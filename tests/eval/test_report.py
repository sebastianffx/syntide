"""End to end on disk: runs written through the trainer seam load back only while their contract holds, and
the evaluator reports validation, then the test fold after the rules were frozen and the fold was scored."""

import json
from dataclasses import asdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from syntide.eval import across
from syntide.eval.analysis import AnalysisConfig, GapClosedRule
from syntide.eval.freeze import freeze_rules, verify_frozen
from syntide.eval.holdout import evaluate as score_test_fold
from syntide.eval.report import write_report
from syntide.eval.runs import RunError, load_run, load_runs
from syntide.eval.stats import SELECTION_METRIC, val_scores
from syntide.eval.tracking import TrackerConfig, make_tracker
from syntide.generate.plan import ArmConfig, ProtocolConfig, RegimeConfig
from syntide.generate.simvea import SimveaConfig
from syntide.trial import Checkpoint, TrainerInfo, make_spec, open_trial, write_spec

TRAINER = TrainerInfo(name="fake", version="1", config={}, device="cpu", predictor="fake:predict")
ARMS = {"A0": 1.0, "A1": 1.5, "A3": 3.0, "C1": 2.0}  # arm -> how well its fake model separates labels
SEEDS = (0, 1)
PROTOCOL = ProtocolConfig(
    id="test-protocol",
    arms={
        "A0": ArmConfig("depleted", "none", "none"),
        "A1": ArmConfig("depleted", "generic", "fixed"),
        "A3": ArmConfig("depleted", "editor", "gap_targeted"),
        "C1": ArmConfig("natural_mix", "none", "none"),
    },
    regimes={"scarcity25": RegimeConfig(manifest="manifests/test/pool25.json", max_steps=30)},
    seeds=list(SEEDS),
)
RULES = AnalysisConfig(
    regime="scarcity25",
    sequence=[],
    diagnostics=[["C1", "A0"]],
    gap_closed=[GapClosedRule(name="natural mix", target="C1", gated=False)],
    n_boot=200,
)


def _probs(labels, quality, seed):
    noise = np.random.default_rng(seed).standard_normal(labels.shape)
    return 1 / (1 + np.exp(-(quality * (labels - 0.5) + noise)))


def _train(protocol_dir, spec, val, parents, calibration, quality=None):
    run_dir = protocol_dir / spec.regime / spec.arm / f"seed{spec.replicate_seed}"
    write_spec(run_dir, spec)
    tracker = make_tracker(TrackerConfig("files"), run_dir, spec.run_name, spec.protocol, spec.to_dict())
    with open_trial(spec, parents, val, tracker, run_dir, calibration=calibration) as trial:
        for _ in trial.batches():
            pass
        quality = ARMS[spec.arm] if quality is None else quality
        probs = {
            step: _probs(val.labels, quality * step / 20, 100 * spec.replicate_seed + step)
            for step in (10, 20, 30)
        }
        history = [{"step": step, **val_scores(val, p)} for step, p in probs.items()]
        best = max(history, key=lambda entry: entry[SELECTION_METRIC])["step"]
        for name in ("best", "final"):
            (run_dir / f"{name}.ckpt").write_bytes(f"{spec.arm}-{spec.replicate_seed}-{name}".encode())
        trial.finish(
            TRAINER,
            best=Checkpoint(run_dir / "best.ckpt", best),
            final=Checkpoint(run_dir / "final.ckpt", 30),
            val_history=history,
            val_probs={"best": probs[best], "final": probs[30]},
            init_sha256=f"{spec.replicate_seed}" * 64,
            wall_time_s=1.0,
        )
    return {"run_dir": str(run_dir.relative_to(protocol_dir)), "trial_id": spec.trial_id}


@pytest.fixture
def protocol_dir(tmp_path, make_samples, make_test_spec, editor_calibration):
    protocol_dir = tmp_path / "runs" / "test-protocol"
    val, parents = make_samples(60, seed=1, prefix="val"), make_samples(20)
    entries = []
    for arm in ARMS:
        for seed in SEEDS:
            spec = make_test_spec(arm, replicate_seed=seed)
            calibration = editor_calibration if arm == "A3" else None
            entries.append(_train(protocol_dir, spec, val, parents, calibration))
    (protocol_dir / "plan.json").write_text(json.dumps({"protocol": "test-protocol", "trials": entries}))
    return protocol_dir


def test_finished_runs_load_back_aligned_and_reproduce_their_score(protocol_dir):
    runs, pending = load_runs(protocol_dir, "val")
    assert pending == [] and len(runs) == len(ARMS) * len(SEEDS)
    assert all(np.array_equal(run.record_id, runs[0].record_id) for run in runs)
    assert list(runs[0].record_id) == sorted(runs[0].record_id)
    assert set(runs[0].probs) == {"best", "final"} and runs[0].steps["final"] == 30


def _rewrite(path, column, change):
    table = pq.read_table(path)
    values = table.column(column).to_numpy(zero_copy_only=False).copy()
    table = table.set_column(table.column_names.index(column), column, pa.array(change(values)))
    pq.write_table(table, path)


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda d: (d / "best.ckpt").write_bytes(b"retrained"), "changed after training"),
        (lambda d: _rewrite(d / "predictions_val.parquet", "y_prob", lambda v: 1 - v), "give AUROC"),
        (
            lambda d: _rewrite(
                d / "predictions_val.parquet", "checkpoint", lambda v: np.where(v == "final", "last", v)
            ),
            "holds checkpoints",
        ),
        (
            lambda d: (d / "trial.json").write_text(
                (d / "trial.json").read_text().replace('"max_steps": 30', '"max_steps": 20')
            ),
            "does not match its trial_id",
        ),
    ],
    ids=["checkpoint", "probabilities", "checkpoint-name", "trial-json"],
)
def test_a_run_whose_outputs_were_changed_is_refused(protocol_dir, tamper, message):
    run_dir = protocol_dir / "scarcity25" / "A3" / "seed0"
    load_run(run_dir, "val")
    tamper(run_dir)
    with pytest.raises(RunError, match=message):
        load_run(run_dir, "val")


def test_a_validation_report_waits_for_a_seed_both_compared_arms_finished(
    tmp_path, make_samples, make_test_spec, editor_calibration
):
    """Mid-training, A3 and A1 can each have finished runs, but on different seeds (parallel pods, a failed
    run): the report says the contrast waits for a common seed instead of failing."""
    protocol_dir = tmp_path / "runs" / "test-protocol"
    val, parents = make_samples(60, seed=1, prefix="val"), make_samples(20)
    entries = [
        _train(
            protocol_dir,
            make_test_spec(arm, replicate_seed=seed),
            val,
            parents,
            editor_calibration if arm == "A3" else None,
        )
        for arm, seed in (("A0", 0), ("A0", 1), ("A1", 1), ("A3", 0), ("C1", 0), ("C1", 1))
    ]
    (protocol_dir / "plan.json").write_text(json.dumps({"protocol": "test-protocol", "trials": entries}))

    out = write_report(protocol_dir, PROTOCOL, RULES, "val", "best", TrackerConfig("files"))
    results = json.loads((out / "comparisons.json").read_text())
    assert results["chain"][0]["missing"] == ["A3 and A1 on a common seed"]
    a3 = next(row for row in results["table"] if row["label"] == "A3")
    assert a3["delta"] == {"against": "A1", "unpaired": "no finished seed in common"}
    assert "no finished runs for A3 and A1 on a common seed" in (out / "results.md").read_text()


def test_validation_report_then_test_report_after_freeze_and_scoring(protocol_dir, make_samples):
    out = write_report(protocol_dir, PROTOCOL, RULES, "val", "best", TrackerConfig("files"))
    results = json.loads((out / "comparisons.json").read_text())
    primary = results["chain"][0]
    assert primary["label"] == "A3 - A1" and primary["status"] in {"passed", "failed"}
    assert primary["artifact"]["seeds"] == list(SEEDS)
    assert [row["label"] for row in results["table"]] == ["A0", "A1", "A3", "C1"]
    assert results["table"][1]["delta"] is None  # A1 is the reference
    assert results["table"][2]["delta"]["artifact"]["estimate"] == primary["artifact"]["estimate"]
    for name in ("results.md", "figures/forest.png", "figures/per_class.png", "figures/learning_curves.png"):
        assert (out / name).exists()
    assert "rules not frozen" in (out / "results.md").read_text()

    with pytest.raises(RunError, match="score the test fold first"):
        write_report(protocol_dir, PROTOCOL, RULES, "test", "best", TrackerConfig("files"))
    freeze_rules(protocol_dir, RULES)
    test = make_samples(50, seed=3, prefix="test")
    predict = lambda checkpoint, x, device: _probs(test.labels, 2.0, len(checkpoint.path.read_bytes()))  # noqa: E731
    score_test_fold(protocol_dir, test=test, load_predictor=lambda name: predict)
    verify_frozen(protocol_dir, RULES, claimed=True)

    out = write_report(protocol_dir, PROTOCOL, RULES, "test", "best", TrackerConfig("files"))
    assert out.name == "test"
    results = json.loads((out / "comparisons.json").read_text())
    assert results["chain"][0]["artifact"]["split"] == "test"
    assert results["gap_gate"]["artifact"]["split"] == "val"  # the gate is judged on validation only
    assert "test fold claimed" in results["freeze"]


def test_an_arm_planned_in_another_protocol_is_compared_with_the_first_protocols_draws(
    protocol_dir, make_samples, make_test_spec
):
    """An arm added after a test claim has its own protocol. ``across`` pairs it with the first protocol's
    runs under that protocol's frozen bootstrap, and reproduces that protocol's own contrasts exactly."""
    out = write_report(protocol_dir, PROTOCOL, RULES, "val", "best", TrackerConfig("files"))
    own = json.loads((out / "comparisons.json").read_text())["diagnostics"][0]
    assert own["label"] == "C1 - A0"
    with pytest.raises(across.AcrossError, match=r"rules\.json does not exist"):
        across.load([protocol_dir], "val")
    freeze_rules(protocol_dir, RULES)

    ext_dir = protocol_dir.parent / "ext-protocol"
    val, parents = make_samples(60, seed=1, prefix="val"), make_samples(20)
    entries = []
    for seed in SEEDS:
        base = make_test_spec(
            "A1",
            protocol="ext-protocol",
            transform_family="simvea",
            generator="simvea@1",
            transform=asdict(SimveaConfig()),
            replicate_seed=seed,
        ).to_dict()
        derived = ("init_seed", "syntide_version", "trial_id")
        spec = make_spec(**{**{key: value for key, value in base.items() if key not in derived}, "arm": "T6"})
        entries.append(_train(ext_dir, spec, val, parents, None, quality=2.5))
    (ext_dir / "plan.json").write_text(json.dumps({"protocol": "ext-protocol", "trials": entries}))

    rules, runs = across.load([protocol_dir, ext_dir], "val")
    assert rules.sha256 == RULES.sha256 and len(runs) == (len(ARMS) + 1) * len(SEEDS)
    contrasts = across.compare(runs, rules, [("C1", "A0"), ("T6", "A0"), ("T6", "A1")])
    for name in ("artifact", "clean", "overall"):
        assert contrasts[0][name] == own[name]  # the same draws as the first protocol's own report
    assert contrasts[1]["label"] == "T6 - A0" and contrasts[1]["overall"]["estimate"] > 0
    assert contrasts[1]["overall"]["seeds"] == list(SEEDS) and set(contrasts[1]["best_step"]) == {"T6", "A0"}
    with pytest.raises(across.AcrossError, match="is in both"):
        across.load([protocol_dir, ext_dir, protocol_dir], "val")

    across.main([str(protocol_dir), str(ext_dir), "--contrast", "T6-A0", "--contrast", "T6-A1"])
    text = (ext_dir / "eval" / "val" / "across.md").read_text()
    assert "| T6 - A0 |" in text and "| T6 - A1 |" in text

    # On test, every protocol's own claim must cover its runs: the extension's fold is scored separately.
    test = make_samples(50, seed=3, prefix="test")
    predict = lambda checkpoint, x, device: _probs(test.labels, 2.0, len(checkpoint.path.read_bytes()))  # noqa: E731
    score_test_fold(protocol_dir, test=test, load_predictor=lambda name: predict)
    with pytest.raises(RunError, match="score the test fold first"):
        across.load([protocol_dir, ext_dir], "test")
    score_test_fold(ext_dir, test=test, load_predictor=lambda name: predict)
    rules, runs = across.load([protocol_dir, ext_dir], "test")
    assert {run.split for run in runs} == {"test"} and len(runs) == (len(ARMS) + 1) * len(SEEDS)
