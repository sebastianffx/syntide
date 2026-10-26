"""Training checks run without model hardware or held-out labels."""

import copy
import datetime as dt
import importlib.util
import sys
import types
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from syntide.qa_baseline_audit import _PROTOCOL


@pytest.fixture
def runner():
    path = Path(__file__).parents[1] / "scripts/runpod/run_diffusets_arm.py"
    spec = importlib.util.spec_from_file_location("run_diffusets_arm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study(runner, tmp_path, monkeypatch):
    original = tmp_path / "original"
    baseline_run = original / "real_qa/seed0"
    root = tmp_path / "extension"
    root.mkdir()
    baseline = {
        **_PROTOCOL,
        "model": "google/gemma-3-270m",
        "base_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "opentslm_commit": "2968f4b891baab4307f7e9d0043e87677b593a30",
        "packages": {"torch": "pinned"},
        "precision": "bf16",
        "prompt_version": "original",
        "trainer_source_sha256": "1" * 64,
        "metrics_source_sha256": "2" * 64,
        "init_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "total_parameters": 343788580,
        "trainable_parameters": 75688484,
        "python": "3.11.11",
        "gpu": "H100",
        "data_files": {
            "train.parquet": "5" * 64,
            "selection.parquet": "6" * 64,
            "signals.npy": "7" * 64,
            "records.json": "8" * 64,
        },
        "token_lengths": {
            split: {"would_truncate_rationale": 0, "sample_questions": 1024}
            for split in ("train", "selection")
        },
    }
    candidate = copy.deepcopy(baseline)
    candidate.update(manifest_sha256="9" * 64, gpu="A100")
    candidate["data_files"]["train.parquet"] = "a" * 64
    manifests = {}
    for name, config in (("real_qa", baseline), (runner.ARM, candidate)):
        manifests[name] = {
            "sha256": config["manifest_sha256"],
            "files": {
                key: {"sha256": value, "bytes": 1}
                for key, value in config["data_files"].items()
            },
            "splits": {
                "train": {"questions": 100000},
                "validation": {"questions": 27264},
                "test": {"questions": 26841},
            },
        }
        manifests[name]["files"].update(
            {
                key + ".parquet": {"sha256": key, "bytes": 1}
                for key in ("validation", "test")
            }
        )

    def save_run(directory, config, *, best_step=12000, truncated=0):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "best.pt").write_bytes(str(directory).encode())
        runner.write(directory / "config.json", config)
        runner.write(
            directory / "train_result.json",
            {
                "seed": 0,
                "max_steps": 15531,
                "manifest_sha256": config["manifest_sha256"],
                "status": "trained",
                "smoke_only": False,
                "distinct_truncated_rationales": truncated,
                "distinct_seen_training_questions": 100000,
                "best_step": best_step,
                "best_selection_accuracy": 0.4,
                "best_before_80_percent": best_step < 0.8 * 15531,
                "best_checkpoint_sha256": runner.sha(directory / "best.pt"),
            },
        )
        runner.write(
            directory / "selection_history.json",
            [
                {
                    "step": step,
                    "accuracy": 0.4 if step == best_step else 0.3,
                    "questions": 2000,
                    "scores": {
                        "groups": {
                            "all": {
                                "accuracy": 0.4 if step == best_step else 0.3
                            }
                        }
                    },
                }
                for step in [*range(1000, 15001, 1000), 15531]
            ],
        )

    save_run(baseline_run, baseline)
    protocol = {
        "study_kind": runner.STUDY_KIND,
        "run_dir": str(root),
        "seeds": [0],
        "candidate_count": 5000,
        "candidate_plan_sha256": "b" * 64,
        "generation_batch_size": 512,
        "evaluation_budget": {
            "deadline_utc": "2100-01-01T00:00:00Z",
            "seconds_per_question": 0.30,
            "safety_factor": 1.25,
            "seconds_per_pass_overhead": 180,
            "final_reserve_seconds": 1800,
        },
        "arms": {
            runner.ARM: {"prepared": str(tmp_path / "prepared" / runner.ARM)}
        },
        "baseline": {
            "run_dir": str(baseline_run),
            "prepared": str(tmp_path / "prepared/real_qa"),
            "original_study_dir": str(original),
            "config_sha256": runner.sha(baseline_run / "config.json"),
            "checkpoint_sha256": runner.sha(baseline_run / "best.pt"),
            "manifest_sha256": baseline["manifest_sha256"],
        },
    }
    runner.freeze_once(root / "protocol.json", protocol)
    preflight = {
        "status": "passed",
        "extension_protocol_sha256": runner.sha(root / "protocol.json"),
        "baseline_config_sha256": protocol["baseline"]["config_sha256"],
        "candidate_plan_sha256": protocol["candidate_plan_sha256"],
        "arms": {
            arm: {
                "expected_config": config,
                "full_training_token_scan": {
                    "questions": 100000,
                    "truncated_rationales": 0,
                },
            }
            for arm, config in (("real_qa", baseline), (runner.ARM, candidate))
        },
    }
    runner.write(root / "preflight.json", preflight)
    data = types.ModuleType("syntide.baseline.data")
    data.verify_prepared = lambda path, **kw: manifests[Path(path).name]
    metrics = types.ModuleType("syntide.baseline.metrics")
    metrics.fit_majority_answers = lambda rows: {"1": "yes"}
    metrics.majority_predictions = lambda rows, majority: rows
    metrics.score_predictions = lambda rows: {"reference": "baseline majority"}
    monkeypatch.setitem(
        sys.modules, "syntide.baseline", types.ModuleType("syntide.baseline")
    )
    monkeypatch.setitem(sys.modules, "syntide.baseline.data", data)
    monkeypatch.setitem(sys.modules, "syntide.baseline.metrics", metrics)
    monkeypatch.setattr(
        pq,
        "read_table",
        lambda path: types.SimpleNamespace(to_pylist=lambda: []),
    )
    monkeypatch.setattr(
        runner.shared, "verify_runtime_sources", lambda cfg: None
    )
    monkeypatch.setattr(
        runner.shared,
        "make_summary",
        lambda root, arm, majority, complete: {"complete": complete},
    )
    monkeypatch.setattr(runner, "write_report", lambda root, summary: None)
    calls = []
    training = {"best_step": 12000, "truncated": 0}

    def execute(command, *args, **kwargs):
        action = command[4]
        directory = Path(command[command.index("--run-dir") + 1])
        assert directory == root / runner.ARM / "seed0"
        if action == "train":
            assert not (root / "freeze.json").exists()
            save_run(directory, candidate, **training)
            calls.append("train")
        else:
            assert (root / "freeze.json").exists() and (
                root / "holdout_claim.json"
            ).exists()
            label = command[command.index("--split") + 1] + (
                "_zero" if "--zero-signal" in command else ""
            )
            calls.append(label)
            runner.write(directory / f"{label}_claim.json", {"batch_size": 512})
            (directory / f"{label}_predictions.jsonl").write_text(
                "published predictions"
            )

    monkeypatch.setattr(runner.shared, "execute", execute)
    return types.SimpleNamespace(
        protocol=protocol,
        root=root,
        baseline=baseline,
        candidate=candidate,
        manifests=manifests,
        calls=calls,
        training=training,
        save_run=save_run,
        baseline_run=baseline_run,
        preflight=preflight,
    )


def test_extension_freezes_once_and_preserves_original(runner, study):
    original = {p: p.read_bytes() for p in study.baseline_run.glob("*")}
    runner.run_study(study.protocol, None)
    assert study.calls == [
        "train",
        "validation",
        "validation_zero",
        "test",
        "test_zero",
    ]
    frozen = runner.read(study.root / "freeze.json")
    assert set(frozen["checkpoints"]) == {runner.ARM}
    assert (
        frozen["candidate_plan_sha256"]
        == study.protocol["candidate_plan_sha256"]
    )
    assert runner.read(study.root / "status.json")["status"] == "complete"
    study.calls.clear()
    runner.run_study(study.protocol, None)
    assert study.calls == ["validation", "validation_zero", "test", "test_zero"]
    assert {p: p.read_bytes() for p in original} == original


def test_late_argmax_keeps_extension_test_closed(runner, study):
    study.training["best_step"] = 15531
    runner.run_study(study.protocol, None)
    assert study.calls == ["train"]
    assert not (study.root / "freeze.json").exists()
    assert (
        runner.read(study.root / "status.json")["status"]
        == "needs_budget_review"
    )


@pytest.mark.parametrize("field", ["init_sha256", "packages", "max_steps"])
def test_changed_training_identity_fails_before_holdout(runner, study, field):
    study.candidate[field] = "changed"
    with pytest.raises(ValueError, match=field):
        runner.run_study(study.protocol, None)
    assert not (study.root / "freeze.json").exists()
    assert all(call == "train" for call in study.calls)


def test_truncated_completed_run_keeps_test_closed(runner, study):
    study.training["truncated"] = 1
    with pytest.raises(ValueError, match="truncated"):
        runner.run_study(study.protocol, None)
    assert study.calls == ["train"]
    assert not (study.root / "freeze.json").exists()


@pytest.mark.parametrize("label", ["test", "test_zero"])
def test_claimed_interrupted_test_is_never_regenerated(runner, study, label):
    runner.run_study(study.protocol, None)
    run = study.root / runner.ARM / "seed0"
    (run / f"{label}_predictions.jsonl").unlink()
    partial = run / f"{label}_predictions.partial"
    partial.write_bytes(b"preserve incomplete generation")
    study.calls.clear()
    with pytest.raises(ValueError, match="refusing repeated test generation"):
        runner.run_study(study.protocol, None)
    assert label not in study.calls and "train" not in study.calls
    assert partial.read_bytes() == b"preserve incomplete generation"


@pytest.mark.parametrize(
    "change",
    [
        "preflight_binding",
        "test_bytes",
        "baseline_checkpoint",
        "baseline_config",
    ],
)
def test_changed_provenance_blocks_before_training(runner, study, change):
    if change == "preflight_binding":
        study.preflight["extension_protocol_sha256"] = "changed"
        runner.write(study.root / "preflight.json", study.preflight)
    elif change == "test_bytes":
        study.manifests[runner.ARM]["files"]["test.parquet"]["sha256"] = (
            "changed"
        )
    elif change == "baseline_checkpoint":
        (study.baseline_run / "best.pt").write_bytes(b"changed")
    else:
        runner.write(study.baseline_run / "config.json", {"changed": True})
    with pytest.raises(ValueError):
        runner.run_study(study.protocol, None)
    assert study.calls == []


@pytest.mark.parametrize(
    "state",
    ["partial_training", "freeze_without_result", "holdout_without_freeze"],
)
def test_existing_incomplete_state_is_preserved(runner, study, state):
    run = study.root / runner.ARM / "seed0"
    if state == "partial_training":
        runner.write(run / "config.json", study.candidate)
    elif state == "freeze_without_result":
        runner.write(study.root / "freeze.json", {"frozen": True})
    else:
        runner.write(run / "test_claim.json", {"batch_size": 512})
    with pytest.raises(ValueError):
        runner.run_study(study.protocol, None)
    assert study.calls == []


def test_extension_cannot_run_inside_original_study(runner, study):
    study.protocol["run_dir"] = str(
        study.baseline_run.parent.parent / "new-arm"
    )
    with pytest.raises(ValueError, match="separate"):
        runner.validate_protocol(study.protocol)


def test_invalid_composition_fails_before_paid_work(runner, study):
    study.protocol["training_corpora"] = ["baseline_original", "grounded"]
    with pytest.raises(ValueError, match="training_corpora must map"):
        runner.validate_protocol(study.protocol)


def test_candidate_plan_bytes_must_match_declared_hash(runner, study, tmp_path):
    path = tmp_path / "candidate-plan.json"
    path.write_text("{}\n")
    study.protocol["candidate_plan_path"] = str(path)
    with pytest.raises(ValueError, match="candidate plan file differs"):
        runner.verify_baseline(study.protocol)


def test_preflight_uses_private_baseline_plus_extension_arm_set(
    runner, study, monkeypatch
):
    module = types.SimpleNamespace(ARMS=("original",))

    def check(inputs, baseline, output):
        assert module.ARMS == ("real_qa", runner.ARM)
        assert set(inputs["arms"]) == {"real_qa", runner.ARM}
        assert baseline == study.baseline
        return copy.deepcopy(study.preflight)

    module.preflight = check
    monkeypatch.setattr(runner, "helper", lambda name: module)
    report = runner.preflight(study.protocol, study.root / "preflight.json")
    assert report["extension_protocol_sha256"] == runner.sha(
        study.root / "protocol.json"
    )
    assert (
        report["baseline_config_sha256"]
        == study.protocol["baseline"]["config_sha256"]
    )


def test_partial_token_scan_cannot_authorize_training(runner, study):
    study.preflight["arms"][runner.ARM]["full_training_token_scan"][
        "questions"
    ] = 1
    runner.write(study.root / "preflight.json", study.preflight)
    with pytest.raises(ValueError, match="every extension training row"):
        runner.run_study(study.protocol, None)
    assert study.calls == []


def test_insufficient_total_time_preserves_checkpoint_before_any_evaluation(
    runner, study, monkeypatch
):
    deadline = dt.datetime.fromisoformat(
        "2100-01-01T00:00:00+00:00"
    ).timestamp()
    monkeypatch.setattr(runner.time, "time", lambda: deadline - 1000)
    runner.run_study(study.protocol, None)
    assert study.calls == ["train"]
    assert (study.root / runner.ARM / "seed0/best.pt").exists()
    assert not list((study.root / runner.ARM / "seed0").glob("*_claim.json"))
    status = runner.read(study.root / "status.json")
    assert (
        status["status"] == "needs_time_budget_review"
        and not status["test_opened"]
    )


def test_completed_validation_rate_is_used_before_test_claim(
    runner, study, monkeypatch
):
    runner.run_study(study.protocol, None)
    run = study.root / runner.ARM / "seed0"
    for label, seconds in (("validation", 15000), ("validation_zero", 16000)):
        runner.write(
            run / f"{label}_metrics.json",
            {"questions": 27264, "elapsed_s": seconds},
        )
    for label in ("test", "test_zero"):
        (run / f"{label}_predictions.jsonl").unlink()
        (run / f"{label}_claim.json").unlink()
    study.calls.clear()
    deadline = dt.datetime.fromisoformat(
        "2100-01-01T00:00:00+00:00"
    ).timestamp()
    monkeypatch.setattr(runner.time, "time", lambda: deadline - 30000)
    runner.run_study(study.protocol, None)
    assert study.calls == ["validation", "validation_zero"]
    assert not list(run.glob("test*_claim.json"))
    status = runner.read(study.root / "status.json")
    assert (
        status["status"] == "needs_time_budget_review"
        and status["phase"] == "test"
    )
    budget = status["budget"]
    assert (
        budget["rate_inputs_seconds_per_question"]["validation_zero"]
        == 16000 / 27264
    )
    assert budget["pending_pass_questions"] == {
        "test": 26841,
        "test_zero": 26841,
    }
    assert not budget["initial_rate_is_measured"]


@pytest.fixture
def operational_override(runner, study, tmp_path):
    operational = runner.helper("run_diffusets_pipeline")
    deadline = operational.utc_timestamp(
        study.protocol["evaluation_budget"]["deadline_utc"], "deadline"
    )
    source = tmp_path / "old/run_diffusets_pipeline.py"
    source.parent.mkdir()
    source.write_text("original frozen pipeline")
    original_config = tmp_path / "original-pipeline.json"
    operational.write(
        original_config,
        {
            "deadline_utc": study.protocol["evaluation_budget"]["deadline_utc"],
            "bindings": {str(source): operational.sha(source)},
        },
    )
    receipt, guard = tmp_path / "receipt.json", tmp_path / "guard.json"
    operational.write(
        receipt,
        {
            "pod": {"id": "owned-pod"},
            "created_utc": dt.datetime.fromtimestamp(
                deadline - 3600, dt.UTC
            ).isoformat(),
            "cost_plan": {
                "max_spend_usd": 40,
                "hourly_rate_upper_bound_usd": 0.62,
            },
        },
    )
    operational.write(guard, {"max_spend_usd": 40})
    payload = {
        "schema_version": 1,
        "pod_id": "owned-pod",
        "created_utc": dt.datetime.fromtimestamp(
            deadline - 1800, dt.UTC
        ).isoformat(),
        "reviewed_at_utc": dt.datetime.fromtimestamp(
            deadline - 1800, dt.UTC
        ).isoformat(),
        "deadline_utc": dt.datetime.fromtimestamp(
            deadline + 86400 + 900, dt.UTC
        ).isoformat(),
        "work_deadline_utc": dt.datetime.fromtimestamp(
            deadline + 86400, dt.UTC
        ).isoformat(),
        "max_spend_usd": 40,
        "hourly_rate_upper_bound_usd": 0.62,
        "storage_reserve_usd": 2.5,
        "reason": "Operational deadline extension only",
        "authorization": "User authorized unchanged $40 spending cap",
        "completion_stop": {"stop_on_completion": True},
        "previous_pipeline_source_sha256": operational.sha(source),
    }
    for path_key, hash_key, path in (
        ("pipeline_config_path", "pipeline_config_sha256", original_config),
        ("protocol_path", "protocol_sha256", study.root / "protocol.json"),
        ("receipt_path", "receipt_sha256", receipt),
        ("base_config_path", "base_config_sha256", guard),
    ):
        payload.update({path_key: str(path), hash_key: operational.sha(path)})
    path = tmp_path / "operations/override.json"
    operational.write(path, payload)
    return operational.deadline_override(
        str(path), operational.sha(path), protocol=study.protocol
    )


def test_deadline_override_preserves_training_and_protocol(
    runner, study, operational_override, monkeypatch
):
    original = (study.root / "protocol.json").read_bytes()
    deadline = dt.datetime.fromisoformat(
        "2100-01-01T00:00:00+00:00"
    ).timestamp()
    monkeypatch.setattr(runner.time, "time", lambda: deadline - 1000)
    runner.run_study(study.protocol, None)
    assert study.calls == ["train"]
    assert (
        runner.read(study.root / "status.json")["status"]
        == "needs_time_budget_review"
    )
    study.calls.clear()
    runner.run_study(
        study.protocol, None, operational_override=operational_override
    )
    assert study.calls == ["validation", "validation_zero", "test", "test_zero"]
    assert (study.root / "protocol.json").read_bytes() == original
    budget = runner.read(study.root / "evaluation_budget.json")
    assert (
        budget["operational_override"]["sha256"]
        == operational_override.expected_sha256
    )
    assert (
        budget["scientific_deadline_utc"]
        == study.protocol["evaluation_budget"]["deadline_utc"]
    )
    assert budget["deadline_utc"] != budget["scientific_deadline_utc"]
    assert runner.read(study.root / "status.json")["status"] == "complete"


def test_override_does_not_authorize_interrupted_test_replay(
    runner, study, operational_override
):
    runner.run_study(study.protocol, None)
    run = study.root / runner.ARM / "seed0"
    (run / "test_predictions.jsonl").unlink()
    study.calls.clear()
    with pytest.raises(ValueError, match="refusing repeated test generation"):
        runner.run_study(
            study.protocol, None, operational_override=operational_override
        )
    assert "test" not in study.calls and "train" not in study.calls


def test_override_revalidates_bound_protocol_before_each_evaluation_guard(
    runner, study, operational_override
):
    run = study.root / runner.ARM / "seed0"
    before = runner.evaluation_budget(
        study.protocol,
        run,
        study.manifests[runner.ARM],
        operational_override=operational_override,
    )
    assert before["status"] == "sufficient_estimated_time"
    (study.root / "protocol.json").write_text("changed scientific protocol")
    with pytest.raises(ValueError, match="extension_protocol binding"):
        runner.evaluation_budget(
            study.protocol,
            run,
            study.manifests[runner.ARM],
            operational_override=operational_override,
        )
