"""Exercise the all-arm holdout gate without a model, GPU, or held-out labels."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from syntide import qa_baseline_audit


@pytest.fixture
def runner():
    path = Path(__file__).parents[1] / "scripts/runpod/run_qa_arms.py"
    spec = importlib.util.spec_from_file_location("run_qa_arms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study(runner, tmp_path, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    protocol = {
        "run_dir": str(root),
        "seeds": [0],
        "generation_batch_size": 512,
        # Sorted JSON changes the arm insertion order; this is an ordinary retry.
        "arms": {arm: {"prepared": str(tmp_path / arm)} for arm in sorted(runner.ARMS)},
    }
    manifests = {
        arm: {
            "sha256": arm,
            "files": {
                name: {"sha256": "same" if name != "train.parquet" else arm, "bytes": 1}
                for name in (
                    "selection.parquet",
                    "validation.parquet",
                    "test.parquet",
                    "train.parquet",
                    "records.json",
                    "signals.npy",
                )
            },
        }
        for arm in runner.ARMS
    }
    baseline = {key: 1 for key in runner.TRAIN_OPTIONS}
    baseline.update(selection_method="option-nll", eval_batch_size=256)

    def save_run(arm, *, converged=True):
        run = root / arm / "seed0"
        run.mkdir(parents=True, exist_ok=True)
        (run / "best.pt").write_bytes(arm.encode())
        config = {
            **baseline,
            "manifest_sha256": arm,
            "data_files": {
                name: manifests[arm]["files"][name]["sha256"]
                for name in ("train.parquet", "selection.parquet", "records.json", "signals.npy")
            },
        }
        runner.write(run / "config.json", config)
        runner.write(
            run / "train_result.json",
            {
                "best_before_80_percent": converged,
                "best_step": 12000 if converged else 15531,
                "best_checkpoint_sha256": runner.sha(run / "best.pt"),
            },
        )
        runner.write(run / "selection_history.json", [])

    save_run("real_qa")
    fake_data = types.ModuleType("syntide.baseline.data")
    fake_data.verify_prepared = lambda prepared, **kw: manifests[Path(prepared).name]
    fake_metrics = types.ModuleType("syntide.baseline.metrics")
    fake_metrics.fit_majority_answers = lambda rows: {"1": "yes"}
    fake_metrics.majority_predictions = lambda rows, majority: rows
    fake_metrics.score_predictions = lambda rows: {"reference": "fixed baseline majority"}
    monkeypatch.setitem(sys.modules, "syntide.baseline", types.ModuleType("syntide.baseline"))
    monkeypatch.setitem(sys.modules, "syntide.baseline.data", fake_data)
    monkeypatch.setitem(sys.modules, "syntide.baseline.metrics", fake_metrics)
    monkeypatch.setattr(pq, "read_table", lambda path: types.SimpleNamespace(to_pylist=lambda: []))
    monkeypatch.setattr(qa_baseline_audit, "audit_config", lambda *args: {"status": "passed"})
    monkeypatch.setattr(
        qa_baseline_audit,
        "audit_run",
        lambda base, cfg, result, history, **kw: {
            "test_eligible": result["best_before_80_percent"],
        },
    )
    monkeypatch.setattr(runner, "verify_runtime_sources", lambda config: None)
    monkeypatch.setattr(runner, "write_comparison_report", lambda root, summaries: None)
    summaries = []
    monkeypatch.setattr(
        runner,
        "make_summary",
        lambda root, arm, majority, complete: (
            summaries.append((arm, complete)) or {"arm": arm, "complete": complete}
        ),
    )
    calls = []
    late = set()

    def execute(command, log, lock, **kwargs):
        phase = command[4]
        run = Path(command[command.index("--run-dir") + 1])
        arm = run.parent.name
        calls.append((phase, arm))
        if phase == "train":
            assert not (root / "freeze.json").exists()
            save_run(arm, converged=arm not in late)
        else:
            assert (root / "freeze.json").is_file()
            assert (root / "holdout_claim.json").is_file()
            assert all((root / name / "seed0/train_result.json").exists() for name in runner.ARMS)

    monkeypatch.setattr(runner, "execute", execute)
    return protocol, root, calls, late, manifests, summaries, save_run


def test_trains_three_arms_then_freezes_all_four_before_evaluation(runner, study):
    protocol, root, calls, _, _, summaries, _ = study
    runner.run_study(protocol, None)
    assert calls[:3] == [("train", arm) for arm in runner.ARMS[1:]]
    assert len([call for call in calls if call[0] == "evaluate"]) == 16
    assert runner.read(root / "status.json")["status"] == "complete"
    assert set(runner.read(root / "freeze.json")["checkpoints"]) == set(runner.ARMS)
    assert sum(complete for _, complete in summaries) == 4
    calls.clear()
    runner.run_study(json.loads(json.dumps(protocol, sort_keys=True)), None)
    assert all(phase == "evaluate" for phase, _ in calls)


def test_one_late_arm_finishes_training_but_keeps_every_test_closed(runner, study):
    protocol, root, calls, late, _, _, _ = study
    late.add("paraphrase")
    runner.run_study(protocol, None)
    assert calls == [("train", arm) for arm in runner.ARMS[1:]]
    assert not (root / "freeze.json").exists()
    assert not (root / "holdout_claim.json").exists()
    status = runner.read(root / "status.json")
    assert status["status"] == "needs_budget_review" and status["arms"] == ["paraphrase"]
    assert status["test_opened"] is False


def test_heldout_dataset_change_fails_before_training(runner, study):
    protocol, _, calls, _, manifests, _, _ = study
    manifests["grounded"]["files"]["test.parquet"]["sha256"] = "other"
    with pytest.raises(ValueError, match="test bytes changed"):
        runner.run_study(protocol, None)
    assert calls == []


def test_holdout_artifacts_without_global_freeze_are_rejected(runner, study):
    protocol, root, calls, *_ = study
    runner.write(root / "real_qa/seed0/test_claim.json", {})
    with pytest.raises(ValueError, match="lacks a matching global"):
        runner.run_study(protocol, None)
    assert calls == []


def test_incomplete_training_is_preserved(runner, study):
    protocol, root, calls, *_ = study
    path = root / "paraphrase/seed0/config.json"
    runner.write(path, {"partial": True})
    with pytest.raises(ValueError, match="preserving files"):
        runner.run_study(protocol, None)
    assert path.read_text() and calls == []


def test_no_training_can_start_after_freeze(runner, study):
    protocol, root, calls, *_ = study
    runner.write(root / "freeze.json", {"already": "frozen"})
    with pytest.raises(ValueError, match="Cannot start training"):
        runner.run_study(protocol, None)
    assert calls == []


def test_changed_checkpoint_after_freeze_is_rejected(runner, study):
    protocol, root, _, _, _, _, _ = study
    runner.run_study(protocol, None)
    result = runner.read(root / "paraphrase/seed0/train_result.json")
    result["best_checkpoint_sha256"] = "replacement"
    runner.write(root / "paraphrase/seed0/train_result.json", result)
    with pytest.raises(ValueError, match="Frozen identity changed"):
        runner.run_study(protocol, None)


def test_train_command_explicitly_preserves_selection_options(runner):
    baseline = {key: 1 for key in runner.TRAIN_OPTIONS}
    baseline.update(selection_method="option-nll", eval_batch_size=256)
    command = runner.train_command("prepared", "run", baseline)
    assert command[command.index("--selection-method") + 1] == "option-nll"
    assert command[command.index("--eval-batch-size") + 1] == "256"


def test_runtime_source_hashes_are_checked_before_evaluation(runner, tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("pinned source\n")
    monkeypatch.setattr(
        runner.importlib.util, "find_spec", lambda name: types.SimpleNamespace(origin=str(source))
    )
    baseline = {"trainer_source_sha256": runner.sha(source), "metrics_source_sha256": runner.sha(source)}
    runner.verify_runtime_sources(baseline)
    source.write_text("changed source\n")
    with pytest.raises(ValueError, match="Runtime differs"):
        runner.verify_runtime_sources(baseline)


@pytest.fixture
def summary_artifacts(runner, tmp_path, monkeypatch):
    """Distinct slice scores at tied maxima expose accidental use of the later event."""
    root = tmp_path / "summary-study"
    run = root / "paraphrase/seed0"
    first = {
        "groups": {
            "all": {"accuracy": 0.39, "template_macro_f1": 0.2, "invalid_rate": 0.0},
            "clean": {"accuracy": 0.465, "template_macro_f1": 0.3, "invalid_rate": 0.0},
            "noisy": {"accuracy": 0.305, "template_macro_f1": 0.1, "invalid_rate": 0.0},
        }
    }
    later = json.loads(json.dumps(first))
    later["groups"]["clean"]["accuracy"] = 0.47
    later["groups"]["noisy"]["template_macro_f1"] = 0.15
    runner.write(run / "train_result.json", {"best_step": 1000})
    runner.write(
        run / "config.json",
        {"selection_method": "option-nll", "selection_questions": 2000, "manifest_sha256": "frozen"},
    )
    runner.write(run / "best_selection_metrics.json", first)
    runner.write(
        run / "selection_history.json",
        [
            {"step": 1000, "accuracy": 0.39, "scores": first},
            {"step": 2000, "accuracy": 0.39, "scores": later},
            {"step": 3000, "accuracy": 0.35, "scores": {}},
        ],
    )
    runner.write(root / "protocol.json", {"arms": {"paraphrase": {"prepared": str(tmp_path / "prepared")}}})
    runner.write(root / "majority_selection.json", {"fixed_reference": True})
    runner.write(root / "paraphrase/config_comparison.json", {"status": "passed"})
    fake_report = types.ModuleType("syntide.baseline.report")

    def no_holdout_access(*args, **kwargs):
        pytest.fail("Invalid selection summary must be rejected before held-out report access")

    fake_report.build_report = no_holdout_access
    monkeypatch.setitem(sys.modules, "syntide.baseline", types.ModuleType("syntide.baseline"))
    monkeypatch.setitem(sys.modules, "syntide.baseline.report", fake_report)
    return root, run, first, later


def test_summary_uses_verified_first_argmax_scores(runner, summary_artifacts):
    root, _, first, _ = summary_artifacts
    summary = runner.make_summary(root, "paraphrase", {}, complete=False)
    assert summary["best_selection_metrics"] == first["groups"]
    assert summary["final_selection_accuracy"] == 0.35
    assert summary["seed_sd"] is None
    assert runner.read(root / "paraphrase/summary.json") == summary


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("mismatch", ["later_tied_maximum", "stale_clean_f1"])
def test_summary_rejects_stale_slice_metrics_before_publication(
    runner, summary_artifacts, complete, mismatch
):
    root, run, first, later = summary_artifacts
    saved = json.loads(json.dumps(later if mismatch == "later_tied_maximum" else first))
    if mismatch == "stale_clean_f1":
        saved["groups"]["clean"]["template_macro_f1"] = 0.99
    # Overall selection accuracy remains correct: checking only that scalar is insufficient.
    assert saved["groups"]["all"]["accuracy"] == first["groups"]["all"]["accuracy"]
    runner.write(run / "best_selection_metrics.json", saved)
    with pytest.raises(ValueError, match="best-selection metrics differ from the selected history event"):
        runner.make_summary(root, "paraphrase", {}, complete=complete)
    assert not (root / "paraphrase/summary.json").exists()
    assert not (root / "paraphrase/verified_report.json").exists()


@pytest.mark.parametrize("label", ["test", "test_zero"])
@pytest.mark.parametrize("has_partial", [False, True])
def test_claimed_unpublished_test_never_starts_a_second_generation_pass(
    runner, study, monkeypatch, label, has_partial
):
    protocol, root, *_ = study
    runner.run_study(protocol, None)  # establish valid, matching all-arm freeze and holdout claim
    run = root / "real_qa/seed0"
    claim = run / f"{label}_claim.json"
    runner.write(claim, {"batch_size": protocol["generation_batch_size"]})
    partial = run / f"{label}_predictions.partial"
    if has_partial:
        partial.write_text('{"task_id":"first-test-question","output_text":"partial answer"}\n')
    if label == "test_zero":
        # A completed real-signal test can be rescored while its interrupted ablation stays blocked.
        runner.write(run / "test_claim.json", {"batch_size": protocol["generation_batch_size"]})
        (run / "test_predictions.jsonl").write_text("published real-signal predictions\n")
    before = {path: path.read_bytes() for path in (claim, partial) if path.exists()}
    commands = []
    monkeypatch.setattr(runner, "execute", lambda command, *args, **kwargs: commands.append(command))
    with pytest.raises(ValueError, match=f"interrupted {label} claim requires review"):
        runner.run_study(protocol, None)
    assert all(command[4] == "evaluate" for command in commands)
    evaluated_labels = [
        command[command.index("--split") + 1] + ("_zero" if "--zero-signal" in command else "")
        for command in commands
    ]
    assert label not in evaluated_labels
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not (run / f"{label}_predictions.jsonl").exists()


@pytest.mark.parametrize("label", ["test", "test_zero"])
def test_published_test_predictions_remain_eligible_for_verified_rescoring(runner, study, monkeypatch, label):
    protocol, root, *_ = study
    runner.run_study(protocol, None)
    run = root / "real_qa/seed0"
    runner.write(run / f"{label}_claim.json", {"batch_size": protocol["generation_batch_size"]})
    predictions = run / f"{label}_predictions.jsonl"
    predictions.write_text("published predictions delegated to unchanged evaluator for verification\n")
    original = predictions.read_bytes()
    commands = []
    monkeypatch.setattr(runner, "execute", lambda command, *args, **kwargs: commands.append(command))
    runner.run_study(protocol, None)
    assert all(command[4] == "evaluate" for command in commands)
    assert any(
        command[command.index("--run-dir") + 1] == str(run)
        and command[command.index("--split") + 1] == "test"
        and ("--zero-signal" in command) == (label == "test_zero")
        for command in commands
    )
    assert predictions.read_bytes() == original
