"""A changed experiment or incomplete checkpoint must never unlock test scoring."""

import copy
import hashlib

import pytest

from syntide.qa_baseline_audit import _PROTOCOL, audit_config, audit_run


@pytest.fixture
def experiment(tmp_path):
    config = {
        **_PROTOCOL,
        "model": "google/gemma-3-270m",
        "base_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "opentslm_commit": "2968f4b891baab4307f7e9d0043e87677b593a30",
        "packages": {"torch": "2.8.0.dev20250319+cu128", "numpy": "2.2.6"},
        "precision": "bf16 autocast; frozen base bf16; trainable weights fp32",
        "prompt_version": "ecgqa-cot-single-joint-media-v1",
        "trainer_source_sha256": "1" * 64,
        "metrics_source_sha256": "2" * 64,
        "init_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "total_parameters": 343788580,
        "trainable_parameters": 75688484,
        "python": "3.11.11",
        "gpu": "NVIDIA H100 80GB HBM3",
        "data_files": {
            "train.parquet": "5" * 64,
            "selection.parquet": "6" * 64,
            "signals.npy": "7" * 64,
            "records.json": "8" * 64,
        },
        "token_lengths": {
            split: {"would_truncate_rationale": 0, "sample_questions": 1024, "max_prompt_tokens": 842}
            for split in ("train", "selection")
        },
    }
    candidate = copy.deepcopy(config)
    candidate["manifest_sha256"] = "9" * 64
    candidate["data_files"]["train.parquet"] = "a" * 64
    candidate["token_lengths"]["train"]["max_prompt_tokens"] = 800
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    result = {
        "seed": 0,
        "max_steps": 15531,
        "manifest_sha256": candidate["manifest_sha256"],
        "status": "trained",
        "smoke_only": False,
        "distinct_truncated_rationales": 0,
        "distinct_seen_training_questions": 100000,
        "best_step": 12000,
        "best_selection_accuracy": 0.4,
        "best_before_80_percent": True,
        "best_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    history = [
        {
            "step": step,
            "accuracy": 0.4 if step == 12000 else 0.3,
            "questions": 2000,
            "scores": {"groups": {"all": {"accuracy": 0.4 if step == 12000 else 0.3}}},
        }
        for step in [*range(1000, 15001, 1000), 15531]
    ]
    return config, candidate, result, history, checkpoint


def test_matches_baseline_and_verifies_frozen_checkpoint(experiment):
    baseline, candidate, result, history, checkpoint = experiment
    report = audit_run(
        baseline,
        candidate,
        result,
        history,
        checkpoint_path=checkpoint,
        frozen_checkpoint_sha256=result["best_checkpoint_sha256"],
    )
    assert report["status"] == "passed"
    assert report["changed_training_files"] == ["train.parquet"]
    assert report["checkpoint_bytes_verified"] and report["frozen_checkpoint_verified"]


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "base_revision",
        "opentslm_commit",
        "packages",
        "precision",
        "prompt_version",
        "batch_size",
        "gradient_accumulation",
        "learning_rate",
        "warmup_fraction",
        "weight_decay",
        "gradient_clip",
        "max_length",
        "max_new_tokens",
        "max_steps",
        "val_every",
        "eval_batch_size",
        "selection_method",
        "selection_questions",
        "selection_limit",
        "trainer_source_sha256",
        "metrics_source_sha256",
        "total_parameters",
        "trainable_parameters",
        "init_sha256",
        "seed",
        "python",
    ],
)
def test_rejects_changed_invariant(experiment, field):
    baseline, candidate, *_ = experiment
    candidate[field] = "changed"
    with pytest.raises(ValueError, match=field):
        audit_config(baseline, candidate)


@pytest.mark.parametrize("change", ["missing", "extra", "selection", "selection_tokens", "truncated_tokens"])
def test_rejects_unexplained_config_difference(experiment, change):
    baseline, candidate, *_ = experiment
    if change == "missing":
        del candidate["python"]
    elif change == "extra":
        candidate["new_trainer_option"] = True
    elif change == "selection":
        candidate["data_files"]["selection.parquet"] = "f" * 64
    elif change == "selection_tokens":
        candidate["token_lengths"]["selection"]["max_prompt_tokens"] = 843
    else:
        candidate["token_lengths"]["train"]["would_truncate_rationale"] = 1
    with pytest.raises(ValueError):
        audit_config(baseline, candidate)


def test_changed_gpu_is_disclosed_and_signal_fingerprints_may_change(experiment):
    baseline, candidate, *_ = experiment
    candidate["gpu"] = "NVIDIA A100-SXM4-80GB"
    candidate["data_files"]["signals.npy"] = "b" * 64
    candidate["data_files"]["records.json"] = "c" * 64
    report = audit_config(baseline, candidate)
    assert report["gpu"]["candidate"] == "NVIDIA A100-SXM4-80GB"
    assert report["changed_training_files"] == ["records.json", "signals.npy", "train.parquet"]


def test_changed_train_requires_new_manifest(experiment):
    baseline, candidate, *_ = experiment
    candidate["manifest_sha256"] = baseline["manifest_sha256"]
    with pytest.raises(ValueError, match="new manifest"):
        audit_config(baseline, candidate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("smoke_only", True),
        ("max_steps", 15530),
        ("seed", 1),
        ("manifest_sha256", "b" * 64),
        ("distinct_truncated_rationales", 1),
        ("distinct_seen_training_questions", 0),
        ("distinct_seen_training_questions", 250000),
        ("best_step", 10000),
        ("best_selection_accuracy", 0.5),
        ("best_before_80_percent", False),
    ],
)
def test_rejects_incomplete_or_inconsistent_result(experiment, field, value):
    baseline, candidate, result, history, _ = experiment
    result[field] = value
    with pytest.raises(ValueError):
        audit_run(baseline, candidate, result, history)


def test_late_checkpoint_keeps_test_closed_even_if_flag_is_true(experiment):
    baseline, candidate, result, history, _ = experiment
    history[-1]["accuracy"] = 0.5
    history[-1]["scores"]["groups"]["all"]["accuracy"] = 0.5
    result.update(best_step=15531, best_selection_accuracy=0.5, best_before_80_percent=True)
    with pytest.raises(ValueError, match="keep test closed"):
        audit_run(baseline, candidate, result, history)


def test_tied_maximum_must_choose_first_checkpoint(experiment):
    baseline, candidate, result, history, _ = experiment
    history[0]["accuracy"] = 0.4
    history[0]["scores"]["groups"]["all"]["accuracy"] = 0.4
    with pytest.raises(ValueError, match="first selection-accuracy maximum"):
        audit_run(baseline, candidate, result, history)


def test_orchestrator_can_audit_completion_without_unlocking_late_checkpoint(experiment):
    baseline, candidate, result, history, _ = experiment
    history[-1]["accuracy"] = 0.5
    history[-1]["scores"]["groups"]["all"]["accuracy"] = 0.5
    result.update(best_step=15531, best_selection_accuracy=0.5, best_before_80_percent=False)
    report = audit_run(baseline, candidate, result, history, require_convergence=False)
    assert report["status"] == "needs_budget_review"
    assert report["test_eligible"] is False
    result["best_before_80_percent"] = True
    with pytest.raises(ValueError, match="guard disagrees"):
        audit_run(baseline, candidate, result, history, require_convergence=False)


@pytest.mark.parametrize("change", ["incomplete", "wrong_question_count", "nonfinite", "score_differs"])
def test_corrupt_history_cannot_unlock_evaluation(experiment, change):
    baseline, candidate, result, history, _ = experiment
    if change == "incomplete":
        history.pop()
    elif change == "wrong_question_count":
        history[0]["questions"] = 1999
    elif change == "nonfinite":
        history[0]["accuracy"] = float("nan")
    else:
        history[0]["scores"]["groups"]["all"]["accuracy"] = 0.9
    with pytest.raises(ValueError):
        audit_run(baseline, candidate, result, history)


def test_checkpoint_bytes_must_match_completed_and_frozen_run(experiment):
    baseline, candidate, result, history, checkpoint = experiment
    with pytest.raises(ValueError, match="frozen checkpoint"):
        audit_run(baseline, candidate, result, history, frozen_checkpoint_sha256="f" * 64)
    checkpoint.write_bytes(b"replacement checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        audit_run(baseline, candidate, result, history, checkpoint_path=checkpoint)
