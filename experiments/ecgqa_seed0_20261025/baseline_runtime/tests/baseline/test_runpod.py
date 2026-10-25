"""RunPod coordinator tests using fake artifacts/subprocesses; no model execution."""

import fcntl
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/runpod/run_baseline.py"
SPEC = importlib.util.spec_from_file_location("runpod_baseline_coordinator", SCRIPT)
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    sources = {
        "trainer_source_sha256": "trainer",
        "metrics_source_sha256": "metrics",
        "orchestrator_source_sha256": "coordinator",
    }
    monkeypatch.setattr(coordinator, "source_hashes", lambda: sources)
    manifest = {
        "sha256": "dataset",
        "splits": {"selection": {"questions": 4}},
        "files": {
            name: {"sha256": name}
            for name in ("records.json", "signals.npy", "train.parquet", "selection.parquet")
        },
    }
    requested = {
        "run_dir": str(tmp_path),
        "prepared": str(tmp_path / "prepared"),
        "seeds": [0, 1, 2],
        "max_steps": 10,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "val_every": 5,
        "eval_batch_size": 256,
        "max_length": 1024,
        "max_new_tokens": 384,
        "selection_method": "option-nll",
        "base_revision": "b" * 40,
    }
    protocol = coordinator.prepare_protocol(requested, tmp_path, manifest)
    return SimpleNamespace(root=tmp_path, manifest=manifest, requested=requested, protocol=protocol)


def test_protocol_records_implicit_learning_rate_and_requires_immutable_model_revision(experiment):
    assert experiment.protocol["learning_rate"] == 2e-4
    assert experiment.protocol["generation_batch_size"] == 256
    assert experiment.protocol["manifest_sha256"] == experiment.manifest["sha256"]
    with pytest.raises(ValueError, match="immutable 40-character revision"):
        coordinator.prepare_protocol(
            {**experiment.requested, "base_revision": "main"}, experiment.root, experiment.manifest
        )


def test_generation_batch_must_be_positive_and_claim_recovery_must_match(experiment):
    with pytest.raises(ValueError, match="generation_batch_size must be a positive integer"):
        coordinator.prepare_protocol(
            {**experiment.requested, "generation_batch_size": 0}, experiment.root, experiment.manifest
        )
    coordinator.write(experiment.root / "test_claim.json", {"batch_size": 256})
    with pytest.raises(ValueError, match="generation batch size differs"):
        coordinator.existing_evaluation(experiment.root, "test", False, 0, {}, experiment.manifest, 512)


def test_interrupted_training_is_preserved_before_restart(experiment):
    path = experiment.root / "seed0"
    path.mkdir()
    (path / "config.json").write_text("original configuration")
    coordinator.archive_incomplete(experiment.root, path, 0)
    archives = list((experiment.root / "incomplete").glob("seed0-*"))
    assert len(archives) == 1
    assert (archives[0] / "config.json").read_text() == "original configuration"
    assert not path.exists()
    assert json.loads((experiment.root / "attempt_history.jsonl").read_text())["seed"] == 0


@pytest.mark.parametrize("guard", ["freeze.json", "holdout_claim.json", "seed1/test.partial"])
def test_exposed_or_frozen_training_is_never_archived(experiment, guard):
    path = experiment.root / "seed0"
    path.mkdir()
    evidence = experiment.root / guard
    evidence.parent.mkdir(exist_ok=True)
    evidence.write_text("evidence must remain untouched")
    with pytest.raises(ValueError, match=r"Cannot restart training|Existing test artifacts"):
        coordinator.archive_incomplete(experiment.root, path, 0)
    assert path.is_dir()
    assert not (experiment.root / "incomplete").exists()


@pytest.fixture
def completed_seed(experiment, monkeypatch):
    protocol, manifest = experiment.protocol, experiment.manifest
    path = experiment.root / "seed0"
    path.mkdir()
    config = {key: protocol[key] for key in coordinator.TRAIN_OPTIONS}
    config.update(
        {
            "seed": 0,
            "selection_limit": 0,
            "effective_batch_size": 4,
            "selection_questions": 4,
            "weight_decay": 0.01,
            "warmup_fraction": 0.1,
            "gradient_clip": 1.0,
            "manifest_sha256": "dataset",
            "trainer_source_sha256": "trainer",
            "metrics_source_sha256": "metrics",
            "data_files": {key: value["sha256"] for key, value in manifest["files"].items()},
        }
    )
    history = [
        {
            "step": step,
            "accuracy": accuracy,
            "questions": 4,
            "scores": {"groups": {"all": {"accuracy": accuracy}}},
        }
        for step, accuracy in ((5, 0.6), (10, 0.5))
    ]
    (path / "best.pt").write_bytes(b"fake checkpoint bytes")
    result = {
        "status": "trained",
        "seed": 0,
        "smoke_only": False,
        "max_steps": 10,
        "manifest_sha256": "dataset",
        "best_step": 5,
        "best_selection_accuracy": 0.6,
        "best_before_80_percent": True,
        "best_checkpoint_sha256": coordinator.sha(path / "best.pt"),
    }
    payload = {"config": config, "step": 5, "selection_accuracy": 0.6}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *args, **kwargs: payload))
    coordinator.write(path / "config.json", config)
    coordinator.write(path / "train_result.json", result)
    coordinator.write(path / "selection_history.json", history)
    return SimpleNamespace(**vars(experiment), path=path, config=config, result=result, payload=payload)


def test_completed_seed_checks_frozen_checkpoint_metadata(completed_seed):
    seed = completed_seed
    result, config = coordinator.validate_seed(seed.path, 0, seed.protocol, seed.manifest)
    assert result == seed.result and config == seed.config
    seed.payload["step"] = 10
    with pytest.raises(ValueError, match="checkpoint configuration/selection differs"):
        coordinator.validate_seed(seed.path, 0, seed.protocol, seed.manifest)


@pytest.mark.parametrize("field,value", [("seed", 1), ("eval_batch_size", 16), ("learning_rate", 1e-4)])
def test_reused_seed_must_match_seed_learning_rate_and_evaluation_batch(completed_seed, field, value):
    seed = completed_seed
    coordinator.write(seed.path / "config.json", {**seed.config, field: value})
    with pytest.raises(ValueError, match=r"another seed|protocol mismatch"):
        coordinator.validate_seed(seed.path, 0, seed.protocol, seed.manifest)


def test_convergence_is_recomputed_instead_of_trusting_result_flag(completed_seed):
    seed = completed_seed
    coordinator.write(seed.path / "train_result.json", {**seed.result, "best_before_80_percent": False})
    with pytest.raises(ValueError, match="convergence flag is inconsistent"):
        coordinator.validate_seed(seed.path, 0, seed.protocol, seed.manifest)


@pytest.mark.parametrize("late", [False, True])
def test_all_seed_validation_precedes_holdout_and_late_best_keeps_test_closed(experiment, monkeypatch, late):
    from syntide.baseline import data

    validations, commands, prepared_splits = [], [], []
    experiment.requested["generation_batch_size"] = 512

    def prepared(path, *, splits):
        prepared_splits.append(splits)
        return experiment.manifest

    def validate(path, seed, protocol, manifest):
        validations.append(seed)
        return {"seed": seed, "best_step": 9 if late else 5}, {"seed": seed, "init_sha256": str(seed)}

    def subprocess(command, **kwargs):
        assert validations == [0, 1, 2]
        assert (experiment.root / "freeze.json").is_file()
        assert (experiment.root / "holdout_claim.json").is_file()
        assert kwargs["pass_fds"]
        commands.append(command)

    monkeypatch.setattr(data, "verify_prepared", prepared)
    monkeypatch.setattr(coordinator, "validate_seed", validate)
    monkeypatch.setattr(coordinator.subprocess, "run", subprocess)
    for seed in (0, 1, 2):
        path = experiment.root / f"seed{seed}"
        path.mkdir()
        (path / "train_result.json").write_text("{}")
    with (experiment.root / ".run.lock").open("a+") as lock:
        coordinator.run_locked(experiment.requested, experiment.root, lock)
    assert prepared_splits == [("train", "selection")]
    status = coordinator.read(experiment.root / "status.json")
    if late:
        assert status["status"] == "needs_budget_review"
        assert validations == [0]
        assert status["results"] == [{"seed": 0, "best_step": 9}]
        assert status["seeds_remaining"] == [1, 2]
        assert not commands
        assert not (experiment.root / "freeze.json").exists()
        assert not (experiment.root / "holdout_claim.json").exists()
    else:
        assert status["status"] == "complete"
        assert len(commands) == 13  # four evaluations per seed, followed by the report
        assert all(command[command.index("--batch-size") + 1] == "512" for command in commands[:12])
        protocol = coordinator.read(experiment.root / "protocol.json")
        assert protocol["eval_batch_size"] == 256
        assert protocol["generation_batch_size"] == 512


def test_duplicate_coordinator_exits_before_touching_data(experiment, monkeypatch):
    request = experiment.root / "requested.json"
    coordinator.write(request, experiment.requested)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(request)])
    with (experiment.root / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit, match="already has an active coordinator"):
            coordinator.main()
