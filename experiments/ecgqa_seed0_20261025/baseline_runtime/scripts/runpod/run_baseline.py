"""Run the frozen three-seed experiment on RunPod and score holdout once.

Usage: PYTHONPATH=src:trainers/src:/workspace/OpenTSLM/src python scripts/runpod/run_baseline.py protocol.json
The protocol is selected from CUDA smoke timings before any test scoring. Restart
this command to recover: unfinished training attempts are preserved under
``incomplete/`` and restarted only before checkpoint freeze/holdout access.
"""

import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TRAIN_OPTIONS = (
    "max_steps",
    "batch_size",
    "gradient_accumulation",
    "val_every",
    "eval_batch_size",
    "max_length",
    "max_new_tokens",
    "selection_method",
    "base_revision",
    "learning_rate",
)


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def source_hashes():
    from syntide.baseline import metrics

    spec = importlib.util.find_spec("syntide_trainers.opentslm_loop")
    if spec is None or spec.origin is None:
        raise ValueError("Trainer source is unavailable on PYTHONPATH")
    return {
        "trainer_source_sha256": sha(Path(spec.origin)),
        "metrics_source_sha256": sha(Path(metrics.__file__)),
        "orchestrator_source_sha256": sha(Path(__file__)),
    }


def prepare_protocol(requested, root, manifest):
    """Make implicit defaults and source identities explicit before training."""
    protocol = {**requested, "run_dir": str(root), "prepared": str(Path(requested["prepared"]).resolve())}
    protocol.setdefault("learning_rate", 2e-4)
    protocol.setdefault("generation_batch_size", protocol.get("eval_batch_size"))
    if protocol.get("seeds") != [0, 1, 2]:
        raise ValueError("Baseline protocol requires seeds [0,1,2]")
    for key in TRAIN_OPTIONS:
        if key not in protocol:
            raise ValueError(f"Missing protocol setting: {key}")
    for key in TRAIN_OPTIONS[:7]:
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(protocol["generation_batch_size"]) is not int or protocol["generation_batch_size"] < 1:
        raise ValueError("generation_batch_size must be a positive integer")
    if protocol["selection_method"] not in {"generation", "option-nll"}:
        raise ValueError("Unknown checkpoint selection method")
    if not math.isfinite(protocol["learning_rate"]) or protocol["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not re.fullmatch(r"[0-9a-f]{40}", protocol["base_revision"]):
        raise ValueError(
            "base_revision must be the immutable 40-character revision resolved by the CUDA smoke"
        )
    for key, expected in {"manifest_sha256": manifest["sha256"], **source_hashes()}.items():
        if key in protocol and protocol[key] != expected:
            raise ValueError(f"Frozen protocol identity differs: {key}")
        protocol[key] = expected
    return protocol


def validate_seed(path, seed, protocol, manifest):
    """Validate completed training and checkpoint metadata without reading test data."""
    import torch

    result, config = read(path / "train_result.json"), read(path / "config.json")
    if result.get("status") != "trained" or result.get("smoke_only") or config.get("selection_limit") != 0:
        raise ValueError(f"Seed {seed}: incomplete or smoke run cannot fill a main seed")
    if result.get("seed") != seed or config.get("seed") != seed:
        raise ValueError(f"Seed {seed}: run belongs to another seed")
    for key in TRAIN_OPTIONS:
        if config.get(key) != protocol[key]:
            raise ValueError(f"Seed {seed}: protocol mismatch in {key}")
    for key in ("manifest_sha256", "trainer_source_sha256", "metrics_source_sha256"):
        if config.get(key) != protocol[key]:
            raise ValueError(f"Seed {seed}: source identity mismatch in {key}")
    if (
        result.get("manifest_sha256") != manifest["sha256"]
        or result.get("max_steps") != protocol["max_steps"]
    ):
        raise ValueError(f"Seed {seed}: training result manifest/budget mismatch")
    expected_files = {
        name: manifest["files"][name]["sha256"]
        for name in ("records.json", "signals.npy", "train.parquet", "selection.parquet")
    }
    if config.get("data_files") != expected_files:
        raise ValueError(f"Seed {seed}: training file identities differ")
    fixed = {
        "effective_batch_size": protocol["batch_size"] * protocol["gradient_accumulation"],
        "selection_questions": manifest["splits"]["selection"]["questions"],
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "gradient_clip": 1.0,
    }
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError(f"Seed {seed}: fixed training settings differ")
    history = read(path / "selection_history.json")
    expected_steps = list(range(protocol["val_every"], protocol["max_steps"] + 1, protocol["val_every"]))
    if not expected_steps or expected_steps[-1] != protocol["max_steps"]:
        expected_steps.append(protocol["max_steps"])
    if [event["step"] for event in history] != expected_steps:
        raise ValueError(f"Seed {seed}: selection history does not cover the fixed training budget")
    for event in history:
        if (
            not math.isfinite(event["accuracy"])
            or not 0 <= event["accuracy"] <= 1
            or event["questions"] != fixed["selection_questions"]
            or event["scores"]["groups"]["all"]["accuracy"] != event["accuracy"]
        ):
            raise ValueError(f"Seed {seed}: invalid checkpoint selection event")
    selected = max(history, key=lambda item: item["accuracy"])
    if result["best_step"] != selected["step"] or result["best_selection_accuracy"] != selected["accuracy"]:
        raise ValueError(f"Seed {seed}: best result is not the first validation argmax")
    if result["best_before_80_percent"] != (selected["step"] < 0.8 * protocol["max_steps"]):
        raise ValueError(f"Seed {seed}: recorded convergence flag is inconsistent")
    checkpoint = path / "best.pt"
    if sha(checkpoint) != result["best_checkpoint_sha256"]:
        raise ValueError(f"Seed {seed}: best checkpoint checksum mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload["config"] != config
        or payload["step"] != selected["step"]
        or payload["selection_accuracy"] != selected["accuracy"]
    ):
        raise ValueError(f"Seed {seed}: checkpoint configuration/selection differs from the completed run")
    return result, config


def archive_incomplete(root, path, seed):
    """Preserve interrupted training; never replace a frozen or exposed checkpoint."""
    if (root / "freeze.json").exists() or (root / "holdout_claim.json").exists():
        raise ValueError("Cannot restart training after checkpoint freeze or holdout claim")
    if any(root.glob("seed*/test*")):
        raise ValueError("Existing test artifacts prevent restarting training")
    if path.exists():
        archive = root / "incomplete" / f"seed{seed}-{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        path.rename(archive)
        with (root / "attempt_history.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "seed": seed,
                        "archived": str(archive),
                        "unix": time.time(),
                        "reason": "No completed train_result.json; restarting the same seed",
                    }
                )
                + "\n"
            )


def existing_evaluation(path, split, zero, seed, result, manifest, generation_batch_size):
    label = split + ("_zero" if zero else "")
    metrics_path, predictions_path = path / f"{label}_metrics.json", path / f"{label}_predictions.jsonl"
    claim_path = path / f"{label}_claim.json"
    if claim_path.exists() and read(claim_path).get("batch_size") != generation_batch_size:
        raise ValueError(f"{claim_path}: generation batch size differs from the frozen protocol")
    if not metrics_path.exists():
        return False
    if not predictions_path.is_file() or not claim_path.is_file():
        raise ValueError(f"{metrics_path}: metrics exist without predictions or provenance claim")
    expected = {
        "split": split,
        "zero_signal": zero,
        "seed": seed,
        "checkpoint_step": result["best_step"],
        "checkpoint_sha256": result["best_checkpoint_sha256"],
        "manifest_sha256": manifest["sha256"],
        "prediction_sha256": sha(predictions_path),
        "claim_sha256": sha(claim_path),
        "questions": manifest["splits"][split]["questions"],
    }
    saved = read(metrics_path)
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{metrics_path}: saved evaluation provenance differs from the frozen run")
    return True


def run_locked(requested, root, lock):
    from syntide.baseline.data import verify_prepared

    # Neither question table nor labels from validation/test are opened here.
    manifest = verify_prepared(requested["prepared"], splits=("train", "selection"))
    protocol = prepare_protocol(requested, root, manifest)
    saved = root / "protocol.json"
    if saved.exists() and read(saved) != protocol:
        raise ValueError("Frozen protocol differs; use a new run directory before any holdout access")
    if not saved.exists():
        write(saved, protocol)
    started = time.time()
    state = read(root / "status.json") if (root / "status.json").exists() else {}
    state.setdefault("started_unix", started)
    state.update(protocol_sha256=sha(saved), status="running", restarted_unix=started)
    state.pop("error", None)
    write(root / "status.json", state)
    executable = [sys.executable, "-u", "-m", "syntide_trainers.opentslm_loop"]
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}

    def run(command, log_name, phase, seed=None):
        state.update(phase=phase, seed=seed, updated_unix=time.time())
        write(root / "status.json", state)
        with (root / log_name).open("a") as log:
            # The child retains the flock if the coordinator is interrupted.
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, pass_fds=(lock.fileno(),)
            )

    try:
        seeds, results, configs = protocol["seeds"], [], []
        for seed in seeds:
            path = root / f"seed{seed}"
            if not (path / "train_result.json").exists():
                archive_incomplete(root, path, seed)
                command = [
                    *executable,
                    "train",
                    "--prepared",
                    protocol["prepared"],
                    "--run-dir",
                    str(path),
                    "--seed",
                    str(seed),
                ]
                for key in TRAIN_OPTIONS:
                    command += ["--" + key.replace("_", "-"), str(protocol[key])]
                run(command, f"seed{seed}_train.log", "training", seed)
            result, config = validate_seed(path, seed, protocol, manifest)
            comparison = {key: value for key, value in config.items() if key not in {"seed", "init_sha256"}}
            if configs and comparison != configs[0]:
                raise ValueError("Seed configurations differ beyond seed and initial weights")
            configs.append(comparison)
            results.append(result)
            if result["best_step"] >= 0.8 * protocol["max_steps"]:
                state.update(
                    status="needs_budget_review",
                    phase="convergence",
                    results=results,
                    seeds_remaining=seeds[len(results) :],
                    updated_unix=time.time(),
                )
                write(root / "status.json", state)
                print(
                    f"Seed {seed} has a late best step. Remaining seeds are deferred; "
                    "test stays closed pending a shared-budget review and rerun of all seeds.",
                    flush=True,
                )
                return
        current_sources = source_hashes()
        if current_sources != {key: protocol[key] for key in current_sources}:
            raise ValueError("Experiment source changed during training")
        frozen = {
            "protocol_sha256": sha(saved),
            "seeds": results,
            "created_unix": time.time(),
            "rule": "All three seeds complete; first validation accuracy argmax; test previously unused.",
        }
        freeze_path, claim_path = root / "freeze.json", root / "holdout_claim.json"
        if freeze_path.exists():
            old = read(freeze_path)
            if old["protocol_sha256"] != frozen["protocol_sha256"] or old["seeds"] != results:
                raise ValueError("Frozen checkpoint set changed")
        else:
            if claim_path.exists() or any(root.glob("seed*/test*")):
                raise ValueError("Existing holdout access has no matching checkpoint freeze")
            write(freeze_path, frozen)
        claim = {
            "freeze_sha256": sha(freeze_path),
            "claimed_unix": time.time(),
            "evaluations": "one real and one fixed-text zero-signal prediction per test question per seed",
        }
        if claim_path.exists():
            if read(claim_path)["freeze_sha256"] != claim["freeze_sha256"]:
                raise ValueError("Holdout has already been claimed by different checkpoints")
        else:
            if any(root.glob("seed*/test*")):
                raise ValueError("Test artifacts exist without a holdout claim")
            write(claim_path, claim)  # Atomic replace is exclusive under the inherited run lock.
        for seed, result in zip(seeds, results, strict=True):
            path = root / f"seed{seed}"
            for split in ("validation", "test"):
                for zero in (False, True):
                    label = split + ("_zero" if zero else "")
                    if existing_evaluation(
                        path, split, zero, seed, result, manifest, protocol["generation_batch_size"]
                    ):
                        continue
                    command = [
                        *executable,
                        "evaluate",
                        "--prepared",
                        protocol["prepared"],
                        "--run-dir",
                        str(path),
                        "--split",
                        split,
                        "--batch-size",
                        str(protocol["generation_batch_size"]),
                    ]
                    if zero:
                        command += ["--zero-signal"]
                    # evaluate recovers missing metrics from published predictions without regenerating them.
                    run(command, f"seed{seed}_{label}.log", label, seed)
        run(
            [
                sys.executable,
                "-m",
                "syntide.baseline.report",
                "--run-dir",
                str(root),
                "--data-dir",
                protocol["prepared"],
                "--output",
                str(root / "report"),
            ],
            "report.log",
            "reporting",
        )
        state.update(
            status="complete",
            phase="complete",
            updated_unix=time.time(),
            wall_seconds=time.time() - state["started_unix"],
        )
        write(root / "status.json", state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_unix=time.time())
        write(root / "status.json", state)
        raise


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/runpod/run_baseline.py protocol.json")
    requested = read(Path(sys.argv[1]).resolve())
    root = Path(requested["run_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(
                "This run directory already has an active coordinator or training process"
            ) from error
        run_locked(requested, root, lock)


if __name__ == "__main__":
    main()
