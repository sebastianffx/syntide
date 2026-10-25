#!/usr/bin/env python3
"""Selection-calibrated, once-only continuation of an already frozen snapshot.

All scientific inputs and the original runner remain byte-identical. Only the
private runner instance's runtime estimate is replaced, using two completed
selection generation measurements. No accuracy is used by this timing policy.
The outer successor watchdog owns this process group and its absolute cutoff.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


DEADLINE = "2026-10-25T01:12:47Z"
PROTOCOL_SHA = "eef0fea26c80ee00cefeaeb10c9cb6324e72c3163ac85c4d9dcde900ccd250c9"
RUNNER_SHA = "a36d43b84f77ff0898d0479fd94d4fb75fa796b721073c49af48a498af56bede"
PASS_LABELS = ("validation", "validation_zero", "test", "test_zero")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def seconds(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def measured_budget(protocol, run, manifest, rates, *, now=None, operational_override=None):
    """Preserve all original margins; completed validation may only raise rate."""
    require(operational_override is None, "A deadline override is forbidden")
    budget = protocol["evaluation_budget"]
    require(budget["deadline_utc"] == DEADLINE, "Snapshot cutoff differs")
    require(budget["safety_factor"] == 1.25
            and budget["seconds_per_pass_overhead"] == 180
            and budget["final_reserve_seconds"] == 1800, "Original margins differ")
    observed = dict(rates)
    require(set(observed) == {"selection_real_wall", "selection_zero_wall"}, "Both calibrations required")
    for label in ("validation", "validation_zero"):
        path = run / f"{label}_metrics.json"
        if path.exists():
            metric = read(path)
            require(metric["questions"] == manifest["splits"]["validation"]["questions"],
                    "Completed validation question count differs")
            observed[label] = metric["elapsed_s"] / metric["questions"]
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in observed.values()),
            "Invalid timing measurement")
    pending = {label: manifest["splits"][label.split("_")[0]]["questions"]
               for label in PASS_LABELS if not (run / f"{label}_predictions.jsonl").exists()}
    needed = sum(pending.values()) * max(observed.values()) * 1.25 + len(pending) * 180 + 1800
    available = seconds(DEADLINE) - (time.time() if now is None else now)
    return {"status": "sufficient_estimated_time" if available >= needed else "needs_time_budget_review",
            "deadline_utc": DEADLINE, "scientific_deadline_utc": budget["deadline_utc"],
            "available_seconds": available, "required_seconds_estimate": needed,
            "pending_pass_questions": pending, "rate_inputs_seconds_per_question": observed,
            "safety_factor": 1.25, "seconds_per_pass_overhead": 180, "final_reserve_seconds": 1800,
            "initial_rate_is_measured": True, "superseded_planning_rate": budget["seconds_per_question"],
            "policy": "maximum frozen-checkpoint full-selection real/zero end-to-end rate, raised by validation",
            "limitation": "Selection-based estimate with margin, not a guaranteed full-pass bound; no score-based decisions."}


def check_bindings(config):
    for path, digest in config["bindings"].items():
        require(sha(path) == digest, f"Bound file changed: {path}")


def closed_passes(run):
    for label in PASS_LABELS:
        require(not any(run.glob(label + "_*")), f"Held-out pass already started: {label}")


def immutable_inventory(protocol, root, run):
    paths = [root / name for name in ("protocol.json", "preflight.json", "freeze.json", "holdout_claim.json")]
    paths += [run / name for name in ("config.json", "train_result.json", "selection_history.json", "best.pt")]
    paths += [Path(protocol["arms"]["grounded_acq_gen"]["prepared"]) / "manifest.json"]
    return {str(path): sha(path) for path in paths}


def unchanged(inventory):
    require(all(sha(path) == digest for path, digest in inventory.items()), "Frozen scientific artifacts changed")


def validate_frozen(runner, protocol, root, run):
    """Model-free full audit before any calibration or held-out access."""
    from syntide.qa_baseline_audit import audit_run

    runner.validate_protocol(protocol)
    baseline = runner.verify_baseline(protocol)
    manifest = runner.verify_data(protocol, baseline)
    runner.check_preflight(protocol, baseline, manifest)
    runner.shared.verify_holdout_state(root)
    frozen = read(root / "freeze.json")
    require(frozen["protocol_sha256"] == sha(root / "protocol.json")
            and frozen["preflight_sha256"] == sha(root / "preflight.json")
            and frozen["seeds"] == [0], "Freeze identity differs")
    require(read(root / "protocol.json") == protocol, "Frozen protocol differs")
    result = read(run / "train_result.json")
    require(result["status"] == "trained" and result["max_steps"] == 15531
            and result["seed"] == 0 and result["distinct_truncated_rationales"] == 0,
            "Fixed-budget training incomplete")
    digest = frozen["checkpoints"]["grounded_acq_gen"]["checkpoint_sha256"]
    require(digest == result["best_checkpoint_sha256"], "Checkpoint freeze/result mismatch")
    audit = audit_run(baseline, read(run / "config.json"), result, read(run / "selection_history.json"),
                      checkpoint_path=run / "best.pt", frozen_checkpoint_sha256=digest)
    require(audit["test_eligible"] and audit["status"] == "passed", "Convergence gate did not pass")
    require(manifest["splits"]["selection"]["questions"] == 2000, "Selection must contain all 2000 questions")
    return manifest, audit


def calibrate(runner, config, protocol, root, run, output, lock, inventory):
    """Evaluate only selection with the unchanged evaluator and exact settings."""
    records = []
    for zero in (False, True):
        check_bindings(config)
        unchanged(inventory)
        closed_passes(run)
        # Conservative allowance for calibration itself, preserving the final reserve.
        remaining_calibrations = 1 if zero else 2
        require(seconds(DEADLINE) - time.time() >= remaining_calibrations * (2000 * .3 * 1.25 + 180) + 1800,
                "Insufficient time even for bounded calibration; test stays closed")
        name = "selection_zero" if zero else "selection"
        directory = output / ("calibration-zero" if zero else "calibration-real")
        directory.mkdir()  # Never overwrite/replay a partial calibration automatically.
        for filename in ("config.json", "train_result.json", "selection_history.json", "best.pt"):
            shutil.copy2(run / filename, directory / filename)
            require(sha(directory / filename) == inventory[str(run / filename)], "Calibration copy differs")
        command = [sys.executable, "-u", "-m", "syntide_trainers.opentslm_loop", "evaluate", "--prepared",
                   protocol["arms"]["grounded_acq_gen"]["prepared"], "--run-dir", str(directory),
                   "--split", "selection", "--batch-size", "256"]
        if zero:
            command.append("--zero-signal")
        write(output / "continuation-status.json", {"status": "calibrating", "split": "selection", "zero": zero})
        started = time.monotonic()
        runner.shared.execute(command, output / f"{name}-calibration.log", lock)
        wall = time.monotonic() - started
        metric = read(directory / f"{name}_metrics.json")
        claim_path = directory / f"{name}_claim.json"
        claim = read(claim_path)
        identity = claim["identity"]
        frozen_config = read(run / "config.json")
        frozen_result = read(run / "train_result.json")
        require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                    for value in (wall, metric["elapsed_s"])), "Invalid calibration duration")
        require(all(identity[key] == frozen_config[key] for key in (
                    "manifest_sha256", "seed", "max_length", "max_new_tokens",
                    "trainer_source_sha256", "metrics_source_sha256"))
                and identity["checkpoint_step"] == frozen_result["best_step"],
                "Calibration claim configuration differs")
        require(metric["questions"] == identity["questions"] == 2000
                and metric["split"] == identity["split"] == "selection"
                and metric["zero_signal"] == identity["zero_signal"] == zero
                and claim["batch_size"] == 256 and identity["max_new_tokens"] == 384
                and identity["checkpoint_sha256"] == inventory[str(run / "best.pt")]
                and identity["config_sha256"] == inventory[str(run / "config.json")]
                and metric["checkpoint_sha256"] == identity["checkpoint_sha256"]
                and metric["manifest_sha256"] == frozen_config["manifest_sha256"]
                and not metric["metrics_recovered_from_predictions"]
                and metric["prediction_sha256"] == sha(directory / f"{name}_predictions.jsonl")
                and metric["claim_sha256"] == sha(claim_path), "Calibration provenance differs")
        rate = max(wall, metric["elapsed_s"]) / 2000
        require(math.isfinite(rate) and rate > 0, "Invalid calibration duration")
        unchanged(inventory)
        records.append({"zero": zero, "questions": 2000, "wall_seconds": wall,
                        "evaluator_seconds": metric["elapsed_s"], "seconds_per_question": rate,
                        "claim_sha256": sha(claim_path), "metrics_sha256": sha(directory / f"{name}_metrics.json"),
                        "prediction_sha256": metric["prediction_sha256"], "config_sha256": identity["config_sha256"],
                        "checkpoint_sha256": identity["checkpoint_sha256"], "batch_size": 256, "max_new_tokens": 384})
    return records


def continue_frozen(config, output):
    check_bindings(config)
    require(sha(config["protocol_path"]) == PROTOCOL_SHA and sha(config["runner_path"]) == RUNNER_SHA,
            "Unexpected snapshot source or protocol")
    require(config["deadline_utc"] == DEADLINE and config["max_spend_usd"] == 40, "Runtime limits changed")
    protocol = read(config["protocol_path"])
    require(protocol["evaluation_budget"]["deadline_utc"] == DEADLINE
            and protocol["generation_batch_size"] == 256, "Scientific evaluation settings changed")
    root = Path(protocol["run_dir"])
    run = root / "grounded_acq_gen/seed0"
    spec = importlib.util.spec_from_file_location("_timing_frozen_snapshot", config["runner_path"])
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with (root / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(read(root / "status.json")["status"] == "needs_time_budget_review", "Not a natural timing-review return")
        closed_passes(run)
        manifest, audit = validate_frozen(runner, protocol, root, run)
        inventory = immutable_inventory(protocol, root, run)
        # One durable attempt; any interruption requires inspection, never replay.
        with (output / "continuation-claim.json").open("x") as stream:
            json.dump({"started_utc": dt.datetime.now(dt.UTC).isoformat(), "inventory": inventory,
                       "configuration": config}, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        original_execute = runner.shared.execute

        def no_training(*args, **kwargs):
            raise RuntimeError("Training is forbidden in a frozen-checkpoint continuation")

        def evaluate_only(command, *args, **kwargs):
            require(command[:5] == [sys.executable, "-u", "-m", "syntide_trainers.opentslm_loop", "evaluate"],
                    "Only unchanged evaluation commands are allowed")
            unchanged(inventory)
            check_bindings(config)
            return original_execute(command, *args, **kwargs)

        runner.shared.train_command = no_training
        runner.shared.execute = evaluate_only
        records = calibrate(runner, config, protocol, root, run, output, lock, inventory)
        rates = {"selection_zero_wall" if x["zero"] else "selection_real_wall": x["seconds_per_question"] for x in records}
        policy = {"status": "calibrated", "calibration": records, "inventory": inventory,
                  "deadline_utc": DEADLINE, "cap_usd": 40, "safety_factor": 1.25,
                  "seconds_per_pass_overhead": 180, "final_reserve_seconds": 1800,
                  "original_rate_assumption": .3, "scores_used_for_timing_decisions": False,
                  "scientific_configuration_changed": False, "run_audit": audit}
        write(output / "calibrated-policy.json", policy)
        policy_sha = sha(output / "calibrated-policy.json")

        def budget_adapter(protocol_arg, run_arg, manifest_arg, *, now=None, operational_override=None):
            require(sha(output / "calibrated-policy.json") == policy_sha, "Calibration policy changed")
            result = measured_budget(protocol_arg, run_arg, manifest_arg, rates, now=now,
                                     operational_override=operational_override)
            result["operational_timing_policy_sha256"] = policy_sha
            result["operational_timing_policy_path"] = str(output / "calibrated-policy.json")
            return result

        runner.evaluation_budget = budget_adapter
        initial = budget_adapter(protocol, run, manifest)
        write(output / "admission.json", initial)
        if initial["status"] != "sufficient_estimated_time":
            write(output / "continuation-result.json", {"status": "needs_time_budget_review", "test_opened": False,
                                                       "budget": initial, "checkpoint_retained": True})
            return
        closed_passes(run)
        unchanged(inventory)
        runner.run_study(protocol, lock)
        unchanged(inventory)
        status = read(root / "status.json")
        write(output / "continuation-result.json", {"status": status["status"], "scientific_status": status,
                                                   "test_opened": any(run.glob("test*_claim.json")),
                                                   "checkpoint_retained": True, "inventory_verified": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.config) == args.config_sha256, "Continuation config changed")
    config = read(args.config)
    output = Path(config["root"])
    require(output == args.config.resolve().parent, "Unexpected operational output")
    require(config["bindings"].get(str(Path(__file__).resolve())) == sha(__file__), "Adapter not pinned")
    try:
        continue_frozen(config, output)
    except BaseException as exc:
        write(output / "continuation-result.json", {"status": "failed_preserved", "error": f"{type(exc).__name__}: {exc}",
                                                   "checkpoint_retained": True, "automatic_retry_allowed": False})
        raise


if __name__ == "__main__":
    main()
