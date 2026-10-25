#!/usr/bin/env python3
"""One-shot, prospectively fixed original-exposure ablation and four-pass evaluation.

The outer queue owns the process group and absolute cutoff. This runner inherits
its guard-recognized lock and passes it to every child. Nothing is retrained or
regenerated on retry; an interrupted run requires inspection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PASSES = ("validation", "validation_zero", "test", "test_zero")
WORK_DEADLINE = "2026-10-25T22:56:47Z"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
    else:
        temp = path.with_suffix(path.suffix + ".partial")
        with temp.open("w") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        temp.replace(path)


def stamp(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def bindings(protocol):
    if "_expected_protocol_sha256" in protocol:
        require(sha(protocol["protocol_path"]) == protocol["_expected_protocol_sha256"], "Protocol changed during execution")
    for path, digest in protocol["bindings"].items():
        require(sha(path) == digest, f"Bound file changed: {path}")


def evaluation_budget(protocol, completed, rates, *, now=None):
    require(all(math.isfinite(x) and x > 0 for x in rates.values()), "Invalid generation timing")
    require({"selection", "selection_zero"}.issubset(rates), "Both selection calibrations are required")
    pending = {name: 27264 if name.startswith("validation") else 26841
               for name in PASSES if name not in completed}
    estimate = sum(pending.values()) * max(rates.values()) * 1.25 + 180*len(pending) + 1800
    available = stamp(protocol["work_deadline_utc"]) - (time.time() if now is None else now)
    return {"status": "admitted" if available >= estimate else "needs_time_budget_review",
            "pending_pass_questions": pending, "available_seconds": available,
            "required_seconds_estimate": estimate, "rates_seconds_per_question": rates,
            "safety_factor": 1.25, "seconds_per_pass_overhead": 180,
            "final_reserve_seconds": 1800, "deadline_utc": protocol["work_deadline_utc"]}


def execute(command, log, hold_fd, deadline):
    require(time.time() < deadline, "Ablation work deadline reached")
    with Path(log).open("x") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                 pass_fds=(hold_fd,))  # inherit outer registered group
        try:
            code = child.wait(timeout=max(1, deadline-time.time()))
            require(code == 0, f"Child exited {code}; inspect {log}")
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            raise


def audit_training(protocol, root, run):
    bindings(protocol)
    baseline = read(protocol["baseline_config"])
    config, result, history = [read(run/name) for name in
                               ("config.json", "train_result.json", "selection_history.json")]
    helper_path = Path(protocol["trainer_path"]).with_name("exposure_protocol.py")
    spec = importlib.util.spec_from_file_location("_exposure_contract", helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    helper.validate_baseline(config, baseline)
    require(result["status"] == "trained" and result["seed"] == 0
            and result["distinct_truncated_rationales"] == 0
            and result["distinct_seen_training_questions"] == 94501,
            "Training incomplete, missing sources, or target truncation")
    require(config["trainer_source_sha256"] == sha(protocol["trainer_path"])
            and config["exposure_helper_sha256"] == sha(helper_path)
            and config["source_manifest_sha256"] == sha(Path(protocol["prepared"])/"source-manifest.json"),
            "Training source bindings differ")
    checkpoint = run/"primary.pt"
    require(sha(checkpoint) == result["primary_checkpoint_sha256"], "Checkpoint SHA differs")
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(payload["config"] == config, "Checkpoint configuration differs")
    helper.validate_primary_checkpoint(config, result, payload, history)
    del payload
    # Independently reconstruct and compare the entire recorded source-draw stream.
    import pyarrow.parquet as pq
    rows = pq.read_table(Path(protocol["prepared"])/"train.parquet").to_pylist()
    sources = helper.validate_sources(rows, read(Path(protocol["prepared"])/"source-manifest.json"))
    for name, batch in (("original", 16), ("generated", 2)):
        stream = helper.PermutationStream(sources[name], 0)
        for _ in range(12000):
            stream.take(batch)
        require(stream.audit() == result["source_exposure"][name], f"Actual {name} exposure/order differs")
    audit = {"status": "passed", "checkpoint_policy": "fixed_original_exposure",
             "primary_step": 12000, "original_draws": 192000, "generated_draws": 24000,
             "draw_order_verified": True, "truncated_rationales": 0,
             "baseline_config_sha256": sha(protocol["baseline_config"]),
             "intentional_differences": protocol["intentional_differences"]}
    write(root/"run_audit.json", audit, exclusive=True)
    inventory = {str(run/name): sha(run/name) for name in
                 ("config.json", "train_result.json", "selection_history.json", "primary.pt")}
    freeze = {"protocol_sha256": sha(protocol["protocol_path"]), "inventory": inventory,
              "checkpoint_sha256": result["primary_checkpoint_sha256"],
              "primary_step": 12000, "seed": 0, "audit_sha256": sha(root/"run_audit.json")}
    write(root/"freeze.json", freeze, exclusive=True)
    return config, result, freeze


def verify_evaluation(run, label, config, result):
    value, claim = read(run/f"{label}_metrics.json"), read(run/f"{label}_claim.json")
    identity = claim["identity"]
    split, zero = label.replace("_zero", ""), label.endswith("_zero")
    count = 2000 if split == "selection" else 27264 if split == "validation" else 26841
    expected = {"checkpoint_sha256": result["primary_checkpoint_sha256"],
                "checkpoint_step": 12000, "manifest_sha256": config["manifest_sha256"],
                "questions": count, "seed": 0, "split": split, "zero_signal": zero}
    for key, target in expected.items():
        require(value[key] == identity[key] == target, f"Evaluation identity differs: {label}/{key}")
    require(identity["config_sha256"] == sha(run/"config.json")
            and identity["trainer_source_sha256"] == config["trainer_source_sha256"]
            and identity["metrics_source_sha256"] == config["metrics_source_sha256"]
            and identity["exposure_helper_sha256"] == config["exposure_helper_sha256"]
            and identity["max_new_tokens"] == 384 and identity["max_length"] == 1536
            and claim["batch_size"] == 256, "Evaluation settings differ")
    prediction = run/f"{label}_predictions.jsonl"
    require(value["prediction_sha256"] == sha(prediction)
            and value["claim_sha256"] == sha(run/f"{label}_claim.json"), "Evaluation artifact SHA differs")
    with prediction.open("rb") as stream:
        require(sum(1 for _ in stream) == count, "Incomplete published predictions")
    require(not value["metrics_recovered_from_predictions"], "Unexpected inference retry")
    return value


def inherited_preflight(protocol, prepared):
    source = read(Path(protocol["prepared"])/"source-manifest.json")
    audit = read(protocol["preparation_audit"])
    prior = read(protocol["prior_snapshot_preflight"])
    baseline = read(protocol["baseline_config"])
    arm = prior["arms"]["grounded_acq_gen"]
    require(audit["status"] == "verified" and audit["original_rows"] == 82829
            and audit["generated_rows"] == 11672 and audit["total_rows"] == 94501
            and audit["signals"] == 13105, "Prepared subset counts/audit differ")
    require(all(audit.get(k) is True for k in (
                "baseline_row_order_and_values_preserved", "baseline_signal_prefix_bitwise_identical",
                "generated_fields_unchanged_except_signal_index")), "Subset content-preservation proof missing")
    require(audit["prepared_manifest"]["sha256"] == sha(Path(protocol["prepared"])/"manifest.json")
            and audit["source_manifest"]["sha256"] == sha(Path(protocol["prepared"])/"source-manifest.json"),
            "Preparation audit binds different output files")
    scan = arm["full_training_token_scan"]
    require(prior["status"] == "passed" and scan["questions"] == 221394
            and scan["truncated_rationales"] == 0 and scan["max_encoded_tokens"] <= baseline["max_length"],
            "Prior complete training scan does not prove zero truncation")
    require(arm["expected_config"]["manifest_sha256"] == source["snapshot_generated_binding"]["prepared_manifest_content_sha256"],
            "Prior token scan covers a different snapshot")
    require(prior["baseline_config_sha256"] == sha(protocol["baseline_config"])
            and prior["trainer_source_sha256"] == baseline["trainer_source_sha256"]
            and prior["metrics_source_sha256"] == baseline["metrics_source_sha256"],
            "Prior token scan baseline/source binding differs")
    for key in ("max_length", "prompt_version", "base_revision", "model", "packages"):
        require(arm["expected_config"][key] == baseline[key], f"Inherited tokenizer/prompt differs: {key}")
    require(prepared["sha256"] == protocol["prepared_manifest_content_sha256"], "Prepared identity differs")
    return {"status": "passed", "basis": "byte/logical-identical subset of fully scanned snapshot",
            "prior_preflight_sha256": sha(protocol["prior_snapshot_preflight"]),
            "preparation_audit_sha256": sha(protocol["preparation_audit"]),
            "training_questions": 94501, "runtime_initialization_check": "enforced by trainer before first update"}


def run_ablation(protocol):
    require(protocol["schema"] == "syntide.exposure3030-run@1"
            and protocol["max_spend_usd"] == 40 and protocol["work_deadline_utc"] == WORK_DEADLINE,
            "Protocol scope or budget differs")
    bindings(protocol)
    root, run = Path(protocol["training_root"]), Path(protocol["training_root"])/"seed0"
    root.mkdir(exist_ok=True); run.mkdir(exist_ok=True)
    hold_fd = int(os.environ["ABLATION_HOLD_FD"])
    stat = os.fstat(hold_fd); expected = Path("/workspace/diffusets-study/generation/.writer.lock").stat()
    require((stat.st_dev, stat.st_ino) == (expected.st_dev, expected.st_ino), "Missing inherited guard lock")
    deadline = stamp(protocol["work_deadline_utc"])
    with (root/".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root/"launch-claim.json", {"protocol_sha256": sha(protocol["protocol_path"]),
              "started_utc": datetime.now(timezone.utc).isoformat()}, exclusive=True)
        # Existing full scan covered this exact subset; builder must prove content equivalence.
        prepared = read(Path(protocol["prepared"])/"manifest.json")
        write(root/"preflight.json", inherited_preflight(protocol, prepared), exclusive=True)
        write(root/"status.json", {"status": "running", "phase": "training"})
        command = [sys.executable, "-u", protocol["trainer_path"], "train", "--prepared", protocol["prepared"],
                   "--run-dir", str(run), "--baseline-config", protocol["baseline_config"],
                   "--source-manifest", str(Path(protocol["prepared"])/"source-manifest.json")]
        for key, val in protocol["train_arguments"].items():
            command += ["--"+key.replace("_", "-"), str(val)]
        execute(command, root/"train.log", hold_fd, deadline)
        config, result, freeze = audit_training(protocol, root, run)
        completed, rates, metrics = [], {}, {}
        for label in ("selection", "selection_zero", *PASSES):
            bindings(protocol)
            require(all(sha(path) == digest for path,digest in freeze["inventory"].items()), "Frozen checkpoint/data changed")
            require(not any((run/(label+suffix)).exists() for suffix in
                            ("_claim.json", "_metrics.json", "_predictions.jsonl", "_predictions.partial")),
                    f"Evaluation already claimed: {label}")
            if label in PASSES:
                budget = evaluation_budget(protocol, completed, rates)
                write(root/"evaluation-budget.json", budget)
                require(budget["status"] == "admitted", "Insufficient time for all remaining passes; test remains closed")
                if not (root/"holdout-claim.json").exists():
                    write(root/"holdout-claim.json", {"freeze_sha256": sha(root/"freeze.json")}, exclusive=True)
            else:
                require(deadline-time.time() > 1800+((2 if label == "selection" else 1)*(2000*.3*1.25+180)),
                        "Insufficient time for selection calibration")
            # Durable per-pass attempt claim forbids replay even if inference never wrote predictions.
            write(root/f"{label}-attempt.json", {"freeze_sha256": sha(root/"freeze.json"), "label": label}, exclusive=True)
            split = label.replace("_zero", "")
            command = [sys.executable, "-u", protocol["trainer_path"], "evaluate", "--prepared", protocol["prepared"],
                       "--run-dir", str(run), "--split", split, "--batch-size", "256"]
            if label.endswith("_zero"):
                command.append("--zero-signal")
            write(root/"status.json", {"status": "running", "phase": label})
            started = time.monotonic(); execute(command, root/f"{label}.log", hold_fd, deadline)
            wall = time.monotonic()-started
            value = verify_evaluation(run, label, config, result)
            metrics[label] = value
            if label in ("selection", "selection_zero", "validation", "validation_zero"):
                rates[label] = max(wall,value["elapsed_s"])/value["questions"]
            completed.append(label)
        summary = {"schema": "syntide.exposure3030-result@1", "status": "single_seed_complete",
                   "arm": "baseline_diffusets3030_exposure_matched", "seed": 0, "seed_sd": None,
                   "training": result, "config": config, "evaluation": metrics,
                   "audit": read(root/"run_audit.json"), "protocol_sha256": sha(protocol["protocol_path"])}
        write(root/"summary.json", summary, exclusive=True)
        write(root/"result.json", {"status": "complete", "summary_sha256": sha(root/"summary.json"),
              "freeze_sha256": sha(root/"freeze.json"), "completed_passes": list(PASSES)}, exclusive=True)
        write(root/"status.json", {"status": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    require(sha(args.protocol) == args.protocol_sha256, "Protocol SHA differs")
    protocol = read(args.protocol)
    require(Path(protocol["protocol_path"]).resolve() == args.protocol.resolve(), "Protocol path differs")
    protocol["_expected_protocol_sha256"] = args.protocol_sha256
    require(not (Path(protocol["training_root"])/"launch-claim.json").exists(), "Run was already claimed; no replay")
    try:
        run_ablation(protocol)
    except BaseException as error:
        root = Path(protocol["training_root"])
        if root.exists():
            write(root/"status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
