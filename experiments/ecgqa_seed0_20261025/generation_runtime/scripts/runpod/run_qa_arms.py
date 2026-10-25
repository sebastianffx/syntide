"""Run seed 0 for three QA arms using the byte-identical archived baseline trainer.

The runtime must contain the baseline's syntide.baseline package and trainer.
No test labels are read until all four checkpoints pass the audit and are frozen.
Interrupted training is never silently overwritten; evaluation reuses saved
published predictions using the unchanged baseline evaluator.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
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
ARMS = ("real_qa", "paraphrase", "grounded", "grounded_acq")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    partial.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze_once(path, value):
    """Create an immutable protocol/claim, accepting only identical retries."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen identity changed: {path}")
    else:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


def train_command(prepared, run, baseline):
    command = [
        sys.executable,
        "-u",
        "-m",
        "syntide_trainers.opentslm_loop",
        "train",
        "--prepared",
        str(prepared),
        "--run-dir",
        str(run),
        "--seed",
        "0",
    ]
    for field in TRAIN_OPTIONS:
        command.extend(["--" + field.replace("_", "-"), str(baseline[field])])
    return command


def verify_runtime_sources(baseline):
    """Check source identity before the evaluator can open a held-out table."""
    for module, key in (
        ("syntide_trainers.opentslm_loop", "trainer_source_sha256"),
        ("syntide.baseline.metrics", "metrics_source_sha256"),
    ):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin or sha(spec.origin) != baseline[key]:
            raise ValueError(f"Runtime differs from baseline: {key}")


def verify_holdout_state(root):
    """A retry must retain the global freeze and claim for any test artifact."""
    freeze, claim = root / "freeze.json", root / "holdout_claim.json"
    artifacts = any(root.glob("*/seed0/test*"))
    if (claim.exists() and not freeze.exists()) or (artifacts and not (claim.exists() and freeze.exists())):
        raise ValueError("Existing holdout access lacks a matching global checkpoint freeze and claim")
    if claim.exists() and read(claim).get("freeze_sha256") != sha(freeze):
        raise ValueError("Holdout claim differs from global checkpoint freeze")


def execute(command, log, lock, *, baseline=None, config_path=None):
    from syntide.qa_baseline_audit import audit_config

    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
    with Path(log).open("a") as stream:
        child = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=env, pass_fds=(lock.fileno(),)
        )
        checked = False
        try:
            while child.poll() is None:
                if config_path and Path(config_path).exists() and not checked:
                    audit_config(baseline, read(config_path))
                    checked = True
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"Process exited {child.returncode}; see {log}")
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def make_summary(root, arm, baseline_majority, *, complete):
    from syntide.baseline.report import build_report

    directory = root / arm
    run = directory / "seed0"
    result = read(run / "train_result.json")
    config = read(run / "config.json")
    best = read(run / "best_selection_metrics.json")
    history = read(run / "selection_history.json")
    selected = max(history, key=lambda event: event["accuracy"])
    if best != selected["scores"]:
        raise ValueError(f"{arm}: saved best-selection metrics differ from the selected history event")
    prepared = Path(read(root / "protocol.json")["arms"][arm]["prepared"])
    summary = {
        "status": "single_seed_complete" if complete else "training_complete_evaluation_pending",
        "arm": arm,
        "completed_seeds": [0],
        "seed_sd": None,
        "training": result,
        "selection_method": config["selection_method"],
        "selection_questions": config["selection_questions"],
        "best_selection_metrics": {
            name: {key: value for key, value in group.items() if key != "per_template"}
            for name, group in best["groups"].items()
        },
        "final_selection_accuracy": history[-1]["accuracy"],
        "majority_comparison": read(root / "majority_selection.json"),
        "config_comparison": read(directory / "config_comparison.json"),
        "manifest_sha256": config["manifest_sha256"],
        "full_validation_generation": "pending_global_freeze",
        "test_generation": "closed_pending_global_freeze",
        "waveform_ablation": "pending_global_freeze",
    }
    if complete:
        verified = build_report(
            directory,
            read(prepared / "manifest.json"),
            baseline_majority,
            expected_seeds=(0,),
            allow_partial=False,
        )
        verified["arm"] = arm
        verified["training_composition"] = (
            read(root / "protocol.json").get("training_corpora", {}).get(arm, ["baseline_original"])
        )
        verified["deviations"] = [
            "Training retains the baseline original questions and adds the arm's declared QA corpora."
            if arm != "real_qa" and line.startswith("Training uses the real-only")
            else line
            for line in verified["deviations"]
        ]
        write(directory / "verified_report.json", verified)
        summary.update(
            full_validation_generation=read(run / "validation_metrics.json"),
            test_generation=read(run / "test_metrics.json"),
            waveform_ablation={
                split: read(run / f"{split}_zero_metrics.json") for split in ("validation", "test")
            },
        )
    write(directory / "summary.json", summary)
    return summary


def write_comparison_report(root, summaries):
    lines = [
        "# ECG-QA synthetic-data comparison, seed 0",
        "",
        "The original baseline checkpoint is reused. Each new arm has 15,531 updates of 16 examples. "
        "All runs use the same trainer, prompt, selection questions, initialization and model revision. "
        "There is one seed per arm; across-seed standard deviation is unavailable.",
        "",
        "| Arm | Selection accuracy | Clean | Noisy | Best step | Distinct training questions seen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, item in summaries.items():
        metrics, training = item["best_selection_metrics"], item["training"]
        values = [f"{100 * metrics[slice_]['accuracy']:.2f}%" for slice_ in ("all", "clean", "noisy")]
        lines.append(
            f"| {arm} | {' | '.join(values)} | {training['best_step']:,} | "
            f"{training['distinct_seen_training_questions']:,} |"
        )
    lines += [
        "",
        "Selection uses option likelihood. The following scores use greedy generation after "
        "the global checkpoint freeze.",
        "",
        "| Arm | Split | Slice | Accuracy | Template macro-F1 | Invalid rate | Zero ECG accuracy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for arm, item in summaries.items():
        for split, key in (("validation", "full_validation_generation"), ("test", "test_generation")):
            for group in ("all", "clean", "noisy"):
                score = item[key]["strict"]["groups"][group]
                zero = item["waveform_ablation"][split]["strict"]["groups"][group]["accuracy"]
                lines.append(
                    f"| {arm} | {split} | {group} | {100 * score['accuracy']:.2f}% | "
                    f"{100 * score['template_macro_f1']:.2f}% | "
                    f"{100 * score['invalid_rate']:.2f}% | {100 * zero:.2f}% |"
                )
    lines += [
        "",
        "Two-ECG comparisons and short cropped signals are excluded because the unchanged "
        "trainer accepts one 12-lead, 1,000-sample signal. Numeric training questions retain an empty "
        "options line. Prepared manifests record all exclusions. Equal updates imply unequal passes "
        "through the arm datasets.",
        "",
        "Each arm folder contains summary.json, seed0/config.json, seed0/train_result.json, "
        "config_comparison.json and verified predictions. The shared majority reference is computed "
        "once from baseline training rows and scored on the fixed selection subset.",
        "",
        "Observed account cost and pod termination are recorded by the local supervisor after "
        "checksum-verified collection; no unobserved per-arm charge is presented as measured cost.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def run_study(protocol, lock):
    import pyarrow.parquet as pq
    from syntide.baseline.data import verify_prepared
    from syntide.baseline.metrics import fit_majority_answers, majority_predictions, score_predictions

    from syntide.qa_baseline_audit import audit_config, audit_run

    root = Path(protocol["run_dir"])
    if set(protocol["arms"]) != set(ARMS) or protocol["seeds"] != [0]:
        raise ValueError("Protocol must contain precisely the four arms and seed 0")
    if type(protocol["generation_batch_size"]) is not int or protocol["generation_batch_size"] < 1:
        raise ValueError("generation_batch_size must be a positive integer")
    freeze_once(root / "protocol.json", protocol)
    baseline = read(root / "real_qa/seed0/config.json")
    audit_config(baseline, baseline)
    verify_runtime_sources(baseline)
    verify_holdout_state(root)
    baseline_prepared = Path(protocol["arms"]["real_qa"]["prepared"])
    manifests = {
        arm: verify_prepared(spec["prepared"], splits=("train", "selection"))
        for arm, spec in protocol["arms"].items()
    }
    for arm, manifest in manifests.items():
        for split in ("selection", "validation", "test"):
            filename = f"{split}.parquet"
            if manifest["files"][filename] != manifests["real_qa"]["files"][filename]:
                raise ValueError(f"{arm}: {split} bytes changed")
    majority = fit_majority_answers(pq.read_table(baseline_prepared / "train.parquet").to_pylist())
    selection = pq.read_table(baseline_prepared / "selection.parquet").to_pylist()
    freeze_once(
        root / "majority_selection.json", score_predictions(majority_predictions(selection, majority))
    )
    frozen = {}
    convergence = []
    for arm in ARMS:
        run = root / arm / "seed0"
        prepared = Path(protocol["arms"][arm]["prepared"])
        run.parent.mkdir(parents=True, exist_ok=True)
        if not (run / "train_result.json").exists():
            if (root / "freeze.json").exists() or (root / "holdout_claim.json").exists():
                raise ValueError("Cannot start training after checkpoint freeze or holdout claim")
            if arm == "real_qa":
                raise ValueError("Existing baseline seed 0 must be supplied; never retrain it")
            if (run / "config.json").exists():
                raise ValueError(f"{arm}: interrupted training requires explicit recovery; preserving files")
            write(
                root / "status.json",
                {"status": "running", "phase": "training", "arm": arm, "updated_unix": time.time()},
            )
            execute(
                train_command(prepared, run, baseline),
                run.parent / "train.log",
                lock,
                baseline=baseline,
                config_path=run / "config.json",
            )
        config, result = read(run / "config.json"), read(run / "train_result.json")
        comparison = audit_config(baseline, config)
        write(run.parent / "config_comparison.json", comparison)
        if config["manifest_sha256"] != manifests[arm]["sha256"]:
            raise ValueError(f"{arm}: dataset fingerprint differs")
        for name, digest in config["data_files"].items():
            if manifests[arm]["files"][name]["sha256"] != digest:
                raise ValueError(f"{arm}: data hash mismatch: {name}")
        # Audit module distinguishes identity/completion errors from the convergence guard.
        audit = audit_run(
            baseline,
            config,
            result,
            read(run / "selection_history.json"),
            checkpoint_path=run / "best.pt",
            require_convergence=False,
        )
        write(run.parent / "run_audit.json", audit)
        if not audit["test_eligible"]:
            convergence.append(arm)
        frozen[arm] = {
            "checkpoint_sha256": result["best_checkpoint_sha256"],
            "config_sha256": sha(run / "config.json"),
            "manifest_sha256": manifests[arm]["sha256"],
            "best_step": result["best_step"],
        }
        make_summary(root, arm, majority, complete=False)
    if convergence:
        write(
            root / "status.json",
            {
                "status": "needs_budget_review",
                "arms": convergence,
                "test_opened": False,
                "updated_unix": time.time(),
            },
        )
        return
    verify_runtime_sources(baseline)
    freeze_once(
        root / "freeze.json",
        {
            "protocol_sha256": sha(root / "protocol.json"),
            "checkpoints": frozen,
            "seeds": [0],
            "rule": "all arms complete, identical settings, first argmax, best before 80%",
        },
    )
    freeze_once(
        root / "holdout_claim.json",
        {
            "freeze_sha256": sha(root / "freeze.json"),
            "evaluations": "one real and one zero-waveform pass per split per arm",
        },
    )
    summaries = {}
    for arm in ARMS:
        run = root / arm / "seed0"
        if sha(run / "best.pt") != frozen[arm]["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after freeze")
        for split in ("validation", "test"):
            for zero in (False, True):
                label = split + ("_zero" if zero else "")
                claim = run / f"{label}_claim.json"
                if claim.exists() and read(claim).get("batch_size") != protocol["generation_batch_size"]:
                    raise ValueError(f"{arm}: existing {label} generation batch size differs from protocol")
                if split == "test" and claim.exists() and not (run / f"{label}_predictions.jsonl").exists():
                    raise ValueError(
                        f"{arm}: interrupted {label} claim requires review; preserving partial predictions "
                        "and refusing a repeated test pass"
                    )
                write(
                    root / "status.json",
                    {"status": "running", "phase": label, "arm": arm, "updated_unix": time.time()},
                )
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "syntide_trainers.opentslm_loop",
                    "evaluate",
                    "--prepared",
                    protocol["arms"][arm]["prepared"],
                    "--run-dir",
                    str(run),
                    "--split",
                    split,
                    "--batch-size",
                    str(protocol["generation_batch_size"]),
                ]
                if zero:
                    command.append("--zero-signal")
                execute(command, run.parent / f"{label}.log", lock)
        summaries[arm] = make_summary(root, arm, majority, complete=True)
    write(
        root / "comparison.json",
        {"seeds": [0], "seed_sd": None, "arms": summaries, "freeze_sha256": sha(root / "freeze.json")},
    )
    write_comparison_report(root, summaries)
    write(root / "status.json", {"status": "complete", "updated_unix": time.time()})


def main():
    protocol = read(sys.argv[1])
    root = Path(protocol["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run_study(protocol, lock)
        except BaseException as error:
            write(
                root / "status.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "updated_unix": time.time(),
                },
            )
            raise


if __name__ == "__main__":
    main()
