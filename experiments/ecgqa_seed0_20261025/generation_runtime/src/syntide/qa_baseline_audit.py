"""Strict, model-free comparison with the frozen single-seed ECG-QA baseline.

This is deliberately separate from the checksum-pinned baseline trainer. It
does not open held-out questions. Call ``audit_config`` when training writes its
configuration, then ``audit_run`` before freezing any compared checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

_DATA_FILES = {"train.parquet", "selection.parquet", "records.json", "signals.npy"}
_VARIABLE_CONFIG = {"manifest_sha256", "data_files", "token_lengths", "gpu"}
_REQUIRED_CONFIG = {
    "model",
    "base_revision",
    "opentslm_commit",
    "packages",
    "precision",
    "prompt_version",
    "batch_size",
    "gradient_accumulation",
    "effective_batch_size",
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
    *_VARIABLE_CONFIG,
}
_PROTOCOL = {
    "seed": 0,
    "max_steps": 15531,
    "batch_size": 8,
    "gradient_accumulation": 2,
    "effective_batch_size": 16,
    "learning_rate": 2e-4,
    "warmup_fraction": 0.1,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "max_length": 1536,
    "max_new_tokens": 384,
    "val_every": 1000,
    "eval_batch_size": 256,
    "selection_method": "option-nll",
    "selection_questions": 2000,
    "selection_limit": 0,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _check(issues: list[str]) -> None:
    if issues:
        raise ValueError("Baseline comparison failed: " + "; ".join(issues))


def audit_config(baseline: dict, candidate: dict) -> dict:
    """Reject any unexplained configuration change, including new/absent fields.

    Different GPU hardware is recorded, while Python, recorded package versions,
    initial weights and all training settings must match. Training data may
    change; the selection file and its token summary must remain identical.
    """
    issues = []
    for label, config in (("baseline", baseline), ("candidate", candidate)):
        missing = _REQUIRED_CONFIG - config.keys()
        if missing:
            issues.append(f"{label} missing config fields: {sorted(missing)}")
        for key, expected in _PROTOCOL.items():
            if config.get(key) != expected:
                issues.append(f"{label} {key} must be {expected!r}")
        for key in ("manifest_sha256", "trainer_source_sha256", "metrics_source_sha256", "init_sha256"):
            if not _digest(config.get(key)):
                issues.append(f"{label} invalid SHA-256 in {key}")
        files = config.get("data_files", {})
        if set(files) != _DATA_FILES or any(not _digest(value) for value in files.values()):
            issues.append(f"{label} invalid training data file fingerprints")
        lengths = config.get("token_lengths", {})
        if set(lengths) != {"train", "selection"}:
            issues.append(f"{label} missing train/selection token summaries")
        for split, summary in lengths.items():
            if summary.get("would_truncate_rationale") != 0 or summary.get("sample_questions", 0) <= 0:
                issues.append(f"{label} {split} token summary has truncation or no samples")
    if set(baseline) != set(candidate):
        issues.append("configuration field inventory differs")
    equal_fields = sorted((set(baseline) | set(candidate)) - _VARIABLE_CONFIG)
    for key in equal_fields:
        if baseline.get(key) != candidate.get(key):
            issues.append(f"config field differs: {key}")
    if baseline.get("data_files", {}).get("selection.parquet") != candidate.get("data_files", {}).get(
        "selection.parquet"
    ):
        issues.append("selection.parquet fingerprint differs")
    if baseline.get("token_lengths", {}).get("selection") != candidate.get("token_lengths", {}).get(
        "selection"
    ):
        issues.append("selection token summary differs")
    train_changed = baseline.get("data_files", {}).get("train.parquet") != candidate.get(
        "data_files", {}
    ).get("train.parquet")
    manifest_changed = baseline.get("manifest_sha256") != candidate.get("manifest_sha256")
    if train_changed and not manifest_changed:
        issues.append("changed training data requires a new manifest fingerprint")
    _check(issues)
    return {
        "status": "passed",
        "identical_config_fields": equal_fields,
        "selection_sha256": candidate["data_files"]["selection.parquet"],
        "manifest_changed": manifest_changed,
        "changed_training_files": sorted(
            name for name in _DATA_FILES if baseline["data_files"][name] != candidate["data_files"][name]
        ),
        "gpu": {"baseline": baseline["gpu"], "candidate": candidate["gpu"]},
    }


def audit_run(
    baseline: dict,
    config: dict,
    result: dict,
    history: list[dict],
    *,
    checkpoint_path: str | Path | None = None,
    frozen_checkpoint_sha256: str | None = None,
    require_convergence: bool = True,
) -> dict:
    """Validate completion and first-argmax selection before any test access.

    Pass ``checkpoint_path`` to verify the physical checkpoint before freeze.
    Pass the frozen digest again before evaluation. This audits bytes, without
    deserializing model weights; the unchanged evaluator also verifies payloads.
    A late checkpoint raises by default. ``require_convergence=False`` allows an
    orchestrator to finish the other arms while returning ``test_eligible=False``;
    it never permits inconsistent provenance or an incorrectly recorded flag.
    """
    config_audit = audit_config(baseline, config)
    issues = []
    if result.get("status") != "trained" or result.get("smoke_only") is not False:
        issues.append("training did not complete a full production run")
    for key in ("seed", "max_steps", "manifest_sha256"):
        if result.get(key) != config[key]:
            issues.append(f"training result differs from config: {key}")
    if result.get("distinct_truncated_rationales") != 0:
        issues.append("training truncated rationales")
    exposed = result.get("distinct_seen_training_questions")
    if type(exposed) is not int or not 0 < exposed <= config["max_steps"] * config["effective_batch_size"]:
        issues.append("invalid distinct training-question exposure count")
    expected_steps = [*range(config["val_every"], config["max_steps"] + 1, config["val_every"])]
    if not expected_steps or expected_steps[-1] != config["max_steps"]:
        expected_steps.append(config["max_steps"])
    if [event.get("step") for event in history] != expected_steps:
        issues.append("selection history does not cover the exact update budget")
    valid_history = bool(history)
    for event in history:
        accuracy = event.get("accuracy")
        if (
            not isinstance(accuracy, (int, float))
            or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1
            or event.get("questions") != config["selection_questions"]
            or event.get("scores", {}).get("groups", {}).get("all", {}).get("accuracy") != accuracy
        ):
            valid_history = False
            issues.append("invalid selection history measurement")
    converged = False
    if valid_history:
        best = max(history, key=lambda event: event["accuracy"])
        if (
            result.get("best_step") != best["step"]
            or result.get("best_selection_accuracy") != best["accuracy"]
        ):
            issues.append("selected checkpoint is not the first selection-accuracy maximum")
        converged = best["step"] < 0.8 * config["max_steps"]
        if require_convergence and not converged:
            issues.append("best checkpoint is at/after 80% of updates; keep test closed for budget review")
    if result.get("best_before_80_percent") is not converged:
        issues.append("recorded convergence guard disagrees with the selected checkpoint")
    checkpoint_digest = result.get("best_checkpoint_sha256")
    if not _digest(checkpoint_digest):
        issues.append("invalid best-checkpoint SHA-256")
    if checkpoint_path is not None and file_sha256(checkpoint_path) != checkpoint_digest:
        issues.append("best checkpoint bytes differ from completed training")
    if frozen_checkpoint_sha256 is not None and frozen_checkpoint_sha256 != checkpoint_digest:
        issues.append("best checkpoint differs from frozen checkpoint")
    _check(issues)
    return {
        **config_audit,
        "status": "passed" if converged else "needs_budget_review",
        "test_eligible": converged,
        "seed": config["seed"],
        "manifest_sha256": config["manifest_sha256"],
        "best_step": result["best_step"],
        "best_selection_accuracy": result["best_selection_accuracy"],
        "best_checkpoint_sha256": checkpoint_digest,
        "checkpoint_bytes_verified": checkpoint_path is not None,
        "frozen_checkpoint_verified": frozen_checkpoint_sha256 is not None,
        "distinct_seen_training_questions": exposed,
        "distinct_truncated_rationales": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing config.json")
    parser.add_argument("--config-only", action="store_true", help="Audit before training completes")
    parser.add_argument("--frozen-checkpoint-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def read(path: Path):
        return json.loads(path.read_text())

    baseline, config = read(args.baseline_config), read(args.run_dir / "config.json")
    if args.config_only:
        report = audit_config(baseline, config)
    else:
        report = audit_run(
            baseline,
            config,
            read(args.run_dir / "train_result.json"),
            read(args.run_dir / "selection_history.json"),
            checkpoint_path=args.run_dir / "best.pt",
            frozen_checkpoint_sha256=args.frozen_checkpoint_sha256,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
