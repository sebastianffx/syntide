"""Verify the fixed training environment and all training targets before launch.

The unchanged baseline model is initialized once. Only training and selection
data are opened. Validation and test labels stay closed. Shared tasks are
encoded once only after their text, patient identity, and waveform bytes match.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

TASK_FIELDS = (
    "task_id",
    "record_id",
    "subject_uid",
    "question",
    "clinical_context",
    "answer_options",
    "rationale",
    "answer",
    "template_id",
    "is_artifact",
)
ARMS = ("real_qa", "paraphrase", "grounded", "grounded_acq")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_sources(baseline):
    for module, field in (
        ("syntide_trainers.opentslm_loop", "trainer_source_sha256"),
        ("syntide.baseline.metrics", "metrics_source_sha256"),
    ):
        spec = importlib.util.find_spec(module)
        if not spec or not spec.origin or sha(spec.origin) != baseline[field]:
            raise ValueError(f"Runtime source differs from baseline: {field}")


def check_runtime(baseline, actual):
    differing = [
        key
        for key, value in actual.items()
        if key != "gpu" and baseline.get(key) != value
    ]
    if differing:
        raise ValueError(
            f"Runtime differs from baseline: {', '.join(sorted(differing))}"
        )


def clear_caches(collator):
    for name in ("prompt_cache", "signal_cache", "training_token_cache"):
        getattr(collator, name).clear()


def scan_rows(collator, rows, seen, *, on_progress=None, progress_every=10000):
    """Exercise original encode/truncation logic with at most 128 cached tasks.

    ``seen`` persists across arms. Waveform digests are recomputed once per
    record in each prepared folder so a shared task never masks changed signals.
    """
    signal_digests, local_ids = {}, set()
    encoded, reused, maximum = 0, 0, 0
    truncated = []
    started = time.monotonic()
    clear_caches(collator)
    collator.truncated_tasks.clear()
    for position, row in enumerate(rows, 1):
        task, record = row["task_id"], row["record_id"]
        if task in local_ids:
            raise ValueError(f"Duplicate task ID within training arm: {task}")
        local_ids.add(task)
        if record not in signal_digests:
            signal = collator.prepared.signal(row)
            signal_digests[record] = hashlib.sha256(
                signal.tobytes()
            ).hexdigest()
        identity = hashlib.sha256(
            json.dumps(
                {
                    "row": {key: row[key] for key in TASK_FIELDS},
                    "signal_sha256": signal_digests[record],
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if task in seen:
            previous_identity, length, was_truncated = seen[task]
            if previous_identity != identity:
                raise ValueError(
                    f"Shared task ID has changed text, patient or "
                    f"waveform: {task}"
                )
            reused += 1
        else:
            ids, _, _ = collator.encode(row, training=True)
            length = len(ids)
            was_truncated = task in collator.truncated_tasks
            seen[task] = (identity, length, was_truncated)
            encoded += 1
            if encoded % 128 == 0:
                clear_caches(collator)
        maximum = max(maximum, length)
        if was_truncated:
            truncated.append(task)
        if on_progress and (
            position % progress_every == 0 or position == len(rows)
        ):
            on_progress(
                {
                    "scanned": position,
                    "total": len(rows),
                    "newly_encoded": encoded,
                    "reused_verified_tasks": reused,
                    "truncated_rationales": len(truncated),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
    clear_caches(collator)
    return {
        "questions": len(rows),
        "newly_encoded": encoded,
        "reused_verified_tasks": reused,
        "records_verified": len(signal_digests),
        "max_encoded_tokens": maximum,
        "truncated_rationales": len(truncated),
        "truncated_task_ids": truncated,
        "elapsed_seconds": time.monotonic() - started,
    }


def preflight(protocol, baseline, output):
    from syntide.qa_baseline_audit import audit_config

    check_sources(baseline)
    from syntide_trainers import opentslm_loop as trainer

    if set(protocol["arms"]) != set(ARMS) or protocol["seeds"] != [0]:
        raise ValueError(
            "Preflight requires the four comparison arms and seed 0"
        )
    audit_config(baseline, baseline)
    trainer.require_cuda()
    started = time.monotonic()
    report = {
        "status": "loading_model",
        "started_epoch_s": time.time(),
        "arms": {},
    }
    write(output, report)
    model, revision = trainer.load_model(0, baseline["base_revision"])
    actual = {
        "model": trainer.BASE_MODEL,
        "base_revision": revision,
        "opentslm_commit": trainer.UPSTREAM_COMMIT,
        "prompt_version": trainer.PROMPT_VERSION,
        "packages": {
            package: trainer.package_version(package)
            for package in baseline["packages"]
        },
        "python": sys.version,
        "gpu": trainer.torch.cuda.get_device_name(),
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "init_sha256": trainer.parameter_fingerprint(model),
    }
    check_runtime(baseline, actual)
    report.update(status="scanning", runtime=actual)
    write(output, report)
    seen, baseline_files = {}, None
    for arm in ARMS:
        prepared = trainer.Prepared(
            Path(protocol["arms"][arm]["prepared"]), ("train", "selection")
        )
        if baseline_files is None:
            if (
                prepared.manifest_sha256 != baseline["manifest_sha256"]
                or prepared.file_sha256 != baseline["data_files"]
            ):
                raise ValueError(
                    "real_qa prepared data differs from the original baseline"
                )
            baseline_files = prepared.manifest["files"]
        for split in ("selection", "validation", "test"):
            filename = f"{split}.parquet"
            if prepared.manifest["files"][filename] != baseline_files[filename]:
                raise ValueError(
                    f"{arm}: {split} file identity differs from baseline"
                )
        collator = trainer.Collator(
            model.text_tokenizer, prepared, baseline["max_length"]
        )
        candidate = {
            **baseline,
            **actual,
            "manifest_sha256": prepared.manifest_sha256,
            "data_files": prepared.file_sha256,
            "token_lengths": {
                split: trainer.token_length_summary(
                    collator, prepared.rows[split]
                )
                for split in ("train", "selection")
            },
        }
        comparison = audit_config(baseline, candidate)
        report["arms"][arm] = {
            "config_comparison": comparison,
            "expected_config": candidate,
        }

        def progress(value, arm=arm):
            report["arms"][arm]["progress"] = value
            write(output, report)
            print(
                json.dumps({"event": "token_preflight", "arm": arm, **value}),
                flush=True,
            )

        scan = scan_rows(
            collator, prepared.rows["train"], seen, on_progress=progress
        )
        report["arms"][arm]["full_training_token_scan"] = scan
        write(output, report)
        if scan["truncated_rationales"]:
            raise ValueError(
                f"{arm}: {scan['truncated_rationales']} training "
                f"rationales would truncate"
            )
        del collator, prepared
    check_sources(baseline)
    report.update(
        status="passed",
        unique_tasks_encoded=len(seen),
        elapsed_seconds=time.monotonic() - started,
        completed_epoch_s=time.time(),
        trainer_source_sha256=baseline["trainer_source_sha256"],
        metrics_source_sha256=baseline["metrics_source_sha256"],
    )
    write(output, report)
    print(
        json.dumps(
            {
                "event": "preflight_passed",
                "output": str(output),
                "unique_tasks_encoded": len(seen),
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        preflight(read(args.protocol), read(args.baseline_config), args.output)
    except BaseException as error:
        report = read(args.output) if args.output.exists() else {}
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write(args.output, report)
        raise


if __name__ == "__main__":
    main()
