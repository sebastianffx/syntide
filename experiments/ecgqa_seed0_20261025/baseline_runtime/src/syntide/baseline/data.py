"""Export a checked ECG-QA manifest into simple files consumed by the CUDA trainer.

Run on RunPod, after building the TimeNet registry::

    python -m syntide.baseline.data --registry datasets/timenet-registry \
        --manifest manifests/ecgqa-artifact-50/pool50.json --output runs/ecgqa-data

``train.parquet`` retains every real training question/rationale. Validation and test
deduplicate (record_id, question), keeping the lowest task_id. ``selection.parquet``
is the only validation table the training loop needs. Test answers are never an
input to selection; the final evaluation opens ``test.parquet`` after freezing the
checkpoint. All tables have ``signal_index`` into ``signals.npy``/``records.json``.

The output is immutable: existing exports are verified and reused only when their
source and preparation settings match. Use a new directory for a changed protocol.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from syntide.data.schema import FS_HZ, LEADS

SCHEMA_VERSION = 1
SELECTION_ALGORITHM = "balanced-template-noise-patient-round-robin-v1"
_FILES = (
    "train.parquet",
    "validation.parquet",
    "test.parquet",
    "selection.parquet",
    "records.json",
    "signals.npy",
)


def canonical_sha256(value: Any) -> str:
    """Portable JSON fingerprint; excludes machine paths and timestamps."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_export_manifest(output: str | Path, *, verify_files: bool = True) -> dict:
    """Verify an export without reading any question/answer table into the model.

    Hashing the held-out file checks storage integrity only. Its labels are not
    parsed or used for model selection.
    """
    output = Path(output)
    payload = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    recorded = payload.get("sha256")
    content = {key: value for key, value in payload.items() if key != "sha256"}
    if not recorded or canonical_sha256(content) != recorded:
        raise ValueError(f"{output}: export manifest checksum mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{output}: unsupported export schema {payload.get('schema_version')}")
    if set(payload.get("files", {})) != set(_FILES):
        raise ValueError(f"{output}: unexpected export file inventory")
    if verify_files:
        for name, expected in payload["files"].items():
            path = output / name
            if not path.is_file() or file_sha256(path) != expected["sha256"]:
                raise ValueError(f"{path}: missing file or checksum mismatch")
            if path.stat().st_size != expected["bytes"]:
                raise ValueError(f"{path}: file length mismatch")
    return payload


def verify_prepared(output: str | Path, *, splits: tuple[str, ...] = ("train", "selection")) -> dict:
    """Verify only the files needed in this phase, without opening held-out tables.

    The stable dataset fingerprint is the returned dictionary's ``sha256``. Call
    with the default during training, or ``splits=("validation", "test")`` after
    freezing the selected checkpoint for final evaluation.
    """
    if not set(splits) <= {"train", "selection", "validation", "test"}:
        raise ValueError(f"unknown dataset splits: {splits}")
    output = Path(output)
    payload = load_export_manifest(output, verify_files=False)
    for name in ("records.json", "signals.npy", *(f"{split}.parquet" for split in splits)):
        expected, path = payload["files"][name], output / name
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{path}: missing file or file length mismatch")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"{path}: checksum mismatch")
    return payload


def distinct_questions(tasks: pa.Table) -> pa.Table:
    """Keep the lowest task_id for each exact (ECG, question) pair, deterministically."""
    columns = tasks.select(["task_id", "record_id", "question"]).to_pydict()
    order = sorted(range(tasks.num_rows), key=lambda i: columns["task_id"][i])
    seen, selected = set(), []
    for i in order:
        key = (columns["record_id"][i], columns["question"][i])
        if key not in seen:
            seen.add(key)
            selected.append(i)
    # Preserve the adapter's ECG/task order after resolving which rationale wins.
    return tasks.take(pa.array(sorted(selected), type=pa.int64()))


def check_question_truth(tasks: pa.Table) -> None:
    """Rationale variants of one (ECG, question) must have the same scoring truth.

    Question text can legitimately occur on different ECGs with different answers.
    Only duplicates within the same ECG are compared. Option ordering is irrelevant.
    """
    truth: dict[tuple[str, str], tuple] = {}
    names = ["task_id", "record_id", "question", "template_id", "answer", "answer_options"]
    columns = tasks.select(names).to_pydict()
    for task, record, question, template, answer, options in zip(*columns.values(), strict=True):
        key = (record, question)
        value = (template, answer, tuple(sorted(options)))
        previous = truth.setdefault(key, value)
        if previous != value:
            raise ValueError(f"conflicting answer/template/options for repeated (ECG, question): {task}")


def _verify_signal_source(registry: str | Path, source: dict) -> None:
    """Check physical PTB-XL bytes, not only the checksums recorded in its manifest."""
    from timenet.reader import TimeFReader
    from timenet.registry import LocalRegistry

    signals = source["signals"]
    version = LocalRegistry(Path(registry)).open_version(signals["dataset"], signals["version"])
    with TimeFReader(version) as reader:
        reader.verify()


def selection_subset(validation: pa.Table, size: int = 2000, seed: int = 0) -> pa.Table:
    """Fixed answer-blind, patient-first balanced template/noise validation sample.

    Each nonempty (template_id, is_artifact) stratum receives a turn. Within each
    stratum patients are ordered by a seeded hash and supply one question per
    cycle; questions within a patient have an independent seeded order. Depleted
    strata redistribute their quota automatically. This takes exactly ``size``
    questions (or all available questions), without requiring whole-patient size
    rounding. Train/validation/test boundaries themselves remain whole-patient.
    """
    if size <= 0:
        raise ValueError("selection size must be positive")
    if seed < 0:
        raise ValueError("selection seed must be nonnegative")
    if not validation.num_rows:
        raise ValueError("validation split is empty")
    columns = validation.select(["task_id", "subject_uid", "template_id", "is_artifact"]).to_pydict()
    strata: dict[tuple[int, bool], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i in range(validation.num_rows):
        key = (columns["template_id"][i], bool(columns["is_artifact"][i]))
        strata[key][columns["subject_uid"][i]].append(i)

    def priority(kind: str, identity: str) -> str:
        return canonical_sha256([seed, kind, identity])

    streams = []
    for key in sorted(strata):
        patients = strata[key]
        queues = deque(
            deque(sorted(patients[patient], key=lambda i: (priority("task", columns["task_id"][i]), i)))
            for patient in sorted(patients, key=lambda patient: (priority("patient", patient), patient))
        )
        ordered = deque()
        while queues:
            patient = queues.popleft()
            ordered.append(patient.popleft())
            if patient:
                queues.append(patient)
        streams.append(ordered)
    active, selected = deque(streams), []
    while active and len(selected) < min(size, validation.num_rows):
        stream = active.popleft()
        selected.append(stream.popleft())
        if stream:
            active.append(stream)
    return validation.take(pa.array(sorted(selected), type=pa.int64()))


def _task_checksum(tasks: pa.Table) -> str:
    """Hash ordered complete task contents, independent of Parquet compression."""
    digest = hashlib.sha256()
    for batch in tasks.to_batches(max_chunksize=4096):
        for row in batch.to_pylist():
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _counts(tasks: pa.Table) -> dict:
    columns = tasks.select(["task_id", "record_id", "subject_uid", "template_id", "is_artifact"]).to_pydict()
    records = dict(zip(columns["record_id"], columns["is_artifact"], strict=True))
    strata = Counter(
        f"{template}:{'noisy' if noisy else 'clean'}"
        for template, noisy in zip(columns["template_id"], columns["is_artifact"], strict=True)
    )
    return {
        "questions": tasks.num_rows,
        "records": len(records),
        "patients": len(set(columns["subject_uid"])),
        "noisy_records": sum(bool(noisy) for noisy in records.values()),
        "noisy_questions": sum(bool(noisy) for noisy in columns["is_artifact"]),
        "templates": len(set(columns["template_id"])),
        "template_noise_counts": dict(sorted(strata.items())),
        "task_ids_sha256": canonical_sha256(columns["task_id"]),
        "tasks_sha256": _task_checksum(tasks),
    }


def export_dataset(
    registry: str | Path,
    manifest_path: str | Path,
    output: str | Path,
    *,
    selection_size: int = 2000,
    selection_seed: int = 0,
    train_role: str = "pool",
) -> dict:
    """Check the existing Syntide gates, then atomically publish the flat dataset."""
    # TimeNet belongs only to preparation; the CUDA loop consumes plain files.
    from syntide.data.challenge import load_manifest
    from syntide.data.ecgqa.adapter import EcgQaAdapter
    from syntide.data.leakage import check_manifest, check_tasks

    if train_role not in {"pool", "depleted", "natural_mix"}:
        raise ValueError("train_role must be pool, depleted, or natural_mix")
    if selection_size <= 0 or selection_seed < 0:
        raise ValueError("selection size must be positive and selection seed nonnegative")
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = load_manifest(manifest_path)
    adapter = EcgQaAdapter(registry)
    tasks, index, source = adapter.tasks(), adapter.index(), adapter.source()
    check_manifest(manifest, index, source)
    check_tasks(tasks, manifest, index)
    check_question_truth(tasks)
    _verify_signal_source(registry, source)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_manifest": {
            "challenge_id": manifest.challenge_id,
            "sha256": manifest.sha256,
            "file_sha256": file_sha256(manifest_path),
            "spec": manifest.spec,
        },
        "preparation": {
            "train_role": train_role,
            "train_deduplication": "none; preserve every real question/rationale row",
            "evaluation_deduplication": "record_id,question; lowest lexicographic task_id",
            "selection_size_requested": selection_size,
            "selection_seed": selection_seed,
            "selection_algorithm": SELECTION_ALGORITHM,
            "selection_answer_blind": True,
        },
    }
    if output.exists():
        existing = load_export_manifest(output)
        if any(existing.get(key) != value for key, value in identity.items()):
            raise FileExistsError(f"{output}: frozen export has a different source or preparation protocol")
        return existing

    train = adapter.tasks_of(manifest.roles[train_role])
    validation_raw = adapter.tasks_of(manifest.roles["val"])
    test_raw = adapter.tasks_of(manifest.roles["test"])
    tables = {
        "train": train,
        "validation": distinct_questions(validation_raw),
        "test": distinct_questions(test_raw),
    }
    if any(not table.num_rows for table in tables.values()):
        raise ValueError("train, validation and test must each contain questions")
    tables["selection"] = selection_subset(tables["validation"], selection_size, selection_seed)
    wanted = set().union(*(set(table["record_id"].to_pylist()) for table in tables.values()))
    records = [record for record in index["record_id"].to_pylist() if record in wanted]
    positions = {record: i for i, record in enumerate(records)}
    tables = {
        name: table.append_column(
            "signal_index",
            pa.array([positions[record] for record in table["record_id"].to_pylist()], pa.int64()),
        )
        for name, table in tables.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, table in tables.items():
            pq.write_table(table, staging / f"{name}.parquet", compression="zstd")
        (staging / "records.json").write_text(
            json.dumps(records, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        signals = np.lib.format.open_memmap(
            staging / "signals.npy", mode="w+", dtype="<f4", shape=(len(records), len(LEADS), 1000)
        )
        for start in range(0, len(records), 256):
            batch_ids = records[start : start + 256]
            batch = adapter.load(batch_ids)
            if batch.x.shape != (len(batch_ids), len(LEADS), 1000):
                raise ValueError(f"unexpected 100 Hz ECG shape: {batch.x.shape}")
            if batch.record_id.tolist() != batch_ids or not np.isfinite(batch.x).all():
                raise ValueError("ECG order mismatch or nonfinite signal values")
            signals[start : start + len(batch_ids)] = batch.x
        signals.flush()
        del signals
        content = {
            **identity,
            "signal": {
                "shape": [len(records), len(LEADS), 1000],
                "dtype": "float32",
                "fs_hz": FS_HZ,
                "units": "mV",
                "leads": list(LEADS),
                "normalization": "none; trainer applies per-lead normalization",
            },
            "splits": {name: _counts(table) for name, table in tables.items()},
            "deduplication": {
                "validation_rows_removed": validation_raw.num_rows - tables["validation"].num_rows,
                "test_rows_removed": test_raw.num_rows - tables["test"].num_rows,
            },
            "source_questions_dropped": adapter.dropped(),
            "leakage_checks": {
                "source_fingerprint": "passed",
                "physical_source_signal_checksums": "passed",
                "records": "passed",
                "patients": "passed",
                "questions": "passed",
                "duplicate_question_truth": "passed",
            },
            "files": {
                name: {"sha256": file_sha256(staging / name), "bytes": (staging / name).stat().st_size}
                for name in _FILES
            },
        }
        payload = {**content, "sha256": canonical_sha256(content)}
        (staging / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # A concurrent exporter is an error, never a reason to replace published data.
        if output.exists():
            raise FileExistsError(f"{output}: output appeared during export")
        staging.rename(output)
        return payload
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-size", default=2000, type=int)
    parser.add_argument("--selection-seed", default=0, type=int)
    parser.add_argument("--train-role", choices=("pool", "depleted", "natural_mix"), default="pool")
    args = parser.parse_args()
    result = export_dataset(
        args.registry,
        args.manifest,
        args.output,
        selection_size=args.selection_size,
        selection_seed=args.selection_seed,
        train_role=args.train_role,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": result["sha256"],
                "splits": {
                    name: {
                        key: value
                        for key, value in split.items()
                        if key in {"questions", "records", "patients", "noisy_records"}
                    }
                    for name, split in result["splits"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
