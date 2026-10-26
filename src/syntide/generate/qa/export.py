"""Add verified corpus tasks to the baseline training dataset.

The export preserves baseline validation and test data, question formatting,
and training schema. Added tasks must trace back to allowed training recordings
and patients. The complete export is checked before its output directory is
published.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from syntide.data.qa import InputRef, QATask
from syntide.data.schema import FS_HZ, LEADS
from syntide.generate.qa.corpus import dataset_id, verify_corpus_files
from syntide.generate.qa.evidence import waveform_sha256
from syntide.generate.qa.sources import baseline_rationale

EXPORT_VERSION = "syntide.qa-baseline-export@1"
LOG = logging.getLogger(__name__)
FILES = (
    "train.parquet",
    "selection.parquet",
    "validation.parquet",
    "test.parquet",
    "records.json",
    "signals.npy",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_prepared(path: str | Path) -> dict:
    """Check baseline files without opening held-out labels."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != canonical_sha256(body):
        raise ValueError(f"{path}: manifest checksum mismatch")
    if manifest.get("schema_version") != 1 or set(
        manifest.get("files", {})
    ) != set(FILES):
        raise ValueError(f"{path}: unsupported baseline prepared format")
    for name, expected in manifest["files"].items():
        file = path / name
        if not file.is_file() or file.stat().st_size != expected["bytes"]:
            raise ValueError(f"{file}: missing file or file length mismatch")
        if file_sha256(file) != expected["sha256"]:
            raise ValueError(f"{file}: checksum mismatch")
    return manifest


def _tasks(path: Path, info: dict) -> Iterator[QATask]:
    from timenet.reader import TimeFReader
    from timenet.registry import LocalRegistry

    version = LocalRegistry(path / "timenet").open_version(
        dataset_id(info["identity"]), "1.0.0"
    )
    with TimeFReader(version) as reader:
        for task in reader.iter_tasks():
            yield QATask.from_json(task.metadata["qa_task"])


def _check_source(info: dict, baseline: dict) -> None:
    identity, source = info["identity"], info["identity"]["source"]
    if identity["purpose"] != "training" or source.get("purpose") != "training":
        raise PermissionError(
            "evaluation corpora cannot be exported for training"
        )
    if source.get("role") != baseline["preparation"]["train_role"]:
        raise PermissionError("corpus training role differs from the baseline")
    if source.get("manifest_sha256") != baseline["source_manifest"]["sha256"]:
        raise PermissionError("corpus split manifest differs from the baseline")
    if source.get("source") != baseline["source"]:
        raise PermissionError(
            "corpus source fingerprint differs from the baseline"
        )
    if identity.get("context_policy") != "ecgqa-cot-baseline@1":
        raise PermissionError("corpus uses a different student-context policy")


def _check_ref(ref: InputRef, roots: dict[str, dict]) -> None:
    if not ref.root_record_ids or any(
        record not in roots or roots[record]["subject_uid"] != patient
        for record, patient in zip(
            ref.root_record_ids, ref.root_subject_uids, strict=True
        )
    ):
        raise PermissionError(
            f"{ref.waveform_id}: lineage is outside baseline training "
            f"records/patients"
        )
    if (
        ref.origin == "generated"
        and ref.lineage.get("review_status") != "verified"
    ):
        raise PermissionError(
            f"{ref.waveform_id}: generated signal has not passed "
            f"measurement review"
        )


def _incompatible(task: QATask) -> str | None:
    if len(task.inputs) != 1:
        return "multiple_ecg_inputs"
    ref, spec = task.inputs[0], task.inputs[0].signal
    if (
        spec.channels != tuple(LEADS)
        or spec.sample_rate != FS_HZ
        or spec.units != "mV"
    ):
        return "incompatible_signal_spec"
    if spec.num_samples != 1000:
        return "incompatible_signal_length"
    if len(set(ref.root_subject_uids)) != 1:
        return "multiple_root_patients"
    return None


def _clinical_context(task: QATask) -> str:
    if set(task.context) - {"clinical_context"}:
        raise ValueError(
            f"{task.task_id}: unsupported student context; refusing "
            f"to discard it"
        )
    contexts = task.context.get("clinical_context", [])
    if (
        not isinstance(contexts, list)
        or len(contexts) != 1
        or not isinstance(contexts[0], str)
    ):
        raise ValueError(
            f"{task.task_id}: expected exactly one clinical context string"
        )
    return contexts[0]


def _counts(table: pa.Table) -> dict:
    rows = table.to_pylist()
    records = {row["record_id"]: row["is_artifact"] for row in rows}
    strata = Counter(
        f"{r['template_id']}:{'noisy' if r['is_artifact'] else 'clean'}"
        for r in rows
    )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        )
        digest.update(b"\n")
    return {
        "questions": len(rows),
        "records": len(records),
        "patients": len({r["subject_uid"] for r in rows}),
        "noisy_records": sum(bool(v) for v in records.values()),
        "noisy_questions": sum(bool(r["is_artifact"]) for r in rows),
        "templates": len({r["template_id"] for r in rows}),
        "template_noise_counts": dict(sorted(strata.items())),
        "task_ids_sha256": canonical_sha256([r["task_id"] for r in rows]),
        "tasks_sha256": digest.hexdigest(),
    }


def export_prepared(
    corpora: Sequence[str | Path],
    baseline_prepared: str | Path,
    output: str | Path,
) -> dict:
    """Append accepted corpus tasks to baseline training data.

    Corpus order determines the export identity. Duplicate task identifiers are
    included once. Supplied original tasks must agree with the baseline and
    remain
    represented by its unchanged training rows.

    Args:
        corpora: Published corpus directories, in the intended append order.
        baseline_prepared: Original prepared baseline dataset directory.
        output: Directory for the complete augmented dataset.

    Returns:
        Metadata for the verified export.
    """
    if not corpora:
        raise ValueError("at least one training corpus is required")
    baseline_path, output = Path(baseline_prepared), Path(output)
    baseline = verify_prepared(baseline_path)
    if baseline.get("qa_export"):
        raise ValueError(
            "baseline-prepared must be the original baseline, not an "
            "augmented export"
        )
    sources = [(Path(path), verify_corpus_files(path)) for path in corpora]
    if len({info["corpus_id"] for _, info in sources}) != len(sources):
        raise ValueError("the same corpus was supplied more than once")
    for _, info in sources:
        _check_source(info, baseline)
    identity = {
        "version": EXPORT_VERSION,
        "baseline_manifest_sha256": baseline["sha256"],
        "corpus_ids": [info["corpus_id"] for _, info in sources],
        ("training_policy"): (
            "baseline original rows followed by unique compatible corpus tasks"
        ),
        ("signal_policy"): (
            "unchanged baseline prefix; append exact 12x1000, 100 Hz, mV "
            "signals only"
        ),
        "context_policy": "ecgqa-cot-baseline@1",
        ("numeric_options_policy"): (
            "preserve empty options; unchanged trainer prompt"
        ),
    }
    if output.exists():
        existing = verify_prepared(output)
        if existing.get("qa_export", {}).get("identity") != identity:
            raise FileExistsError(
                f"{output}: frozen export has a different corpus or baseline"
            )
        return existing
    train = pq.read_table(baseline_path / "train.parquet")
    original = train.to_pylist()
    roots = {row["record_id"]: row for row in original}
    by_task = {row["task_id"]: row for row in original}
    if len(by_task) != len(original):
        raise ValueError("baseline has duplicate training task IDs")
    records = json.loads((baseline_path / "records.json").read_text())
    positions = {record: i for i, record in enumerate(records)}
    baseline_signals = np.load(
        baseline_path / "signals.npy", mmap_mode="r", allow_pickle=False
    )
    if len(positions) != len(records) or baseline_signals.shape != (
        len(records),
        12,
        1000,
    ):
        raise ValueError(
            "baseline records/signals shape or record uniqueness mismatch"
        )
    if baseline_signals.dtype != np.dtype("float32"):
        raise ValueError("baseline signals must be float32")
    for row in original:
        if positions.get(row["record_id"]) != row["signal_index"]:
            raise ValueError("baseline training signal index mismatch")
    selection = pq.read_table(
        baseline_path / "selection.parquet", columns=["subject_uid"]
    )
    if {r["subject_uid"] for r in original} & set(
        selection["subject_uid"].to_pylist()
    ):
        raise PermissionError(
            "baseline has patient leakage into checkpoint selection"
        )
    templates = sorted(
        {t for _, info in sources for t in info["identity"]["templates"]}
    )
    template_ids = {name: 1000 + i for i, name in enumerate(templates)}
    added, seen, appended, checked = [], set(by_task), {}, {}
    reports = []
    for path, info in sources:
        LOG.info("Exporting %s (%s)", path, info["corpus_id"])
        counts, reasons = Counter(), Counter()
        for task in _tasks(path, info):
            counts["input_tasks"] += 1
            if counts["input_tasks"] % 10000 == 0:
                LOG.info(
                    "%s: checked %d tasks", path.name, counts["input_tasks"]
                )
            if task.verification.get("status") != "accepted":
                raise ValueError(
                    f"{task.task_id}: unaccepted task in a published corpus"
                )
            for ref in task.inputs:
                _check_ref(ref, roots)
            if reason := _incompatible(task):
                reasons[reason] += 1
                continue
            ref = task.inputs[0]
            context = _clinical_context(task)
            if task.task_id in seen:
                counts["duplicate_tasks"] += 1
                continue
            source_id = task.lineage.get("source_task_id")
            parent = (
                by_task.get(source_id)
                if source_id
                else roots[ref.root_record_ids[0]]
            )
            if parent is None:
                raise PermissionError(
                    f"{task.task_id}: source task is outside the "
                    f"baseline training rows"
                )
            if task.composition == "source" and task.paraphrase == "seed":
                if (
                    parent["record_id"] != ref.waveform_id
                    or parent["question"] != task.question
                    or parent["answer"] != task.answer_text
                    or baseline_rationale(parent["rationale"]) != task.rationale
                    or parent["clinical_context"] != context
                    or tuple(parent["answer_options"]) != task.options
                ):
                    raise ValueError(
                        f"{task.task_id}: original source task "
                        f"differs from baseline"
                    )
                counts["original_tasks_already_in_baseline"] += 1
                seen.add(task.task_id)
                continue
            previous = checked.get(ref.waveform_id)
            if previous is not None and previous != ref:
                raise ValueError(
                    f"{ref.waveform_id}: conflicting waveform "
                    f"references across corpora"
                )
            if previous is None:
                signal_file = path / "signals" / f"{ref.waveform_sha256}.npy"
                x = np.load(signal_file, allow_pickle=False)
                if x.shape != (12, 1000) or not np.isfinite(x).all():
                    raise ValueError(
                        f"{ref.waveform_id}: invalid waveform values or shape"
                    )
                if waveform_sha256(x) != ref.waveform_sha256:
                    raise ValueError(
                        f"{ref.waveform_id}: waveform does not match "
                        f"its measured bytes"
                    )
                if ref.waveform_id in positions:
                    if not np.array_equal(
                        x, baseline_signals[positions[ref.waveform_id]]
                    ):
                        raise ValueError(
                            f"{ref.waveform_id}: corpus waveform "
                            f"differs from baseline"
                        )
                else:
                    positions[ref.waveform_id] = len(records)
                    records.append(ref.waveform_id)
                    appended[ref.waveform_id] = (ref, signal_file)
                checked[ref.waveform_id] = ref
            template_id = task.lineage.get("template_id")
            if template_id is None:
                template_id = template_ids[task.template]
            added.append(
                {
                    **parent,
                    "task_id": task.task_id,
                    "record_id": ref.waveform_id,
                    "subject_uid": ref.root_subject_uids[0],
                    "question": task.question,
                    "answer": task.answer_text,
                    "answer_options": list(task.options),
                    "clinical_context": context,
                    "rationale": task.rationale,
                    "has_rationale": bool(task.rationale),
                    "template_id": template_id,
                    "question_type": task.lineage.get(
                        "question_type", task.family
                    ),
                    "cot_split": task.lineage.get("cot_split", "synthetic"),
                    "signal_index": positions[ref.waveform_id],
                }
            )
            seen.add(task.task_id)
            counts["exported_tasks"] += 1
            counts["numeric_tasks"] += task.answer_type == "numeric"
            counts["empty_options_tasks"] += not task.options
        reports.append(
            {
                "corpus_id": info["corpus_id"],
                "counts": dict(counts),
                "excluded": dict(reasons),
            }
        )
    if not added:
        raise ValueError(
            "no compatible new training tasks remain after exclusions"
        )
    combined = pa.concat_tables(
        [train, pa.Table.from_pylist(added, schema=train.schema)]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    try:
        pq.write_table(combined, staging / "train.parquet", compression="zstd")
        for split in ("selection", "validation", "test"):
            shutil.copyfile(
                baseline_path / f"{split}.parquet", staging / f"{split}.parquet"
            )
        if not appended:
            shutil.copyfile(
                baseline_path / "signals.npy", staging / "signals.npy"
            )
            shutil.copyfile(
                baseline_path / "records.json", staging / "records.json"
            )
        else:
            (staging / "records.json").write_text(
                json.dumps(records, separators=(",", ":")) + "\n"
            )
            signals = np.lib.format.open_memmap(
                staging / "signals.npy",
                mode="w+",
                dtype="<f4",
                shape=(len(records), 12, 1000),
            )
            for start in range(0, len(baseline_signals), 256):
                end = min(start + 256, len(baseline_signals))
                signals[start:end] = baseline_signals[start:end]
            for wid, (_, file) in appended.items():
                signals[positions[wid]] = np.load(file, allow_pickle=False)
            signals.flush()
            del signals
        content = {
            **{
                key: value
                for key, value in baseline.items()
                if key not in ("sha256", "files")
            },
            "signal": {**baseline["signal"], "shape": [len(records), 12, 1000]},
            "splits": {**baseline["splits"], "train": _counts(combined)},
            "qa_export": {
                "identity": identity,
                "baseline_rows_preserved": len(original),
                "added_training_rows": len(added),
                "baseline_signal_prefix": len(baseline_signals),
                "appended_waveforms": [
                    {"signal_index": positions[wid], "input": ref.to_json()}
                    for wid, (ref, _) in appended.items()
                ],
                "synthetic_template_ids": template_ids,
                "corpora": reports,
                ("lineage_check"): (
                    "all roots and patients exactly match baseline "
                    "training rows"
                ),
                ("evaluation_policy"): (
                    "selection/validation/test copied byte-for-byte; "
                    "labels not parsed"
                ),
                ("training_slice_metadata"): (
                    "strat_fold/is_artifact/artifact_types inherited "
                    "from first root"
                ),
            },
            "files": {
                name: {
                    "sha256": file_sha256(staging / name),
                    "bytes": (staging / name).stat().st_size,
                }
                for name in FILES
            },
        }
        payload = {**content, "sha256": canonical_sha256(content)}
        (staging / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        verify_prepared(staging)
        if output.exists():
            raise FileExistsError(f"{output}: output appeared during export")
        staging.rename(output)
        return payload
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
