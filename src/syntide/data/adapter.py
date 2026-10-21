"""Read the PTB-XL TimeNet dataset into SYNTIDE's in-memory form.

``index()`` is columnar (control-plane queries only, no signal reads) and feeds the challenge and the
leakage gate. ``load()`` materializes signals for a list of record ids, in clinical lead order.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
from timenet.reader import TimeFReader
from timenet.refs import split_ref
from timenet.registry import LocalRegistry

from syntide.data.ptbxl.connector import TARGET_SCHEMA
from syntide.data.schema import ARTIFACT_TYPES, LEADS, SUPERCLASSES, Samples, subject_uid


class PtbXlAdapter:
    """A read-only view of ``physionet/ptb-xl`` in a local TimeNet registry."""

    def __init__(self, registry: str | Path, dataset: str = "physionet/ptb-xl@1.0.3") -> None:
        dataset_id, version = split_ref(dataset)
        self._version = LocalRegistry(Path(registry)).open_version(dataset_id, version)
        self._reader = TimeFReader(self._version)
        self.dataset_id = dataset_id
        self.dataset_version = str(self._version.manifest.metadata.dataset_version)
        self._index: pa.Table | None = None

    def source(self) -> dict[str, str]:
        """Identify the exact dataset content, for manifests and trial identity."""
        return {"dataset": self.dataset_id, "version": self.dataset_version, "sha256": self.source_sha256()}

    def source_sha256(self) -> str:
        """Fingerprint what SYNTIDE reads: the signal files plus the index.

        TimeNet's control database holds the records, labels and annotations, but its bytes differ on
        every build, so the index derived from it is hashed instead of the file.
        """
        digest = hashlib.sha256()
        for part in sorted(self._version.manifest.files.time_series, key=lambda part: part.path):
            digest.update(f"{part.path} {part.checksum}\n".encode())
        digest.update(json.dumps(self.index().to_pylist(), sort_keys=True).encode())
        return digest.hexdigest()

    def index(self) -> pa.Table:
        """One row per record that has a diagnostic task, ordered by ECG id.

        Columns: ``record_id``, ``subject_uid``, ``strat_fold``, ``labels`` (superclass names),
        ``artifact_types`` (set PTB-XL signal-quality flags) and ``is_artifact``.
        """
        if self._index is None:
            facts = self._record_facts()
            labels = self._labels()
            record_ids = sorted(labels, key=lambda record_id: int(record_id.rpartition("-")[2]))
            artifact_types = [[t for t in ARTIFACT_TYPES if t in facts[r]] for r in record_ids]
            self._index = pa.table(
                {
                    "record_id": record_ids,
                    "subject_uid": [subject_uid(facts[r]["patient_id"]) for r in record_ids],
                    "strat_fold": pa.array([facts[r]["strat_fold"] for r in record_ids], pa.int64()),
                    "labels": [list(labels[r]) for r in record_ids],
                    "artifact_types": pa.array(artifact_types, pa.list_(pa.string())),
                    "is_artifact": [bool(types) for types in artifact_types],
                }
            )
        return self._index

    def load(self, record_ids: Sequence[str]) -> Samples:
        """Materialize signals and labels for ``record_ids``, in that order."""
        rows = self._rows_by_id()
        missing = [record_id for record_id in record_ids if record_id not in rows]
        if missing:
            raise KeyError(f"{len(missing)} record ids are not labelled PTB-XL records, e.g. {missing[:3]}")
        signals = []
        for expected, record in zip(
            record_ids, self._reader.iter_records(record_ids, with_annotations=False), strict=True
        ):
            if record.record_id != expected:
                raise RuntimeError(f"TimeNet returned {record.record_id!r} where {expected!r} was requested")
            by_name = {signal.name: signal for signal in record.signals}
            signals.append(
                np.stack([by_name[lead].to_numpy() for lead in LEADS]).astype(np.float32, copy=False)
            )
        return Samples(
            record_id=np.array(record_ids, dtype=str),
            subject_uid=np.array([rows[r]["subject_uid"] for r in record_ids], dtype=str),
            labels=np.array([[c in rows[r]["labels"] for c in SUPERCLASSES] for r in record_ids], dtype=bool),
            artifacts=np.array(
                [[t in rows[r]["artifact_types"] for t in ARTIFACT_TYPES] for r in record_ids], dtype=bool
            ),
            x=np.stack(signals) if signals else np.empty((0, len(LEADS), 0), np.float32),
        )

    def _rows_by_id(self) -> dict[str, dict]:
        return {row["record_id"]: row for row in self.index().to_pylist()}

    def _record_facts(self) -> dict[str, dict[str, object]]:
        """Pivot the record annotations: record id -> {key: value}."""
        table = self._reader.annotation_table("Record").select(
            ["object_id", "key", "integer_value", "text_value"]
        )
        facts: dict[str, dict[str, object]] = defaultdict(dict)
        for row in table.to_pylist():
            value = row["integer_value"] if row["integer_value"] is not None else row["text_value"]
            facts[row["object_id"]][row["key"]] = value
        return facts

    def _labels(self) -> dict[str, tuple[str, ...]]:
        """Map each record with a diagnostic task to its superclasses."""
        tasks = self._reader.task_table().select(["task_id", "target_schema", "input_record_ids"]).to_pylist()
        record_of = {
            t["task_id"]: t["input_record_ids"][0] for t in tasks if t["target_schema"] == TARGET_SCHEMA
        }
        labels: dict[str, list[str]] = defaultdict(list)
        for target in self._reader.target_table().select(["task_id", "text_value"]).to_pylist():
            if target["task_id"] in record_of:
                labels[record_of[target["task_id"]]].append(target["text_value"])
        return {record_id: tuple(sorted(values)) for record_id, values in labels.items()}
