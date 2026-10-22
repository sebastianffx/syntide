"""Private ECG-QA registry view for discovery, independent of experiment trainers.

Questions come from TimeNet's ECG-QA-CoT registry; waveforms and patient/fold membership
come from the existing PTB-XL adapter. The task columns, ordering and fingerprint preserve
compatibility with ECG-QA experiment manifests and previously saved explorer analyses.
No download/build entry point or experiment training code is needed here.
"""

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
from timenet.reader import TimeFReader
from timenet.refs import split_ref
from timenet.registry import LocalRegistry

from syntide.data.adapter import PtbXlAdapter
from syntide.data.challenge import Manifest
from syntide.data.leakage import LeakageError
from syntide.data.schema import Samples

DATASET = "physionet/ecg-qa-cot@1.0.0"
SIGNALS = "physionet/ptb-xl@1.0.3"
# This exact column order is part of the existing ECG-QA source fingerprint.
TASK_COLUMNS = (
    "task_id",
    "record_id",
    "subject_uid",
    "strat_fold",
    "is_artifact",
    "artifact_types",
    "cot_split",
    "question_type",
    "template_id",
    "question",
    "answer",
    "answer_options",
    "clinical_context",
    "rationale",
    "has_rationale",
)


class EcgQaView:
    """Read a committed ECG-QA registry without importing an optional training package."""

    def __init__(self, registry: str | Path, dataset: str = DATASET, signals: str = SIGNALS):
        dataset_id, version = split_ref(dataset)
        self.signals = PtbXlAdapter(registry, signals)
        self._version = LocalRegistry(Path(registry)).open_version(dataset_id, version)
        self._reader = TimeFReader(self._version)
        self.dataset_id = dataset_id
        self.dataset_version = str(self._version.manifest.metadata.dataset_version)
        self._tasks: pa.Table | None = None
        self._sha256: str | None = None

    def source(self) -> dict:
        return {
            "dataset": self.dataset_id,
            "version": self.dataset_version,
            "sha256": self.source_sha256(),
            "signals": self.signals.source(),
        }

    def source_sha256(self) -> str:
        if self._sha256 is None:
            digest = hashlib.sha256(self.signals.source_sha256().encode())
            columns = self.tasks().to_pydict()
            for row in zip(*(columns[name] for name in TASK_COLUMNS), strict=True):
                digest.update(repr(row).encode())
            self._sha256 = digest.hexdigest()
        return self._sha256

    def tasks(self) -> pa.Table:
        if self._tasks is None:
            self._tasks = self._read_tasks()
        return self._tasks

    def index(self) -> pa.Table:
        counts = Counter(self.tasks()["record_id"].to_pylist())
        index = self.signals.index()
        index = index.filter(pa.array([record_id in counts for record_id in index["record_id"].to_pylist()]))
        return index.append_column(
            "n_tasks", pa.array([counts[r] for r in index["record_id"].to_pylist()], pa.int64())
        )

    def load(self, record_ids: Sequence[str]) -> Samples:
        return self.signals.load(record_ids)

    def _read_tasks(self) -> pa.Table:
        ptbxl = {row["record_id"]: row for row in self.signals.index().to_pylist()}
        splits = self._cot_splits()
        rows = []
        for task in self._reader.iter_tasks():
            if len(task.inputs) != 1:
                raise ValueError(f"{task.id}: ECG-QA discovery expects exactly one input recording")
            record_id = task.inputs[0].record_id
            if record_id not in ptbxl:
                continue
            targets = task.targets or ()
            if len(targets) != 1:
                raise ValueError(f"{task.id}: ECG-QA discovery expects exactly one answer")
            facts = {annotation.key: annotation.value for annotation in task.input_annotations}
            record = ptbxl[record_id]
            rows.append(
                {
                    "task_id": task.id,
                    "record_id": record_id,
                    "subject_uid": record["subject_uid"],
                    "strat_fold": record["strat_fold"],
                    "is_artifact": record["is_artifact"],
                    "artifact_types": record["artifact_types"],
                    "cot_split": splits[record_id],
                    "question_type": str(facts["question_type"]),
                    "template_id": int(facts["template_id"]),
                    "question": task.prompt or "",
                    "answer": str(targets[0]),
                    "answer_options": [str(value) for value in facts.get("answer_options") or []],
                    "clinical_context": str(facts.get("clinical_context", "")),
                    "rationale": task.rationale or "",
                    "has_rationale": bool(task.rationale and task.rationale.strip()),
                }
            )
        rows.sort(key=lambda row: (int(row["record_id"].rpartition("-")[2]), row["task_id"]))
        types = {name: pa.string() for name in TASK_COLUMNS}
        types.update(
            strat_fold=pa.int64(),
            template_id=pa.int64(),
            is_artifact=pa.bool_(),
            has_rationale=pa.bool_(),
            artifact_types=pa.list_(pa.string()),
            answer_options=pa.list_(pa.string()),
        )
        return pa.table({name: pa.array([row[name] for row in rows], types[name]) for name in TASK_COLUMNS})

    def _cot_splits(self) -> dict[str, str]:
        annotations = self._reader.annotation_table("Record").select(["object_id", "key", "text_value"])
        return {
            row["object_id"]: row["text_value"] for row in annotations.to_pylist() if row["key"] == "split"
        }


def check_qa_tasks(tasks: pa.Table, manifest: Manifest, index: pa.Table) -> None:
    """Verify unique tasks, their record/patient join and patient isolation by role."""
    rows = tasks.select(["task_id", "record_id", "subject_uid"]).to_pylist()
    subjects = dict(zip(index["record_id"].to_pylist(), index["subject_uid"].to_pylist(), strict=True))
    counts = Counter(row["task_id"] for row in rows)
    if any(count > 1 for count in counts.values()):
        raise LeakageError("ECG-QA task ids are repeated")
    unknown = {row["record_id"] for row in rows} - subjects.keys()
    if unknown:
        raise LeakageError(f"ECG-QA questions sit on records outside the index: {sorted(unknown)[:5]}")
    mismatched = [row["task_id"] for row in rows if subjects[row["record_id"]] != row["subject_uid"]]
    if mismatched:
        raise LeakageError(f"ECG-QA questions name another patient than their record: {mismatched[:5]}")
    roles = defaultdict(set)
    for role, ids in manifest.roles.items():
        for record_id in ids:
            roles[record_id].add(role)
    heldout_roles = {"val", "test"}
    train_subjects = {row["subject_uid"] for row in rows if roles[row["record_id"]] - heldout_roles}
    heldout_subjects = {row["subject_uid"] for row in rows if roles[row["record_id"]] & heldout_roles}
    overlap = train_subjects & heldout_subjects
    if overlap:
        raise LeakageError(
            f"ECG-QA patients ask questions in both training roles and val/test: {sorted(overlap)[:5]}"
        )
