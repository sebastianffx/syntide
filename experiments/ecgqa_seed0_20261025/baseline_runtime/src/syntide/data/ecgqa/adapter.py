"""Read ECG-QA-CoT's questions on top of PTB-XL's signals.

The questions come from ``physionet/ecg-qa-cot`` (one ``AnswerTask`` per question). The signals, the
patient, the fold and the noise flags come from ``physionet/ptb-xl@1.0.3`` at 100 Hz, the data the
editor is calibrated on. The two join on ``record_id`` (``ptbxl-<ecg_id>``). Only questions whose
ECG is a labelled PTB-XL record are kept.

The CoT release has its own split; it is kept as ``cot_split`` for reporting only. SYNTIDE splits by
PTB-XL folds and whole patients (``syntide.data.challenge``).
"""

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from timenet.reader import TimeFReader
from timenet.refs import split_ref
from timenet.registry import LocalRegistry

from syntide.data.adapter import PtbXlAdapter
from syntide.data.schema import LEADS, Samples

DATASET = "physionet/ecg-qa-cot@1.0.0"
SIGNALS = "physionet/ptb-xl@1.0.3"
# Task columns in order. Record-level columns repeat PTB-XL's index for the question's ECG.
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


class EcgQaAdapter:
    """A read-only view of ECG-QA-CoT's questions, keyed to PTB-XL records."""

    def __init__(self, registry: str | Path, dataset: str = DATASET, signals: str = SIGNALS) -> None:
        dataset_id, version = split_ref(dataset)
        self.signals = PtbXlAdapter(registry, signals)
        self._version = LocalRegistry(Path(registry)).open_version(dataset_id, version)
        self._reader = TimeFReader(self._version)
        self.dataset_id = dataset_id
        self.dataset_version = str(self._version.manifest.metadata.dataset_version)
        self._tasks: pa.Table | None = None
        self._dropped: Counter[str] = Counter()
        self._sha256: str | None = None

    def source(self) -> dict:
        """Identify the questions and the signals they are read with, for manifests and trial identity."""
        return {
            "dataset": self.dataset_id,
            "version": self.dataset_version,
            "sha256": self.source_sha256(),
            "signals": self.signals.source(),
        }

    def source_sha256(self) -> str:
        """Hash every task column plus the signal dataset's own fingerprint.

        The signals SYNTIDE reads are PTB-XL's, so ecg-qa-cot's 500 Hz files are not hashed.
        """
        if self._sha256 is None:
            digest = hashlib.sha256(self.signals.source_sha256().encode())
            columns = self.tasks().to_pydict()
            for row in zip(*(columns[name] for name in TASK_COLUMNS), strict=True):
                digest.update(repr(row).encode())
            self._sha256 = digest.hexdigest()
        return self._sha256

    def tasks(self) -> pa.Table:
        """One row per question whose ECG is a labelled PTB-XL record, ordered by ECG id, then task id."""
        if self._tasks is None:
            self._tasks = self._read_tasks()
        return self._tasks

    def index(self) -> pa.Table:
        """PTB-XL's record index, restricted to records with at least one question, plus ``n_tasks``.

        It has every column ``syntide.data.challenge`` and the leakage gate read, so manifests are
        built and checked exactly as for PTB-XL.
        """
        counts = Counter(self.tasks()["record_id"].to_pylist())
        index = self.signals.index()
        keep = pa.array([record_id in counts for record_id in index["record_id"].to_pylist()])
        index = index.filter(keep)
        return index.append_column(
            "n_tasks", pa.array([counts[r] for r in index["record_id"].to_pylist()], pa.int64())
        )

    def load(self, record_ids: Sequence[str]) -> Samples:
        """Signals and PTB-XL labels for ``record_ids``, in that order (100 Hz, mV)."""
        return self.signals.load(record_ids)

    def raw_signals(self, record_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """ecg-qa-cot's own leads (500 Hz, mV) in clinical order, for the signal check only."""
        out = {}
        for record in self._reader.iter_records(list(record_ids), with_annotations=False):
            by_name = {signal.name.upper(): signal for signal in record.signals}
            out[record.record_id] = np.stack([by_name[lead].to_numpy() for lead in LEADS]).astype(np.float64)
        return out

    def tasks_of(self, record_ids: Sequence[str]) -> pa.Table:
        """The questions on ``record_ids``, in task order."""
        wanted = pa.array(sorted(set(record_ids)))
        tasks = self.tasks()
        return tasks.filter(pc.is_in(tasks["record_id"], value_set=wanted))

    def dropped(self) -> dict[str, int]:
        """Questions not kept, by reason (for the counts report)."""
        self.tasks()
        return dict(self._dropped)

    def _read_tasks(self) -> pa.Table:
        ptbxl = {row["record_id"]: row for row in self.signals.index().to_pylist()}
        split_of = self._cot_splits()
        rows: dict[str, list] = defaultdict(list)
        for task in self._reader.iter_tasks():
            if len(task.inputs) != 1:
                raise ValueError(f"{task.id} has {len(task.inputs)} input records; ECG-QA-CoT has one")
            record_id = task.inputs[0].record_id
            if record_id not in ptbxl:
                self._dropped["ecg_not_a_labelled_ptbxl_record"] += 1
                continue
            facts = {annotation.key: annotation.value for annotation in task.input_annotations}
            record = ptbxl[record_id]
            values = {
                "task_id": task.id,
                "record_id": record_id,
                "subject_uid": record["subject_uid"],
                "strat_fold": record["strat_fold"],
                "is_artifact": record["is_artifact"],
                "artifact_types": record["artifact_types"],
                "cot_split": split_of[record_id],
                "question_type": str(facts["question_type"]),
                "template_id": int(facts["template_id"]),
                "question": task.prompt or "",
                "answer": _single_answer(task),
                "answer_options": [str(option) for option in facts.get("answer_options") or []],
                "clinical_context": str(facts.get("clinical_context", "")),
                "rationale": task.rationale or "",
                "has_rationale": bool(task.rationale and task.rationale.strip()),
            }
            for name in TASK_COLUMNS:
                rows[name].append(values[name])
        order = sorted(
            range(len(rows["task_id"])),
            key=lambda i: (int(rows["record_id"][i].rpartition("-")[2]), rows["task_id"][i]),
        )
        types = {
            "strat_fold": pa.int64(),
            "template_id": pa.int64(),
            "artifact_types": pa.list_(pa.string()),
            "answer_options": pa.list_(pa.string()),
        }
        return pa.table(
            {name: pa.array([rows[name][i] for i in order], types.get(name)) for name in TASK_COLUMNS}
        )

    def _cot_splits(self) -> dict[str, str]:
        table = self._reader.annotation_table("Record").select(["object_id", "key", "text_value"])
        return {row["object_id"]: row["text_value"] for row in table.to_pylist() if row["key"] == "split"}


def _single_answer(task) -> str:
    targets = task.targets or ()
    if len(targets) != 1:
        raise ValueError(f"{task.id} has {len(targets)} answers; ECG-QA-CoT has exactly one per question")
    return str(targets[0])
