"""Real recordings for QA generation, restricted to one manifest role, plus the original ECG-QA-CoT tasks.

A ``QASource`` is the population a build may read: rows (record, patient, annotations, the ECG-QA-CoT
clinical context and original questions) and an ordered waveform loader. ``open_source`` builds it from
the existing PTB-XL adapter (and the ECG-QA view when questions are needed) after the leakage gate
(Gate 1), the QA join check and ``check_lineage`` pass for every record of the role. Waveforms are
loaded only for that role.

``purpose='training'`` accepts training roles only (``pool``, ``depleted``, ``natural_mix``);
``purpose='evaluation'`` accepts ``val`` or ``test`` only. An evaluation source can build evaluation
suites, never a training corpus, and a training source never reads a held-out waveform.

``source_tasks`` turns the original ECG-QA-CoT questions into ``QATask``s with the baseline-compatible
context, unchanged answers and rationales, and ``clinical_review: source_dataset_unreviewed``. Their
answers come from the dataset, not from evidence, so they are never re-derived after a waveform edit.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from syntide.data.qa import InputRef, QATask, SignalInfo
from syntide.data.schema import FS_HZ, LEADS
from syntide.explore.plans import content_id
from syntide.generate.qa.evidence import waveform_sha256

TRAINING_ROLES = ("pool", "depleted", "natural_mix")
EVALUATION_ROLES = ("val", "test")
PTBXL = "physionet/ptb-xl@1.0.3"
ECGQA = "physionet/ecg-qa-cot@1.0.0"
SOURCE_TASK_RULE = "ecgqa-cot-source@1"


@dataclass
class SourceRow:
    record_id: str
    subject_uid: str
    labels: list[str]
    artifact_types: list[str]
    clinical_context: str = ""
    questions: list[dict[str, Any]] = field(default_factory=list)


class QASource:
    """A checked population of real recordings and its restricted loader."""

    def __init__(
        self,
        rows: Sequence[SourceRow],
        loader: Callable[[list[str]], np.ndarray],
        provenance: dict[str, Any],
        *,
        purpose: str,
        signal: SignalInfo | None = None,
        dataset: str = PTBXL,
        checker: Callable[[list[tuple[str, tuple[str, ...], tuple[str, ...]]]], None] | None = None,
    ) -> None:
        self.rows = sorted(
            rows,
            key=lambda row: (
                int(row.record_id.rpartition("-")[2]) if row.record_id.rpartition("-")[2].isdigit() else 0,
                row.record_id,
            ),
        )
        self._by_id = {row.record_id: row for row in self.rows}
        self._loader = loader
        self.provenance = provenance
        self.purpose = purpose
        self.signal = signal or SignalInfo(LEADS, FS_HZ, "mV", 10 * FS_HZ)
        self.dataset = dataset
        self._checker = checker

    @property
    def subject_of(self) -> dict[str, str]:
        return {row.record_id: row.subject_uid for row in self.rows}

    def check_ancestry(self, items: Sequence[tuple[str, Sequence[str], Sequence[str]]]) -> None:
        """Every (input id, roots, subjects) must descend only from this source's role and its patients."""
        items = [(i, tuple(r), tuple(s)) for i, r, s in items]
        if self._checker is not None:
            self._checker(items)
            return
        from syntide.data.leakage import LeakageError

        subjects = self.subject_of
        bad = [
            i
            for i, roots, subs in items
            if not roots
            or len(roots) != len(subs)
            or any(subjects.get(r) != s for r, s in zip(roots, subs, strict=True))
        ]
        if bad:
            raise LeakageError(
                f"{len(bad)} task inputs descend from records outside the source role: {bad[:5]}"
            )

    def without_subjects(self, subjects: set[str]) -> QASource:
        """The same source minus every record of these patients (whole patients only)."""
        rows = [row for row in self.rows if row.subject_uid not in subjects]
        provenance = {
            **self.provenance,
            "excluded_subjects": len({r.subject_uid for r in self.rows} & subjects),
        }
        provenance["fingerprint"] = content_id(
            {k: v for k, v in provenance.items() if k not in ("manifest", "fingerprint")}
            | {"records": [r.record_id for r in rows]}
        )
        return QASource(
            rows,
            self._loader,
            provenance,
            purpose=self.purpose,
            signal=self.signal,
            dataset=self.dataset,
            checker=self._checker,
        )

    def restricted_to(self, record_ids: set[str]) -> QASource:
        """The same source, reading only these recordings."""
        rows = [row for row in self.rows if row.record_id in record_ids]
        provenance = {**self.provenance, "restricted_records": len(rows)}
        provenance["fingerprint"] = content_id(
            {k: v for k, v in provenance.items() if k not in ("manifest", "fingerprint")}
            | {"records": [r.record_id for r in rows]}
        )
        return QASource(
            rows,
            self._loader,
            provenance,
            purpose=self.purpose,
            signal=self.signal,
            dataset=self.dataset,
            checker=self._checker,
        )

    def row(self, record_id: str) -> SourceRow:
        return self._by_id[record_id]

    def load(self, record_ids: Sequence[str]) -> list[tuple[InputRef, np.ndarray]]:
        """Waveforms of population members only, in the requested order, with their input references."""
        ids = list(record_ids)
        outside = sorted(set(ids) - self._by_id.keys())
        if outside:
            raise PermissionError(f"waveform request outside the source population: {outside[:3]}")
        x = self._loader(ids)
        out = []
        for record_id, signal in zip(ids, x, strict=True):
            signal = np.asarray(signal, dtype=np.float32)
            spec = SignalInfo(
                self.signal.channels, self.signal.sample_rate, self.signal.units, signal.shape[-1]
            )
            row = self._by_id[record_id]
            ref = InputRef(
                waveform_id=record_id,
                waveform_sha256=waveform_sha256(signal),
                signal=spec,
                origin="real",
                source_record_ids=(),
                root_record_ids=(record_id,),
                root_subject_uids=(row.subject_uid,),
                dataset=self.dataset,
                lineage={"source": self.provenance.get("fingerprint")},
            )
            out.append((ref, signal))
        return out

    @property
    def fingerprint(self) -> str:
        return self.provenance["fingerprint"]


def open_source(
    registry: str | Path,
    manifest_path: str | Path,
    role: str,
    *,
    purpose: str = "training",
    questions: bool = True,
    dataset: str = PTBXL,
    qa_dataset: str = ECGQA,
    manifest_sha256: str | None = None,
) -> QASource:
    """PTB-XL recordings of one manifest role, with their ECG-QA-CoT questions when ``questions``.

    ``manifest_sha256`` pins the exact split (e.g. the RunPod baseline's ``ecgqa-artifact-50/pool50``);
    any other manifest is refused.
    """
    from syntide.data.adapter import PtbXlAdapter
    from syntide.data.challenge import load_manifest
    from syntide.data.leakage import check_lineage, check_manifest

    _check_role(role, purpose)
    manifest = load_manifest(Path(manifest_path))
    if manifest_sha256 is not None and manifest.sha256 != manifest_sha256:
        raise PermissionError(
            f"{manifest_path} is manifest {manifest.sha256[:12]}, not the pinned {manifest_sha256[:12]}"
        )
    if questions:
        from syntide.explore.qa import EcgQaView, check_qa_tasks

        view = EcgQaView(registry, qa_dataset, dataset)
        index, source = view.index(), view.source()
        check_manifest(manifest, index, source)  # Gate 1, on the QA view's fingerprint
        check_qa_tasks(view.tasks(), manifest, index)
        adapter = view.signals
    else:
        adapter = PtbXlAdapter(registry, dataset)
        index, source = adapter.index(), adapter.source()
        check_manifest(manifest, index, source)
    members = set(manifest.roles[role])
    facts = {row["record_id"]: row for row in index.to_pylist() if row["record_id"] in members}
    check_lineage(((r, (r,), (facts[r]["subject_uid"],)) for r in sorted(facts)), manifest, index, (role,))
    by_record: dict[str, list[dict[str, Any]]] = {}
    contexts: dict[str, str] = {}
    if questions:
        for task in view.tasks().to_pylist():
            if task["record_id"] in members:
                by_record.setdefault(task["record_id"], []).append(task)
                contexts.setdefault(task["record_id"], task["clinical_context"])
    rows = [
        SourceRow(
            record_id=r,
            subject_uid=facts[r]["subject_uid"],
            labels=list(facts[r]["labels"]),
            artifact_types=list(facts[r]["artifact_types"]),
            clinical_context=contexts.get(r, ""),
            questions=by_record.get(r, []),
        )
        for r in sorted(facts)
    ]
    provenance = {
        "source": source,
        "manifest": str(Path(manifest_path)),
        "manifest_sha256": manifest.sha256,
        "challenge_id": manifest.challenge_id,
        "role": role,
        "purpose": purpose,
        "records": len(rows),
        "subjects": len({row.subject_uid for row in rows}),
        "questions": sum(len(row.questions) for row in rows),
        "split_policy": "PTB-XL patient folds through the manifest; the CoT release split is descriptive",
    }
    provenance["fingerprint"] = content_id({k: v for k, v in provenance.items() if k != "manifest"})

    def load(ids: list[str]) -> np.ndarray:
        return adapter.load(ids).x

    def checker(items: list[tuple[str, tuple[str, ...], tuple[str, ...]]]) -> None:
        check_lineage(items, manifest, index, (role,))

    return QASource(rows, load, provenance, purpose=purpose, dataset=dataset, checker=checker)


def open_external_source(
    catalog: str | Path,
    dataset: str,
    *,
    signal: SignalInfo | None = None,
    label_mapping: dict[str, str] | None = None,
) -> QASource:
    """An evaluation-only source from an explorer-style JSONL catalog of another dataset.

    Every row must have ``split: test`` (or ``external``): an external catalog is never a training source.
    Channels go through the generated-signal alias table, the sample rate must equal the target's (no
    silent resampling), and patients are namespaced ``external:<dataset>:<subject>``. ``label_mapping``
    maps the catalog's answer strings onto the canonical option strings and is part of the provenance.
    """
    import json

    from syntide.generate.qa.generated import normalize_channels

    target = signal or SignalInfo(LEADS, FS_HZ, "mV", 10 * FS_HZ)
    path = Path(catalog)
    mapping = dict(label_mapping or {})
    rows, specs = {}, {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("split") not in ("test", "external"):
            raise PermissionError(
                f"{path}:{line_no}: external catalogs hold evaluation rows only (split test)"
            )
        record_id = f"external:{dataset}:{item['record_id']}"
        channels = normalize_channels(item["channels"])
        if set(target.channels) - set(channels) or float(item["sample_rate"]) != target.sample_rate:
            raise ValueError(f"{path}:{line_no}: needs leads {target.channels} at {target.sample_rate} Hz")
        specs[record_id] = (path.parent / item["signal_path"], channels)
        row = rows.setdefault(
            record_id,
            SourceRow(
                record_id, f"external:{dataset}:{item.get('subject_uid') or item['record_id']}", [], []
            ),
        )
        for k, text in enumerate(item.get("texts") or []):
            answer = mapping.get(text["answer"], text["answer"])
            options = [mapping.get(o, o) for o in text.get("answer_options") or []]
            row.questions.append(
                {
                    "task_id": text.get("text_id") or f"{record_id}:q{k}",
                    "record_id": record_id,
                    "question": text["question"],
                    "answer": answer,
                    "answer_options": options,
                    "clinical_context": text.get("clinical_context", ""),
                    "rationale": text.get("rationale", ""),
                    "template_id": int((text.get("text_metadata") or {}).get("template_id", 0)),
                    "question_type": (text.get("text_metadata") or {}).get("question_type", "external"),
                    "cot_split": "external",
                    "is_artifact": False,
                }
            )
            row.clinical_context = row.clinical_context or text.get("clinical_context", "")
    provenance = {
        "dataset": dataset,
        "catalog_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "label_mapping": mapping,
        "purpose": "evaluation",
        "role": "external",
        "records": len(rows),
    }
    provenance["fingerprint"] = content_id(provenance)

    def load(ids: list[str]) -> np.ndarray:
        out = []
        for record_id in ids:
            file, channels = specs[record_id]
            x = np.load(file, allow_pickle=False)
            out.append(np.asarray(x, dtype=np.float32)[[channels.index(lead) for lead in target.channels]])
        return np.stack(out)

    def checker(items: list[tuple[str, tuple[str, ...], tuple[str, ...]]]) -> None:
        from syntide.data.leakage import LeakageError

        bad = [i for i, roots, _ in items if not roots or any(r not in rows for r in roots)]
        if bad:
            raise LeakageError(f"{len(bad)} inputs descend from records outside the external catalog")

    return QASource(
        list(rows.values()),
        load,
        provenance,
        purpose="evaluation",
        signal=target,
        dataset=dataset,
        checker=checker,
    )


def _check_role(role: str, purpose: str) -> None:
    allowed = {"training": TRAINING_ROLES, "evaluation": EVALUATION_ROLES}.get(purpose)
    if allowed is None:
        raise ValueError(f"purpose must be training or evaluation, got {purpose!r}")
    if role not in allowed:
        raise PermissionError(f"a {purpose} source reads roles {allowed}, not {role!r}")


def source_tasks(
    source: QASource, ref: InputRef, *, dedupe: bool = False, qa_dataset: str = ECGQA
) -> list[QATask]:
    """The original ECG-QA-CoT questions of one real recording, as accepted source supervision.

    ``dedupe`` keeps one task per (record, question), the lowest task id (the baseline's evaluation
    rule); training keeps every question/rationale row, as the baseline did.
    """
    row = source.row(ref.waveform_id)
    questions = sorted(row.questions, key=lambda q: q["task_id"])
    if dedupe:
        seen, kept = set(), []
        for q in questions:
            if q["question"] not in seen:
                seen.add(q["question"])
                kept.append(q)
        questions = kept
    out = []
    for q in questions:
        options = tuple(q["answer_options"])
        answer = q["answer"]
        answer_type = "binary" if set(options) == {"yes", "no"} else "choice"
        if answer not in options:  # a source answer outside its declared options is kept, but as an option
            options = (*options, answer)
        out.append(
            QATask(
                family=f"ecgqa.template{int(q['template_id']):02d}",
                template=f"ecg-qa-cot/template{int(q['template_id'])}@source",
                inputs=(ref,),
                question=q["question"],
                answer_type=answer_type,
                answer=answer,
                options=options,
                answerability="answerable",
                rationale=baseline_rationale(q["rationale"]),
                claims=(),
                required_evidence=(),
                context={"clinical_context": [q["clinical_context"]]},
                tolerance=None,
                semantic_key="sem-"
                + content_id([ref.waveform_sha256, q["template_id"], q["question"], answer])[:24],
                paraphrase="seed",
                composition="source",
                generation={"generator": SOURCE_TASK_RULE, "operator": "source"},
                verification={
                    "status": "accepted",
                    "source": qa_dataset,
                    "checks": [{"name": SOURCE_TASK_RULE, "pass": True}],
                    "reason": "",
                    "context_policy": "ecgqa-cot-baseline@1",
                    "context_reveals_answer": "signal quality" in q["clinical_context"].lower()
                    and q["question_type"].startswith("single")
                    and _is_noise_question(q["question"]),
                    "clinical_review": "source_dataset_unreviewed",
                    "rationale_status": "source_unverified",
                },
                lineage={
                    "source_dataset": qa_dataset,
                    "source_task_id": q["task_id"],
                    "template_id": int(q["template_id"]),
                    "question_type": q["question_type"],
                    "cot_split": q["cot_split"],
                    "is_artifact": bool(q["is_artifact"]),
                    "source_rationale_sha256": hashlib.sha256((q["rationale"] or "").encode()).hexdigest(),
                },
            )
        )
    return out


def baseline_rationale(text: str | None) -> str:
    """The source rationale without its own trailing ``Answer:`` (the baseline's ``target_text`` rule), so
    the gold answer is appended exactly once."""
    return re.split(r"(?i)\banswer\s*:", text or "")[0].strip()


def _is_noise_question(question: str) -> bool:
    lowered = question.lower()
    return any(word in lowered for word in ("noise", "drift", "electrode", "artifact"))
