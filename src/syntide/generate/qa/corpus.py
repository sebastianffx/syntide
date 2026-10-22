"""The QA corpus: bounded, restartable construction and atomic, immutable publication.

Construction writes into ``<output>.staging/``:

- ``signals/<sha256>.npy``: each distinct waveform once, however many questions it has;
- ``chunks/<unit>.json``: per work unit, the accepted tasks, every evidence record they cite, and the
  ledger of every candidate (accepted or rejected, with its reason). A chunk is written to a
  ``.partial`` file and renamed, then ``progress.json`` records it with its sha256;
- ``progress.json``: the build identity and the finished units. A rerun with the same identity skips
  finished units (after re-checking their sha256); a different identity is refused.

``publish`` re-verifies every chunk, then writes the committed outputs into the staging directory:

- ``timenet/``: a TimeNet dataset, one ``Record`` per waveform (its evidence in the record metadata)
  and one ``AnswerTask`` per accepted task, with the task's ordered inputs. A comparison task is one
  task over two records, never two tasks;
- ``catalog.jsonl``: the explorer's recording-to-many-``texts`` catalog (one row per waveform);
- ``corpus.json``: identity, counts, a content hash and the sha256 of every file. ``corpus_id`` hashes the
  identity, counts, content and every file except TimeNet's own (rewritten with new bytes on each
  write); those are verified on load but kept out of the id, so a rebuild gives the same id.

Only then is the staging directory renamed to ``<output>``. ``<output>`` therefore exists only for a
complete, verified corpus; an interrupted build or publish leaves nothing there. A published corpus is
never modified: a changed plan, policy or rule set builds a new one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from syntide.data.qa import Evidence, InputRef, QATask
from syntide.explore.plans import content_id

SCHEMA = "syntide.qa-corpus@1"
PURPOSES = ("training", "evaluation")
LEDGER_DECISIONS = ("accepted", "rejected")


class CorpusError(RuntimeError):
    """A corpus build was interrupted inconsistently, or a published corpus does not verify."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(partial, path)
    return _sha256(path)


def ledger_row(task: QATask, unit: str, attempt: str) -> dict[str, Any]:
    """One candidate, accepted or rejected, without its waveform."""
    v = task.verification
    return {
        "task_id": task.task_id,
        "unit": unit,
        "attempt": attempt,
        "family": task.family,
        "template": task.template,
        "composition": task.composition,
        "inputs": [ref.waveform_id for ref in task.inputs],
        "origins": [ref.origin for ref in task.inputs],
        "answerability": task.answerability,
        "decision": v.get("status"),
        "reason": v.get("reason", ""),
        "semantic_key": task.semantic_key,
        "paraphrase": task.paraphrase,
        "generation": task.generation,
    }


def rejection(unit: str, attempt: str, family: str, reason: str, **details: Any) -> dict[str, Any]:
    """A ledger row for a candidate that never became a task (invalid signal, failed remeasurement)."""
    return {
        "task_id": None,
        "unit": unit,
        "attempt": attempt,
        "family": family,
        "template": None,
        "composition": None,
        "inputs": details.pop("inputs", []),
        "origins": details.pop("origins", []),
        "answerability": None,
        "decision": "rejected",
        "reason": reason,
        "semantic_key": None,
        "paraphrase": None,
        "generation": details,
    }


class CorpusWriter:
    """Stage a corpus build. Use ``unit_done``/``write_unit`` per work unit, then ``publish``."""

    def __init__(self, output: str | Path, identity: Mapping[str, Any]) -> None:
        self.output = Path(output).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"{self.output} is already a published corpus; corpora are immutable")
        if identity.get("purpose") not in PURPOSES:
            raise CorpusError(f"identity.purpose must be one of {PURPOSES}")
        self.identity = json.loads(json.dumps(dict(identity), allow_nan=False))
        self.identity_id = content_id(self.identity)
        self.staging = self.output.with_name(self.output.name + ".staging")
        self._progress_path = self.staging / "progress.json"
        if self._progress_path.exists():
            progress = json.loads(self._progress_path.read_text(encoding="utf-8"))
            if progress["identity_id"] != self.identity_id:
                raise CorpusError(
                    f"{self.staging} holds an interrupted build of another configuration "
                    f"({progress['identity_id'][:12]}); remove it on purpose or choose another output"
                )
            for unit, digest in progress["units"].items():
                path = self.staging / "chunks" / f"{unit}.json"
                if not path.exists() or _sha256(path) != digest:
                    raise CorpusError(f"{path} is missing or changed since it was committed")
            self.progress = progress
        else:
            self.progress = {"identity_id": self.identity_id, "identity": self.identity, "units": {}}
            (self.staging / "signals").mkdir(parents=True, exist_ok=True)
            _write_json(self._progress_path, self.progress)

    def unit_done(self, unit: str) -> bool:
        return unit in self.progress["units"]

    def put_signal(self, ref: InputRef, x: np.ndarray) -> None:
        """Store a waveform once, under its sha256."""
        path = self.staging / "signals" / f"{ref.waveform_sha256}.npy"
        if path.exists():
            return
        partial = path.with_name(path.name + ".partial.npy")
        np.save(partial, np.ascontiguousarray(x, dtype=np.float32), allow_pickle=False)
        os.replace(partial, path)

    def write_unit(
        self,
        unit: str,
        tasks: Iterable[QATask],
        evidence: Iterable[Evidence],
        ledger: Iterable[dict[str, Any]],
        waveforms: Iterable[InputRef],
    ) -> None:
        """Commit one work unit. ``waveforms`` are every input it stored, accepted or not."""
        if self.unit_done(unit):
            raise CorpusError(f"unit {unit} is already committed")
        accepted = [t for t in tasks if t.verification.get("status") == "accepted"]
        payload = {
            "unit": unit,
            "tasks": [t.to_json() for t in accepted],
            "evidence": [e.to_json() for e in dict((e.evidence_id, e) for e in evidence).values()],
            "ledger": list(ledger),
            "waveforms": [ref.to_json() for ref in dict((r.waveform_sha256, r) for r in waveforms).values()],
        }
        digest = _write_json(self.staging / "chunks" / f"{unit}.json", payload)
        self.progress["units"][unit] = digest
        _write_json(self._progress_path, self.progress)

    def chunks(self) -> Iterator[dict[str, Any]]:
        for unit, digest in sorted(self.progress["units"].items()):
            path = self.staging / "chunks" / f"{unit}.json"
            if _sha256(path) != digest:
                raise CorpusError(f"{path} changed since it was committed")
            yield json.loads(path.read_text(encoding="utf-8"))

    def publish(self, *, expected_units: Iterable[str]) -> dict[str, Any]:
        """Verify, write TimeNet + catalog + corpus.json, then rename staging to the output."""
        missing = sorted(set(expected_units) - set(self.progress["units"]))
        if missing:
            raise CorpusError(
                f"{len(missing)} work units are not finished (e.g. {missing[:3]}); nothing published"
            )
        tasks: dict[str, QATask] = {}
        evidence: dict[str, Evidence] = {}
        waveforms: dict[str, InputRef] = {}
        ledger: list[dict[str, Any]] = []
        for chunk in self.chunks():
            for item in chunk["evidence"]:
                evidence.setdefault(item["evidence_id"], Evidence.from_json(item))
            for item in chunk["waveforms"]:
                ref = InputRef.from_json(item)
                waveforms.setdefault(ref.waveform_id, ref)
            for item in chunk["tasks"]:
                task = QATask.from_json(item)
                tasks.setdefault(task.task_id, task)
            ledger.extend(chunk["ledger"])
        _check_publishable(tasks.values(), evidence, waveforms, self.staging / "signals")
        used = {ref.waveform_id for task in tasks.values() for ref in task.inputs}
        records = {wid: waveforms[wid] for wid in sorted(used)}
        published = {ref.waveform_sha256 for ref in records.values()}
        evidence = {eid: e for eid, e in evidence.items() if e.waveform_sha256 in published}
        signals = self.staging / "signals"
        for path in signals.glob("*.npy"):  # waveforms of rejected candidates are not published
            if path.stem not in {ref.waveform_sha256 for ref in records.values()}:
                path.unlink()
        ordered = sorted(tasks.values(), key=lambda t: (t.inputs[0].waveform_id, t.family, t.task_id))
        _write_timenet(self.staging / "timenet", self.identity, records, ordered, evidence, signals)
        _write_catalog(self.staging / "catalog.jsonl", records, ordered, evidence, ledger)
        _write_json(self.staging / "ledger.json", ledger)
        counts = corpus_counts(ordered, records, ledger)
        everything = {
            str(path.relative_to(self.staging)): _sha256(path)
            for path in sorted(self.staging.rglob("*"))
            if path.is_file() and path.name not in ("progress.json",) and "chunks" not in path.parts
        }
        # TimeNet rewrites its control database and manifest with new bytes on every write, so the id hashes
        # the content (tasks, evidence, inputs, signals, catalog, ledger); TimeNet's files are verified only.
        files = {k: v for k, v in everything.items() if not k.startswith("timenet/")}
        content = content_id(
            {
                "tasks": [t.task_id for t in ordered],
                "evidence": sorted(evidence),
                "inputs": [records[w].to_json() for w in records],
            }
        )
        body = {
            "schema_version": SCHEMA,
            "identity": self.identity,
            "identity_id": self.identity_id,
            "counts": counts,
            "files": files,
            "content_sha256": content,
        }
        info = {
            **body,
            "corpus_id": "corpus-" + content_id(body)[:24],
            "status": "published",
            "timenet_files": {k: v for k, v in everything.items() if k.startswith("timenet/")},
        }
        _write_json(self.staging / "corpus.json", info)
        shutil.rmtree(self.staging / "chunks")
        self._progress_path.unlink()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.staging, self.output)
        return info


def _check_publishable(
    tasks: Iterable[QATask],
    evidence: Mapping[str, Evidence],
    waveforms: Mapping[str, InputRef],
    signals: Path,
) -> None:
    for task in tasks:
        if task.verification.get("status") != "accepted":
            raise CorpusError(f"{task.task_id} is not accepted; only accepted supervision is published")
        for ref in task.inputs:
            if ref.waveform_id not in waveforms or waveforms[ref.waveform_id] != ref:
                raise CorpusError(f"{task.task_id}: input {ref.waveform_id} has no stored waveform record")
            if not (signals / f"{ref.waveform_sha256}.npy").exists():
                raise CorpusError(f"{task.task_id}: the waveform of {ref.waveform_id} was not stored")
            review = ref.lineage.get("review_status")
            if ref.origin == "generated" and review != "verified":
                raise CorpusError(
                    f"{task.task_id}: generated waveform {ref.waveform_id} is {review!r}; pending generator "
                    "outputs never enter an accepted export"
                )
        for eid in task.required_evidence:
            if eid not in evidence:
                raise CorpusError(f"{task.task_id}: evidence {eid} is not stored")


def corpus_counts(
    tasks: list[QATask], records: Mapping[str, InputRef], ledger: list[dict[str, Any]]
) -> dict[str, Any]:
    """Patients, recordings, waveforms, semantic tasks, examples, paraphrases, attempts and tokens."""
    roots = {r for ref in records.values() for r in ref.root_record_ids}
    subjects = {s for ref in records.values() for s in ref.root_subject_uids}
    origins = Counter(ref.origin for ref in records.values())
    decisions = Counter(row["decision"] for row in ledger)
    reasons = Counter(
        reason
        for row in ledger
        if row["decision"] == "rejected"
        for reason in row["reason"].split(",")
        if reason
    )
    question_tokens = sum(len(t.question.split()) for t in tasks)
    return {
        "unique_patients": len(subjects),
        "root_recordings": len(roots),
        "waveforms": len(records),
        "waveforms_by_origin": dict(sorted(origins.items())),
        "generated_waveforms": origins.get("generated", 0),
        "semantic_tasks": len({t.semantic_key for t in tasks}),
        "qa_examples": len(tasks),
        "paraphrases": sum(t.paraphrase != "seed" for t in tasks),
        "attempts": len(ledger),
        "accepted": decisions.get("accepted", 0),
        "rejected": decisions.get("rejected", 0),
        "rejection_reasons": dict(sorted(reasons.items())),
        "by_family": dict(sorted(Counter(t.family for t in tasks).items())),
        "by_composition": dict(sorted(Counter(t.composition for t in tasks).items())),
        "by_answerability": dict(sorted(Counter(t.answerability for t in tasks).items())),
        "context_reveals_answer": sum(bool(t.verification.get("context_reveals_answer")) for t in tasks),
        "tokens": {
            "question_whitespace_tokens": question_tokens,
            "answer_only_target_whitespace_tokens": sum(len(t.target("answer_only").split()) for t in tasks),
            "rationale_target_whitespace_tokens": sum(len(t.target("rationale").split()) for t in tasks),
            "note": "whitespace tokens; trainers report their own tokenizer counts in result.json",
        },
    }


def _write_timenet(
    root: Path,
    identity: Mapping[str, Any],
    records: Mapping[str, InputRef],
    tasks: list[QATask],
    evidence: Mapping[str, Evidence],
    signals: Path,
) -> None:
    from timenet.dataset import Record, Signal, Source, TimeFDataset
    from timenet.dataset.axis import RegularAxis
    from timenet.types import AnswerTask, TimeSeriesSpec, ureg
    from timenet.types.metadata import DatasetMetadata

    spec = TimeSeriesSpec(spec_type="ecg", name="12-lead ECG", unit_value=ureg.millivolt)
    metadata = DatasetMetadata(
        dataset_id=dataset_id(identity),
        dataset_version="1.0.0",
        name=f"SYNTIDE QA corpus {identity.get('name', '')}".strip(),
        description="Evidence-grounded ECG question-answer-rationale tasks built by syntide.generate.qa. "
        "Synthetic and edited waveforms are not clinically validated.",
        license=identity.get("license", "CC-BY-4.0"),
        tags=("ecg", "qa", "synthetic", identity["purpose"]),
    )
    dataset = TimeFDataset(metadata=metadata)
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence.values():
        by_sha[item.waveform_sha256].append(item.to_json())
    built = {}
    for wid, ref in records.items():
        if ref.signal.units != "mV":
            raise CorpusError(f"{wid}: the corpus stores ECGs in mV, not {ref.signal.units}")
        x = np.load(signals / f"{ref.waveform_sha256}.npy", allow_pickle=False)
        axis = RegularAxis.from_rate_hz(Fraction(str(ref.signal.sample_rate)))
        leads = tuple(
            Signal(spec=spec, time_axis=axis, name=lead, data=x[k], id=f"{wid}/{lead}")
            for k, lead in enumerate(ref.signal.channels)
        )
        record = Record(
            record_id=wid,
            sources=(Source(id=f"{wid}/source", name=ref.origin, signals=leads),),
            metadata={
                "input": ref.to_json(),
                "evidence": sorted(by_sha[ref.waveform_sha256], key=lambda e: e["evidence_id"]),
            },
        )
        built[wid] = dataset.add_record(record=record)
    dataset.add_tasks(
        tasks=[
            AnswerTask(
                id=task.task_id,
                inputs=tuple(built[ref.waveform_id] for ref in task.inputs),
                targets=(task.answer_text,),
                prompt=task.question,
                rationale=task.rationale or None,
                metadata={"qa_task": task.to_json()},
            )
            for task in tasks
        ]
    )
    dataset.write(path=root)


def dataset_id(identity: Mapping[str, Any]) -> str:
    name = "".join(c if c.isalnum() or c in "-_." else "-" for c in str(identity.get("name", "qa"))).strip(
        "-"
    )
    return f"syntide/{name or 'qa'}"


def _write_catalog(
    path: Path,
    records: Mapping[str, InputRef],
    tasks: list[QATask],
    evidence: Mapping[str, Evidence],
    ledger: list[dict[str, Any]],
) -> None:
    """One row per waveform; a task is attached to its first input only (a pair appears once)."""
    anchored: dict[str, list[QATask]] = defaultdict(list)
    paired: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        anchored[task.inputs[0].waveform_id].append(task)
        if len(task.inputs) > 1:
            ids = [ref.waveform_id for ref in task.inputs]
            for wid in ids:
                paired[wid] |= set(ids) - {wid}
    rejected: dict[str, Counter] = defaultdict(Counter)
    for row in ledger:
        if row["decision"] == "rejected":
            for wid in row["inputs"][:1]:
                rejected[wid][row["reason"]] += 1
    rate_of = {
        e.waveform_sha256: e.value for e in evidence.values() if e.attribute == "heart_rate_bpm" and e.usable
    }
    lines = []
    for wid, ref in records.items():
        own = anchored.get(wid, [])
        rate = rate_of.get(ref.waveform_sha256)
        band = "unknown" if rate is None else "<60" if rate < 60 else "60-100" if rate <= 100 else ">100"
        labels = sorted(
            {
                f"origin:{ref.origin}",
                f"rate:{band}",
                *(f"family:{t.family}" for t in own),
                *(f"answerability:{t.answerability}" for t in own),
            }
        )
        lines.append(
            {
                "record_id": wid,
                "subject_uid": ref.root_subject_uids[0],
                "split": "train",
                "signal_path": f"signals/{ref.waveform_sha256}.npy",
                "sample_rate": ref.signal.sample_rate,
                "channels": list(ref.signal.channels),
                "units": ref.signal.units,
                "labels": labels,
                "metadata": {
                    "origin": "real" if ref.origin == "real" else "synthetic",
                    "synthetic": ref.origin != "real",
                    "qa_origin": ref.origin,
                    "source_record_id": ref.root_record_ids[0],
                    "root_record_ids": list(ref.root_record_ids),
                    "display_name": wid if len(wid) <= 24 else f"{wid[:10]}…{wid[-8:]}",
                    "waveform_sha256": ref.waveform_sha256,
                    "generation": ref.lineage,
                    "qa_tasks": len(own),
                    "qa_rejected": sum(rejected[wid].values()),
                    "qa_rejection_reasons": dict(rejected[wid].most_common(5)),
                    "paired_with": sorted(paired.get(wid, ())),
                },
                "texts": [
                    {
                        "text_id": task.task_id,
                        "question": task.question,
                        "answer": task.answer_text,
                        "rationale": task.rationale,
                        "text_metadata": {
                            "family": task.family,
                            "template": task.template,
                            "composition": task.composition,
                            "answerability": task.answerability,
                            "paraphrase": task.paraphrase,
                            "evidence_confidence": _confidence(task, evidence),
                            "status": task.verification.get("status"),
                            "context_reveals_answer": task.verification.get("context_reveals_answer", False),
                            "inputs": [r.waveform_id for r in task.inputs],
                            "pair_id": task.lineage.get("pair_id"),
                        },
                    }
                    for task in own
                ],
            }
        )
    path.write_text(
        "".join(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n" for line in lines),
        encoding="utf-8",
    )


def _confidence(task: QATask, evidence: Mapping[str, Evidence]) -> float | None:
    values = [
        evidence[e].confidence
        for e in task.required_evidence
        if e in evidence and evidence[e].confidence is not None
    ]
    return round(min(values), 4) if values else None


@dataclass
class Corpus:
    """A verified, published corpus, read back from its TimeNet dataset."""

    path: Path
    info: dict[str, Any]
    tasks: list[QATask]
    evidence: dict[str, Evidence]
    inputs: dict[str, InputRef]
    _signals: dict[str, np.ndarray]

    @property
    def corpus_id(self) -> str:
        return self.info["corpus_id"]

    @property
    def purpose(self) -> str:
        return self.info["identity"]["purpose"]

    def signal(self, waveform_id: str) -> np.ndarray:
        return self._signals[waveform_id]


def verify_corpus_files(path: str | Path) -> dict[str, Any]:
    """Check ``corpus.json``'s id and every file checksum. Returns the corpus info."""
    path = Path(path)
    info_path = path / "corpus.json"
    if not info_path.exists():
        raise CorpusError(f"{path} is not a published corpus (no corpus.json)")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    body = {
        k: info[k] for k in ("schema_version", "identity", "identity_id", "counts", "files", "content_sha256")
    }
    if info.get("schema_version") != SCHEMA or info.get("corpus_id") != "corpus-" + content_id(body)[:24]:
        raise CorpusError(f"{info_path} does not match its corpus_id; it was edited")
    for name, digest in {**info["files"], **info["timenet_files"]}.items():
        file = path / name
        if not file.exists() or _sha256(file) != digest:
            raise CorpusError(f"{file} is missing or changed since publication")
    return info


def load_corpus(path: str | Path) -> Corpus:
    """Load a published corpus from its TimeNet dataset, re-checking every task and waveform id."""
    from timenet.reader import TimeFReader
    from timenet.registry import LocalRegistry

    path = Path(path)
    info = verify_corpus_files(path)
    version = LocalRegistry(path / "timenet").open_version(dataset_id(info["identity"]), "1.0.0")
    reader = TimeFReader(version)
    inputs, signals, evidence = {}, {}, {}
    for record in reader.iter_records(reader.record_ids(), with_annotations=False):
        ref = InputRef.from_json(record.metadata["input"])
        by_name = {s.name: s for s in record.signals}
        x = np.stack([np.asarray(by_name[lead].to_numpy(), dtype=np.float32) for lead in ref.signal.channels])
        from syntide.generate.qa.evidence import waveform_sha256

        if waveform_sha256(x) != ref.waveform_sha256:
            raise CorpusError(f"{record.record_id}: stored values do not match waveform_sha256")
        inputs[record.record_id], signals[record.record_id] = ref, x
        for item in record.metadata["evidence"]:
            e = Evidence.from_json(item)
            evidence[e.evidence_id] = e
    tasks = [QATask.from_json(task.metadata["qa_task"]) for task in reader.iter_tasks()]
    tasks.sort(key=lambda t: (t.inputs[0].waveform_id, t.family, t.task_id))
    content = content_id(
        {
            "tasks": [t.task_id for t in tasks],
            "evidence": sorted(evidence),
            "inputs": [inputs[w].to_json() for w in sorted(inputs)],
        }
    )
    if content != info["content_sha256"]:
        raise CorpusError(
            f"{path}: the TimeNet dataset does not hold the published tasks, evidence and inputs"
        )
    return Corpus(path, info, tasks, evidence, inputs, signals)
