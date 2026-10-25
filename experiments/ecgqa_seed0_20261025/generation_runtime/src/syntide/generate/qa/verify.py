"""Acceptance for QA candidates, and the student-context policy.

``verify_task`` runs every blocking check and returns the task with its ``verification`` filled in.
A check that cannot run (missing evidence, an unknown template, an unparseable rationale) returns
``None``, and ``None`` is a failure: an unsupported, missing or inconclusive validator never passes.

Checks (``name@version``):

- ``evidence_present@1``: every required evidence id is in the store and describes one of the inputs.
- ``evidence_supports_answer@1``: an answerable task cites only usable evidence (measured or annotation,
  present or absent); an unanswerable one cites at least one unknown or unobservable fact. Requested
  generator controls and diagnostic hypotheses never support an answer.
- ``answer_recomputed@1``: rebuilding the template from the stored evidence and parameters gives the
  same answer, options and answerability.
- ``rationale_consistent@1``: every number in the rationale is a value of cited evidence (or a value
  derived in the template's own claim), no option other than the answer is asserted, and the
  rationale holds no ``Answer:`` line of its own.
- ``context_clean@1``: the student context holds only permitted keys and no generator control.
- ``lineage_resolved@1``: every input names its real roots and their patients.

Clinical validity is outside what these checks establish; every task records
``clinical_review: not_reviewed`` (or the source dataset's status).

Context policy ``ecgqa-cot-baseline@1`` keeps the student prompt identical to the RunPod baseline:
a real recording's ECG-QA-CoT ``clinical_context`` is passed verbatim. That text names PTB-XL
signal-quality flags, so tasks whose answer it states are marked ``context_reveals_answer``. A
derived recording (edited or generated) keeps the format but drops the "Signal quality:" sentence,
which would describe its parent, not itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from syntide.data.qa import ANSWER_KINDS, INSUFFICIENT, Evidence, QATask, format_number
from syntide.generate.qa.templates import ANNOTATION_FAMILIES, TEMPLATES, View, build_task

CHECKS = (
    "evidence_present@1",
    "evidence_supports_answer@1",
    "answer_recomputed@1",
    "rationale_consistent@1",
    "context_clean@1",
    "lineage_resolved@1",
)
CONTEXT_POLICY = "ecgqa-cot-baseline@1"
CONTEXT_KEYS = frozenset({"clinical_context"})
FORBIDDEN_CONTEXT = re.compile(r"amplitude_mv|seed|recipe|controls|requested|heart_rate=|conditioning", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?")
_SIGNAL_QUALITY = re.compile(r"\s*Signal quality:[^\n]*", re.I)


def student_context(origin: str, source_context: str | None) -> str:
    """The clinical context a student sees for one input under ``ecgqa-cot-baseline@1``."""
    text = (source_context or "").strip()
    if origin == "real":
        return text
    if origin == "transformed":
        return _SIGNAL_QUALITY.sub("", text).strip()
    return "12-lead ECG."


def reveals_answer(task: QATask) -> bool:
    """Whether the student context states this task's answer (kept, by the baseline policy, but marked)."""
    if task.answerability != "answerable":
        return False
    contexts = " ".join(task.context.get("clinical_context", [])).lower()
    if task.family in ANNOTATION_FAMILIES and "signal quality" in contexts:
        return True
    if task.answer_type in ("choice", "multi_choice"):
        answers = task.answer if isinstance(task.answer, list) else [task.answer]
        return any(len(str(a)) > 3 and str(a).lower() in contexts for a in answers)
    return False


def verify_task(task: QATask, store: Mapping[str, Evidence]) -> QATask:
    """Run every check; return the task with ``verification`` recorded (accepted or rejected)."""
    results = {
        "evidence_present@1": _evidence_present(task, store),
        "evidence_supports_answer@1": _supports(task, store),
        "answer_recomputed@1": _recomputed(task, store),
        "rationale_consistent@1": _rationale(task, store),
        "context_clean@1": _context(task),
        "lineage_resolved@1": all(ref.root_record_ids and ref.root_subject_uids for ref in task.inputs),
    }
    failed = [name for name, ok in results.items() if ok is not True]
    verification = {
        "status": "accepted" if not failed else "rejected",
        "checks": [{"name": name, "pass": ok} for name, ok in results.items()],
        "reason": ",".join(failed),
        "context_policy": CONTEXT_POLICY,
        "context_reveals_answer": reveals_answer(task),
        "clinical_review": task.verification.get("clinical_review", "not_reviewed"),
    }
    return replace(task, verification=verification)


def _evidence_present(task: QATask, store: Mapping[str, Evidence]) -> bool | None:
    if not task.required_evidence:
        return None
    shas = {ref.waveform_sha256 for ref in task.inputs}
    return all(eid in store and store[eid].waveform_sha256 in shas for eid in task.required_evidence)


def _supports(task: QATask, store: Mapping[str, Evidence]) -> bool | None:
    cited = [store.get(eid) for eid in task.required_evidence]
    if not cited or any(item is None for item in cited):
        return None
    if any(item.kind not in ANSWER_KINDS for item in cited):
        return False
    if task.answerability == "answerable":
        return all(item.usable for item in cited)
    return any(item.status in ("unknown", "unobservable") for item in cited)


def _recomputed(task: QATask, store: Mapping[str, Evidence]) -> bool | None:
    template = TEMPLATES.get(task.template)
    if template is None or task.lineage.get("source_dataset"):
        return None
    params = task.generation.get("template_params")
    if params is None:
        return None
    # Template-derived facts (a stated lead subset) are recomputed by the template itself.
    evidence = list(store.values())
    views = [View(ref, evidence) for ref in task.inputs]
    rebuilt = build_task(template, params, views, task.context, task.generation)
    if rebuilt is None:
        return False
    again, _ = rebuilt
    return (again.answer, again.options, again.answerability, again.answer_type) == (
        task.answer,
        task.options,
        task.answerability,
        task.answer_type,
    )


def _rationale(task: QATask, store: Mapping[str, Evidence]) -> bool | None:
    text = task.rationale
    if re.search(r"(?im)^\s*answer\s*:", text):
        return False
    cited = [store[eid] for claim in task.claims for eid in claim.evidence_ids if eid in store]
    allowed = set()
    for item in cited:
        allowed |= _numbers_of(item)
    for claim in task.claims:  # numbers the template derived and wrote into its own claims
        allowed |= {float(n) for n in _NUMBER.findall(claim.text)}
    if task.answer_type == "numeric":
        allowed.add(round(float(task.answer), 3))
    allowed |= {float(n) for n in _NUMBER.findall(task.question)}
    for number in _NUMBER.findall(text):
        if not any(
            abs(float(number) - a) <= 5e-4 + 1e-9 for a in allowed
        ):  # numbers are written to 3 decimals
            return False
    if task.answer_type in ("choice", "binary") and task.answer != INSUFFICIENT:
        lowered = text.lower()
        others = [o for o in task.options if o != task.answer and len(o) > 3]
        conclusion = lowered.rsplit(".", 2)[-2] if lowered.count(".") >= 2 else lowered
        if any(f"answer is {o.lower()}" in lowered or f"therefore {o.lower()}" in conclusion for o in others):
            return False
    return True


def _numbers_of(item: Evidence) -> set[float]:
    values: set[float] = set()

    def add(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int | float):
            values.add(round(float(value), 3))
            values.add(float(format_number(value)))
        elif isinstance(value, list | tuple):
            for v in value:
                add(v)

    add(item.value)
    add(item.signal.num_samples)
    add(item.signal.sample_rate)
    if item.attribute == "heart_rate_bpm" and item.value:
        values.add(float(format_number(60 / item.value)))
    for lead in item.leads:  # lead names such as V2 contain digits
        values |= {float(n) for n in _NUMBER.findall(lead)}
    return values


def _context(task: QATask) -> bool:
    if set(task.context) - CONTEXT_KEYS:
        return False
    contexts = task.context.get("clinical_context", [])
    if len(contexts) != len(task.inputs):
        return False
    return not any(FORBIDDEN_CONTEXT.search(text) for text in contexts)
