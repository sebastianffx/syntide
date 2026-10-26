"""Allocate new question-answer tasks using development-set errors.

Coverage groups tasks by measured signal attributes and question family.
Development predictions determine each group's error estimate and uncertainty.
Only development data may guide allocation; protected evaluation patients are
excluded. Generation plans record their parent corpus, random seed, and task
counts so the selected allocation can be reproduced.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from scipy.stats import beta

from syntide.data.qa import Evidence, QATask
from syntide.explore.plans import content_id
from syntide.generate.qa.templates import TEMPLATES

PLAN_SCHEMA = "syntide.qa-generation-plan@1"
GAP_SCHEMA = "syntide.qa-gap-report@1"
SLICES = (
    "family",
    "answer",
    "rate_band",
    "lead_availability",
    "noise",
    "origin",
    "source",
    "composition",
    "answerability",
)


class AllocationError(ValueError):
    """The predictions are not from a development suite, or do not match it."""


def task_slices(
    task: QATask, evidence: Mapping[str, Evidence]
) -> dict[str, str]:
    """Group a task by measured attributes of its first recording."""
    sha = task.inputs[0].waveform_sha256
    own = [e for e in evidence.values() if e.waveform_sha256 == sha]
    return _task_slices(task, own)


def _task_slices(task: QATask, own: Sequence[Evidence]) -> dict[str, str]:
    """Apply grouping rules to evidence from the first recording only."""
    rate = next((e for e in own if e.attribute == "heart_rate_bpm"), None)
    flat = next((e for e in own if e.attribute == "flat_leads"), None)
    noise = [e.value for e in own if e.attribute == "hf_rms_mv" and e.usable]
    if task.answerability != "answerable":
        answer = "insufficient"
    elif task.answer_type == "numeric":
        answer = "numeric"
    else:
        answer = task.answer_text if len(task.answer_text) <= 40 else "long"
    return {
        "family": task.family,
        "answer": f"{task.family}={answer}",
        "rate_band": "unknown"
        if rate is None or not rate.usable
        else "<60"
        if rate.value < 60
        else "60-100"
        if rate.value <= 100
        else ">100",
        "lead_availability": "unknown"
        if flat is None or not flat.usable
        else "all"
        if not flat.value
        else "partial",
        "noise": "unknown"
        if not noise
        else "<0.02mV"
        if max(noise) < 0.02
        else "0.02-0.05mV"
        if max(noise) < 0.05
        else ">=0.05mV",
        "origin": task.inputs[0].origin,
        "source": task.inputs[0].dataset,
        "composition": "single"
        if task.composition in ("attribute", "source")
        else task.composition,
        "answerability": task.answerability,
    }


def coverage(
    tasks: Iterable[QATask], evidence: Mapping[str, Evidence]
) -> dict[str, dict[str, int]]:
    # Preserve evidence insertion order: rate/flat select the first matching
    # fact,
    # even when a later fact for the same attribute is usable and the first is
    # not.
    by_waveform: dict[str, list[Evidence]] = defaultdict(list)
    for fact in evidence.values():
        by_waveform[fact.waveform_sha256].append(fact)
    counts: dict[str, Counter] = defaultdict(Counter)
    for task in tasks:
        own = by_waveform.get(task.inputs[0].waveform_sha256, ())
        for key, value in _task_slices(task, own).items():
            counts[key][value] += 1
    return {key: dict(sorted(counts[key].items())) for key in SLICES}


def gap_report(
    suite_info: Mapping[str, Any],
    tasks: Sequence[QATask],
    evidence: Mapping[str, Evidence],
    correct: Mapping[str, bool],
    *,
    min_n: int = 20,
    prior_strength: float = 10.0,
    protected_subjects: Iterable[str] = (),
) -> dict[str, Any]:
    """Estimate group errors using development predictions only."""
    identity = suite_info["identity"]
    if (
        identity.get("purpose") != "evaluation"
        or identity["config"].get("suite_role") != "development"
    ):
        raise AllocationError(
            "gap reports read predictions on a development suite "
            "only, never selection or test"
        )
    subjects = {
        s
        for task in tasks
        for ref in task.inputs
        for s in ref.root_subject_uids
    }
    shared = subjects & set(protected_subjects)
    if shared:
        raise AllocationError(
            f"{len(shared)} development patients are also in the "
            f"selection or test suites"
        )
    ids = {task.task_id for task in tasks}
    if set(correct) != ids:
        raise AllocationError(
            "predictions must cover exactly the development suite's tasks"
        )
    overall_errors = sum(not v for v in correct.values())
    overall = (overall_errors + 0.5) / (len(correct) + 1)
    cells: dict[tuple[str, str], list[QATask]] = defaultdict(list)
    for task in tasks:
        for key, value in task_slices(task, evidence).items():
            cells[(key, value)].append(task)
    slices = []
    for (key, value), members in sorted(cells.items()):
        n = len(members)
        errors = sum(not correct[t.task_id] for t in members)
        reliable = n >= min_n
        low, high = beta.ppf([0.05, 0.95], errors + 0.5, n - errors + 0.5)
        slices.append(
            {
                "slice": key,
                "value": value,
                "n": n,
                "patients": len(
                    {
                        s
                        for t in members
                        for ref in t.inputs
                        for s in ref.root_subject_uids
                    }
                ),
                "errors": errors,
                "error_rate": errors / n,
                "interval90": [float(low), float(high)],
                "shrunk_error": (errors + prior_strength * overall)
                / (n + prior_strength)
                if reliable
                else overall,
                "reliable": reliable,
            }
        )
    body = {
        "schema_version": GAP_SCHEMA,
        "suite_corpus_id": suite_info["corpus_id"],
        "predictions_sha256": content_id(
            sorted([k, bool(v)] for k, v in correct.items())
        ),
        "rules": {
            "min_n": min_n,
            "prior_strength": prior_strength,
            "slices": list(SLICES),
        },
        "overall": {
            "n": len(correct),
            "errors": overall_errors,
            "error_rate": overall_errors / max(len(correct), 1),
        },
        "slices": slices,
    }
    return {**body, "gap_report_id": "qagap-" + content_id(body)[:16]}


def make_plan(
    gap: Mapping[str, Any],
    budget: int,
    *,
    parent_corpus_id: str | None,
    seed: int,
    families: Sequence[str] | None = None,
    max_share: float = 0.3,
) -> dict[str, Any]:
    """Allocate tasks using development errors and sample counts."""
    body = {k: v for k, v in gap.items() if k != "gap_report_id"}
    if content_id(body)[:16] != gap["gap_report_id"].removeprefix("qagap-"):
        raise AllocationError(
            "the gap report does not match its id; it was edited"
        )
    by_family = {
        row["value"]: row for row in gap["slices"] if row["slice"] == "family"
    }
    generated = {t.family: [] for t in TEMPLATES.values() if t.n_inputs == 1}
    for template in TEMPLATES.values():
        if template.family in generated:
            generated[template.family].append(template.id)
    chosen = [f for f in (families or sorted(generated)) if f in generated]
    if not chosen:
        raise AllocationError("no generatable family to allocate to")
    overall = gap["overall"]["error_rate"]
    weights = {
        f: (by_family[f]["shrunk_error"] if f in by_family else overall)
        for f in chosen
    }
    total = sum(weights.values()) or 1.0
    shares = {f: min(max_share, w / total) for f, w in weights.items()}
    norm = sum(shares.values()) or 1.0
    family_quota = _largest_remainder(
        {f: budget * s / norm for f, s in shares.items()}, budget
    )
    template_quotas: dict[str, int] = {}
    for family, quota in family_quota.items():
        ids = sorted(generated[family])
        for template_id, n in _largest_remainder(
            {t: quota / len(ids) for t in ids}, quota
        ).items():
            if n:
                template_quotas[template_id] = n
    plan = {
        "schema_version": PLAN_SCHEMA,
        "gap_report_id": gap["gap_report_id"],
        "parent_corpus_id": parent_corpus_id,
        "seed": seed,
        "budget": budget,
        "max_share": max_share,
        "family_weights": {f: round(w, 6) for f, w in sorted(weights.items())},
        "family_quotas": dict(sorted(family_quota.items())),
        "template_quotas": dict(sorted(template_quotas.items())),
        "status": "intent_only_builds_a_new_corpus",
    }
    return {**plan, "plan_id": "qaplan-" + content_id(plan)[:16]}


def verify_plan(plan: Mapping[str, Any]) -> None:
    body = {k: v for k, v in plan.items() if k != "plan_id"}
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("plan_id") != "qaplan-" + content_id(body)[:16]
    ):
        raise AllocationError(
            "the QA generation plan does not match its id; export a "
            "new plan instead of editing it"
        )


def _largest_remainder(
    targets: Mapping[str, float], total: int
) -> dict[str, int]:
    floors = {k: math.floor(v) for k, v in targets.items()}
    left = total - sum(floors.values())
    order = sorted(targets, key=lambda k: (-(targets[k] - floors[k]), k))
    for k in order[: max(left, 0)]:
        floors[k] += 1
    return floors
