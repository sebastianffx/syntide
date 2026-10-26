"""Create question variants and explanations from verified waveform evidence.

Seed templates describe individual attributes. Additional templates combine
facts, ask for reasoning, or change the question's stated situation. Optional
language-model responses may reword questions and explanations, while the
original evidence continues to determine the answers. Random choices depend on
the waveform and build seed rather than processing order.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from syntide.data.qa import Evidence, QATask
from syntide.generate.qa.teacher import (
    QUESTION_PROMPT,
    RATIONALE_PROMPT,
    Teacher,
    question_prompt,
    rationale_prompt,
)
from syntide.generate.qa.templates import (
    TEMPLATES,
    Template,
    View,
    build_task,
    candidates,
)
from syntide.generate.qa.verify import verify_task

GENERATOR = "syntide.tsevol@1"
OPERATORS = {
    "breadth": "attribute",
    "depth": "composed",
    "reasoning": "reasoning",
    "situation": "situation",
}
STREAMS = {
    "seed": 11,
    "evolve": 12,
    "paraphrase": 13,
    "pair": 14,
    "acquisition": 15,
}
_TOKEN = re.compile(r"\b(?:[A-Z]{1,3}\d?|V\d|\d+(?:\.\d+)?)\b")


@dataclass
class EvolConfig:
    seed_templates: list[str] = field(
        default_factory=lambda: [
            "acquisition.duration@1",
            "acquisition.flat_leads@1",
            "artifact.wander_in_lead@1",
            "artifact.max_wander_lead@1",
            "rhythm.heart_rate@1",
            "rhythm.beat_count@1",
            "rhythm.regularity@1",
            "rhythm.rate_band@1",
            "quality.noise_annotation@1",
            "diagnosis.superclass@1",
            "interval.measurement@1",
        ]
    )
    # artifact.max_noise_lead@1 is left out by default: on real 100 Hz PTB-XL
    # its margin rule is almost never
    # met, so it would teach a constant "insufficient evidence". It stays
    # available by name.
    max_seed_tasks: int = 8
    evolutions: int = 4
    operators: list[str] = field(default_factory=lambda: list(OPERATORS))
    max_per_family: int = 3
    paraphrase_fraction: float = (
        0.0  # share of accepted tasks that get one teacher paraphrase attempt
    )
    max_paraphrases: int = 1
    rationale_fraction: float = (
        0.0  # share of accepted tasks whose rationale the teacher rewords
    )
    insufficient_fraction: float = (
        0.25  # cap on the share of unanswerable tasks per waveform
    )


def stream(
    seed: int, purpose: str, sha: str, *keys: int
) -> np.random.Generator:
    """Seed randomness by build, purpose, and waveform."""
    return np.random.default_rng(
        [STREAMS[purpose], seed, int(sha[:15], 16), *keys]
    )


@dataclass
class Outcome:
    accepted: list[QATask] = field(default_factory=list)
    rejected: list[QATask] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)


def evolve_waveform(
    view: View,
    evidence: Sequence[Evidence],
    context: dict[str, Any],
    cfg: EvolConfig,
    seed: int,
    *,
    teacher: Teacher | None = None,
    lineage: dict[str, Any] | None = None,
) -> Outcome:
    """Create template tasks and optional rewordings for one recording."""
    out = Outcome(evidence=list(evidence))
    store = {e.evidence_id: e for e in evidence}
    sha = view.ref.waveform_sha256
    families: Counter = Counter()
    semantic: set[str] = set()
    insufficient = 0
    budget = cfg.max_seed_tasks + cfg.evolutions

    def consider(
        template: Template, params: dict[str, Any], generation: dict[str, Any]
    ) -> QATask | None:
        nonlocal insufficient
        if families[template.family] >= cfg.max_per_family:
            return None
        built = build_task(
            template, params, [view], context, generation, lineage=lineage
        )
        if built is None:
            return None
        task, cited = built
        if task.semantic_key in semantic:
            return None
        if (
            task.answerability != "answerable"
            and insufficient + 1 > cfg.insufficient_fraction * budget
        ):
            return None
        for e in cited:
            store.setdefault(e.evidence_id, e)
            out.evidence.append(e)
        task = verify_task(task, store)
        if task.verification["status"] != "accepted":
            out.rejected.append(task)
            return None
        families[template.family] += 1
        semantic.add(task.semantic_key)
        insufficient += task.answerability != "answerable"
        out.accepted.append(task)
        return task

    seeds = _choices(cfg.seed_templates, [view], stream(seed, "seed", sha))
    for template, params in seeds[: cfg.max_seed_tasks * 3]:
        if len(out.accepted) >= cfg.max_seed_tasks:
            break
        consider(template, params, {"generator": GENERATOR, "operator": "seed"})
    parents = list(out.accepted)
    rng = stream(seed, "evolve", sha)
    for step in range(cfg.evolutions):
        if not parents or not cfg.operators:
            break
        parent = parents[int(rng.integers(len(parents)))]
        operator = cfg.operators[int(rng.integers(len(cfg.operators)))]
        level = OPERATORS[operator]
        pool = [
            t.id
            for t in TEMPLATES.values()
            if t.composition == level and t.n_inputs == 1
        ]
        if (
            operator == "breadth"
        ):  # breadth stays within the configured seed templates
            pool = [
                t
                for t in pool
                if t in cfg.seed_templates
                and TEMPLATES[t].family != parent.family
            ]
            pool.append(parent.template)
        for template, params in _choices(pool, [view], rng):
            generation = {
                "generator": GENERATOR,
                "operator": operator,
                "parent_task": parent.task_id,
                "step": step,
            }
            if consider(template, params, generation) is not None:
                break
    if teacher is not None:
        reword(out, store, cfg, teacher, seed, sha)
    return out


def _choices(
    template_ids: Sequence[str], views: Sequence[View], rng: np.random.Generator
) -> list[tuple[Template, dict]]:
    """Yield applicable templates and parameters in a seeded order."""
    pairs = [
        (TEMPLATES[t], p)
        for t in dict.fromkeys(template_ids)
        if t in TEMPLATES
        for p in candidates(TEMPLATES[t], views)
    ]
    order = rng.permutation(len(pairs)) if pairs else []
    return [pairs[i] for i in order]


def reword(
    out: Outcome,
    store: dict[str, Evidence],
    cfg: EvolConfig,
    teacher: Teacher,
    seed: int,
    sha: str,
) -> None:
    """Reword questions and explanations while preserving their meaning."""
    rng = stream(seed, "paraphrase", sha)
    for index, task in enumerate(list(out.accepted)):
        if (
            task.answerability == "answerable"
            and rng.random() < cfg.rationale_fraction
        ):
            response = teacher.ask(
                RATIONALE_PROMPT,
                rationale_prompt(
                    task.question,
                    task.answer_text,
                    [c.text for c in task.claims],
                ),
            )
            candidate = replace(
                task,
                rationale=response.text,
                verification={"status": "pending"},
                generation={
                    **task.generation,
                    "rationale_teacher": response.ref,
                },
            )
            checked = (
                verify_task(candidate, store)
                if response.ok
                else replace(
                    candidate,
                    verification={
                        "status": "rejected",
                        "reason": "teacher_refused_or_empty",
                        "checks": [],
                    },
                )
            )
            if checked.verification["status"] == "accepted":
                out.accepted[index] = task = checked
            else:
                out.rejected.append(checked)
        made = sum(
            t.semantic_key == task.semantic_key and t.paraphrase != "seed"
            for t in out.accepted
        )
        if (
            made >= cfg.max_paraphrases
            or rng.random() >= cfg.paraphrase_fraction
        ):
            continue
        paraphrased = paraphrase(task, teacher)
        if paraphrased is None:
            continue
        checked = (
            verify_task(paraphrased, store)
            if paraphrased.verification.get("status") == "pending"
            else paraphrased
        )
        (
            out.accepted
            if checked.verification["status"] == "accepted"
            else out.rejected
        ).append(checked)


def paraphrase(task: QATask, teacher: Teacher) -> QATask | None:
    """Reword a question without changing its answer or evidence."""
    response = teacher.ask(
        QUESTION_PROMPT, question_prompt(task.question, list(task.options))
    )
    candidate = replace(
        task,
        question=response.text.strip(),
        paraphrase=response.sha256[:16],
        generation={
            **task.generation,
            "paraphrase_teacher": response.ref,
            "paraphrase_of": task.task_id,
        },
        verification={"status": "pending"},
    )
    problem = (
        paraphrase_problem(task, candidate.question)
        if response.ok
        else "teacher_refused_or_empty"
    )
    if problem:
        return replace(
            candidate,
            verification={
                "status": "rejected",
                "reason": problem,
                "checks": [],
            },
        )
    if (
        task.composition == "source"
    ):  # dataset answers are not evidence-backed; only the wording changes
        return replace(
            candidate,
            verification={
                **task.verification,
                "checks": [
                    *task.verification.get("checks", []),
                    {"name": "paraphrase_preserves@1", "pass": True},
                ],
            },
        )
    return candidate


def paraphrase_problem(task: QATask, text: str) -> str:
    """Explain an invalid rewording, or return an empty string."""
    if not text or len(text) > 400 or not text.endswith("?"):
        return "paraphrase_malformed"
    if " ".join(text.lower().split()) == " ".join(
        task.question.lower().split()
    ):
        return "paraphrase_identical"
    if "answer:" in text.lower():
        return "paraphrase_answers"
    missing = set(_TOKEN.findall(task.question)) - set(_TOKEN.findall(text))
    if missing:
        return "paraphrase_dropped_" + "_".join(sorted(missing))[:40]
    answer = task.answer_text.lower()
    if (
        task.answerability == "answerable"
        and len(answer) > 3
        and answer in text.lower()
        and answer not in task.question.lower()
    ):
        return "paraphrase_leaks_answer"
    return ""
