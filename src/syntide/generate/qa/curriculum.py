"""Frozen QA training curricula: which task every batch slot holds, for every step, before training starts.

A curriculum is a list of stages, each ending at an optimizer step (``until_step``; the last one is the
trial's ``max_steps``). A stage selects tasks by composition level and optionally by family and corpus
component, sets its target mode, and replays a share (``replay``) of earlier stages' tasks. Within a
pool, tasks are served in passes: each pass is one seeded permutation, so every task of the pool is
seen once per pass (as the RunPod baseline's loader did) before any repeats.

All randomness is keyed by (curriculum seed, purpose, step or pass), never by the arm, so two arms with
the same pools see the same sequence. ``schedule`` is a pure function of the frozen curriculum and the
corpora's task ids: the same inputs always give the same sequence (``sequence_sha256``).

The target mode is the arm's supervision (``answer_only`` or ``rationale``), except that a stage marked
``target_mode: answer_only`` stays answer-only in a rationale arm (the basic attribute stage).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from syntide.data.qa import QATask
from syntide.explore.plans import content_id

STREAM = 21
ALL = ("source", "attribute", "composed", "reasoning", "situation", "comparison")


@dataclass
class StageConfig:
    name: str = "all"
    until_step: int = 0  # 0 = the trial's max_steps (last stage only)
    compositions: list[str] = field(default_factory=lambda: list(ALL))
    families: list[str] | None = None
    components: list[str] | None = None  # corpus labels (e.g. real_qa, grounded); None = every corpus
    target_mode: str | None = None  # None = the arm's supervision
    replay: float = 0.0  # share of each batch drawn from earlier stages' pools


@dataclass
class CurriculumConfig:
    name: str = "uniform-passes@1"
    seed: int = 0
    stages: list[StageConfig] = field(default_factory=lambda: [StageConfig()])


class CurriculumError(ValueError):
    """The curriculum's stages do not cover the budget, or a stage selects no task."""


@dataclass(frozen=True)
class Slot:
    step: int
    position: int
    stage: str
    pool: str  # stage name, or replay:<stage names>
    task_id: str
    corpus: str
    target_mode: str


class Curriculum:
    """The frozen schedule over labelled corpora (``{label: [QATask, ...]}``)."""

    def __init__(
        self,
        cfg: CurriculumConfig,
        corpora: Mapping[str, Sequence[QATask]],
        *,
        batch_size: int,
        max_steps: int,
        supervision: str,
    ) -> None:
        if supervision not in ("answer_only", "rationale"):
            raise CurriculumError("supervision must be answer_only or rationale")
        self.cfg, self.batch_size, self.max_steps, self.supervision = cfg, batch_size, max_steps, supervision
        self._corpora = {label: list(tasks) for label, tasks in corpora.items()}
        bounds = [stage.until_step or max_steps for stage in cfg.stages]
        if bounds != sorted(bounds) or bounds[-1] != max_steps or bounds[0] <= 0:
            raise CurriculumError(
                f"stage ends {bounds} must increase and the last must be max_steps={max_steps}"
            )
        self.bounds = bounds
        self.pools: list[list[tuple[str, str]]] = []  # per stage: (corpus label, task id), sorted
        for stage in cfg.stages:
            if not 0 <= stage.replay < 1:
                raise CurriculumError(f"stage {stage.name}: replay must be in [0, 1)")
            members = sorted(
                (label, task.task_id)
                for label, tasks in corpora.items()
                if stage.components is None or label in stage.components
                for task in tasks
                if task.composition in stage.compositions
                and (stage.families is None or task.family in stage.families)
            )
            if not members:
                raise CurriculumError(f"stage {stage.name} selects no task from the corpora")
            self.pools.append(members)
        self.replay_pools = [
            sorted({m for pool in self.pools[:i] for m in pool}) for i in range(len(self.pools))
        ]
        self._cursor: dict[str, int] = defaultdict(int)
        self._perms: dict[tuple[str, int], np.ndarray] = {}

    @property
    def sha256(self) -> str:
        """Identity of the frozen curriculum and the exact pools it draws from."""
        return content_id(
            {
                "config": asdict(self.cfg),
                "batch_size": self.batch_size,
                "max_steps": self.max_steps,
                "supervision": self.supervision,
                "pools": [[list(m) for m in pool] for pool in self.pools],
            }
        )

    def stage_at(self, step: int) -> int:
        return next(i for i, end in enumerate(self.bounds) if step <= end)

    def mode(self, stage: int) -> str:
        own = self.cfg.stages[stage].target_mode
        return "answer_only" if self.supervision == "answer_only" or own == "answer_only" else "rationale"

    def _next(self, name: str, pool: Sequence[tuple[str, str]]) -> tuple[str, str]:
        position = self._cursor[name]
        epoch, offset = divmod(position, len(pool))
        key = (name, epoch)
        if key not in self._perms:
            self._perms = {k: v for k, v in self._perms.items() if k[0] != name or k[1] >= epoch - 1}
            self._perms[key] = np.random.default_rng(
                [STREAM, self.cfg.seed, int(content_id(name)[:12], 16), epoch]
            ).permutation(len(pool))
        self._cursor[name] = position + 1
        return pool[int(self._perms[key][offset])]

    def schedule(self) -> Iterator[list[Slot]]:
        """One list of ``batch_size`` slots per step, steps 1..max_steps. Iterate it once per trial."""
        for step in range(1, self.max_steps + 1):
            stage = self.stage_at(step)
            spec = self.cfg.stages[stage]
            rng = np.random.default_rng([STREAM, self.cfg.seed, step])
            earlier = self.replay_pools[stage]
            replay_name = "replay:" + "+".join(s.name for s in self.cfg.stages[:stage])
            slots = []
            for position in range(self.batch_size):
                replay = stage > 0 and earlier and rng.random() < spec.replay
                name, pool = (replay_name, earlier) if replay else (spec.name, self.pools[stage])
                label, task_id = self._next(name, pool)
                slots.append(Slot(step, position, spec.name, name, task_id, label, self.mode(stage)))
            yield slots


def sequence_sha256(curriculum: Curriculum) -> str:
    """Hash of the whole task sequence, from a fresh copy of the schedule (the original is not consumed)."""
    fresh = Curriculum(
        curriculum.cfg,
        curriculum._corpora,
        batch_size=curriculum.batch_size,
        max_steps=curriculum.max_steps,
        supervision=curriculum.supervision,
    )
    return content_id([[[slot.task_id, slot.target_mode] for slot in slots] for slots in fresh.schedule()])


def exposure(slots: Sequence[Slot], tasks: Mapping[str, QATask]) -> dict[str, Any]:
    """Examples, distinct tasks and whitespace tokens, overall and per stage, corpus and family."""
    by: dict[str, Counter] = {
        "stage": Counter(),
        "corpus": Counter(),
        "family": Counter(),
        "target_mode": Counter(),
    }
    tokens = Counter()
    seen = set()
    for slot in slots:
        task = tasks[slot.task_id]
        seen.add(slot.task_id)
        by["stage"][slot.stage] += 1
        by["corpus"][slot.corpus] += 1
        by["family"][task.family] += 1
        by["target_mode"][slot.target_mode] += 1
        tokens["prompt"] += len(task.question.split()) + sum(
            len(c.split()) for c in task.context.get("clinical_context", [])
        )
        tokens["target"] += len(task.target(slot.target_mode).split())
    return {
        "examples": len(slots),
        "distinct_tasks": len(seen),
        "distinct_semantic_tasks": len({tasks[t].semantic_key for t in seen}),
        "whitespace_tokens": dict(tokens),
        **{f"by_{k}": dict(sorted(v.items())) for k, v in by.items()},
    }
