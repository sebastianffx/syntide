"""The paired training stream: every batch a trial trains on, and the exposure it counts.

The stream, not the trainer, owns the budget. It yields exactly ``max_steps`` batches of
``batch_size`` and then stops, whatever the trainer does.

Pairing (training doc §4A). Randomness comes from separate streams (INIT, PARENTS, SLOTS,
TRANSFORM) keyed by (replicate_seed, pass or step, slot, attempt) and **never by the arm**. So
within a (regime, seed) block, A0-A3 see the same parents in the same order, the same transform
slots and the same model initialization, and differ only in what happens inside a transform slot:

- ``none``: the parent goes in unchanged (A0; C1 and C2 have no transform slots at all);
- ``generic``: A1's per-lead augmentation (``syntide.generate.generic``);
- ``editor``: the electrode-space artifact editor. A2 and A3 draw the same uniforms ``u`` for a
  (step, slot, attempt) and differ only in the policy that maps them to controls.

An editor candidate that fails a blocking validator is retried with the same parent and slot and the
next draw (``attempt`` + 1), up to ``max_retries`` times. A slot whose every attempt fails keeps its
parent unchanged and is counted in ``exhausted_slots``: never silently replaced.

Every candidate gets a ledger row, rejected ones included, and at each ``val_every`` boundary the
stream logs its ``budget/*`` counters (and ``generator/acceptance_rate`` when it generates) to the
tracker, so every trainer's dashboard shows them.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from syntide.data.schema import FS_HZ, Samples
from syntide.generate import editor, generic, policy, validate
from syntide.generate.ledger import LedgerWriter, Recipe, ledger_row

if TYPE_CHECKING:
    from syntide.eval.tracking import Tracker
    from syntide.trial import TrialSpec

STREAMS = {"init": 0, "parents": 1, "slots": 2, "transform": 3, "gate": 4}
FAMILIES = ("none", "generic", "editor")


def rng(seed: int, stream: str, *keys: int) -> np.random.Generator:
    """One independent random stream per purpose and key. Keys never include the arm id."""
    return np.random.default_rng([STREAMS[stream], seed, *keys])


def init_seed(replicate_seed: int) -> int:
    """The model-initialization seed shared by every arm of a (regime, replicate_seed) block."""
    return int(rng(replicate_seed, "init").integers(2**63))


@dataclass(frozen=True)
class Batch:
    step: int  # the optimizer update this batch is for, 1..max_steps
    x: np.ndarray  # (B, 12, T) float32, millivolts
    y: np.ndarray  # (B, 5) float32 multi-hot diagnostic superclasses
    transformed: np.ndarray  # (B,) bool: the slot holds a transformed parent
    record_id: np.ndarray  # (B,) str: the parent record of each slot


class TrainStream:
    """Single-use iterator over one trial's batches. Editor arms need the regime's calibration."""

    def __init__(
        self,
        spec: "TrialSpec",
        parents: Samples,
        tracker: "Tracker | None" = None,
        ledger: LedgerWriter | None = None,
        calibration: dict[str, Any] | None = None,
    ) -> None:
        if spec.transform_family not in FAMILIES:
            raise ValueError(f"unknown transform family {spec.transform_family!r}")
        if len(parents) == 0:
            raise ValueError("a training stream needs at least one parent record")
        if spec.transform_family == "editor":
            if calibration is None:
                raise policy.CalibrationError(f"arm {spec.arm} edits ECGs and needs its regime's calibration")
            if calibration["envelope"] != spec.transform["envelope"]:
                raise policy.CalibrationError("the calibration was fit for another envelope; recalibrate")
        self.spec = spec
        self.parents = parents
        self.tracker = tracker
        self.ledger = ledger
        self.calibration = calibration
        self.n_slots = round(spec.transform_fraction * spec.batch_size)
        self._generic = (
            generic.GenericAugConfig(**spec.transform) if spec.transform_family == "generic" else None
        )
        self._perms: dict[int, np.ndarray] = {}
        self._started = False
        self.steps_served = 0
        self.presentations = 0
        self.transform_slots = 0
        self.candidates = 0
        self.accepted = 0
        self.exhausted_slots = 0
        self._recipes: set[str] = set()

    @property
    def consumed(self) -> bool:
        return self.steps_served == self.spec.max_steps

    @property
    def exposure(self) -> dict[str, int]:
        return {
            "steps": self.steps_served,
            "presentations": self.presentations,
            "transform_slots": self.transform_slots,
            "candidates": self.candidates,
            "accepted": self.accepted,
            "rejected": self.candidates - self.accepted,
            "exhausted_slots": self.exhausted_slots,
            "distinct_recipes": len(self._recipes),
        }

    def __iter__(self) -> Iterator[Batch]:
        if self._started:
            raise RuntimeError("a training stream can be iterated only once")
        self._started = True
        return self._batches()

    def _batches(self) -> Iterator[Batch]:
        spec = self.spec
        transform = {"generic": self._augment, "editor": self._edit}.get(spec.transform_family)
        started = time.monotonic()
        for step in range(1, spec.max_steps + 1):
            index = self.parent_positions(step)
            mask = self.slot_mask(step)
            x = self.parents.x[index]  # fancy indexing copies, so edits never touch the parents
            transformed = np.zeros(spec.batch_size, dtype=bool)
            if transform is not None:
                for slot in np.flatnonzero(mask):
                    edited = transform(x[slot], int(index[slot]), step, int(slot))
                    if edited is not None:
                        x[slot] = edited
                        transformed[slot] = True
            self.presentations += spec.batch_size
            self.transform_slots += int(mask.sum())
            self.steps_served = step
            if self.tracker is not None and step % spec.val_every == 0:
                self.tracker.log_scalars(self._budget(time.monotonic() - started), step)
            yield Batch(
                step=step,
                x=x,
                y=self.parents.labels[index].astype(np.float32),
                transformed=transformed,
                record_id=self.parents.record_id[index],
            )

    def parent_positions(self, step: int) -> np.ndarray:
        """Parent indices for ``step``: consecutive slices of one seeded permutation per pass."""
        n, size = len(self.parents), self.spec.batch_size
        passes, offsets = np.divmod(np.arange((step - 1) * size, step * size), n)
        index = np.empty(size, dtype=np.int64)
        for p in np.unique(passes):
            if p not in self._perms:
                self._perms = {q: perm for q, perm in self._perms.items() if q >= p - 1}
                self._perms[p] = rng(self.spec.replicate_seed, "parents", int(p)).permutation(n)
            index[passes == p] = self._perms[p][offsets[passes == p]]
        return index

    def slot_mask(self, step: int) -> np.ndarray:
        """Which batch positions are transform slots at ``step`` (exactly ``n_slots`` of them)."""
        mask = np.zeros(self.spec.batch_size, dtype=bool)
        if self.n_slots:
            chosen = rng(self.spec.replicate_seed, "slots", step).choice(
                self.spec.batch_size, self.n_slots, replace=False
            )
            mask[chosen] = True
        return mask

    def _rng_stream(self, step: int, slot: int, attempt: int) -> str:
        return f"{self.spec.arm}/s{self.spec.replicate_seed}/step{step}/slot{slot}/try{attempt}"

    def _augment(self, x: np.ndarray, parent: int, step: int, slot: int) -> np.ndarray:
        draw = rng(self.spec.replicate_seed, "transform", step, slot, 0)
        controls = generic.draw_controls(draw, self._generic, x.shape[0], FS_HZ)
        seed_key = int(draw.integers(2**63))
        recipe = Recipe(
            str(self.parents.record_id[parent]), generic.GENERATOR, generic.VERSION, controls, seed_key, None
        )
        self.candidates += 1
        self.accepted += 1
        self._recipes.add(recipe.recipe_id)
        if self.ledger is not None:
            self.ledger.append(
                ledger_row(
                    recipe,
                    step=step,
                    slot=slot,
                    attempt=0,
                    rng_stream=self._rng_stream(step, slot, 0),
                    parent_subject_uid=str(self.parents.subject_uid[parent]),
                    label_effect=generic.LABEL_EFFECT,
                )
            )
        return generic.apply(x, controls, seed_key, FS_HZ)

    def _edit(self, x: np.ndarray, parent: int, step: int, slot: int) -> np.ndarray | None:
        cfg, calibration = self.spec.transform, self.calibration
        parent_id = str(self.parents.record_id[parent])
        for attempt in range(cfg["max_retries"] + 1):
            draw = rng(self.spec.replicate_seed, "transform", step, slot, attempt)
            u, seed_key = draw.random(len(policy.U_FIELDS)), int(draw.integers(2**63))
            controls = policy.controls_for(self.spec.sampling_policy, u, calibration, x.shape[-1], FS_HZ)
            recipe = Recipe(
                parent_id, editor.GENERATOR, editor.VERSION, controls, seed_key, calibration["gap_report_id"]
            )
            candidate = editor.edit(x, recipe, FS_HZ)
            checks, realized = validate.evaluate(candidate, x, recipe, cfg, FS_HZ)
            ok = validate.accepted(checks)
            self.candidates += 1
            if self.ledger is not None:
                self.ledger.append(
                    ledger_row(
                        recipe,
                        step=step,
                        slot=slot,
                        attempt=attempt,
                        rng_stream=self._rng_stream(step, slot, attempt),
                        parent_subject_uid=str(self.parents.subject_uid[parent]),
                        label_effect=editor.LABEL_EFFECT,
                        controls_realized=realized,
                        validations=[check.to_json() for check in checks],
                        decision="accepted" if ok else "rejected",
                        reason=validate.failed(checks),
                    )
                )
            if ok:
                self.accepted += 1
                self._recipes.add(recipe.recipe_id)
                return candidate.x
        self.exhausted_slots += 1
        return None

    def _budget(self, elapsed_s: float) -> dict[str, float]:
        values = {
            "budget/sample_presentations": self.presentations,
            "budget/transform_slots": self.transform_slots,
            "budget/distinct_recipes": len(self._recipes),
            "budget/wall_time_seconds": elapsed_s,
        }
        if self.candidates:
            values["generator/acceptance_rate"] = self.accepted / self.candidates
        if self.spec.transform_family == "editor":
            values["generator/exhausted_slots"] = self.exhausted_slots
        return values
