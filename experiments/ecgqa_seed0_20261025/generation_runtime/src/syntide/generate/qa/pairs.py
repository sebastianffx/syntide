"""Controlled variations and paired comparisons over two recordings.

A ``ControlledVariation`` declares the attributes it is meant to change (and in which direction) and
the attributes it must leave alone. Both outputs are remeasured; the pair is accepted only when every
declared change is observed in its direction by at least its minimum effect, and every invariant holds
within its tolerance. A shared seed, or a generator's conditioning, establishes neither.

Supported (narrow, tested) variations:

- ``amplitude_up@1``: a calibrated gain of ``factor`` (``ecg.acquisition@1`` gain). Changes amplitude;
  keeps rate, beat count, rhythm regularity, duration and lead availability.
- ``wander_up@1``: one electrode-space baseline drift (``ecg.electrode_artifact@1.0.0``, unchanged).
  Changes the largest baseline wander; keeps rate, beat count, duration and lead availability.
- ``diffusets.heart_rate_condition@1``: two generated samples with the same seed whose conditioning
  differs only in ``heart_rate``. Changes the measured heart rate; keeps duration and lead availability.

Not supported, and refused explicitly: changing the heart rate of a recorded ECG by time-warping (it
changes every interval, not only the rate) and any diagnosis change (no validated physiological editor).

A pair is a controlled **signal** comparison (``claim_scope``), not a clinical treatment effect. Its
comparison task has explicit inputs in a seeded order (A, B), and its answer comes from the remeasured
evidence, never from the intended change. Natural comparisons pair two different real recordings of the
same work unit; partners come from a seeded permutation, never a Cartesian product.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from syntide.data.qa import Evidence, InputRef, QATask
from syntide.explore.plans import content_id
from syntide.generate.qa.templates import TEMPLATES, View, build_task, candidates, scalar
from syntide.generate.qa.verify import student_context, verify_task

CLAIM_SCOPE = "controlled_signal_comparison"
INVARIANT_TOLERANCE = {"heart_rate_bpm": 3.0, "beat_count": 1.0, "duration_s": 1e-6}
EXACT = ("rr_regularity", "flat_leads")
UNSUPPORTED = {
    "heart_rate": "time-warping a recorded ECG changes every interval, not only the heart rate",
    "diagnosis": "there is no validated physiological editor for diagnostic findings",
}


class UnsupportedVariation(ValueError):
    """A physiological change this version cannot make or verify."""


@dataclass(frozen=True)
class ControlledVariation:
    name: str
    changes: dict[str, tuple[str, float]]  # attribute -> (increase|decrease, minimum effect)
    invariants: tuple[str, ...]
    recipe: dict[str, Any] = field(default_factory=dict)  # the acquisition recipe that makes the variant


VARIATIONS = {
    "amplitude_up@1": ControlledVariation(
        "amplitude_up@1",
        {"amplitude_mv": ("increase", 0.2)},
        ("heart_rate_bpm", "beat_count", "rr_regularity", "duration_s", "flat_leads"),
        {"op": "gain", "params": {"factor": 1.5}},
    ),
    "wander_up@1": ControlledVariation(
        "wander_up@1",
        {"max_wander_mv": ("increase", 0.3)},
        ("heart_rate_bpm", "beat_count", "duration_s", "flat_leads"),
        {
            "op": "electrode_artifact",
            "params": {
                "controls": {
                    "artifact_type": "baseline_drift",
                    "electrode": "LA",
                    "amplitude_mv": 1.5,
                    "freq_hz": 0.2,
                }
            },
        },
    ),
    "diffusets.heart_rate_condition@1": ControlledVariation(
        "diffusets.heart_rate_condition@1",
        {"heart_rate_bpm": ("from_conditioning", 5.0)},
        ("duration_s", "flat_leads"),
    ),
}


def variation(name: str) -> ControlledVariation:
    if name in UNSUPPORTED:
        raise UnsupportedVariation(f"{name}: unsupported, {UNSUPPORTED[name]}")
    if name not in VARIATIONS:
        raise UnsupportedVariation(f"unknown variation {name!r}; supported: {sorted(VARIATIONS)}")
    return VARIATIONS[name]


@dataclass(frozen=True)
class PairRecord:
    variation: str
    reference: InputRef
    variant: InputRef
    intended: dict[str, Any]
    observed: dict[str, list[Any]]
    invariant_checks: list[dict[str, Any]]
    status: str
    reason: str

    @property
    def pair_id(self) -> str:
        return (
            "pair-"
            + content_id([self.variation, self.reference.waveform_sha256, self.variant.waveform_sha256])[:20]
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "variation": self.variation,
            "reference": self.reference.waveform_id,
            "variant": self.variant.waveform_id,
            "intended": self.intended,
            "observed": self.observed,
            "invariant_checks": self.invariant_checks,
            "status": self.status,
            "reason": self.reason,
            "claim_scope": CLAIM_SCOPE,
        }


def _value(view: View, attribute: str) -> Evidence | None:
    return scalar(view, attribute)


def check_pair(
    spec: ControlledVariation, reference: View, variant: View, intended: dict[str, Any] | None = None
) -> PairRecord:
    """Remeasured changes and invariants of a variant against its reference."""
    intended = dict(intended or {name: direction for name, (direction, _) in spec.changes.items()})
    observed, problems, checks = {}, [], []
    for attribute, (direction, minimum) in spec.changes.items():
        a, b = _value(reference, attribute), _value(variant, attribute)
        if a is None or b is None or not (a.usable and b.usable):
            problems.append(f"change_unmeasured:{attribute}")
            continue
        observed[attribute] = [a.value, b.value]
        wanted = intended.get(attribute, direction)
        delta = b.value - a.value
        if (wanted == "increase" and delta < minimum) or (wanted == "decrease" and -delta < minimum):
            problems.append(f"change_not_observed:{attribute}")
    for attribute in spec.invariants:
        a, b = _value(reference, attribute), _value(variant, attribute)
        if a is None or b is None or not (a.usable and b.usable):
            checks.append({"attribute": attribute, "reference": None, "variant": None, "pass": False})
            problems.append(f"invariant_unmeasured:{attribute}")
            continue
        same = (
            a.value == b.value
            if attribute in EXACT
            else abs(b.value - a.value) <= INVARIANT_TOLERANCE.get(attribute, 0.0)
        )
        checks.append({"attribute": attribute, "reference": a.value, "variant": b.value, "pass": bool(same)})
        if not same:
            problems.append(f"invariant_violated:{attribute}")
    return PairRecord(
        spec.name,
        reference.ref,
        variant.ref,
        intended,
        observed,
        checks,
        "rejected" if problems else "accepted",
        ",".join(problems),
    )


def comparison_tasks(
    first: View,
    second: View,
    contexts: Sequence[str],
    rng: np.random.Generator,
    *,
    pair: PairRecord | None = None,
    attributes: Sequence[str] | None = None,
) -> tuple[list[QATask], list[QATask], list[Evidence]]:
    """Comparison tasks over two recordings, in a seeded order. Returns (accepted, rejected, evidence)."""
    flip = bool(rng.integers(2))
    views = [second, first] if flip else [first, second]
    ctx = list(reversed(contexts)) if flip else list(contexts)
    template = TEMPLATES["compare.attribute@1"]
    accepted, rejected, cited_all = [], [], []
    store = {e.evidence_id: e for v in views for e in v.evidence}
    for params in candidates(template, views):
        if attributes is not None and params["attribute"] not in attributes:
            continue
        lineage = (
            {"pair_id": pair.pair_id, "pair": pair.to_json()}
            if pair
            else {"pair_id": None, "natural_pair": True}
        )
        generation = {
            "generator": "syntide.pairs@1",
            "operator": "comparison",
            "order": "flipped" if flip else "as_given",
        }
        context = {
            "clinical_context": [student_context(v.ref.origin, c) for v, c in zip(views, ctx, strict=True)]
        }
        built = build_task(template, params, views, context, generation, lineage=lineage)
        if built is None:
            continue
        task, cited = built
        cited_all.extend(cited)
        checked = verify_task(task, {**store, **{e.evidence_id: e for e in cited}})
        (accepted if checked.verification["status"] == "accepted" else rejected).append(checked)
    return accepted, rejected, cited_all


def natural_partners(
    record_ids: Sequence[str], max_pairs: int, rng: np.random.Generator
) -> list[tuple[str, str]]:
    """Disjoint pairs of different recordings from a seeded permutation (at most ``max_pairs``)."""
    order = [record_ids[i] for i in rng.permutation(len(record_ids))]
    return [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)][:max_pairs]
