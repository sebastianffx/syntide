"""Create recording-condition variations and recheck their question answers.

Supported operations change sample rate, filter frequency content, crop time,
mask leads, change gain, or add electrode artifacts. Recipes record the
operation, settings, and random seed. Every transformed waveform is measured
again before its questions are accepted.

Each operation declares whether it preserves, determines, or invalidates an
attribute. A task that should retain its answer is rejected if remeasurement
changes that answer. Other supported tasks receive updated answers and
explanations, including explicit insufficient-evidence answers when needed.
Inapplicable templates are rejected.

Source annotations are inherited only when their meaning is preserved. A
preserved diagnosis does not imply every waveform measurement remains
observable. Original dataset questions are not carried onto edited waveforms
because their answers cannot be verified through remeasurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from syntide.data.qa import Evidence, InputRef, QATask, SignalInfo
from syntide.explore.plans import content_id
from syntide.generate import editor, validate
from syntide.generate.editor import EditorConfig
from syntide.generate.ledger import Recipe
from syntide.generate.qa.evidence import (
    ANNOTATIONS,
    EvidenceExtractor,
    waveform_sha256,
)
from syntide.generate.qa.templates import TEMPLATES, View, build_task
from syntide.generate.qa.verify import student_context, verify_task

GENERATOR = "ecg.acquisition"
VERSION = "1"
GROUPS = (
    "duration",
    "rate",
    "amplitude",
    "artifact",
    "lead_availability",
    "diagnosis",
    "signal_quality",
    "intervals",
)
# Which group each template's answer belongs to.
TEMPLATE_GROUP = {
    "acquisition.duration": "duration",
    "acquisition.flat_leads": "lead_availability",
    "artifact.wander_in_lead": "artifact",
    "artifact.max_wander_lead": "artifact",
    "artifact.max_noise_lead": "artifact",
    "rhythm.heart_rate": "rate",
    "rhythm.beat_count": "duration",
    "rhythm.regularity": "rate",
    "rhythm.rate_band": "rate",
    "quality.noise_annotation": "signal_quality",
    "diagnosis.superclass": "diagnosis",
    "interval.measurement": "intervals",
    "compose.rate_and_wander": "artifact",
    "reason.support_refute": "rate",
    "situation.lead_subset": "artifact",
    "situation.supplied_rule": "rate",
}
_P, _D, _I = "preserves", "determines", "invalidates"
LABEL_EFFECT = {
    "resample": {
        "duration": _P,
        "rate": _P,
        "amplitude": _D,
        "artifact": _D,
        "lead_availability": _P,
        "diagnosis": _P,
        "signal_quality": _I,
        "intervals": _I,
    },
    "lowpass": {
        "duration": _P,
        "rate": _P,
        "amplitude": _D,
        "artifact": _D,
        "lead_availability": _P,
        "diagnosis": _P,
        "signal_quality": _I,
        "intervals": _I,
    },
    "highpass": {
        "duration": _P,
        "rate": _P,
        "amplitude": _D,
        "artifact": _D,
        "lead_availability": _P,
        "diagnosis": _P,
        "signal_quality": _I,
        "intervals": _I,
    },
    "crop": {
        "duration": _D,
        "rate": _D,
        "amplitude": _D,
        "artifact": _D,
        "lead_availability": _P,
        "diagnosis": _I,
        "signal_quality": _I,
        "intervals": _I,
    },
    "lead_mask": {
        "duration": _P,
        "rate": _D,
        "amplitude": _D,
        "artifact": _D,
        "lead_availability": _D,
        "diagnosis": _P,
        "signal_quality": _I,
        "intervals": _I,
    },
    "gain": {
        "duration": _P,
        "rate": _P,
        "amplitude": _D,
        "artifact": _D,
        "lead_availability": _P,
        "diagnosis": _P,
        "signal_quality": _I,
        "intervals": _P,
    },
    "electrode_artifact": {
        "duration": _P,
        "rate": _D,
        "amplitude": _D,
        "artifact": _D,
        "lead_availability": _P,
        "diagnosis": editor.LABEL_EFFECT["diagnosis"],
        "signal_quality": _I,
        "intervals": _I,
    },
}
OPERATIONS = tuple(LABEL_EFFECT)


@dataclass
class AcquisitionConfig:
    """Signal transformations and ranges to sample for each source recording."""

    operations: list[str] = field(
        default_factory=lambda: [
            "resample",
            "lowpass",
            "crop",
            "lead_mask",
            "gain",
            "electrode_artifact",
        ]
    )
    per_waveform: int = 1
    resample_rates_hz: list[float] = field(default_factory=lambda: [50.0])
    lowpass_hz: list[float] = field(default_factory=lambda: [25.0, 35.0])
    highpass_hz: list[float] = field(default_factory=lambda: [0.67])
    crop_s: list[float] = field(default_factory=lambda: [5.0])
    lead_masks: list[list[str]] = field(
        default_factory=lambda: [
            ["V1", "V2", "V3", "V4", "V5", "V6"],
            ["I", "II", "III", "AVR", "AVL", "AVF"],
            ["V2"],
        ]
    )
    gains: list[float] = field(default_factory=lambda: [0.5, 2.0])
    mv_limit: float = 10.0
    editor: EditorConfig = field(default_factory=EditorConfig)


class UnsupportedTransformation(ValueError):
    """The requested operation is not one this version supports."""


@dataclass(frozen=True)
class Variant:
    ref: InputRef
    x: np.ndarray
    recipe: dict[str, Any]
    checks: list[dict[str, Any]]
    ok: bool


def draw_recipe(
    op: str, cfg: AcquisitionConfig, rng: np.random.Generator, spec: SignalInfo
) -> dict[str, Any]:
    if op not in LABEL_EFFECT:
        raise UnsupportedTransformation(
            f"{GENERATOR}@{VERSION} does not support {op!r}; choose "
            f"one of {OPERATIONS}"
        )
    pick = lambda values: values[int(rng.integers(len(values)))]  # noqa: E731
    seed_key = int(rng.integers(2**53))
    if op == "resample":
        params = {"rate_hz": float(pick(cfg.resample_rates_hz))}
    elif op == "lowpass":
        params = {"cutoff_hz": float(pick(cfg.lowpass_hz))}
    elif op == "highpass":
        params = {"cutoff_hz": float(pick(cfg.highpass_hz))}
    elif op == "crop":
        duration = float(pick(cfg.crop_s))
        if duration >= spec.duration_s:
            raise UnsupportedTransformation(
                f"crop of {duration} s needs a longer recording than "
                f"{spec.duration_s} s"
            )
        start = round(float(rng.uniform(0, spec.duration_s - duration)), 2)
        params = {"start_s": start, "duration_s": duration}
    elif op == "lead_mask":
        params = {
            "leads": [
                lead for lead in pick(cfg.lead_masks) if lead in spec.channels
            ]
        }
    elif op == "gain":
        params = {"factor": float(pick(cfg.gains))}
    else:
        env = asdict(cfg.editor.envelope)
        kind = editor.ARTIFACTS[int(rng.integers(len(editor.ARTIFACTS)))]
        controls: dict[str, Any] = {
            "artifact_type": kind,
            "electrode": editor.ELECTRODES[
                int(rng.integers(len(editor.ELECTRODES)))
            ],
        }
        for name in editor.CONTROLS[kind]:
            low, high = editor.control_range(env, kind, name, spec.duration_s)
            controls[name] = float(rng.uniform(low, high))
        params = {"controls": controls}
    return {
        "generator": f"{GENERATOR}@{VERSION}",
        "op": op,
        "params": params,
        "seed_key": seed_key,
    }


def apply(
    x: np.ndarray, ref: InputRef, recipe: dict[str, Any], cfg: AcquisitionConfig
) -> Variant:
    """Transform a waveform and return its reference and signal checks."""
    op, params = recipe["op"], recipe["params"]
    spec, fs = ref.signal, ref.signal.sample_rate
    x64 = np.asarray(x, dtype=np.float64)
    checks: list[dict[str, Any]] = []
    channels, n = spec.channels, spec.num_samples
    if op == "resample":
        down = Fraction(params["rate_hz"]).limit_denominator(1000) / Fraction(
            fs
        ).limit_denominator(1000)
        low = resample_poly(x64, down.numerator, down.denominator, axis=-1)
        y = resample_poly(low, down.denominator, down.numerator, axis=-1)[:, :n]
        if y.shape[-1] < n:
            y = np.pad(y, ((0, 0), (0, n - y.shape[-1])), mode="edge")
    elif op in ("lowpass", "highpass"):
        if not 0 < params["cutoff_hz"] < fs / 2:
            raise UnsupportedTransformation(
                f"cutoff {params['cutoff_hz']} Hz is outside (0, {fs / 2}) Hz"
            )
        sos = butter(4, params["cutoff_hz"], btype=op, fs=fs, output="sos")
        y = sosfiltfilt(sos, x64, axis=-1)
    elif op == "crop":
        start = round(params["start_s"] * fs)
        n = round(params["duration_s"] * fs)
        y = x64[:, start : start + n]
    elif op == "lead_mask":
        unknown = set(params["leads"]) - set(channels)
        if unknown or not params["leads"]:
            raise UnsupportedTransformation(
                f"lead_mask names leads {sorted(unknown)} the signal "
                f"does not have"
            )
        y = x64.copy()
        y[[channels.index(lead) for lead in params["leads"]]] = 0.0
    elif op == "gain":
        y = x64 * params["factor"]
    elif op == "electrode_artifact":
        edit_recipe = Recipe(
            ref.waveform_id,
            editor.GENERATOR,
            editor.VERSION,
            params["controls"],
            recipe["seed_key"],
            None,
        )
        candidate = editor.edit(x, edit_recipe, fs)
        found, _ = validate.evaluate(
            candidate, x, edit_recipe, asdict(cfg.editor), fs
        )
        checks.extend(check.to_json() for check in found)
        y = candidate.x.astype(np.float64)
    else:
        raise UnsupportedTransformation(
            f"{GENERATOR}@{VERSION} does not support {op!r}"
        )
    finite = bool(np.isfinite(y).all())
    peak = float(np.abs(y).max()) if finite and y.size else float("inf")
    limit = max(
        cfg.mv_limit, float(np.abs(x64).max())
    )  # bounds@1's rule: never past the limit or the parent's peak
    checks.append(
        {
            "name": "signal_valid@1",
            "version": "1",
            "value": peak,
            "pass": finite and y.shape == (len(channels), n) and peak <= limit,
        }
    )
    y32 = y.astype(np.float32)
    new_spec = SignalInfo(channels, fs, spec.units, n)
    recipe_id = content_id({"parent": ref.waveform_sha256, **recipe})
    out = InputRef(
        waveform_id=f"acq-{recipe_id[:20]}",
        waveform_sha256=waveform_sha256(y32),
        signal=new_spec,
        origin="transformed",
        source_record_ids=(ref.waveform_id,),
        root_record_ids=ref.root_record_ids,
        root_subject_uids=ref.root_subject_uids,
        dataset=ref.dataset,
        lineage={
            "recipe_id": recipe_id,
            "recipe": recipe,
            "parent_waveform_sha256": ref.waveform_sha256,
            "label_effect": LABEL_EFFECT[op],
            "time_offset_s": params.get("start_s", 0.0)
            if op == "crop"
            else 0.0,
        },
    )
    ok = all(c["pass"] for c in checks)
    return Variant(out, y32, recipe, checks, ok)


def inherit_annotations(
    parent: list[Evidence], variant: Variant, extractor: EvidenceExtractor
) -> list[Evidence]:
    """Keep annotations only when a transformation preserves their meaning."""
    effect = variant.ref.lineage["label_effect"]
    out = []
    for item in parent:
        if item.method != ANNOTATIONS:
            continue
        group = (
            "diagnosis"
            if item.attribute == "diagnostic_superclasses"
            else "signal_quality"
        )
        keep = effect[group] == "preserves"
        out.append(
            Evidence(
                **{
                    **{
                        name: getattr(item, name)
                        for name in item.__dataclass_fields__
                    },
                    "waveform_sha256": variant.ref.waveform_sha256,
                    "signal": variant.ref.signal,
                    "record_id": variant.ref.waveform_id,
                    "value": item.value if keep else None,
                    "status": item.status if keep else "unknown",
                    "confidence": item.confidence,
                    "observability": (
                        (
                            f"inherited from {item.record_id}: "
                            f"{variant.recipe['op']} preserves {group}"
                        )
                        if keep
                        else (
                            f"{variant.recipe['op']} invalidates the "
                            f"{group} annotation of {item.record_id}"
                        )
                    ),
                    "provenance": {
                        **item.provenance,
                        "inherited_from": item.evidence_id,
                        "label_effect": effect[group],
                    },
                    "config_sha256": extractor.config_sha256,
                }
            )
        )
    return out


def carry(
    task: QATask,
    variant: Variant,
    variant_evidence: list[Evidence],
    parent_context: str,
) -> tuple[QATask | None, str, tuple[Evidence, ...]]:
    """Rebuild and check a task using the transformed signal."""
    if task.composition in ("source", "comparison") or len(task.inputs) != 1:
        return None, "not_carried", ()
    template = TEMPLATES.get(task.template)
    if template is None:
        return None, "rejected:unknown_template", ()
    effect = variant.ref.lineage["label_effect"][
        TEMPLATE_GROUP[template.family]
    ]
    view = View(variant.ref, variant_evidence)
    context = {
        "clinical_context": [student_context("transformed", parent_context)]
    }
    generation = {
        "generator": f"{GENERATOR}@{VERSION}",
        "operator": "acquisition",
        "parent_task": task.task_id,
        "recipe_id": variant.ref.lineage["recipe_id"],
        "op": variant.recipe["op"],
    }
    built = build_task(
        template,
        task.generation["template_params"],
        [view],
        context,
        generation,
        lineage={**task.lineage, "carried_from": task.task_id},
    )
    if built is None:
        return None, "rejected:template_no_longer_applies", ()
    rebuilt, cited = built
    same = (rebuilt.answer, rebuilt.answerability) == (
        task.answer,
        task.answerability,
    )
    if same:
        decision = "preserved"
    elif effect == "preserves":
        return (
            None,
            f"rejected:invariant_violation:{TEMPLATE_GROUP[template.family]}",
            (),
        )
    elif rebuilt.answerability != "answerable":
        decision = "insufficient_after_transform"
    else:
        decision = "remeasured_updated"
    store = {e.evidence_id: e for e in [*variant_evidence, *cited]}
    checked = verify_task(
        replace(
            rebuilt,
            generation={**rebuilt.generation, "carry_decision": decision},
        ),
        store,
    )
    if checked.verification["status"] != "accepted":
        return checked, "rejected:" + checked.verification["reason"], cited
    return checked, decision, cited
