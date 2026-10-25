"""Seed and compositional question templates over verified evidence, with deterministic rationales.

A template turns the evidence of one or two waveforms plus explicit parameters into a ``Draft``: the
question, the canonical answer, the options, the numeric tolerance, the evidence it requires and one
claim per rationale sentence, each citing evidence ids. Templates never read a waveform or a generator
control: only ``Evidence`` whose kind is ``measured`` or ``annotation`` can establish an answer.

When required evidence is ``unknown`` or ``unobservable`` the template states that explicitly: the
draft answers ``INSUFFICIENT`` and its claim cites the inconclusive evidence. A template that cannot
apply at all (no such evidence, an ambiguous parameter near a decision boundary) returns ``None``.

Composition levels (ChatTS TSEvol, adapted): ``attribute`` (one measured fact), ``composed`` (depth:
several facts), ``reasoning`` (support or refute a statement from evidence), ``situation`` (a changed
lead set or a supplied interpretation rule) and ``comparison`` (two recordings, explicit order A, B).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from syntide.data.qa import INSUFFICIENT, Claim, Evidence, InputRef, QATask, format_number
from syntide.explore.plans import content_id
from syntide.generate.qa.evidence import SUPERCLASS_NAMES

LIMB = ("I", "II", "III", "AVR", "AVL", "AVF")
PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")
NOISE_NAMES = {
    "baseline_drift": "baseline drift",
    "static_noise": "static noise",
    "burst_noise": "burst noise",
    "electrodes_problems": "electrode problems",
}
RATE_BANDS = ("below 60 bpm", "60 to 100 bpm", "above 100 bpm")
RATE_BAND_RULE = "rate_bands@1"
BOUNDARY_MARGIN_BPM = 3.0


class View:
    """The evidence of one waveform, indexed by attribute and lead."""

    def __init__(self, ref: InputRef, evidence: Sequence[Evidence]) -> None:
        self.ref = ref
        self.evidence = [e for e in evidence if e.waveform_sha256 == ref.waveform_sha256]
        self._index = {
            (e.attribute, e.leads if e.attribute in ("wander_mv", "hf_rms_mv") else ()): e
            for e in self.evidence
        }

    def get(self, attribute: str, lead: str | None = None) -> Evidence | None:
        return self._index.get((attribute, (lead,) if lead else ()))

    @property
    def channels(self) -> tuple[str, ...]:
        return self.ref.signal.channels


@dataclass(frozen=True)
class Draft:
    question: str
    answer_type: str
    answer: Any
    options: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    claims: tuple[tuple[str, tuple[Evidence, ...]], ...]
    tolerance: dict[str, float] | None = None

    @property
    def answerable(self) -> bool:
        return self.answer != INSUFFICIENT


@dataclass(frozen=True)
class Template:
    id: str  # name@version
    family: str
    composition: str
    n_inputs: int
    build: Callable[[Sequence[View], dict[str, Any]], Draft | None]
    parameters: Callable[[Sequence[View]], list[dict[str, Any]]] = field(default=lambda views: [{}])


def _insufficient(
    question: str, answer_type: str, options: Sequence[str], missing: Sequence[Evidence]
) -> Draft:
    text = "; ".join(
        f"{e.attribute.replace('_', ' ')}{_where(e)} is {e.status} ({e.observability})" for e in missing
    )
    return Draft(
        question,
        answer_type,
        INSUFFICIENT,
        tuple(options),
        tuple(missing),
        ((f"The recording does not establish this: {text}.", tuple(missing)),),
    )


def _where(e: Evidence) -> str:
    return f" in lead {e.leads[0]}" if len(e.leads) == 1 else ""


def _blocked(*items: Evidence | None) -> list[Evidence] | None:
    """None if any item is missing or cannot answer; the inconclusive items if some are unknown."""
    if any(item is None or item.kind not in ("measured", "annotation") for item in items):
        return None
    return [item for item in items if not item.usable]


# ---------------------------------------------------------------- attribute templates (seed)


def _duration(views, params):
    e = views[0].get("duration_s")
    question = "How long is this recording, in seconds?"
    if (blocked := _blocked(e)) is None:
        return None
    if blocked:
        return _insufficient(question, "numeric", (), blocked)
    claim = (
        f"The recording holds {e.signal.num_samples} samples at {format_number(e.signal.sample_rate)} Hz, "
        f"which is {format_number(e.value)} s."
    )
    return Draft(question, "numeric", e.value, (), (e,), ((claim, (e,)),), {"abs": 0.05, "rel": 0.0})


def _flat_leads(views, params):
    e = views[0].get("flat_leads")
    if (blocked := _blocked(e)) is None:
        return None
    question = "Which leads of this recording carry no signal?"
    if blocked:
        return _insufficient(question, "lead_set", views[0].channels, blocked)
    claim = (
        f"Leads {', '.join(e.value)} are flat (robust spread and peak-to-peak below the flat threshold)."
        if e.value
        else "Every lead varies above the flat threshold, so no lead is missing."
    )
    return Draft(question, "lead_set", list(e.value), views[0].channels, (e,), ((claim, (e,)),))


def _lead_params(views):
    return [{"lead": lead} for lead in views[0].channels]


def _wander_in_lead(views, params):
    lead = params["lead"]
    e = views[0].get("wander_mv", lead)
    if (blocked := _blocked(e)) is None:
        return None
    question = (
        f"What is the baseline wander amplitude in lead {lead}, in mV (half the peak-to-peak below 0.7 Hz)?"
    )
    if blocked:
        return _insufficient(question, "numeric", (), blocked)
    claim = (
        f"Low-pass filtering lead {lead} below 0.7 Hz leaves a slow component of "
        f"{format_number(e.value)} mV half peak-to-peak."
    )
    return Draft(question, "numeric", e.value, (), (e,), ((claim, (e,)),), {"abs": 0.05, "rel": 0.15})


def _strongest_lead(attribute: str, words: str):
    def build(views, params):
        view = views[0]
        e = view.get(attribute)
        if (blocked := _blocked(e)) is None:
            return None
        available = view.get("available_leads")
        options = tuple(available.value) if available and available.usable else view.channels
        question = f"In which lead is the {words} largest?"
        if blocked:
            return _insufficient(question, "choice", options, blocked)
        if e.value not in options:
            return None
        claim = f"Lead {e.value} measures the largest {words}, clearly above every other lead."
        return Draft(question, "choice", e.value, options, (e,), ((claim, (e,)),))

    return build


def _heart_rate(views, params):
    e = views[0].get("heart_rate_bpm")
    if (blocked := _blocked(e)) is None:
        return None
    question = "What is the heart rate of this recording, in beats per minute?"
    if blocked:
        return _insufficient(question, "numeric", (), blocked)
    claim = (
        f"R-peak detection gives a median RR interval of {format_number(60 / e.value)} s, "
        f"so the heart rate is {format_number(e.value)} bpm."
    )
    return Draft(question, "numeric", e.value, (), (e,), ((claim, (e,)),), {"abs": 3.0, "rel": 0.05})


def _beat_count(views, params):
    e = views[0].get("beat_count")
    if (blocked := _blocked(e)) is None:
        return None
    question = "How many heartbeats does this recording contain?"
    if blocked:
        return _insufficient(question, "numeric", (), blocked)
    claim = f"R-peak detection finds {e.value} beats in the recording."
    return Draft(question, "numeric", e.value, (), (e,), ((claim, (e,)),), {"abs": 1.0, "rel": 0.0})


def _regularity(views, params):
    e = views[0].get("rr_regularity")
    cv = views[0].get("rr_cv")
    if (blocked := _blocked(e)) is None:
        return None
    question = "Is the rhythm of this recording regular or irregular (RR intervals)?"
    options = ("regular", "irregular")
    if blocked:
        return _insufficient(question, "choice", options, blocked)
    cited = (e, cv) if cv is not None and cv.usable else (e,)
    claim = (
        f"The RR intervals vary by a coefficient of {format_number(cv.value)}, so they are {e.value}."
        if len(cited) == 2
        else f"The RR intervals are {e.value}."
    )
    return Draft(question, "choice", e.value, options, cited, ((claim, cited),))


def rate_band(hr: float) -> str | None:
    """The rate band of ``hr``, or None within BOUNDARY_MARGIN_BPM of a band edge (rule rate_bands@1)."""
    if min(abs(hr - 60), abs(hr - 100)) < BOUNDARY_MARGIN_BPM:
        return None
    return RATE_BANDS[0] if hr < 60 else RATE_BANDS[1] if hr <= 100 else RATE_BANDS[2]


def _rate_band(views, params):
    e = views[0].get("heart_rate_bpm")
    if (blocked := _blocked(e)) is None:
        return None
    question = "Is the heart rate below 60 bpm, between 60 and 100 bpm, or above 100 bpm?"
    if blocked:
        return _insufficient(question, "choice", RATE_BANDS, blocked)
    band = rate_band(e.value)
    if band is None:
        return None
    claim = f"The measured heart rate is {format_number(e.value)} bpm, which is {band}."
    return Draft(question, "choice", band, RATE_BANDS, (e,), ((claim, (e,)),))


def _noise_params(views):
    return [{"artifact": kind} for kind in NOISE_NAMES]


def _noise_annotation(views, params):
    kind = params["artifact"]
    e = views[0].get(f"annotated_{kind}")
    if (blocked := _blocked(e)) is None:
        return None
    question = f"Does this ECG show {NOISE_NAMES[kind]}?"
    if blocked:
        return _insufficient(question, "binary", ("yes", "no"), blocked)
    answer = "yes" if e.value else "no"
    verb = "lists" if e.value else "does not list"
    claim = f"The recording's signal-quality annotation {verb} {NOISE_NAMES[kind]}."
    return Draft(question, "binary", answer, ("yes", "no"), (e,), ((claim, (e,)),))


def _superclasses(views, params):
    e = views[0].get("diagnostic_superclasses")
    if (blocked := _blocked(e)) is None:
        return None
    options = tuple(SUPERCLASS_NAMES[c] for c in sorted(SUPERCLASS_NAMES))
    question = "Which diagnostic superclasses are annotated for this ECG?"
    if blocked:
        return _insufficient(question, "multi_choice", options, blocked)
    answer = [SUPERCLASS_NAMES[c] for c in e.value if c in SUPERCLASS_NAMES]
    claim = f"The diagnostic annotation of the recording lists {', '.join(answer) or 'no superclass'}."
    return Draft(question, "multi_choice", answer, options, (e,), ((claim, (e,)),))


INTERVALS = {
    "qrs_duration_ms": "QRS duration",
    "pr_interval_ms": "PR interval",
    "qt_interval_ms": "QT interval",
    "qtc_ms": "corrected QT interval",
}


def _interval_params(views):
    return [{"attribute": a} for a in INTERVALS if views[0].get(a) is not None]


def _interval(views, params):
    attribute = params["attribute"]
    e = views[0].get(attribute)
    if (blocked := _blocked(e)) is None:
        return None
    question = f"What is the {INTERVALS[attribute]} of this ECG, in milliseconds?"
    if blocked:
        return _insufficient(question, "numeric", (), blocked)
    claim = (
        f"The measurement sidecar ({e.provenance.get('extractor')}) gives a {INTERVALS[attribute]} "
        f"of {format_number(e.value)} ms."
    )
    return Draft(question, "numeric", e.value, (), (e,), ((claim, (e,)),), {"abs": 10.0, "rel": 0.05})


# ---------------------------------------------------------------- composed, reasoning, situation


def _rate_and_wander_params(views):
    return [
        {"threshold_bpm": t, "lead": lead, "wander_mv": w}
        for t in (60, 100)
        for w in (0.2, 0.5)
        for lead in ("II", "V2")
    ]


def _rate_and_wander(views, params):
    view, thr, lead, mv = views[0], params["threshold_bpm"], params["lead"], params["wander_mv"]
    hr, wander = view.get("heart_rate_bpm"), view.get("wander_mv", lead)
    if (blocked := _blocked(hr, wander)) is None:
        return None
    question = (
        f"Is the heart rate above {thr} bpm and the baseline wander in lead {lead} "
        f"below {format_number(mv)} mV?"
    )
    if blocked:
        return _insufficient(question, "binary", ("yes", "no"), blocked)
    if abs(hr.value - thr) < BOUNDARY_MARGIN_BPM or abs(wander.value - mv) < 0.05:
        return None
    fast, calm = hr.value > thr, wander.value < mv
    answer = "yes" if fast and calm else "no"
    claims = (
        (
            f"The heart rate is {format_number(hr.value)} bpm, {'above' if fast else 'not above'} {thr} bpm.",
            (hr,),
        ),
        (
            f"Lead {lead} wanders {format_number(wander.value)} mV, "
            f"{'below' if calm else 'not below'} {format_number(mv)} mV.",
            (wander,),
        ),
    )
    return Draft(question, "binary", answer, ("yes", "no"), (hr, wander), claims)


def _rate_statement(low: float | None, high: float | None):
    """A heart-rate statement's truth, or None within the boundary margin of an edge it names."""

    def judge(e: Evidence) -> tuple[bool, str] | None:
        edges = [edge for edge in (low, high) if edge is not None]
        if any(abs(e.value - edge) < BOUNDARY_MARGIN_BPM for edge in edges):
            return None
        holds = (low is None or e.value >= low) and (high is None or e.value <= high)
        return holds, f"The measured heart rate is {format_number(e.value)} bpm."

    return judge


def _regularity_statement(wanted: str):
    return lambda e: (e.value == wanted, f"The RR intervals are {e.value}.")


def _flat_statement(lead: str | None):
    def judge(e: Evidence) -> tuple[bool, str]:
        if lead is None:
            holds = not e.value
            return holds, "No lead is flat." if holds else f"Leads {', '.join(e.value)} are flat."
        holds = lead in e.value
        return holds, f"Lead {lead} is {'flat' if holds else 'not flat'}."

    return judge


# Statements come in complementary sets (abnormal and normal), so neither answer is a shortcut.
STATEMENTS = {
    "irregular": ("The rhythm is irregular.", "rr_regularity", _regularity_statement("irregular")),
    "regular": ("The rhythm is regular.", "rr_regularity", _regularity_statement("regular")),
    "fast": ("The heart rate exceeds 100 bpm.", "heart_rate_bpm", _rate_statement(100, None)),
    "slow": ("The heart rate is below 60 bpm.", "heart_rate_bpm", _rate_statement(None, 60)),
    "normal_rate": ("The heart rate is between 60 and 100 bpm.", "heart_rate_bpm", _rate_statement(60, 100)),
    "flat_v2": ("Lead V2 carries no signal.", "flat_leads", _flat_statement("V2")),
    "all_leads": ("Every lead carries signal.", "flat_leads", _flat_statement(None)),
}


def _statement_params(views):
    return [{"statement": key} for key in STATEMENTS]


def _support_refute(views, params):
    text, attribute, judge = STATEMENTS[params["statement"]]
    e = views[0].get(attribute)
    if (blocked := _blocked(e)) is None:
        return None
    question = f'Statement: "{text}" Is this statement supported or refuted by the recording?'
    options = ("supported", "refuted")
    if blocked:
        return _insufficient(question, "choice", options, blocked)
    judged = judge(e)
    if judged is None:
        return None
    holds, fact = judged
    return Draft(question, "choice", "supported" if holds else "refuted", options, (e,), ((fact, (e,)),))


SUBSETS = {"limb": LIMB, "precordial": PRECORDIAL}


def _subset_params(views):
    return [{"subset": s, "lead": lead} for s in SUBSETS for lead in ("II", "V2")]


def _lead_subset(views, params):
    """Situation: only a subset of leads was recorded. A question about another lead is unanswerable."""
    subset, lead = SUBSETS[params["subset"]], params["lead"]
    e = views[0].get("wander_mv", lead)
    if e is None:
        return None
    question = (
        f"Suppose only the {params['subset']} leads ({', '.join(subset)}) were recorded. "
        f"What is the baseline wander amplitude in lead {lead}, in mV?"
    )
    if lead not in subset:
        # A derived fact under its own attribute, so it never shadows the measured wander of that lead.
        unseen = Evidence(
            **{
                **_fields(e),
                "attribute": "wander_mv_in_stated_subset",
                "value": None,
                "status": "unobservable",
                "confidence": None,
                "observability": f"lead {lead} is outside the stated {params['subset']} leads",
                "rules": ("stated_lead_subset@1",),
            }
        )
        return _insufficient(question, "numeric", (), [unseen])
    inner = _wander_in_lead(views, {"lead": lead})
    return None if inner is None else replace(inner, question=question)


def _fields(e: Evidence) -> dict[str, Any]:
    return {name: getattr(e, name) for name in e.__dataclass_fields__}


def _rule_params(views):
    return [{"threshold_bpm": t} for t in (90, 110, 120)]


def _supplied_rule(views, params):
    """Situation: the question supplies the interpretation rule."""
    thr = params["threshold_bpm"]
    e = views[0].get("heart_rate_bpm")
    if (blocked := _blocked(e)) is None:
        return None
    question = f"Using the rule that a heart rate above {thr} bpm counts as fast, is this recording fast?"
    if blocked:
        return _insufficient(question, "binary", ("yes", "no"), blocked)
    if abs(e.value - thr) < BOUNDARY_MARGIN_BPM:
        return None
    fast = e.value > thr
    side = "above" if fast else "not above"
    claim = f"The heart rate is {format_number(e.value)} bpm, {side} the stated {thr} bpm."
    return Draft(question, "binary", "yes" if fast else "no", ("yes", "no"), (e,), ((claim, (e,)),))


# ---------------------------------------------------------------- comparison (two inputs, order A, B)

COMPARED = {
    "heart_rate_bpm": ("heart rate", "bpm", 5.0),
    "amplitude_mv": ("signal amplitude (largest robust peak-to-peak over leads)", "mV", 0.1),
    "max_wander_mv": ("largest baseline wander over leads", "mV", 0.1),
}


def _compare_params(views):
    return [{"attribute": a} for a in COMPARED]


def scalar(view: View, attribute: str) -> Evidence | None:
    """A whole-recording scalar, including ones derived from per-lead evidence."""
    if attribute != "max_wander_mv":
        return view.get(attribute)
    per_lead = [view.get("wander_mv", lead) for lead in view.channels]
    usable = [e for e in per_lead if e is not None and e.usable]
    return (
        max(usable, key=lambda e: e.value) if usable else next((e for e in per_lead if e is not None), None)
    )


def _compare(views, params):
    attribute = params["attribute"]
    words, units, margin = COMPARED[attribute]
    a, b = scalar(views[0], attribute), scalar(views[1], attribute)
    if (blocked := _blocked(a, b)) is None:
        return None
    question = f"Which recording, A or B, has the higher {words}?"
    options = ("A", "B")
    if blocked:
        return _insufficient(question, "choice", options, blocked)
    if abs(a.value - b.value) < max(margin, 0.05 * max(a.value, b.value)):
        return None
    answer = "A" if a.value > b.value else "B"
    claims = (
        (f"Recording A measures {format_number(a.value)} {units}.", (a,)),
        (f"Recording B measures {format_number(b.value)} {units}.", (b,)),
    )
    return Draft(question, "choice", answer, options, (a, b), claims)


TEMPLATES: dict[str, Template] = {
    t.id: t
    for t in (
        Template("acquisition.duration@1", "acquisition.duration", "attribute", 1, _duration),
        Template("acquisition.flat_leads@1", "acquisition.flat_leads", "attribute", 1, _flat_leads),
        Template(
            "artifact.wander_in_lead@1",
            "artifact.wander_in_lead",
            "attribute",
            1,
            _wander_in_lead,
            _lead_params,
        ),
        Template(
            "artifact.max_wander_lead@1",
            "artifact.max_wander_lead",
            "attribute",
            1,
            _strongest_lead("max_wander_lead", "baseline wander"),
        ),
        Template(
            "artifact.max_noise_lead@1",
            "artifact.max_noise_lead",
            "attribute",
            1,
            _strongest_lead("max_hf_noise_lead", "high-frequency noise"),
        ),
        Template("rhythm.heart_rate@1", "rhythm.heart_rate", "attribute", 1, _heart_rate),
        Template("rhythm.beat_count@1", "rhythm.beat_count", "attribute", 1, _beat_count),
        Template("rhythm.regularity@1", "rhythm.regularity", "attribute", 1, _regularity),
        Template("rhythm.rate_band@1", "rhythm.rate_band", "attribute", 1, _rate_band),
        Template(
            "quality.noise_annotation@1",
            "quality.noise_annotation",
            "attribute",
            1,
            _noise_annotation,
            _noise_params,
        ),
        Template("diagnosis.superclass@1", "diagnosis.superclass", "attribute", 1, _superclasses),
        Template(
            "interval.measurement@1", "interval.measurement", "attribute", 1, _interval, _interval_params
        ),
        Template(
            "compose.rate_and_wander@1",
            "compose.rate_and_wander",
            "composed",
            1,
            _rate_and_wander,
            _rate_and_wander_params,
        ),
        Template(
            "reason.support_refute@1",
            "reason.support_refute",
            "reasoning",
            1,
            _support_refute,
            _statement_params,
        ),
        Template(
            "situation.lead_subset@1", "situation.lead_subset", "situation", 1, _lead_subset, _subset_params
        ),
        Template(
            "situation.supplied_rule@1",
            "situation.supplied_rule",
            "situation",
            1,
            _supplied_rule,
            _rule_params,
        ),
        Template("compare.attribute@1", "compare.attribute", "comparison", 2, _compare, _compare_params),
    )
}
# Families whose answer is a PTB-XL annotation. The baseline-compatible clinical context names the same
# signal-quality flags, so these tasks are marked context_reveals_answer.
ANNOTATION_FAMILIES = ("quality.noise_annotation", "diagnosis.superclass")


def rationale_text(claims: Sequence[tuple[str, Sequence[Evidence]]]) -> str:
    return " ".join(text for text, _ in claims)


def build_task(
    template: Template,
    params: dict[str, Any],
    views: Sequence[View],
    context: dict[str, Any],
    generation: dict[str, Any],
    *,
    lineage: dict[str, Any] | None = None,
    question: str | None = None,
    paraphrase: str = "seed",
    rationale: str | None = None,
) -> tuple[QATask, tuple[Evidence, ...]] | None:
    """Instantiate ``template``: the ``QATask`` (not yet verified) and every evidence record it cites.

    The evidence may include facts the template derived itself (a situation's unobservable lead), so the
    corpus stores it next to the extracted evidence.
    """
    if len(views) != template.n_inputs:
        raise ValueError(f"{template.id} takes {template.n_inputs} inputs, got {len(views)}")
    draft = template.build(views, params)
    if draft is None:
        return None
    inputs = tuple(view.ref for view in views)
    semantic_key = (
        "sem-"
        + content_id(
            {
                "template": template.id,
                "params": params,
                "inputs": [ref.waveform_sha256 for ref in inputs],
                "answer": draft.answer,
            }
        )[:24]
    )
    claims = tuple(Claim(text, tuple(e.evidence_id for e in cited)) for text, cited in draft.claims)
    task = QATask(
        family=template.family,
        template=template.id,
        inputs=inputs,
        question=question or draft.question,
        answer_type="insufficient" if not draft.answerable else draft.answer_type,
        answer=draft.answer,
        options=draft.options,
        answerability="answerable" if draft.answerable else "insufficient_evidence",
        rationale=rationale if rationale is not None else rationale_text(draft.claims),
        claims=claims,
        required_evidence=tuple(dict.fromkeys(e.evidence_id for e in draft.evidence)),
        context=context,
        tolerance=draft.tolerance,
        semantic_key=semantic_key,
        paraphrase=paraphrase,
        composition=template.composition,
        generation={**generation, "template_params": params},
        verification={"status": "pending"},
        lineage=lineage or {},
    )
    return task, tuple({e.evidence_id: e for e in draft.evidence}.values())


def candidates(template: Template, views: Sequence[View]) -> list[dict[str, Any]]:
    """The parameterizations of ``template`` that apply to these views, in a fixed order."""
    return [params for params in template.parameters(views) if template.build(views, params) is not None]
