"""Define waveform references, evidence, and question-answer tasks.

Each object has an identifier derived from its contents:

- ``InputRef`` identifies a waveform, its exact bytes, and the real
  recordings and patients from which it was derived.
- ``Evidence`` records one fact about a waveform, including units, location,
  observation status, and the method used to establish it. Measured evidence
  stays distinct from annotations, generation requests, and unverified
  claims.
- ``QATask`` contains a question, ordered inputs, expected answer,
  supporting evidence, and explanation. Model inputs exclude training
  answers.

``INSUFFICIENT`` means the evidence cannot answer the question. This differs
from the ECG-QA dataset answer "none", which means no finding of the
specified kind. The collection writer maps these objects to TimeNet records
and tasks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from syntide.explore.plans import content_id

EVIDENCE_SCHEMA = "syntide.ecg-evidence@1"
TASK_SCHEMA = "syntide.qa-task@1"
INPUT_SCHEMA = "syntide.qa-input@1"

STATUSES = ("present", "absent", "unknown", "unobservable")
# measured: computed from this waveform's bytes; annotation: an existing label
# or measurement sidecar;
# requested_control: what a generator was asked to produce; hypothesis: an
# unverified diagnostic claim.
KINDS = ("measured", "annotation", "requested_control", "hypothesis")
# Only these kinds may establish an answer. A requested control or a hypothesis
# never does.
ANSWER_KINDS = ("measured", "annotation")
ORIGINS = ("real", "transformed", "generated")
ANSWER_TYPES = (
    "binary",
    "choice",
    "multi_choice",
    "numeric",
    "lead_set",
    "insufficient",
)
ANSWERABILITY = ("answerable", "insufficient_evidence")
INSUFFICIENT = "insufficient evidence"
# The label-effect vocabulary of the editor and per-lead augmentation
# (``LABEL_EFFECT``).
LABEL_EFFECTS = ("preserves", "determines", "invalidates")


class ContractError(ValueError):
    """An evidence record or task violates its format or content hash."""


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("evidence and task values must be finite")
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class SignalInfo:
    channels: tuple[str, ...]
    sample_rate: float
    units: str
    num_samples: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "sample_rate", float(self.sample_rate))
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ContractError("channels must be unique and non-empty")
        if (
            not (self.sample_rate > 0 and math.isfinite(self.sample_rate))
            or self.num_samples <= 0
        ):
            raise ContractError("sample_rate and num_samples must be positive")

    @property
    def duration_s(self) -> float:
        return self.num_samples / self.sample_rate

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "channels": list(self.channels)}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> SignalInfo:
        return cls(**value)


@dataclass(frozen=True)
class InputRef:
    """A waveform reference with its hash and source patients."""

    waveform_id: str
    waveform_sha256: str
    signal: SignalInfo
    origin: str  # real | transformed | generated
    source_record_ids: tuple[
        str, ...
    ]  # immediate parents (empty for a real recording)
    root_record_ids: tuple[str, ...]  # the real recordings it descends from
    root_subject_uids: tuple[str, ...]  # their patients
    # e.g. physionet/ptb-xl@1.0.3, or the generator id for generated waveforms
    dataset: str
    lineage: dict[str, Any] = field(
        default_factory=dict
    )  # recipe ids, run ids, generator provenance
    schema_version: str = INPUT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_record_ids",
            "root_record_ids",
            "root_subject_uids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.origin not in ORIGINS:
            raise ContractError(
                f"origin must be one of {ORIGINS}, got {self.origin!r}"
            )
        if self.origin == "real" and self.root_record_ids != (
            self.waveform_id,
        ):
            raise ContractError("a real input is its own root recording")
        if len(self.root_record_ids) != len(self.root_subject_uids):
            raise ContractError("each root record needs its patient")
        _finite(self.lineage)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["signal"] = self.signal.to_json()
        for name in (
            "source_record_ids",
            "root_record_ids",
            "root_subject_uids",
        ):
            out[name] = list(out[name])
        return out

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> InputRef:
        return cls(**{**value, "signal": SignalInfo.from_json(value["signal"])})


@dataclass(frozen=True)
class Evidence:
    """A waveform fact identified by the hash of all its fields."""

    waveform_sha256: str
    signal: SignalInfo
    record_id: str
    root_record_ids: tuple[str, ...]
    root_subject_uids: tuple[str, ...]
    attribute: str
    # number, string, bool, list, or None when status is unknown/unobservable
    value: Any
    units: str | None
    status: str
    kind: str
    leads: tuple[
        str, ...
    ]  # leads the fact applies to; () for a whole-recording fact
    interval_s: tuple[float, float] | None
    # Confidence is in [0, 1], or None when the method does not provide it.
    confidence: float | None
    observability: (
        str  # how directly the waveform shows it, in the method's words
    )
    method: str  # name@version
    config_sha256: str  # hash of the method's settings
    rules: tuple[str, ...] = ()  # name@version of every decision rule applied
    provenance: dict[str, Any] = field(
        default_factory=dict
    )  # sidecar checksums, annotation source
    schema_version: str = EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("root_record_ids", "root_subject_uids", "leads", "rules"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.interval_s is not None:
            object.__setattr__(
                self, "interval_s", tuple(float(v) for v in self.interval_s)
            )
        if self.status not in STATUSES:
            raise ContractError(
                f"{self.attribute}: status must be one of {STATUSES}"
            )
        if self.kind not in KINDS:
            raise ContractError(
                f"{self.attribute}: kind must be one of {KINDS}"
            )
        if (
            self.status in ("unknown", "unobservable")
            and self.value is not None
        ):
            raise ContractError(
                f"{self.attribute}: an {self.status} fact carries no value"
            )
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ContractError(
                f"{self.attribute}: confidence must be in [0, 1]"
            )
        unknown = set(self.leads) - set(self.signal.channels)
        if unknown:
            raise ContractError(
                f"{self.attribute}: leads {sorted(unknown)} are not "
                "channels of the signal"
            )
        object.__setattr__(self, "value", _finite(self.value))
        _finite(self.provenance)

    @property
    def evidence_id(self) -> str:
        return "ev-" + content_id(self._content())[:24]

    @property
    def usable(self) -> bool:
        """Whether this fact may establish an answer."""
        return self.kind in ANSWER_KINDS and self.status in (
            "present",
            "absent",
        )

    def _content(self) -> dict[str, Any]:
        out = asdict(self)
        out["signal"] = self.signal.to_json()
        for name in ("root_record_ids", "root_subject_uids", "leads", "rules"):
            out[name] = list(out[name])
        if out["interval_s"] is not None:
            out["interval_s"] = list(out["interval_s"])
        return out

    def to_json(self) -> dict[str, Any]:
        return {**self._content(), "evidence_id": self.evidence_id}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> Evidence:
        body = {k: v for k, v in value.items() if k != "evidence_id"}
        body["signal"] = SignalInfo.from_json(body["signal"])
        if body.get("interval_s") is not None:
            body["interval_s"] = tuple(body["interval_s"])
        evidence = cls(**body)
        if (
            "evidence_id" in value
            and value["evidence_id"] != evidence.evidence_id
        ):
            raise ContractError(
                f"evidence {value['evidence_id']} does not match its content"
            )
        return evidence


@dataclass(frozen=True)
class Claim:
    text: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def to_json(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class QATask:
    """A question with ordered inputs and a content-derived identifier."""

    family: str  # semantic family, e.g. rhythm.heart_rate; paraphrases share it
    template: str  # template id and version, e.g. rhythm.heart_rate@1
    inputs: tuple[InputRef, ...]
    question: str
    answer_type: str
    answer: Any  # canonical value; INSUFFICIENT when the task is unanswerable
    options: tuple[str, ...]
    answerability: str
    rationale: str
    claims: tuple[Claim, ...]
    required_evidence: tuple[str, ...]  # evidence ids the answer depends on
    context: dict[
        str, Any
    ]  # student-visible context, already filtered by the context policy
    tolerance: (
        dict[str, float] | None
    )  # numeric answers: {"abs": ..., "rel": ...}
    semantic_key: (
        str  # family + parameters + inputs + answer; identical for paraphrases
    )
    paraphrase: (
        str  # "seed" or the teacher response hash of the question wording
    )
    composition: str  # attribute | composed | comparison | situation | source
    generation: dict[
        str, Any
    ]  # generator/operator, parent task ids, seed, teacher reference
    verification: dict[str, Any]  # checks and their outcome
    lineage: dict[str, Any] = field(
        default_factory=dict
    )  # source dataset task ids, pair id
    schema_version: str = TASK_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(
            self, "required_evidence", tuple(self.required_evidence)
        )
        if not self.inputs:
            raise ContractError(
                f"{self.template}: a task needs at least one input"
            )
        if len({ref.waveform_id for ref in self.inputs}) != len(self.inputs):
            raise ContractError(
                f"{self.template}: a task lists the same input twice"
            )
        if self.answer_type not in ANSWER_TYPES:
            raise ContractError(
                f"{self.template}: answer_type must be one of {ANSWER_TYPES}"
            )
        if self.answerability not in ANSWERABILITY:
            raise ContractError(
                f"{self.template}: answerability must be one of {ANSWERABILITY}"
            )
        insufficient = self.answerability == "insufficient_evidence"
        if insufficient != (self.answer == INSUFFICIENT):
            raise ContractError(
                f"{self.template}: an unanswerable task answers "
                f"{INSUFFICIENT!r}, only it"
            )
        if (
            self.answer_type in ("choice", "binary")
            and not insufficient
            and self.answer not in self.options
        ):
            raise ContractError(
                f"{self.template}: answer {self.answer!r} is not one of "
                "its options"
            )
        if (
            self.answer_type in ("multi_choice", "lead_set")
            and not insufficient
        ):
            if not isinstance(self.answer, list | tuple) or set(
                self.answer
            ) - set(self.options):
                raise ContractError(
                    f"{self.template}: a set answer must be a subset of "
                    "its options"
                )
            object.__setattr__(
                self, "answer", sorted(self.answer, key=self.options.index)
            )
        if self.answer_type == "numeric" and not insufficient:
            if not isinstance(self.answer, int | float) or isinstance(
                self.answer, bool
            ):
                raise ContractError(
                    f"{self.template}: a numeric answer must be a number"
                )
            if not self.tolerance:
                raise ContractError(
                    f"{self.template}: a numeric answer needs a tolerance"
                )
        for claim in self.claims:
            missing = set(claim.evidence_ids) - set(self.required_evidence)
            if missing:
                raise ContractError(
                    f"{self.template}: a claim cites evidence the task "
                    "does not require"
                )
        object.__setattr__(self, "answer", _finite(self.answer))
        _finite(self.context)

    @property
    def task_id(self) -> str:
        return "qa-" + content_id(self._content())[:24]

    @property
    def answer_text(self) -> str:
        return answer_text(self.answer_type, self.answer)

    def student_view(self) -> dict[str, Any]:
        """Return only the inputs a model may read.

        Include waveform references, the question, options, and permitted
        context. Exclude answers, explanations, evidence, and generation
        settings.
        """
        return {
            "inputs": [
                {"waveform_id": ref.waveform_id, "signal": ref.signal.to_json()}
                for ref in self.inputs
            ],
            "question": self.question,
            "options": list(self.options),
            "context": dict(self.context),
        }

    def target(self, mode: str) -> str:
        """Return an answer, optionally preceded by its explanation."""
        line = f"Answer: {self.answer_text}"
        if mode == "answer_only":
            return line
        if mode == "rationale":
            return (
                f"{self.rationale.strip()}\n{line}"
                if self.rationale.strip()
                else line
            )
        raise ContractError(
            f"unknown target mode {mode!r}; choose answer_only or rationale"
        )

    def _content(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["inputs"] = [ref.to_json() for ref in self.inputs]
        out["options"] = list(self.options)
        out["claims"] = [claim.to_json() for claim in self.claims]
        out["required_evidence"] = list(self.required_evidence)
        if isinstance(out["answer"], tuple):
            out["answer"] = list(out["answer"])
        return out

    def to_json(self) -> dict[str, Any]:
        return {**self._content(), "task_id": self.task_id}

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> QATask:
        body = {k: v for k, v in value.items() if k != "task_id"}
        body["inputs"] = tuple(
            InputRef.from_json(ref) for ref in body["inputs"]
        )
        body["claims"] = tuple(
            Claim(c["text"], tuple(c["evidence_ids"])) for c in body["claims"]
        )
        task = cls(**body)
        if "task_id" in value and value["task_id"] != task.task_id:
            raise ContractError(
                f"task {value['task_id']} does not match its content; it "
                "was edited"
            )
        return task


def answer_text(answer_type: str, answer: Any) -> str:
    """Format the expected answer following the Answer: label."""
    if answer == INSUFFICIENT:
        return INSUFFICIENT
    if answer_type in ("multi_choice", "lead_set"):
        return ", ".join(answer) if answer else "none"
    if answer_type == "numeric":
        return format_number(answer)
    return str(answer)


def format_number(value: float) -> str:
    """Format numbers with up to three decimal places and no exponent."""
    rounded = round(float(value), 3)
    text = f"{rounded:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text
