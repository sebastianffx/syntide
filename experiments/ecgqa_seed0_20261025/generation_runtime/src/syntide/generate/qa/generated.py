"""Generated waveforms (``syntide.generate.full`` bundles, e.g. DiffuSETS) into the QA verification pipeline.

A full-generation bundle stays ``generated_pending_domain_and_label_review``; nothing here changes it.
This module reads the bundle (checking ``run.json``'s artifact hashes and each signal's sha256), then
turns each sample into a QA input only after these explicit steps:

1. **Leads.** Channel names are normalized through ``LEAD_ALIASES`` (DiffuSETS writes ``aVR``, PTB-XL
   ``AVR``) and reordered to the target lead order. An unknown or duplicated lead is refused.
2. **Rate.** DiffuSETS is native 1,024 samples at 102.4 Hz over 10 s. A different target rate needs
   ``allow_resample``; the conversion is a real polyphase resampling (102.4 -> 100 Hz is 125/128, so
   1,024 -> 1,000 samples) and is recorded. A mismatch without it is refused, never relabelled.
3. **Ancestry.** The conditioning's ``source_record_ids`` are the real recordings the sample descends
   from; their patients come from the index. A sample without them has no resolvable ancestry and is
   refused (its fresh ``synthetic:...`` id proves nothing).
4. **Verification.** The waveform is measured like any other. It becomes ``verified`` for exactly the
   attributes in ``required`` whose evidence is usable with enough confidence, plus plausibility
   checks; otherwise it stays pending and never enters an accepted export.

Requested labels and the conditioning (report, heart rate, age, sex) are recorded as
``requested_control`` lineage only: they are never evidence that can establish an answer. The 1,536-d
``text-embedding-ada-002`` conditioning stays the backend's concern (``diffusets.py``); explorer
embeddings are not a substitute. The pretrained generator's training cohort is recorded when known;
its overlap with an external benchmark stays ``unknown`` and visible in evaluation metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from syntide.data.qa import Evidence, InputRef, SignalInfo
from syntide.data.schema import LEADS
from syntide.generate.qa.evidence import EvidenceExtractor, waveform_sha256

LEAD_ALIASES = {"aVR": "AVR", "aVL": "AVL", "aVF": "AVF", "avr": "AVR", "avl": "AVL", "avf": "AVF"}
PENDING = "generated_pending_domain_and_label_review"
GENERATOR_COHORTS = {
    "diffusets@1": {
        "training_cohort": "MIMIC-IV-ECG (as described by the DiffuSETS authors)",
        "overlap_with_ptbxl": "none declared",
        "overlap_with_external_benchmarks": "unknown",
    }
}


class GeneratedInputError(ValueError):
    """A generated sample cannot become a QA input: unknown leads, an unconverted rate, no ancestry."""


@dataclass(frozen=True)
class GeneratedSample:
    record_id: str
    run_id: str
    generator: str
    seed: int
    conditioning: dict[str, Any]
    source_record_ids: tuple[str, ...]
    channels: tuple[str, ...]
    sample_rate: float
    units: str
    signal_path: Path
    signal_sha256: str
    status: str
    requested_labels: tuple[str, ...]

    def load(self) -> np.ndarray:
        """The waveform, read only after the caller checked this sample's ancestry."""
        if _sha256(self.signal_path) != self.signal_sha256:
            raise GeneratedInputError(f"{self.signal_path} does not match its ledger sha256")
        return np.load(self.signal_path, allow_pickle=False)


def normalize_channels(channels: Sequence[str]) -> tuple[str, ...]:
    out = tuple(LEAD_ALIASES.get(ch, ch) for ch in channels)
    if len(set(out)) != len(out):
        raise GeneratedInputError(f"channels {list(channels)} collide after alias normalization")
    unknown = set(out) - set(LEADS)
    if unknown:
        raise GeneratedInputError(f"channels {sorted(unknown)} are not ECG leads this pipeline knows")
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bundle(path: str | Path) -> list[GeneratedSample]:
    """Read a bundle's catalog after checking ``run.json``'s artifacts. No waveform is read here."""
    path = Path(path)
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    for name, digest in run["artifacts"].items():
        if _sha256(path / name) != digest:
            raise GeneratedInputError(f"{path / name} changed since generation")
    samples = []
    for line in (path / "catalog.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        row = entry["metadata"]["generation"]
        signal_path = path / entry["signal_path"]
        samples.append(
            GeneratedSample(
                record_id=entry["record_id"],
                run_id=row["run_id"],
                generator=row["generator"],
                seed=int(row["seed"]),
                conditioning=row["conditioning"],
                source_record_ids=tuple(row.get("source_record_ids") or ()),
                channels=tuple(entry["channels"]),
                sample_rate=float(entry["sample_rate"]),
                units=entry["units"],
                signal_path=signal_path,
                signal_sha256=row["signal_sha256"],
                status=row["status"],
                requested_labels=tuple(entry["metadata"].get("requested_labels", ())),
            )
        )
    return samples


def to_input(
    sample: GeneratedSample,
    target: SignalInfo,
    subject_of: Mapping[str, str],
    *,
    allow_resample: bool,
) -> tuple[InputRef, np.ndarray]:
    """A pending QA input for ``sample`` in the target lead order and rate. Not yet verified."""
    if sample.units != target.units:
        raise GeneratedInputError(f"{sample.record_id}: units {sample.units} are not {target.units}")
    channels = normalize_channels(sample.channels)
    missing = set(target.channels) - set(channels)
    if missing:
        raise GeneratedInputError(f"{sample.record_id}: missing leads {sorted(missing)}")
    if not sample.source_record_ids:
        raise GeneratedInputError(
            f"{sample.record_id}: no conditioning source records; its ancestry is unresolved"
        )
    unknown = [r for r in sample.source_record_ids if r not in subject_of]
    if unknown:
        raise GeneratedInputError(f"{sample.record_id}: source records {unknown[:3]} are not in the index")
    x = np.asarray(sample.load(), dtype=np.float64)[[channels.index(lead) for lead in target.channels]]
    resampling = None
    if not np.isclose(sample.sample_rate, target.sample_rate):
        if not allow_resample:
            raise GeneratedInputError(
                f"{sample.record_id}: {sample.sample_rate} Hz is not {target.sample_rate} Hz; "
                "set allow_resample to "
                "convert explicitly"
            )
        ratio = Fraction(str(target.sample_rate)) / Fraction(str(sample.sample_rate))
        x = resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)
        resampling = {
            "from_hz": sample.sample_rate,
            "to_hz": target.sample_rate,
            "method": "polyphase",
            "up": ratio.numerator,
            "down": ratio.denominator,
        }
    if x.shape[-1] != target.num_samples:
        raise GeneratedInputError(
            f"{sample.record_id}: {x.shape[-1]} samples, the target needs {target.num_samples}"
        )
    x32 = x.astype(np.float32)
    ref = InputRef(
        waveform_id=sample.record_id,
        waveform_sha256=waveform_sha256(x32),
        signal=target,
        origin="generated",
        source_record_ids=sample.source_record_ids,
        root_record_ids=sample.source_record_ids,
        root_subject_uids=tuple(subject_of[r] for r in sample.source_record_ids),
        dataset=sample.generator,
        lineage={
            "run_id": sample.run_id,
            "generator": sample.generator,
            "seed": sample.seed,
            "bundle_status": sample.status,
            "review_status": "pending",
            "requested_controls": sample.conditioning,
            "requested_labels": list(sample.requested_labels),
            "resampling": resampling,
            "channel_aliases": {a: b for a, b in zip(sample.channels, channels, strict=True) if a != b},
            "generator_cohort": GENERATOR_COHORTS.get(
                sample.generator,
                {"training_cohort": "unknown", "overlap_with_external_benchmarks": "unknown"},
            ),
        },
    )
    return ref, x32


def verify_generated(
    ref: InputRef,
    x: np.ndarray,
    extractor: EvidenceExtractor,
    required: Sequence[str],
    *,
    min_confidence: float = 0.8,
    amplitude_mv: tuple[float, float] = (0.2, 10.0),
) -> tuple[InputRef, list[Evidence], dict[str, Any]]:
    """Measure a generated input; mark it verified only for usable, confident ``required`` attributes."""
    evidence = extractor.extract(x, ref)
    by_name = {e.attribute: e for e in evidence if e.attribute in required or e.attribute == "amplitude_mv"}
    reasons = []
    for attribute in required:
        item = by_name.get(attribute)
        if item is None or not item.usable:
            reasons.append(f"unmeasured:{attribute}")
        elif item.confidence is not None and item.confidence < min_confidence:
            reasons.append(f"low_confidence:{attribute}")
    amplitude = by_name.get("amplitude_mv")
    if amplitude is None or not amplitude.usable or not amplitude_mv[0] <= amplitude.value <= amplitude_mv[1]:
        reasons.append("implausible_amplitude")
    review = {
        "status": "verified" if not reasons else "rejected",
        "verified_attributes": [] if reasons else list(required),
        "reasons": reasons,
        "checks": ["finite@1", "amplitude_plausible@1", *(f"usable:{a}" for a in required)],
        "clinical_validation": "not_performed",
    }
    lineage = {**ref.lineage, "review_status": review["status"], "review": review}
    verified = InputRef(
        **{**{name: getattr(ref, name) for name in ref.__dataclass_fields__}, "lineage": lineage}
    )
    return verified, evidence, review
