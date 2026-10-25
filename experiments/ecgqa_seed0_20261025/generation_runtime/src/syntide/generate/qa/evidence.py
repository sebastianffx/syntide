"""Verified ECG attributes: versioned, deterministic extractors and a per-signal evidence cache.

Extractors (``name@version``; each version is part of every evidence id it produces):

- ``ecg.acquisition@1``: duration, sampling rate, channels and which leads carry signal. A lead whose
  robust spread and peak-to-peak are both below ``flat_mv`` is flat (unavailable).
- ``ecg.artifact_descriptors@1``: the existing ``ecg.electrode_artifacts@1`` descriptors per lead
  (baseline wander, high-frequency noise, the loudest burst) and the leads where they are strongest.
  Their documented limitation is carried in ``observability``: physiology leaks into both bands, so a
  natural ECG measures above a pure artifact. They measure the recording; they do not diagnose noise.
- ``ecg.beats@1``: R-peaks on a 5-15 Hz energy envelope pooled over the available leads, then the
  heart rate, the beat count and RR regularity. Every lead is also detected on its own, and the median
  per-lead agreement with the pooled peaks is the confidence. Fewer than ``min_beats`` beats, an
  agreement below ``min_agreement`` or an implausible RR makes the rate ``unknown``: inconclusive is
  never a guess. Regularity has an explicit grey zone (``unknown``) between its two bands. At 100 Hz
  this extractor does **not** delineate P, QRS or T waves; interval questions need a sidecar.
- ``ptbxl.annotations@1``: PTB-XL's existing signal-quality flags and diagnostic superclasses, as
  ``annotation`` evidence of the unmodified source recording only. A superclass never stands in for a
  morphology finding.
- ``measurement.sidecar@1``: interval and axis measurements from a PTB-XL+ style CSV. The file sha256,
  the column map and the external extractor's name and version are in every evidence id, so a changed
  sidecar or mapping makes new evidence. Sidecar values describe the original recording only.

``EvidenceCache`` stores the evidence list of one exact (waveform bytes, signal spec, record lineage,
extractor settings, annotations, sidecar) combination once; every question about that signal reuses
it, and any change to one of those inputs is a cache miss.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

from syntide.data.qa import Evidence, InputRef
from syntide.data.schema import ARTIFACT_TYPES
from syntide.explore.plans import content_id
from syntide.generate import descriptors

ACQUISITION = "ecg.acquisition@1"
ARTIFACT = "ecg.artifact_descriptors@1"
BEATS = "ecg.beats@1"
ANNOTATIONS = "ptbxl.annotations@1"
SIDECAR = "measurement.sidecar@1"
LOCALIZATION_RULE = "localization_margin@1"
REGULARITY_RULE = "rr_regularity_bands@1"
SUPERCLASS_NAMES = {
    "CD": "conduction disturbance",
    "HYP": "hypertrophy",
    "MI": "myocardial infarction",
    "NORM": "normal ecg",
    "STTC": "st/t change",
}


@dataclass
class BeatConfig:
    band_hz: list[float] = field(default_factory=lambda: [5.0, 15.0])
    integrate_s: float = 0.1  # moving-average window of the energy envelope
    refractory_s: float = 0.25  # no two beats closer than this (240 bpm)
    height: float = 0.3  # peak threshold, as a fraction of the envelope's 98th percentile
    match_s: float = 0.06  # a lead's peak matches a pooled peak within this
    min_beats: int = 4
    min_agreement: float = 0.6
    rr_range_s: list[float] = field(default_factory=lambda: [0.25, 2.5])
    regular_cv: float = 0.08  # RR coefficient of variation at or below: regular
    irregular_cv: float = 0.15  # at or above: irregular; between the two: unknown


@dataclass
class SidecarConfig:
    """A PTB-XL+ style measurement file. ``columns`` maps attribute -> [column, units]."""

    path: str = ""
    sha256: str = ""  # required: the file is refused if its bytes differ
    id_column: str = "ecg_id"
    record_id_format: str = "ptbxl-{id}"
    extractor: str = ""  # e.g. GE 12SL or Uni-G, as published with the sidecar
    extractor_version: str = ""
    columns: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ExtractorConfig:
    flat_mv: float = 0.005
    localization_margin: float = 1.25  # strongest lead must exceed the runner-up by this factor
    localization_floor_mv: float = 0.05  # ... and measure at least this
    beats: BeatConfig = field(default_factory=BeatConfig)
    annotations: bool = True
    sidecar: SidecarConfig | None = None

    def sha256(self) -> str:
        return content_id(asdict(self))


class EvidenceError(ValueError):
    """An extractor input is malformed, or a sidecar does not match its declared checksum."""


class Sidecar:
    """A checksummed measurement file, indexed by record id."""

    def __init__(self, config: SidecarConfig) -> None:
        path = Path(config.path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not config.sha256 or digest != config.sha256:
            raise EvidenceError(
                f"{path}: sha256 {digest[:12]} does not match the declared {config.sha256[:12]}"
            )
        if not config.extractor or not config.extractor_version or not config.columns:
            raise EvidenceError("a sidecar needs its extractor name, extractor version and column map")
        self.config = config
        self.rows: dict[str, dict[str, str]] = {}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                raw = str(row[config.id_column]).strip()
                key = raw if not raw.replace(".", "", 1).isdigit() else str(int(float(raw)))
                self.rows[config.record_id_format.format(id=key)] = row
        missing = {column for column, _ in config.columns.values()} - set(next(iter(self.rows.values()), {}))
        if self.rows and missing:
            raise EvidenceError(f"{path} has no columns {sorted(missing)}")


class EvidenceExtractor:
    """Measure one waveform into ``Evidence``; real recordings may add annotations and sidecar values."""

    def __init__(self, config: ExtractorConfig | None = None, cache: EvidenceCache | None = None) -> None:
        self.config = config or ExtractorConfig()
        self.sidecar = (
            Sidecar(self.config.sidecar) if self.config.sidecar and self.config.sidecar.path else None
        )
        self.cache = cache
        self.config_sha256 = self.config.sha256()

    def extract(
        self, x: np.ndarray, ref: InputRef, annotations: Mapping[str, Any] | None = None
    ) -> list[Evidence]:
        """Every fact this extractor set can state about ``x`` (channels, time), in mV."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (len(ref.signal.channels), ref.signal.num_samples):
            raise EvidenceError(f"{ref.waveform_id}: shape {x.shape} does not match its signal spec")
        if not np.isfinite(x).all():
            raise EvidenceError(f"{ref.waveform_id}: the waveform is not finite")
        if waveform_sha256(x.astype(np.float32)) != ref.waveform_sha256:
            raise EvidenceError(f"{ref.waveform_id}: the waveform bytes do not match waveform_sha256")
        annotations = dict(annotations or {}) if ref.origin == "real" and self.config.annotations else {}
        key = content_id(
            {
                "input": ref.to_json(),
                "extractors": [ACQUISITION, ARTIFACT, BEATS, ANNOTATIONS, SIDECAR],
                "config_sha256": self.config_sha256,
                "annotations": annotations,
                "sidecar_sha256": self.config.sidecar.sha256 if self.sidecar else None,
            }
        )
        if self.cache is not None and (cached := self.cache.get(key)) is not None:
            return cached
        out = [*self._acquisition(x, ref), *self._artifacts(x, ref), *self._beats(x, ref)]
        if annotations:
            out.extend(self._annotations(ref, annotations))
        if self.sidecar is not None and ref.origin == "real":
            out.extend(self._sidecar(ref))
        if self.cache is not None:
            self.cache.put(key, out)
        return out

    def _make(self, ref: InputRef, method: str, attribute: str, value: Any, **kw: Any) -> Evidence:
        kw.setdefault("status", "present")
        kw.setdefault("kind", "measured")
        kw.setdefault("units", None)
        kw.setdefault("leads", ())
        kw.setdefault("interval_s", None)
        kw.setdefault("confidence", 1.0)
        kw.setdefault("observability", "direct measurement of the stored waveform")
        return Evidence(
            waveform_sha256=ref.waveform_sha256,
            signal=ref.signal,
            record_id=ref.waveform_id,
            root_record_ids=ref.root_record_ids,
            root_subject_uids=ref.root_subject_uids,
            attribute=attribute,
            value=value,
            method=method,
            config_sha256=self.config_sha256,
            **kw,
        )

    def available_leads(self, x: np.ndarray, channels: Sequence[str]) -> list[str]:
        spread = descriptors.robust_sd(x)
        ptp = np.ptp(x, axis=-1)
        return [
            ch
            for ch, s, p in zip(channels, spread, ptp, strict=True)
            if s >= self.config.flat_mv or p >= 2 * self.config.flat_mv
        ]

    def _acquisition(self, x: np.ndarray, ref: InputRef) -> list[Evidence]:
        spec = ref.signal
        available = self.available_leads(x, spec.channels)
        flat = [ch for ch in spec.channels if ch not in available]
        whole = (0.0, spec.duration_s)
        return [
            self._make(
                ref, ACQUISITION, "duration_s", round(spec.duration_s, 6), units="s", interval_s=whole
            ),
            self._make(ref, ACQUISITION, "sample_rate_hz", spec.sample_rate, units="Hz"),
            self._make(ref, ACQUISITION, "channels", list(spec.channels)),
            self._make(ref, ACQUISITION, "available_leads", available, leads=tuple(available)),
            self._make(ref, ACQUISITION, "flat_leads", flat, leads=tuple(flat)),
            self._amplitude(x, ref, available),
        ]

    def _amplitude(self, x: np.ndarray, ref: InputRef, available: Sequence[str]) -> Evidence:
        """The largest robust peak-to-peak (99.5th - 0.5th percentile) over the available leads."""
        if not available:
            return self._make(
                ref,
                ACQUISITION,
                "amplitude_mv",
                None,
                units="mV",
                status="unobservable",
                confidence=None,
                observability="every lead is flat",
            )
        rows = [ref.signal.channels.index(lead) for lead in available]
        spans = np.percentile(x[rows], 99.5, axis=-1) - np.percentile(x[rows], 0.5, axis=-1)
        k = int(np.argmax(spans))
        return self._make(
            ref, ACQUISITION, "amplitude_mv", round(float(spans[k]), 4), units="mV", leads=(available[k],)
        )

    def _artifacts(self, x: np.ndarray, ref: InputRef) -> list[Evidence]:
        spec, fs = ref.signal, ref.signal.sample_rate
        available = self.available_leads(x, spec.channels)
        observability = (
            "ecg.electrode_artifacts@1 descriptor of the recording; physiology leaks into the band, so it "
            "measures above the artifact alone and is not a noise diagnosis"
        )
        out, wander, noise = [], {}, {}
        for k, lead in enumerate(spec.channels):
            if lead not in available:
                for attribute in ("wander_mv", "hf_rms_mv"):
                    out.append(
                        self._make(
                            ref,
                            ARTIFACT,
                            attribute,
                            None,
                            units="mV",
                            leads=(lead,),
                            status="unobservable",
                            confidence=None,
                            observability="the lead is flat",
                        )
                    )
                continue
            wander[lead] = round(descriptors.wander_mv(x[k], fs), 4)
            noise[lead] = round(descriptors.hf_rms_mv(x[k], fs), 4)
            for attribute, value in (("wander_mv", wander[lead]), ("hf_rms_mv", noise[lead])):
                out.append(
                    self._make(
                        ref,
                        ARTIFACT,
                        attribute,
                        value,
                        units="mV",
                        leads=(lead,),
                        observability=observability,
                    )
                )
        for attribute, values in (("max_wander_lead", wander), ("max_hf_noise_lead", noise)):
            out.append(self._localize(ref, attribute, values))
        if available:
            rows = [spec.channels.index(lead) for lead in available]
            bursts = [(descriptors.burst(x[k], fs), spec.channels[k]) for k in rows]
            found, lead = max(bursts, key=lambda item: item[0]["burst_rms_mv"])
            start = found["burst_onset_s"]
            out.append(
                self._make(
                    ref,
                    ARTIFACT,
                    "burst_rms_mv",
                    round(found["burst_rms_mv"], 4),
                    units="mV",
                    leads=(lead,),
                    interval_s=(round(start, 3), round(start + found["burst_duration_s"], 3)),
                    observability=observability,
                )
            )
        return out

    def _localize(self, ref: InputRef, attribute: str, values: Mapping[str, float]) -> Evidence:
        cfg = self.config
        ranked = sorted(values.items(), key=lambda item: (-item[1], ref.signal.channels.index(item[0])))
        rules = (LOCALIZATION_RULE,)
        if len(ranked) < 2:
            return self._make(ref, ARTIFACT, attribute, None, status="unknown", confidence=None, rules=rules)
        (top, value), (_, second) = ranked[0], ranked[1]
        if value < cfg.localization_floor_mv or value < cfg.localization_margin * second:
            return self._make(
                ref,
                ARTIFACT,
                attribute,
                None,
                status="unknown",
                confidence=None,
                rules=rules,
                observability="no lead is clearly strongest",
            )
        return self._make(
            ref,
            ARTIFACT,
            attribute,
            top,
            leads=(top,),
            rules=rules,
            confidence=round(min(1.0, (value / second - 1) / (cfg.localization_margin - 1) / 2 + 0.5), 4),
            observability="the lead with the largest descriptor, by a margin",
        )

    def _beats(self, x: np.ndarray, ref: InputRef) -> list[Evidence]:
        spec, cfg = ref.signal, self.config.beats
        available = self.available_leads(x, spec.channels)
        rows = [spec.channels.index(lead) for lead in available]
        rules = (REGULARITY_RULE,)
        whole = (0.0, round(spec.duration_s, 6))
        unknown = {
            "status": "unknown",
            "confidence": None,
            "interval_s": whole,
            "leads": tuple(available),
        }
        if not rows:
            return [
                self._make(ref, BEATS, name, None, units=units, **{**unknown, "status": "unobservable"})
                for name, units in (("heart_rate_bpm", "bpm"), ("beat_count", None), ("rr_regularity", None))
            ]
        pooled, per_lead = detect_beats(x[rows], spec.sample_rate, cfg)
        agreement = float(
            np.median([_match_f1(pooled, own, cfg.match_s * spec.sample_rate) for own in per_lead])
        )
        rr = np.diff(pooled) / spec.sample_rate
        plausible = len(rr) > 0 and bool(((rr >= cfg.rr_range_s[0]) & (rr <= cfg.rr_range_s[1])).all())
        conclusive = len(pooled) >= cfg.min_beats and agreement >= cfg.min_agreement and plausible
        reason = (
            f"{len(pooled)} beats, lead agreement {agreement:.2f}, RR "
            f"{'plausible' if plausible else 'implausible'}"
        )
        if not conclusive:
            return [
                self._make(
                    ref, BEATS, name, None, units=units, observability=f"inconclusive: {reason}", **unknown
                )
                for name, units in (("heart_rate_bpm", "bpm"), ("beat_count", None), ("rr_regularity", None))
            ]
        confidence = round(agreement, 4)
        hr = round(60.0 / float(np.median(rr)), 1)
        cv = float(np.std(rr) / np.mean(rr))
        regularity = "regular" if cv <= cfg.regular_cv else "irregular" if cv >= cfg.irregular_cv else None
        common = {
            "interval_s": whole,
            "leads": tuple(available),
            "confidence": confidence,
            "observability": f"R-peak detection pooled over available leads ({reason})",
        }
        return [
            self._make(ref, BEATS, "heart_rate_bpm", hr, units="bpm", **common),
            self._make(ref, BEATS, "beat_count", len(pooled), **common),
            self._make(ref, BEATS, "rr_cv", round(cv, 4), **common),
            self._make(
                ref,
                BEATS,
                "rr_regularity",
                regularity,
                rules=rules,
                **({**common, "status": "unknown", "confidence": None} if regularity is None else common),
            ),
        ]

    def _annotations(self, ref: InputRef, annotations: Mapping[str, Any]) -> list[Evidence]:
        out = []
        source = {"source": "PTB-XL 1.0.3 ptbxl_database.csv", "record_id": ref.waveform_id}
        flags = set(annotations.get("artifact_types", ()))
        for kind in ARTIFACT_TYPES:
            out.append(
                self._make(
                    ref,
                    ANNOTATIONS,
                    f"annotated_{kind}",
                    kind in flags,
                    kind="annotation",
                    status="present" if kind in flags else "absent",
                    confidence=None,
                    observability="PTB-XL signal-quality annotation of the unmodified recording; coarse, not "
                    "lead- or time-resolved",
                    provenance=source,
                )
            )
        labels = annotations.get("labels")
        if labels is not None:
            out.append(
                self._make(
                    ref,
                    ANNOTATIONS,
                    "diagnostic_superclasses",
                    sorted(labels),
                    kind="annotation",
                    confidence=None,
                    provenance={**source, "aggregation": "SCP diagnostic codes -> superclass"},
                    observability="PTB-XL diagnostic superclass annotation; not a waveform measurement and "
                    "not a "
                    "morphology finding",
                )
            )
        return out

    def _sidecar(self, ref: InputRef) -> list[Evidence]:
        cfg = self.config.sidecar
        row = self.sidecar.rows.get(ref.waveform_id)
        provenance = {
            "sidecar_sha256": cfg.sha256,
            "extractor": cfg.extractor,
            "extractor_version": cfg.extractor_version,
            "column_map_sha256": content_id(cfg.columns),
        }
        out = []
        for attribute, (column, units) in sorted(cfg.columns.items()):
            raw = (row or {}).get(column, "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = None
            ok = value is not None and np.isfinite(value)
            out.append(
                self._make(
                    ref,
                    SIDECAR,
                    attribute,
                    round(value, 3) if ok else None,
                    units=units,
                    kind="annotation",
                    status="present" if ok else "unknown",
                    confidence=None,
                    provenance=provenance,
                    observability=f"{cfg.extractor} {cfg.extractor_version} measurement of the original "
                    "recording",
                )
            )
        return out


def detect_beats(x: np.ndarray, fs: float, cfg: BeatConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    """Pooled R-peak sample indices, and each lead's own peaks, for leads ``x`` (leads, time)."""
    low, high = cfg.band_hz
    high = min(high, 0.45 * fs)
    sos = butter(2, [low, high], btype="bandpass", fs=fs, output="sos")
    band = sosfiltfilt(sos, x, axis=-1)
    energy = np.diff(band, axis=-1, prepend=band[:, :1]) ** 2
    scale = np.percentile(energy, 99, axis=-1, keepdims=True)
    energy = np.minimum(energy / np.where(scale > 0, scale, 1.0), 4.0)
    width = max(1, round(cfg.integrate_s * fs))
    energy = uniform_filter1d(energy, size=width, axis=-1, mode="nearest")
    distance = max(1, round(cfg.refractory_s * fs))

    def peaks(trace: np.ndarray) -> np.ndarray:
        top = np.percentile(trace, 98)
        if top <= 0:
            return np.empty(0, dtype=int)
        found, _ = find_peaks(trace, height=cfg.height * top, distance=distance)
        return found

    return peaks(energy.mean(axis=0)), [peaks(lead) for lead in energy]


def _match_f1(reference: np.ndarray, found: np.ndarray, tolerance: float) -> float:
    if len(reference) == 0 or len(found) == 0:
        return 0.0
    distance = np.abs(reference[:, None] - found[None, :])
    hits = int((distance.min(axis=1) <= tolerance).sum())
    precision = float((distance.min(axis=0) <= tolerance).mean())
    recall = hits / len(reference)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def waveform_sha256(x: np.ndarray) -> str:
    """The identity of a waveform: its float32 bytes, C order, plus its shape."""
    array = np.ascontiguousarray(x, dtype=np.float32)
    digest = hashlib.sha256(f"{array.shape}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def by_attribute(evidence: Sequence[Evidence]) -> dict[tuple[str, tuple[str, ...]], Evidence]:
    """Index evidence by (attribute, leads). Raises if a fact is stated twice."""
    out: dict[tuple[str, tuple[str, ...]], Evidence] = {}
    for item in evidence:
        key = (item.attribute, item.leads if item.attribute in PER_LEAD else ())
        if key in out:
            raise EvidenceError(f"{item.record_id}: {key} is stated twice")
        out[key] = item
    return out


PER_LEAD = frozenset({"wander_mv", "hf_rms_mv"})


class EvidenceCache:
    """Evidence lists on disk, one JSON file per exact extraction input."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> list[Evidence] | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["key"] != key:
            raise EvidenceError(f"{path} holds another extraction")
        self.hits += 1
        return [Evidence.from_json(item) for item in payload["evidence"]]

    def put(self, key: str, evidence: Sequence[Evidence]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "evidence": [item.to_json() for item in evidence]}
        # Several corpus builders can extract the same waveform concurrently.
        # Each writer needs its own staging file before atomic publication.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".partial",
            delete=False,
        ) as stream:
            partial = Path(stream.name)
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
        try:
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
