"""``ecg.electrode_artifact@1.0.0``: the electrode-space artifact editor (arms A2 and A3).

Real artifacts arise at electrodes, not leads (MVE challenge #4). The editor adds one artifact to one
of the nine electrodes (RA, LA, LL, V1-V6) and projects it into the 12 leads with the fixed
Einthoven / Goldberger / Wilson matrix: I = LA-RA, II = LL-RA, III = LL-LA, aVR = RA-(LA+LL)/2, ...,
Vi = Vi - WCT with WCT = (RA+LA+LL)/3. A left-arm artifact therefore shows in every lead except II.

Artifact types (v1): ``baseline_drift`` (a sinusoid), ``static_noise`` (high-frequency noise over the
whole record) and ``burst_noise`` (the same noise in one interval). ``electrodes_problems`` is deferred.
Amplitudes are in descriptor units: every lead's coefficient is at most 1 in magnitude, so the
strongest lead of the projected residual carries the requested amplitude.

A recipe (parent, generator, version, controls, seed_key, gap report) fully determines the edit: the
waveform randomness (noise, phase) comes only from ``seed_key``, so ``edit`` regenerates bit-exactly.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from syntide.data.schema import LEADS
from syntide.generate import descriptors
from syntide.generate.ledger import Recipe

__all__ = ["Recipe"]  # re-exported: a recipe is the editor's unit of provenance

GENERATOR = "ecg.electrode_artifact"
VERSION = "1.0.0"
ELECTRODES = ("RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6")
ARTIFACTS = ("baseline_drift", "static_noise", "burst_noise")
LABEL_EFFECT = {"diagnosis": "preserves", "artifact_type": "determines", "signal_quality": "invalidates"}
# The continuous controls each artifact type uses, besides the categorical artifact_type and electrode.
CONTROLS = {
    "baseline_drift": ("amplitude_mv", "freq_hz"),
    "static_noise": ("amplitude_mv",),
    "burst_noise": ("amplitude_mv", "onset_s", "duration_s"),
}


def _lead_matrix() -> np.ndarray:
    ra, la, ll = (ELECTRODES.index(name) for name in ("RA", "LA", "LL"))
    limb = {
        "I": {la: 1, ra: -1},
        "II": {ll: 1, ra: -1},
        "III": {ll: 1, la: -1},
        "AVR": {ra: 1, la: -0.5, ll: -0.5},
        "AVL": {la: 1, ra: -0.5, ll: -0.5},
        "AVF": {ll: 1, ra: -0.5, la: -0.5},
    }
    matrix = np.zeros((len(LEADS), len(ELECTRODES)))
    for row, lead in enumerate(LEADS):
        weights = limb.get(lead) or {ELECTRODES.index(lead): 1, ra: -1 / 3, la: -1 / 3, ll: -1 / 3}
        for column, weight in weights.items():
            matrix[row, column] = weight
    return matrix


LEAD_FROM_ELECTRODE = _lead_matrix()  # (12 leads, 9 electrodes)


@dataclass
class DriftEnvelope:
    amplitude_mv: list[float] = field(default_factory=lambda: [0.1, 3.5])
    freq_hz: list[float] = field(default_factory=lambda: [0.1, 0.4])


@dataclass
class StaticEnvelope:
    amplitude_mv: list[float] = field(default_factory=lambda: [0.005, 0.1])


@dataclass
class BurstEnvelope:
    amplitude_mv: list[float] = field(default_factory=lambda: [0.01, 0.6])
    duration_s: list[float] = field(default_factory=lambda: [0.5, 3.0])


@dataclass
class Envelope:
    """The prior envelope: the range each control may take. Onset is uniform over the record."""

    baseline_drift: DriftEnvelope = field(default_factory=DriftEnvelope)
    static_noise: StaticEnvelope = field(default_factory=StaticEnvelope)
    burst_noise: BurstEnvelope = field(default_factory=BurstEnvelope)


@dataclass
class EditorConfig:
    """The editor's envelope, its acceptance rules and the targeted policy's smoothing."""

    mv_limit: float = 10.0  # bounds@1: an edit may not push a sample past this (or the parent's own peak)
    alignment_tol: float = 0.15  # alignment@1: relative error allowed between realized and requested severity
    max_retries: int = 5  # further draws for the same parent and slot after a rejection
    type_smoothing: float = 0.1  # A3 type weights are prior / (retained frequency + type_smoothing)
    density_floor: float = 1.0  # A3 severity density is prior / (retained KDE + density_floor x prior)
    envelope: Envelope = field(default_factory=Envelope)


def control_range(envelope: dict, artifact_type: str, name: str, record_s: float) -> tuple[float, float]:
    """The [low, high] a continuous control may take; onset depends on the shortest burst."""
    if name == "onset_s":
        return 0.0, record_s - envelope[artifact_type]["duration_s"][0]
    low, high = envelope[artifact_type][name]
    return float(low), float(high)


@dataclass(frozen=True)
class Candidate:
    x: np.ndarray  # (12, T) float32: the edited ECG, mV
    residual: np.ndarray  # (12, T) float64: exactly what the edit added to each lead
    interval: tuple[float, float]  # seconds the artifact spans
    affected_leads: tuple[str, ...]
    strongest_lead: int  # a lead whose coefficient is +-1, where the requested amplitude appears


def edit(parent_x: np.ndarray, recipe: Recipe, fs: int) -> Candidate:
    """Add the recipe's artifact to one ECG (12, T) in mV."""
    controls = recipe.controls
    weights = LEAD_FROM_ELECTRODE[:, ELECTRODES.index(controls["electrode"])]
    wave, interval = artifact_waveform(
        controls, parent_x.shape[-1], fs, np.random.default_rng(recipe.seed_key)
    )
    residual = np.outer(weights, wave)
    return Candidate(
        x=(parent_x.astype(np.float64) + residual).astype(np.float32),
        residual=residual,
        interval=interval,
        affected_leads=tuple(lead for lead, w in zip(LEADS, weights, strict=True) if w != 0),
        strongest_lead=int(np.argmax(np.abs(weights))),
    )


def artifact_waveform(
    controls: dict[str, Any], n_samples: int, fs: int, waveform: np.random.Generator
) -> tuple[np.ndarray, tuple[float, float]]:
    """The electrode-space artifact (T,) in mV, and the interval it spans in seconds."""
    kind, amplitude = controls["artifact_type"], controls["amplitude_mv"]
    t = np.arange(n_samples) / fs
    whole = (0.0, n_samples / fs)
    if kind == "baseline_drift":
        phase = waveform.uniform(0.0, 2 * np.pi)
        return amplitude * np.sin(2 * np.pi * controls["freq_hz"] * t + phase), whole
    if kind == "static_noise":
        return amplitude * _hf_noise(n_samples, fs, waveform), whole
    if kind == "burst_noise":
        onset, duration = controls["onset_s"], controls["duration_s"]
        gate = (t >= onset) & (t < onset + duration)
        noise = descriptors.highpass(waveform.standard_normal(n_samples), fs)[0]
        noise *= gate / descriptors.robust_sd(noise[gate])  # unit robust RMS inside the burst
        return amplitude * noise, (onset, onset + duration)
    raise ValueError(f"unknown artifact type {kind!r}")


def _hf_noise(n_samples: int, fs: int, waveform: np.random.Generator) -> np.ndarray:
    """Gaussian noise in the descriptors' high-frequency band, with unit robust RMS."""
    noise = descriptors.highpass(waveform.standard_normal(n_samples), fs)[0]
    return noise / descriptors.robust_sd(noise)
