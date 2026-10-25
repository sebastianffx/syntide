"""A1's standard augmentation, ``per_lead_aug@1``: what common practice does.

Each lead is treated independently: per-lead gain, Gaussian noise and sinusoidal baseline wander,
plus one circular time shift shared by all leads. Real artifacts arise at electrodes and project
into several leads at once (the editor, arms A2/A3, models that), so this family is deliberately
not physical.

The controls are drawn from the recipe's random stream and recorded in the ledger; the noise
waveform and the per-lead wander phases come only from ``seed_key``, so a ledger row regenerates
its signal bit-exactly with ``apply``.
"""

from dataclasses import dataclass, field

import numpy as np

GENERATOR = "per_lead_aug"
VERSION = "1"
LABEL_EFFECT = {"diagnosis": "preserves", "signal_quality": "invalidates"}


@dataclass
class GenericAugConfig:
    """A1's envelope: each control is drawn uniformly from its [low, high] range."""

    gain: list[float] = field(default_factory=lambda: [0.9, 1.1])
    max_shift_s: float = 0.5
    noise_rms_mv: list[float] = field(default_factory=lambda: [0.0, 0.1])
    wander_mv: list[float] = field(default_factory=lambda: [0.0, 3.5])
    wander_hz: list[float] = field(default_factory=lambda: [0.05, 0.7])


def draw_controls(rng: np.random.Generator, cfg: GenericAugConfig, n_leads: int, fs: int) -> dict:
    max_shift = round(cfg.max_shift_s * fs)
    return {
        "gain": [float(g) for g in rng.uniform(*cfg.gain, size=n_leads)],
        "shift_samples": int(rng.integers(-max_shift, max_shift + 1)),
        "noise_rms_mv": float(rng.uniform(*cfg.noise_rms_mv)),
        "wander_mv": float(rng.uniform(*cfg.wander_mv)),
        "wander_hz": float(rng.uniform(*cfg.wander_hz)),
    }


def apply(x: np.ndarray, controls: dict, seed_key: int, fs: int) -> np.ndarray:
    """Augment one ECG ``x`` (leads, T) in mV. Deterministic in (x, controls, seed_key, fs)."""
    n_leads, n_samples = x.shape
    waveform = np.random.default_rng(seed_key)
    phase = waveform.uniform(0.0, 2 * np.pi, size=(n_leads, 1))
    noise = waveform.standard_normal((n_leads, n_samples))
    t = np.arange(n_samples) / fs
    out = np.roll(x.astype(np.float64), controls["shift_samples"], axis=1)
    out *= np.asarray(controls["gain"])[:, None]
    out += controls["wander_mv"] * np.sin(2 * np.pi * controls["wander_hz"] * t + phase)
    out += controls["noise_rms_mv"] * noise
    return out.astype(np.float32)
