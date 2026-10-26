"""Apply standard augmentation independently to each recording lead.

Each lead receives a gain adjustment, Gaussian noise, and sinusoidal baseline
drift. All leads share one circular time shift. These operations do not model
the way an artifact at one electrode affects several leads.

The generation record stores every setting. The random seed determines the
noise and drift phases, allowing ``apply`` to reproduce the signal exactly.
"""

from dataclasses import dataclass, field

import numpy as np

GENERATOR = "per_lead_aug"
VERSION = "1"
LABEL_EFFECT = {"diagnosis": "preserves", "signal_quality": "invalidates"}


@dataclass
class GenericAugConfig:
    """Uniform ranges for the standard per-lead augmentation settings."""

    gain: list[float] = field(default_factory=lambda: [0.9, 1.1])
    max_shift_s: float = 0.5
    noise_rms_mv: list[float] = field(default_factory=lambda: [0.0, 0.1])
    wander_mv: list[float] = field(default_factory=lambda: [0.0, 3.5])
    wander_hz: list[float] = field(default_factory=lambda: [0.05, 0.7])


def draw_controls(
    rng: np.random.Generator, cfg: GenericAugConfig, n_leads: int, fs: int
) -> dict:
    max_shift = round(cfg.max_shift_s * fs)
    return {
        "gain": [float(g) for g in rng.uniform(*cfg.gain, size=n_leads)],
        "shift_samples": int(rng.integers(-max_shift, max_shift + 1)),
        "noise_rms_mv": float(rng.uniform(*cfg.noise_rms_mv)),
        "wander_mv": float(rng.uniform(*cfg.wander_mv)),
        "wander_hz": float(rng.uniform(*cfg.wander_hz)),
    }


def apply(x: np.ndarray, controls: dict, seed_key: int, fs: int) -> np.ndarray:
    """Apply reproducible per-lead augmentation to one recording.

    Args:
        x: Source signal with shape (leads, samples), in millivolts.
        controls: Recorded gains, noise amplitudes, drift, and time shift.
        seed_key: Seed for noise and baseline-drift phases.
        fs: Sampling frequency in hertz.

    Returns:
        The augmented signal, with the same shape as ``x``.
    """
    n_leads, n_samples = x.shape
    waveform = np.random.default_rng(seed_key)
    phase = waveform.uniform(0.0, 2 * np.pi, size=(n_leads, 1))
    noise = waveform.standard_normal((n_leads, n_samples))
    t = np.arange(n_samples) / fs
    out = np.roll(x.astype(np.float64), controls["shift_samples"], axis=1)
    out *= np.asarray(controls["gain"])[:, None]
    out += controls["wander_mv"] * np.sin(
        2 * np.pi * controls["wander_hz"] * t + phase
    )
    out += controls["noise_rms_mv"] * noise
    return out.astype(np.float32)
