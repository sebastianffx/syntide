"""``ecg_aug@1``: single augmentation techniques, for the augmentation sweep (``ptbxl_aug_sweep``).

A1's ``per_lead_aug@1`` applies four operations at once. This family applies only the operations an arm
switches on, so each technique can be measured alone. Each one is a common ECG augmentation with a
known risk for one diagnostic superclass:

- ``stretch``: resample the time axis by a factor (heart rate and every interval scale with it). Risk:
  conduction disturbances (CD) are defined by interval lengths, such as QRS >= 120 ms.
- ``scale``: multiply every lead by one factor. Risk: hypertrophy (HYP) criteria are voltage thresholds.
- ``time_mask_s``: flatten one window of all leads to 0 mV (cutout).
- ``lead_dropout``: set some leads to 0 mV. Risk: MI is localized to lead groups.
- ``mixup_alpha``: blend the ECG with a random parent and blend the labels by the same weight
  (lambda ~ Beta(alpha, alpha)). The only operation that changes the labels.

The operations run in that order; mixup comes last and blends with the partner's unaugmented ECG.
Every control is drawn from the stream's random stream and recorded in the ledger. ``apply`` uses no
other randomness, so a ledger row regenerates its input bit-exactly from the parent and the partner.
"""

from dataclasses import dataclass

import numpy as np

GENERATOR = "ecg_aug"
VERSION = "1"
OPS = ("stretch", "scale", "time_mask_s", "lead_dropout", "mixup_alpha")


@dataclass
class EcgAugConfig:
    """An operation is on when its field is set. Factor ranges are drawn log-uniformly."""

    stretch: list[float] | None = None  # [low, high] time-axis speed factor; > 1 = faster heart rate
    scale: list[float] | None = None  # [low, high] amplitude factor, the same for every lead
    time_mask_s: list[float] | None = None  # [low, high] duration of one flat window, seconds
    lead_dropout: list[int] | None = None  # [low, high] number of leads set to 0 mV
    mixup_alpha: float | None = None  # lambda ~ Beta(alpha, alpha)


def enabled(cfg: EcgAugConfig) -> list[str]:
    return [op for op in OPS if getattr(cfg, op) is not None]


def check(cfg: EcgAugConfig, n_leads: int = 12) -> None:
    """Refuse a config that switches nothing on or sets an impossible range."""
    if not enabled(cfg):
        raise ValueError(f"an ecg_aug arm must switch on at least one of {OPS}")
    for op in ("stretch", "scale", "time_mask_s", "lead_dropout"):
        value = getattr(cfg, op)
        if value is not None and (len(value) != 2 or not 0 < value[0] <= value[1]):
            raise ValueError(f"ecg_aug.{op} must be [low, high] with 0 < low <= high, got {value}")
    if cfg.lead_dropout is not None and cfg.lead_dropout[1] >= n_leads:
        raise ValueError(f"ecg_aug.lead_dropout must keep at least one of {n_leads} leads")
    if cfg.mixup_alpha is not None and cfg.mixup_alpha <= 0:
        raise ValueError(f"ecg_aug.mixup_alpha must be positive, got {cfg.mixup_alpha}")


def label_effect(controls: dict) -> dict[str, str]:
    diagnosis = "mixes" if "mixup_lambda" in controls else "preserves"
    return {"diagnosis": diagnosis, "signal_quality": "invalidates"}


def _log_uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))


def draw_controls(
    rng: np.random.Generator, cfg: EcgAugConfig, n_leads: int, n_samples: int, fs: int, n_parents: int
) -> dict:
    """The controls of one augmented input. ``mixup_partner`` is a parent index; the stream stores its id."""
    controls: dict = {}
    if cfg.stretch is not None:
        controls["stretch"] = _log_uniform(rng, cfg.stretch)
        controls["stretch_offset"] = int(rng.integers(n_samples))
    if cfg.scale is not None:
        controls["scale"] = _log_uniform(rng, cfg.scale)
    if cfg.time_mask_s is not None:
        width = min(round(rng.uniform(*cfg.time_mask_s) * fs), n_samples)
        controls["time_mask"] = [int(rng.integers(n_samples - width + 1)), int(width)]
    if cfg.lead_dropout is not None:
        k = int(rng.integers(cfg.lead_dropout[0], cfg.lead_dropout[1] + 1))
        controls["lead_dropout"] = sorted(int(i) for i in rng.choice(n_leads, k, replace=False))
    if cfg.mixup_alpha is not None:
        controls["mixup_lambda"] = float(rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
        controls["mixup_partner"] = int(rng.integers(n_parents))
    return controls


def apply(x: np.ndarray, controls: dict, partner_x: np.ndarray | None = None) -> np.ndarray:
    """Augment one ECG ``x`` (leads, T) in mV. Deterministic in (x, controls, partner_x)."""
    out = x.astype(np.float64)
    n_samples = out.shape[-1]
    if "stretch" in controls:
        # Read the record at ``stretch`` samples per output sample, linearly interpolated, wrapping around
        # the end as A1's circular shift does.
        positions = controls["stretch_offset"] + np.arange(n_samples) * controls["stretch"]
        left = np.floor(positions)
        weight = positions - left
        left = left.astype(np.int64) % n_samples
        out = out[:, left] * (1 - weight) + out[:, (left + 1) % n_samples] * weight
    if "scale" in controls:
        out *= controls["scale"]
    if "time_mask" in controls:
        start, width = controls["time_mask"]
        out[:, start : start + width] = 0.0
    if "lead_dropout" in controls:
        out[controls["lead_dropout"]] = 0.0
    if "mixup_lambda" in controls:
        if partner_x is None:
            raise ValueError("a mixup recipe needs its partner's ECG")
        lam = controls["mixup_lambda"]
        out = lam * out + (1 - lam) * partner_x.astype(np.float64)
    return out.astype(np.float32)


def mix_labels(y: np.ndarray, partner_y: np.ndarray, controls: dict) -> np.ndarray:
    """The labels of an augmented input: blended by mixup's lambda, unchanged otherwise."""
    if "mixup_lambda" not in controls:
        return y
    lam = controls["mixup_lambda"]
    return (lam * y.astype(np.float64) + (1 - lam) * partner_y.astype(np.float64)).astype(np.float32)
