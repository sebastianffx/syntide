"""Apply individual electrocardiogram augmentation methods.

Each configured operation can be evaluated separately. Time stretching changes
heart rate and interval lengths; amplitude scaling changes voltage
measurements;
time masking removes one interval; lead dropout removes selected leads; and
mixup blends two recordings and their labels with the same weight.

Operations run in that order. Mixup uses the partner's original recording.
All random choices are saved in the generation record, so ``apply`` reproduces
the same result from the source recordings and recorded settings.
"""

from dataclasses import dataclass

import numpy as np

GENERATOR = "ecg_aug"
VERSION = "1"
OPS = ("stretch", "scale", "time_mask_s", "lead_dropout", "mixup_alpha")


@dataclass
class EcgAugConfig:
    """Settings for independently enabled signal augmentation operations."""

    stretch: list[float] | None = (
        None  # [low, high] time-axis speed factor; > 1 = faster heart rate
    )
    scale: list[float] | None = (
        None  # [low, high] amplitude factor, the same for every lead
    )
    time_mask_s: list[float] | None = (
        None  # [low, high] duration of one flat window, seconds
    )
    lead_dropout: list[int] | None = (
        None  # [low, high] number of leads set to 0 mV
    )
    mixup_alpha: float | None = None  # lambda ~ Beta(alpha, alpha)


def enabled(cfg: EcgAugConfig) -> list[str]:
    return [op for op in OPS if getattr(cfg, op) is not None]


def check(cfg: EcgAugConfig, n_leads: int = 12) -> None:
    """Reject disabled configurations and invalid augmentation ranges."""
    if not enabled(cfg):
        raise ValueError(f"an ecg_aug arm must switch on at least one of {OPS}")
    for op in ("stretch", "scale", "time_mask_s", "lead_dropout"):
        value = getattr(cfg, op)
        if value is not None and (
            len(value) != 2 or not 0 < value[0] <= value[1]
        ):
            raise ValueError(
                f"ecg_aug.{op} must be [low, high] with 0 < low <= "
                f"high, got {value}"
            )
    if cfg.lead_dropout is not None and cfg.lead_dropout[1] >= n_leads:
        raise ValueError(
            f"ecg_aug.lead_dropout must keep at least one of {n_leads} leads"
        )
    if cfg.mixup_alpha is not None and cfg.mixup_alpha <= 0:
        raise ValueError(
            f"ecg_aug.mixup_alpha must be positive, got {cfg.mixup_alpha}"
        )


def label_effect(controls: dict) -> dict[str, str]:
    diagnosis = "mixes" if "mixup_lambda" in controls else "preserves"
    return {"diagnosis": diagnosis, "signal_quality": "invalidates"}


def _log_uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))


def draw_controls(
    rng: np.random.Generator,
    cfg: EcgAugConfig,
    n_leads: int,
    n_samples: int,
    fs: int,
    n_parents: int,
) -> dict:
    """Draw and record the settings for one augmented recording."""
    controls: dict = {}
    if cfg.stretch is not None:
        controls["stretch"] = _log_uniform(rng, cfg.stretch)
        controls["stretch_offset"] = int(rng.integers(n_samples))
    if cfg.scale is not None:
        controls["scale"] = _log_uniform(rng, cfg.scale)
    if cfg.time_mask_s is not None:
        width = min(round(rng.uniform(*cfg.time_mask_s) * fs), n_samples)
        controls["time_mask"] = [
            int(rng.integers(n_samples - width + 1)),
            int(width),
        ]
    if cfg.lead_dropout is not None:
        k = int(rng.integers(cfg.lead_dropout[0], cfg.lead_dropout[1] + 1))
        controls["lead_dropout"] = sorted(
            int(i) for i in rng.choice(n_leads, k, replace=False)
        )
    if cfg.mixup_alpha is not None:
        controls["mixup_lambda"] = float(
            rng.beta(cfg.mixup_alpha, cfg.mixup_alpha)
        )
        controls["mixup_partner"] = int(rng.integers(n_parents))
    return controls


def apply(
    x: np.ndarray, controls: dict, partner_x: np.ndarray | None = None
) -> np.ndarray:
    """Augment a recording using only the supplied settings.

    Args:
        x: Source signal with shape (leads, samples), in millivolts.
        controls: Operation settings returned by ``draw_controls``.
        partner_x: Original partner signal when mixup is enabled.

    Returns:
        The augmented signal, with the same shape as ``x``.
    """
    out = x.astype(np.float64)
    n_samples = out.shape[-1]
    if "stretch" in controls:
        # Read the record at ``stretch`` samples per output sample, linearly
        # interpolated, wrapping around
        # the end as the standard circular shift does.
        positions = (
            controls["stretch_offset"]
            + np.arange(n_samples) * controls["stretch"]
        )
        left = np.floor(positions)
        weight = positions - left
        left = left.astype(np.int64) % n_samples
        out = (
            out[:, left] * (1 - weight)
            + out[:, (left + 1) % n_samples] * weight
        )
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


def mix_labels(
    y: np.ndarray, partner_y: np.ndarray, controls: dict
) -> np.ndarray:
    """Blend labels with the mixup weight, or return them unchanged."""
    if "mixup_lambda" not in controls:
        return y
    lam = controls["mixup_lambda"]
    return (
        lam * y.astype(np.float64) + (1 - lam) * partner_y.astype(np.float64)
    ).astype(np.float32)
