"""Reconstruct chest leads using the published SIMVEA augmentation method.

The method models small differences in chest-electrode placement by replacing
a lead with a weighted sum of two neighboring leads. It follows Lim, Lee,
Jang, and Joo, Biomedical Engineering Letters 15:377-388 (2025),
doi:10.1007/s13534-024-00455-3.

Only chest leads V2 through V5 can be replaced. Each has a replacement
probability of one half. The lead pair and its weights are drawn independently
for each replacement from the choices in the paper. The scale ``alpha`` ranges
from 0.1 to 1.0 and the mixing weight ``beta`` from 0.0 to 2.0, both in steps
of 0.1. The replacement equals ``alpha * (beta * first + (2 - beta) *
second)``.
Setting ``beta`` to one gives the paper's unweighted variant.

New leads always use original recorded leads. Every setting is recorded, so
``apply`` reproduces the augmented signal exactly. The trainer keeps values in
millivolts, so the scale also changes the replacement lead's amplitude.
"""

from dataclasses import dataclass, field

import numpy as np

from syntide.data.schema import LEADS

GENERATOR = "simvea"
VERSION = "1"
# A chest electrode's position changes which leads look like which; the
# diagnosis stays (the paper's premise).
# The PTB-XL signal-quality flags name leads, and a mixed lead carries its
# neighbours' noise.
LABEL_EFFECT = {"diagnosis": "preserves", "signal_quality": "invalidates"}
# Fig. 1c: the pairs each replaced lead can be built from. * = the pair
# straddles the lead's angle exactly.
PAIRS = {
    "V2": (("V1", "V4"), ("V1", "V3")),  # (V1, V4)*
    "V3": (("V2", "V4"), ("V1", "V5")),  # both *
    "V4": (("V3", "V5"), ("V2", "V5"), ("V1", "V6")),  # (V2, V5)*, (V1, V6)*
    "V5": (("V4", "V6"), ("V3", "V6")),  # (V4, V6)*
}


@dataclass
class SimveaConfig:
    """Settings for random chest-lead reconstruction."""

    alpha: list[float] = field(
        default_factory=lambda: [0.1, 1.0]
    )  # [low, high] scale of the new lead
    beta: list[float] = field(
        default_factory=lambda: [0.0, 2.0]
    )  # [low, high] weight of the pair's 1st lead
    # alpha and beta are drawn uniformly from low, low + step, ..., high
    step: float = 0.1
    lead_p: float = 0.5  # chance that each of V2-V5 is replaced


def check(cfg: SimveaConfig, n_leads: int = len(LEADS)) -> None:
    """Check reconstruction settings and require twelve standard leads."""
    if n_leads != len(LEADS):
        raise ValueError(
            f"simvea mixes chest leads by name and needs the "
            f"{len(LEADS)} leads {LEADS}"
        )
    if cfg.step <= 0:
        raise ValueError(f"simvea.step must be positive, got {cfg.step}")
    alpha, beta = cfg.alpha, cfg.beta
    if len(alpha) != 2 or not 0 < alpha[0] <= alpha[1]:
        raise ValueError(
            f"simvea.alpha must be [low, high] with 0 < low <= high, "
            f"got {alpha}"
        )
    if len(beta) != 2 or not 0 <= beta[0] <= beta[1] <= 2:
        raise ValueError(
            f"simvea.beta must be [low, high] with 0 <= low <= high "
            f"<= 2, got {beta}"
        )
    for name, (low, high) in (("alpha", alpha), ("beta", beta)):
        if not np.isclose(
            (high - low) / cfg.step, round((high - low) / cfg.step)
        ):
            raise ValueError(
                f"simvea.{name} {[low, high]} is not a whole number "
                f"of steps of {cfg.step}"
            )
    if not 0 < cfg.lead_p <= 1:
        raise ValueError(f"simvea.lead_p must be in (0, 1], got {cfg.lead_p}")


def grid(bounds: list[float], step: float) -> np.ndarray:
    """Return evenly spaced values with floating-point errors corrected."""
    count = round((bounds[1] - bounds[0]) / step) + 1
    return np.round(bounds[0] + step * np.arange(count), 10)


def draw_controls(rng: np.random.Generator, cfg: SimveaConfig) -> dict:
    """Draw lead pairs, scales, and weights for one augmented input.

    Every lead consumes the same four draws whether or not it is replaced, so
    each lead's draw is fixed by
    the slot alone.
    """
    alphas, betas = grid(cfg.alpha, cfg.step), grid(cfg.beta, cfg.step)
    replaced = []
    for lead, pairs in PAIRS.items():
        replace = rng.random() < cfg.lead_p
        pair = pairs[int(rng.integers(len(pairs)))]
        alpha = float(alphas[rng.integers(len(alphas))])
        beta = float(betas[rng.integers(len(betas))])
        if replace:
            replaced.append(
                {"lead": lead, "pair": list(pair), "alpha": alpha, "beta": beta}
            )
    return {"leads": replaced}


def apply(x: np.ndarray, controls: dict) -> np.ndarray:
    """Reconstruct selected chest leads from the recorded weighted lead pairs.

    Args:
        x: Source signal with shape (12, samples), in millivolts.
        controls: Recorded lead pairs, scales, and mixing weights.

    Returns:
        The signal with selected leads replaced; other leads remain unchanged.
    """
    measured = x.astype(np.float64)
    out = measured.copy()
    for edit in controls["leads"]:
        i, j, z = (LEADS.index(name) for name in (edit["lead"], *edit["pair"]))
        beta = edit["beta"]
        out[i] = edit["alpha"] * (beta * measured[j] + (2 - beta) * measured[z])
    return out.astype(np.float32)
