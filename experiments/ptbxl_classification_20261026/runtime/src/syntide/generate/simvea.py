"""``simvea@1``: SIMVEA_h, precordial lead vector addition, for the augmentation sweep's arm T6.

Lim, Lee, Jang and Joo, "Specialized ECG data augmentation method: leveraging precordial lead positional
variability", Biomedical Engineering Letters 15:377-388 (2025), doi:10.1007/s13534-024-00455-3.

Each chest lead looks at the heart from its own angle on the horizontal plane (Fig. 1a): V1 120°, V2 90°,
V3 75°, V4 60°, V5 30°, V6 0°. Chest electrodes are placed by palpating ribs, so their positions vary
between recordings of one patient while the diagnosis does not. SIMVEA_h imitates a misplaced electrode by
replacing lead i with a vector sum of two leads on either side of it (Table 2):

    X^(i) = alpha * (beta * X^(j) + (2 - beta) * X^(z))

``beta`` sets how close the electrode sits to each neighbour; ``beta = 1`` is plain SIMVEA. ``alpha`` scales
the new lead. The pairs are 30-120° apart, so with beta = 1 their sum is 1.0-1.9 times as long as one lead,
and alpha 0.1-1.0 gives a new lead 0.1-1.9 times as long as a measured one.

Settings, all from the paper:

- only V2-V5 are replaced, since V1 and V6 have no neighbour on one side (§2.2);
- the pair comes from Fig. 1c, drawn uniformly per lead (Supplementary Appendix 1). Starred pairs straddle
  the lead's angle exactly (V2 at 90° = V1 at 120° with V4 at 60°); the others are the nearest neighbours;
- each of V2-V5 is replaced independently with probability 1/2 (§2.4: "a random number was generated
  internally for each lead. If this value exceeded 0.5, the augmented lead was used"). So 1 in 16 augmented
  slots is served unchanged, as in the paper;
- alpha is uniform on {0.1, 0.2, ..., 1.0} and beta on {0.0, 0.1, ..., 2.0} (Supplementary Appendix 1:
  "in increments of 0.1"). Because beta and 2 - beta have the same distribution, which lead of a pair is j
  does not matter;
- a new lead is a sum of measured leads, never of a lead replaced in the same draw.

Where the paper is silent, this module chooses:

- alpha, beta and the pair are drawn for each replaced lead, not once per ECG. The supplement describes
  generating "each new augmented lead"; github.com/rpopgmp/TimofeevAV (code/simvea.py) does the same;
- the paper augments before band-pass filtering and normalization (Fig. 2). Filtering is linear and the same
  for every lead, so it commutes with a sum of leads. Its normalization is not described. Our trainer feeds
  millivolts unchanged, so alpha changes the replaced lead's amplitude, as in Supplementary Fig. 1.

The paper's PTB-XL result is a null: with random alpha and beta, SIMVEA_h scored NORM AUROC 0.9571 and MI
0.9453 against a baseline of 0.9577 and 0.9460 (Table 3).

Every control is drawn from the stream's random stream and recorded in the ledger. ``apply`` uses no other
randomness, so a ledger row regenerates its training input bit-exactly from the parent.
"""

from dataclasses import dataclass, field

import numpy as np

from syntide.data.schema import LEADS

GENERATOR = "simvea"
VERSION = "1"
# A chest electrode's position changes which leads look like which; the diagnosis stays (the paper's premise).
# The PTB-XL signal-quality flags name leads, and a mixed lead carries its neighbours' noise.
LABEL_EFFECT = {"diagnosis": "preserves", "signal_quality": "invalidates"}
# Fig. 1c: the pairs each replaced lead can be built from. * = the pair straddles the lead's angle exactly.
PAIRS = {
    "V2": (("V1", "V4"), ("V1", "V3")),  # (V1, V4)*
    "V3": (("V2", "V4"), ("V1", "V5")),  # both *
    "V4": (("V3", "V5"), ("V2", "V5"), ("V1", "V6")),  # (V2, V5)*, (V1, V6)*
    "V5": (("V4", "V6"), ("V3", "V6")),  # (V4, V6)*
}


@dataclass
class SimveaConfig:
    """The paper's SIMVEA_h with random hyperparameters. ``beta: [1.0, 1.0]`` gives plain SIMVEA."""

    alpha: list[float] = field(default_factory=lambda: [0.1, 1.0])  # [low, high] scale of the new lead
    beta: list[float] = field(default_factory=lambda: [0.0, 2.0])  # [low, high] weight of the pair's 1st lead
    step: float = 0.1  # alpha and beta are drawn uniformly from low, low + step, ..., high
    lead_p: float = 0.5  # chance that each of V2-V5 is replaced


def check(cfg: SimveaConfig, n_leads: int = len(LEADS)) -> None:
    """Refuse a config the formula cannot follow, or ECGs without the 12 standard leads."""
    if n_leads != len(LEADS):
        raise ValueError(f"simvea mixes chest leads by name and needs the {len(LEADS)} leads {LEADS}")
    if cfg.step <= 0:
        raise ValueError(f"simvea.step must be positive, got {cfg.step}")
    alpha, beta = cfg.alpha, cfg.beta
    if len(alpha) != 2 or not 0 < alpha[0] <= alpha[1]:
        raise ValueError(f"simvea.alpha must be [low, high] with 0 < low <= high, got {alpha}")
    if len(beta) != 2 or not 0 <= beta[0] <= beta[1] <= 2:
        raise ValueError(f"simvea.beta must be [low, high] with 0 <= low <= high <= 2, got {beta}")
    for name, (low, high) in (("alpha", alpha), ("beta", beta)):
        if not np.isclose((high - low) / cfg.step, round((high - low) / cfg.step)):
            raise ValueError(f"simvea.{name} {[low, high]} is not a whole number of steps of {cfg.step}")
    if not 0 < cfg.lead_p <= 1:
        raise ValueError(f"simvea.lead_p must be in (0, 1], got {cfg.lead_p}")


def grid(bounds: list[float], step: float) -> np.ndarray:
    """The values a hyperparameter is drawn from: low, low + step, ..., high (rounding error removed)."""
    count = round((bounds[1] - bounds[0]) / step) + 1
    return np.round(bounds[0] + step * np.arange(count), 10)


def draw_controls(rng: np.random.Generator, cfg: SimveaConfig) -> dict:
    """The replaced leads of one augmented input, each with its pair, alpha and beta.

    Every lead consumes the same four draws whether or not it is replaced, so each lead's draw is fixed by
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
            replaced.append({"lead": lead, "pair": list(pair), "alpha": alpha, "beta": beta})
    return {"leads": replaced}


def apply(x: np.ndarray, controls: dict) -> np.ndarray:
    """Augment one ECG ``x`` (12, T) in mV. Deterministic in (x, controls); other leads are returned as is."""
    measured = x.astype(np.float64)
    out = measured.copy()
    for edit in controls["leads"]:
        i, j, z = (LEADS.index(name) for name in (edit["lead"], *edit["pair"]))
        beta = edit["beta"]
        out[i] = edit["alpha"] * (beta * measured[j] + (2 - beta) * measured[z])
    return out.astype(np.float32)
