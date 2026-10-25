"""Paired contrasts between arms, and the decision rules applied to them. Pure NumPy; no files.

A contrast ``a - b`` compares two arms seed by seed on the same records:

- **estimate**: the mean over seeds of (macro-AUROC of a's seed-s model - b's seed-s model);
- **CI**: a crossed bootstrap. Each draw resamples the slice's patients (every record of a drawn patient
  counts once per draw) and, independently, the seeds. Both arms are scored under the same draw, and every
  contrast on a slice uses the same draws, so contrasts on one split are paired with each other too.

Macro-AUROC skips a class without both a weighted positive and a weighted negative, like
``syntide.eval.stats.macro_auroc``, which it matches exactly at unit weights. Pairing is refused
(``PairingError``) unless the arms share seeds, records and labels, trainer fingerprint, budget, the initial
weights of each seed and, between transform arms that declare the same share of transformed slots, those
slots. A dose arm that declares its own share (``ptbxl_aug_sweep``) pairs with any arm.
"""

import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from syntide.data.schema import SUPERCLASSES
from syntide.eval.runs import Run

SLICES = ("artifact", "clean", "overall")


class PairingError(ValueError):
    """Two arms cannot be compared seed by seed: they differ in something other than the arm."""


def auroc_draws(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted AUROC of scores ``p`` (n,) for labels ``y`` (n,) under each row of ``weights`` (B, n).

    Ties count one half, as in ``sklearn.metrics.roc_auc_score``. A row without a weighted positive and a
    weighted negative gives NaN.
    """
    order = np.argsort(p, kind="mergesort")
    p, y, w = p[order], np.asarray(y, dtype=bool)[order], weights[:, order]
    starts = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])  # first index of each run of tied scores
    pos = np.add.reduceat(np.where(y, w, 0.0), starts, axis=1)
    neg = np.add.reduceat(np.where(y, 0.0, w), starts, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    n_pos, n_neg = pos.sum(axis=1), neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (pos * (neg_below + 0.5 * neg)).sum(axis=1) / (n_pos * n_neg)
    return np.where((n_pos > 0) & (n_neg > 0), auc, np.nan)


def class_aurocs(labels: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-class AUROC (B, K) under each row of ``weights`` (B, n); NaN where a class is undefined."""
    return np.stack([auroc_draws(labels[:, k], probs[:, k], weights) for k in range(labels.shape[1])], axis=1)


def macro(per_class: np.ndarray) -> np.ndarray:
    """Mean over the defined classes of (..., K); NaN only if no class is defined."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(per_class, axis=-1)


def slice_mask(is_artifact: np.ndarray, name: str) -> np.ndarray:
    if name == "artifact":
        return is_artifact.copy()
    if name == "clean":
        return ~is_artifact
    if name == "overall":
        return np.ones_like(is_artifact)
    raise ValueError(f"unknown slice {name!r}; choose one of {', '.join(SLICES)}")


@dataclass(frozen=True)
class UtilityReport:
    """The paired effect of arm ``a`` over arm ``b`` (``<arm>@<regime>``) on one slice of one split."""

    a: str
    b: str
    split: str
    checkpoint: str
    slice: str
    metric: str
    estimate: float  # mean over seeds of the paired difference
    ci_low: float
    ci_high: float
    ci_level: float
    n_boot: int
    seeds: list[int]
    per_seed: list[float]  # the paired difference of each seed, in ``seeds`` order
    mean_a: float
    mean_b: float
    n_records: int
    n_patients: int
    per_class: dict[str, float | None]  # mean over seeds of the per-class difference; None if undefined
    n_positive: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_pairing(a: Sequence[Run], b: Sequence[Run]) -> None:
    """Refuse two arms that differ in anything but the arm. Records and labels are checked when scored."""
    for side in (a, b):
        if not side:
            raise PairingError("a contrast needs at least one run per arm")
        if len({run.key for run in side}) != 1:
            raise PairingError(f"one side of the contrast mixes arms {sorted({run.key for run in side})}")
        if len({run.seed for run in side}) != len(side):
            raise PairingError(f"{side[0].key} lists a seed twice")
    seeds_a, seeds_b = sorted(run.seed for run in a), sorted(run.seed for run in b)
    if seeds_a != seeds_b:
        raise PairingError(f"the arms have different seeds: {seeds_a} vs {seeds_b}")
    runs = [*a, *b]
    if len({run.fingerprint for run in runs}) != 1:
        raise PairingError("the arms were trained by different trainers or trainer configs")
    if len({run.exposure["presentations"] for run in runs}) != 1:
        raise PairingError("the arms saw different numbers of training presentations; budgets differ")
    by_seed = {run.seed: run for run in b}
    for run_a in a:
        run_b = by_seed[run_a.seed]
        if run_a.init_sha256 != run_b.init_sha256:
            raise PairingError(f"seed {run_a.seed} starts from different initial weights in the two arms")
        # Two transform arms that declare the same share of transformed slots must have served the same
        # slots. A0's slots hold unchanged parents, and a dose arm declares its own share: that is the arm.
        transforms = all(
            run.spec.transform_family != "none" and run.spec.transform_fraction > 0 for run in (run_a, run_b)
        )
        same_share = run_a.spec.transform_fraction == run_b.spec.transform_fraction
        if (
            transforms
            and same_share
            and run_a.exposure["transform_slots"] != run_b.exposure["transform_slots"]
        ):
            raise PairingError(f"seed {run_a.seed} has different transform slots in the two arms")


class PairedScorer:
    """Scores runs of one split and checkpoint under shared bootstrap draws, so every contrast is paired.

    ``records`` is any run of the split: every scored run must have exactly its records and labels.
    """

    def __init__(self, records: Run, *, checkpoint: str, n_boot: int, seed: int) -> None:
        self.record_id, self.subject_uid = records.record_id, records.subject_uid
        self.labels, self.is_artifact = records.labels, records.is_artifact
        self.split, self.checkpoint, self.n_boot, self.seed = records.split, checkpoint, n_boot, seed
        self._weights: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self._seed_draws: dict[int, np.ndarray] = {}
        self._scores: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def weights(self, name: str) -> tuple[np.ndarray, np.ndarray, int]:
        """(record mask, record weights (B, n_slice), number of patients) of one slice."""
        if name not in self._weights:
            mask = slice_mask(self.is_artifact, name)
            patients, inverse = np.unique(self.subject_uid[mask], return_inverse=True)
            rng = np.random.default_rng([self.seed, SLICES.index(name)])
            uniform = np.full(len(patients), 1 / len(patients))
            counts = rng.multinomial(len(patients), uniform, size=self.n_boot)
            self._weights[name] = (mask, counts[:, inverse].astype(np.float64), len(patients))
        return self._weights[name]

    def seed_draws(self, n_seeds: int) -> np.ndarray:
        """(B, n_seeds) indices of the seeds each draw uses."""
        if n_seeds not in self._seed_draws:
            rng = np.random.default_rng([self.seed, len(SLICES), n_seeds])
            self._seed_draws[n_seeds] = rng.integers(0, n_seeds, size=(self.n_boot, n_seeds))
        return self._seed_draws[n_seeds]

    def scores(self, run: Run, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-class AUROC of one run on one slice: at unit weights (K,) and under every draw (B, K)."""
        cache = (str(run.run_dir), name)
        if cache not in self._scores:
            same = (
                run.split == self.split
                and np.array_equal(run.record_id, self.record_id)
                and np.array_equal(run.labels, self.labels)
                and np.array_equal(run.is_artifact, self.is_artifact)
                and np.array_equal(run.subject_uid, self.subject_uid)
            )
            if not same:
                raise PairingError(f"{run.run_dir} was scored on other records or labels than the other runs")
            mask, weights, _ = self.weights(name)
            labels, probs = self.labels[mask], run.probs[self.checkpoint][mask]
            point = class_aurocs(labels, probs, np.ones((1, int(mask.sum()))))[0]
            self._scores[cache] = (point, class_aurocs(labels, probs, weights))
        return self._scores[cache]

    def point(self, run: Run, name: str) -> float:
        """The run's macro-AUROC on a slice (equal to ``syntide.eval.stats.macro_auroc``)."""
        return float(macro(self.scores(run, name)[0]))

    def contrast(
        self, a: Sequence[Run], b: Sequence[Run], name: str, *, ci_level: float, labels: tuple[str, str]
    ) -> UtilityReport:
        check_pairing(a, b)
        a, b = sorted(a, key=lambda run: run.seed), sorted(b, key=lambda run: run.seed)
        points_a, draws_a = zip(*(self.scores(run, name) for run in a), strict=True)
        points_b, draws_b = zip(*(self.scores(run, name) for run in b), strict=True)
        per_class = np.array(points_a) - np.array(points_b)  # (S, K)
        per_seed = macro(np.array(points_a)) - macro(np.array(points_b))  # (S,)
        pairs = zip(draws_a, draws_b, strict=True)
        differences = np.stack([macro(da) - macro(db) for da, db in pairs], axis=1)  # (B, S)
        effects = np.take_along_axis(differences, self.seed_draws(len(a)), axis=1).mean(axis=1)  # (B,)
        effects = effects[np.isfinite(effects)]
        tail = (1 - ci_level) / 2
        low, high = np.quantile(effects, [tail, 1 - tail])
        mask, _, n_patients = self.weights(name)
        positives = self.labels[mask].sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            class_effect = np.nanmean(per_class, axis=0)
        return UtilityReport(
            a=labels[0],
            b=labels[1],
            split=self.split,
            checkpoint=self.checkpoint,
            slice=name,
            metric="macro_auroc",
            estimate=float(per_seed.mean()),
            ci_low=float(low),
            ci_high=float(high),
            ci_level=ci_level,
            n_boot=len(effects),
            seeds=[run.seed for run in a],
            per_seed=[float(value) for value in per_seed],
            mean_a=float(macro(np.array(points_a)).mean()),
            mean_b=float(macro(np.array(points_b)).mean()),
            n_records=int(mask.sum()),
            n_patients=n_patients,
            per_class={
                k: None if np.isnan(v) else float(v) for k, v in zip(SUPERCLASSES, class_effect, strict=True)
            },
            n_positive={k: int(v) for k, v in zip(SUPERCLASSES, positives, strict=True)},
        )


def findings(
    artifact: UtilityReport, clean: UtilityReport, *, practical_target: float, noninferiority_margin: float
) -> dict[str, Any]:
    """The separate questions of docs/timenet_syntide.md §3B, and one verdict that combines them."""
    positive = artifact.ci_low > 0
    target_supported = artifact.ci_low > practical_target
    target_ruled_out = artifact.ci_high < practical_target
    noninferior = clean.ci_low > noninferiority_margin
    benefit = positive and noninferior
    target = "supported" if target_supported else "ruled out" if target_ruled_out else "plausible"
    if benefit and target_supported:
        verdict = f"benefit: the artifact gain exceeds +{practical_target:g} and the clean slice holds"
    elif benefit and target_ruled_out:
        verdict = f"benefit below target: an artifact gain under +{practical_target:g}; the clean slice holds"
    elif benefit:
        verdict = f"benefit: an artifact gain, +{practical_target:g} still plausible; the clean slice holds"
    elif positive:
        verdict = f"artifact gain, but a clean-slice loss beyond {noninferiority_margin:g} is not ruled out"
    elif target_ruled_out:
        verdict = f"useful negative: no artifact gain shown, and a gain of +{practical_target:g} is ruled out"
    else:
        verdict = "inconclusive"
    return {
        "positive_effect_supported": positive,
        "target_effect_supported": target_supported,
        "target_effect_ruled_out": target_ruled_out,
        "target_effect": target,
        "clean_noninferiority_supported": noninferior,
        "benefit": benefit,
        "verdict": verdict,
    }


def fixed_sequence(benefits: Sequence[bool | None]) -> list[str]:
    """Status of each step of a fixed sequence, given its benefit (None: its arms did not run).

    Testing stops at the first step without a benefit; every later step is "not tested".
    """
    statuses, stopped = [], False
    for benefit in benefits:
        if stopped:
            statuses.append("not tested")
        elif benefit is None:
            statuses.append("not run")
            stopped = True
        else:
            statuses.append("passed" if benefit else "failed")
            stopped = not benefit
    return statuses


def gap_closed(
    numerator: UtilityReport, denominator: UtilityReport, *, gated: bool, gate_passed: bool | None
) -> dict[str, Any]:
    """(arm - baseline) / (target - baseline), descriptive (docs/timenet_syntide.md §3C).

    Numerator and denominator are always reported. The ratio is given only when the denominator's CI is
    above 0 and, for a gated entry, the gap gate passed on validation.
    """
    reason = None
    if gated and not gate_passed:
        reason = "the gap gate did not pass on validation"
    elif denominator.ci_low <= 0:
        reason = "the denominator's CI is not above 0"
    return {
        "numerator": numerator.to_dict(),
        "denominator": denominator.to_dict(),
        "ratio": None if reason else numerator.estimate / denominator.estimate,
        "withheld_because": reason,
    }
