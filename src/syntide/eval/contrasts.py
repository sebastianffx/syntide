"""Compare matched experiment treatments using shared resampling draws.

A comparison subtracts each baseline model's score from the treatment
model's score at the same seed. The effect estimate averages those
differences. Confidence intervals resample patients and seeds independently,
using the same draws for both treatments and all comparisons on the data
subset.

Scores average area under the receiver operating characteristic curve across
classes. A class requires both positive and negative examples with nonzero
weight. Compared runs must share records, labels, trainer settings, training
budgets, and initial weights. Treatments with the same transformed batch
fraction must also transform the same batch positions.
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
    """Compared runs differ in more than the intended treatment."""


def auroc_draws(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Compute weighted classification scores for each resampling draw.

    Args:
        y: Binary labels for the recordings.
        p: Predicted scores aligned with y.
        weights: One row of recording weights per resampling draw.

    Returns:
        One score per draw. Tied predictions contribute one half. A draw
        without both a positive and a negative weighted example returns a
        non-finite value.
    """
    order = np.argsort(p, kind="mergesort")
    p, y, w = p[order], np.asarray(y, dtype=bool)[order], weights[:, order]
    starts = np.flatnonzero(
        np.r_[True, p[1:] != p[:-1]]
    )  # first index of each run of tied scores
    pos = np.add.reduceat(np.where(y, w, 0.0), starts, axis=1)
    neg = np.add.reduceat(np.where(y, 0.0, w), starts, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    n_pos, n_neg = pos.sum(axis=1), neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (pos * (neg_below + 0.5 * neg)).sum(axis=1) / (n_pos * n_neg)
    return np.where((n_pos > 0) & (n_neg > 0), auc, np.nan)


def class_aurocs(
    labels: np.ndarray, probs: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Compute a classification score for each resampling draw and class.

    Return an array shaped (draws, classes). A class lacking positive or
    negative examples has a non-finite score.
    """
    return np.stack(
        [
            auroc_draws(labels[:, k], probs[:, k], weights)
            for k in range(labels.shape[1])
        ],
        axis=1,
    )


def macro(per_class: np.ndarray) -> np.ndarray:
    """Average defined class scores, or return a non-finite value."""
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
    raise ValueError(
        f"unknown slice {name!r}; choose one of {', '.join(SLICES)}"
    )


@dataclass(frozen=True)
class UtilityReport:
    """A paired treatment comparison for one data subset."""

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
    per_seed: list[
        float
    ]  # the paired difference of each seed, in ``seeds`` order
    mean_a: float
    mean_b: float
    n_records: int
    n_patients: int
    per_class: dict[
        str, float | None
    ]  # mean over seeds of the per-class difference; None if undefined
    n_positive: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_pairing(a: Sequence[Run], b: Sequence[Run]) -> None:
    """Reject runs that differ in more than the intended treatment."""
    for side in (a, b):
        if not side:
            raise PairingError("a contrast needs at least one run per arm")
        if len({run.key for run in side}) != 1:
            raise PairingError(
                "one side of the contrast mixes arms "
                f"{sorted({run.key for run in side})}"
            )
        if len({run.seed for run in side}) != len(side):
            raise PairingError(f"{side[0].key} lists a seed twice")
    seeds_a, seeds_b = (
        sorted(run.seed for run in a),
        sorted(run.seed for run in b),
    )
    if seeds_a != seeds_b:
        raise PairingError(
            f"the arms have different seeds: {seeds_a} vs {seeds_b}"
        )
    runs = [*a, *b]
    if len({run.fingerprint for run in runs}) != 1:
        raise PairingError(
            "the arms were trained by different trainers or trainer configs"
        )
    if len({run.exposure["presentations"] for run in runs}) != 1:
        raise PairingError(
            "the arms saw different numbers of training "
            "presentations; budgets differ"
        )
    by_seed = {run.seed: run for run in b}
    for run_a in a:
        run_b = by_seed[run_a.seed]
        if run_a.init_sha256 != run_b.init_sha256:
            raise PairingError(
                f"seed {run_a.seed} starts from different initial weights "
                "in the two arms"
            )
        # Two transform arms that declare the same share of transformed slots
        # must have served the same
        # slots. A0's slots hold unchanged parents, and a dose arm declares its
        # own share: that is the arm.
        transforms = all(
            run.spec.transform_family != "none"
            and run.spec.transform_fraction > 0
            for run in (run_a, run_b)
        )
        same_share = (
            run_a.spec.transform_fraction == run_b.spec.transform_fraction
        )
        if (
            transforms
            and same_share
            and run_a.exposure["transform_slots"]
            != run_b.exposure["transform_slots"]
        ):
            raise PairingError(
                f"seed {run_a.seed} has different transform slots in the "
                "two arms"
            )


class PairedScorer:
    """Compare run scores using the same patient and seed samples.

    ``records`` is any run of the split: every scored run must have exactly
    its records and labels.
    """

    def __init__(
        self, records: Run, *, checkpoint: str, n_boot: int, seed: int
    ) -> None:
        self.record_id, self.subject_uid = (
            records.record_id,
            records.subject_uid,
        )
        self.labels, self.is_artifact = records.labels, records.is_artifact
        self.split, self.checkpoint, self.n_boot, self.seed = (
            records.split,
            checkpoint,
            n_boot,
            seed,
        )
        self._weights: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        self._seed_draws: dict[int, np.ndarray] = {}
        self._scores: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def weights(self, name: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Return the record mask, sample weights, and patient count."""
        if name not in self._weights:
            mask = slice_mask(self.is_artifact, name)
            patients, inverse = np.unique(
                self.subject_uid[mask], return_inverse=True
            )
            rng = np.random.default_rng([self.seed, SLICES.index(name)])
            uniform = np.full(len(patients), 1 / len(patients))
            counts = rng.multinomial(len(patients), uniform, size=self.n_boot)
            self._weights[name] = (
                mask,
                counts[:, inverse].astype(np.float64),
                len(patients),
            )
        return self._weights[name]

    def seed_draws(self, n_seeds: int) -> np.ndarray:
        """(B, n_seeds) indices of the seeds each draw uses."""
        if n_seeds not in self._seed_draws:
            rng = np.random.default_rng([self.seed, len(SLICES), n_seeds])
            self._seed_draws[n_seeds] = rng.integers(
                0, n_seeds, size=(self.n_boot, n_seeds)
            )
        return self._seed_draws[n_seeds]

    def scores(self, run: Run, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute class scores with unit and resampling weights.

        Return the unweighted class scores and an array shaped (draws,
        classes).
        """
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
                raise PairingError(
                    f"{run.run_dir} was scored on other records or labels "
                    "than the other runs"
                )
            mask, weights, _ = self.weights(name)
            labels, probs = self.labels[mask], run.probs[self.checkpoint][mask]
            point = class_aurocs(labels, probs, np.ones((1, int(mask.sum()))))[
                0
            ]
            self._scores[cache] = (point, class_aurocs(labels, probs, weights))
        return self._scores[cache]

    def point(self, run: Run, name: str) -> float:
        """Return the mean score over defined classes in the subset."""
        return float(macro(self.scores(run, name)[0]))

    def contrast(
        self,
        a: Sequence[Run],
        b: Sequence[Run],
        name: str,
        *,
        ci_level: float,
        labels: tuple[str, str],
    ) -> UtilityReport:
        check_pairing(a, b)
        a, b = (
            sorted(a, key=lambda run: run.seed),
            sorted(b, key=lambda run: run.seed),
        )
        points_a, draws_a = zip(
            *(self.scores(run, name) for run in a), strict=True
        )
        points_b, draws_b = zip(
            *(self.scores(run, name) for run in b), strict=True
        )
        per_class = np.array(points_a) - np.array(points_b)  # (S, K)
        per_seed = macro(np.array(points_a)) - macro(np.array(points_b))  # (S,)
        pairs = zip(draws_a, draws_b, strict=True)
        differences = np.stack(
            [macro(da) - macro(db) for da, db in pairs], axis=1
        )  # (B, S)
        effects = np.take_along_axis(
            differences, self.seed_draws(len(a)), axis=1
        ).mean(axis=1)  # (B,)
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
                k: None if np.isnan(v) else float(v)
                for k, v in zip(SUPERCLASSES, class_effect, strict=True)
            },
            n_positive={
                k: int(v) for k, v in zip(SUPERCLASSES, positives, strict=True)
            },
        )


def findings(
    artifact: UtilityReport,
    clean: UtilityReport,
    *,
    practical_target: float,
    noninferiority_margin: float,
) -> dict[str, Any]:
    """Assess benefit, harm, and support for the practical target."""
    positive = artifact.ci_low > 0
    target_supported = artifact.ci_low > practical_target
    target_ruled_out = artifact.ci_high < practical_target
    noninferior = clean.ci_low > noninferiority_margin
    benefit = positive and noninferior
    target = (
        "supported"
        if target_supported
        else "ruled out"
        if target_ruled_out
        else "plausible"
    )
    if benefit and target_supported:
        verdict = (
            f"benefit: the artifact gain exceeds +{practical_target:g} and "
            "the clean slice holds"
        )
    elif benefit and target_ruled_out:
        verdict = (
            "benefit below target: an artifact gain under +"
            f"{practical_target:g}; the clean slice holds"
        )
    elif benefit:
        verdict = (
            f"benefit: an artifact gain, +{practical_target:g} still "
            "plausible; the clean slice holds"
        )
    elif positive:
        verdict = (
            "artifact gain, but a clean-slice loss beyond "
            f"{noninferiority_margin:g} is not ruled out"
        )
    elif target_ruled_out:
        verdict = (
            "useful negative: no artifact gain shown, and a gain of +"
            f"{practical_target:g} is ruled out"
        )
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
    """Evaluate comparisons in order until one fails to show benefit.

    Testing stops at the first step without a benefit; every later step is
    "not tested".
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
    numerator: UtilityReport,
    denominator: UtilityReport,
    *,
    gated: bool,
    gate_passed: bool | None,
) -> dict[str, Any]:
    """Describe the fraction of the reference performance gap recovered.

    Compute (treatment - baseline) / (target - baseline). Always report
    numerator and denominator. Report the ratio only when the denominator's
    confidence interval is positive and any required validation check
    passes.
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
