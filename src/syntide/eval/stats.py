"""Metrics SYNTIDE owns, so that no trainer can choose its own.

``val_scores`` is the checkpoint-selection metric: trainers call it during
training, and ``Trial.finish`` recomputes it from the probabilities the
trainer hands back.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from syntide.data.schema import Samples

SELECTION_METRIC = "val/auroc_overall"
_EPS = 1e-7


def macro_auroc(
    y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Compute mean area under the receiver operating characteristic curve.

    Classes without both a positive and negative weighted example are
    skipped.

    Args:
        y_true: Binary labels, shaped (recordings, classes).
        y_prob: Predicted probabilities in the same shape as y_true.
        weights: Optional per-recording weights, such as resampling counts.

    Returns:
        The unweighted mean of the scores for the defined classes.

    Raises:
        ValueError: No class has both a positive and a negative example.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    w = (
        np.ones(len(y_true))
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    aurocs = []
    for k in range(y_true.shape[1]):
        positive, present = y_true[:, k], w > 0
        if (positive & present).any() and (~positive & present).any():
            aurocs.append(
                roc_auc_score(positive, y_prob[:, k], sample_weight=w)
            )
    if not aurocs:
        raise ValueError(
            "no class has both a positive and a negative record; "
            "AUROC is undefined"
        )
    return float(np.mean(aurocs))


def val_scores(val: Samples, probs: np.ndarray) -> dict[str, float]:
    """Compute the shared validation scores used for checkpoint selection.

    Args:
        val: Validation recordings with labels and signal-quality flags.
        probs: Predicted class probabilities aligned with val.labels.

    Returns:
        One mean binary cross-entropy loss across all recordings and three
        classification scores: all recordings, noisy recordings, and clean
        recordings. Values use the fixed metric names expected by trainers.

    Raises:
        ValueError: Probabilities have the wrong shape, are invalid, or no
            class in a scored subset has both positive and negative
            examples.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != val.labels.shape:
        raise ValueError(
            f"probs have shape {probs.shape}, expected {val.labels.shape}"
        )
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError("probs must be finite probabilities in [0, 1]")
    y = val.labels
    p = np.clip(probs, _EPS, 1 - _EPS)
    artifact = val.is_artifact
    return {
        SELECTION_METRIC: macro_auroc(y, probs),
        "val/auroc_artifact": macro_auroc(y[artifact], probs[artifact]),
        "val/auroc_clean": macro_auroc(y[~artifact], probs[~artifact]),
        "val/loss": float(-np.mean(np.where(y, np.log(p), np.log1p(-p)))),
    }
