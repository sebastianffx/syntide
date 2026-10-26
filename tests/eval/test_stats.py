"""Selection metrics remain finite for small evaluation groups."""

import numpy as np
import pytest

from syntide.eval.stats import macro_auroc, val_scores


def test_macro_auroc_skips_classes_without_both_labels_and_respects_weights():
    y = np.array(
        [[1, 0], [0, 0], [1, 0], [0, 0]], dtype=bool
    )  # class 1 has no positive
    p = np.array([[0.9, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])
    assert (
        macro_auroc(y, p) == 0.75
    )  # class 0 alone; class 1 is skipped instead of giving NaN
    # Zero weight removes record 3 (a negative scored above the positive 0.4),
    # so class 0 is perfect.
    assert macro_auroc(y, p, weights=np.array([1, 1, 1, 0])) == 1.0
    with pytest.raises(ValueError, match="undefined"):
        macro_auroc(y[:, 1:], p[:, 1:])


def test_val_scores_report_every_slice_and_reject_non_probabilities(
    make_samples,
):
    val = make_samples(40)
    probs = np.where(val.labels, 0.8, 0.2)
    scores = val_scores(val, probs)
    assert (
        scores["val/auroc_overall"]
        == scores["val/auroc_artifact"]
        == scores["val/auroc_clean"]
        == 1.0
    )
    assert scores["val/loss"] == pytest.approx(-np.log(0.8))
    with pytest.raises(ValueError, match="probabilities"):
        val_scores(val, probs + 1)
