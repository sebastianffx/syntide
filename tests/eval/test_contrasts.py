"""Paired contrasts: identical arms cancel, a real effect is found, unpaired arms are refused, and the
decision rules read the CIs as frozen."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from syntide.data.schema import SUPERCLASSES
from syntide.eval.analysis import AnalysisConfig, gap_gate_outcome
from syntide.eval.contrasts import (
    PairedScorer,
    PairingError,
    UtilityReport,
    auroc_draws,
    findings,
    fixed_sequence,
    gap_closed,
)
from syntide.eval.runs import Run
from syntide.eval.stats import macro_auroc

N, SEEDS = 240, (0, 1, 2)


@pytest.fixture(scope="module")
def records():
    rng = np.random.default_rng(7)
    patients = rng.integers(0, 150, N)  # some patients have several ECGs
    return {
        "record_id": np.array([f"ptbxl-{i:04d}" for i in range(N)]),
        "subject_uid": np.array([f"ptb-xl:{p}" for p in patients]),
        "labels": rng.random((N, len(SUPERCLASSES))) < 0.3,
        "is_artifact": rng.random(N) < 0.3,
    }


@pytest.fixture
def make_run(records, make_test_spec):
    def build(arm, seed, probs, *, regime="scarcity25", result=None, **record_changes):
        result = {
            "trainer": {"fingerprint": "f" * 64, "init_sha256": f"{seed}" * 64},
            "exposure": {"presentations": 240, "transform_slots": 120},
            "wall_time_s": 1.0,
        } | (result or {})
        return Run(
            run_dir=Path(f"runs/{regime}/{arm}/seed{seed}"),
            spec=make_test_spec(arm, replicate_seed=seed, regime=regime),
            result=result,
            split="val",
            probs={"best": probs},
            steps={"best": 10},
            **(records | record_changes),
        )

    return build


def _probs(labels, quality, seed):
    noise = np.random.default_rng(seed).standard_normal(labels.shape)
    return 1 / (1 + np.exp(-(quality * (labels - 0.5) + noise)))


def _scorer(run, n_boot=500):
    return PairedScorer(run, checkpoint="best", n_boot=n_boot, seed=0)


def test_auroc_draws_match_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y, p = rng.random(300) < 0.3, np.round(rng.random(300), 1)  # coarse scores: many ties
    weights = rng.integers(0, 3, size=(6, 300)).astype(float)
    expected = [roc_auc_score(y, p, sample_weight=w) for w in weights]
    np.testing.assert_allclose(auroc_draws(y, p, weights), expected, rtol=0, atol=1e-12)
    assert np.isnan(auroc_draws(np.zeros(300, bool), p, weights)).all()  # no positive: undefined


def test_identical_arms_cancel_exactly(make_run, records):
    probs = [_probs(records["labels"], 2.0, seed) for seed in SEEDS]
    a = [make_run("A3", s, probs[s]) for s in SEEDS]
    b = [make_run("A1", s, probs[s]) for s in SEEDS]
    report = _scorer(a[0]).contrast(a, b, "artifact", ci_level=0.95, labels=("A3", "A1"))
    assert report.estimate == report.ci_low == report.ci_high == 0.0
    assert report.per_seed == [0.0, 0.0, 0.0]


def test_a_real_effect_is_detected_and_every_seed_is_reported(make_run, records):
    labels = records["labels"]
    a = [make_run("A3", s, _probs(labels, 4.0, s)) for s in SEEDS]
    b = [make_run("A1", s, _probs(labels, 1.0, 10 + s)) for s in SEEDS]
    scorer = _scorer(a[0])
    report = scorer.contrast(a, b, "overall", ci_level=0.95, labels=("A3", "A1"))
    assert report.ci_low > 0 and report.ci_low < report.estimate < report.ci_high
    expected = [
        macro_auroc(labels, run_a.probs["best"]) - macro_auroc(labels, run_b.probs["best"])
        for run_a, run_b in zip(a, b, strict=True)
    ]
    np.testing.assert_allclose(report.per_seed, expected, atol=1e-12)  # unit weights = the selection metric
    assert report.seeds == list(SEEDS) and report.n_records == N
    artifact = scorer.contrast(a, b, "artifact", ci_level=0.95, labels=("A3", "A1"))
    assert artifact.n_records == records["is_artifact"].sum()
    assert artifact.n_patients == len(set(records["subject_uid"][records["is_artifact"]]))


def test_a_class_without_positives_is_skipped_not_nan(make_run, records):
    labels = records["labels"].copy()
    labels[:, 1] = False  # HYP never occurs
    a = [make_run("A3", s, _probs(labels, 3.0, s), labels=labels) for s in SEEDS]
    b = [make_run("A1", s, _probs(labels, 1.0, s), labels=labels) for s in SEEDS]
    report = _scorer(a[0]).contrast(a, b, "artifact", ci_level=0.95, labels=("A3", "A1"))
    assert np.isfinite([report.estimate, report.ci_low, report.ci_high]).all()
    assert report.per_class["HYP"] is None and report.n_positive["HYP"] == 0


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda run, s: run if s else None, "different seeds"),
        (
            lambda run, s: replace(
                run, result=run.result | {"trainer": {**run.result["trainer"], "fingerprint": "e"}}
            ),
            "different trainers",
        ),
        (
            lambda run, s: replace(
                run, result=run.result | {"trainer": {**run.result["trainer"], "init_sha256": "x"}}
            ),
            "different initial weights",
        ),
        (
            lambda run, s: replace(
                run, result=run.result | {"exposure": {"presentations": 480, "transform_slots": 120}}
            ),
            "budgets differ",
        ),
        (
            lambda run, s: replace(
                run, result=run.result | {"exposure": {"presentations": 240, "transform_slots": 96}}
            ),
            "different transform slots",
        ),
        (lambda run, s: replace(run, labels=~run.labels), "other records or labels"),
        (lambda run, s: replace(run, record_id=np.roll(run.record_id, 1)), "other records or labels"),
    ],
    ids=["seeds", "fingerprint", "init", "presentations", "transform-slots", "labels", "records"],
)
def test_arms_that_differ_in_more_than_the_arm_are_refused(make_run, records, change, message):
    a = [make_run("A3", s, _probs(records["labels"], 2.0, s)) for s in SEEDS]
    b = [make_run("A1", s, _probs(records["labels"], 2.0, s)) for s in SEEDS]
    changed = [run for run in (change(run, s) for s, run in enumerate(b)) if run is not None]
    with pytest.raises(PairingError, match=message):
        _scorer(a[0]).contrast(a, changed, "overall", ci_level=0.95, labels=("A3", "A1"))


def test_an_arm_without_transform_slots_pairs_with_a_transform_arm(make_run, records):
    a = [make_run("C1", s, _probs(records["labels"], 2.0, s)) for s in SEEDS]
    a = [
        replace(run, result=run.result | {"exposure": {"presentations": 240, "transform_slots": 0}})
        for run in a
    ]
    b = [make_run("A0", s, _probs(records["labels"], 2.0, s)) for s in SEEDS]
    _scorer(a[0]).contrast(a, b, "overall", ci_level=0.95, labels=("C1", "A0"))


def _report(estimate, low, high, slice_name="artifact"):
    return UtilityReport(
        a="A3", b="A1", split="val", checkpoint="best", slice=slice_name, metric="macro_auroc",
        estimate=estimate, ci_low=low, ci_high=high, ci_level=0.95, n_boot=2000, seeds=[0],
        per_seed=[estimate], mean_a=0.9, mean_b=0.9, n_records=10, n_patients=10, per_class={},
        n_positive={},
    )  # fmt: skip


def test_the_decision_rules_are_separate_questions():
    small_gain = findings(
        _report(0.009, 0.003, 0.015), _report(0.0, -0.004, 0.004, "clean"), practical_target=0.02,
        noninferiority_margin=-0.01,
    )  # fmt: skip
    assert small_gain["positive_effect_supported"] and small_gain["target_effect_ruled_out"]
    assert small_gain["benefit"] and small_gain["verdict"].startswith("benefit below target")

    clean_loss = findings(
        _report(0.009, 0.003, 0.015), _report(-0.005, -0.02, 0.01, "clean"), practical_target=0.02,
        noninferiority_margin=-0.01,
    )  # fmt: skip
    assert not clean_loss["clean_noninferiority_supported"] and not clean_loss["benefit"]

    null = findings(
        _report(0.001, -0.006, 0.012), _report(0.0, -0.004, 0.004, "clean"), practical_target=0.02,
        noninferiority_margin=-0.01,
    )  # fmt: skip
    assert null["verdict"].startswith("useful negative") and null["target_effect"] == "ruled out"


def test_the_fixed_sequence_stops_at_the_first_step_without_a_benefit():
    assert fixed_sequence([True, False, True]) == ["passed", "failed", "not tested"]
    assert fixed_sequence([None, True]) == ["not run", "not tested"]
    assert fixed_sequence([True, True]) == ["passed", "passed"]


def test_gap_closed_withholds_the_ratio_unless_the_gate_passed_and_the_denominator_is_clear():
    numerator, denominator = _report(0.01, 0.0, 0.02), _report(0.02, 0.005, 0.035)
    assert gap_closed(numerator, denominator, gated=True, gate_passed=True)["ratio"] == pytest.approx(0.5)
    assert gap_closed(numerator, denominator, gated=True, gate_passed=False)["ratio"] is None
    assert gap_closed(numerator, denominator, gated=False, gate_passed=False)["ratio"] == pytest.approx(0.5)
    unclear = gap_closed(numerator, _report(0.02, -0.001, 0.04), gated=False, gate_passed=None)
    assert unclear["ratio"] is None and "denominator" in unclear["withheld_because"]


def test_the_gap_gate_is_inclusive(make_run, records):
    labels = records["labels"]
    c1 = [make_run("C1", s, _probs(labels, 3.0, s)) for s in SEEDS]
    c1 = [
        replace(run, result=run.result | {"exposure": {"presentations": 240, "transform_slots": 0}})
        for run in c1
    ]
    a0 = [make_run("A0", s, _probs(labels, 1.0, s)) for s in SEEDS]
    rules = AnalysisConfig(regime="scarcity25", n_boot=200)
    estimate = gap_gate_outcome([*c1, *a0], rules)["artifact"]["estimate"]
    assert gap_gate_outcome(
        [*c1, *a0], replace(rules, gap_gate=replace(rules.gap_gate, min_effect=estimate))
    )["passed"]
    above = np.nextafter(estimate, np.inf)
    assert not gap_gate_outcome(
        [*c1, *a0], replace(rules, gap_gate=replace(rules.gap_gate, min_effect=above))
    )["passed"]
