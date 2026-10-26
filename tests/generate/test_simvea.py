"""SIMVEA lead reconstruction preserves its published settings and replay."""

import json
from collections import Counter
from dataclasses import asdict

import numpy as np
import pyarrow.parquet as pq
import pytest

from syntide.data.schema import LEADS
from syntide.generate import simvea
from syntide.generate.ledger import LedgerWriter, recipe_id
from syntide.generate.simvea import PAIRS, SimveaConfig
from syntide.generate.stream import TrainStream

RNG = np.random.default_rng(5)
X = RNG.standard_normal((12, 200)).astype(np.float32)
V = {name: LEADS.index(name) for name in LEADS}
ANGLE = {
    "V1": 120,
    "V2": 90,
    "V3": 75,
    "V4": 60,
    "V5": 30,
    "V6": 0,
}  # Fig. 1a, degrees


def test_a_replaced_lead_is_the_papers_weighted_sum_of_measured_leads():
    edits = [
        {"lead": "V2", "pair": ["V1", "V4"], "alpha": 0.5, "beta": 1.5},
        # V2 is replaced too; V3 must still be built from the measured V2.
        {"lead": "V3", "pair": ["V2", "V4"], "alpha": 0.8, "beta": 0.3},
    ]
    out = simvea.apply(X, {"leads": edits})
    x = X.astype(np.float64)
    np.testing.assert_allclose(
        out[V["V2"]], 0.5 * (1.5 * x[V["V1"]] + 0.5 * x[V["V4"]]), rtol=1e-6
    )
    np.testing.assert_allclose(
        out[V["V3"]], 0.8 * (0.3 * x[V["V2"]] + 1.7 * x[V["V4"]]), rtol=1e-6
    )
    untouched = [i for i in range(12) if i not in (V["V2"], V["V3"])]
    assert np.array_equal(out[untouched], X[untouched])
    assert out.dtype == np.float32
    # beta = 1 is plain SIMVEA, alpha * (X_j + X_z); no replaced lead returns
    # the ECG unchanged.
    plain = simvea.apply(
        X,
        {
            "leads": [
                {"lead": "V5", "pair": ["V4", "V6"], "alpha": 0.6, "beta": 1.0}
            ]
        },
    )
    np.testing.assert_allclose(
        plain[V["V5"]], 0.6 * (x[V["V4"]] + x[V["V6"]]), rtol=1e-6
    )
    assert np.array_equal(simvea.apply(X, {"leads": []}), X)


def test_every_pair_is_fig_1c_and_straddles_its_lead():
    assert {lead: len(pairs) for lead, pairs in PAIRS.items()} == {
        "V2": 2,
        "V3": 2,
        "V4": 3,
        "V5": 2,
    }
    for lead, pairs in PAIRS.items():
        for j, z in pairs:
            assert (
                ANGLE[j] > ANGLE[lead] > ANGLE[z]
            )  # one neighbour on each side of the lead
    exact = sorted(
        (lead, j, z)
        for lead, pairs in PAIRS.items()
        for j, z in pairs
        if ANGLE[j] + ANGLE[z] == 2 * ANGLE[lead]
    )
    # The pairs Fig. 1c stars: they straddle the lead's angle exactly.
    assert exact == [
        ("V2", "V1", "V4"),
        ("V3", "V1", "V5"),
        ("V3", "V2", "V4"),
        ("V4", "V1", "V6"),
        ("V4", "V2", "V5"),
        ("V5", "V4", "V6"),
    ]


def test_draws_follow_the_papers_grids_pairs_and_per_lead_coin():
    rng = np.random.default_rng(0)
    draws = [simvea.draw_controls(rng, SimveaConfig()) for _ in range(8000)]
    edits = [edit for draw in draws for edit in draw["leads"]]
    per_lead = Counter(edit["lead"] for edit in edits)
    assert set(per_lead) == {
        "V2",
        "V3",
        "V4",
        "V5",
    }  # V1 and V6 are never replaced
    for lead in per_lead:
        assert per_lead[lead] / len(draws) == pytest.approx(0.5, abs=0.02)
        pairs = Counter(
            tuple(edit["pair"]) for edit in edits if edit["lead"] == lead
        )
        assert set(pairs) == set(PAIRS[lead])  # uniform over the lead's pairs
        assert min(pairs.values()) / max(pairs.values()) > 0.85
    assert sum(not draw["leads"] for draw in draws) / len(
        draws
    ) == pytest.approx(1 / 16, abs=0.01)
    assert sorted({edit["alpha"] for edit in edits}) == [
        round(0.1 * k, 1) for k in range(1, 11)
    ]
    assert sorted({edit["beta"] for edit in edits}) == [
        round(0.1 * k, 1) for k in range(21)
    ]
    # A plain-SIMVEA config pins beta at 1.
    plain = SimveaConfig(beta=[1.0, 1.0])
    assert {
        e["beta"]
        for _ in range(50)
        for e in simvea.draw_controls(rng, plain)["leads"]
    } == {1.0}


@pytest.mark.parametrize(
    "cfg, n_leads, message",
    [
        (SimveaConfig(alpha=[0.0, 1.0]), 12, "0 < low"),
        (SimveaConfig(beta=[0.0, 2.5]), 12, "high <= 2"),
        (SimveaConfig(alpha=[0.1, 1.05]), 12, "whole number of steps"),
        (SimveaConfig(step=0.0), 12, "positive"),
        (SimveaConfig(lead_p=0.0), 12, r"\(0, 1\]"),
        (SimveaConfig(), 8, "needs the 12 leads"),
    ],
)
def test_impossible_configs_are_refused(cfg, n_leads, message):
    with pytest.raises(ValueError, match=message):
        simvea.check(cfg, n_leads)


def _spec(make_test_spec):
    transform = asdict(SimveaConfig())
    return make_test_spec(
        "A1",
        transform_family="simvea",
        generator="simvea@1",
        transform=transform,
    )


def test_a_simvea_arm_pairs_with_a0_and_changes_only_chest_leads_in_its_slots(
    make_samples, make_test_spec
):
    parents = make_samples(20)
    a0 = list(TrainStream(make_test_spec("A0"), parents))
    t6 = list(TrainStream(_spec(make_test_spec), parents))
    chest = [V[name] for name in PAIRS]
    others = [i for i in range(12) if i not in chest]
    for b0, b6 in zip(a0, t6, strict=True):
        assert np.array_equal(b0.record_id, b6.record_id) and np.array_equal(
            b0.y, b6.y
        )  # labels kept
        assert b6.transformed.sum() == 4  # A1's half of the batch
        assert np.array_equal(b6.x[~b6.transformed], b0.x[~b6.transformed])
        assert np.array_equal(
            b6.x[:, others], b0.x[:, others]
        )  # limb leads, V1 and V6 never change
    changed = np.concatenate(
        [
            (b6.x != b0.x).any(axis=2)[b6.transformed]
            for b0, b6 in zip(a0, t6, strict=True)
        ]
    )
    assert changed[:, chest].mean() == pytest.approx(0.5, abs=0.12)


def test_every_simvea_ledger_row_regenerates_its_input(
    make_samples, make_test_spec, tmp_path
):
    parents = make_samples(20)
    ledger = LedgerWriter(tmp_path / "recipes.parquet")
    served = {
        b.step: b
        for b in TrainStream(_spec(make_test_spec), parents, ledger=ledger)
    }
    rows = pq.read_table(ledger.close()).to_pylist()
    row_of = {record_id: i for i, record_id in enumerate(parents.record_id)}

    assert len(rows) == 30 * 4
    for row in rows:
        controls = json.loads(row["controls_requested"])
        regenerated = simvea.apply(
            parents.x[row_of[row["parent_record_id"]]], controls
        )
        assert np.array_equal(regenerated, served[row["step"]].x[row["slot"]])
        assert json.loads(row["label_effect"]) == simvea.LABEL_EFFECT
        assert row["generator"] == "simvea" and row["decision"] == "accepted"
        assert row["recipe_id"] == recipe_id(
            row["parent_record_id"],
            "simvea",
            "1",
            controls,
            row["seed_key"],
            None,
        )
