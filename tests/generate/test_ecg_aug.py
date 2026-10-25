"""The single-technique augmentations: each operation does only what it claims, the stream pairs them with
A0, mixup blends labels by its lambda, and every ledger row regenerates its training input bit-exactly."""

import json
from dataclasses import asdict

import numpy as np
import pyarrow.parquet as pq
import pytest

from syntide.generate import ecg_aug
from syntide.generate.ecg_aug import EcgAugConfig
from syntide.generate.ledger import LedgerWriter, recipe_id
from syntide.generate.stream import TrainStream

RNG = np.random.default_rng(3)
X = RNG.standard_normal((12, 200)).astype(np.float32)


def test_each_operation_changes_only_what_it_claims():
    assert np.array_equal(ecg_aug.apply(X, {"stretch": 1.0, "stretch_offset": 0}), X)
    assert np.array_equal(ecg_aug.apply(X, {"stretch": 1.0, "stretch_offset": 5}), np.roll(X, -5, axis=1))
    faster = ecg_aug.apply(X, {"stretch": 2.0, "stretch_offset": 0})
    assert np.array_equal(faster[:, :100], X[:, ::2])  # twice the rate: every second sample, then wraps

    assert np.allclose(ecg_aug.apply(X, {"scale": 1.3}), 1.3 * X)

    masked = ecg_aug.apply(X, {"time_mask": [40, 25]})
    assert not masked[:, 40:65].any()
    assert np.array_equal(np.delete(masked, np.s_[40:65], axis=1), np.delete(X, np.s_[40:65], axis=1))

    dropped = ecg_aug.apply(X, {"lead_dropout": [1, 6]})
    assert not dropped[[1, 6]].any()
    assert np.array_equal(np.delete(dropped, [1, 6], axis=0), np.delete(X, [1, 6], axis=0))

    partner = RNG.standard_normal((12, 200)).astype(np.float32)
    mixed = ecg_aug.apply(X, {"mixup_lambda": 0.75}, partner)
    assert np.allclose(mixed, 0.75 * X + 0.25 * partner, atol=1e-6)
    y, partner_y = np.array([1, 0, 0, 1, 0], np.float32), np.array([0, 0, 1, 1, 0], np.float32)
    assert np.allclose(ecg_aug.mix_labels(y, partner_y, {"mixup_lambda": 0.75}), [0.75, 0, 0.25, 1, 0])
    assert ecg_aug.mix_labels(y, partner_y, {"scale": 1.1}) is y
    with pytest.raises(ValueError, match="partner"):
        ecg_aug.apply(X, {"mixup_lambda": 0.5})


def test_controls_stay_inside_the_configured_ranges():
    cfg = EcgAugConfig(
        stretch=[0.85, 1.18], scale=[0.7, 1.43], time_mask_s=[0.5, 2.0], lead_dropout=[1, 4], mixup_alpha=0.4
    )
    rng = np.random.default_rng(0)
    for _ in range(500):
        c = ecg_aug.draw_controls(rng, cfg, 12, 1000, 100, n_parents=50)
        assert 0.85 <= c["stretch"] <= 1.18 and 0 <= c["stretch_offset"] < 1000
        assert 0.7 <= c["scale"] <= 1.43
        start, width = c["time_mask"]
        assert 50 <= width <= 200 and start >= 0 and start + width <= 1000
        assert 1 <= len(c["lead_dropout"]) <= 4 and len(set(c["lead_dropout"])) == len(c["lead_dropout"])
        assert 0 <= c["mixup_lambda"] <= 1 and 0 <= c["mixup_partner"] < 50
    assert set(ecg_aug.draw_controls(rng, EcgAugConfig(scale=[0.9, 1.1]), 12, 1000, 100, 50)) == {"scale"}


@pytest.mark.parametrize(
    "cfg, message",
    [
        (EcgAugConfig(), "at least one"),
        (EcgAugConfig(scale=[1.2, 0.8]), "low <= high"),
        (EcgAugConfig(time_mask_s=[0.0, 1.0]), "0 < low"),
        (EcgAugConfig(lead_dropout=[1, 12]), "at least one of 12 leads"),
        (EcgAugConfig(mixup_alpha=0.0), "positive"),
    ],
)
def test_impossible_configs_are_refused(cfg, message):
    with pytest.raises(ValueError, match=message):
        ecg_aug.check(cfg)


def _spec(make_test_spec, cfg: EcgAugConfig):
    return make_test_spec("A1", transform_family="ecg_aug", generator="ecg_aug@1", transform=asdict(cfg))


def test_a_technique_arm_pairs_with_a0_and_changes_only_its_slots(make_samples, make_test_spec):
    parents = make_samples(20)
    a0 = list(TrainStream(make_test_spec("A0"), parents))
    t3 = list(TrainStream(_spec(make_test_spec, EcgAugConfig(scale=[0.7, 1.43])), parents))
    for b0, b3 in zip(a0, t3, strict=True):
        assert np.array_equal(b0.record_id, b3.record_id) and np.array_equal(b0.y, b3.y)
        assert b3.transformed.sum() == 4
        assert np.array_equal(b3.x[~b3.transformed], b0.x[~b3.transformed])
        ratio = b3.x[b3.transformed] / b0.x[b3.transformed]
        assert np.allclose(ratio, ratio[:, :1, :1], rtol=1e-4)  # one factor for every lead and sample


def test_every_ecg_aug_ledger_row_regenerates_its_input_and_mixup_its_labels(
    make_samples, make_test_spec, tmp_path
):
    parents = make_samples(20)
    cfg = EcgAugConfig(stretch=[0.85, 1.18], time_mask_s=[0.05, 0.2], lead_dropout=[1, 4], mixup_alpha=0.4)
    ledger = LedgerWriter(tmp_path / "recipes.parquet")
    served = {b.step: b for b in TrainStream(_spec(make_test_spec, cfg), parents, ledger=ledger)}
    rows = pq.read_table(ledger.close()).to_pylist()
    row_of = {record_id: i for i, record_id in enumerate(parents.record_id)}

    assert len(rows) == 30 * 4
    for row in rows:
        controls = json.loads(row["controls_requested"])
        parent, partner = row_of[row["parent_record_id"]], row_of[controls["mixup_partner"]]
        batch = served[row["step"]]
        regenerated = ecg_aug.apply(parents.x[parent], controls, parents.x[partner])
        assert np.array_equal(regenerated, batch.x[row["slot"]])
        lam = controls["mixup_lambda"]
        expected_y = lam * parents.labels[parent] + (1 - lam) * parents.labels[partner]
        assert np.allclose(batch.y[row["slot"]], expected_y, atol=1e-6)
        assert json.loads(row["label_effect"])["diagnosis"] == "mixes"
        assert row["recipe_id"] == recipe_id(
            row["parent_record_id"], "ecg_aug", "1", controls, row["seed_key"], None
        )
    # Slots outside the transform keep their parents' hard labels.
    for batch in served.values():
        assert set(np.unique(batch.y[~batch.transformed])) <= {0.0, 1.0}
