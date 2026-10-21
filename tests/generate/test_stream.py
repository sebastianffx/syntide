"""The paired training stream: arms share their schedule, the budget is exact, recipes regenerate,
and rejected edits are retried or recorded, never replaced."""

import itertools
import json
from dataclasses import asdict

import numpy as np
import pyarrow.parquet as pq
import pytest

from syntide.data.schema import FS_HZ
from syntide.generate import editor, generic, validate
from syntide.generate.editor import EditorConfig
from syntide.generate.ledger import LedgerWriter, Recipe, recipe_id
from syntide.generate.policy import CalibrationError
from syntide.generate.stream import TrainStream


def test_paired_arms_share_parents_slots_and_init_and_differ_only_inside_transform_slots(
    make_samples, make_test_spec
):
    parents = make_samples(20)
    a0, a1 = make_test_spec("A0"), make_test_spec("A1")
    assert a0.init_seed == a1.init_seed
    batches = {arm: list(TrainStream(spec, parents)) for arm, spec in (("A0", a0), ("A1", a1))}
    row_of = {record_id: i for i, record_id in enumerate(parents.record_id)}

    for b0, b1 in zip(batches["A0"], batches["A1"], strict=True):
        assert np.array_equal(b0.record_id, b1.record_id)
        assert np.array_equal(b0.y, b1.y)
        # A0's slots hold unchanged parents; A1 transforms exactly its slots and nothing else.
        assert np.array_equal(b0.x, parents.x[[row_of[r] for r in b0.record_id]])
        assert not b0.transformed.any()
        assert b1.transformed.sum() == 4
        assert np.array_equal(b1.x[~b1.transformed], b0.x[~b1.transformed])
        assert not np.allclose(b1.x[b1.transformed], b0.x[b1.transformed])

    other_seed = list(TrainStream(make_test_spec("A0", replicate_seed=1), parents))
    assert any(
        not np.array_equal(a.record_id, b.record_id) for a, b in zip(batches["A0"], other_seed, strict=True)
    )


def test_stream_serves_exactly_the_budget_once_and_presents_every_parent_equally(
    make_samples, make_test_spec
):
    parents = make_samples(20)
    stream = TrainStream(make_test_spec("A1"), parents)
    batches = list(stream)

    assert [b.step for b in batches] == list(range(1, 31))
    assert stream.consumed
    assert stream.exposure["presentations"] == 30 * 8
    assert stream.exposure["transform_slots"] == stream.exposure["accepted"] == 30 * 4
    assert stream.exposure["distinct_recipes"] == 30 * 4
    # 240 presentations over 20 parents = 12 full passes, each a permutation (batches straddle passes).
    _, counts = np.unique(np.concatenate([b.record_id for b in batches]), return_counts=True)
    assert set(counts) == {12}
    with pytest.raises(RuntimeError, match="only once"):
        iter(stream)

    c1 = TrainStream(make_test_spec("C1"), parents)
    assert not any(b.transformed.any() for b in c1)
    assert c1.exposure["transform_slots"] == 0


def test_every_a1_ledger_row_regenerates_its_training_input_bit_exactly(
    make_samples, make_test_spec, tmp_path
):
    parents = make_samples(20)
    ledger = LedgerWriter(tmp_path / "recipes.parquet", flush_every=7)  # several row groups
    served = {b.step: b for b in TrainStream(make_test_spec("A1"), parents, ledger=ledger)}
    rows = pq.read_table(ledger.close()).to_pylist()
    row_of = {record_id: i for i, record_id in enumerate(parents.record_id)}

    assert len(rows) == 30 * 4
    assert max(row["step"] for row in rows) <= 30
    for row in rows:
        controls = json.loads(row["controls_requested"])
        parent = parents.x[row_of[row["parent_record_id"]]]
        regenerated = generic.apply(parent, controls, row["seed_key"], FS_HZ)
        assert np.array_equal(regenerated, served[row["step"]].x[row["slot"]])
        assert row["recipe_id"] == recipe_id(
            row["parent_record_id"], row["generator"], row["version"], controls, row["seed_key"], None
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("parent_id", "rec-2"),
        ("generator", "other"),
        ("version", "2"),
        ("controls", {"wander_mv": 0.51}),
        ("seed_key", 8),
        ("gap_report_id", "gap-1"),
    ],
)
def test_recipe_id_changes_with_every_field(field, value):
    base = {
        "parent_id": "rec-1",
        "generator": "per_lead_aug",
        "version": "1",
        "controls": {"wander_mv": 0.5},
        "seed_key": 7,
        "gap_report_id": None,
    }
    assert recipe_id(**{**base, field: value}) != recipe_id(**base)


def _editor_run(spec, parents, calibration, tmp_path, name):
    ledger = LedgerWriter(tmp_path / f"{name}.parquet")
    stream = TrainStream(spec, parents, ledger=ledger, calibration=calibration)
    batches = {b.step: b for b in stream}
    return stream, batches, pq.read_table(ledger.close()).to_pylist()


def test_a2_and_a3_share_parents_slots_and_draws_and_every_edit_regenerates(
    make_samples, make_test_spec, editor_calibration, tmp_path
):
    parents = make_samples(20, n_samples=1000)
    row_of = {record_id: i for i, record_id in enumerate(parents.record_id)}
    runs = {
        arm: _editor_run(make_test_spec(arm), parents, editor_calibration, tmp_path, arm)
        for arm in ("A2", "A3")
    }
    (_, b2, rows2), (_, b3, rows3) = runs["A2"], runs["A3"]
    for step in b2:
        assert np.array_equal(b2[step].record_id, b3[step].record_id)

    # The same (step, slot, attempt) is the same draw and parent under both policies; controls differ.
    by_key = {(r["step"], r["slot"], r["attempt"]): r for r in rows3}
    shared = [
        (r, by_key[(r["step"], r["slot"], r["attempt"])])
        for r in rows2
        if (r["step"], r["slot"], r["attempt"]) in by_key
    ]
    assert len(shared) >= 30 * 4
    assert all(
        a["seed_key"] == b["seed_key"] and a["parent_record_id"] == b["parent_record_id"] for a, b in shared
    )
    assert sum(a["controls_requested"] != b["controls_requested"] for a, b in shared) == len(shared)

    for rows, batches in ((rows2, b2), (rows3, b3)):
        for row in rows:
            if row["decision"] != "accepted":
                continue
            recipe = Recipe(
                row["parent_record_id"],
                row["generator"],
                row["version"],
                json.loads(row["controls_requested"]),
                row["seed_key"],
                row["gap_report_id"],
            )
            assert recipe.recipe_id == row["recipe_id"]
            regenerated = editor.edit(parents.x[row_of[row["parent_record_id"]]], recipe, FS_HZ).x
            assert np.array_equal(regenerated, batches[row["step"]].x[row["slot"]])


def test_a_rejected_edit_retries_the_same_parent_and_slot_with_the_next_draw(
    make_samples, make_test_spec, editor_calibration, tmp_path, monkeypatch
):
    calls = itertools.count()
    real = validate.evaluate

    def reject_every_first_attempt(*args, **kwargs):
        checks, realized = real(*args, **kwargs)
        if next(calls) % 2 == 0:
            checks = [checks[0], validate.Check("alignment", "1", 1.0, False, blocking=True)]
        return checks, realized

    monkeypatch.setattr(validate, "evaluate", reject_every_first_attempt)
    spec = make_test_spec("A3", transform={**asdict(EditorConfig()), "alignment_tol": 10.0})
    parents = make_samples(20, n_samples=1000)
    stream, batches, rows = _editor_run(spec, parents, editor_calibration, tmp_path, "retry")

    slots = {}
    for row in rows:
        slots.setdefault((row["step"], row["slot"]), []).append(row)
    assert len(slots) == 30 * 4
    for (step, slot), tries in slots.items():
        first, second = tries
        assert (first["attempt"], first["decision"], first["reason"]) == (0, "rejected", "alignment@1")
        assert (second["attempt"], second["decision"]) == (1, "accepted")
        assert first["parent_record_id"] == second["parent_record_id"]
        assert first["seed_key"] != second["seed_key"]
        assert batches[step].transformed[slot]
    assert stream.exposure["candidates"] == 2 * 30 * 4 and stream.exposure["exhausted_slots"] == 0


def test_an_exhausted_slot_is_recorded_and_keeps_its_parent(
    make_samples, make_test_spec, editor_calibration, tmp_path
):
    spec = make_test_spec("A3", transform={**asdict(EditorConfig()), "alignment_tol": -1.0, "max_retries": 2})
    parents = make_samples(20, n_samples=1000)
    row_of = {record_id: i for i, record_id in enumerate(parents.record_id)}
    stream, batches, rows = _editor_run(spec, parents, editor_calibration, tmp_path, "exhausted")

    assert stream.exposure["exhausted_slots"] == stream.exposure["transform_slots"] == 30 * 4
    assert stream.exposure["accepted"] == 0 and len(rows) == 3 * 30 * 4
    assert {row["attempt"] for row in rows} == {0, 1, 2} and {row["decision"] for row in rows} == {"rejected"}
    for batch in batches.values():
        assert not batch.transformed.any()
        assert np.array_equal(batch.x, parents.x[[row_of[r] for r in batch.record_id]])


def test_editor_arms_refuse_a_missing_or_mismatched_calibration(
    make_samples, make_test_spec, editor_calibration
):
    parents = make_samples(5, n_samples=1000)
    with pytest.raises(CalibrationError, match="needs its regime's calibration"):
        TrainStream(make_test_spec("A2"), parents)
    other = {
        **editor_calibration,
        "envelope": {**editor_calibration["envelope"], "static_noise": {"amplitude_mv": [0, 1]}},
    }
    with pytest.raises(CalibrationError, match="another envelope"):
        TrainStream(make_test_spec("A2"), parents, calibration=other)
