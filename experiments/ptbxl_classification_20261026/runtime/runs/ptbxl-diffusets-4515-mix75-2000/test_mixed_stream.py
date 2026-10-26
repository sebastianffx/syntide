"""Checks the sampler contract without training or loading any ECG data."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from syntide.data.schema import Samples

module_spec = importlib.util.spec_from_file_location("mixed_stream", Path(__file__).with_name("mixed_stream.py"))
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
MixedStream = module.MixedStream


def build(n_real=11, n_generated=7, real_per_batch=3, generated_per_batch=1, steps=30, seed=0):
    n = n_real + n_generated
    sample = Samples(
        record_id=np.asarray([f"record-{i}" for i in range(n)]),
        subject_uid=np.asarray([f"patient-{i}" for i in range(n)]),
        labels=(np.arange(n)[:, None] + np.arange(5)[None, :]) % 2 == 0,
        artifacts=np.zeros((n, 4), dtype=bool),
        x=np.broadcast_to(np.arange(n, dtype=np.float32)[:, None, None], (n, 12, 1)).copy(),
    )
    spec = SimpleNamespace(
        transform_family="none", transform_fraction=0.0,
        batch_size=real_per_batch + generated_per_batch,
        max_steps=steps, val_every=10, replicate_seed=seed,
    )
    return MixedStream(spec, sample, n_real=n_real, real_per_batch=real_per_batch)


def test_exact_full_trial_mix_coverage_and_trace(tmp_path):
    stream = build(8579, 4515, 48, 16, 2000)
    for step, batch in enumerate(stream, 1):
        rows = batch.x[:, 0, 0].astype(int)
        assert batch.step == step
        assert np.sum(rows < 8579) == 48
        assert np.sum(rows >= 8579) == 16
        assert np.array_equal(rows, stream.batch_indices[step - 1])
        assert np.array_equal(batch.y, stream.parents.labels[rows].astype(np.float32))
        assert np.array_equal(batch.record_id, stream.parents.record_id[rows])
        assert batch.x.dtype == batch.y.dtype == np.float32
        assert not batch.transformed.any()
    assert stream.consumed and stream.steps_served == 2000
    assert stream.exposure["presentations"] == 128000
    assert stream.exposure["real_presentations"] == 96000
    assert stream.exposure["generated_presentations"] == 32000
    assert stream.exposure["distinct_real_records"] == 8579
    assert stream.exposure["distinct_generated_records"] == 4515
    assert stream.exposure["transform_slots"] == 0
    assert set(stream.row_counts[:8579]) == {11, 12}
    assert set(stream.row_counts[8579:]) == {7, 8}
    audit = stream.write_audit(tmp_path)
    assert audit["all_rows_seen"] and audit["every_batch_has_exact_mix"]
    assert stream.write_audit(tmp_path) == audit
    assert len((tmp_path / "batch_composition.csv").read_text().splitlines()) == 2001
    assert len((tmp_path / "row_exposure.csv").read_text().splitlines()) == 13095


def test_per_source_cycles_exhaust_before_repeating_even_across_batch_boundaries():
    stream = build(n_real=5, n_generated=2, real_per_batch=7, generated_per_batch=3, steps=10)
    for source, size, count in ((0, 5, 7), (1, 2, 3)):
        draws = np.concatenate([stream._positions(step, size, count, source) for step in range(1, 11)])
        for start in range(0, len(draws) - size + 1, size):
            assert sorted(draws[start:start + size]) == list(range(size))
        assert np.array_equal(stream._positions(2, size, count, source), draws[count:count * 2])


def test_determinism_seed_change_and_independent_sources():
    a, b, c = build(seed=4), build(seed=4), build(seed=5)
    a_rows = np.stack([batch.x[:, 0, 0] for batch in a])
    b_rows = np.stack([batch.x[:, 0, 0] for batch in b])
    c_rows = np.stack([batch.x[:, 0, 0] for batch in c])
    assert np.array_equal(a_rows, b_rows)
    assert not np.array_equal(a_rows, c_rows)
    other = build(n_generated=3, generated_per_batch=4, seed=4)
    for step in range(1, 31):
        assert np.array_equal(a._positions(step, 11, 3, 0), other._positions(step, 11, 3, 0))


def test_single_use_partial_audit_and_parent_immutability(tmp_path):
    stream = build()
    before = stream.parents.x.copy()
    iterator = iter(stream)
    batch = next(iterator)
    batch.x[:] = -99
    assert np.array_equal(stream.parents.x, before)
    assert not stream.consumed
    with pytest.raises(RuntimeError, match="only once"):
        iter(stream)
    with pytest.raises(ValueError, match="every planned batch"):
        stream.write_audit(tmp_path)
    list(iterator)
    assert stream.consumed
    with pytest.raises(RuntimeError, match="only once"):
        iter(stream)


def test_rejects_changed_existing_trace(tmp_path):
    a, b = build(seed=1), build(seed=2)
    list(a)
    list(b)
    a.write_audit(tmp_path)
    with pytest.raises(ValueError, match="existing sampling trace differs"):
        b.write_audit(tmp_path)
