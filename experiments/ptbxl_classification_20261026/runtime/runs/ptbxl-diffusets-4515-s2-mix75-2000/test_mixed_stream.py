"""Scientific sampling/augmentation contract checks without training on ECGs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from syntide.data.schema import Samples
from syntide.eval.stats import val_scores
from syntide.eval.tracking import FilesTracker
from syntide.generate import generic
from syntide.generate.ledger import LedgerWriter
from syntide.trial import Checkpoint, TrainerInfo, make_spec, open_trial


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module('mixed_s2_stream', Path(__file__).with_name('mixed_stream.py'))
MixedStream, S2_CONFIG = MODULE.MixedStream, MODULE.S2_CONFIG
OLD = load_module('original_mixed_stream', Path(__file__).parents[1] /
                  'ptbxl-diffusets-4515-mix75-2000/mixed_stream.py')


def samples(n=20, duration=4):
    return Samples(
        record_id=np.asarray([f'record-{i}' for i in range(n)]),
        subject_uid=np.asarray([f'patient-{i}' for i in range(n)]),
        labels=(np.arange(n)[:, None] + np.arange(5)[None, :]) % 2 == 0,
        artifacts=np.column_stack((np.arange(n) % 3 == 0, np.zeros((n, 3), bool))),
        x=np.broadcast_to(np.arange(n, dtype=np.float32)[:, None, None] / 100,
                          (n, 12, duration)).copy(),
    )


def build(n_real=13, n_generated=7, steps=6, seed=0, ledger=None, duration=4):
    spec = SimpleNamespace(transform_family='generic', transform_fraction=.25,
                           transform=dict(S2_CONFIG), batch_size=64, max_steps=steps,
                           val_every=2, replicate_seed=seed, arm='S2mix')
    return MixedStream(spec, samples(n_real + n_generated, duration), ledger=ledger,
                       n_real=n_real, real_per_batch=48)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_full_three_seed_schedule_matches_previous_source_draws(seed):
    stream = build(8579, 4515, 2000, seed)
    old_spec = SimpleNamespace(transform_family='none', transform_fraction=0,
                              batch_size=64, max_steps=2000, val_every=100, replicate_seed=seed)
    old = OLD.MixedStream(old_spec, stream.parents, n_real=8579, real_per_batch=48)
    draws, masks = [], []
    for step in range(1, 2001):
        ix, mask = stream.parent_positions(step), stream.slot_mask(step)
        assert np.array_equal(ix, old.parent_positions(step))
        assert (ix < 8579).sum() == 48 and (ix >= 8579).sum() == 16
        assert mask.sum() == 16
        assert ((ix < 8579) & mask).sum() == 12
        assert ((ix >= 8579) & mask).sum() == 4
        draws.append(ix)
        masks.append(mask)
    draws, masks = np.stack(draws), np.stack(masks)
    counts = np.bincount(draws.ravel(), minlength=13094)
    assert set(counts[:8579]) == {11, 12}
    assert set(counts[8579:]) == {7, 8}
    assert masks.sum() == 32000
    assert ((draws < 8579) & masks).sum() == 24000
    assert ((draws >= 8579) & masks).sum() == 8000


def test_native_augmentation_ledger_replays_exactly_and_preserves_targets(tmp_path):
    ledger = LedgerWriter(tmp_path / 'recipes.parquet', flush_every=20)
    stream = build(steps=4, ledger=ledger, duration=100)
    before = stream.parents.x.copy()
    batches = list(stream)
    ledger.close()
    rows = pq.read_table(tmp_path / 'recipes.parquet').to_pylist()
    assert len(rows) == 4 * 16
    assert np.array_equal(stream.parents.x, before)
    for batch in batches:
        ix = stream.batch_indices[batch.step - 1]
        mask = batch.transformed
        assert np.array_equal(batch.y, stream.parents.labels[ix].astype(np.float32))
        assert np.array_equal(batch.record_id, stream.parents.record_id[ix])
        assert np.array_equal(batch.x[~mask], before[ix[~mask]])
        assert np.isfinite(batch.x).all() and batch.x.dtype == np.float32
        assert mask.sum() == 16
    for row in rows:
        step, slot = row['step'], row['slot']
        ix = stream.batch_indices[step - 1, slot]
        controls = json.loads(row['controls_requested'])
        assert row['generator'] == 'per_lead_aug' and row['decision'] == 'accepted'
        assert row['parent_record_id'] == str(stream.parents.record_id[ix])
        assert all(.95 <= gain <= 1.05 for gain in controls['gain'])
        assert abs(controls['shift_samples']) <= 25
        assert 0 <= controls['noise_rms_mv'] <= .05
        assert 0 <= controls['wander_mv'] <= 1.75
        reconstructed = generic.apply(before[ix], controls, row['seed_key'], 100)
        assert np.array_equal(batches[step - 1].x[slot], reconstructed)
    audit = stream.write_audit(tmp_path)
    assert audit['every_batch_has_exact_mix'] and audit['every_batch_has_exact_s2_dose']
    assert audit['exposure']['augmented_real_presentations'] == 48
    assert audit['exposure']['augmented_generated_presentations'] == 16
    assert audit['exposure']['transform_slots'] == audit['exposure']['accepted'] == 64
    assert stream.write_audit(tmp_path) == audit


def test_determinism_and_single_use(tmp_path):
    a, b, c = build(seed=1), build(seed=1), build(seed=2)
    ba, bb, bc = list(a), list(b), list(c)
    assert all(np.array_equal(x.x, y.x) and np.array_equal(x.transformed, y.transformed)
               for x, y in zip(ba, bb, strict=True))
    assert not np.array_equal(ba[0].x, bc[0].x)
    with pytest.raises(RuntimeError, match='only once'):
        iter(a)
    partial = build()
    next(iter(partial))
    with pytest.raises(ValueError, match='every planned batch'):
        partial.write_audit(tmp_path)


def test_trial_finish_uses_native_ledger_and_exposure(tmp_path):
    parents, val = samples(20), samples(12)
    spec = make_spec(protocol='s2-test', regime='pool50', arm='S2mix', replicate_seed=0,
        source_role='pool_plus_offline', transform_family='generic', sampling_policy='fixed',
        generator='diffusets+s2', validators=[], transform=dict(S2_CONFIG), dataset='test',
        source_sha256='a' * 64, manifest_sha256='b' * 64, calibration_sha256=None,
        batch_size=64, max_steps=4, val_every=2, transform_fraction=.25,
        registry='test', manifest='test', calibration=None)
    tracker = FilesTracker(tmp_path)
    with open_trial(spec, parents, val, tracker, tmp_path) as trial:
        trial._stream = stream = MixedStream(spec, parents, tracker, trial._ledger, n_real=13)
        list(trial.batches())
        probs = .2 + .6 * val.labels
        history = [{'step': step, **val_scores(val, probs)} for step in [2, 4]]
        best, final = tmp_path / 'best.pt', tmp_path / 'final.pt'
        best.write_bytes(b'test-best')
        final.write_bytes(b'test-final')
        result = trial.finish(TrainerInfo(name='test', version='1', config={}, device='cpu', predictor='test'),
            best=Checkpoint(best, 2), final=Checkpoint(final, 4), val_history=history,
            val_probs={'best': probs, 'final': probs}, init_sha256='c' * 64, wall_time_s=1.)
        audit = stream.write_audit(tmp_path)
    assert result['status'] == 'complete' and result['exposure'] == audit['exposure']
    assert pq.read_table(tmp_path / 'recipes.parquet').num_rows == 64
    assert pq.read_table(tmp_path / 'predictions_val.parquet').num_rows == 120


def test_rejects_changed_s2_config_and_source_dosage():
    valid = build()
    valid.spec.transform['max_shift_s'] = .5
    with pytest.raises(ValueError, match='native half-strength'):
        MixedStream(valid.spec, valid.parents, n_real=13)
    valid.spec.transform['max_shift_s'] = .25
    with pytest.raises(ValueError, match='divide by four'):
        MixedStream(valid.spec, valid.parents, n_real=13, real_per_batch=47)
