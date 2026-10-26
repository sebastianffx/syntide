#!/usr/bin/env python3
"""Three frozen seed trials: all pool50 + DiffuSETS, 48:16 batches, native S2 dose.

The existing reference trainer and generic augmentation are used unchanged.
Each batch augments 12 of 48 real and 4 of 16 generated slots. Evaluation opens
the held-out test fold only after all three trials and their audits complete.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from syntide.data.adapter import PtbXlAdapter
from syntide.data.challenge import load_manifest
from syntide.data.leakage import check_lineage, check_manifest
from syntide.data.schema import Samples, SUPERCLASSES
from syntide.eval.holdout import evaluate
from syntide.eval.tracking import FilesTracker
from syntide.generate.plan import write_plan
from syntide.trial import is_complete, make_spec, open_trial
from syntide_trainers.torch_loop import ResNet1d, TorchLoopConfig, fit, state_sha256, trainer_info

from mixed_stream import MixedStream, S2_CONFIG

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREVIOUS = PROJECT / 'runs/ptbxl-diffusets-4515-mix75-2000'
S2_ROOT = PROJECT / 'runs/ptbxl-aug-sweep/scarcity50/S2'
MANIFEST = PROJECT / 'manifests/ptbxl-scaling/pool50.json'
REGISTRY = PROJECT / 'datasets/timenet-registry'
N_REAL, N_SYNTH, STEPS = 8579, 4515, 2000
SEEDS = (0, 1, 2)
POLICY = 'provisional_conditioning_parent_superclasses'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def freeze(path, payload):
    if path.exists():
        if read(path) != payload:
            raise ValueError(f'Frozen file differs: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')


class ProgressTracker(FilesTracker):
    def __init__(self, run_dir, seed):
        super().__init__(run_dir)
        self.seed = seed

    def log_scalars(self, values, step):
        super().log_scalars(values, step)
        if 'val/auroc_overall' in values:
            print(f"seed {self.seed}, step {step:4d}/{STEPS}: val AUROC {values['val/auroc_overall']:.6f}, "
                  f"noisy {values['val/auroc_artifact']:.6f}, clean {values['val/auroc_clean']:.6f}", flush=True)


def audit_trial(spec, info, baseline):
    directory = ROOT / spec.run_name
    result = read(directory / 'result.json')
    assert result['status'] == 'complete' and result['trial_id'] == spec.trial_id
    assert result['final_step'] == STEPS
    assert result['trainer']['fingerprint'] == info.fingerprint
    assert result['trainer']['init_sha256'] == baseline['trainer']['init_sha256']
    history = result['val_history']
    assert [r['step'] for r in history] == list(range(100, STEPS + 1, 100))
    best = max(history, key=lambda r: r['val/auroc_overall'])
    assert best['step'] == result['best_step'] and best['val/auroc_overall'] == result['best_val_auroc']
    exposure = result['exposure']
    for key, expected in {'steps': 2000, 'presentations': 128000, 'real_presentations': 96000,
                          'generated_presentations': 32000, 'distinct_real_records': 8579,
                          'distinct_generated_records': 4515, 'transform_slots': 32000,
                          'candidates': 32000, 'accepted': 32000, 'rejected': 0}.items():
        assert exposure[key] == expected, (key, exposure[key], expected)
    sampling = read(directory / 'sampling_audit.json')
    assert sampling['all_rows_seen'] and sampling['every_batch_has_exact_mix']
    with np.load(directory / 'sampling_trace.npz', allow_pickle=False) as trace:
        indices = trace['indices']
        record_ids = trace['record_id']
        transformed = trace['transformed']
        assert indices.shape == transformed.shape == (STEPS, 64)
        assert transformed.dtype == np.bool_
        is_real = indices < N_REAL
        assert np.all(is_real.sum(axis=1) == 48)
        assert np.all((is_real & transformed).sum(axis=1) == 12)
        assert np.all((~is_real & transformed).sum(axis=1) == 4)
        assert len(np.unique(indices)) == N_REAL + N_SYNTH
    ledger = pq.read_table(directory / 'recipes.parquet').to_pylist()
    assert len(ledger) == 32000
    observed = np.zeros((STEPS, 64), dtype=bool)
    for recipe in ledger:
        step, slot = recipe['step'] - 1, recipe['slot']
        assert not observed[step, slot]
        observed[step, slot] = True
        assert recipe['parent_record_id'] == str(record_ids[indices[step, slot]])
        assert recipe['generator'] == 'per_lead_aug' and recipe['version'] == '1'
        assert recipe['decision'] == 'accepted'
        controls = json.loads(recipe['controls_requested'])
        assert len(controls['gain']) == 12 and all(.95 <= v <= 1.05 for v in controls['gain'])
        assert -25 <= controls['shift_samples'] <= 25
        assert 0 <= controls['noise_rms_mv'] <= .05
        assert 0 <= controls['wander_mv'] <= 1.75 and .05 <= controls['wander_hz'] <= .7
    assert np.array_equal(observed, transformed)
    for name, checkpoint in result['provenance']['checkpoints'].items():
        assert sha(directory / checkpoint['path']) == checkpoint['sha256'], name
    audit = {'passed': True, 'seed': spec.replicate_seed, 'steps': STEPS,
             'best_step': result['best_step'], 'best_val_auroc': result['best_val_auroc'],
             'same_initial_weights_as_historical_s2': True, 'all_rows_seen': True,
             'every_batch_has_48_real_16_generated': True,
             'every_batch_augments_12_real_4_generated': True,
             'all_32000_native_s2_recipes_verified': True,
             'result_sha256': sha(directory / 'result.json'),
             'recipes_sha256': sha(directory / 'recipes.parquet'),
             'sampling_audit_sha256': sha(directory / 'sampling_audit.json')}
    freeze(directory / 'training_audit.json', audit)
    return audit


def main():
    os.chdir(PROJECT)
    adapter = PtbXlAdapter(REGISTRY, 'physionet/ptb-xl@1.0.3')
    manifest = load_manifest(MANIFEST)
    index, source = adapter.index(), adapter.source()
    check_manifest(manifest, index, source)
    assert tuple(map(len, (manifest.roles['pool'], manifest.roles['val'], manifest.roles['test']))) == (N_REAL, 2146, 2158)
    baselines = {seed: read(S2_ROOT / f'seed{seed}/result.json') for seed in SEEDS}
    for seed in SEEDS:
        original_spec = read(S2_ROOT / f'seed{seed}/trial.json')
        assert original_spec['transform'] == S2_CONFIG
        assert original_spec['transform_fraction'] == .25 and original_spec['batch_size'] == 64
        assert original_spec['replicate_seed'] == seed
    cfg = TorchLoopConfig(device=baselines[0]['trainer']['device'])
    info = trainer_info(cfg)
    assert all(b['status'] == 'complete' and b['trainer']['fingerprint'] == info.fingerprint for b in baselines.values())
    assert info.device == 'mps' and torch.backends.mps.is_available()
    inputs = {str(p.relative_to(PROJECT)): sha(p) for p in sorted((PREVIOUS / 'data').iterdir()) if p.is_file()}
    previous_freeze = read(PREVIOUS / 'experiment_freeze.json')
    for relative, expected in previous_freeze['inputs_sha256'].items():
        assert sha(PREVIOUS / relative) == expected
    data_manifest = read(PREVIOUS / 'data/manifest.json')
    assert data_manifest['count'] == N_SYNTH and data_manifest['label_policy'] == POLICY
    for entry in data_manifest['files'].values():
        assert sha(PREVIOUS / entry['path']) == entry['sha256']
    provenance = read(PREVIOUS / 'data/provenance.json')
    assert provenance['classification_manifest_sha256'] == manifest.sha256
    assert provenance['source_sha256'] == source['sha256']
    assert provenance['heldout_record_overlap'] == provenance['heldout_patient_overlap'] == 0
    code_paths = [ROOT / 'run_experiment.py', ROOT / 'mixed_stream.py',
                  PROJECT / 'trainers/src/syntide_trainers/torch_loop.py', PROJECT / 'src/syntide/trial.py',
                  PROJECT / 'src/syntide/generate/stream.py', PROJECT / 'src/syntide/generate/generic.py',
                  PROJECT / 'src/syntide/generate/ledger.py', PROJECT / 'src/syntide/eval/stats.py',
                  PROJECT / 'src/syntide/eval/holdout.py']
    code = {str(p.relative_to(PROJECT)): sha(p) for p in code_paths}
    original_paths = [p for seed in SEEDS for p in (S2_ROOT / f'seed{seed}').iterdir() if p.is_file()]
    original_paths += [p for p in (PROJECT / 'runs/ptbxl-scaling').glob('pool*/real/seed0/*') if p.is_file()]
    originals = {str(p.relative_to(PROJECT)): sha(p) for p in original_paths}
    protocol = {
        'id': ROOT.name, 'seeds': list(SEEDS), 'max_steps': STEPS,
        'step_budget_basis': 'Retain 2000 steps per seed from the immediately preceding trial.',
        'real_records': N_REAL, 'generated_records': N_SYNTH, 'batch_size': 64,
        'real_per_batch': 48, 'generated_per_batch': 16,
        'augmentation': {'name': 'S2', 'family': 'generic', 'fraction': .25, 'strength_relative_to_A1': .5,
                         'dose': .125, 'real_slots': 12, 'generated_slots': 4, 'envelope': S2_CONFIG,
                         'implementation': 'unchanged syntide.generate.generic; labels preserved'},
        'sampling': 'Independent shuffled exhaustive source cycles, identical to prior mixed trial source order; random source-stratified augmentation slots.',
        'val_every': 100, 'model': asdict(cfg.model), 'optimizer': asdict(cfg.optim), 'device': info.device,
        'loss': 'BCEWithLogitsLoss, equal weight per example and per class',
        'checkpoint_selection': 'first maximum validation overall macro-AUROC across steps100..2000',
        'label_policy': POLICY, 'class_order': list(SUPERCLASSES),
        'diagnostic_labels_verified_on_generated_signals': False,
        'manifest': str(MANIFEST.relative_to(PROJECT)), 'manifest_sha256': manifest.sha256,
        'source_sha256': source['sha256'], 'input_files_sha256': inputs, 'code_sha256': code,
        'comparisons': {'scaling': 'historical seed0,12000 steps; also restrict selection to<=2000',
                        'S2': 'historical seeds0,1,2,6389 artifact-depleted real ECGs,10000 steps; all best checkpoints<=2000',
                        'caution': 'Compared with historical S2, new recipe restores2190 real ECGs and adds4515 generated ECGs; source-stratified dose; not an isolated generated-data effect.',
                        'uncertainty': 'mean and sample SD across3 seeds; SD is not a confidence interval',
                        'test': 'exploratory; prior test results known; all3 trials frozen and complete before test scoring'},
        'conditioning_caveat': {'generated_rows_with_parents_outside_real_pool': 2212,
                                'additional_source_records': 1794, 'additional_source_patients': 1765,
                                'heldout_patient_overlap': 0},
    }
    freeze(ROOT / 'protocol.json', protocol)
    specs = [make_spec(protocol=ROOT.name, regime='pool50', arm='mix75_s2', replicate_seed=seed,
        source_role='pool_plus_offline_diffusets', transform_family='generic', sampling_policy='fixed',
        generator='diffusets4515+S2@' + sha(ROOT / 'protocol.json'), validators=[], transform=S2_CONFIG,
        dataset='physionet/ptb-xl@1.0.3', source_sha256=source['sha256'], manifest_sha256=manifest.sha256,
        calibration_sha256=None, batch_size=64, max_steps=STEPS, val_every=100, transform_fraction=.25,
        registry=str(REGISTRY), manifest=str(MANIFEST), calibration=None) for seed in SEEDS]
    write_plan(specs, ROOT.parent)
    frozen = {'protocol_sha256': sha(ROOT / 'protocol.json'), 'plan_sha256': sha(ROOT / 'plan.json'),
              'inputs_sha256': inputs, 'code_sha256': code, 'original_files_sha256': originals}
    if (ROOT / 'experiment_freeze.json').exists():
        assert {k: v for k, v in read(ROOT / 'experiment_freeze.json').items() if k != 'frozen_at'} == frozen
    else:
        assert not (ROOT / 'test_claim.json').exists()
        assert not any(ROOT.glob('pool50/mix75_s2/seed*/result.json'))
        freeze(ROOT / 'experiment_freeze.json', {**frozen, 'frozen_at': datetime.now(UTC).isoformat()})

    # Load only training and validation here. Held-out waveforms load in evaluate() after all trials.
    real, val = adapter.load(manifest.roles['pool']), adapter.load(manifest.roles['val'])
    with np.load(PREVIOUS / 'data/generated.npz', allow_pickle=False) as archive:
        generated = {k: archive[k] for k in archive.files}
    assert generated['x'].shape == (N_SYNTH, 12, 1000) and generated['x'].dtype == np.float32
    assert np.isfinite(generated['x']).all() and generated['labels'].dtype == np.bool_
    source_rows = {r['record_id']: r for r in index.to_pylist()}
    check_lineage([(str(record), [str(root)], [str(subject)]) for record, root, subject in
                   zip(generated['record_id'], generated['root_record_id'], generated['root_subject_uid'], strict=True)],
                  manifest, index, allowed_roles=['pool', 'unused'])
    expected_labels = np.asarray([[c in source_rows[str(root)]['labels'] for c in SUPERCLASSES]
                                  for root in generated['root_record_id']], dtype=bool)
    assert np.array_equal(generated['labels'], expected_labels)
    assert np.array_equal(generated['subject_uid'], generated['root_subject_uid'])
    combined = Samples(**{k: np.concatenate([getattr(real, k), generated[k]]) for k in
                          ('record_id', 'subject_uid', 'labels', 'artifacts', 'x')})
    assert len(set(combined.record_id)) == N_REAL + N_SYNTH
    del real, generated
    audits = []
    for spec in specs:
        seed, run_dir = spec.replicate_seed, ROOT / spec.run_name
        torch.manual_seed(spec.init_seed)
        probe = ResNet1d(**asdict(cfg.model))
        assert state_sha256(probe) == baselines[seed]['trainer']['init_sha256']
        del probe
        if not is_complete(run_dir, info):
            assert not (ROOT / 'test_claim.json').exists()
            tracker = ProgressTracker(run_dir, seed)
            with open_trial(spec, combined, val, tracker, run_dir) as trial:
                stream = MixedStream(spec, combined, tracker, trial._ledger,
                                     n_real=N_REAL, real_per_batch=48)
                trial._stream = stream
                print(f'Starting seed{seed}: {N_REAL}+{N_SYNTH} ECGs,48+16 per batch,12+4 S2-augmented.', flush=True)
                result = fit(trial, cfg, info)
                stream.write_audit(run_dir)
            print(f"Completed seed{seed}: best val {result['best_val_auroc']:.6f} at {result['best_step']}; {result['wall_time_s']:.1f}s", flush=True)
        audits.append(audit_trial(spec, info, baselines[seed]))
    for path, expected in {**inputs, **originals, **code}.items():
        assert sha(PROJECT / path) == expected, path
    freeze(ROOT / 'training_audit.json', {'passed': True, 'trials': audits,
           'all_original_inputs_and_results_unchanged': True, 'all_training_code_unchanged': True})
    scores = evaluate(ROOT)
    assert len(scores) == 3 and all(s['n_test'] == 2158 for s in scores)
    for path, expected in originals.items():
        assert sha(PROJECT / path) == expected, path
    freeze(ROOT / 'evaluation_audit.json', {'passed': True, 'seeds': list(SEEDS), 'n_test': 2158,
           'all_trials_complete_before_test': True, 'validation_selected_checkpoints': [s['step'] for s in scores],
           'original_outputs_unchanged': True,
           'score_files_sha256': {spec.run_name: sha(ROOT / spec.run_name / 'test_scores.json') for spec in specs}})
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == '__main__':
    main()
