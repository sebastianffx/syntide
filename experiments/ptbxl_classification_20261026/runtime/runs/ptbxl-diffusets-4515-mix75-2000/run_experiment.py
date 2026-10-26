#!/usr/bin/env python3
"""Train/evaluate one frozen seed-0 trial with exact 48-real/16-generated batches.

Run from the project environment: .venv/bin/python runs/ptbxl-diffusets-4515-mix75-2000/run_experiment.py
Preparation is separate (prepare_data.py). Original runs and source bundles are read-only.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
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

from mixed_stream import MixedStream

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASELINE = PROJECT / 'runs/ptbxl-scaling/pool50/real/seed0'
MANIFEST = PROJECT / 'manifests/ptbxl-scaling/pool50.json'
REGISTRY = PROJECT / 'datasets/timenet-registry'
N_REAL, N_SYNTH, STEPS = 8579, 4515, 2000
POLICY = 'provisional_conditioning_parent_superclasses'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


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
    def log_scalars(self, values, step):
        super().log_scalars(values, step)
        if 'val/auroc_overall' in values:
            print(f"step {step:4d}/{STEPS}: val AUROC {values['val/auroc_overall']:.6f}, "
                  f"noisy {values['val/auroc_artifact']:.6f}, clean {values['val/auroc_clean']:.6f}",
                  flush=True)


def main():
    os.chdir(PROJECT)
    adapter = PtbXlAdapter(REGISTRY, 'physionet/ptb-xl@1.0.3')
    manifest = load_manifest(MANIFEST)
    index, source = adapter.index(), adapter.source()
    check_manifest(manifest, index, source)
    assert tuple(map(len, (manifest.roles['pool'], manifest.roles['val'], manifest.roles['test']))) == (N_REAL, 2146, 2158)
    baseline = read(BASELINE / 'result.json')
    assert baseline['status'] == 'complete'
    assert baseline['provenance']['source_sha256'] == source['sha256']
    assert baseline['provenance']['manifest_sha256'] == manifest.sha256
    cfg = TorchLoopConfig(device=baseline['trainer']['device'])
    info = trainer_info(cfg)
    assert info.fingerprint == baseline['trainer']['fingerprint']
    assert info.device == 'mps' and torch.backends.mps.is_available()

    # Bind every prepared input and all numerical/stream implementation before training.
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'data').glob('*')) if p.is_file()}
    assert {'data/generated.npz', 'data/conditioning_labels.json', 'data/provenance.json', 'data/manifest.json'} <= inputs.keys()
    data_manifest = read(ROOT / 'data/manifest.json')
    assert data_manifest['count'] == N_SYNTH and data_manifest['label_policy'] == POLICY
    for entry in data_manifest['files'].values():
        assert inputs[entry['path']] == entry['sha256']
    provenance = read(ROOT / 'data/provenance.json')
    assert provenance['classification_manifest_sha256'] == manifest.sha256
    assert provenance['source_sha256'] == source['sha256']
    assert provenance['heldout_record_overlap'] == provenance['heldout_patient_overlap'] == 0
    code_paths = [ROOT / 'run_experiment.py', ROOT / 'mixed_stream.py', ROOT / 'prepare_data.py',
                  PROJECT / 'trainers/src/syntide_trainers/torch_loop.py', PROJECT / 'src/syntide/trial.py',
                  PROJECT / 'src/syntide/eval/stats.py', PROJECT / 'src/syntide/eval/holdout.py']
    code = {str(p.relative_to(PROJECT)): sha(p) for p in code_paths}
    originals = {str(p.relative_to(PROJECT)): sha(p) for p in BASELINE.iterdir() if p.is_file()}
    settings = {
        'id': ROOT.name, 'label_policy': POLICY,
        'label_choice_authorization': 'User: use the labels that makes more sense for those 4,515 ecgs',
        'real_records': N_REAL, 'generated_records': N_SYNTH,
        'real_per_batch': 48, 'generated_per_batch': 16, 'batch_size': 64,
        'max_steps': STEPS, 'val_every': 100, 'seed': 0,
        'sampling': 'independent shuffled cyclic source pools; exact source ratio in every batch',
        'model': asdict(cfg.model), 'optimizer': asdict(cfg.optim), 'device': info.device,
        'loss': 'BCEWithLogitsLoss, equally weighted examples/classes',
        'checkpoint_selection': 'first maximum validation overall macro-AUROC across steps 100..2000',
        'class_order': list(SUPERCLASSES), 'manifest': str(MANIFEST.relative_to(PROJECT)),
        'manifest_sha256': manifest.sha256, 'source_sha256': source['sha256'],
        'diagnostic_labels_verified_on_generated_signals': False,
        'comparison': 'Exploratory single-seed trial. Original test results already known. Original scaling budget 12000; also compare histories restricted to <=2000.',
        'inputs_sha256': inputs, 'code_sha256': code,
    }
    freeze(ROOT / 'protocol.json', settings)
    spec = make_spec(protocol=ROOT.name, regime='pool50', arm='mix75', replicate_seed=0,
        source_role='pool_plus_offline_diffusets', transform_family='none', sampling_policy='none',
        generator='diffusets@1', validators=[], transform={
            'offline_loader': 'run_experiment.py', 'protocol_sha256': sha(ROOT / 'protocol.json'),
            'label_policy': POLICY, 'sampling': settings['sampling'],
            'real_per_batch': 48, 'generated_per_batch': 16, 'data_sha256': inputs,
        }, dataset='physionet/ptb-xl@1.0.3', source_sha256=source['sha256'],
        manifest_sha256=manifest.sha256, calibration_sha256=None, batch_size=64,
        max_steps=STEPS, val_every=100, transform_fraction=0.0,
        registry=str(REGISTRY), manifest=str(MANIFEST), calibration=None)
    run_dir = ROOT / spec.run_name
    write_plan([spec], ROOT.parent)
    frozen = {'protocol_sha256': sha(ROOT / 'protocol.json'), 'plan_sha256': sha(ROOT / 'plan.json'),
              'inputs_sha256': inputs, 'code_sha256': code, 'baseline_files_sha256': originals}
    freeze_path = ROOT / 'experiment_freeze.json'
    if freeze_path.exists():
        previous = read(freeze_path)
        assert {k: v for k, v in previous.items() if k != 'frozen_at'} == frozen
    else:
        assert not (ROOT / 'test_claim.json').exists() and not (run_dir / 'result.json').exists()
        freeze(freeze_path, {**frozen, 'frozen_at': datetime.now(UTC).isoformat()})

    torch.manual_seed(spec.init_seed)
    probe = ResNet1d(**asdict(cfg.model))
    assert state_sha256(probe) == baseline['trainer']['init_sha256']
    del probe
    if not is_complete(run_dir, info):
        assert not (ROOT / 'test_claim.json').exists()
        real = adapter.load(manifest.roles['pool'])
        val = adapter.load(manifest.roles['val'])
        with np.load(ROOT / 'data/generated.npz', allow_pickle=False) as archive:
            generated = {k: archive[k] for k in archive.files}
        assert generated['x'].shape == (N_SYNTH, 12, 1000)
        assert generated['x'].dtype == np.float32 and np.isfinite(generated['x']).all()
        assert generated['labels'].shape == (N_SYNTH, 5) and generated['labels'].dtype == np.bool_
        assert len(set(generated['record_id'])) == N_SYNTH
        source_rows = {r['record_id']: r for r in index.to_pylist()}
        root_ids = generated['root_record_id']
        root_subjects = generated['root_subject_uid']
        check_lineage([(str(record), [str(root)], [str(subject)]) for record, root, subject in
                       zip(generated['record_id'], root_ids, root_subjects, strict=True)],
                      manifest, index, allowed_roles=['pool', 'unused'])
        expected_labels = np.asarray([[c in source_rows[str(root)]['labels'] for c in SUPERCLASSES]
                                      for root in root_ids], dtype=bool)
        assert np.array_equal(generated['labels'], expected_labels)
        assert np.array_equal(generated['subject_uid'], root_subjects)
        assert all(source_rows[str(root)]['subject_uid'] == subject for root, subject in zip(root_ids, root_subjects, strict=True))
        combined = Samples(**{k: np.concatenate([getattr(real, k), generated[k]]) for k in
                              ('record_id', 'subject_uid', 'labels', 'artifacts', 'x')})
        assert len(set(combined.record_id)) == N_REAL + N_SYNTH
        tracker = ProgressTracker(run_dir)
        with open_trial(spec, combined, val, tracker, run_dir) as trial:
            stream = MixedStream(spec, combined, tracker, n_real=N_REAL, real_per_batch=48)
            trial._stream = stream
            print(f'Training {N_REAL} real + {N_SYNTH} generated; 48+16 per batch; {STEPS} steps.', flush=True)
            result = fit(trial, cfg, info)
            stream.write_audit(run_dir)
        assert result['exposure']['presentations'] == 128000
        print(f"Training complete: best val {result['best_val_auroc']:.6f} at {result['best_step']}; {result['wall_time_s']:.1f}s", flush=True)
    else:
        result = read(run_dir / 'result.json')
    assert result['final_step'] == STEPS and result['trainer']['init_sha256'] == baseline['trainer']['init_sha256']
    assert [r['step'] for r in result['val_history']] == list(range(100, STEPS + 1, 100))
    assert (run_dir / 'sampling_audit.json').is_file()
    sampling = read(run_dir / 'sampling_audit.json')
    assert sampling['every_batch_has_exact_mix'] and sampling['all_rows_seen']
    assert sampling['real_per_batch'] == 48 and sampling['generated_per_batch'] == 16
    assert result['exposure']['real_presentations'] == 96000
    assert result['exposure']['generated_presentations'] == 32000
    assert result['exposure']['distinct_real_records'] == N_REAL
    assert result['exposure']['distinct_generated_records'] == N_SYNTH
    for path, expected in inputs.items():
        assert sha(ROOT / path) == expected
    for path, expected in {**originals, **code}.items():
        assert sha(PROJECT / path) == expected
    freeze(ROOT / 'training_audit.json', {
        'passed': True, 'steps': STEPS, 'validation_checkpoints': 20,
        'initial_weights_match_scaling_seed0': True, 'trainer_matches_scaling': True,
        'original_outputs_unchanged': True, 'inputs_unchanged': True,
        'sampler_audit_sha256': sha(run_dir / 'sampling_audit.json'),
        'result_sha256': sha(run_dir / 'result.json'),
    })
    scores = evaluate(ROOT)
    assert len(scores) == 1 and scores[0]['n_test'] == 2158
    for path, expected in originals.items():
        assert sha(PROJECT / path) == expected
    freeze(ROOT / 'evaluation_audit.json', {
        'passed': True, 'n_test': 2158, 'selected_step': result['best_step'],
        'selection': 'validation overall AUROC only', 'baseline_outputs_unchanged': True,
        'test_scores_sha256': sha(run_dir / 'test_scores.json'),
    })
    print(json.dumps(scores[0], indent=2), flush=True)


if __name__ == '__main__':
    main()
