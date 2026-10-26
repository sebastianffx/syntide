#!/usr/bin/env python3
"""Build a compact result report and verify the reused real-only test comparator."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
TRIAL = ROOT / 'pool50/mix75/seed0'
BASE = PROJECT / 'runs/ptbxl-scaling/pool50/real/seed0'
CONTROL = PROJECT / 'runs/ptbxl-artifact-50-val100/scarcity50/C2/seed0'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    real, mixed = read(BASE / 'result.json'), read(TRIAL / 'result.json')
    control = read(CONTROL / 'result.json')
    real_test, mixed_test = read(CONTROL / 'test_scores.json'), read(TRIAL / 'test_scores.json')
    a = torch.load(BASE / 'best.pt', weights_only=True, map_location='cpu')
    b = torch.load(CONTROL / 'best.pt', weights_only=True, map_location='cpu')
    assert a['model_config'] == b['model_config']
    assert a['model'].keys() == b['model'].keys()
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
    assert a['step'] == b['step'] == 1200
    assert real['trainer'] == control['trainer']
    assert real_test['trial_id'] == control['trial_id']
    assert real['provenance']['source_sha256'] == control['provenance']['source_sha256']
    # The manifests differ by name/metadata but use identical validation and test records.
    manifest_a = read(PROJECT / real['provenance']['manifest'])
    manifest_b = read(PROJECT / control['provenance']['manifest'])
    assert manifest_a['roles']['val'] == manifest_b['roles']['val']
    assert manifest_a['roles']['test'] == manifest_b['roles']['test']
    for directory, result in ((BASE, real), (CONTROL, control)):
        assert sha(directory / 'best.pt') == result['provenance']['checkpoints']['best']['sha256']
    equivalence = {
        'passed': True, 'all_model_parameters_and_buffers_exactly_equal': True,
        'model_config_equal': True, 'trainer_equal': True, 'validation_and_test_record_lists_equal': True,
        'checkpoint_step': 1200,
        'scaling_checkpoint': str((BASE / 'best.pt').relative_to(PROJECT)),
        'control_checkpoint': str((CONTROL / 'best.pt').relative_to(PROJECT)),
        'test_scores_source': str((CONTROL / 'test_scores.json').relative_to(PROJECT)),
        'file_sha256': {str(p.relative_to(PROJECT)): sha(p) for p in
                       (BASE / 'best.pt', CONTROL / 'best.pt', CONTROL / 'test_scores.json')},
        'note': 'File hashes differ because saved trial metadata differs. All model tensors are identical; reuse existing scores without new baseline test inference.',
    }
    write(ROOT / 'baseline_checkpoint_equivalence.json', equivalence)
    rb = max((r for r in real['val_history'] if r['step'] <= 2000), key=lambda r: r['val/auroc_overall'])
    mb = max(mixed['val_history'], key=lambda r: r['val/auroc_overall'])
    assert rb['step'] == 1200 and mb['step'] == mixed['best_step']
    rows = []
    for split, old, new in [('validation', rb, mb), ('test', real_test, mixed_test)]:
        prefix = 'val' if split == 'validation' else 'test'
        for subset in ('overall', 'artifact', 'clean'):
            key = f'{prefix}/auroc_{subset}'
            rows.append({'split': split, 'slice': subset, 'metric': 'macro_auroc',
                         'real50': old[key], 'real50_plus_diffusets4515': new[key],
                         'difference': new[key] - old[key],
                         'real_checkpoint_step': 1200, 'mixed_checkpoint_step': mb['step']})
    with (ROOT / 'test_and_validation_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sampling = read(TRIAL / 'sampling_audit.json')
    with np.load(TRIAL / 'sampling_trace.npz', allow_pickle=False) as trace:
        ix = trace['indices']
        assert ix.shape == (2000, 64)
        assert np.all((ix < 8579).sum(axis=1) == 48)
        assert np.all((ix >= 8579).sum(axis=1) == 16)
        assert len(np.unique(ix)) == 13094
    for filename, expected in sampling['files_sha256'].items():
        assert sha(TRIAL / filename) == expected
    write(ROOT / 'results_summary.json', {
        'status': 'complete', 'seed': 0, 'max_steps': 2000, 'best_step': mb['step'],
        'n_real': 8579, 'n_generated': 4515, 'batch_real': 48, 'batch_generated': 16,
        'validation_best': mb, 'validation_final': mixed['val_history'][-1], 'test': mixed_test,
        'comparison': rows, 'sampler_trace_independently_verified': True,
        'training_wall_time_s': mixed['wall_time_s'], 'seed_standard_deviation': None,
    })
    lines = [
        'PTB-XL + DiffuSETS classification trial — 26 October 2026',
        '',
        'Completed 2,000 optimizer steps; seed 0; 8,579 real ECGs + all 4,515 accepted generated ECGs.',
        'Every batch: 48 real + 16 generated. Total draws: 96,000 real + 32,000 generated.',
        'Every real ECG was used 11–12 times and every generated ECG 7–8 times.',
        'Same ResNet1d, AdamW (lr .001, weight decay .01), BCE loss and initial weights as scaling seed 0.',
        'Five multi-label targets: CD, HYP, MI, NORM, STTC. Generated targets are provisional conditioning-parent labels.',
        'Validation every 100 steps on 2,146 ECGs; best overall validation AUROC selects step 1,700.',
        'Test uses 2,158 ECGs, including 513 artifact-bearing and 1,645 clean records.',
        '',
        'Split       Slice       Real 50%     + DiffuSETS    Difference',
    ]
    lines += [f"{r['split']:<11} {r['slice']:<11} {r['real50']:.6f}     {r['real50_plus_diffusets4515']:.6f}      {r['difference']:+.6f}" for r in rows]
    lines += [
        '',
        f"Final step 2,000 validation AUROC: {mixed['val_history'][-1]['val/auroc_overall']:.6f}.",
        f"Training wall time: {mixed['wall_time_s']:.1f} seconds on MPS.",
        'The validation gain did not carry over to the test fold. This is one seed; no variance or significance claim.',
        '',
        'Interpretation:',
        '- Acceptance checked technical quality and rhythm, not the five diagnostic labels.',
        '- 2,212 generated ECGs use conditioning parents outside the 50% real pool (1,794 distinct records / 1,765 patients); all are from training folds 1–8.',
        '- No conditioning record or patient overlaps validation/test. Generator pretraining membership cannot be independently audited.',
        '- This is exploratory: the test fold was viewed in earlier experiments.',
        '- Original scaling runs trained 12,000 steps. The budget-matched plot limits selection to the first 2,000; real50 already peaks at step 1,200.',
        '- Real50 test scores reuse an existing checkpoint with exactly identical model tensors; baseline_checkpoint_equivalence.json records the verification.',
        '',
        'Plot and data files:',
        'figures/scaling_saturation.png/.svg — original scaling context plus new mixed trial at 50% real patients.',
        'figures/scaling_saturation_budget_matched.png/.svg — checkpoint selection restricted to <=2,000 steps.',
        'figures/learning_curves.png/.svg — validation histories through 2,000 steps.',
        'figures/saturation_points.csv, validation_history.csv, comparisons.csv, per_class_metrics.csv, summary.json.',
        'test_and_validation_comparison.csv — baseline and mixed results by split and slice.',
        'pool50/mix75/seed0/ — checkpoints, metrics, validation/test predictions, and complete sampling audit.',
        '',
        'Recreate report/figures from the project root:',
        '.venv/bin/python runs/ptbxl-diffusets-4515-mix75-2000/plot_results.py',
        '.venv/bin/python runs/ptbxl-diffusets-4515-mix75-2000/build_report.py',
        'Preparation and training scripts retain frozen input/code fingerprints and do not overwrite completed trials.',
    ]
    (ROOT / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:24]))


if __name__ == '__main__':
    main()
