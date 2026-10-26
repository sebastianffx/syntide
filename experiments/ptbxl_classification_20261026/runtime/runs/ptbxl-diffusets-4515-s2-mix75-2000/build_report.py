#!/usr/bin/env python3
"""Summarize completed three-seed S2 + DiffuSETS trials and saved historical S2."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PARTS = ('overall', 'artifact', 'clean')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def verify_predictions(directory, split, scores):
    table = pq.read_table(directory / f'predictions_{split}.parquet').to_pandas()
    table = table[table['checkpoint'] == 'best']
    assert set(table['step']) == {scores['step']}
    for subset in PARTS:
        selected = table if subset == 'overall' else table[table.is_artifact == (subset == 'artifact')]
        aucs = [roc_auc_score(part.y_true, part.y_prob) for _, part in selected.groupby('class_id')]
        assert len(aucs) == 5
        assert np.isclose(np.mean(aucs), scores[f'{split}/auroc_{subset}'], rtol=0, atol=1e-12)


def main():
    assert read(ROOT / 'training_audit.json')['passed']
    assert read(ROOT / 'evaluation_audit.json')['passed']
    frozen = read(ROOT / 'experiment_freeze.json')
    for group in ('inputs_sha256', 'code_sha256', 'original_files_sha256'):
        for relative, expected in frozen[group].items():
            assert sha(PROJECT / relative) == expected, relative
    claim = read(ROOT / 'test_claim.json')
    assert datetime.fromisoformat(read(ROOT / 'experiment_freeze.json')['frozen_at']) <= datetime.fromisoformat(claim['claimed_at'])
    per_seed, files, training_times = [], {}, []
    for arm, template in (
        ('historical_S2', PROJECT / 'runs/ptbxl-aug-sweep/scarcity50/S2'),
        ('S2_plus_DiffuSETS', ROOT / 'pool50/mix75_s2'),
    ):
        for seed in range(3):
            directory = template / f'seed{seed}'
            result, test = read(directory / 'result.json'), read(directory / 'test_scores.json')
            assert result['status'] == 'complete' and result['replicate_seed'] == seed
            best = max(result['val_history'], key=lambda h: h['val/auroc_overall'])
            matched = max([h for h in result['val_history'] if h['step'] <= 2000], key=lambda h: h['val/auroc_overall'])
            assert best['step'] == matched['step'] == result['best_step'] == test['step']
            assert result['trial_id'] == test['trial_id']
            if arm == 'S2_plus_DiffuSETS':
                assert result['final_step'] == 2000
                assert datetime.fromisoformat(result['finished_at']) <= datetime.fromisoformat(claim['claimed_at'])
                assert datetime.fromisoformat(claim['claimed_at']) <= datetime.fromisoformat(test['scored_at'])
                verify_predictions(directory, 'val', best)
                verify_predictions(directory, 'test', test)
                training_times.append(result['wall_time_s'])
            for split, scores in (('val', best), ('test', test)):
                for subset in PARTS:
                    per_seed.append({'arm': arm, 'seed': seed, 'split': split, 'slice': subset,
                        'macro_auroc': scores[f'{split}/auroc_{subset}'], 'checkpoint_step': result['best_step'],
                        'n_real': 6389 if arm == 'historical_S2' else 8579,
                        'n_generated': 0 if arm == 'historical_S2' else 4515,
                        'training_steps_completed': result['final_step'], 'selection_budget_steps': 2000})
            for filename in ('result.json', 'test_scores.json', 'predictions_val.parquet', 'predictions_test.parquet'):
                files[str((directory / filename).relative_to(PROJECT))] = sha(directory / filename)
    aggregated = []
    for split in ('val', 'test'):
        for subset in PARTS:
            old = np.array([r['macro_auroc'] for r in per_seed if r['arm'] == 'historical_S2' and r['split'] == split and r['slice'] == subset])
            new = np.array([r['macro_auroc'] for r in per_seed if r['arm'] == 'S2_plus_DiffuSETS' and r['split'] == split and r['slice'] == subset])
            assert len(old) == len(new) == 3
            aggregated.append({'split': split, 'slice': subset,
                'historical_S2_mean': float(old.mean()), 'historical_S2_sd': float(old.std(ddof=1)),
                'S2_plus_DiffuSETS_mean': float(new.mean()), 'S2_plus_DiffuSETS_sd': float(new.std(ddof=1)),
                'difference_mean': float((new-old).mean()), 'paired_difference_sd': float((new-old).std(ddof=1)),
                'n_seeds': 3})
    for filename, rows in [('seed_metrics.csv', per_seed), ('aggregate_metrics.csv', aggregated)]:
        with (ROOT / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(ROOT / 'results_summary.json', {'status': 'complete', 'seeds': [0,1,2],
        'steps_per_seed': 2000, 'per_seed': per_seed, 'aggregate': aggregated,
        'historical_S2_best_checkpoints_unchanged_by_2000_step_limit': True,
        'new_best_at_budget_boundary_seeds': [r['seed'] for r in per_seed if r['arm'] == 'S2_plus_DiffuSETS' and r['split'] == 'val' and r['slice'] == 'overall' and r['checkpoint_step'] == 2000],
        'training_wall_time_s': training_times, 'total_training_wall_time_s': sum(training_times),
        'validation_and_test_aurocs_recomputed_from_saved_predictions': True,
        'test_claim_after_all_three_completed': True, 'evidence_sha256': files,
        'interpretation': 'Descriptive exploratory comparison. Historical S2 has6389depleted real records; the new recipe restores2190realrecords and adds4515generated. Differences do not isolate generated-data effects.',
        'error_bars': 'sample standard deviation across3seeds, not confidence intervals'})
    lines = [
        'Three-seed PTB-XL classification: S2 + DiffuSETS', '',
        'Completed seeds 0, 1 and 2, each for 2,000 optimizer steps.',
        'Every seed uses all 8,579 real ECGs + all 4,515 accepted generated ECGs.',
        'Every batch: exactly 48 real + 16 generated; exactly 12 real + 4 generated receive S2.',
        'S2: gain 0.95–1.05, circular shift ±0.25 s, noise RMS 0–0.05 mV, baseline wander 0–1.75 mV at 0.05–0.7 Hz.',
        'Dose = 1/4 batch × 1/2 strength = 1/8. Labels unchanged by augmentation.',
        'Native ResNet1d/AdamW/BCE trainer and native generic augmentation unchanged.',
        'Each seed: 96,000 real + 32,000 generated presentations; 24,000 real + 8,000 generated augmented presentations.',
        'Every real ECG was seen 11–12 times and every generated ECG 7–8 times per seed.',
        'Validation every 100 steps; first maximum overall validation macro-AUROC selects the checkpoint.',
        'Validation fold 9: 2,146 records. Test fold 10: 2,158 records, scored after all three seeds finished.',
        '', 'Macro-AUROC, mean ± sample SD across three seeds:',
        'Split  Slice       Historical S2          S2 + DiffuSETS         Difference',
    ]
    for row in aggregated:
        lines.append(f"{row['split']:<6} {row['slice']:<11} {row['historical_S2_mean']:.6f} ± {row['historical_S2_sd']:.6f}   "
                     f"{row['S2_plus_DiffuSETS_mean']:.6f} ± {row['S2_plus_DiffuSETS_sd']:.6f}   {row['difference_mean']:+.6f}")
    lines += ['', 'New-trial per-seed overall results:']
    for seed in range(3):
        rows = [r for r in per_seed if r['arm'] == 'S2_plus_DiffuSETS' and r['seed'] == seed and r['slice'] == 'overall']
        values = {r['split']: r for r in rows}
        lines.append(f"Seed {seed}: best step {values['val']['checkpoint_step']}; validation {values['val']['macro_auroc']:.6f}; test {values['test']['macro_auroc']:.6f}.")
    lines += ['', 'Interpretation and comparison limits:',
        '- Historical S2 uses 6,389 artifact-depleted real ECGs; this experiment uses 8,579 real + 4,515 generated. Differences include restoring 2,190 real records and adding generated records.',
        '- Historical S2 trained 10,000 steps, but all three selected checkpoints already lie within 2,000. Its saved test scores therefore also apply to the 2,000-step selection comparison.',
        '- The original real-data scaling line is seed 0 and trained 12,000 steps. Separate plots restrict every validation selection and smoothed peak to <=2,000 steps.',
        '- Error bars are sample SD, not confidence intervals; no significance claim is made.',
        '- Seed 2 selects the final step (2,000); these runs do not establish eventual convergence or saturation.',
        '- Generated labels are provisional conditioning-parent CD/HYP/MI/NORM/STTC targets; acceptance did not validate those diagnoses.',
        '- 2,212 generated ECGs use conditioning parents outside the real 50% pool (1,794 records / 1,765 patients). All are in training folds 1–8; no held-out record or patient overlap.',
        '- Previous test results were known; this is an exploratory experiment.',
        '', 'Artifacts:',
        'figures/ contains overall-only and full scaling comparisons, zoomed comparisons, PNG/SVG figures and plotted CSV/JSON data.',
        'seed_metrics.csv and aggregate_metrics.csv contain full-precision validation/test metrics.',
        'results_summary.json contains metrics and evidence fingerprints.',
        'pool50/mix75_s2/seed{0,1,2}/ contains checkpoints, predictions, learning histories, all augmentation recipes and batch/source audits.',
        '', 'Rebuild outputs from the synthetic_data_engine project root:',
        '.venv/bin/python runs/ptbxl-diffusets-4515-s2-mix75-2000/plot_results.py',
        '.venv/bin/python runs/ptbxl-diffusets-4515-s2-mix75-2000/build_report.py',
    ]
    (ROOT / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[13:26]))


if __name__ == '__main__':
    main()
