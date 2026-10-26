#!/usr/bin/env python3
"""Prepare the explicitly authorized exploratory classification extension.

Reads the immutable 1,530-signal snapshot. Does not alter the generation catalog,
original experiments, or remote job. New training labels are explicitly proxies.
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import os

import numpy as np

from syntide.data.adapter import PtbXlAdapter
from syntide.data.challenge import load_manifest
from syntide.data.leakage import check_lineage, check_manifest
from syntide.data.schema import LEADS, SUPERCLASSES

RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[1]
SNAPSHOT = PROJECT / 'outputs/ptbxl-diffusets-classification-1530/snapshot'
MANIFEST = PROJECT / 'manifests/ptbxl-artifact-50/pool50.json'
AUTHORIZATION = 'Run with provisional conditioning labels and the extra control'
LABEL_POLICY = 'provisional_conditioning_parent_superclasses'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def freeze_json(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if read(path) != payload:
            raise ValueError(f'Frozen file differs: {path}')
    else:
        with path.open('x') as f:
            f.write(text)


def save_npz(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError(f'Existing data differs: {path}')
        return
    partial = path.with_suffix('.partial')
    with partial.open('wb') as f:
        np.savez(f, **arrays)
    os.replace(partial, path)


def main():
    os.chdir(PROJECT)
    from collect_snapshot import verify_snapshot
    verify_snapshot()
    snapshot = read(SNAPSHOT / 'snapshot_manifest.json')
    inventory = snapshot['remote_inventory']
    selected = inventory['accepted_slot_ids']
    assert len(selected) == 1530 and selected == sorted(set(selected))
    plan = read(SNAPSHOT / 'candidate-plan.json')
    manifest = load_manifest(MANIFEST)
    adapter = PtbXlAdapter('datasets/timenet-registry', 'physionet/ptb-xl@1.0.3')
    index, source = adapter.index(), adapter.source()
    check_manifest(manifest, index, source)
    index_rows = {r['record_id']: r for r in index.to_pylist()}
    base_ids = list(manifest.roles['pool'])
    assert len(base_ids) == 8579
    parent_ids = [plan['candidates'][s]['source_record_id'] for s in selected]
    unique_ids = list(dict.fromkeys(base_ids + parent_ids))
    print(f'Loading {len(unique_ids)} distinct real ECGs for the two matched training arms.', flush=True)
    real = adapter.load(unique_ids)
    real_position = {r: i for i, r in enumerate(real.record_id.tolist())}
    base_pos = np.array([real_position[r] for r in base_ids])
    parent_pos = np.array([real_position[r] for r in parent_ids])
    signals, synthetic_ids, ancestry, rows = [], [], [], []
    for slot, parent_id in zip(selected, parent_ids, strict=True):
        folder = SNAPSHOT / f'candidates/{slot:05d}'
        result, candidate = read(folder / 'result.json'), read(folder / 'candidate.json')
        catalog, ledger = read(folder / 'catalog.jsonl'), read(folder / 'ledger.jsonl')
        acceptance = read(folder / 'acceptance.json')
        assert result['accepted'] is True and result['status'] == acceptance['status'] == 'verified'
        assert candidate == plan['candidates'][slot]
        assert ledger['source_record_ids'] == [parent_id]
        assert catalog['labels'] == [] and catalog['metadata']['label_status'] == 'pending_review'
        assert [x.upper() for x in catalog['channels']] == list(LEADS)
        assert catalog['sample_rate'] == 100 and catalog['units'] == 'mV'
        assert catalog['signal_path'] == ledger['signal_path'] == 'signals/00000000.npy'
        signal_path = folder / ledger['signal_path']
        assert sha(signal_path) == ledger['signal_sha256'] == result['artifacts'][ledger['signal_path']]
        signal = np.load(signal_path, allow_pickle=False)
        assert signal.shape == (12, 1000) and signal.dtype == np.float32 and np.isfinite(signal).all()
        parent = index_rows[parent_id]
        assert parent['subject_uid'] == candidate['subject_uid'] and parent['strat_fold'] in range(1, 9)
        synthetic_id = catalog['record_id']
        assert synthetic_id == result['record_id'] == ledger['record_id']
        synthetic_ids.append(synthetic_id)
        signals.append(signal)
        ancestry.append((synthetic_id, [parent_id], [parent['subject_uid']]))
        rows.append({
            'slot_id': slot, 'record_id': synthetic_id, 'conditioning_record_id': parent_id,
            'conditioning_subject_uid': parent['subject_uid'], 'conditioning_fold': parent['strat_fold'],
            'conditioning_in_original_c2_pool': parent_id in manifest.roles['pool'],
            'conditioning_labels': parent['labels'], 'label_policy': LABEL_POLICY,
            'diagnostic_label_validation': 'not_performed', 'generated_artifact_status': 'unknown',
            'result_sha256': sha(folder / 'result.json'), 'signal_sha256': ledger['signal_sha256'],
            'acceptance_sha256': sha(folder / 'acceptance.json'), 'seed': candidate['seed'],
            'rate_band': candidate['rate_band'], 'regularity': candidate['regularity'],
        })
    assert len(set(synthetic_ids)) == 1530
    check_lineage(ancestry, manifest, index, allowed_roles=['pool', 'unused'])
    base_subjects = real.subject_uid[base_pos]
    parent_subjects = real.subject_uid[parent_pos]
    labels = np.concatenate([real.labels[base_pos], real.labels[parent_pos]])
    roots = np.asarray(base_ids + parent_ids, dtype=str)
    subjects = np.concatenate([base_subjects, parent_subjects])
    common = dict(labels=labels, subject_uid=subjects, root_record_id=roots, root_subject_uid=subjects)
    synthetic_x = np.stack(signals)
    data_dir = RUN / 'data'
    data_dir.mkdir(exist_ok=True)
    arms = {}
    for arm in ('G1', 'R1'):
        synthetic = arm == 'G1'
        extra_ids = synthetic_ids if synthetic else [f'real-parent-control:{slot:05d}:{parent}' for slot, parent in zip(selected, parent_ids, strict=True)]
        arrays = dict(
            **common,
            record_id=np.asarray(base_ids + extra_ids, dtype=str),
            x=np.concatenate([real.x[base_pos], synthetic_x if synthetic else real.x[parent_pos]]),
            artifacts=np.concatenate([real.artifacts[base_pos], np.zeros((1530, 4), dtype=bool) if synthetic else real.artifacts[parent_pos]]),
            is_synthetic=np.concatenate([np.zeros(8579, dtype=bool), np.full(1530, synthetic, dtype=bool)]),
        )
        path = data_dir / f'{arm}.npz'
        save_npz(path, arrays)
        arms[arm] = {'path': str(path.relative_to(RUN)), 'sha256': sha(path), 'n_rows': 10109,
                     'n_original_pool_rows': 8579, 'n_additional_rows': 1530}
        print(f'Prepared {arm}: 10,109 rows, SHA256 {arms[arm]["sha256"][:12]}', flush=True)
        del arrays
    freeze_json(data_dir / 'conditioning_labels.json', rows)
    coverage = Counter('inside_original_pool' if r['conditioning_in_original_c2_pool'] else 'outside_original_pool' for r in rows)
    provenance = {
        'schema_version': 'ptbxl-diffusets-classification-provenance@1',
        'label_policy': LABEL_POLICY, 'label_policy_authorization': AUTHORIZATION,
        'class_order': list(SUPERCLASSES), 'lead_order': list(LEADS), 'sample_rate_hz': 100, 'units': 'mV',
        'snapshot_manifest': str((SNAPSHOT / 'snapshot_manifest.json').relative_to(PROJECT)),
        'snapshot_manifest_sha256': sha(SNAPSHOT / 'snapshot_manifest.json'),
        'generation_id': inventory['generation_id'], 'snapshot_completed_candidates': 1721,
        'snapshot_accepted_candidates': 1530, 'snapshot_rejected_candidates': 191,
        'accepted_slot_ids': selected, 'conditioning_coverage': dict(coverage),
        'original_pool_ecgs': 8579, 'original_pool_patients': len(set(base_subjects.tolist())),
        'distinct_conditioning_ecgs': len(set(parent_ids)), 'distinct_conditioning_patients': len(set(parent_subjects.tolist())),
        'combined_distinct_real_root_ecgs': len(unique_ids), 'combined_distinct_real_root_patients': len(set(subjects.tolist())),
        'additional_distinct_real_root_ecgs': len(set(parent_ids) - set(base_ids)),
        'additional_distinct_real_root_patients': len(set(parent_subjects.tolist()) - set(base_subjects.tolist())),
        'conditioning_class_counts': {c: int(real.labels[parent_pos, i].sum()) for i, c in enumerate(SUPERCLASSES)},
        'accepted_rate_bands': dict(Counter(r['rate_band'] for r in rows)),
        'accepted_regularity': dict(Counter(r['regularity'] for r in rows)),
        'validation_ecgs': len(manifest.roles['val']), 'test_ecgs': len(manifest.roles['test']),
        'conditioning_lineage_gate': 'PASS: only folds 1–8; no validation/test records or patients',
        'source_sha256': source['sha256'], 'classification_manifest_sha256': manifest.sha256,
        'synthetic_artifact_flags': 'Unknown. False placeholders are unused by loss/sampling; never interpreted as clean truth.',
        'real_control': 'Same 1,530 conditioning parents in accepted-slot order, including duplicates, appended to the original 8,579. Matches labels and row sampling with G1.',
        'generator_pretraining': 'Reported MIMIC-IV-ECG; no PTB-XL overlap declared. Checkpoint membership unavailable for independent holdout-overlap verification.',
        'caveats': [
            'Synthetic diagnosis targets are conditioning-parent proxy labels, not verified generated diagnoses.',
            'Some conditioning patients lie outside the original 50% classification pool; G1 versus C2 includes extra source information.',
            'The accepted snapshot is an early generation prefix, not a random sample of the final 5,000-candidate allocation.',
            'The test fold has been viewed in earlier experiments; this is an exploratory extension.',
        ],
    }
    freeze_json(data_dir / 'provenance.json', provenance)
    data_manifest = {
        'schema_version': 'ptbxl-offline-training@1', 'arms': arms,
        'conditioning_labels': {'path': 'data/conditioning_labels.json', 'sha256': sha(data_dir / 'conditioning_labels.json')},
        'provenance': {'path': 'data/provenance.json', 'sha256': sha(data_dir / 'provenance.json')},
        'label_policy': LABEL_POLICY,
    }
    freeze_json(data_dir / 'manifest.json', data_manifest)
    protocol = {
        'schema_version': 'ptbxl-diffusets-classification@1', 'id': RUN.name, 'regime': 'scarcity50',
        'label_policy': LABEL_POLICY, 'label_policy_authorized': True, 'label_policy_authorization': AUTHORIZATION,
        'original_run': 'runs/ptbxl-artifact-50-val100', 'manifest': 'manifests/ptbxl-artifact-50/pool50.json',
        'registry': 'datasets/timenet-registry', 'dataset': 'physionet/ptb-xl@1.0.3',
        'data_manifest': {'path': 'data/manifest.json', 'sha256': sha(data_dir / 'manifest.json')},
        'lineage_allowed_roles': ['pool', 'unused'], 'seeds': [0, 1, 2], 'batch_size': 64,
        'max_steps': 6500, 'val_every': 100,
        'analysis': {'primary': ['G1', 'C2'], 'key_control': ['G1', 'R1'], 'diagnostic': ['R1', 'C2'],
                     'contrasts': [['G1', 'C2'], ['G1', 'R1'], ['R1', 'C2']],
                     'slices': ['overall', 'clean', 'artifact'], 'n_boot': 2000, 'ci_level': 0.95, 'bootstrap_seed': 0},
    }
    freeze_json(RUN / 'protocol.json', protocol)
    print(json.dumps({'status': 'prepared', 'counts': provenance['conditioning_coverage'],
                      'additional_distinct_real_root_ecgs': provenance['additional_distinct_real_root_ecgs'],
                      'additional_distinct_real_root_patients': provenance['additional_distinct_real_root_patients'],
                      'protocol': str(RUN / 'protocol.json')}, indent=2))


if __name__ == '__main__':
    main()
