"""Reproduce the selection-tuned DiffuSETS allocation; reads TRAIN + selection only."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = Path('/Users/sebastian/Documents/Codex/2026-10-22/please-run-and-report-back-after-2/outputs/runpod-baseline')
MANIFEST = ROOT / 'manifests/ecgqa-artifact-50/pool50.json'
RAW = ROOT / 'datasets/raw/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv'
DB = ROOT / 'outputs/qa/grounded/timenet/syntide/ecgqa-cot-grounded/1.0.0/control.duckdb'
ARMS = ('real_qa', 'paraphrase', 'grounded', 'grounded_acq')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(name, value):
    path = OUT / name
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n'
    if path.exists() and path.read_text() != payload:
        raise RuntimeError(f'Refusing to replace an existing different frozen plan file: {path}')
    path.write_text(payload)


def cluster(report):
    """Lexical conditioning clusters only; neither diagnoses nor generated labels."""
    text = report.lower()
    for label, pattern in (
        ('pacing_mention', r'pacemaker|schrittmacher|paced'),
        ('atrial_irregularity_mention', r'fibrillat|flimmer|flutter|flattern'),
        ('slow_rate_mention', r'brady'),
        ('fast_rate_mention', r'tachy'),
        ('sinus_mention', r'sinus'),
    ):
        if re.search(pattern, text):
            return label
    return 'other_report'


def selection_analysis():
    sources = {}
    metrics = {}
    checks = {}
    for arm in ARMS:
        directory = BASE / 'run/seed0' if arm == 'real_qa' else ROOT / f'outputs/qa-runs/interim/{arm}/seed0'
        path = directory / 'best_selection_metrics.json'
        history = read(directory / 'selection_history.json')
        result = read(directory / 'train_result.json')
        best = max(history, key=lambda event: event['accuracy'])
        metrics[arm] = read(path)['groups']
        assert read(path) == best['scores']
        assert best['step'] == result['best_step']
        assert len(history) == 16 and history[-1]['step'] == 15531
        checks[arm] = {'first_selection_argmax_step': best['step'], 'selection_events': 16, 'max_steps': 15531}
        for filename in ('best_selection_metrics.json', 'selection_history.json', 'train_result.json'):
            source = directory / filename
            sources[str(source)] = sha(source)
    training = pq.read_table(BASE / 'prepared/train.parquet', columns=['template_id', 'question']).to_pylist()
    examples = {}
    counts = collections.Counter()
    for row in training:
        key = str(row['template_id'])
        counts[key] += 1
        examples.setdefault(key, row['question'])
    sources[str(BASE / 'prepared/train.parquet')] = sha(BASE / 'prepared/train.parquet')
    overview = {arm: {s: {k: v for k, v in group.items() if k != 'per_template'} for s, group in groups.items()} for arm, groups in metrics.items()}
    templates = {}
    for tid in sorted(examples, key=int):
        templates[tid] = {'training_rows': counts[tid], 'question_example_from_train': examples[tid], 'selection': {}}
        for arm in ARMS:
            templates[tid]['selection'][arm] = {}
            for subset in ('all', 'clean', 'noisy'):
                item = metrics[arm][subset]['per_template'][tid]
                templates[tid]['selection'][arm][subset] = {
                    'n': item['n'], 'accuracy': item['accuracy'], 'macro_f1': item['macro_f1'],
                    'delta_accuracy_vs_baseline_pp': 100 * (item['accuracy'] - metrics['real_qa'][subset]['per_template'][tid]['accuracy']),
                    'predicted_label_counts': {label: value['tp'] + value['fp'] for label, value in item['per_class'].items() if value['tp'] + value['fp']},
                }
    families = {'noise_26_34': range(26, 35), 'numeric_intervals_39_42': range(39, 43), 'rhythm_7_10_16': (7, 10, 16)}
    aggregate = {}
    for family, ids in families.items():
        aggregate[family] = {}
        for arm in ARMS:
            aggregate[family][arm] = {}
            for subset in ('all', 'clean', 'noisy'):
                group = [metrics[arm][subset]['per_template'][str(tid)] for tid in ids]
                aggregate[family][arm][subset] = {'n': sum(g['n'] for g in group), 'accuracy': sum(g['accuracy'] * g['n'] for g in group) / sum(g['n'] for g in group), 'mean_template_macro_f1': sum(g['macro_f1'] for g in group) / len(group)}
    return {
        'scope': 'Fixed 2,000-question selection subset and original TRAIN question wording only. No full validation or test metric files, labels, predictions or rows are read by this script.',
        'study_status': 'Selection-tuned exploratory extension, separate from the original four-arm confirmatory protocol. The same selection subset has already chosen checkpoints and now informs allocation.',
        'sources_sha256': sources, 'checkpoint_checks': checks, 'overview': overview, 'template_families': aggregate, 'templates': templates,
        'findings': [
            'grounded_acq has the strongest selection accuracy (46.95% all, 40.10695% noisy), but its macro-F1 is below baseline in all, clean and noisy subsets.',
            'Large gains are not proof of improved diagnostic signal reasoning: acquisition templates 5, 7, 10 and 18 predict none for every selection question; template 4 predicts none 50/54 times.',
            'Acquisition template 26 (lead baseline drift) falls 23.0769 percentage points; template 28 (any noise in a lead) falls 50 points and always predicts yes. Template 34 noisy lead localization remains 0% and predicts none for all items.',
            'Template 39 RR-range discrimination predicts no for all acquisition-arm selection items, while template 42 numeric-feature listing predicts none for all items. Measured HR/rate/regularity coverage is actionable; unvalidated interval or diagnosis labels are not.',
            'Per-template cells are small (typically about 52 total and 26 per clean/noisy slice); no significance or reliable expected gain is asserted.',
        ],
    }


def main():
    manifest = read(MANIFEST)
    pool = set(manifest['roles']['pool'])
    assert len(pool) == 5778
    raw = {}
    with RAW.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            rid = f"ptbxl-{row['ecg_id']}"
            if rid in pool:
                raw[rid] = {k: row[k] for k in ('age', 'sex', 'report')}
    connection = duckdb.connect(str(DB), read_only=True)
    rows = connection.execute('SELECT record_id, metadata FROM records WHERE record_id IN (SELECT unnest(?))', [sorted(pool)]).fetchall()
    connection.close()
    assert {rid for rid, _ in rows} == pool
    eligible = []
    exclusions = collections.Counter()
    excluded_records = {}
    for rid, text in rows:
        source = raw[rid]
        metadata = json.loads(text)
        evidence = metadata['evidence']
        by_name = {entry['attribute']: entry for entry in evidence}
        issues = []
        try:
            age = float(source['age'])
            sex = float(source['sex'])
        except (TypeError, ValueError):
            age, sex = 0, -1
            issues.append('non_numeric_demographics')
        if not math.isfinite(age) or not 0 < age < 110:
            issues.append('age_outside_supported_range')
        if sex not in (0, 1):
            issues.append('unsupported_sex')
        if not source['report'].strip():
            issues.append('empty_report')
        if source['report'].strip().lower() == 'trace only requested.':
            issues.append('noninformative_trace_only_report')
        for attr in ('heart_rate_bpm', 'beat_count', 'rr_regularity'):
            entry = by_name.get(attr, {})
            if entry.get('status') != 'present' or (entry.get('confidence') or 0) < 0.8:
                issues.append(f'unusable_{attr}')
        if issues:
            exclusions.update(issues)
            excluded_records[rid] = issues
            continue
        heart_rate = by_name['heart_rate_bpm']['value']
        assert math.isfinite(heart_rate) and heart_rate > 0
        max_wander = max(e['value'] for e in evidence if e['attribute'] == 'wander_mv' and e['value'] is not None)
        max_hf = max(e['value'] for e in evidence if e['attribute'] == 'hf_rms_mv' and e['value'] is not None)
        eligible.append({
            'source_record_id': rid, 'subject_uid': metadata['input']['root_subject_uids'][0],
            'report': source['report'], 'age': age, 'sex': 'M' if sex == 0 else 'F',
            'heart_rate': heart_rate, 'regularity': by_name['rr_regularity']['value'],
            'parent_max_wander_mv': max_wander, 'parent_max_hf_rms_mv': max_hf,
            'rate_band': 'slow' if heart_rate < 60 else 'fast' if heart_rate > 100 else 'middle',
            'report_cluster': cluster(source['report']),
            'report_sha256': hashlib.sha256(source['report'].encode()).hexdigest(),
            'parent_waveform_sha256': metadata['input']['waveform_sha256'],
            'required_evidence_ids': {attr: by_name[attr]['evidence_id'] for attr in ('heart_rate_bpm', 'beat_count', 'rr_regularity')},
        })
    thresholds = {'parent_max_wander_mv_q75': float(np.quantile([r['parent_max_wander_mv'] for r in eligible], 0.75)), 'parent_max_hf_rms_mv_q75': float(np.quantile([r['parent_max_hf_rms_mv'] for r in eligible], 0.75))}
    pools = collections.defaultdict(list)
    for row in eligible:
        row['parent_noise_stratum'] = 'higher' if row['parent_max_wander_mv'] >= thresholds['parent_max_wander_mv_q75'] or row['parent_max_hf_rms_mv'] >= thresholds['parent_max_hf_rms_mv_q75'] else 'lower'
        row['stratum'] = '|'.join((row['rate_band'], row['regularity'], row['parent_noise_stratum']))
        row['stable_order'] = hashlib.sha256(f"diffusets-allocation|0|{row['source_record_id']}".encode()).hexdigest()
        pools[row['stratum']].append(row)
    quotas = {}
    for rate, count in [('slow', 1000), ('middle', 3000), ('fast', 1000)]:
        for regularity, fraction in [('regular', 0.8), ('irregular', 0.2)]:
            for noise in ('lower', 'higher'):
                quotas[f'{rate}|{regularity}|{noise}'] = int(count * fraction / 2)
    assert sum(quotas.values()) == 5000 and set(quotas) == set(pools)
    record_uses = collections.Counter()
    subject_uses = collections.Counter()
    report_uses = collections.Counter()
    counts = collections.Counter()
    candidates = []
    used_seeds = set()
    while len(candidates) < 5000:
        for stratum in sorted(quotas):
            if counts[stratum] == quotas[stratum]:
                continue
            row = min(pools[stratum], key=lambda r: (record_uses[r['source_record_id']], subject_uses[r['subject_uid']], report_uses[r['report_sha256']], r['stable_order']))
            slot = len(candidates)
            seed = int.from_bytes(hashlib.sha256(f"diffusets-candidate|0|{slot}|{row['source_record_id']}".encode()).digest()[:4], 'big')
            while seed in used_seeds:
                seed = (seed + 1) % (2**32)
            used_seeds.add(seed)
            candidates.append({k: v for k, v in row.items() if k != 'stable_order'} | {'slot_id': slot, 'seed': seed, 'parent_replicate': record_uses[row['source_record_id']]})
            record_uses[row['source_record_id']] += 1
            subject_uses[row['subject_uid']] += 1
            report_uses[row['report_sha256']] += 1
            counts[stratum] += 1
    assert dict(counts) == quotas
    assert len(candidates) == len(used_seeds) == 5000
    assert all(row['source_record_id'] in pool and row['report'] == raw[row['source_record_id']]['report'] for row in candidates)
    selection = selection_analysis()
    dump('selection-analysis.json', selection)
    source_sha = {str(path): sha(path) for path in (MANIFEST, RAW, DB, Path(__file__))}
    selected_ids = sorted(record_uses, key=lambda x: int(x.split('-')[1]))
    profile = {
        'scope': 'TRAIN pool only, records filtered to manifest pool before metadata analysis. No diagnosis annotations used as eligibility, quotas or generated labels.',
        'pool_size': len(pool), 'eligible_parents': len(eligible), 'excluded_parent_count': len(excluded_records),
        'exclusion_reason_counts_overlap': dict(exclusions), 'excluded_records': excluded_records,
        'noise_thresholds': thresholds, 'noise_definition': 'higher if either maximum per-lead wander or high-frequency RMS is at or above the TRAIN-eligible 75th percentile; otherwise lower. These are relative measurement strata, not clinical clean/noisy labels.',
        'eligible_stratum_counts': {k: len(v) for k, v in sorted(pools.items())},
        'eligible_report_clusters': dict(collections.Counter(r['report_cluster'] for r in eligible)),
        'selected_candidate_report_clusters': dict(collections.Counter(r['report_cluster'] for r in candidates)),
        'selected_unique_records': len(record_uses), 'selected_unique_subjects': len(subject_uses), 'selected_unique_report_texts': len(report_uses),
        'max_candidates_per_record': max(record_uses.values()), 'max_candidates_per_subject': max(subject_uses.values()),
        'parent_replica_count_histogram': dict(sorted(collections.Counter(record_uses.values()).items())),
        'sources_sha256': source_sha,
    }
    dump('pool-profile.json', profile)
    plan = {
        'schema_version': 'syntide.diffusets-candidate-plan@1', 'plan_seed': 0, 'training_seed': 0,
        'candidate_count': 5000, 'manifest_sha256': sha(MANIFEST), 'manifest_content_sha256': manifest['sha256'],
        'manifest_path': str(MANIFEST),
        'provenance': {'scope': 'Training pool plus fixed selection metrics only; no full validation or held-out test inputs to allocation.', 'study_status': selection['study_status'], 'sources_sha256': source_sha, 'selection_analysis_sha256': sha(OUT / 'selection-analysis.json')},
        'allocation': {
            'rate_quota': {'slow_measured_bpm_lt60': 1000, 'middle_measured_bpm_60_to100_inclusive': 3000, 'fast_measured_bpm_gt100': 1000},
            'within_each_rate': {'regular': 0.8, 'irregular': 0.2, 'higher_parent_noise': 0.5, 'lower_parent_noise': 0.5},
            'exact_stratum_quota': quotas, 'noise_thresholds': thresholds,
            'target_heart_rate_rule': 'Use each source waveform measured heart_rate_bpm unchanged; do not invent rates that contradict the preserved report.',
            'parent_choice_rule': 'Round-robin lexicographically sorted strata. Within stratum minimize (record uses, subject uses, exact-report uses, SHA256(diffusets-allocation|0|record_id)).',
            'seed_rule': 'First four bytes big-endian SHA256(diffusets-candidate|0|slot_id|record_id), increment modulo 2**32 only on collision.',
            'source_report_rule': 'Exact source report with age and sex unchanged; PTB-XL sex 0 maps M, 1 maps F. Exclude missing/unsupported demographics and exact noninformative trace-only report.',
            'report_cluster_rule': 'Lexical descriptions only, priority pacing, atrial irregularity, slow rate, fast rate, sinus, other. Clusters describe conditioning text and never establish generated answers.',
            'distinct_parent_records': len(record_uses), 'distinct_parent_subjects': len(subject_uses), 'distinct_report_texts': len(report_uses), 'maximum_parent_replicates': max(record_uses.values()),
        },
        'generation_and_acceptance': {
            'exact_attempt_budget': 5000, 'replace_rejected_candidates': False, 'accepted_count': 'unknown until measurement',
            'embedding_policy': 'Use only selected TRAIN reports. Root may use exact-report-matched published legacy text-embedding-ada-002 vectors with explicit original API-input provenance unknown; no fabricated equivalence to newly embedded exact prompt.',
            'backend': 'Pinned DiffuSETS, batch one, 1000 steps; one immutable candidate bundle per slot for resumption, as implemented by the generation driver.',
            'gate': 'Existing generated ancestry/hash/shape/lead checks; explicit 102.4-to-100 Hz resampling; required HR/beat-count/RR regularity evidence usable at confidence >=0.8; measured amplitude 0.2–10 mV. Keep only accepted candidates, record every rejection.',
            'no_noise_control_claim': 'DiffuSETS receives report, age, sex and HR, not source waveform/noise measurements. Parent noise quotas balance provenance and do not guarantee any realized noise quota. Measure generated waveforms; report realized HR/regularity/noise and requested-versus-realized distributions.',
            'no_diagnosis_ground_truth': 'Conditioning diagnoses, patient metadata and requested HR are lineage only. Every generated answer must be remeasured. No diagnostic, infarct, axis or unvalidated interval labels may be copied from parent reports.',
        },
        'qa_recommendations_not_applied': {
            'prioritize': ['rhythm.heart_rate@1', 'rhythm.beat_count@1', 'rhythm.regularity@1', 'rhythm.rate_band@1'],
            'optional_existing_config_extension': {'add_required': ['wander_mv'], 'add_templates': ['artifact.wander_in_lead@1'], 'reason': 'Directly measured per-lead slow variation can support noise reasoning without treating parent noise annotations as generated labels.'},
            'deprioritize_constant_targets': ['acquisition.duration@1', 'acquisition.flat_leads@1'],
            'unchanged_trainer_limitation': 'Two-ECG comparisons remain excluded and numerical answers retain the unchanged empty-options prompt.',
        },
        'candidates': candidates,
    }
    dump('candidate-plan.json', plan)
    dump('selected-parent-ids.json', {'source_record_ids': selected_ids, 'count': len(selected_ids), 'candidate_plan_sha256': sha(OUT / 'candidate-plan.json')})
    dump('planning-receipt.json', {'files_sha256': {name: sha(OUT / name) for name in ('candidate-plan.json', 'selected-parent-ids.json', 'selection-analysis.json', 'pool-profile.json', 'build_plan.py')}})
    print(json.dumps({'candidate_count': 5000, 'plan_sha256': sha(OUT / 'candidate-plan.json'), 'unique_parents': len(record_uses), 'unique_subjects': len(subject_uses), 'unique_reports': len(report_uses), 'maximum_parent_replicates': max(record_uses.values()), 'eligible': len(eligible), 'quota': quotas}, indent=2))


if __name__ == '__main__':
    main()
