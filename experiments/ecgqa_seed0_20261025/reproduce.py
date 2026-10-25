#!/usr/bin/env python3
"""Verify the frozen package, render the seven-arm table, or print CUDA commands.

Standard library only. No model loading, inference, network access or cloud actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
ARMS = ('real_qa', 'paraphrase', 'grounded', 'grounded_acq',
        'diffusets3030_stack', 'diffusets_full_stack', 'exposure_matched')
PHASES = ('validation', 'validation_zero', 'test', 'test_zero')
GROUPS = ('all', 'clean', 'noisy')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def results(root=ROOT):
    data = read(root / 'evidence/results.json')
    require(data['seed'] == 0 and data['seed_sd'] is None, 'Expected single seed 0, SD null')
    require(set(data['arms']) == set(ARMS), 'Expected exactly the seven reported arms')
    for arm, item in data['arms'].items():
        for phase in PHASES:
            groups = item['metrics'][phase]
            n = 27264 if phase.startswith('validation') else 26841
            require(groups['all']['n'] == n, f'{arm}/{phase}: wrong question count')
            require(groups['clean']['n'] + groups['noisy']['n'] == n, 'Slice counts differ')
            for group in GROUPS:
                for metric in ('accuracy', 'template_macro_f1', 'invalid_rate'):
                    value = groups[group][metric]
                    require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
                            f'{arm}/{phase}/{group}: invalid {metric}')
            for metric in ('accuracy', 'invalid_rate'):
                combined = sum(groups[g][metric] * groups[g]['n'] for g in ('clean', 'noisy')) / n
                require(abs(combined - groups['all'][metric]) < 1e-12, 'Weighted slice metric differs')
    return data


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = read(root / 'manifest.json')
    require(manifest['schema'] == 'ecgqa-seven-arm-reproduction-files@1', 'Wrong package manifest')
    require(bool(manifest['files']), 'Empty file manifest')
    for relative, expected in manifest['files'].items():
        name = Path(relative)
        require(not name.is_absolute() and '..' not in name.parts, 'Unsafe package path')
        path = root / name
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'Package symlink refused')
        require(path.is_file() and path.stat().st_size == expected['bytes']
                and sha(path) == expected['sha256'], f'Package hash mismatch: {relative}')
    data = results(root)
    return {'status': 'passed', 'files': len(manifest['files']), 'arms': list(data['arms']),
            'scope': 'Packaged source/config/evidence integrity; external datasets and checkpoints are not included.'}


def table(root=ROOT):
    data = results(root)
    lines = ['# ECG-QA seven-arm comparison, seed 0', '',
             'Reproduced from the committed full-precision saved metrics; no inference is run.', '',
             '| Arm | Test accuracy | Clean accuracy | Noisy accuracy | Template macro-F1 | Invalid | Real minus zero |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in ARMS:
        item = data['arms'][name]
        m, z = item['metrics']['test'], item['metrics']['test_zero']
        values = [f"{100*m[g]['accuracy']:.2f}%" for g in GROUPS]
        values += [f"{100*m['all']['template_macro_f1']:.2f}%", f"{100*m['all']['invalid_rate']:.3f}%",
                   f"{100*(m['all']['accuracy']-z['all']['accuracy']):+.2f} pp"]
        lines.append('| ' + item['label'] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'The exposure-matched arm uses fixed update 12,000: 192,000 original plus 24,000 '
              'generated draws, batch 18. Earlier arms use selection-chosen checkpoints and batch 16.',
              'It also excludes grounded/acquisition data. Differences from the earlier stacks do not '
              'isolate original exposure alone. All arms use one seed; seed SD is null.',
              'Validation and zero-waveform controls do not establish a reliable general improvement. '
              'See evidence/results.json for every validation/test slice and control.', '']
    return '\n'.join(lines)


def command(arm, prepared, run_dir, python='python3', phase='train', zero_signal=False, root=ROOT):
    config = read(root / 'configs/arms' / arm / 'config.json')
    require(config['seed'] == 0, 'Only the recorded seed is supported')
    base = root / 'baseline_runtime'
    exposure = arm == 'exposure_matched'
    program = ([str(root / 'exposure/trainer/opentslm_exposure.py')] if exposure
               else ['-m', 'syntide_trainers.opentslm_loop'])
    argv = [python, *program, 'train' if phase == 'train' else 'evaluate',
            '--prepared', str(Path(prepared).resolve()), '--run-dir', str(Path(run_dir).resolve())]
    if phase == 'train':
        require(not zero_signal, 'Zero signal is an evaluation control only')
        for key in ('seed', 'max_steps', 'batch_size', 'gradient_accumulation', 'eval_batch_size',
                    'learning_rate', 'max_length', 'max_new_tokens', 'base_revision', 'selection_method', 'val_every'):
            argv += ['--' + key.replace('_', '-'), str(config[key])]
        if exposure:
            argv += ['--baseline-config', str(root / 'configs/arms/real_qa/config.json'),
                     '--source-manifest', str(Path(prepared).resolve() / 'source-manifest.json')]
    else:
        generation_batch = 512 if arm in ARMS[:4] else 256
        argv += ['--split', phase, '--batch-size', str(generation_batch)]
        if zero_signal:
            argv += ['--zero-signal']
    env_paths = [str(base / 'src'), str(base / 'trainers/src')]
    return {'environment': {'PYTHONPATH': os.pathsep.join(env_paths), 'TOKENIZERS_PARALLELISM': 'false',
                            'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}, 'argv': argv,
            'note': 'Printed only. Use a fresh run directory and the pinned CUDA/OpenTSLM environment. '
                    'Pass preflight and checkpoint/exposure audits before each held-out evaluation; '
                    'never replay a claimed interrupted evaluation. Historical protocols are provenance, not launch defaults.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('verify')
    render = commands.add_parser('table')
    render.add_argument('--output', type=Path)
    planned = commands.add_parser('command')
    planned.add_argument('--arm', choices=ARMS, required=True)
    planned.add_argument('--prepared', required=True)
    planned.add_argument('--run-dir', required=True)
    planned.add_argument('--python', default='python3')
    planned.add_argument('--phase', choices=('train', 'selection', 'validation', 'test'), default='train')
    planned.add_argument('--zero-signal', action='store_true')
    planned.add_argument('--shell', action='store_true', help='Print shell-quoted environment and command')
    args = parser.parse_args()
    checked = verify()
    if args.action == 'verify':
        print(json.dumps(checked, indent=2))
    elif args.action == 'table':
        rendered = table()
        if args.output:
            with args.output.open('x') as stream:
                stream.write(rendered)
        else:
            print(rendered, end='')
    else:
        result = command(args.arm, args.prepared, args.run_dir, args.python, args.phase, args.zero_signal)
        if args.shell:
            print('# ' + result['note'])
            print(shlex.join(['env', *(k + '=' + v for k, v in result['environment'].items()), *result['argv']]))
        else:
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
