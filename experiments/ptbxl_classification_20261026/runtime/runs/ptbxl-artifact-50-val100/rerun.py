"""Reproduce the requested validation-cadence experiment using the existing CLI contracts."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / 'runs/ptbxl-artifact-50-val100'
ORIGINAL = PROJECT / 'runs/ptbxl-artifact-50'
OVERRIDES = ['protocol=ptbxl_artifact_50', 'protocol.id=ptbxl-artifact-50-val100', 'protocol.val_every=100']


def status(phase, state='running'):
    payload = {'phase': phase, 'status': state, 'updated_at': datetime.now(timezone.utc).isoformat()}
    (RUN / 'rerun_status.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload), flush=True)


def run(phase, arguments):
    status(phase)
    command = [sys.executable, '-u', '-m', *arguments]
    with (RUN / f'{phase}.log').open('a') as log:
        log.write('\nCOMMAND: ' + repr(command) + '\n')
        log.flush()
        process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f'{phase} exited with code {code}')


def verify_training():
    status('verify-training')
    rows = []
    for entry in json.loads((RUN / 'plan.json').read_text())['trials']:
        rel = entry['run_dir']
        old = json.loads((ORIGINAL / rel / 'result.json').read_text())
        new = json.loads((RUN / rel / 'result.json').read_text())
        a = {h['step']: h for h in old['val_history']}
        b = {h['step']: h for h in new['val_history']}
        shared = sorted(a.keys() & b.keys())
        same_history = all(a[k] == b[k] for k in shared)
        same_trainer = old['trainer']['fingerprint'] == new['trainer']['fingerprint']
        same_init = old['trainer']['init_sha256'] == new['trainer']['init_sha256']
        row = {'run_dir': rel, 'shared_checkpoints': len(shared), 'shared_metrics_exactly_equal': same_history,
               'trainer_matches': same_trainer, 'initial_weights_match': same_init,
               'validation_steps': len(new['val_history']), 'old_best_step': old['best_step'], 'new_best_step': new['best_step']}
        rows.append(row)
        if not (same_history and same_trainer and same_init and len(shared) == 13 and len(b) == 65):
            (RUN / 'training_equivalence.json').write_text(json.dumps(rows, indent=2) + '\n')
            raise RuntimeError(f'Training equivalence differs: {row}')
    scaling = json.loads((PROJECT / 'runs/ptbxl-scaling/pool100/real/seed0/result.json').read_text())
    new = json.loads((RUN / 'full100/C2/seed0/result.json').read_text())
    if new['val_history'] != scaling['val_history'][:65]:
        raise RuntimeError('Full100 seed0 does not exactly match scaling through step6500')
    (RUN / 'training_equivalence.json').write_text(json.dumps({'trials': rows, 'scaling_seed0_first_65_rows_exactly_equal': True}, indent=2) + '\n')
    print('Verified: all 21 training trajectories agree at shared checkpoints; full100 seed0 reproduces scaling exactly.', flush=True)


def main():
    run('train', ['syntide_trainers.torch_loop', 'runs/ptbxl-artifact-50-val100/*/*/seed*'])
    verify_training()
    run('evaluate-val', ['syntide.eval', *OVERRIDES, 'split=val'])
    run('evaluate-val-final', ['syntide.eval', *OVERRIDES, 'checkpoint=final'])
    run('evaluate-test', ['syntide.eval', *OVERRIDES, 'split=test'])
    run('arms-figures', ['syntide.eval.arms_figure', str(RUN)])
    run('audit', ['syntide.eval', *OVERRIDES, 'mode=audit'])
    status('complete', 'complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        status('failed', 'failed')
        raise
