import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(__file__).resolve().parent
result_files = sorted(root.glob('*/*/seed*/result.json'))
completed = [json.loads(p.read_text()) for p in result_files]
report = {'completed': len(completed), 'total': 21,
          'phase': json.loads((root / 'rerun_status.json').read_text()), 'active': []}
for directory in sorted(root.glob('*/*/seed*')):
    if (directory / 'result.json').exists() or not list(directory.glob('events.out.tfevents.*')):
        continue
    accumulator = EventAccumulator(str(directory), size_guidance={'scalars': 0})
    accumulator.Reload()
    tags = accumulator.Tags()['scalars']
    steps = [accumulator.Scalars(t)[-1].step for t in tags if accumulator.Scalars(t)]
    history = accumulator.Scalars('val/auroc_overall') if 'val/auroc_overall' in tags else []
    best = max(history, key=lambda h: h.value) if history else None
    report['active'].append({'trial': str(directory.relative_to(root)), 'step': max(steps, default=0),
                             'budget': 6500, 'best_val': best.value if best else None,
                             'best_step': best.step if best else None})
print(json.dumps(report))
