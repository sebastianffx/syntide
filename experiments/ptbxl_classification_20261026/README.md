# Reproduce the PTB-XL classification experiments

This archive preserves the classification experiments used in `syntide.key`,
including source files that previously existed only in ignored run directories.
It contains eight experiment groups and 85 completed training trials: data
scaling, the 50% patient comparison, the augmentation sweep, the two SIMVEA
variants, and the three DiffuSETS classification comparisons.

## Verify and summarize offline

From the repository root, using Python 3.11 or later:

```sh
python3 experiments/ptbxl_classification_20261026/reproduce.py verify
python3 experiments/ptbxl_classification_20261026/reproduce.py table
python3 -m unittest discover \
  -s experiments/ptbxl_classification_20261026 -p test_reproduce.py
```

These commands use only the standard library. Verification checks package
integrity, all 17 recorded historical source hashes, and the saved trial
identities. The table reads saved scores; it does not rerun inference or rescore
prediction tables. Values are mean area under the receiver operating
characteristic curve across diagnostic classes and seeds. The scaling run
directories have no adjacent test scores; their separate later test evaluation
and checkpoint-equivalence evidence are preserved under the first 4,515-signal
run's `heldout_scaling_evaluation/` directory.

## What is preserved

| Path | Contents |
|---|---|
| `runtime/src/`, `runtime/trainers/` | Classification implementation and trainer, with historical numerical sources recovered from Git |
| `runtime/conf/`, `runtime/manifests/` | Protocols, evaluation rules, patient splits and calibration metadata |
| `runtime/runs/` | Byte-exact experiment scripts, frozen protocols, trial definitions, training histories and saved aggregate scores |
| `runtime/uv.lock` | Dependency lock for the original project environment |
| `sources.json` | Original paths, Git revision, captured hashes and historical source checks |
| `external_inputs.json` | Required waveform, prediction and checkpoint locations omitted from this compact archive |
| `manifest.json` | Byte counts and hashes for the complete package |

Archived source files retain their original text because changing even a
docstring invalidates historical source hashes. Maintained source lives in the
repository's top-level `src/` and `trainers/` directories. Recorded arm keys in
data files remain unchanged so saved trials and comparisons keep their identity.
For readable descriptions, consult the corresponding protocol's transform and
source-role fields and the repository's experiment guides.

## Rebuild the classification experiments

Work in a separate copy of `runtime/`, preserving its directory structure. The
original scripts derive the project root from their location. Install the locked
Python 3.13 environment there with `uv sync --locked`. Restore the external data
listed in `external_inputs.json` and verify their recorded hashes before use.
The recorded runs used Apple's Metal backend; matching the files does not ensure
identical arithmetic on another device or library build.

For a new scaling run, prepare the registry and manifests, then plan and train:

```sh
uv run python -m syntide.data challenge.id=ptbxl-scaling \
  'challenge.pool_fractions=[0.1,0.2,0.3,0.5,0.7,1.0]'
uv run python -m syntide.generate protocol=ptbxl_scaling mode=plan \
  protocol.id=ptbxl-scaling-reproduction
uv run python -m syntide_trainers.torch_loop \
  'runs/ptbxl-scaling-reproduction/*/*/seed*'
```

The displayed 50% comparison uses validation every 100 updates, rather than the
250-update default in `ptbxl_artifact_50.yaml`. Its planning command is:

```sh
uv run python -m syntide.generate protocol=ptbxl_artifact_50 mode=plan \
  protocol.id=ptbxl-artifact-50-val100-reproduction protocol.val_every=100
```

The sweep and SIMVEA protocols are `ptbxl_aug_sweep`, `ptbxl_aug_simvea`, and
`ptbxl_aug_simvea_whole`. Their saved plans and trial definitions are authoritative
for the executed settings. Existing result files represent completed historical
trials. To train fresh experiments, use a separate output tree and record its new
identities; do not remove historical test claims to rerun scoring.

The unique DiffuSETS scripts remain beside their protocols under `runtime/runs/`:

- `ptbxl-diffusets-1530/`: `collect_snapshot.py`, `prepare_data.py`,
  `run_experiment.py` and `compare_results.py`. The runner exposes `plan`, `train`,
  `verify` and `evaluate` commands. The collection scripts record historical
  remote locations; restore the frozen snapshot instead of assuming those hosts
  still exist.
- `ptbxl-diffusets-4515-mix75-2000/`: `prepare_data.py`, `mixed_stream.py`,
  `run_experiment.py` and report/figure helpers. Each batch contains 48 real and
  16 generated recordings, for 2,000 updates and one seed.
- `ptbxl-diffusets-4515-s2-mix75-2000/`: its own stream, runner and report helpers.
  It shares the preceding prepared data, trains three seeds, and augments 12 real
  and four generated recordings in each batch at half the standard strength.

The latter runners also bind the original scaling and augmentation reference
results, checkpoint identities and source hashes. A bare copied runner is
insufficient. Restore all dependencies listed in its frozen protocol and
`experiment_freeze.json`. The archived historical scripts are evidence and
reproduction implementations, not portable one-command launchers.

Generated diagnosis labels are inherited from conditioning parents and were not
verified on the generated signals. These comparisons are exploratory: prior test
results were already known, and the later augmented mixture also changes the
real-data pool. Keep these qualifications when reporting the saved results.

## Local code checks

The archive integrity suite has three tests. The two preserved mixed-stream
suites have 12 tests covering exact source proportions, independent sampling,
augmentation and exposure accounting. They use synthetic arrays, without data
downloads, training or held-out inference. Run them separately in the copied
runtime with the project environment:

```sh
PYTHONPATH=src:trainers/src python -m pytest -q -c /dev/null \
  -p no:cacheprovider --import-mode=importlib \
  runs/ptbxl-diffusets-4515-mix75-2000/test_mixed_stream.py
PYTHONPATH=src:trainers/src python -m pytest -q -c /dev/null \
  -p no:cacheprovider --import-mode=importlib \
  runs/ptbxl-diffusets-4515-s2-mix75-2000/test_mixed_stream.py
```
