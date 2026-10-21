# SYNTIDE

A synthetic data engine for temporal AI. The first experiment asks whether targeted, physically grounded
ECG artifact editing beats standard augmentation on naturally artifact-bearing ECGs from unseen
patients. See `docs/SYNTIDE_Architecture.md` and `docs/SYNTIDE_MVE.md`.

The work is built in three phases, each with its own package, tests and file outputs:

| Phase | Package | Output |
|---|---|---|
| 1. PTB-XL in TimeNet | `syntide.data` | TimeNet registry `physionet/ptb-xl@1.0.3`, `manifests/ptbxl-artifact-v2/pool{10,25}.json` |
| 2. Training + generator | `syntide.train`, `syntide.generate` | one run directory per trial |
| 3. Evaluation | `syntide.eval` | paired comparisons, TensorBoard views, results table |

## Setup

```bash
uv sync          # Python 3.13 (Hydra 1.3 does not run on 3.14); installs all dependencies
uv run pytest    # fast, synthetic fixtures only; no PTB-XL download needed
```

## Phase 1: PTB-XL as a TimeNet dataset

```bash
uv run python -m syntide.data
```

This command:

1. Downloads PTB-XL 1.0.3 (1.84 GB) into `datasets/raw/ptb-xl` once, then reuses it.
2. Builds the TimeNet dataset `physionet/ptb-xl@1.0.3` into `datasets/timenet-registry`. The build is skipped when that version already exists.
3. Writes one challenge manifest per scarcity pool to `manifests/ptbxl-artifact-v2/`, after it passes the leakage gate (Gate 1).

Any setting can be overridden from the command line, for example `challenge.seed=1` for another pool draw or
`registry=/data/timenet`. See `conf/data.yaml` and `DataConfig` in `src/syntide/data/__main__.py`.

### How PTB-XL maps onto TimeNet

| PTB-XL | TimeNet |
|---|---|
| `ecg_id` | `Record` `ptbxl-{ecg_id}` (the same ids as TimeNet's ECG-QA CoT) |
| 12 leads, 100 Hz (`records100`) | 12 lazy `Signal`s in mV, spec `ecg` |
| `patient_id`, `strat_fold` | integer record annotations |
| `baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems` | text record annotations, present only when the flag is set |
| `scp_codes` | `Record.metadata["ptbxl.scp_codes"]` |
| diagnostic superclasses (CD, HYP, MI, NORM, STTC) | one multi-label `ClassificationTask` per record, with `metadata.split` = train (folds 1–8), validation (9) or test (10); the ~400 records with no diagnostic superclass have no task |

TimeNet 0.11 has no split field and does not store subject ids, so patient and fold are kept as
annotations. TimeNet also returns a record's leads sorted by name. `PtbXlAdapter.load()` restores the
clinical order (I, II, III, aVR, aVL, aVF, V1–V6).

### Reading it from Python

```python
from syntide.data.adapter import PtbXlAdapter
from syntide.data.challenge import load_manifest

adapter = PtbXlAdapter("datasets/timenet-registry")
index = adapter.index()                     # pyarrow table, one row per labelled record, no signal reads
manifest = load_manifest("manifests/ptbxl-artifact-v2/pool25.json")
train = adapter.load(manifest.roles["depleted"])   # Samples: x (n, 12, 1000) mV, labels (n, 5), ...
```

### Challenge manifests

Each manifest lists record ids per role and carries a sha256 over its content. Hand edits are
rejected on load.

| Role | Meaning | Used by |
|---|---|---|
| `pool` / `unused` | a seeded fraction of training *subjects* (all their ECGs) and the rest; the 10% pool is nested in the 25% pool | C2 (full oracle) |
| `retained_artifact` / `withheld` | the pool's artifact ECGs, split 10% / 90% | generator calibration / oracle-only |
| `depleted` | pool − withheld | A0–A3 parents |
| `natural_mix` | pool minus as many *random* ECGs as were withheld | C1 (matched-N control) |
| `val` / `test` | folds 9 / 10, untouched | validation / the frozen test |

The leakage gate (`syntide.data.leakage`) checks four things:

- train, val and test are patient-disjoint, and every labelled ECG is in exactly one outer partition;
- `retained_artifact` and `withheld` split the pool's artifact ECGs exactly;
- `depleted` excludes every withheld ECG;
- `natural_mix` comes from the pool and has the same size as `depleted`.

The trainer runs it again before every trial.
