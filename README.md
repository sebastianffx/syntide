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
3. Writes one challenge manifest per scarcity pool to `manifests/<challenge.id>/` (by default `manifests/ptbxl-artifact-v2/`), after it passes the leakage gate (Gate 1).

Any setting can be overridden from the command line, for example `registry=/data/timenet`, or
`challenge.id=ptbxl-artifact-v2-seed1 challenge.seed=1` for another pool draw, which is written to
`manifests/ptbxl-artifact-v2-seed1/`. A manifest that already exists is never overwritten with
different content. To rebuild one, delete it first. See `conf/data.yaml` and `DataConfig` in
`src/syntide/data/__main__.py`.

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

`PtbXlAdapter.source()` fingerprints the data: it hashes the signal files plus the index. It does not
hash TimeNet's `control.duckdb`, whose bytes change on every build, so rebuilding the same data
reproduces the fingerprint.

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

Each manifest lists record ids per role. It also carries the fingerprint of the data it was built from
and a sha256 over its content, so hand edits are rejected on load. Every draw takes **whole patients**,
so no patient is split between roles that are meant to differ.

| Role | Meaning | Used by |
|---|---|---|
| `pool` / `unused` | a seeded fraction of training patients (all their ECGs) and the rest; the 10% pool is nested in the 25% pool | C2 (full oracle) |
| `retained_artifact` / `withheld` | the pool's artifact patients, split 10% / 90%. Retained patients' artifact ECGs calibrate the generator. Withheld patients leave with all their ECGs, clean ones included | generator calibration / oracle-only |
| `depleted` | pool − withheld | A0–A3 parents |
| `natural_mix` | pool minus random whole patients holding as many ECGs as were withheld | C1 (matched-N control) |
| `val` / `test` | folds 9 / 10, untouched | validation / the frozen test |

| | pool10 | pool25 |
|---|---|---|
| pool | 1,741 ECGs / 1,483 patients | 4,279 / 3,706 |
| artifact patients (retained / withheld) | 373 (38 / 335) | 941 (95 / 846) |
| withheld ECGs (artifact + clean) | 485 (372 + 113) | 1,140 (908 + 232) |
| `depleted` = `natural_mix` size | 1,256 | 3,139 |
| artifact rate: `depleted` / `natural_mix` / pool | 3.1% / 23.2% / 23.6% | 3.4% / 23.4% / 23.7% |

The test fold has 2,158 ECGs. Its artifact slice is 513 ECGs from 477 patients. `docs/SYNTIDE_MVE.md`
quotes 37 / 93 retained artifact ECGs, which came from an earlier per-ECG draw. The whole-patient counts
above replace them.

The leakage gate (Gate 1, `syntide.data.leakage.check_manifest`) checks that:

- the manifest was built from the local data (same fingerprint);
- train, val and test are exactly folds 1–8, 9 and 10, and no record is in two outer partitions;
- no patient is in more than one of train, val and test, or in both the pool and `unused`;
- `retained_artifact` and `withheld` cover the pool's artifact ECGs, and no withheld patient keeps an
  ECG in `depleted`;
- `natural_mix` is the pool minus whole patients and has the same size as `depleted`.

For Phase 2 it also provides `check_calibration_ids` (calibration reads only `retained_artifact`) and
`check_recipe_parents` (every recipe starts from `depleted`). The trainer runs the gate again before
every trial.

### How the split compares with OpenTSLM and ECG-QA

We use the split recommended by the PTB-XL authors (Wagner et al., 2020), which the PTB-XL benchmark
also uses: folds 1–8 / 9 / 10. Patients never cross folds. Folds 9 and 10 hold only patients whose
ECGs were all validated by a cardiologist, and fold 10 is the recommended holdout. So our macro-AUROC
is comparable with the PTB-XL classification literature.

OpenTSLM builds on ECG-QA. ECG-QA's published split (`{train,valid,test}_ecgs.tsv`) is
patient-disjoint but is **not** the fold split:

- it uses 16,054 of PTB-XL's 21,799 ECGs, split about 72/8/20 and drawn from all ten folds;
- 1,547 of its training ECGs, from 1,353 patients, are in our test fold 10.

OpenTSLM's ECG-QA loader reads that split directly. Its ECG-QA-CoT data, described as "80/10/10" in the
paper, ships as ready-made CSVs whose host was unreachable, so its split is unverified. A model
pretrained on ECG-QA, and possibly on ECG-QA-CoT (such as OpenTSLM's released ECG checkpoint), has
seen patients from our test set. Exclude those patients before evaluating such a model on fold 10.
This MVE trains from scratch, so it is not affected. The OpenTSLM TeeMoE paper does not use PTB-XL.
