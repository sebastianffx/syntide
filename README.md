# SYNTIDE

A synthetic data engine for temporal AI. The first experiment asks whether targeted, physically grounded
ECG artifact editing beats standard augmentation on naturally artifact-bearing ECGs from unseen
patients. See `docs/SYNTIDE_Architecture.md` and `docs/SYNTIDE_MVE.md`.

The work is built in three phases, each with its own package, tests and file outputs:

| Phase | Package | Output |
|---|---|---|
| 1. PTB-XL in TimeNet | `syntide.data` | TimeNet registry `physionet/ptb-xl@1.0.3`, `manifests/ptbxl-artifact-v2/pool{10,25}.json` |
| 2. Generator + training stream | `syntide.generate`, with the trainer seam `syntide.trial`; the trainer lives in `trainers/` | one run directory per trial |
| 3. Evaluation | `syntide.eval` | paired comparisons, dashboard views, results table |

Training is not part of SYNTIDE:
- SYNTIDE writes a frozen `trial.json` per run. It serves the batches, enforces the step budget, records every recipe, and checks what the trainer hands back.
- Any loop (plain PyTorch, Lightning, Hydra) that goes through `syntide.trial` can train a trial. The first one is a plain PyTorch trainer in `trainers/`, a separate workspace package that imports syntide.
- Trainers and evaluation log to the same dashboard: local files, TensorBoard or Weights & Biases, chosen in `conf/tracker/*.yaml`.

See `docs/implementation_plan_v2.md`, Phases 2 and 3.

## Setup

```bash
uv sync          # Python 3.13 (Hydra 1.3 does not run on 3.14); installs syntide, the dev tools and the trainer
uv run pytest    # fast, synthetic fixtures only; no PTB-XL download needed
uv sync --extra wandb   # only for the Weights & Biases dashboard
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
`check_recipe_parents` (every recipe starts from `depleted`). `syntide.trial.prepare` runs the gate again
before every trial, whichever trainer runs it.

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

## Phase 2: calibrate the generator, plan trials, then train them with any trainer

Phase 2 holds:
- the electrode-space artifact editor, its calibration and Gate 4 (arms A2/A3);
- A1's standard augmentation;
- the planner;
- the trainer seam (`syntide.trial`) and the paired training stream;
- the switchable dashboard;
- the plain PyTorch reference trainer (`trainers/`).

```bash
# 0. Calibrate the editor and run Gate 4 for both regimes (about 20 s; needed before A2/A3 are planned).
uv run python -m syntide.generate
# 1. Plan: write one frozen trial.json per (regime, arm, seed). Quote the brackets in zsh.
uv run python -m syntide.generate mode=plan 'select.arms=[A0,C1]' 'select.regimes=[scarcity25]' 'select.seeds=[0,1,2]'
# 2. Train: the reference trainer skips the trials it already finished.
uv run python -m syntide_trainers.torch_loop "runs/ptbxl-artifact-v2/scarcity25/*/seed*"
uv run tensorboard --logdir runs
```

The whole experiment is one file, `conf/protocol/ptbxl_artifact_v2.yaml`: the arms, the regimes, the seeds, the batch
size, the validation cadence, the step budgets, and the A1 and editor envelopes. A1 sets the step budgets in a pilot,
in its own runs directory:

```bash
uv run python -m syntide.generate mode=plan runs_dir=runs/pilot2 'select.arms=[A1]' 'select.seeds=[0]' \
  protocol.val_every=100 protocol.regimes.scarcity10.max_steps=6000 protocol.regimes.scarcity25.max_steps=6000
uv run python -m syntide_trainers.torch_loop "runs/pilot2/ptbxl-artifact-v2/*/A1/seed0"
```

Validation AUROC peaks early and then overfits, so each budget is twice the step where `val/auroc_overall`, smoothed
over 5 validations, peaks, rounded up to a multiple of `val_every`:

| Regime | Smoothed peak | `max_steps` | Then, by step 6,000 |
|---|---|---|---|
| scarcity25 | 0.905 at step 1,600 | 3,250 | 0.878; val loss 0.33 → 0.83 |
| scarcity10 | 0.885 at step 700 | 1,500 | 0.857; val loss 0.35 → 1.08 |

The margin covers arms that should peak later than A1 (C2 trains on 36% more ECGs). The best checkpoint is still
chosen on validation, so the extra steps cost only compute and the final-checkpoint sensitivity. The full grid of 72
trials takes about 2–3 hours on an M1 Max.

### The artifact editor (A2, A3) and Gate 4

`ecg.electrode_artifact@1.0.0` adds one artifact to one of the nine electrodes (RA, LA, LL, V1–V6) and projects it
into the 12 leads with the Einthoven, Goldberger and Wilson equations. A left-arm artifact therefore shows in every
lead except II. The v1 artifact types are `baseline_drift` (a sinusoid), `static_noise` (high-frequency noise over the
whole record) and `burst_noise` (the same noise over 0.5–3 s). `electrodes_problems` is deferred. Every edit is a
recipe (parent, controls, seed), so a ledger row regenerates it bit-exactly.

Severities live in descriptor space (`ecg.electrode_artifacts@1`), the same measurements for natural ECGs and edits:

| Type | Descriptor | Retained ECGs (pool25, physiology included) | Envelope |
|---|---|---|---|
| `baseline_drift` | half peak-to-peak below 0.7 Hz | median 0.70 mV, max 2.9 mV | 0.1–3.5 mV, 0.1–0.4 Hz |
| `static_noise` | robust RMS (1.4826 × MAD) above 15 Hz | median 0.015 mV, max 0.056 mV | 0.005–0.1 mV |
| `burst_noise` | robust RMS of the loudest segment above 15 Hz | median 0.029 mV, max 0.094 mV | 0.01–0.6 mV, 0.5–3 s |

A1's wander and noise share the drift and static-noise ceilings, and planning checks this.

`mode=calibrate` reads only each regime's `retained_artifact` ECGs. It writes the gap report to
`manifests/<protocol>/calibration/poolXX.json`. A2 (`coverage_matched`) draws types and severities as the retained
ECGs have them. A3 (`gap_targeted`) up-weights rare types and severities where the retained density is low. Both
policies map the same random draws, so A2 and A3 differ only in their controls. At 25%, A2 asks for 69% static
noise, 22% drift and 9% bursts at the retained median severities, while A3 asks for 53% bursts and puts most of its
severities above what the retained ECGs show.

Gate 4 then edits 1,000 recipes per policy over `depleted` parents. For every control of every type, it regresses
realized on requested (slope in [0.9, 1.1], |intercept| ≤ 5% and MAE ≤ 10% of the range, corr ≥ 0.9). It passes for
both regimes: every slope is between 0.96 and 1.00, and acceptance is 99.7% (A2) and about 98.6% (A3). It writes
`generator_gate_poolXX.json`, every candidate (`generator_gate_poolXX/recipes.parquet`) and a 40-recipe review sample
at the severity extremes, with an empty `verdict` column for an expert
(`generator_gate_poolXX/review_sample.csv`, never overwritten).

During training, a candidate that fails `bounds@1` or `alignment@1` is retried with the same parent and slot and the
next draw, up to `max_retries` times. A slot whose every attempt fails keeps its parent unchanged and is counted in
`exhausted_slots`. Calibrations and gate results are frozen once written: delete them on purpose to rebuild them, and
re-plan the editor trials, whose ids depend on the calibration's hash.

### What SYNTIDE enforces, whichever trainer runs

| When | Check |
|---|---|
| Planning | Gate 1 on each manifest. Arms train only on `depleted`, `natural_mix` or `pool`, and a transform arm only on `depleted`. `max_steps` is a positive multiple of `val_every`. A1 and the editor share their severity ceilings. Editor arms need a gate that passed on exactly the regime's calibration file, fit on the same manifest and editor settings. If any check fails, nothing is written, and a planned `trial.json` is never replaced by a different trial. |
| `prepare` | `trial.json` was not edited, and the syntide version, the local data fingerprint, the manifest and (for editor arms) the calibration are the ones it was planned with. Gate 1 runs again. Only the arm's role and `val` are loaded; `test` never is. |
| The stream | Exactly `max_steps` batches of `batch_size`, counted by SYNTIDE: presentations, transform slots and recipes. |
| `finish` | Refuses a partly consumed stream, a final checkpoint before `max_steps`, a validation history off the `val_every` cadence, a best checkpoint that is not the first argmax of `val/auroc_overall`, probabilities that do not reproduce the reported scores, missing checkpoint files, and a run directory that already holds a result. |

Pairing: randomness comes from separate streams for initialization, parent order, transform slots and transforms. They
are keyed by the replicate seed, the step and the slot, and never by the arm. So within a (regime, seed) block, A0–A3
see the same parents in the same order and the same transform slots, and start from the same weights. A0's slots hold
unchanged parents. C1 and C2 have no transform slots.

### A run directory

`runs/<protocol>/<regime>/<arm>/seed<n>/` holds:

| File | Written by | Content |
|---|---|---|
| `trial.json` | the planner | the frozen `TrialSpec`; its `trial_id` hashes every scientific input, not paths |
| `recipes.parquet` | the stream | one row per generated candidate (empty for A0, C1, C2); every row regenerates its signal bit-exactly |
| `exposure.json` | the stream | presentations, transform slots, candidates, accepted, exhausted slots, distinct recipes |
| `predictions_val.parquet` | `finish` | per record × class probabilities of the best and final checkpoints |
| `result.json` | `finish`, last | best step and score, validation history, exposure, trainer fingerprint and `init_sha256`, checkpoint shas, environment |
| `best.pt`, `final.pt` | the trainer | its checkpoints |
| `metrics.jsonl` or TensorBoard events | the tracker | the dashboard: `train/loss`, `val/*`, `budget/*`, `generator/acceptance_rate` |

### The reference trainer

`trainers/conf/torch_loop.yaml` sets the model width, AdamW's learning rate and weight decay, the device, the
`train/loss` cadence and the tracker. Override any key on the command line (`model.width=128 device=cpu`), and pick
the dashboard with `--tracker conf/tracker/files.yaml` (or `tensorboard.yaml`, `wandb.yaml`). The model is a 1-D ResNet
with about 0.95M parameters; it trains at about 30 ms per step on an M1 Max (MPS).

The model and optimizer settings and the device form the trainer fingerprint. Compared runs must share it, so a run
directory trained with one fingerprint refuses another. The model gate's wider model therefore needs its own
`runs_dir`. A new trainer (for example, Lightning) is a package that makes the same `syntide.trial` calls; see the
docstrings in `src/syntide/trial.py` and `trainers/src/syntide_trainers/torch_loop.py`.

## The 50% experiment: every arm against 100% of the patients

`conf/protocol/ptbxl_artifact_50.yaml` runs the six arms on one data size, 50% of the training patients, and adds
the target they are measured against: C2 on every training patient (`full100`, which lists `arms: [C2]`, so no other
arm runs there). Each runs 3 seeds, so there are 21 trials. They all share one budget of 6,500 steps, twice the
step where real-only training on 100% of the patients peaks (`runs/ptbxl-scaling`).

```bash
# Manifests: pool50 and pool100, whole patients, checked by the leakage gate.
uv run python -m syntide.data challenge.id=ptbxl-artifact-50 'challenge.pool_fractions=[0.5,1.0]'
# Calibrate the editor and run Gate 4 (only regimes with an editor arm), then plan and train the 21 trials.
uv run python -m syntide.generate protocol=ptbxl_artifact_50
uv run python -m syntide.generate protocol=ptbxl_artifact_50 mode=plan
uv run python -m syntide_trainers.torch_loop "runs/ptbxl-artifact-50/*/*/seed*"
# Score the held-out test fold once, then draw the figures and the summary table.
uv run python -m syntide.eval.holdout runs/ptbxl-artifact-50
uv run python -m syntide.eval.arms_figure runs/ptbxl-artifact-50
```

At 50% the pool holds 8,579 ECGs. Withholding 90% of the noisy patients leaves 6,389 ECGs for A0–A3, and C1 gets
the same number drawn at the natural noisy rate. Gate 4 passed: 99.8% (A2) and 98.7% (A3) of generated edits were
accepted.

`syntide.eval.holdout` is where the test fold is spent, once:
- it refuses to run until every trial in `plan.json` has finished;
- it checks that each best checkpoint is unchanged since training;
- it writes `test_claim.json` (create-only) naming every trial and checkpoint it scores, so a later call can only
  finish that same list;
- each run gets `predictions_test.parquet` and `test_scores.json`, which are never rescored.

`syntide.eval.arms_figure` writes:
- `figures/arms_overall.png`: validation and test macro-AUROC per arm, with every seed, the mean ± SD, the 100%
  target as a line, and each arm's gap to it;
- `figures/arms_slices.png`: the same for noisy and clean records;
- `arms_summary.md`: the numbers, each arm's seed-paired change over A0, and a convergence check.

## Phase 3: judging the results

`python -m syntide.eval` judges a protocol's runs by decision rules frozen before the test fold is scored. The rules
are one file per protocol, `conf/analysis/<protocol>.yaml`, and `protocol=<name>` selects both.

```bash
# 1. Before the test fold is claimed: freeze the decision rules into runs/<protocol>/eval/rules.json (create-only).
uv run python -m syntide.eval protocol=ptbxl_artifact_50 mode=freeze
# 2. Any time: the validation report (partial while trials train), and the final-checkpoint sensitivity check.
uv run python -m syntide.eval protocol=ptbxl_artifact_50
uv run python -m syntide.eval protocol=ptbxl_artifact_50 checkpoint=final
# 3. What the generator made: coverage, acceptance, signal examples and a label audit (report-only, ~1 min on CPU).
uv run python -m syntide.eval protocol=ptbxl_artifact_50 mode=audit
# 4. Once every trial is finished: score the test fold through syntide.eval.holdout, then report on it.
uv run python -m syntide.eval protocol=ptbxl_artifact_50 split=test
```

Each report goes to `runs/<protocol>/eval/<split>[-final]/` (the audit to `eval/audit/`):

| File | Content |
|---|---|
| `comparisons.json` | every paired contrast (estimate, CI, per-seed and per-class effects, counts) and the findings |
| `results.md` | the primary question, the fixed sequence, the results table, gap closed, diagnostics, exposure, gates |
| `figures/*.png` | `forest` (paired effects), `per_class`, `learning_curves`; the audit adds `coverage_*` and `signal_examples_*` |
| `dashboard/` | the same figures and text in the chosen dashboard: `tracker=files`, `tensorboard` (default) or `wandb` |

How a contrast `a - b` is computed (`syntide.eval.contrasts`):
- **Estimate:** the mean over seeds of the paired difference in macro-AUROC on one slice: artifact (ECGs with any
  PTB-XL signal-quality flag), clean or all.
- **CI:** a crossed bootstrap (2,000 draws). Each draw resamples the slice's patients, with all of a drawn patient's
  ECGs, and, independently, the seeds. Both arms are scored on the same draw, and every contrast uses the same draws.
- **Pairing:** a contrast is refused unless both arms share seeds, records and labels, the trainer fingerprint, the
  number of presentations, each seed's initial weights, and (between transform arms) the transform slots. An arm in
  another regime is named `C2@full100`; it pairs when those all match.
- **Run contract:** before scoring, every run is re-checked. `trial.json` must be unedited, the checkpoints must
  have their recorded sha256, and the predictions must cover every record and class and reproduce the score the
  run reported.

The frozen rules of `ptbxl-artifact-50` (sha256 `e6a66b593c2c…`, frozen 2026-10-22 14:16 UTC):

| Rule | Value |
|---|---|
| Primary | A3 − A1 on the artifact slice, scarcity50 |
| Benefit | artifact CI low > 0 **and** clean CI low > −0.01 (non-inferiority) |
| Practical target | +0.02, reported as supported (CI low > 0.02), ruled out (CI high < 0.02) or plausible |
| Fixed sequence | A3 − A1 → A3 − A2 → A2 − A1; each step only if every earlier one showed a benefit |
| Gap gate (Gate 3) | C1 − A0 ≥ 0.02 on the validation artifact slice; judged on validation even in the test report |
| Gap closed | (arm − A0) / (C1 − A0), only if the gap gate passed; (arm − A0) / (C2@full100 − A0). Each ratio only when its denominator's CI is above 0 |
| Diagnostics | A1 − A0, C1 − A0, C2 − A0, C2@full100 − A0, C2@full100 − C2; reported, never evidence |

The test report needs three things:
- the rules it runs with must equal the frozen ones;
- `plan.json` must be unchanged since the freeze;
- `test_claim.json` must be dated after the freeze.

`split=test` checks the first two, scores the test fold once through `syntide.eval.holdout` (runs already scored
are only read back), and then checks the claim date.
