# SYNTIDE

Syntide helps researchers decide **what synthetic time-series data to generate**, create it,
and measure whether it improves downstream models. It starts with PTB-XL ECGs and supports
exploring custom `(time_series, text)` and `(time_series, question, answer)` datasets.

![Syntide module goals and data flow: data feeds explore and generate; explore exports generation plans; generate returns catalogs to explore and supplies trials to trainers; eval compares trainer results.](docs/images/syntide-workflow.png)

| Module | Goal | Main output |
|---|---|---|
| `syntide.data` | Prepare signals, labels and patient-separated splits | TimeNet registry and challenge manifests |
| `syntide.explore` | Find coverage gaps and choose generation targets | Interactive explorer and generation plan |
| `syntide.generate` | Augment, edit or generate signals; plan controlled experiments | Recipes, synthetic catalogs and frozen trials |
| `trainers` | Train downstream models through `syntide.trial` | Checkpoints and predictions |
| `syntide.eval` | Audit generation and compare downstream results | Metrics, figures and reports |

Training lives in the separate `trainers/` package. Syntide supplies its data stream and checks
the trial contract; other trainers can implement the same interface.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run commands from this
repository's root (`synthetic_data_engine/`). Python **3.13** is required.

```bash
uv sync --extra explore --extra diffusets
```

This installs Syntide, the reference trainer and the optional exploration/DiffuSETS dependencies.
Use `uv sync` for the core artifact experiment only, or add `--extra wandb` for Weights & Biases.
Exploration and full generation support CPU, Apple MPS and CUDA; `--device auto` selects an
available device. Full generation sets `device` in its JSON configuration.

## Prepare the data

```bash
uv run python -m syntide.data
```

The first run downloads PTB-XL 1.0.3 (about 1.84 GB), builds `datasets/timenet-registry/`, and
writes patient-separated manifests to `manifests/ptbxl-artifact-v2/`. Later runs reuse the data.
Training uses folds 1–8, validation uses fold 9, and the held-out test uses fold 10.

## Explore and generate

### 1. Find a coverage gap

```bash
uv run python -m syntide.explore analyze \
  --dataset ptbxl \
  --manifest manifests/ptbxl-artifact-v2/pool25.json \
  --sample 512 --backend chronos --device auto \
  --output outputs/explore/ptbxl --serve
```

This embeds 512 multichannel recordings from the manifest's depleted training pool. The explorer
shows signal clusters, label counts, metadata slices, text novelty and channel statistics.
Select recordings, choose generation methods and counts, then export a generation plan.

Use `--sample-fraction 0.1` instead of `--sample 512` to sample a fraction. Other sources are
`--dataset ecgqa` and `--dataset jsonl --input catalog.jsonl`; the other signal backend is
`--backend opentslm`. See the [explorer guide](docs/coverage_explorer.md) for formats and options.

To reopen an existing analysis:

```bash
uv run python -m syntide.explore serve outputs/explore/ptbxl
```

### 2. Generate complete ECGs with DiffuSETS

For this example, export a plan containing **only diffusion** and save it as
`outputs/explore/ptbxl/generation-plan.json`. Create `diffusets-config.json` using the
[configuration example](docs/full_generation.md#conditioning-and-configuration).
It specifies model paths, clinical reports, demographics and compatible
**1536-dimensional `text-embedding-ada-002` vectors**. Explorer embeddings cannot replace these vectors.

```bash
# Download pinned model assets once.
uv run python -m syntide.generate.full.assets --output outputs/diffusets-assets

# Execute the first (zero-based) request in the exported plan.
uv run python -m syntide.generate.full \
  --plan outputs/explore/ptbxl/generation-plan.json \
  --config diffusets-config.json --request-index 0 \
  --output outputs/generated/ecg-001
```

The output includes waveforms, a JSONL catalog, per-sample seeds and model/conditioning provenance.
DiffuSETS generates 10-second, 12-lead ECGs. Its native rate is 102.4 Hz; the configuration can
explicitly request resampling to PTB-XL's 100 Hz. See [full generation](docs/full_generation.md)
for conditioning, output schemas and adding another model or modality.

Explorer plans record intentions. The direct plan executor currently handles diffusion requests;
standard augmentation and artifact editing run through the trial workflow below. Generated labels
remain unverified, and full synthetic catalogs need explicit dataset integration before training.

### 3. Inspect generated signals

```bash
uv run python -m syntide.explore analyze \
  --dataset jsonl --input outputs/generated/ecg-001/catalog.jsonl \
  --sample-fraction 1 --backend chronos --no-text \
  --output outputs/explore/generated-ecg --serve
```

A catalog containing both real and synthetic recordings enables origin coloring and paired
waveform comparisons. Use a common channel set and sample rate, and preserve the source-reference
metadata described in the [explorer guide](docs/coverage_explorer.md#compare-real-and-synthetic-signals).

## Train and evaluate the artifact experiment

This workflow compares unchanged real data (A0), standard augmentation (A1), coverage-matched
artifact editing (A2), gap-targeted editing (A3), and real-data controls (C1/C2).
The default protocol plans **72 trials** across two scarcity regimes and six seeds.

```bash
# Calibrate the editor and check its generated artifacts.
uv run python -m syntide.generate protocol=ptbxl_artifact_v2 mode=calibrate

# Write frozen trial.json files.
uv run python -m syntide.generate protocol=ptbxl_artifact_v2 mode=plan

# Freeze evaluation rules before scoring any held-out test data.
uv run python -m syntide.eval protocol=ptbxl_artifact_v2 mode=freeze

# Train planned trials; completed runs are skipped.
uv run python -m syntide_trainers.torch_loop "runs/ptbxl-artifact-v2/*/*/seed*"

# Report validation performance and audit the artifact generator.
uv run python -m syntide.eval protocol=ptbxl_artifact_v2 split=val
uv run python -m syntide.eval protocol=ptbxl_artifact_v2 mode=audit

# Once every planned trial finishes, score the test set once and report.
uv run python -m syntide.eval protocol=ptbxl_artifact_v2 split=test
```

Use the same explicit `protocol` for planning and evaluation: their defaults differ.
Reports are saved under `runs/ptbxl-artifact-v2/eval/`.
Open the local training dashboard with `uv run tensorboard --logdir runs`.

For a small training check, use a separate run directory and select two arms and one seed:

```bash
uv run python -m syntide.generate protocol=ptbxl_artifact_v2 mode=plan runs_dir=runs/smoke \
  'select.arms=[A0,A1]' 'select.regimes=[scarcity25]' 'select.seeds=[0]'
uv run python -m syntide_trainers.torch_loop "runs/smoke/ptbxl-artifact-v2/*/*/seed*"
```

This subset checks training; use the complete protocol for the full set of scientific comparisons.
Quote list overrides and glob patterns as shown, including in zsh.

## Development and guides

```bash
uv run --extra explore --extra diffusets pytest
uv run ruff check src tests trainers
```

Tests use synthetic fixtures and do not require a PTB-XL download or cloud GPU.

- [Coverage explorer](docs/coverage_explorer.md): datasets, embeddings, metrics, comparisons and plan export.
- [Full generation](docs/full_generation.md): DiffuSETS setup, conditioning and backend extensions.
- [PTB-XL experiment reference](docs/ptbxl_artifact_workflow.md): manifests, calibration, leakage checks, arms, budgets and evaluation rules.
- [Trainer interface](docs/training_loop_syntide.md): connecting a training loop through `syntide.trial`.
- [Architecture](docs/SYNTIDE_Architecture.md) and [experiment design](docs/SYNTIDE_MVE.md).
