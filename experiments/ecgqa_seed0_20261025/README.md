# Reproduce the seven ECG-QA experiments (25 October 2026)

This directory preserves the scientific code and evidence behind the seven-row
seed-0 comparison. `RESULTS.md` is reproducible offline from the committed metrics.
The actual trainers, data exporters and augmentation builders are frozen copies
of the versions used for these runs; `sources.json` records their original hashes.

**Training reproduction also requires external data and model assets.** Raw ECGs,
prepared datasets, generated signal bundles, embeddings, per-question predictions
and checkpoints are not stored in Git. Their fingerprints are in `inputs/`.
The package does not download them or start a cloud job. Matching the software and
inputs is necessary, but is not a guarantee of bit-identical GPU arithmetic across
different hardware.

## Recreate the reported table

From the repository root, using Python 3.11 or later:

```sh
python3 experiments/ecgqa_seed0_20261025/reproduce.py verify
python3 experiments/ecgqa_seed0_20261025/reproduce.py table
python3 -m unittest discover -s experiments/ecgqa_seed0_20261025 -p test_reproduce.py
```

These commands use only the standard library. They verify the packaged hashes,
seven arms, question counts and weighted slice accounting. `table` reads saved
metrics; it does not rerun inference or independently rescore external predictions.
`evidence/results.json` retains full-precision real/zero validation and test
accuracy, template macro-F1 and invalid rates for all, clean and noisy groups.
`evidence/provenance.json` and `collection-inventory-extracts.json` identify the
original summaries and their verified collection entries.

## What each row means

| Command arm | Training questions | Training updates | Reported checkpoint | Data added to original QA |
|---|---:|---:|---:|---|
| `real_qa` | 82,829 | 15,531 | 12,000 | None |
| `paraphrase` | 165,658 | 15,531 | 10,000 | Offline deterministic template paraphrases |
| `grounded` | 152,118 | 15,531 | 4,000 | Waveform-grounded QA |
| `grounded_acq` | 209,722 | 15,531 | 12,000 | Grounded + acquisition QA |
| `diffusets3030_stack` | 221,394 | 15,531 | 8,000 | Grounded + acquisition + 3,030 accepted generated ECGs |
| `diffusets_full_stack` | 227,231 | 15,531 | 10,000 | Grounded + acquisition + 4,515 accepted generated ECGs |
| `exposure_matched` | 94,501 | 12,000 | **Fixed 12,000** | Only QA from the frozen 3,030 generated ECGs |

The first six arms select checkpoints using option-NLL on 2,000 selection
questions. They use batch 8 with accumulation 2, so their total optimizer-example
budget is equal but their original-QA exposure is not. The final arm preserves
the baseline's exact first 192,000 original draws and adds 24,000 generated draws
from an independent sampler: 16 original + 2 generated per update. It uses loss
weights 8/18, 8/18 and 2/18 and retains the baseline LR horizon of 15,531 and
1,554-step warmup. Its diagnostic selection peak never replaces `primary.pt`.

All seven use seed 0, the same base revision, answer/rationale training target,
held-out files, greedy generation and 384-token generation limit. The original
baseline used an H100, original augmentations an A100, and the three DiffuSETS
arms an A40. Generated-answer batch size was 512 for the original four and 256
for the later three. Validation has 27,264 questions; test has 26,841 (18,805 clean,
8,036 noisy). The all-group F1 is not a weighted average of slice F1s.

The final arm was designed after prior held-out results were known. Comparisons
with the older stacks change data mix, sampling and checkpoint policy together.
Seed SD is null. The small test accuracy improvement over baseline does not
repeat on validation, and the controls do not show stronger waveform dependence.
Zero-waveform evaluation retains the text and original mean/std context.

## Files and environments

| Location | Purpose |
|---|---|
| `baseline_runtime/` | Exact baseline trainer, data/metrics implementation, audit code and tests |
| `generation_runtime/` | Frozen DiffuSETS and QA generation/export code; original versus executed runners |
| `generation_runtime/original_qa_overrides/` | Earlier QA code required for the original four-arm corpora |
| `exposure/` | Exact original-exposure builder, sampler, trainer, scientific runner and tests |
| `configs/arms/` | Seven actual configs, training results and run audits |
| `configs/protocols/` | Exact historical protocols, including immutable scientific bindings |
| `inputs/` | External dataset, corpus, candidate, embedding and checkpoint fingerprints |
| `evidence/` | Compact metrics, realized generation distributions and provenance |
| `manifest.json` | Size/SHA256 inventory of this package (excluding itself and Python caches) |

Use separate generation and training environments. Both source trees contain a
regular `syntide` package; putting both on `PYTHONPATH` does not merge them.
The exact observed packages are saved in each `runtime-pip-freeze.txt` and in the
arm configs. Training used Python 3.11.11, torch
`2.8.0.dev20250319+cu128`, transformers `4.53.3`, numpy `2.2.6`, pyarrow `21.0.0`,
and open-flamingo `2.0.1`. The requirements list alone does not recreate the CUDA
image: preserve the pinned torch build and install open-flamingo without replacing
its dependencies.

Install OpenTSLM from a Git checkout at commit
`2968f4b891baab4307f7e9d0043e87677b593a30`, exposing its `src` directory or installing
it editable. Its Git identity is checked at runtime. The frozen base is
`google/gemma-3-270m` revision `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`.
The trainer requires CUDA with BF16 support. Do not substitute an unpinned current
OpenTSLM, tokenizer, checkpoint or torch version and call it the same run.

Historical protocols/runners contain expired paths and timing limits. They are
execution evidence, not launch defaults. Fresh runs need fresh output directories
and a separately recorded operational configuration; leave the archived files
and completed evaluation claims intact.

## Restore or rebuild the inputs

The shortest route to a fresh training run is to restore the seven prepared
datasets and verify every file against `inputs/prepared-datasets.json`.
`inputs/checkpoint-artifacts.json` identifies the original selected checkpoints.
The frozen evaluator also requires the corresponding config, training result and
selection history; a checkpoint alone is insufficient. Preserve completed claims
and predictions and use those predictions to rescore completed evaluations.
External paths in these manifests are relative to the archived study, not public
download URLs.

To rebuild instead, use the frozen code and the exact upstream TimeNet registry,
ECG-QA-CoT/PTB-XL revisions and pool manifest. The builder uses patient folds
1–8/9/10, the fixed 50% training pool, all original training rows and distinct
evaluation questions; paired/cropped tasks are excluded. A selection-size or
row-order change produces a different experiment. The selection subset is fixed
at 2,000 with seed 0. In the training environment:

```sh
REPRO="$(pwd)/experiments/ecgqa_seed0_20261025"
export PYTHONPATH="$REPRO/baseline_runtime/src:$REPRO/baseline_runtime/trainers/src:/path/to/OpenTSLM/src"
python -m syntide.baseline.data --registry /path/to/frozen-registry \
  --manifest "$REPRO/baseline_runtime/manifests/ecgqa-artifact-50/pool50.json" \
  --output /path/to/fresh-baseline-prepared --selection-size 2000 \
  --selection-seed 0 --train-role pool
```

For original paraphrase, grounded and acquisition corpora, copy the generation
runtime to a separate working directory and apply `original_qa_overrides/src`
over that copy's `src`. Use its `syntide.generate.qa build --config ... --output ...`
entry point and the archived corpus configurations. Rebind only filesystem
locations in copies of configs; record the resulting config hashes. Paraphrases
use the deterministic template provider, not a paid teacher model. Export each
corpus with the baseline prefix using
`syntide.generate.qa export --baseline-prepared ... --corpus ... --output ...`;
pass grounded then acquisition corpora in that order for the combined arm.
Do not modify the frozen later DiffuSETS runtime while rebuilding these corpora.

For DiffuSETS, restore the exact frozen 5,000-slot plan, published embeddings,
model assets and source parents listed in `inputs/diffusets-inputs.json` and
`configs/diffusets-generation-identity.json`. `inputs/candidate-slots.csv` is an
audit index, not a replacement generator plan. The frozen generator's `validate`,
`generate` and `prepare` entry points are in
`generation_runtime/scripts/runpod/diffusets_generate.py` (see `--help`). Keep
exactly 5,000 attempts with no rejected-slot replacements. Full generation accepted
4,515 (90.3%); the snapshot uses 3,030 accepted signals from slots 0–3,379, not a
proportional sample of the full allocation. Snapshot preparation is preserved in
`generation_runtime/scripts/prepare_snapshot_qa.py`.

That snapshot builder requires the complete inventoried prefix directory,
including rejected candidates and its frozen candidate plan, identity,
`snapshot_manifest.json` and `source_heartbeat.json`; restoring accepted signal
directories alone is insufficient. Its `--help` lists the registry, baseline,
grounded/acquisition corpus and fresh output arguments. The archived
`generation_runtime/planning/build_plan.py` preserves the allocation recipe with
historical host paths; it is provenance, not a portable launcher. The included
embedding extractor is captured local source whose origin is explicitly marked
in `sources.json`, rather than a remotely verified source pin. It needs the exact
metadata CSV named in the frozen plan.

Conditioning used 3,874 training parents, 1,000 slow/3,000 middle/1,000 fast slots
and published exact-report-matched ada-002 vectors. Their legacy API input
provenance is unknown; no new embedding API calls were used. Parent diagnoses and
noise are not generated labels. Accepted signals are measured anew to produce
11,672 snapshot or 17,509 full generated QA rows. Preparation freezes the original
evaluation files byte-for-byte. Realized distributions are saved separately in
`evidence/diffusets-measurement-summary.json`.

Build the last arm from the restored baseline, snapshot and generated corpus:

```sh
python "$REPRO/exposure/prepare_exposure_data.py" \
  --baseline /path/to/baseline-prepared \
  --snapshot-prepared /path/to/snapshot-prepared \
  --generated-corpus /path/to/snapshot-generated-corpus \
  --snapshot-manifest /path/to/snapshot_manifest.json \
  --output /path/to/fresh-exposure-prepared
```

This checks the original row prefix and order, uses only measured generated QA,
remaps signal indices, and preserves held-out files. Check its source manifest
and builder audit against `inputs/exposure3030/`. Original source-path strings
are part of some provenance hashes: an otherwise equivalent fresh export at a
different path can have a different manifest hash. Keep that distinction explicit
instead of editing an archived manifest to force a match.

## Train and evaluate a fresh run

The helper prints, but does not execute, the exact numerical trainer arguments
for an arm. Its printed `PYTHONPATH` selects the frozen baseline runtime; install
OpenTSLM editable in that environment (or append its pinned checkout's `src`).

```sh
python3 "$REPRO/reproduce.py" command --arm exposure_matched \
  --prepared /path/to/exposure-prepared --run-dir /path/to/new-run --shell
python3 "$REPRO/reproduce.py" command --arm exposure_matched \
  --prepared /path/to/exposure-prepared --run-dir /path/to/new-run \
  --phase validation --shell
```

Use any arm key from the table. `--phase selection`, `validation`, or `test`
prints its evaluation command; `--zero-signal` adds the corresponding control.
Training and selection need no access to held-out answers beyond the frozen
selection subset. Before opening validation/test, run the frozen full-token
preflight, verify initialization, packages, source/data hashes and zero truncation,
then freeze and audit the selected checkpoint. For the final arm, reconstruct
both sampler streams and require exactly 192,000 original + 24,000 generated
draws at fixed update 12,000. The frozen runners and their tests encode these
checks. A printed trainer command alone is not a complete audited study runner.

The snapshot continuation and final exposure arm calibrated generated-answer time
on selection before their four held-out passes. The full-candidate runner retained
its conservative 0.3-second-per-question floor. Option-NLL selection timing is
different from generated answer timing. Execute each real/zero validation and test pass once, with frozen
settings and its durable claim; preserve an interrupted claim rather than
rerunning it under the same identity. Never substitute a diagnostic peak for the
final arm's primary checkpoint or use the old equal-max-steps audit for it.

## CPU verification

The root helper tests need only Python. The scientific unit tests need the
dependencies of the corresponding runtime, but do not perform training or
held-out inference. Baseline tests are under `baseline_runtime/tests/baseline/`;
the exposure test launcher is `exposure/run_cpu_tests.py`. Use each runtime's
own import path. Test fixtures model corrupt artifacts, sampler reconstruction,
data preservation and checkpoint/claim checks, not new scientific results.

The packaged suites passed 100 baseline, 64 generation and 38 exposure tests.
To run them with absolute paths to Python environments containing their test
dependencies:

```sh
(cd "$REPRO/baseline_runtime" && PYTHONPATH=src:trainers/src "$TRAIN_PY" -m pytest \
  -q -c /dev/null -p no:cacheprovider --confcutdir=. \
  tests/baseline tests/test_qa_baseline_audit.py trainers/tests/test_opentslm_recovery.py)
(cd "$REPRO/generation_runtime" && PYTHONPATH=.:src "$GEN_PY" -m pytest \
  -q -c /dev/null -p no:cacheprovider --confcutdir=. tests)
"$TRAIN_PY" "$REPRO/exposure/run_cpu_tests.py"
```

These CPU checks validate code behavior and packaging. They do not certify the
local test environment as the original CUDA training environment.
