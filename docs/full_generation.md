# Complete signal generation with DiffuSETS

The explorer's `diffusion` method defaults to `diffusets@1` for ECG. Exporting a plan still only
records the researcher's intentions. A separate command executes one selected diffusion request,
with explicit model assets, clinical conditioning, output format and per-sample seeds:

```sh
uv sync --extra explore --extra diffusets
uv run python -m syntide.generate.full.assets --output outputs/diffusets-assets
uv run python -m syntide.generate.full \
  --plan outputs/explore/generation-plan.json \
  --config diffusets-config.json \
  --request-index 2 \
  --output outputs/generated/ecg-run-001
```

Run these commands from `synthetic_data_engine/`. The index is zero-based: in a plan containing
augmentation, editing and diffusion in that order, diffusion is request `2`. Multiple/mixed
requests require an explicit index. Only that request executes; other indices are recorded as
unexecuted. Counts are total outputs, not outputs per selected recording.

The asset helper downloads approximately 162 MB of compressed inference weights, the two pinned
architecture files, and one published demonstration embedding. It fetches only the needed ZIP
members, verifies their pinned SHA256 hashes, and never downloads the archive's patient datasets.
`assets.json` records source/model revisions and file hashes. Architecture stays outside the
Python package in this cache. Existing cached files must match their hashes.

## Conditioning and configuration

This adapter uses the original [DiffuSETS experiment](https://github.com/Raiiyf/DiffuSETS_Exp)
and [public weights](https://huggingface.co/Laiyf/DiffuSETS), pinned to source
`f9106528a9e5348d94a118a71eb79d459ee4197a` and weights revision
`80b04013aef594c7c3012a8b932e5e09524e581a`. It loads the U-Net and VAE decoder with
`weights_only=True`, checks their shapes strictly, and runs inference in float32.

The checkpoint requires **1536-dimensional `text-embedding-ada-002` vectors**, plus age, sex and
heart rate. The explorer's MiniLM, Chronos and OpenTSLM vectors are not compatible conditioning.
The separate DiffuSETS-KT checkpoint uses another embedding model and is not this adapter.

Store each clinical text vector as a local NumPy `.npy` file with shape `(1536,)`. Produce the
embedding from the exact output of `format_report(report)`, which implements the upstream
report prompt. `|` separates diagnoses. For example:

```python
from syntide.generate.full.diffusets import format_report

embedding_input = format_report("Sinus rhythm")
# Obtain text-embedding-ada-002 for embedding_input through your chosen workflow,
# then save the returned vector with numpy.save("conditioning/sinus.npy", vector).
```

No embedding API is called by Syntide. Supply the matching vector and its exact input explicitly;
the input comparison checks the declaration, not the external provider's training or execution.
An older published vector whose input was not recorded can instead supply `embedding_source`;
its provenance is explicitly marked `unknown_legacy_input`.

An execution configuration for native output:

```json
{
  "generator": "diffusets@1",
  "source_dir": "outputs/diffusets-assets/source",
  "unet_checkpoint": "outputs/diffusets-assets/checkpoints/unet_all.pth",
  "vae_checkpoint": "outputs/diffusets-assets/checkpoints/vae_model.pth",
  "device": "auto",
  "num_inference_steps": 1000,
  "signal": {
    "modality": "ecg",
    "channels": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "sample_rate": 102.4,
    "num_samples": 1024,
    "units": "mV"
  },
  "conditions": [
    {
      "conditioning": {
        "report": "Sinus rhythm",
        "age": 50,
        "sex": "M",
        "heart_rate": 80,
        "text_embedding_path": "conditioning/sinus.npy",
        "text_embedding_model": "text-embedding-ada-002",
        "embedding_input": "Most importantly, the 1st diagnosis is {Sinus rhythm}."
      },
      "source_record_ids": [],
      "source_text_ids": []
    }
  ]
}
```

Paths inside the configuration are resolved relative to the command's working directory;
absolute paths are useful when moving configurations between directories. Optional
`unet_sha256`, `vae_sha256`, and conditioning `text_embedding_sha256` enforce expected file
contents. All actual hashes are saved even when those expectations are omitted.

Conditions cycle in their declared order until the requested count is met. Reference IDs are
optional; when used, they must belong to the exported plan's selected records/texts. They are
conditioning references, not synthetic patients. Reports and demographics are explicit: no
diagnoses, heart rates or QA answers are inferred automatically from the selected cluster.
Plan parameters bind the execution configuration; conflicting overrides fail before generation.

`device` accepts `auto`, `cpu`, `mps`, or `cuda`; `auto` chooses CUDA, then Apple MPS, then CPU.
An explicitly requested unavailable device fails. A sample's seed controls the initial latent
and every DDPM noise draw through its own CPU random generator. Equal seeds can still yield
different numerical results across devices or library versions; those versions and the device
are recorded. `num_inference_steps` defaults to the original 1000 steps. Fewer steps are useful
for implementation checks but their signal quality has not been established by this integration.

## Signal format and output artifacts

DiffuSETS generates **10-second, 12-lead ECGs at 102.4 Hz**, with 1024 samples per lead in mV.
To request PTB-XL's 100 Hz format, explicitly add `"resample": true` and use
`"sample_rate": 100, "num_samples": 1000`. The adapter applies polyphase resampling after native
generation and records both formats. It rejects other modalities, nonstandard channel order,
other durations and non-mV units. No implicit truncation or amplitude normalization is applied.

A successful run publishes a new directory containing:

- `run.json`: immutable run identity, resolved model/configuration provenance and artifact hashes.
- `plan.json` and `config.json`: the source plan and resolved execution configuration.
- `signals/*.npy`: float32 waveforms with shape `[channels, time]`.
- `ledger.jsonl`: sample seeds, conditioning/reference IDs, signal hashes and validation results.
- `catalog.jsonl`: the same generated dataset in the explorer's generic JSONL format.

```sh
uv run python -m syntide.explore analyze \
  --dataset jsonl --input outputs/generated/ecg-run-001/catalog.jsonl \
  --sample-fraction 1 --backend chronos --no-text \
  --output outputs/explore/generated-ecg
```

The runner checks signal shape and finite numeric values. It does not establish physiological
validity or agreement with the requested diagnosis. Generated labels are empty; intended labels,
conditioning and source QA references remain metadata pending review. QA answers/rationales are
not copied into generated supervision. Generation does not train a model or modify an existing
experiment's calibration/trial pipeline.

All conditions and assets are checked before loading model weights. A run writes to a temporary
staging directory and publishes only on complete success; failure does not leave an apparently
complete dataset. Existing output directories are never overwritten.

## Adding models and modalities

The full-generation interface is independent of ECG-specific `Samples` and training streams.
`SignalSpec` carries modality, named channels, sample rate, sample count and units;
`GenerationRequest` carries this specification, a seed and a JSON conditioning object;
`GeneratedSignal` returns the array, the exact specification and JSON metadata.

A backend factory accepts a configuration dictionary and returns an object with:

- `validate_request(request)`: validate capabilities and all conditions without loading weights.
- `generate(request) -> GeneratedSignal`: perform seeded full-signal inference.
- `provenance() -> dict`: return stable, JSON-compatible model/checkpoint/control provenance,
  fully resolved after validation and unchanged by inference.

Register a factory directly with `register_generator("my-imu-model@1", factory)` for Python
usage, or expose it from an installed package:

```toml
[project.entry-points."syntide.full_generators"]
"my-imu-model@1" = "my_package.synthesis:create_backend"
```

The CLI discovers these entry points lazily. `python -m syntide.generate.full --list` lists
available names without loading weights. Each backend owns its conditioning schema and signal
constraints. Choose its versioned ID and modality in the explorer's full-generation controls,
then provide its execution configuration. The engine, seed derivation, ledger and JSONL output
do not need to change for another modality. Tests include a non-ECG PPG backend exercising this
same workflow.

## Fixed 5,000-candidate QA extension

`scripts/runpod/diffusets_generate.py` runs the frozen seed-0 allocation in
`outputs/qa-runs/diffusets-extension/planning/candidate-plan.json`. This is a separate,
selection-tuned exploratory extension. The plan contains exactly 5,000 ordered slots with unique
seeds and training-pool source records. A rejected candidate consumes its slot; there is no top-up.

The driver uses the pinned original DiffuSETS backend, float32, batch size one and 1,000 diffusion
steps, with explicit output resampling to 12 × 1,000 samples at 100 Hz. It makes no embedding API
calls. `--embedding-index` points to JSON with `entries` keyed by source record ID. Each entry binds
its exact source `report`, `age`, `sex`, `subject_uid`, `text_embedding_model`, vector path (relative
to the index), vector SHA256, and provenance. Published legacy vectors must declare
`embedding_source` and `prompt_provenance: unknown_legacy_input`; they must not invent an
`embedding_input`. The index's `source_csv_sha256` must match the frozen plan's PTB-XL metadata file.

For example, from the generation checkout with its compatible dependencies installed:

```sh
PYTHONPATH=src python scripts/runpod/diffusets_generate.py validate \
  --plan /workspace/diffusets-study/planning/candidate-plan.json \
  --plan-sha256 2bc86083e7a684340bd5a10e8a4ea174671e665e3b1f5903e78a865ec3f009f4 \
  --manifest manifests/ecgqa-artifact-50/pool50.json \
  --embedding-index /workspace/diffusets-study/conditioning/embedding-index.json \
  --output /workspace/diffusets-study/generated
```

Use the same arguments with `generate`, plus `--assets /workspace/diffusets-assets --device cuda`,
to execute the plan. Whole-output locking prevents concurrent writers. Each candidate gets a durable
claim followed by an atomically published, checksummed bundle. Rerunning the command verifies and
reuses completed slots, including rejected slots; an interrupted slot retries the same seed and
condition. Thus retries do not increase the number of unique candidate slots, although interrupted
computation may be repeated. Existing published artifacts, plan identity, model/runtime provenance,
measurement code and QA policy cannot change during resume.

Acceptance requires usable measured heart rate, beat count and RR regularity at confidence ≥ 0.8,
plus measured amplitude 0.2–10 mV. Conditioning reports and requested HR remain lineage, not answers.
`acceptance.json` saves the full measured evidence per candidate. After all slots finish,
`summary.json` records accepted/rejected counts and checksums; `measurement-summary.json` reports
requested versus realized HR, regularity, per-record maximum wander/high-frequency/burst descriptors,
and acceptance by the allocation's parent strata. Those descriptors do not establish a noise or
clinical diagnosis.

After completion, `prepare` uses the same `--plan`, `--plan-sha256`, `--manifest` and `--output`, plus:

```sh
  --registry datasets/timenet-registry \
  --qa-output outputs/qa/generated-5000-seed0 \
  --baseline-prepared /workspace/qa-data/real_qa \
  --grounded outputs/qa/grounded --acquisition outputs/qa/acquisition \
  --prepared-output /workspace/qa-data/grounded_acq_gen
```

Run preparation in the same measurement environment as generation. It freezes the complete QA config
and input fingerprints, remeasures accepted ECGs, and exports the original baseline rows plus the
unchanged grounded/acquisition corpora and new generated QA. The baseline selection file stays
byte-identical. QA uses only the four rhythm templates (heart rate, beat count, regularity and rate
band), without evolutions or comparison pairs. Each accepted ECG supplies **3–4 QA rows**: rate-band
questions abstain within 3 bpm of 60 or 100 bpm. Regularity rationales may cite measured RR coefficient
of variation, which supports the already accepted regularity measurement. Original diagnosis labels,
constant duration and missing-lead questions are not added to this extension.

Training runs separately in the unchanged baseline runtime; the generation environment and dependencies
must never replace that runtime. The project's default `uv` environment is Python 3.13, while this
driver and its required TimeNet 0.11 APIs also run on Python 3.11 with explicitly compatible installed
dependencies and the checkout supplied through `PYTHONPATH`.
