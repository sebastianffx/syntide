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
