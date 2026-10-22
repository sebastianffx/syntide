# Coverage explorer

`python -m syntide.explore` helps a researcher choose which recordings and semantic regions
should receive more synthetic examples. It embeds a reproducible sample of training recordings,
clusters signal embeddings, reports label coverage and text novelty, and writes an interactive
explorer. A selected region can become a reproducible generation plan containing augmentation,
synthetic editing, diffusion generation, or a mixture with explicit counts.

The command exports generation intentions. It does not create signals, launch training, rent
hardware, or convert the plan into an existing SYNTIDE experiment trial automatically.

## Install and first analysis

Run commands from `synthetic_data_engine/` with the existing local registry available:

```sh
uv sync --extra explore

uv run python -m syntide.explore analyze \
  --dataset ptbxl \
  --registry datasets/timenet-registry \
  --sample 512 --seed 42 \
  --backend chronos --device auto \
  --clusters 8 --projection umap \
  --waveform-points 256 \
  --output runs/explore/ptbxl-chronos-512 \
  --serve
```

The default signal model is pinned `amazon/chronos-t5-tiny`. The initial model download needs
network access; later runs can use the local Hugging Face cache. The optional dependency group
includes Chronos, sentence transformers and UMAP.

`--device auto` chooses CUDA when available, then Apple MPS, then CPU. Use `--device mps` for an
explicit Apple Silicon run, `--device cuda` on a GPU machine, or `--device cpu` for a CPU run.
An explicitly requested unavailable device raises an error. `--batch-size` controls model batches;
reduce it when memory is limited. Recording lengths and channel counts also affect cost.

For an existing scarcity experiment, pass its manifest so discovery uses that experiment's
depleted population:

```sh
uv run python -m syntide.explore analyze \
  --dataset ptbxl --registry datasets/timenet-registry \
  --manifest manifests/ptbxl-artifact-50/pool50.json \
  --sample-fraction 0.10 --seed 42 \
  --backend chronos --device mps \
  --target-count 100 --target-fraction 0.02 \
  --waveform-points 256 \
  --output runs/explore/ptbxl-depleted
```

Use the path of the actual manifest being studied. A manifest defaults to `--role depleted`;
without one, the default role is `train`. Validation/test recordings never enter discovery.
Manifest selection also excludes withheld records and patients: even a nominal training role
such as `pool` is rejected if it includes withheld members. Without a manifest, discovery uses
the full official training partition and cannot infer withholding from a separate experiment.

## ECG-QA and OpenTSLM-SP

```sh
uv run python -m syntide.explore analyze \
  --dataset ecgqa --registry datasets/timenet-registry \
  --manifest /path/to/ecgqa-artifact-50/pool50.json \
  --sample 512 --seed 42 \
  --backend opentslm --device cuda --batch-size 16 \
  --text-fields question answer \
  --clusters 8 --waveform-points 256 \
  --output runs/explore/ecgqa-sp-512
```

Point `--manifest` at the upcoming experiment’s actual manifest, including when it lives in a separate
checkout. Omit it to explore the registered dataset’s full training partition. The explorer has its
own read-only QA adapter and does not import the experiment’s trainer or data-builder package.

The same command supports `--device mps` or `--device cpu` on a Mac. This implementation loads
the OpenTSLM signal encoder only; it does not load its language model or run the upcoming
OpenTSLM-Flamingo training experiment.

The default OpenTSLM checkpoint is pinned `OpenTSLM/llama-3.2-1b-tsqa-sp`, trained at the TSQA
curriculum stage. It avoids the OpenTSLM ECG-QA checkpoints that overlap this experiment's test
population. Custom checkpoints require an explicit training-data provenance declaration and
`contains_ecgqa: false`; changing a checkpoint's filename does not establish its eligibility.
The checkpoint loader strictly checks encoder weights and records the checkpoint hash.
Supply that declaration with `--checkpoint-provenance checkpoint-provenance.json`, alongside
`--checkpoint path/to/model_checkpoint.pt` or a custom `--model`. For example:

```json
{
  "training_data": "Exact dataset references and training split description",
  "contains_ecgqa": false,
  "source": "Training manifest or model-card reference"
}
```

Complete the declaration from the checkpoint's actual training history; it is provenance to
review, not an automated proof that its training data are disjoint.

ECG-QA questions are joined to the existing PTB-XL 100 Hz signals. Membership follows PTB-XL
patient folds 1–8 / 9 / 10, not the CoT release's own split. The release split remains available
as `cot_split` metadata. Repeated questions with different rationales remain visible, while
each multichannel recording contributes exactly one point and one unit to label counts.

The source also retains template IDs, answer options, question types, clinical context and
rationales. This supports inspecting the coming OpenTSLM ECG-QA experiment, without approving its proposed label-effect table
or rationale eligibility rules.

## Generic `(time_series, text)` and `(time_series, question, answer)` data

Use `--dataset jsonl --input path/to/catalog.jsonl`. The catalog refers to local `.npy` or `.npz`
signals; paths are relative to the catalog. Each array has shape `[channels, time]`, with finite
real values. A minimal catalog can contain both kinds of tuple:

```jsonl
{"record_id":"rec-001","subject_uid":"patient-001","split":"train","signal_path":"signals/rec-001.npy","sample_rate":100,"channels":["I","II"],"units":"mV","labels":["rhythm:sinus"],"text_id":"report-001","text":"Sinus rhythm with normal conduction."}
{"record_id":"rec-002","subject_uid":"patient-002","split":"train","signal_path":"signals/rec-002.npy","sample_rate":100,"channels":["I","II"],"units":"mV","labels":["rhythm:afib"],"text_id":"qa-002","question":"Which rhythm is present?","answer":"Atrial fibrillation","text_metadata":{"template_id":4}}
{"record_id":"rec-002","subject_uid":"patient-002","split":"train","signal_path":"signals/rec-002.npy","sample_rate":100,"channels":["I","II"],"units":"mV","labels":["rhythm:afib"],"text_id":"qa-003","question":"Is the rhythm regular?","answer":"No","text_metadata":{"template_id":5}}
```

```sh
uv run python -m syntide.explore analyze \
  --dataset jsonl --input datasets/custom/catalog.jsonl \
  --sample-fraction 1 --seed 42 \
  --backend chronos --device auto \
  --text-fields text question answer \
  --projection pca --clusters 4 --waveform-points 256 \
  --output runs/explore/custom
```

Required fields are `record_id`, `signal_path`, `sample_rate`, `channels`, and explicit `split`
(`train`, `validation`/`val`, or `test`). Supply a stable `subject_uid` to enforce patient-level
separation; if omitted, the record ID becomes the subject ID, so separate recordings from an
unknown common patient cannot be detected. `units` defaults to `unknown`. Labels are optional.

Repeated `record_id` rows attach more tuples to the same recording. Their signal location,
sample rate, channels, units, subject, split and recording metadata must agree; their labels
are combined. An identical repeated `text_id` is deduplicated, while conflicting reuse fails.
Without `text_id`/`task_id`, stable catalog line-based text IDs are assigned. Alternatively,
put tuple objects inside a recording's `texts` array. Use `metadata` for recording-level slices
and `text_metadata` for tuple-level properties; a nested text object's `metadata` is also accepted.
`rationale` is retained separately from question and answer.

An NPZ archive may use a `signal` member, a single array member, or an explicit `signal_key`.
All recordings in an embedding cohort must have the same channel-name set; channel order is
aligned by name. Sample rates must match unless `--target-sample-rate HZ` explicitly resamples
them to a common rate with polyphase filtering. The waveform inspector and descriptive channel
statistics retain the source waveform and original sample rate.

Train/validation/test subject overlap is rejected before embedding. Generic catalogs use their
explicit splits; PTB-XL challenge manifests are not accepted for generic JSONL data. Signal
bytes and catalog metadata are fingerprinted, and changes to a selected signal after cataloging
are rejected during loading. Only selected training waveforms are decoded.

## Sampling, labels and model controls

`--sample N` and `--sample-fraction F` are mutually exclusive. The default is 512 recordings;
fractions must be in `(0, 1]`. Sampling is uniform without replacement after stable record-ID
ordering, controlled by `--seed`. It samples recordings, not questions or patients. The saved
analysis records the exact sampled IDs and full selected-population counts.

PTB-XL labels use three separate namespaces:

| Namespace | Example | Meaning |
|---|---|---|
| `superclass:` | `superclass:MI` | Existing diagnostic superclass |
| `scp:` | `scp:AFIB` | All stored SCP codes, including diagnostic, rhythm and form codes |
| `artifact:` | `artifact:static_noise` | Existing signal-quality flag |

SCP likelihood values remain in metadata. Human descriptions and code categories come from
`scp_statements.csv` when found beside the local raw cache; `--scp-statements PATH` supplies it
explicitly. Available raw PTB-XL metadata adds age, sex, device, report and other descriptive
slices. The catalog retains the existing adapter's eligibility rule: PTB-XL recordings need at
least one diagnostic superclass; ECG-QA additionally requires a joined question. Rhythm-only
recordings omitted by that adapter are therefore outside this population.

Signal controls are `--backend chronos|opentslm`, `--model`, `--revision`, `--checkpoint`,
`--checkpoint-provenance`, `--device`, `--batch-size`, `--target-sample-rate`, and
`--context-length`. The context length controls chunk size, in samples, within the model's
native limit; OpenTSLM also requires a multiple of its patch size. Shorter contexts still cover
the complete recording through additional chunks. Use a new output directory when changing a model or configuration.
The built-in model revisions are pinned; for other model IDs, provide an explicit revision
for repeatable fresh runs. The default OpenTSLM checkpoint does not need a separate declaration.

Text embeddings default to pinned `sentence-transformers/all-MiniLM-L6-v2`. Configure
`--text-model`, `--text-revision` and `--text-fields FIELD ...`; default fields are
`question answer text`. Fields may also name tuple metadata such as `clinical_context`, or
the separate `rationale` field. Non-empty selected fields are joined for each tuple; identical
field values within that tuple are included once. `--no-text` retains tuples for inspection
while disabling sentence embedding and semantic novelty. Choosing fields matters: a common
clinical-context paragraph or rationale boilerplate can dominate semantic similarity.

## What the metrics measure

Both signal backends embed leads independently, pool across time and concatenate lead vectors
in a fixed channel order. Long signals are processed in non-overlapping chunks with
sample-weighted aggregation, so the representation covers the entire recording. There is no
cross-channel attention in this aggregation. OpenTSLM uses per-recording, per-lead z-scores;
Chronos applies its own tokenizer scaling. The saved provenance records these choices.

K-means fits **L2-normalized original signal embeddings**, without using labels. Labels describe
the resulting clusters; they do not influence cluster fitting. `--clusters` sets the requested
cluster count, capped by the number of distinct embeddings. PCA is always available; UMAP is
also computed for `--projection umap`, with `--neighbors` controlling its neighborhood size.
The 2D coordinates are display views. Cluster distances are never measured on that projection.

| Evidence | Definition and interpretation |
|---|---|
| Label counts | Recording counts in the entire selected population and in the embedded sample, reported separately; patient counts are separate too. |
| Relative scarcity | `1 - label_count / largest_label_count`; inspect comparable label namespaces when interpreting it. |
| Target deficit | Shortfall against `--target-count` or `ceil(--target-fraction × population_records)`, taking the larger target when both are supplied. |
| Within-cluster distance | Mean Euclidean distance to the cluster centroid in normalized signal-embedding space; additional spread statistics are saved. |
| Between-cluster distance | Euclidean distance between centroids; the nearest other cluster is reported. |
| Signal novelty | Mean distance to other sampled recording embeddings; a larger neighborhood radius means lower local density. |
| Silhouette | Separation relative to neighboring clusters; unavailable for degenerate cases and explicitly subsampled for large analyses. |
| Label purity | Fraction of cluster members having the most common complete label set. |
| Label entropy | Mean binary presence entropy over labels present in the embedded sample. |
| Sentence novelty | Mean cosine distance to neighboring distinct, non-empty embedded text strings; repeated strings do not become independent neighbors. |
| Recording text novelty | Maximum sentence novelty among that recording's tuples, with a mean also saved. |

Text novelty is semantic isolation in the chosen sentence model, not simply the count of an
answer or sentence. A sentence far from other distinct strings receives a larger score.
Duplicate-string counts remain available so researchers can distinguish repeated content
from sparse semantic regions.

Population label counts are exact for the selected source; cluster geometry, novelty and
semantic coverage describe only the embedded sample. Small uniform samples can miss rare
labels, and patients with several recordings can contribute several points. Thresholds and
cluster IDs are specific to the model, sample and configuration. Target fractions refer to
the population **before generation**. Labels overlap, so adding per-label deficits does not
give a valid total number of new recordings automatically.

## Explore and export

Each analysis directory contains:

- `index.html`: a standalone interactive report with its data embedded, requiring no CDN.
- `analysis.json`: cluster metrics, recording summaries, tuples, source/model provenance and
  a content-derived `analysis_id`.
- `embeddings.npz`: signal embeddings with `record_ids`; when enabled, text embeddings with
  `text_ids` and `text_record_ids`. Arrays load with `allow_pickle=False`.

Open `index.html` directly, pass `--serve` during analysis, or start the local server later:

```sh
uv run python -m syntide.explore serve runs/explore/ecgqa-sp-512 --port 8765
```

The server binds only to `127.0.0.1`. On a remote CUDA host, use `--no-open` and an SSH tunnel
to view it locally, or copy the standalone HTML to your Mac. The report includes waveform
previews and source text, so treat it as a data artifact when sharing it.

Filter by label, cluster, metadata values or numeric ranges, text search, signal or sentence novelty percentile, cluster
size, within-cluster distance, and nearest-cluster separation. Click a point to inspect all
channels or a single channel, channel means/std/RMS/ranges, recording duration and attached
tuples. Drag a rectangle to select a region, Shift-click individual points, or select every
visible recording. The selected set persists while filters change.

`--waveform-points 256` is the default preview limit per channel; the maximum is 1000. Statistics
still use the full waveform. Uniformly sampled previews can omit short peaks; they are an
inspection aid rather than a replacement for the original signal. Each channel has its own
amplitude axis; when comparing recordings, both traces share that channel's axis.

Choose one or more generation methods and a positive integer count for each, then export the
generation plan. For example, 100 augmentation, 50 editing and 25 diffusion requests means
175 requested outputs in total, not 175 per selected parent. Parameter JSON expresses intended
controls; it is not validated against a generator's configuration schema at this stage.
The plan includes selected recording/text IDs, labels, clusters, filter criteria, source
fingerprints, embedding provenance, a generation seed, method counts and its own content hash.

A selection can also be finalized through the CLI:

```sh
uv run python -m syntide.explore plan runs/explore/ecgqa-sp-512 \
  --selection syntide-selection.json \
  --output runs/explore/ecgqa-generation-plan.json
```

Example selection structure:

```json
{
  "record_ids": ["ptbxl-1"],
  "seed": 42,
  "requests": [
    {"method": "augmentation", "count": 100, "parameters": {}},
    {"method": "editing", "count": 50, "parameters": {}},
    {"method": "diffusion", "generator": "diffusets@1", "modality": "ecg", "count": 25,
     "parameters": {"num_inference_steps": 1000}}
  ],
  "criteria": {"reason": "researcher-reviewed sparse signal and sentence region"}
}
```

Replace the example record ID with one in the saved analysis. Optional `text_ids` restricts
which attached tuples are targeted; otherwise all tuples on selected recordings are included.
The CLI verifies the analysis checksum and checks selection membership. Static-browser plan
export requires Web Crypto; if unavailable, use the local server or export a selection and
finalize it with the CLI.

Augmentation and editing reference existing generator identifiers but still require recipe
configuration, calibration and validation. Full generation defaults to DiffuSETS for ECG;
the UI also accepts another versioned generator ID and modality. The separate
`python -m syntide.generate.full` command resolves checkpoints and explicit conditioning and
executes a selected diffusion request. See [full generation](full_generation.md) for the
configuration and plugin contract. Counts do not determine exact
parent-to-output assignments or guarantee accepted outputs. The exported schema is
`syntide.generation-plan@1`, with status
`intent_only_requires_generator_configuration_and_label_review`.

In particular, noise edits can invalidate noise answers; gain changes can affect voltage
criteria; a rationale may contradict an edited waveform. Review label, answer and rationale
effects for every selected method, preserve the existing calibration/parent/holdout gates,
and record accepted/rejected recipes in a generation ledger. The proposed ECG-QA label-effect
table remains subject to its experiment's review process. Selecting a sparse cluster does
not establish that an augmentation creates a new diagnosis or a valid new QA answer.

## Compare real and synthetic signals

Create one JSONL catalog containing the real reference recordings and generated recordings,
then analyze it with `--dataset jsonl --input comparison-catalog.jsonl`. Preserve unique
record IDs, explicit training splits, channel names, units, sample rates and provenance.
When moving entries into a new catalog, resolve each `signal_path` against its original
catalog directory and rewrite it as an absolute path or a path relative to the new catalog.
Concatenating files without adjusting their relative paths can point at the wrong signals.

The full-generation runner marks its records with `metadata.synthetic: true`; the viewer
recognizes this directly. Custom catalogs can use `metadata.origin: "real"` or `"synthetic"`.
Use `metadata.display_name` for a short readable name. To pair a synthetic recording with a
reference, set `metadata.source_record_id` to that real recording's ID. The viewer also reads
the runner's `metadata.generation.source_record_ids` and chooses a matching reference in the
analysis. References must be included in the embedded sample to be available for comparison.
Use `--sample-fraction 1` when you want every catalog entry in the view.

With both origins present, the map defaults to **Real / synthetic** coloring. Select a synthetic
point to inspect its paired reference, or choose any recording in **Compare with recording**.
Choose one channel or all channels. Traces share actual time in seconds and amplitude in the
declared units within each channel; there is no amplitude normalization or beat alignment.
Different units prevent overlay, and missing channels are identified. Channel statistics
describe the selected recording. Intended diagnoses appear as **Requested labels (unverified)**,
separately from observed labels.

Embedding a mixed catalog requires matching channel-name sets and sample rates, or an explicit
`--target-sample-rate` for resampling. For PTB-XL comparisons, full generation can directly
request 100 Hz output as described in the [DiffuSETS guide](full_generation.md#signal-format-and-output-artifacts).
Use `--waveform-points 1000` to inspect all samples of a 10-second, 100 Hz recording.
A new joint analysis refits its clusters and projection; cluster numbers should not be compared
directly with a previous discovery analysis.

## Reuse and reproducibility

Repeat the same analyze command with `--reuse` to reopen its frozen analysis without rerunning
the model. A different configuration or source fingerprint requires a new output directory;
existing analysis and plan files are not silently replaced with different contents.

Provenance covers the selected scope, sample seed and IDs, source and metadata hashes, model
revisions/checkpoint hashes, preprocessing, device and relevant package versions. Seeds make
sampling and configured analysis repeatable; floating-point model results can still differ
between devices or library versions. The saved analysis preserves the actual measured results
used to choose a generation plan.
