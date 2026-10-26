# Evidence-grounded ECG-QA-CoT generation

`syntide.generate.qa` turns real, edited and generated ECGs into question–answer–rationale tasks whose
answers come from verified evidence. It publishes immutable training corpora and evaluation suites,
plans QA trials through a versioned QA trial seam, and scores them once on frozen held-out suites. It
reuses SYNTIDE's existing boundaries:

- the manifests and the leakage gate;
- the electrode editor;
- the full-generation registry;
- the tracker;
- create-only freezing.

**Status.** The pipeline runs end to end on synthetic fixtures and on the local PTB-XL / ECG-QA-CoT
registry. No QA model has been trained on these corpora yet, so no benchmark improvement is claimed.

**The baseline it compares against.** The RunPod OpenTSLM baseline has one completed seed (Gemma-3-270M
Flamingo). Its best checkpoint scored 39.0% option-likelihood accuracy on the 2,000-question selection
subset. Its seeds 1–2, full validation and test scoring were cancelled.

## Architecture changes, and why

| Change | Where | Why it was needed |
|---|---|---|
| `Evidence`, `InputRef`, `QATask` contracts | `syntide/data/qa.py` (new) | Classification `Samples` has no place for questions, evidence or multi-input lineage. `Samples` is unchanged. |
| `check_lineage`, `check_disjoint_subjects` | `syntide/data/leakage.py` (added functions) | A generated or edited waveform's new id proves nothing. Its *real roots* and their patients must respect the manifest roles. |
| Generation package | `syntide/generate/qa/` (new) | The five capabilities, the corpus writer and the CLI. |
| QA trial seam | `syntide/qa_trial.py` (new) | The classification seam is AUROC- and `(n, 5)`-specific. A separate versioned spec (`syntide.qa-trial@1`) leaves `TrialSpec`, its fields and its ids untouched. |
| QA scoring and frozen evaluation | `syntide/eval/qa_metrics.py`, `syntide/eval/qa_holdout.py` (new) | Answer parsing, numeric tolerance and answerability need their own frozen contract. The existing eval modules are not modified. |
| Offline consumer, OpenTSLM adapter | `trainers/.../qa_consumer.py`, `opentslm_qa.py` (new) | Model code stays outside `syntide`. The consumer exercises the whole contract offline. |
| Configs | `conf/qa/` (new) | Separate from `ptbxl_artifact_*`. Pinned to the baseline's split, selection subset and budget. |

Nothing else changed: the existing modules, configs, explorer UI and plan schema are as they were.

## Contracts

- **`Evidence`** (`syntide.ecg-evidence@1`): one fact about one waveform. It records:
  - the waveform's sha256 and signal spec (channels, rate, units, length);
  - the record and its real roots and patients;
  - attribute, value, units, leads and interval;
  - a `status`: present | absent | unknown | unobservable;
  - a `kind`: measured | annotation | requested_control | hypothesis;
  - confidence and observability;
  - method `name@version`, config sha256, rule versions, and provenance (sidecar sha256, annotation source).

  Its id hashes all of it. Only *measured* or *annotation* evidence with a present or absent status can
  establish an answer.
- **`InputRef`**: one waveform. It holds its bytes' sha256, its spec and its origin (real | transformed
  | generated), its immediate parents, its real roots and their patients, and its lineage: recipe, run,
  resampling, lead aliases, requested controls, generator training cohort and review status.
- **`QATask`** (`syntide.qa-task@1`) holds:
  - family and template version;
  - ordered inputs;
  - the question;
  - the canonical answer type and value, and the options;
  - a short rationale whose claims cite evidence ids;
  - the required evidence and an explicit `answerability`;
  - the student-visible context;
  - generation, verification and lineage records.

  `student_view()` is everything a model may see. The answer, rationale and evidence are privileged.

An unanswerable task answers `insufficient evidence`. That answer is distinct from ECG-QA's `none`.

Committed outputs are TimeNet datasets: one `Record` per waveform, with its evidence in the record
metadata, and one `AnswerTask` per task. A comparison is one task with two ordered inputs. The explorer
catalog (`catalog.jsonl`) keeps one row per waveform and attaches each task once, to its first input.

## The five capabilities

### A. Evidence-grounded QA from real ECGs

Pipeline: source ECG → extractors (cached per exact waveform, settings, annotations and sidecar) →
eligible templates → canonical answers → deterministic rationales → `verify_task` → accepted tasks.

| Extractor | Facts | Limits |
|---|---|---|
| `ecg.acquisition@1` | duration, rate, channels, flat/available leads, robust amplitude | – |
| `ecg.artifact_descriptors@1` | per-lead wander and HF noise, loudest burst, strongest lead (with a margin rule) | The existing `ecg.electrode_artifacts@1` descriptors: physiology leaks into both bands, so they measure the recording and do not diagnose noise. |
| `ecg.beats@1` | heart rate, beat count, RR CV, RR regularity | Pooled 5–15 Hz energy R-peaks, with per-lead agreement as confidence. Fewer than 4 beats, agreement < 0.6 or implausible RR makes the result `unknown`. Regularity has an explicit grey zone. No P/QRS/T delineation at 100 Hz. |
| `ptbxl.annotations@1` | PTB-XL signal-quality flags, diagnostic superclasses | Annotations of the unmodified recording only. A superclass never stands in for morphology. |
| `measurement.sidecar@1` | intervals (QRS, PR, QT, QTc) from a PTB-XL+ style CSV | The file sha256, column map and external extractor version are part of every id. Values describe the original recording only. |

Seed templates cover:

- duration and flat leads;
- wander in a lead, and the strongest wander lead (the strongest-noise-lead template exists, but is off by
  default: its margin rule is almost never met on real 100 Hz PTB-XL, so it would teach a constant
  "insufficient evidence");
- heart rate, beat count, regularity and rate band (with a ±3 bpm boundary rule);
- the annotated noise type;
- annotated superclasses;
- sidecar intervals.

`verify_task` runs six blocking checks. A check that cannot run counts as a failure:

| Check | What it requires |
|---|---|
| `evidence_present@1` | every cited evidence id exists and belongs to an input |
| `evidence_supports_answer@1` | the cited evidence can support this answer |
| `answer_recomputed@1` | rebuilding the template from the evidence gives the same answer |
| `rationale_consistent@1` | every number appears in the cited evidence; no other option is asserted; no `Answer:` line |
| `context_clean@1` | the context has only permitted keys and no generator controls |
| `lineage_resolved@1` | every input names its real roots and their patients |

Clinical validity is not established by these checks. Each task records `clinical_review: not_reviewed`,
or `source_dataset_unreviewed` for original ECG-QA-CoT tasks.

### B. ECG-TSEvol and compositional questions

This follows ChatTS's TSEvol (Xie et al., VLDB 2025). Each step draws from the waveform's verified
evidence pool, and `verify_task` is the evidence-based eliminator that discards failing tasks. The
operators:

- **breadth:** another attribute or lead;
- **depth:** a composed rate and wander question;
- **reasoning:** support or refute a statement (statements come in complementary sets, so neither
  answer is a shortcut);
- **situation:** a stated lead subset (asking about a lead outside it gives an insufficient-evidence
  target), or a supplied rule.

Comparisons over two recordings use seeded, disjoint pairs, never a Cartesian product.

Sampling is keyed by the waveform's sha256. Caps apply per waveform and family. Semantic keys dedupe
tasks, and paraphrases share the key of their semantic task.

Optional teacher rewording:

- **Providers:** `template` (offline) or `anthropic` (opt-in and paid; `claude-opus-5-5` with
  server-side refusal fallbacks).
- **Storage:** every request and response is stored with its hash. `mode: replay` rebuilds the same
  task ids from the store and never calls a provider.
- **Checks:** reworded text goes through the same checks. A teacher never sets an answer.

### C. Acquisition variation with task-aware supervision

The operations are versioned as `ecg.acquisition@1`:

- `resample` (down and back to the source rate);
- `lowpass` and `highpass`;
- `crop` (with its time offset recorded);
- `lead_mask` (lead identities and order are kept);
- `gain`;
- `electrode_artifact` (the existing `ecg.electrode_artifact@1.0.0` editor, unchanged, with its `bounds@1`
  and `alignment@1` validators).

Each operation declares a label effect per attribute group, using the existing `preserves | determines
| invalidates` vocabulary. Every evidence-backed task is then rebuilt from remeasured evidence:

| Rebuilt task | `preserves` | `determines` | `invalidates` |
|---|---|---|---|
| same answer | kept | kept | kept |
| different answer | rejected (invariant) | updated | updated |
| now insufficient | rejected (invariant) | insufficient-evidence task | insufficient-evidence task |

Answerability is recorded separately from the label effect. Annotation evidence is inherited only for
groups the operation preserves, and becomes `unknown` otherwise. Preserving the diagnosis label and
preserving what the waveform shows are separate checks. Original ECG-QA-CoT tasks are never carried onto
an edited waveform.

### D. Controlled waveform generation and pairs

`generated.py` reads `syntide.generate.full` bundles (for example DiffuSETS) and refuses any sample that
fails one of these steps:

1. Check every artifact hash.
2. Resolve each sample's real ancestry (the conditioning `source_record_ids`) against the source role
   *before* reading its waveform.
3. Normalize the `aVR/aVL/aVF` aliases explicitly.
4. Convert 102.4 → 100 Hz only with `allow_resample` (polyphase 125/128, so 1,024 → 1,000 samples, recorded).
5. Remeasure the waveform. It becomes `verified` only for the configured attributes when they are usable
   with confidence ≥ 0.8.

Pending samples never reach an accepted export: `publish` refuses them. Requested conditioning and
requested labels are lineage only and never answers. The DiffuSETS checkpoint still needs its 1,536-d
`text-embedding-ada-002` vectors (see `full_generation.md`).

Controlled variations declare their changes and invariants, and both waveforms are remeasured:

| Variation | Changes | Invariants checked |
|---|---|---|
| `amplitude_up@1` | amplitude (×1.5 gain) | rate, beat count, regularity, duration, lead availability |
| `wander_up@1` | baseline wander (one electrode drift) | rate, beat count, duration, lead availability |
| `diffusets.heart_rate_condition@1` | measured heart rate (same seed, heart-rate conditioning only differs) | duration, lead availability |

Each pair records its id, intended and observed changes, invariant checks and lineage, with
`claim_scope: controlled_signal_comparison`. Pair tasks come in a seeded A/B order, and the answer
follows the measurement.

Time-warp heart-rate changes and diagnosis changes are refused (`UnsupportedVariation`).

### E. Targeted allocation and curriculum

`allocation.py` computes:

- **coverage** over measured slices: family and answer, rate band, lead availability, noise level,
  origin, source dataset, composition and answerability;
- **gap reports** (`qagap-…`) from *development-suite* predictions only. Each slice reports patient
  counts, Jeffreys intervals, and an error rate shrunk toward the overall rate. Slices below `min_n`
  are unreliable.

`make_plan` writes a content-addressed `syntide.qa-generation-plan@1` with per-template quotas. A plan
only builds a **new** corpus: its id is part of the build identity.

`curriculum.py` freezes the curriculum: stages, `until_step` boundaries, compositions or families, target
mode, replay share, seeds and the pools it draws from. A curriculum's sha256 is part of the trial id;
`sequence_sha256` reproduces the exact sequence and `exposure` counts it.

## Corpora: bounded, restartable, immutable

A build processes work units of recordings. Each unit is committed as one chunk (`.partial` file, then
an atomic rename, then `progress.json`). Rerunning the same command resumes; a different configuration
is refused. Nothing exists at the output path until `publish` has re-verified every chunk and written:

- the TimeNet dataset;
- the explorer catalog;
- the ledger of every candidate, with its reason;
- `corpus.json`.

Only then is staging renamed. `corpus_id` hashes the identity, counts, content and file checksums.
TimeNet's own files are verified on load but kept out of the id, because TimeNet rewrites them with new
bytes on every write. Each waveform is stored once.

The counts report patients, root recordings, waveforms by origin, semantic tasks, QA examples,
paraphrases, attempts, accepted and rejected candidates (with reasons), and whitespace token counts.

## Trials, trainers and evaluation

- **Trials.** `python -m syntide.generate.qa plan-trials` writes one `trial.json` per arm, supervision
  and seed. A trial pins its corpus ids, selection suite, curriculum sha256, budget, metric
  (`val/qa_accuracy`, max, earliest tie), parser `strict-answer-line@1`, answerability policy, decoding
  and context policy.
- **Loading.** `load_any_spec` dispatches on `schema_version`, so legacy classification `trial.json`
  files load exactly as before.
- **Preparing.** `prepare_qa` re-verifies the corpora and refuses evaluation corpora as training data
  and any patient shared with the selection suite.
- **Finishing.** `finish` re-parses the saved outputs, recomputes the reported scores and refuses a best
  checkpoint that is not the first argmax.
- **Trainers.**
  - `syntide_trainers.qa_consumer` is the offline contract exerciser (NumPy only; not a model of record).
  - `syntide_trainers.opentslm_qa` reproduces the baseline's `ecgqa-cot-single-joint-media-v1` prompt
    and target byte for byte for original tasks, with documented `-numeric` and `-pair` extensions.

`python -m syntide.eval.qa_holdout` runs the held-out steps:

| Mode | What it does |
|---|---|
| `freeze` | Create-only. Records the rules, the plan sha256, the parser and metric, suite corpus ids and `corpus.json` sha256 (dataset version, eligibility, label mapping, preprocessing), best-checkpoint sha256 and declared generator-cohort overlap. Checks that no suite shares a patient with any training corpus. |
| `score` | Re-verifies everything, claims the suites once (create-only), then scores each run's best checkpoint on every suite. The primary suite also gets the **context-only** (zeroed signals) and **waveform-swap** (same family, other answer) controls on waveform-evidence tasks. |
| `report` | Recomputes every number from saved outputs: accuracy, family-macro accuracy and F1, invalid rate, numeric tolerance and median error, abstention precision and recall, per-slice results, rationale contradictions, per-arm mean ± SD, and paired contrasts with patient-cluster bootstrap intervals. `report-val` reports on the selection suite. |

## Baseline compatibility

Everything is the same as the RunPod baseline:

- **split:** `ecgqa-artifact-50/pool50`, content sha `6c46d78a…`. Main's `EcgQaView` reproduces its source
  fingerprint `ac0dcb42…`, and Gate 1 plus the QA join pass on it;
- **selection subset:** `selection.parquet`, sha `29d7b99e…`;
- **budget:** 15,531 updates × 16 examples, validation every 1,000 updates;
- **selection:** option-likelihood selection and the strict parser;
- **prompt:** `ecgqa-cot-single-joint-media-v1`.

Model and optimizer settings belong to the trainer's config and fingerprint.

The baseline-compatible context policy (`ecgqa-cot-baseline@1`) passes the original `clinical_context`
verbatim. That text states PTB-XL signal-quality flags, so tasks whose answer it reveals are marked
`context_reveals_answer`, and the context-only control measures how much that matters. Derived recordings
keep the format but drop the stale "Signal quality:" sentence.

### Export for the unchanged baseline trainer

`syntide.generate.qa export` augments the original baseline prepared folder. It keeps every original
training row, including its source task id and rationale, as an unchanged prefix. It appends accepted
corpus tasks and any new waveform arrays. The baseline's entire signal array remains an exact prefix,
so selection, validation and test can be copied byte-for-byte with their original signal indices.
The exporter hashes these held-out files but does not parse validation or test labels.

Use the complete original prepared folder (all six data files and `manifest.json`), not just the
selection file copied for corpus construction:

```bash
BASELINE=/absolute/path/to/original/prepared
uv run python -m syntide.generate.qa export --baseline-prepared "$BASELINE" \
  --corpus outputs/qa/paraphrase --output runs/ecgqa-data-paraphrase
uv run python -m syntide.generate.qa export --baseline-prepared "$BASELINE" \
  --corpus outputs/qa/grounded --output runs/ecgqa-data-grounded
uv run python -m syntide.generate.qa export --baseline-prepared "$BASELINE" \
  --corpus outputs/qa/grounded outputs/qa/acquisition --output runs/ecgqa-data-grounded_acq
```

The source fingerprint, split manifest, training role, every root recording and its patient must match
the baseline. Original source tasks supplied as a corpus are checked against the existing training
rows and do not duplicate them. Identical corpus task ids across inputs are included once. The
`qa_export` section of the new checksummed `manifest.json` records corpus ids, exclusions, appended
waveform lineage and template ids. Synthetic rows retain corpus task ids, making their full evidence
and provenance recoverable from the frozen corpus. Training slice annotations (`is_artifact`, fold,
artifact types) describe the first root recording; the evaluation slices remain exactly the baseline's.

The unchanged trainer has fixed input and prompt constraints:

- Two-ECG comparisons are excluded and counted as `multiple_ecg_inputs`.
- Short crops are excluded and counted as `incompatible_signal_length`. The exporter does not pad,
  stretch or repeat a waveform, since that would change what its measured answer describes.
- Other lead orders, rates, units or signals combining multiple root patients are excluded explicitly.
- Numeric answers retain empty options. The prompt therefore has an empty options line.
- Student context must consist of exactly one `clinical_context` string; unrepresentable context is
  an error. Stated lead subsets and interpretation rules already appear in the question itself.

Publication is atomic. Re-running an identical export verifies and reuses it; changed corpora or
baseline files require a new output directory. Each arm gets a new dataset fingerprint. For arms that
add questions on existing ECGs only, `records.json` and `signals.npy` remain byte-identical to baseline.
The baseline trainer, prompt, normalization and selection procedure need no code changes.

## Explicitly unsupported

- P/QRS/T delineation and interval measurement from the waveform at 100 Hz. Intervals come only from a
  checksummed sidecar.
- Morphology from superclass labels.
- Clinical validation of any edited or generated ECG.
- Diagnosis-changing edits.
- Time-warped heart-rate changes.
- Carrying original ECG-QA-CoT answers onto edited waveforms.
- Retrieval sources: none exist in this pipeline.
- Generator-cohort overlap with external benchmarks: recorded as `unknown`, never assumed absent.

## Commands

### Offline (synthetic fixtures, about 30 s)

```bash
uv run --extra explore --extra diffusets pytest tests/qa -q
```

`tests/qa/test_end_to_end.py` runs the whole chain:

1. a synthetic PTB-XL + ECG-QA-CoT registry (built through the production TimeNet connectors);
2. evidence;
3. QA generation and verification;
4. the accepted export, reopened in the explorer;
5. frozen QA trials, trained by the consumer;
6. recomputed validation scores;
7. frozen held-out evaluation with both controls.

### Real data (local registry, baseline split)

Prerequisites:

- Copy `manifests/ecgqa-artifact-50/` from the `runpod-baseline` branch. Its sha256 is pinned in every
  config.
- Copy the baseline's `prepared/selection.parquet` to `outputs/runpod-baseline/prepared/`.

```bash
Q="uv run python -m syntide.generate.qa"
# Corpora (about 15 min each for the 5,778 pool ECGs on one CPU process; restartable).
$Q build --config conf/qa/corpus_real_qa.yaml     --output outputs/qa/real_qa
$Q build --config conf/qa/corpus_paraphrase.yaml  --output outputs/qa/paraphrase
$Q build --config conf/qa/corpus_grounded.yaml    --output outputs/qa/grounded
$Q build --config conf/qa/corpus_acquisition.yaml --output outputs/qa/acquisition
$Q build --config conf/qa/corpus_generated.yaml   --output outputs/qa/generated   # needs generated bundles
# Suites.
$Q build --config conf/qa/suite_selection.yaml    --output outputs/qa/selection
$Q build --config conf/qa/suite_development.yaml  --output outputs/qa/development
$Q build --config conf/qa/suite_test.yaml         --output outputs/qa/test
# A small check first: add --max-records 64 to any build.

$Q plan-trials --protocol conf/qa/protocol_ecgqa_cot_qa50.yaml        # 5 arms x 2 supervision x 3 seeds
uv run python -m syntide_trainers.qa_consumer "runs/ecgqa-cot-qa50/*/*/seed*"   # offline consumer, or the OpenTSLM loop
uv run python -m syntide.eval.qa_holdout report-val runs/ecgqa-cot-qa50 --rules conf/qa/eval_ecgqa_cot_qa50.yaml
uv run python -m syntide.eval.qa_holdout freeze     runs/ecgqa-cot-qa50 --rules conf/qa/eval_ecgqa_cot_qa50.yaml
uv run python -m syntide.eval.qa_holdout score      runs/ecgqa-cot-qa50 --rules conf/qa/eval_ecgqa_cot_qa50.yaml
uv run python -m syntide.eval.qa_holdout report     runs/ecgqa-cot-qa50 --rules conf/qa/eval_ecgqa_cot_qa50.yaml
```

Targeted regeneration from development errors (it writes a new plan and builds a new corpus; nothing
running is changed):

```bash
$Q gap --suite outputs/qa/development --outputs runs/.../qa_development_outputs.parquet \
  --protected outputs/qa/selection outputs/qa/test --output outputs/qa/gap.json
$Q allocate --gap outputs/qa/gap.json --budget 20000 --parent outputs/qa/grounded --output outputs/qa/plan.json
$Q build --config conf/qa/corpus_grounded.yaml --output outputs/qa/grounded-v2 \
  --plan outputs/qa/plan.json --parent outputs/qa/grounded
```

## Measuring validation and external-generalization gains

- **Arms** (`conf/qa/protocol_ecgqa_cot_qa50.yaml`):
  1. original real QA (the baseline's data);
  2. plus paraphrases;
  3. plus grounded expansion;
  4. plus acquisition variation;
  5. plus verified generated waveforms.
- **Supervision:** each arm runs both `rationale` and `answer_only`.
- **Seeds:** three per arm.
- **Fixed across arms:** encoder, LLM, optimizer, decoding, selection suite and the primary budget
  (updates × examples). Compare token exposure from each run's `exposure.json` (whitespace counts, plus
  the trainer's own tokenizer counts).
- **Validation:** `report-val` on the selection suite, as often as wanted.
- **Held-out test**, once, after `freeze`:
  - internal fold-10 suite (original plus grounded questions);
  - optionally an external evaluation-only suite (`suite_external_example.yaml`), with its overlap with
    the generators' training cohorts declared, and `unknown` when not established.
- **Primary contrast:** `grounded_acq_gen/rationale − real_qa/rationale` on the internal test suite, with
  a patient-cluster bootstrap interval over seeds and tasks. Report the other contrasts and the
  supervision contrasts as secondary.
- **Required alongside:** the context-only and waveform-swap controls. A gain that survives with zeroed
  or swapped waveforms is not a waveform gain.
- **Curriculum:** studied separately (`protocol_ecgqa_cot_qa50_curriculum.yaml`) at the same budget.
