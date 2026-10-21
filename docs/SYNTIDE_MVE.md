# SYNTIDE — MVE review and modular engine design

Oct 21, 2026 · @seb

## Verdict

Approve the core design, but do not run the 72-run grid as written. Three issues can produce a positive result that is not about synthetic data: the oracle mixes more data with better coverage, the 4,000-step budget means 189 passes per record at 10% scarcity, and standard augmentation and untargeted editing are not separated by a stated difference.

The v2 protocol below keeps the question, the dataset and the run count (72). It adds a go/no-go gate, a coverage-only control at matched N, an electrode-space editor, and recipe-based generation. It also makes the engine modular, so PTB-XL becomes the first plugin set rather than the design.

| Decision | What |
| --- | --- |
| Keep | PTB-XL folds 1–8 / 9 / 10, natural artifact test slice as primary, label-preserving editing, matched presentations, paired seeds, patient bootstrap, falsification rules |
| Change | Oracle → matched-N natural-mix control; arms defined by two factors (transform family × sampling policy); step budget set per regime by pilot; 8,192-item cache → recipe stream |
| Add | Gap gate before the grid, electrode-space artifact model, clean-slice non-inferiority, fixed-sequence secondary tests, blinded gap-discovery check, held-out real-noise stress test |
| Drop | Edited-only arm: with editing, train-on-synthetic is just heavy augmentation of real parents |

## What to keep

Most of the MVE is already the right experiment; these choices should survive unchanged.

- **Dataset and split.** Official patient-disjoint folds, 513 artifact-bearing test ECGs from 477 patients, and a target slice defined from metadata before training.
- **Natural test slice as primary.** Evaluating on naturally artifact-bearing ECGs, never on synthetically corrupted test signals, is what makes a positive result credible.
- **Editing over generation.** Real parents keep authentic disease morphology, so labels survive as long as edits stay away from QRS, ST and T.
- **Pairing.** Same parent schedule, labels, output count and RNG across transform arms; only the parameter policy differs between targeted and untargeted.
- **Calibration discipline.** Artifact distributions fit only on retained records (37 or 93); withheld records are oracle-only.
- **Pre-registration.** Primary outcome frozen before test, patient-clustered bootstrap, every seed shown, explicit "distrust if" list.
- **TimeNet split caveat.** Confirmed in the docs: splits live on Tasks and one Record can serve several splits, so subject separation must be enforced outside the schema.

## Challenges, ranked

The first three can manufacture a positive result; the rest weaken what any result means. Numbers below are computed from the MVE's own manifest counts.

| # | Issue | Why it matters | Fix in v2 |
| --- | --- | --- | --- |
| 1 | Oracle mixes quantity with coverage | Oracle restores 325 / 835 records, about 24% more real data. Its gain cannot be credited to artifact coverage, so it cannot define headroom. | **Matched-N natural-mix control:** from the same pool, drop 325 / 835 records at random instead of artifact ones. Same N, natural prevalence: \~292 / 745 artifact records vs 37 / 93. |
| 2 | Budget turns augmentation into a regularizer | 256k presentations = 189 passes per record at 10%, 76 at 25%. Real-only at its final step is likely overfit, so any transform arm "wins" and the real-only and oracle references are distorted. | Set steps per regime from a pilot where standard augmentation plateaus on validation. Primary checkpoint = best overall-validation macro-AUROC (every 250 steps, same rule for all arms); final checkpoint as sensitivity. |
| 3 | Standard aug vs untargeted editing is undefined | If standard aug already adds Gaussian noise and sinusoidal wander, the two arms share a transform family. A null difference is then uninterpretable. | Define arms by two factors: **transform family** (generic per-lead vs electrode-space artifact model) × **sampling policy** (coverage-matched vs gap-targeted). |
| 4 | Per-lead noise is physically wrong | Artifacts arise at electrodes. Limb leads are differences of RA/LA/LL, and V1–V6 reference the Wilson terminal, so a left-arm artifact shows in every lead except II. Independent per-lead noise is easy to detect and unlike natural artifacts. | Inject artifacts on 9 electrodes (RA, LA, LL, V1–V6) and project to 12 leads with the fixed Einthoven/Goldberger/Wilson matrix. |
| 5 | No go/no-go on the gap | "Failed challenge construction" is only discovered after \~10 h of grid. | **Gap gate** before the grid: natural-mix minus depleted ≥ 0.02 macro-AUROC on the validation artifact slice (3 seeds, 25%). These runs are reused as grid seeds. |
| 6 | Threshold sits at the detection limit | Macro-AUROC MDE on 513 ECGs ≈ 1.6–2.8 points (inter-class correlation 0–0.5); 95% CI half-width ≈ 1.1–1.9. A true +0.01 effect will mostly read "inconclusive". | Estimate power from the gate's validation bootstrap and say so before test. Add clean-slice non-inferiority (margin −0.01), fixed-sequence secondary tests, and log-loss as a lower-variance secondary. |
| 7 | Edits are reused \~16× | 128k synthetic slots over 8,192 cached edits = 15.6 presentations per edit. Diversity collapses toward memorizing specific edits. | **Recipe stream:** store (parent, generator version, controls, seed) and regenerate on the fly. \~128k distinct recipes per arm, same for standard aug. |
| 8 | Targeting is told the answer | The gap was planted by the experimenter, so the engine's "decide what to generate" step is never tested. | **Blinded gap-discovery check** (report-only): the gap detector sees retained labels plus unlabeled signals from unused training subjects and must rank the artifact region in its top 3. It never feeds the primary arms. |
| 9 | One scarcity pool | 1,482 / 3,705 subjects are a single draw; seeds do not capture pool-to-pool variance. | State that results are conditional on the pool; next week, repeat over 5 pool draws. |
| 10 | Cross-dataset leakage via TimeNet | TimeNet's ECG-QA CoT is built on PTB-XL ECGs. Once SYNTIDE exports PTB-XL tasks beside it, one patient can sit in train in one dataset and test in another. | Global subject namespace (`ptb-xl:<patient_id>`); the leakage guard checks every dataset in a training mix, not one at a time. |
| 11 | Edited-only arm answers nothing new | With editing, every input has a real parent, so "synthetic-only" is heavy augmentation. | Drop it; its compute pays for the natural-mix control. |

## Revised MVE v2

Same question, same 72 runs: when artifact ECGs are scarce in training, does targeted, physically grounded editing beat standard augmentation on naturally artifact-bearing ECGs from unseen patients?

### Gates (each must pass before the next step)

1. **Leakage gate.** Assert subject-disjointness across train pool, withheld set, unlabeled pool, validation and test, and that every recipe's parent is in the train pool. Fail = stop.
2. **Model gate.** On the 25% pool, longer training and a 2× wider model change validation macro-AUROC by < 0.01. Fail = fix the model, not the data.
3. **Gap gate.** Natural-mix control minus depleted real-only ≥ 0.02 on the validation artifact slice (3 seeds). Fail = retain 0% artifact records and retest once, else report a construction failure and stop.
4. **Generator gate.** For every control, realized vs requested value correlates ≥ 0.9 on 1,000 recipes; all outputs finite and within the mV envelope. Two-sample and label audits are reported, not used to filter.
5. **Freeze.** Commit config hashes, manifests and the decision rules. Then evaluate test once.

### Arms

| Arm | Transform family | Sampling policy | Question it answers |
| --- | --- | --- | --- |
| A0 Real-only | none (parents repeat in transform slots) | — | Floor |
| A1 Standard aug | Generic per-lead: gain, shift, Gaussian noise, sinusoidal wander | Fixed ranges, same severity envelope | What common practice achieves |
| A2 Editor, coverage-matched | Electrode-space artifact model | Match retained descriptor distribution | Does realism help without targeting? |
| **A3 Editor, targeted** | Electrode-space artifact model | Up-weight low-density regions inside the prior envelope | **Primary** |
| C1 Natural-mix control | none | — | Headroom from real coverage at the same N |
| C2 Full oracle | none | — | Reference with more real data |

6 arms × 6 seeds × 2 regimes = 72 runs. Gate-3 runs are the first three seeds of A0 and C1 at 25%, so they cost nothing extra.

A1–A3 fill 50% of slots with transformed parents from the same schedule and paired RNG. Steps per regime come from the model-gate pilot and are identical across arms. Keep NSTDB real noise (bw, ma, em) out of training so it stays an independent stress test.

### Statistics and decision rules

- **Primary estimand:** A3 − A1, macro-AUROC, artifact test slice, 25% regime, mean over seeds. 95% CI from a crossed bootstrap over patients and seeds.
- **Benefit:** CI lower bound > 0 **and** clean-slice non-inferiority (A3 − A1 > −0.01).
- **Useful negative:** CI upper bound < +0.02.
- **Inconclusive:** anything else; report it as such.
- **Fixed sequence at α = 0.05, only if the previous step passes:** A3 − A2 (targeting) → A2 − A1 (realism) → the same chain at 10%.
- **Gap closed** = (A3 − A0) / (C1 − A0), reported only when the gap gate passed.
- **Diagnostics, never evidence:** clean test ECGs with NSTDB noise added at 6, 12 and 18 dB; log-loss; per-class AUROC.

### The one results table

| Arm | Artifact slice, 25% | Clean slice, 25% | Δ vs A1 \[95% CI\] | Gap closed | Artifact slice, 10% |
| --- | --- | --- | --- | --- | --- |
| A0 Real-only |  |  |  |  |  |
| A1 Standard aug |  |  | reference |  |  |
| A2 Editor, matched |  |  |  |  |  |
| A3 Editor, targeted |  |  | **primary** |  |  |
| C1 Natural-mix |  |  |  | 100% by definition |  |
| C2 Full oracle |  |  |  |  |  |

## SYNTIDE architecture

SYNTIDE owns decisions — what to generate, what to keep, whether it helped; TimeNet owns data — ingest, format, immutable versions, task schema. The split follows TimeNet's own scope: it states that training, inference, model definitions and evaluation metrics are out of scope, and its benchmark suite is still planned.

&#91;embedded content: SYNTIDE control loop · 5 stages, one ledger, TimeNet as data plane\]

Every stage writes to the ledger; only the exporter writes to TimeNet, so real data stays read-only.

The full component diagram, with gates, plugin boundaries and how one recipe becomes TimeNet tasks, is in [SYNTIDE Architecture](https://claude.ai/artifact/UnumWRg4fYoTCVS4EcRN9r).

A dataset or modality is a set of five plugins (marked ●). Everything else is shared and never imports modality code.

| Component | Does | Plugin? | PTB-XL implementation |
| --- | --- | --- | --- |
| DatasetAdapter | Loads a TimeNet dataset id into samples with labels, `subject_uid` and metadata | ● | New `physionet/ptb-xl` connector on `BasePhysioNetConnector` (only ECG-QA CoT is in the catalog) |
| LeakageGuard | Asserts subject-disjointness across every manifest and every dataset in a mix | shared | `ptb-xl:<patient_id>` namespace; also checks ECG-QA CoT |
| ChallengeSpec | Declares a scarcity pool and a depletion rule; emits retained / withheld manifests | ● (config) | Artifact flags, retain 10%; 10% and 25% pools |
| DescriptorExtractor | Maps a signal to named, unit-bearing descriptors: the coverage space | ● | Wander power < 0.7 Hz, high-frequency residual after beat template, burst envelope, per-electrode attribution |
| GapDetector | Compares labeled coverage with a reference (prior envelope or unlabeled pool) plus validation error slices; emits a target distribution | shared | Density ratio + slice deficits |
| Generator | Turns (parent, controls, seed) into a signal plus the annotations it caused; declares controls and label effects | ● | Electrode-space artifact editor v1 |
| ControlPlanner | Inverts generator calibration: target descriptors → control samples | shared | Quantile map fit on retained records |
| Validators + AcceptancePolicy | Typed, versioned checks; blocking or report-only | shared + ● | Bounds, realized ≈ requested, duplicates and nearest neighbours, two-sample, label audit |
| Ledger | Append-only recipe store with every result and decision | shared | DuckDB / Parquet |
| Mixer + Harness | Budget-matched streams, arms from config, crossed bootstrap, decision rules | shared | 1M ResNet, MPS |
| TimeNetExporter | The only module that imports `timenet`; publishes accepted recipes as a versioned dataset | shared | Classification, editing, localization, QA tasks |

**Coverage lives in descriptor space, not control space.** The gap detector never sees generator knobs, so an editor, a simulator and a learned model can be compared on the same target.

**Every generator declares a label-effect contract**, per task family: it `preserves`, `determines` or `invalidates` that label. The artifact editor preserves diagnosis, determines artifact type, leads and timing, and invalidates signal-quality labels. The exporter emits only tasks whose label is preserved or determined.

```python
LabelEffect = Literal["preserves", "determines", "invalidates"]

@dataclass(frozen=True)
class Control:                      # one generator knob
    name: str; unit: str; low: float; high: float

@dataclass(frozen=True)
class Recipe:                       # the unit of provenance
    parent_id: str | None; generator: str; version: str
    controls: Mapping[str, float]; seed: int; gap_report_id: str

class DatasetAdapter(Protocol):
    timenet_id: str
    def samples(self, split: Split) -> Iterator[Sample]: ...
    def subject_uid(self, record_id: str) -> str: ...

class DescriptorExtractor(Protocol):
    names: tuple[str, ...]
    def __call__(self, x: np.ndarray, fs_hz: float) -> np.ndarray: ...

class GapDetector(Protocol):
    def report(self, labeled: np.ndarray, reference: np.ndarray,
               val_slices: Mapping[str, float]) -> GapReport: ...

class Generator(Protocol):
    name: str; version: str
    controls: tuple[Control, ...]
    label_effect: Mapping[str, LabelEffect]   # task family -> effect
    def __call__(self, parent: Sample | None, r: Recipe) -> Candidate: ...

class Validator(Protocol):
    name: str; version: str; blocking: bool
    def __call__(self, c: Candidate) -> ValidationResult: ...
```

**Expert review is spent on calibration, not volume.** A cardiologist reviews a stratified sample once per generator version, concentrated at the severity extremes; their verdicts set the severity envelope. After that, review goes to label-audit disagreements and to gap reports before a new generator is built. Per-example review does not scale and is not needed while edits stay label-preserving.

## Provenance and TimeNet mapping

Store recipes, not tensors: every synthetic example is reproducible from its parent, generator version, controls and seed, and its ID is a hash of those. The ledger row below lives in SYNTIDE; TimeNet's manifest records only the build environment and file checksums, so it cannot carry per-example history.

| Ledger field | Example | Why it exists |
| --- | --- | --- |
| `recipe_id` | sha256 of the fields below | Dedup; stable reference from TimeNet |
| `parent_record_id`, `parent_subject_uid` | `ptb-xl:ecg_01234`, `ptb-xl:p_18792` | Leakage checks; nearest-neighbour exclusion |
| `source_version` | `physionet/ptb-xl@1.0.3` + manifest sha256 | Rebuild from the exact source |
| `challenge_id`, `manifest_hash` | `artifact-retain10/25pct` | Which scarcity world it belongs to |
| `generator`, `version`, `code_hash` | `electrode_artifact@1.0.0` | Rollback and A/B of generator versions |
| `controls_requested` / `controls_realized` | wander 0.4 / 0.38 mV | Alignment check; honest instruction text |
| `seed`, `rng_stream` | 7, `A3/s3/step1200` | Bit-exact regeneration; arm pairing |
| `gap_report_id` | `gap-2026-10-22-a` | Why it was generated |
| `validations[]` | name, version, value, pass | Audit; re-run when a validator changes |
| `decision`, `reason` | accepted | Rejection-rate monitoring |
| `label_effect` | diagnosis: preserves | Which tasks may be exported |
| `review` | unreviewed / approved / rejected + reviewer | Where experts spent time |
| `exported_task_ids` | TimeNet task ids | Trace a training example back to its recipe |

The exporter maps each accepted recipe onto TimeNet's data model:

- **A new Record with its own Signals.** TimeNet gives each Signal one owning Source, so an edit never reuses the parent's Signal.
- **Record-level provenance as span-less Annotations.** Records have no metadata slot, so `subject_uid`, `syntide.recipe_id`, `syntide.parent_record` and `syntide.generator` are static annotations.
- **Each injected artifact as a TimeInterval Annotation** scoped to the leads it touched via `time_series_ids`. This is free, verifiable supervision.
- **Tasks from one recipe:** a ClassificationTask inheriting the parent's diagnosis via `from_tasks`; a TSEditingTask (parent → edit, prompt rendered from realized controls); a TemporalLocalizationTask (find the burst); AnswerTasks (which electrode, what frequency).
- **Split = the parent's split.** LeakageGuard, not the schema, enforces the subject rule.
- **A separate dataset id**, e.g. `aionic/ptb-xl-syntide-artifacts`, tagged `synthetic` and `generated-supervision` — the tag TimeNet's SensorCaps entry already uses for model-written captions. Each publish is an immutable version with checksummed files.

Example prompt, rendered from realized values: "Add 0.38 mV baseline wander at 0.3 Hz on the left-arm electrode and a 1.2 s motion burst at 6.0 s." The paired localization target spans every lead except II, because II is the only lead that does not use the left arm.

Pin the TimeNet version. Its Tasks page shows `inputs` / `targets` / `metadata`, while its Types page shows `record_ids` / `target` / `source_record_id`; keeping `timenet` imports inside the exporter makes that drift a one-file change.

## Extending to other datasets and modalities

A new dataset is five plugins and one config file; the gap detector, ledger, mixer, harness, statistics and exporter do not change. The table shows the plugin set for the MVE and three next candidates.

| Dataset (TimeNet id) | Task type | Challenge spec | Descriptors | Generator → label effect |
| --- | --- | --- | --- | --- |
| PTB-XL (`physionet/ptb-xl`, to build) | 5-class multilabel ClassificationTask | Artifact records, retain 10% | Electrode-space artifact descriptors | Electrode artifact editor → preserves diagnosis |
| PPG-DaLiA (to build) | ScalarPredictionTask, HR in bpm | High-HR windows depleted | Realized HR, respiratory rate, accelerometer motion energy | PPG simulator → determines HR |
| Sleep-EDFx (`physionet/sleep-edfx`, in catalog) | Per-epoch stage ClassificationTask | N1 epochs or noisy EEG scarce | Band powers, EMG and electrode-pop artifacts | EEG artifact editor → preserves stage within a bounded severity |
| ECG-QA CoT (`physionet/ecg-qa-cot`, in catalog) | AnswerTask with rationale | Under-represented question types | Question-type and answer coverage | Templated QA from editor annotations → determines answer |

Adding a dataset:

1. Point the adapter at a TimeNet id, or write a connector (`download` + `convert`) and build it.
2. Define `subject_uid` and register the dataset's lineage (e.g. ECG-QA CoT → PTB-XL) with LeakageGuard.
3. Write the ChallengeSpec in config.
4. Implement a DescriptorExtractor and its unit tests.
5. Implement or wrap a Generator with its controls and label-effect contract.
6. Add modality validators (e.g. physiological HR range); generic ones come free.

The PTB-XL experiment, as config:

```yaml
experiment: ptbxl-artifact-v2
dataset:
  adapter: timenet:physionet/ptb-xl@1.0.3
  subject_uid: "ptb-xl:{patient_id}"
  splits: {train: [1,2,3,4,5,6,7,8], val: [9], test: [10]}
challenge:
  pool_fraction: [0.10, 0.25]
  slice: static_noise | burst_noise | baseline_drift | electrodes_problems
  retain: 0.10
descriptors: ecg.electrode_artifacts@1
gap: {detector: density_ratio@1, reference: prior_envelope}
generators:
  generic: per_lead_aug@1                # A1
  editor: ecg.electrode_artifact@1       # A2, A3
validators: [bounds@1, alignment@1, dedup_nn@1, two_sample@1:report, label_audit@1:report]
arms:
  A0: {family: none}
  A1: {family: generic, policy: fixed}
  A2: {family: editor, policy: coverage_matched}
  A3: {family: editor, policy: gap_targeted}
  C1: {data: natural_mix_matched_n}
  C2: {data: full_oracle}
training: {model: resnet1d_1m, batch: 64, steps: from_pilot, transform_fraction: 0.5, seeds: 6}
evaluation:
  primary: {metric: macro_auroc, slice: artifact, regime: 0.25, contrast: [A3, A1]}
  non_inferiority: {slice: clean, margin: -0.01}
  sequence: [[A3, A2], [A2, A1]]
export:
  timenet_id: aionic/ptb-xl-syntide-artifacts
  tasks: [classification, ts_editing, temporal_localization, answer]
```

The PPG-DaLiA version changes only `dataset`, `challenge`, `descriptors`, `generators` and the primary metric (subject-macro MAE on high-HR windows).

## Plan, evidence that would change the conclusion, next week

The three days keep the original compute envelope (≈ 8–13 h on the M1 Max); the gates move the expensive decisions to Day 1.

| Day | Build | Gate passed by end of day |
| --- | --- | --- |
| 1 | Adapter, manifests, LeakageGuard; electrode-space editor and descriptors; model and budget pilot; first 3 seeds of A0 and C1 at 25% | Leakage, model, gap; protocol frozen |
| 2 | Recipe streams, validators, ledger; remaining grid runs; blinded gap-discovery check; export 1,000 recipes to TimeNet | Generator gate; grid complete on validation only |
| 3 | Test evaluated once; crossed bootstrap; the one results table; diagram and slides | Decision rule applied as written |

What would change the conclusion:

- **A positive A3 − A1 weakens** if it disappears at the final checkpoint, if A2 ≈ A3 (the gain is realism, not targeting), or if the clean slice drops by more than 0.01.
- **A null A3 − A1 becomes informative** only if the gap gate passed clearly: then the model already learns this invariance from 37–93 examples, and artifacts are the wrong gap to target.
- **A failed gap gate is itself the finding:** coarse PTB-XL artifact flags do not mark a data-limited slice for this model, and the engine should move to a gap with measurable headroom.
- **A strong NSTDB stress-test gain with no natural-slice gain** means the editor teaches synthetic-noise robustness that does not transfer to real acquisition artifacts.

With one more week and a cluster:

1. 20 seeds × 5 scarcity-pool draws, to put pool-to-pool variance in the interval.
2. Dose-response: transformed fraction 0.25 / 0.5 / 0.75 and recipe count, for a synthetic-data scaling curve.
3. A learned, text-conditioned edit model against the mechanistic editor, under the same harness and descriptors.
4. External natural-artifact test set from another hospital and device, to separate artifact robustness from site shift.
5. Tier B on CUDA: fine-tune an OpenTSLM model on ECG-QA CoT plus exported localization and QA tasks, with the ECG-QA patient overlap removed.
6. Run the PPG-DaLiA plugin set unchanged through the engine, to show the modularity is real.

## Sources

TimeNet pages opened on 21 October 2026; counts and the MVE design come from the attached MVE write-up and the Aionic brief.

- [TimeNet overview and scope](https://docs.timenet.ai)
- [TimeNet Tasks](https://docs.timenet.ai/data-model/tasks.html) and [Types](https://docs.timenet.ai/types.html)
- [TimeNet Records](https://docs.timenet.ai/data-model/records.html), [Annotations](https://docs.timenet.ai/data-model/annotations.html), [Time series](https://docs.timenet.ai/data-model/time-series.html)
- [TimeNet Connectors](https://docs.timenet.ai/connectors.html), [Manifest](https://docs.timenet.ai/manifest.html), [Splitting](https://docs.timenet.ai/api/splitting.html), [Provenance](https://docs.timenet.ai/api/provenance.html)
- [TimeNet catalog](https://docs.timenet.ai/catalog/datasets.html) and [Benchmarks (planned)](https://docs.timenet.ai/catalog/benchmarks.html)
- [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/1.0.0/)
- [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
