# SYNTIDE — three-phase implementation plan

## Context

The Aionic brief (docs/Synthetic_Data_Engine_Task.pdf) asks for a small proof of concept with one central question: **can synthetic data improve performance on held-out real data?** The experiment must use a defensible baseline, isolate the synthetic effect, prevent leakage across generation, training and evaluation, match training budgets, and repeat runs. Total effort is capped at three days, and the work ends in a 45-minute presentation.

SYNTIDE's design (docs/SYNTIDE_Architecture.md, docs/SYNTIDE_MVE.md) answers this on PTB-XL. It edits real training ECGs with a physically grounded artifact generator and compares it with standard augmentation on naturally artifact-bearing test ECGs. The repo has only docs, a hello-world `main.py` and `timenet[cli,torch]==0.11.0` in `.venv`.

The plan keeps the three requested phases:
1. PTB-XL in TimeNet;
2. training with Hydra + structured dataclasses + YAML groups;
3. evaluation with TensorBoard / SummaryWriter.

Each phase has its own package, tests and file outputs, and each phase's outputs are the next phase's inputs.

### What the brief requires that the first draft missed
| Brief requirement | First draft | Added in this revision |
|---|---|---|
| "Generate synthetic examples" and compare with vs without them | Generator deferred; only A1 standard augmentation | **Electrode-space artifact editor** plugged into Phase 2's mixer → arms A2/A3 and the full 72-run grid (this widens the earlier "A0, A1, C1, C2" choice) |
| Decide *what* to generate, not just reproduce the distribution | Missing | Gap report + two sampling policies: **coverage-matched (A2)** vs **gap-targeted (A3)**, both fit on retained records only |
| Evaluate quality, reject unsuitable outputs | Missing | Blocking validators (bounds, alignment), acceptance with retries, **generator gate** (Gate 4), calibration checked beyond correlation (timenet_syntide §2) |
| Plausibility, diversity, semantic alignment, memorization | Missing | Coverage view (natural vs requested vs accepted descriptors), distinct-recipe count per run, label audit (report-only), parents limited to the depleted pool |
| Record complete provenance | Missing | Recipe ledger (`recipes.parquet` per trial). Each recipe id is a sha256 of its parent, generator version, controls and seed, and regenerates bit-exactly |
| Where expert review adds value | Missing | The generator gate writes a stratified review sample (severity extremes) for one-time calibration review |
| Prevent leakage between *generation*, training and evaluation | Training and evaluation only | Calibration may read only `retained_artifact` records; recipe parents must be in `depleted` |
| Explain the result; say what would change it; plan next week | Missing | Phase 3 writes the MVE's "one results table", fixed-sequence findings, gap-closed numerator and denominator, and figures (PNG) for slides |
| Three-day cap, presentation | Missing | Schedule + deliverables section |

Verified facts this plan depends on:
- `timenet.engine.run_pipeline(connector, root, *, cache_dir, keep_cache, force)` builds a connector straight from Python. `timenet-connectors==0.11.0` provides `BasePhysioNetConnector` (direct format-16 reader, header parser). Its ECG-QA CoT connector is a working PTB-XL template (`ptbxl-{ecg_id}` record ids).
- TimeNet 0.11.0 has **no split API**, and `Record.subject_ids` is not stored. Patient and fold are stored by us. `Record.metadata` (JSON) and annotations (bool/int/float/str) round-trip.
- Lead Signals come back **sorted by name**, not in clinical order. The adapter must reorder them.
- All deps resolve on Python 3.14: hydra-core 1.3.7, lightning 2.6.6, tensorboard 2.21, scikit-learn 1.9 (which brings scipy), matplotlib 3.11, timenet-connectors 0.11.0 (which pulls `timenet[build]` for loading the card).
- PTB-XL 1.0.3 zip is at `https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip` (1.84 GB, plain HTTPS).

## Cross-cutting decisions

| Decision | Choice | Why |
|---|---|---|
| Package | `src/syntide/` with uv build backend (`module-name = "syntide"`); delete `main.py` | One importable package |
| TimeNet boundary | Only `syntide.data` imports `timenet`. Pin `timenet[torch]==0.11.0`, `timenet-connectors==0.11.0` | timenet_syntide.md §7 |
| Generator boundary | `syntide.generate` is pure NumPy/SciPy: no timenet, no torch | Generators, validators and planner stay independent of the data plane (timenet_syntide §7) |
| Config | Hydra 1.3 + one structured dataclass per entry point (`ConfigStore`) + small YAML groups in `conf/` | training_loop doc §1/§3. Hydra stays at the entry points |
| Training loop | **Lightning `Trainer`** (`lightning==2.6.*`) + `TensorBoardLogger` (its `.experiment` is a `SummaryWriter`) | Chosen by user. Used as a plain trainer: no DataModule, no custom callbacks |
| Sampling rate | 100 Hz (`records100`, 1,000 samples × 12 leads) | PTB-XL benchmark convention; 5× cheaper than 500 Hz |
| Labels | 5 diagnostic superclasses, multi-label, from `scp_statements.csv` (`diagnostic == 1`) | MVE task |
| No speculative abstractions | Concrete PTB-XL / ECG code; no plugin Protocols, registries or hydra-zen until a second dataset exists | User constraint |
| Dependencies | runtime: the TimeNet pins, `hydra-core>=1.3.7,<1.4`, `lightning==2.6.*`, `tensorboard`, `scikit-learn`, `scipy`, `matplotlib`; dev group: `pytest`, `ruff`, `wfdb` (fixtures only) | |
| Git | Ignore `.timenet-registry/`, `.cache/`, `runs/`, `.DS_Store`. **Commit `manifests/`** (manifests, calibrations, gate results: small JSON that is the frozen protocol) | |

Final layout:
```
conf/  data.yaml  generate.yaml  train.yaml  evaluate.yaml
       protocol/ptbxl_artifact_v2.yaml  arm/{A0,A1,A2,A3,C1,C2}.yaml
       regime/{scarcity10,scarcity25}.yaml  runtime/{mps,cpu}.yaml
src/syntide/
  data/      __main__.py schema.py adapter.py challenge.py leakage.py ptbxl/{__init__,connector}.py ptbxl/dataset.yaml
  generate/  __main__.py editor.py descriptors.py policy.py validate.py
  train/     __main__.py config.py model.py mixer.py trial.py
  eval/      __main__.py config.py stats.py freeze.py tensorboard.py report.py
tests/  data/  generate/  train/  eval/  test_seams.py
manifests/ptbxl-artifact-v2/  pool{10,25}.json  calibration/pool{10,25}.json  generator_gate_pool{10,25}.json
```

---

## Phase 1 — PTB-XL as a TimeNet dataset, plus manifests and the leakage gate

**Step 0 (scaffold):**
- Rewrite `pyproject.toml`: deps, uv build backend, ruff and pytest config.
- Extend `.gitignore` and delete `main.py`.
- `uv sync`, then run the 3.14 smoke check (see Risks).
- `README.md` grows with each phase.

### Taken from timenet_syntide.md (basic items only)
- Isolate TimeNet imports (§7).
- Typed facts go in annotations; provenance goes in namespaced `Record.metadata` (§7, row 1).
- Splits come from SYNTIDE's frozen fold policy and are stored on the Task (§7, row 5).
- Pin TimeNet and add a round-trip contract test (§7).
- Global subject namespace `ptb-xl:<patient_id>` and patient separation across the outer partitions. Explicit membership rules between retained, withheld and control manifests (§4A; training doc §5).
- **Deferred:** task-evidence compiler (§1), derivation graph and input-view policy (§4), joint coverage planning (§5), transfer ladder (§6), exporter corrections (§7, rows 2–4 and 6).

### Files
- `data/ptbxl/connector.py`: `PtbXlConnector(BasePhysioNetConnector[PtbXlSource])`, modelled on ECG-QA CoT.
  - `download_async`: `ensure_archive(PTBXL_URL, cache_dir)` → `find_dir_containing(..., "ptbxl_database.csv")`. Idempotent.
  - `convert`: one Record per `ecg_id` (mapping table below). Uses the CSV's `filename_lr` and the inherited `_read_header`/`_lead_loader` (lazy, mV, float32). `CONNECTOR = PtbXlConnector`.
  - Pure helpers: `diagnostic_superclasses(scp_codes, statements)` and `artifact_types(row)`. A noise column counts only if it is non-empty after `strip()`.
- `data/ptbxl/dataset.yaml`: `physionet/ptb-xl` 1.0.3, CC-BY-4.0, cardiology/health, tags, source_url. The card schema forbids extra keys.
- `data/schema.py` (no timenet import):
  - `LEADS` (I, II, III, aVR, aVL, aVF, V1–V6), `SUPERCLASSES`, `ARTIFACT_TYPES`;
  - `subject_uid(patient_id)`;
  - frozen `Samples(record_id, subject_uid, is_artifact, artifact_types, labels[n,5], x[n,12,T])`.
- `data/adapter.py`: `PtbXlAdapter(registry, dataset="physionet/ptb-xl@1.0.3")`.
  - `index() -> pa.Table`: one row per labelled record, built columnar from `annotation_table("Record")` + `target_table()`.
  - `load(ids) -> Samples`: `iter_records`, with leads reordered into `LEADS`.
  - `source_sha256()`: hash of the sorted `(path, checksum)` pairs of `manifest.files`.
- `data/challenge.py`: `ChallengeSpec` (also the Hydra schema node), `build_manifest`, `save_manifest`, and `load_manifest`, which re-hashes the file and rejects edits.
- `data/leakage.py`: `LeakageError`, `check_manifest(manifest, subject_of)` (Gate 1), and `check_calibration_ids(ids, manifest)` (calibration ⊆ `retained_artifact`).
- `data/__main__.py` + `conf/data.yaml`: `python -m syntide.data`. It builds the registry if missing (`run_pipeline(..., cache_dir=download_cache, keep_cache=True)`), then writes and checks `pool10.json` and `pool25.json`.

### PTB-XL → TimeNet mapping
| PTB-XL | TimeNet |
|---|---|
| `ecg_id` | `Record(record_id="ptbxl-{ecg_id}")` (same ids as ECG-QA CoT) |
| 12 leads, `records100` | 12 lazy `Signal`s, spec `ecg` (mV), ids `ecg-{ecg_id}-{lead}`, one `Source` |
| `patient_id`, `strat_fold` | int record annotations |
| 4 noise columns | str record annotations (stripped lead description), only when present |
| `scp_codes` | `Record.metadata["ptbxl.scp_codes"]` |
| diagnostic superclasses | `ClassificationTask(targets=sorted superclasses, target_schema="ptbxl.diagnostic_superclass", metadata={"split": …})`. Only emitted when ≥1 superclass exists |

### Challenge manifests
Fields: `challenge_id`, `spec` (folds, pool_fraction, retain, seed), `source` (dataset, version, source_sha256), `roles`, `sha256`.

How the roles are built:
- Outer partitions come from folds: train = 1–8, val = 9, test = 10.
- `pool`: `⌈fraction × n_subjects⌉` train **subjects** after one seeded shuffle. A subject's records go together, and the pools are nested (10% ⊂ 25%). `unused` = train − pool.
- `retained_artifact` = `⌈retain × n_artifact_in_pool⌉` random artifact records. `withheld` = the rest of the pool's artifact records.
- `depleted` = pool − withheld (parents for A0–A3).
- `natural_mix` = pool minus `|withheld|` random records (C1). `pool` itself is C2.

The leakage gate checks:
- the pool, val and test subjects are disjoint;
- every labelled record sits in exactly one outer partition;
- retained and withheld split the pool's artifacts exactly;
- `withheld ∩ depleted = ∅`;
- `natural_mix ⊆ pool` and `|natural_mix| = |depleted|`.

### Tests (tests/data)
A fixture writes 4 synthetic format-16 WFDB records (`wfdb.wrsamp`) plus the two CSVs. Two records share a patient, one has `burst_noise`, one has only rhythm codes, one has a whitespace-only noise string, and the folds vary. The fixture then runs `convert` → `store_dataset` → adapter.
1. **Signal fidelity:** values come back in mV (within one ADC step) and in clinical lead order.
2. **Label and task rules:** rhythm codes are ignored, superclasses are deduplicated, a record with no superclass has no task and is absent from `index()`, and the split follows the folds.
3. **Index semantics:** namespaced `subject_uid`. `is_artifact` and `artifact_types` count only real noise strings.
4. **Manifest construction** (synthetic index): subject-level, nested pools; ceil-rule counts; depleted excludes withheld; natural_mix matches depleted's size while keeping more artifact records.
5. **Determinism and integrity:** same spec → same sha; new seed → new pool; a hand-edited manifest is rejected.
6. **Leakage gate** (parametrized over each invariant, plus a calibration id outside `retained_artifact`): each raises `LeakageError` naming the offender.

**Done when:** the tests pass, `python -m syntide.data` builds the registry and both manifests, and the logged counts are close to the MVE's (pool25 ≈ 928 artifact records / ≈ 93 retained; test artifact slice ≈ 513 ECGs / 477 patients). Loading the pool25 depleted set takes under about a minute; otherwise add an `.npy` cache keyed by `source_sha256`.

---

## Phase 2 — Training harness, with the synthetic generator plugged into the mixer

### 2A. Harness (Hydra + structured dataclasses + Lightning)
Config (`train/config.py`; OmegaConf-safe types):
- `ProtocolConfig`: id, registry, dataset, batch_size=64, val_every=250, transform_fraction=0.5, `ModelConfig`, `OptimConfig`, `GenericAugConfig` (A1 envelope), `EditorConfig` (prior envelope per control, `mv_limit=10`, `alignment_tol=0.15`, `max_retries=5`).
- `ArmConfig`: id, `source_role` (depleted | natural_mix | pool), `transform_family` (none | generic | editor), `sampling_policy` (none | fixed | coverage_matched | gap_targeted).
- `RegimeConfig`: id, manifest, `calibration` (required for editor arms), `max_steps: int = MISSING`.
- `RuntimeConfig`: accelerator (mps | cpu), deterministic, log_every_n_steps.
- `TrainConfig`: the above plus replicate_seed (`train_schema`).

Budgets live in **regime**, so arms within a regime match by construction.

`conf/train.yaml`:
- run and sweep dirs both resolve to `runs/${protocol.id}/${regime.id}/${arm.id}/seed${replicate_seed}`;
- `hydra.sweeper.params`: `arm: A0,A1,A2,A3,C1,C2`, `regime: scarcity10,scarcity25`, `replicate_seed: 0..5` (72 trials).

Modules:
- `train/__main__.py` (the only Hydra code):
  1. `OmegaConf.to_object` → `TrainConfig`.
  2. `load_manifest` + `check_manifest` (Gate 1).
  3. `validate(spec, manifest)`:
     - the arm's role exists;
     - any transform family requires `source_role == depleted`;
     - `max_steps % val_every == 0`;
     - editor arms need a calibration whose `generator_gate` passed (Gate 4).
  4. Adapter load.
  5. `run_trial`.
- `train/model.py`: `ResNet1d` (~1M params, 12 → 5) and `EcgModule(L.LightningModule)`.
  - `training_step`: BCE. Counts presentations and transform slots. Logs `train/loss` and `budget/*`.
  - `on_validation_epoch_end`: sklearn macro-AUROC overall/artifact/clean, logs `val/auroc_*` and `val/loss`, and appends to `val_history`.
  - `configure_optimizers`: AdamW.
  - Plain functions `load_model(ckpt)` and `predict_proba(model, x, device)` are reused by Phase 3.
- `train/mixer.py` (training doc §1, §4A):
  - `rng(seed, stream, *keys)`: separate INIT, PARENTS, SLOTS and TRANSFORM streams, keyed by `(replicate_seed, step, slot, attempt)` and **never the arm id**.
  - `batch_plan(...)`: a seeded permutation per pass over the parents, plus a slot mask per step.
  - Transform families:
    - `none`: identity (A0's parents repeat in transform slots);
    - `generic`: A1 per-lead gain, shift, Gaussian noise and sinusoidal wander;
    - `editor`: §2B. The same uniform draws `u` go through A2's or A3's policy, so A2 and A3 differ only in the controls.
  - `PairedStream(IterableDataset)` yields `(x, y, mask)` per step through `DataLoader(batch_size=None, num_workers=0)`. It appends one ledger row per candidate (including rejected ones).
- `train/trial.py`:
  - `trial_id`: sha256 of the scientific inputs (protocol, arm, regime, seed, manifest sha, source sha, calibration sha, generator and validator versions, `syntide.__version__`, accelerator). It excludes logging settings and paths.
  - `run_trial(spec, train, val, out_dir, calibration) -> RunResult`:
    1. If a completed `result.json` with the same id exists, return it. A different id is refused.
    2. `L.seed_everything` from INIT.
    3. `L.Trainer` with: `max_steps`; `val_check_interval=val_every`; `check_val_every_n_epoch=None`; `accumulate_grad_batches=1`; `num_sanity_val_steps=0`; `ModelCheckpoint(filename="best", monitor="val/auroc_overall", mode="max", save_top_k=1)`; and `TensorBoardLogger(save_dir=out_dir, name="", version="")`.
    4. Fit.
    5. Save `final.ckpt` explicitly (training doc §4B).
    6. Read the best step from the best checkpoint's `global_step`.
    7. Write val predictions for both checkpoints and the ledger. Ledger rows from Lightning's one-batch prefetch beyond `max_steps` are dropped.

### 2B. Synthetic generator: `ecg.electrode_artifact@1.0.0` (`syntide/generate/`)
- `editor.py`:
  - **Electrodes and projection:**
    - `ELECTRODES = (RA, LA, LL, V1–V6)`.
    - The fixed 12×9 matrix `LEAD_FROM_ELECTRODE`: I = LA−RA, II = LL−RA, III = LL−LA, Goldberger augmented leads, and Vi − WCT with WCT = (RA+LA+LL)/3 (MVE challenge #4).
  - **Artifact types:** v1 covers `baseline_drift`, `static_noise` and `burst_noise`. `electrodes_problems` (~30 records) is deferred.
  - **Controls:** `Control(name, unit, low, high)` for `amplitude_mv` (the severity), `freq_hz`, `onset_s` and `duration_s`, plus categorical `artifact_type` and `electrode`.
  - **Recipes:** `Recipe(parent_id, generator, version, controls, seed_key, gap_report_id)`. `recipe_id` = sha256 of the canonical JSON.
  - **`edit(parent_x, recipe, fs)`** returns a `Candidate`: the edited x, the projected residual, the interval and the affected leads. Waveform noise comes only from `seed_key`, so regeneration is bit-exact.
  - **`LABEL_EFFECT`:** `{diagnosis: preserves, artifact_type: determines, signal_quality: invalidates}`, recorded in the ledger.
- `descriptors.py` (`ecg.electrode_artifacts@1`; the coverage space, so the policies never see generator knobs):
  - `wander_mv`: half the peak-to-peak of a < 0.7 Hz zero-phase low-pass;
  - `hf_rms_mv`: 1.4826·MAD of a > 15 Hz high-pass (robust to QRS without beat detection);
  - `burst_rms_mv` and `burst_duration_s`: from a 0.25 s windowed HF-RMS envelope;
  - each is the max over leads.
- `policy.py` (gap detector and control planner, v1):
  - `fit_calibration(manifest, retained: Samples, envelope)`:
    - calls `check_calibration_ids` (only `retained_artifact` records);
    - measures each retained record's severity descriptor for its artifact type;
    - builds a **GapReport**: retained type frequencies, plus the retained severity density vs the prior envelope on a grid;
    - writes `manifests/<protocol>/calibration/poolXX.json` with its sha. This is the `gap_report_id`.
  - `controls_for(policy, u, calibration)`:
    - `coverage_matched` (A2): type ~ retained frequencies; severity = retained empirical quantile map `Q_ret(u)`, clipped to the envelope;
    - `gap_targeted` (A3): type weights ∝ prior / (retained + α); severity = inverse CDF of `envelope / (KDE_ret + ε)`, so low-density regions inside the envelope are up-weighted;
    - nuisance controls map identically for both policies.
  - Per-type 1-D severity maps avoid jointly implausible combinations (timenet_syntide §5).
- `validate.py` (timenet_syntide §2 layers, basic subset):
  - `bounds@1` (blocking): finite, shape intact, |x| ≤ `mv_limit`.
  - `alignment@1` (blocking): severity measured on the projected residual is within `alignment_tol` of the requested value.
  - Report-only: in-context severity measured on the edited signal.
  - **Rejection handling:** the parent and slot stay fixed while the next draw is tried (`attempt` key), up to `max_retries`. Exhaustion is recorded (`exhausted_slots` in RunResult), never silently substituted (training doc §4A).
- **Ledger** (`recipes.parquet` per trial; DuckDB can query it), with fields from the MVE table:
  - recipe_id, parent_record_id, parent_subject_uid;
  - generator, version;
  - controls_requested, controls_realized;
  - seed_key, rng_stream (`A3/s3/step1200/slot17/try0`), gap_report_id;
  - validations (name, version, value, pass), decision, reason;
  - label_effect.
  - A1 recipes are logged the same way.
- `generate/__main__.py` + `conf/generate.yaml`, run per regime before the grid:
  1. Fit the calibration.
  2. Run the **generator gate (Gate 4)** on 1,000 recipes per policy over depleted parents. For each continuous control, realized vs requested needs slope ∈ [0.9, 1.1], |intercept| ≤ 5% of the range, MAE ≤ 10% of the range and corr ≥ 0.9. Correlation alone would pass a 2×+0.2 bias. All outputs must be finite and within bounds.
  3. Write `generator_gate_poolXX.json`, `gate_recipes.parquet` (for Phase 3's audit views), and `review_sample.csv`. The review sample holds 40 recipes, stratified toward severity extremes, with an empty `verdict` column for one-time expert calibration review.

### Outputs per trial (Phase 3's input contract)
`.hydra/config.yaml`, TensorBoard events, `checkpoints/{best,final}.ckpt`, `recipes.parquet`, and:
- `predictions_val.parquet`: `trial_id, arm, regime, seed, checkpoint, step, record_id, subject_uid, class_id, y_true, y_prob, is_artifact`;
- `result.json` (RunResult):
  - trial info: status, ids, best_step, best_val_auroc, final_step;
  - exposure: presentations, transform_slots, candidates, accepted, exhausted_slots, distinct_recipes;
  - timing: wall_time_s;
  - provenance: manifest, source and calibration shas; registry, dataset and paths; checkpoint shas;
  - env.

No test-split access anywhere in Phase 2.

### Tests (tests/generate, tests/train), using synthetic arrays and CPU (no TimeNet)
1. **Electrode projection:** an LA artifact appears in every lead except II, where it is exactly zero. An RA artifact is absent from III, and a V3 artifact appears only in V3.
2. **Recipe provenance:** regenerating from a ledger row is bit-identical, and `recipe_id` changes when any field changes.
3. **Descriptor calibration and gate:**
   - on a synthetic ECG with QRS-like spikes, injected wander, noise and burst are recovered within tolerance;
   - the gate fails a generator with realized = 2×requested + 0.2 despite corr = 1.
4. **Policies:**
   - with the same `u`, A2 and A3 both stay within the envelope;
   - A2 reproduces the retained quantiles;
   - A3 moves mass to low-density severities and under-represented types;
   - fitting on a withheld record raises `LeakageError`.
5. **Acceptance:** a rejected candidate retries with the same parent, slot and next draw, and exhaustion is recorded, not replaced.
6. **Pairing:** with the same seed, A0–A3 share parent indices, slot masks and initial weights, and A2/A3 share `u` while their controls differ. A0's batches equal their parents. A new seed gives a new schedule.
7. **Budget and checkpoints** (Lightning on CPU, 30 steps, val_every 10):
   - presentations = steps × batch;
   - transform_slots = Σmask (0 for C1);
   - `best_step` = argmax of `val_history`, and `final.ckpt` has `global_step == 30`;
   - the ledger holds only rows with step ≤ 30;
   - the predictions cover both checkpoints × all records × 5 classes.
8. **Trial identity and reuse:** the id ignores logging settings and paths, and changes with seed, max_steps, manifest, calibration and accelerator. A completed trial is reused, and a different trial in the same dir is refused.
9. **Config boundary** (Hydra `compose`, parametrized): each of these is refused before any data loads: missing `max_steps`, a transform arm on a non-depleted role, a non-divisible `max_steps`, and an editor arm without a passed gate.

**Done when:**
1. The tests pass.
2. The pilot (`arm=A1 regime=scarcity25 regime.max_steps=20000 hydra.run.dir=runs/pilot/A1-25`) sets numeric `max_steps` in both regime YAMLs.
3. The model gate runs (width 128 and 2× steps, in separate run dirs).
4. The gap-gate subset runs (`-m arm=A0,C1 regime=scarcity25 replicate_seed=0,1,2`).
5. `python -m syntide.generate` passes Gate 4 for both regimes.
6. The full sweep runs and skips the 6 reused trials.

---

## Phase 3 — Evaluation, test-once discipline, TensorBoard views, results for the presentation

### Modules
- `eval/config.py` + `conf/evaluate.yaml`: `EvaluateConfig` with:
  - `runs_dir`, `split` (val | test), `checkpoint` (best | final), `mode` (evaluate | freeze | audit);
  - `n_boot=2000`, `ci_level=0.95`, `bootstrap_seed`;
  - **primary** `[A3, A1]` (artifact slice, scarcity25);
  - **fixed sequence** `[[A3,A2],[A2,A1]]`, then the same chain at scarcity10;
  - diagnostics `[[A1,A0],[C1,A0],[C2,A0]]`;
  - `practical_target=0.02`, `noninferiority_margin=-0.01`, `gap_gate: {contrast: [C1,A0], regime: scarcity25, min_effect: 0.02}`.
  - Contrast labels follow timenet_syntide §3A: A2−A1 is "two recipes", not "realism".
- `eval/stats.py` (pure):
  - `macro_auroc(y, p, weights)`: classes without both weighted labels are skipped, so the result never contains NaN.
  - `paired_contrast(...) -> UtilityReport`:
    - `PairingError` if the seeds, records or labels differ between arms;
    - estimate = mean over seeds of the paired difference;
    - crossed bootstrap: patient multiplicities become record weights, and seeds are resampled; the same draw serves both arms.
  - `findings(...)`, as separate flags (timenet_syntide §3B):
    - `positive_effect_supported` (artifact ci_low > 0);
    - `target_effect_ruled_out` (artifact ci_high < 0.02);
    - `clean_noninferiority_supported` (clean **ci_low** > −0.01);
    - `gap_gate_passed` (estimate ≥ 0.02).
  - `fixed_sequence(...)`: stops at the first contrast that fails; later ones are marked "not tested".
  - `gap_closed(...)`: reports the numerator (A3−A0) and denominator (C1−A0) separately; the ratio only when the gap gate passed and the denominator > 0 (timenet_syntide §3C).
- `eval/freeze.py` (Gate 5):
  - `write_freeze` records the completed trial ids, checkpoint shas, manifest and calibration shas, a hash of the decision rules, and the git rev;
  - `verify_freeze` raises `FreezeError` naming each change;
  - `claim_test_evaluation` is a one-shot marker (`open(path, "x")`).
- `eval/tensorboard.py` (functions taking a `SummaryWriter`; untested presentation code; every figure is also saved as PNG under `figures/` for slides). The five views from results_visualization.md:
  - **Training health:** comes from Phase 2's scalars.
  - **Downstream utility:** `log_paired_effects`, a forest plot per regime with artifact and clean panels, CI and seed dots, and reference lines at 0/+0.02 and 0/−0.01.
  - **Coverage & acceptance:** `log_coverage` shows retained natural vs requested vs accepted severity per type and policy, acceptance rate by type, severity bin and rejection reason, plus distinct recipes per transformed presentation (diversity / memorization).
  - **Signal inspection:** `log_signal_examples` shows fixed development records (original / edited / residual), with consistent mV axes and the edited interval shaded.
  - **Failure slices:** `log_slice_differences` gives per-class Δ with counts.
  - Plus `log_hparams` (one row per trial) and `log_findings` (markdown).
- `eval/report.py`: `results.md`, holding:
  - the MVE's **one results table** (arm × artifact 25% / clean 25% / Δ vs A1 [CI] / gap closed / artifact 10%, with seed SD per arm);
  - the findings flags and the fixed-sequence outcome;
  - gate outcomes 1–5, read from their JSON files;
  - run counts and compute time.
- `eval/__main__.py`:
  - `mode=audit`: coverage and examples from `gate_recipes.parquet`, plus a **label audit** (report-only). The audit takes the mean |Δp| and the flip rate of A0's best checkpoint between each parent and its edit over the gate recipes, which shows whether edits erase diagnostic evidence.
  - `mode=freeze`: writes `freeze.json`.
  - `split=val`: reports, TensorBoard and `results.md` from `predictions_val.parquet`.
  - `split=test`: `verify_freeze` + `claim_test_evaluation`, then inference on fold 10 for the **frozen** trials only (Phase 1 adapter, Phase 2 `load_model`/`predict_proba`), then the same reports.

### Tests (tests/eval and tests/test_seams.py)
1. **Paired arms cancel:** identical predictions give estimate 0 and a CI of exactly [0, 0].
2. **Known effect detected:** the CI excludes 0, and every seed's effect is reported.
3. **Pairing guard:** differing seeds, records or labels raise `PairingError`.
4. **Degenerate class:** it is excluded, and the result is finite.
5. **Decision rules:**
   - CI [0.003, 0.015] sets both `positive_effect_supported` and `target_effect_ruled_out`;
   - a clean estimate of −0.005 with ci_low −0.02 does not support non-inferiority;
   - the gap gate is inclusive;
   - the fixed sequence stops at the first failure;
   - gap closed is withheld when the gate fails or the denominator ≤ 0.
6. **Freeze guard** (parametrized): a changed checkpoint, rule or calibration is rejected, and a second test claim is refused.
7. **Seam test:** a tiny Phase 2 run of A0 and A3 produces outputs that Phase 3's `paired_contrast` accepts, and its ledger parents are all in `depleted`.

**Done when:** the tests pass; `mode=audit` and `split=val` populate all five TensorBoard views and `results.md`; the test-once flow works on a throwaway runs dir (a second `split=test` is refused); and the real `split=test` is run exactly once after freezing.

---

## Phase seams
| From → to | Contract |
|---|---|
| Phase 1 → 2 | Registry `physionet/ptb-xl@1.0.3`; `manifests/<protocol>/pool{10,25}.json`; `PtbXlAdapter.load → Samples`; `load_manifest`, `check_manifest`, `check_calibration_ids` |
| Phase 2 → 3 | Run dirs (`predictions_val.parquet`, `result.json`, `recipes.parquet`, `checkpoints/*.ckpt`); calibration + gate JSON + `gate_recipes.parquet`; `load_model`, `predict_proba`, `edit` |
| Phase 3 → presentation | `results.md`, `figures/*.png`, `comparisons.json` (UtilityReports), TensorBoard |

## Schedule against the brief's 3-day budget, and deliverables
| Day | Build | Run / gate |
|---|---|---|
| 1 | Phase 1; Phase 2A | Gate 1; pilot + model gate (Gate 2); gap-gate runs A0/C1 × 3 seeds |
| 2 | Phase 2B; Phase 3 stats + views | Gap gate verdict (Gate 3; MVE fallback: retain 0% and retest once); Gate 4; remaining 66 trials (≈ 8–13 h on the M1 Max) |
| 3 | Phase 3 report | Freeze (Gate 5) → test once → `results.md` + figures → slides for the 45-minute discussion |

Deliverables:
- the code repo, with a README covering how to reproduce each gate;
- `results.md` and figures;
- slides covering the design (from the architecture and MVE docs) and the experiment;
- a notes section filled from the actual findings: what evidence would change the conclusion, and next week's plan with a larger cluster (from MVE §Plan).

## Out of scope (future plugins at these seams)
TimeNet exporter (provenance lives in the ledger until then), `electrodes_problems` artifact type, electrode-attribution descriptor, joint multi-control planning, validation-error slices in the gap detector, two-sample plausibility test, blinded gap discovery, NSTDB stress test, learned or text-conditioned generators, Plotly export, hydra-zen, Ray/Slurm launchers.

## Risks
- Hydra 1.3 and Lightning 2.6 resolve on Python 3.14, but Lightning only declares support up to 3.13. Step 0 runs an import + compose + 2-step CPU `Trainer.fit` smoke check; if it fails, pin `.python-version` to 3.13.
- Retained artifact counts are small (≈ 37 / 93 split over 3 types), so the per-type quantile maps are noisy. A KDE floor `ε` and clipping to the envelope regularize them; the numbers are reported in the gap report.
- Descriptors measured on natural ECGs mix with physiological content (T waves in the wander band, QRS in the high-frequency band). The robust estimators limit this, and the coverage view shows the residual bias.
- TimeNet per-lead reads may be slow; measure, and add the `.npy` cache only if needed.
- MPS is not bitwise deterministic. Pairing holds at the RNG and data level, and the accelerator is part of the trial id.
- Disk: 1.84 GB download + about 3 GB extracted + the registry.

## Verification (end to end)
```bash
uv sync && uv run ruff check && uv run pytest
uv run python -m syntide.data                                    # registry + manifests (Gate 1)
TIMENET_REGISTRY=.timenet-registry uv run timenet info physionet/ptb-xl
uv run python -m syntide.train arm=A1 regime=scarcity25 regime.max_steps=20000 hydra.run.dir=runs/pilot/A1-25
uv run python -m syntide.train -m arm=A0,C1 regime=scarcity25 replicate_seed=0,1,2   # gap-gate runs
uv run python -m syntide.generate                                # calibration + Gate 4 + review sample
uv run python -m syntide.eval mode=audit                         # coverage, examples, label audit
uv run python -m syntide.train -m                                # full 72-trial grid (reuses 6)
uv run python -m syntide.eval split=val                          # gap gate, forest plot, results.md (val)
uv run python -m syntide.eval mode=freeze && uv run python -m syntide.eval split=test   # once
uv run tensorboard --logdir runs
```
