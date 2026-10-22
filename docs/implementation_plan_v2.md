# SYNTIDE — three-phase implementation plan

## Context

The Aionic brief (docs/Synthetic_Data_Engine_Task.pdf) asks for a small proof of concept with one central question: **can synthetic data improve performance on held-out real data?** The experiment must use a defensible baseline, isolate the synthetic effect, prevent leakage across generation, training and evaluation, match training budgets, and repeat runs. Total effort is capped at three days, and the work ends in a 45-minute presentation.

SYNTIDE's design (docs/SYNTIDE_Architecture.md, docs/SYNTIDE_MVE.md) answers this on PTB-XL. It edits real training ECGs with a physically grounded artifact generator and compares it with standard augmentation on naturally artifact-bearing test ECGs. The repo has only docs, a hello-world `main.py` and `timenet[cli,torch]==0.11.0` in `.venv`.

The plan keeps the three requested phases:
1. PTB-XL in TimeNet;
2. the generator and the paired training stream, consumed by a trainer that lives outside the package;
3. evaluation, with the dashboard backend (local files, TensorBoard or Weights & Biases) chosen in YAML.

SYNTIDE is three packages: `syntide.data`, `syntide.generate` and `syntide.eval`. Training is not part of it.
- Any loop (plain PyTorch, Lightning, Hydra) can train a trial if it reads SYNTIDE's dataclasses from `syntide.trial` and hands back the outputs SYNTIDE defines.
- The first trainer is plain PyTorch, in `trainers/`.
- SYNTIDE still produces everything the comparison depends on: the batches, the step budget, exposure counts, the recipe ledger, the selection metric and the test predictions. A trainer therefore cannot weaken the experiment. It can only fail SYNTIDE's checks.

Each phase has its own package, tests and file outputs, and each phase's outputs are the next phase's inputs.

### What the brief requires that the first draft missed
| Brief requirement | First draft | Added in this revision |
|---|---|---|
| "Generate synthetic examples" and compare with vs without them | Generator deferred; only A1 standard augmentation | **Electrode-space artifact editor** plugged into Phase 2's training stream → arms A2/A3 and the full 72-run grid (this widens the earlier "A0, A1, C1, C2" choice) |
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
- All deps resolve on Python 3.13 (Hydra 1.3 crashes on 3.14): hydra-core 1.3.7, tensorboard 2.21, torch 2.14 (from `timenet[torch]`), scikit-learn 1.9 (which brings scipy), matplotlib 3.11, timenet-connectors 0.11.0 (which pulls `timenet[build]` for loading the card).
- PTB-XL 1.0.3 zip is at `https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip` (1.84 GB, plain HTTPS).

## Cross-cutting decisions

| Decision | Choice | Why |
|---|---|---|
| Package | `src/syntide/` with uv build backend (`module-name = "syntide"`); delete `main.py` | One importable package |
| TimeNet boundary | Only `syntide.data` imports `timenet`. Pin `timenet[torch]==0.11.0`, `timenet-connectors==0.11.0` | timenet_syntide.md §7 |
| Generator boundary | `syntide.generate` is pure NumPy/SciPy: no timenet, no torch | Generators, validators and planner stay independent of the data plane (timenet_syntide §7) |
| Config | Hydra 1.3 + one structured dataclass per SYNTIDE entry point (`ConfigStore`) + small YAML groups in `conf/`. The whole experiment (arms, regimes, seeds, budgets) is one file, `conf/protocol/ptbxl_artifact_v2.yaml` | training_loop doc §1/§3. Hydra stays at SYNTIDE's entry points. Trainers get plain dataclasses and need no Hydra |
| Training | **Outside the package.** `syntide/trial.py` is the seam (Phase 2A). The first trainer is **plain PyTorch** in `trainers/`, a uv workspace member (`syntide-trainers`) that imports syntide; syntide never imports it | Chosen by user on 2026-10-21, replacing Lightning. Any framework (PyTorch, Lightning, Hydra) works if it passes SYNTIDE's checks |
| Tracking | `syntide.eval.tracking`: one `Tracker` Protocol with `files`, `tensorboard` and `wandb` backends, chosen by `conf/tracker/*.yaml`. Trainers and eval use the same backend | Chosen by user. The evidence files never depend on the backend |
| Sampling rate | 100 Hz (`records100`, 1,000 samples × 12 leads) | PTB-XL benchmark convention; 5× cheaper than 500 Hz |
| Labels | 5 diagnostic superclasses, multi-label, from `scp_statements.csv` (`diagnostic == 1`) | MVE task |
| No speculative abstractions | Concrete PTB-XL / ECG code; no plugin Protocols, registries or hydra-zen until a second dataset exists | User constraint |
| Dependencies | syntide runtime: the TimeNet pins, `hydra-core>=1.3.7,<1.4`, `tensorboard`, `scikit-learn`, `scipy`, `matplotlib`; optional extra `wandb`; dev group: `pytest`, `ruff`, `wfdb` (fixtures only). `trainers/`: `syntide` (workspace source) + `torch`. No `lightning` | |
| Git | Ignore `datasets/`, `runs/` (which also holds the tracker output, `wandb/` included), `outputs/`, `.DS_Store`. **Commit `manifests/`** (manifests, calibrations, gate results: small JSON that is the frozen protocol) | |

Final layout:
```
conf/  data.yaml  generate.yaml  evaluate.yaml
       protocol/ptbxl_artifact_v2.yaml  tracker/{files,tensorboard,wandb}.yaml
src/syntide/      # data, generate, eval + the trainer seam; no training code
  trial.py   TrialSpec, Batch, TrainerInfo, Checkpoint, Predictor; prepare, open_trial, is_complete, Trial.finish
  data/      __main__.py schema.py adapter.py challenge.py leakage.py ptbxl/{__init__,connector}.py ptbxl/dataset.yaml
  generate/  __main__.py editor.py descriptors.py policy.py validate.py stream.py
  eval/      __main__.py config.py runs.py stats.py freeze.py tracking.py views.py report.py
trainers/         # uv workspace member syntide-trainers: imports syntide, never imported by it
  pyproject.toml  conf/torch_loop.yaml  src/syntide_trainers/{__init__,torch_loop}.py  tests/test_torch_loop.py
tests/  data/  generate/  eval/  test_trial.py  test_seams.py
manifests/ptbxl-artifact-v2/  pool{10,25}.json  calibration/pool{10,25}.json  generator_gate_pool{10,25}.json
runs/<protocol>/<regime>/<arm>/seed<n>/   trial.json from syntide, then the run outputs (Phase 2)
```

### Where the experiment's controls live, now that training is outside the package
| Brief requirement | Enforced by |
|---|---|
| Control for training budget | The stream yields exactly `max_steps` batches of `batch_size` and counts exposure itself. `finish` refuses an early stop. Eval refuses unequal presentations and different trainer fingerprints |
| Isolate the synthetic effect | A0–A3 share the stream's parent and slot schedule. Eval requires the same trainer fingerprint, and the same `init_sha256` per (regime, seed) |
| No leakage between generation, training and evaluation | Planning and `prepare` both run Gate 1. The trainer gets only its arm's role and `val`. Test predictions are made only inside `syntide.eval`, after the freeze |
| Complete provenance | The stream writes `recipes.parquet` and `exposure.json`, not the trainer. `result.json` records the trainer's name, version, config and predictor, and the checkpoint shas |

---

## Phase 1 — PTB-XL as a TimeNet dataset, plus manifests and the leakage gate

**Step 0 (scaffold):**
- Rewrite `pyproject.toml`: deps, uv build backend, ruff and pytest config.
- Extend `.gitignore` and delete `main.py`.
- `uv sync` on Python 3.13 (see Risks).
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

## Phase 2 — Generator, paired training stream and the trainer seam

> **Status (2026-10-21): Phase 2 is built (2A, 2B, 2C).** Tests 1–10 are in place, plus the Phase 3 tracking tests (8), because trainers need `syntide.eval.tracking` and `val_scores` now. `python -m syntide.generate` passes Gate 4 for both regimes. Where the code differs from the text below:
> - Extra modules in `generate/`:
>   - `plan.py`: the protocol schema and the planner, so `__main__` stays thin;
>   - `ledger.py`: `Recipe`, recipe ids and the Parquet writer, shared by A1 and the editor;
>   - `generic.py`: A1, `per_lead_aug@1`;
>   - `gate.py`: Gate 4 and the review sample.
> - `max_steps: null` instead of `???`, so one regime can be planned while the other waits for its pilot.
> - Each regime names its `generator_gate` JSON. Planning an editor arm needs `"passed": true` from a gate run on exactly the regime's calibration file, fit on the same manifest and editor settings.
> - `TrialSpec` also carries:
>   - `transform`: the family's settings, so a new A1 envelope changes A1's id but not A0's;
>   - `generator` and `validators`;
>   - `registry`: a path, kept out of the id.
>
>   `Batch` also carries each slot's parent `record_id`.
> - C1 and C2 are planned with `transform_fraction = 0`. A0 keeps the shared slot schedule, so its `transform_slots` equals A1–A3's.
> - `result.json` is write-once: `prepare` and `finish` refuse a finished run directory.
> - `syntide-trainers` is also in syntide's dev group, so a plain `uv sync` installs it. syntide still never imports it (a test checks this).
> - `ResNet1d` stage widths are (w, w, 2w, 2w, 3w), 0.95M parameters at w=64. AdamW uses a constant learning rate.
> - **Burst descriptor:** 0.5 s windows, not 0.25 s. At 100 Hz the 15 Hz high-pass makes each QRS ring for about 0.15 s, which fills a 0.25 s window, so the windowed RMS measured beats rather than noise. It now takes a robust (median) envelope, grows the segment around the loudest window, and refines the edges on a 0.1 s power trace. Clean ECGs read about 0.014 mV and burst-flagged ones 0.023–0.03 mV.
> - **Amplitudes are in descriptor units.** Every lead coefficient is at most 1 in magnitude, so the residual's strongest lead carries the requested amplitude. Realized controls are measured on that lead, which equals the maximum over leads. Burst noise is normalized inside its interval, and drift frequency is 0.1–0.4 Hz, so 10 s always hold a full cycle (frequency is measured from zero crossings).
> - **Envelope:** drift 0.1–3.5 mV, static noise 0.005–0.1 mV, burst noise 0.01–0.6 mV over 0.5–3 s. It covers every retained ECG of both pools. A1's wander was widened to 3.5 mV to share the drift ceiling, and planning enforces the shared ceilings.
> - `in_context@1` (report-only) runs in the gate, not in the training stream, to keep editor arms at about 50 ms per step against 34 ms for A0.
> - `bounds@1` allows a parent that already exceeds `mv_limit`, but an edit may not push past `max(mv_limit, parent peak)`.
> - An exhausted slot keeps its unchanged parent and is counted in `exhausted_slots`. Gate outputs live in `manifests/<protocol>/generator_gate_poolXX/` (`recipes.parquet`, `review_sample.csv`).
> - **Pilot and budgets:** A1's validation AUROC peaks early and then overfits, instead of plateauing. With the final config (`runs/pilot2`, seed 0, validation every 100), the smoothed curve peaks at step 1,600 (25%, 0.905) and step 700 (10%, 0.885). So `max_steps` is twice that, rounded up to a multiple of `val_every`: 3,250 and 1,500. An earlier 20k-step pilot, run with 1 mV wander, peaked at steps 1,000 and 500.

Training is not part of the package. SYNTIDE plans each trial, serves its batches, and checks and records what the trainer hands back. Any trainer that goes through `syntide.trial` is acceptable. The first one is plain PyTorch (§2C).

### 2A. Experiment plan, trainer seam and training stream

**Experiment definition.** `conf/protocol/ptbxl_artifact_v2.yaml` (schema `ProtocolConfig`, with OmegaConf-safe types) holds the whole experiment:
- id, registry, dataset, batch_size=64, val_every=250, transform_fraction=0.5, `GenericAugConfig` (A1 envelope) and `EditorConfig` (prior envelope per control, `mv_limit=10`, `alignment_tol=0.15`, `max_retries=5`);
- `arms:` A0–A3, C1 and C2. Each has a `source_role` (depleted | natural_mix | pool), a `transform_family` (none | generic | editor) and a `sampling_policy` (none | fixed | coverage_matched | gap_targeted);
- `regimes:` scarcity10 and scarcity25. Each has a `manifest`, a `calibration` (required for editor arms) and `max_steps: ???`, which the pilot sets;
- `seeds: [0, 1, 2, 3, 4, 5]`.

Budgets live in the regime, so arms within a regime match by construction. Model, optimizer and device belong to the trainer's own config (§2C).

**Planning.** `python -m syntide.generate mode=plan` writes the plan before anything runs (training doc §4D). It can be narrowed with `select.arms=[A0,C1] select.regimes=[scarcity25] select.seeds=[0,1,2]`; `runs_dir` defaults to `runs`.
1. It runs `load_manifest` + `check_manifest` (Gate 1), then the semantic checks. Nothing is written if one fails:
   - the arm's role exists;
   - any transform family requires `source_role == depleted`;
   - `max_steps` is set and `max_steps % val_every == 0`;
   - editor arms need a calibration whose `generator_gate` passed (Gate 4).
2. It writes one frozen `TrialSpec` per selected (regime, arm, seed) to `runs/<protocol>/<regime>/<arm>/seed<n>/trial.json`, plus a `runs/<protocol>/plan.json` index. An existing `trial.json` with a different `trial_id` is refused.

**The seam: `syntide/trial.py`.** A trainer needs only this module and `syntide.eval.tracking`. The dataclasses hold NumPy arrays, and nothing here imports torch.
- `TrialSpec`:
  - protocol id, arm, source_role, transform_family, sampling_policy, and the generator and validator versions;
  - regime, manifest path + sha, source sha, calibration path + sha;
  - replicate_seed, and `init_seed` from the INIT stream (trainers seed model initialization with it);
  - batch_size, max_steps, val_every, transform_fraction;
  - `trial_id`: sha256 of all of the above plus `syntide.__version__`. It excludes trainer settings, logging settings and paths.
- `Batch(step, x[B,12,T] float32 mV, y[B,5] float32, transformed[B] bool)`.
- `TrainerInfo(name, version, config, device, predictor="module:function")`, with a `fingerprint` (sha256 of name, version, config and device). Compared arms must share it.
- `Checkpoint(path, step)`, and the `Predictor` Protocol: `(checkpoint, x, device) -> probs[n,5]`.
- `prepare(run_dir, *, registry, tracker) -> Trial`:
  1. reads `trial.json` and reruns Gate 1 (`PtbXlAdapter.index/source`, `check_manifest`);
  2. checks the calibration gate for editor arms;
  3. loads the arm's role and `val` with `PtbXlAdapter.load`. It never loads `test`;
  4. calls `open_trial(spec, parents, val, tracker)`, which builds the stream and opens the tracker. `open_trial` does no I/O, so tests call it directly.

  `Trial` exposes `spec`, `batches()`, `val` and `tracker`.
- `is_complete(run_dir, trainer) -> bool` lets a trainer skip finished trials.
- `Trial.finish(trainer, best, final, val_history, val_probs={"best", "final"}, init_sha256, wall_time_s)`:
  - It refuses when:
    - the stream was not fully consumed;
    - `final.step != max_steps`;
    - `best.step` is not a multiple of `val_every`, or is not the argmax of `val/auroc_overall` in `val_history`;
    - `val_scores(val_probs["best"])` does not reproduce the reported score;
    - a checkpoint file is missing;
    - an existing `result.json` holds a different trial_id or trainer fingerprint.
  - Otherwise it writes the run outputs (see "Outputs per trial") and closes the tracker.
- SYNTIDE owns the selection metric. Trainers call `syntide.eval.stats.val_scores(val, probs)`, which returns `val/auroc_{overall,artifact,clean}` (sklearn macro-AUROC) and `val/loss`.

**The training stream: `generate/stream.py`** (training doc §1, §4A). It is framework-free.
- `rng(seed, stream, *keys)`: separate INIT, PARENTS, SLOTS and TRANSFORM streams, keyed by `(replicate_seed, step, slot, attempt)` and **never the arm id**.
- `batch_plan(...)`: a seeded permutation per pass over the parents, plus a slot mask per step.
- Transform families:
  - `none`: identity (A0's parents repeat in transform slots);
  - `generic`: A1 per-lead gain, shift, Gaussian noise and sinusoidal wander;
  - `editor`: §2B. The same uniform draws `u` go through A2's or A3's policy, so A2 and A3 differ only in the controls.
- `TrainStream` yields exactly `max_steps` NumPy `Batch`es and then stops, so the data enforces the budget whatever the trainer does.
  - It keeps its own exposure counters: presentations, transform slots, candidates, accepted, exhausted slots and distinct recipes.
  - It appends one ledger row per candidate, rejected ones included.
  - At each `val_every` boundary it logs `budget/*` and `generator/acceptance_rate` to the tracker, so every trainer's dashboard shows them.
  - A PyTorch trainer wraps each batch with `torch.from_numpy`. No `DataLoader` is needed.

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
- `generate/__main__.py` + `conf/generate.yaml` with `mode=calibrate` (the default; `mode=plan` is §2A), run per regime before the editor arms are planned:
  1. Fit the calibration.
  2. Run the **generator gate (Gate 4)** on 1,000 recipes per policy over depleted parents. For each continuous control, realized vs requested needs slope ∈ [0.9, 1.1], |intercept| ≤ 5% of the range, MAE ≤ 10% of the range and corr ≥ 0.9. Correlation alone would pass a 2×+0.2 bias. All outputs must be finite and within bounds.
  3. Write `generator_gate_poolXX.json`, `gate_recipes.parquet` (for Phase 3's audit views), and `review_sample.csv`. The review sample holds 40 recipes, stratified toward severity extremes, with an empty `verdict` column for one-time expert calibration review.

### 2C. Reference trainer: plain PyTorch in `trainers/` (outside the package)
- `trainers/pyproject.toml` defines the `syntide-trainers` package, which depends on `syntide` (workspace source) and `torch`. The root `pyproject.toml` lists it under `[tool.uv.workspace] members`, and `uv sync --all-packages` installs both.
- `trainers/conf/torch_loop.yaml`: model width, lr, weight_decay, device, `log_every`, and the tracker YAML to use (default `conf/tracker/tensorboard.yaml`).
- `trainers/src/syntide_trainers/torch_loop.py`, about 150 lines:
  - `ResNet1d` (~1M params, 12 → 5).
  - `predict(checkpoint, x, device)`, which satisfies the `Predictor` Protocol. Phase 3 calls it for the test split and the label audit.
  - `train(run_dir, cfg, tracker_cfg)`:
    1. Call `prepare(...)`, and skip the trial if `is_complete`.
    2. Call `torch.manual_seed(spec.init_seed)`, build the model and hash its initial state (`init_sha256`).
    3. Train with AdamW + BCE over `trial.batches()`, logging `train/loss` every `log_every` steps.
    4. At each `val_every`, compute `val_scores` on `trial.val`, log them, and save `best.pt` when `val/auroc_overall` improves.
    5. Save `final.pt` and call `trial.finish(...)`.
  - `main()`: argparse over run-dir globs, with `--config`, `--tracker conf/tracker/<backend>.yaml` and `key=value` overrides through `OmegaConf.from_dotlist`. It does not use Hydra.
- A Lightning or Hydra trainer would be another package that makes the same calls. Nothing in syntide changes for it.

### Outputs per trial (Phase 3's input contract)
- **Written by SYNTIDE:** `trial.json` (at planning), plus `recipes.parquet` and `exposure.json` (from the stream, at `finish`).
- **Written by the trainer:** its checkpoints (`best.pt` and `final.pt` for the reference trainer) and the tracker output (`metrics.jsonl`, TensorBoard events or `wandb/`).
- **Written by `finish`, from what the trainer hands over:**
  - `predictions_val.parquet`: `trial_id, arm, regime, seed, checkpoint, step, record_id, subject_uid, class_id, y_true, y_prob, is_artifact`;
  - `result.json` (RunResult):
    - trial info: status, ids, best_step, best_val_auroc, final_step, val_history;
    - exposure: presentations, transform_slots, candidates, accepted, exhausted_slots, distinct_recipes;
    - trainer: name, version, config, device, predictor, fingerprint, and `init_sha256`;
    - timing: wall_time_s;
    - provenance: manifest, source and calibration shas; registry, dataset and paths; checkpoint shas;
    - env.

No test-split access anywhere in Phase 2.

### Tests (tests/generate, tests/test_trial.py, trainers/tests), using synthetic arrays and CPU (no TimeNet)
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
6. **Pairing:** with the same seed, A0–A3 share parent indices, slot masks and `init_seed`, and A2/A3 share `u` while their controls differ. A0's batches equal their parents. A new seed gives a new schedule.
7. **Run contract** (`tests/test_trial.py`: a fake NumPy trainer through `open_trial`, with no torch):
   - presentations = steps × batch;
   - transform_slots = Σmask (0 for C1);
   - the ledger holds only rows with step ≤ max_steps;
   - the predictions cover both checkpoints × all records × 5 classes;
   - `finish` refuses (parametrized) a partly consumed stream, a wrong final step, a best step that is not the argmax, an unreproducible val score, and a different trainer in the same dir.
8. **Trial identity and reuse:** the id ignores trainer settings, logging settings and paths, and changes with seed, max_steps, manifest and calibration. `is_complete` reports a finished trial, and a different trial in the same dir is refused.
9. **Plan boundary** (parametrized): each of these is refused before any `trial.json` is written: a missing `max_steps`, a transform arm on a non-depleted role, a non-divisible `max_steps`, and an editor arm without a passed gate.
10. **Reference trainer** (`trainers/tests/test_torch_loop.py`; CPU, 30 steps, val_every 10, synthetic `Samples`): `finish` accepts the run, `final.pt` is at step 30, and `predict(best.pt)` reproduces `val_probs["best"]`.

**Done when:**
1. The tests pass, and `syntide` imports neither `lightning` nor `syntide_trainers`.
2. The pilot sets a numeric `max_steps` for both regimes in the protocol YAML. It is planned with `mode=plan runs_dir=runs/pilot select.arms=[A1] select.regimes=[scarcity25] select.seeds=[0] protocol.regimes.scarcity25.max_steps=20000`, then trained by the reference trainer on that run dir.
3. The model gate runs (width 128 and 2× steps, in separate runs dirs).
4. The gap-gate subset runs: `mode=plan select.arms=[A0,C1] select.regimes=[scarcity25] select.seeds=[0,1,2]`, then the trainer.
5. `python -m syntide.generate` passes Gate 4 for both regimes.
6. The full plan runs, and the trainer skips the 6 completed trials.

---

## Phase 3 — Evaluation, test-once discipline, dashboard views, results for the presentation

> **Status (2026-10-22): built** (see the README, "Phase 3"). Where it differs from the list below:
> - The test fold is frozen, claimed and scored by `syntide.eval.holdout`. `eval/freeze.py` freezes only the
>   decision rules (`runs/<protocol>/eval/rules.json`), and the test report checks they were frozen before the claim.
> - The decision rules are `conf/analysis/<protocol>.yaml` (schema `AnalysisConfig` in `eval/analysis.py`), chosen
>   with `protocol=`. There is no `eval/config.py`.
> - The paired statistics are in `eval/contrasts.py`. `eval/stats.py` keeps only the selection metric, which the
>   trainer seam imports.
> - An arm can be named in another regime (`C2@full100`). Gap closed needs its denominator's CI above 0, not only
>   its estimate.
> - The rules were frozen for `ptbxl-artifact-50` (one 50% regime plus the 100% reference), which replaced the
>   10%/25% grid below.

### Modules
- `eval/config.py` + `conf/evaluate.yaml`: `EvaluateConfig` with:
  - `runs_dir`, `split` (val | test), `checkpoint` (best | final), `mode` (evaluate | freeze | audit);
  - `tracker`: `defaults: [tracker: tensorboard]`, and `tracker=files` or `tracker=wandb` switches the dashboard;
  - `n_boot=2000`, `ci_level=0.95`, `bootstrap_seed`;
  - **primary** `[A3, A1]` (artifact slice, scarcity25);
  - **fixed sequence** `[[A3,A2],[A2,A1]]`, then the same chain at scarcity10;
  - diagnostics `[[A1,A0],[C1,A0],[C2,A0]]`;
  - `practical_target=0.02`, `noninferiority_margin=-0.01`, `gap_gate: {contrast: [C1,A0], regime: scarcity25, min_effect: 0.02}`.
  - Contrast labels follow timenet_syntide §3A: A2−A1 is "two recipes", not "realism".
- `eval/runs.py`: `load_run(run_dir)` re-checks the run contract, whichever trainer produced the run:
  - `trial.json` and `result.json` agree;
  - the checkpoint shas match;
  - the predictions cover both checkpoints × all val records × 5 classes.
- `eval/stats.py` (pure):
  - `macro_auroc(y, p, weights)`: classes without both weighted labels are skipped, so the result never contains NaN.
  - `val_scores(val, probs)`: the checkpoint-selection metric that trainers and `finish` use (Phase 2A).
  - `paired_contrast(...) -> UtilityReport`:
    - `PairingError` when any of these holds:
      - the seeds, records or labels differ between arms;
      - the compared runs have different trainer fingerprints;
      - `init_sha256` differs within a (regime, seed) block;
      - presentations differ within a regime;
      - transform slots differ across A0–A3;
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
  - `write_freeze` records the completed trial ids, checkpoint shas, trainer fingerprints and predictor paths, manifest and calibration shas, a hash of the decision rules, and the git rev;
  - `verify_freeze` raises `FreezeError` naming each change;
  - `claim_test_evaluation` is a one-shot marker (`open(path, "x")`).
- `eval/tracking.py`: the dashboard backends, which trainers and eval share.
  - `TrackerConfig(backend="files"|"tensorboard"|"wandb", project="syntide", entity=None, mode="online")` loads from `conf/tracker/{files,tensorboard,wandb}.yaml`. `TrackerConfig.from_yaml(path)` serves trainers that don't use Hydra.
  - `Tracker` Protocol: `log_scalars(values, step)`, `log_figure(name, fig, step)`, `log_text(name, md, step)`, `log_summary(hparams, metrics)` and `close()`.
  - `make_tracker(cfg, log_dir, run_name, group, config)` imports the backend lazily. An unknown backend name raises an error that lists the choices.
    - `files`: `metrics.jsonl` (one `{"step", **values}` line per call), `figures/*.png`, `text/*.md` and `summary.json`. No extra dependencies.
    - `tensorboard`: `torch.utils.tensorboard.SummaryWriter(log_dir)` with `add_scalar`, `add_figure`, `add_text` and `add_hparams`.
    - `wandb`: `run = wandb.init(project, entity, group=<protocol id>, name="<regime>/<arm>/seed<n>", dir=log_dir, config=..., mode=...)`, then `run.log(..., step=)`, `wandb.Image(fig)`, `run.summary.update`, and `run.finish()` in `close()`. Using the returned run object lets one process train several run dirs in turn.
  - Metric names follow results_visualization.md, and the step axis is optimizer updates.
  - The evidence files are always written, whatever the backend: predictions, `result.json`, the ledger, `comparisons.json`, `results.md` and the slide PNGs. The backend only decides where the dashboard goes.
- `eval/views.py` (functions taking a `Tracker`; untested presentation code; every figure is also saved as PNG under `figures/` for slides). The five views from results_visualization.md:
  - **Training health:** comes from the trainer's and the stream's scalars.
  - **Downstream utility:** `log_paired_effects`, a forest plot per regime with artifact and clean panels, CI and seed dots, and reference lines at 0/+0.02 and 0/−0.01.
  - **Coverage & acceptance:** `log_coverage` shows retained natural vs requested vs accepted severity per type and policy, acceptance rate by type, severity bin and rejection reason, plus distinct recipes per transformed presentation (diversity / memorization).
  - **Signal inspection:** `log_signal_examples` shows fixed development records (original / edited / residual), with consistent mV axes and the edited interval shaded.
  - **Failure slices:** `log_slice_differences` gives per-class Δ with counts.
  - Plus `log_hparams` (one row per trial, through `log_summary`) and `log_findings` (markdown, through `log_text`).
- `eval/report.py`: `results.md`, holding:
  - the MVE's **one results table** (arm × artifact 25% / clean 25% / Δ vs A1 [CI] / gap closed / artifact 10%, with seed SD per arm);
  - the findings flags and the fixed-sequence outcome;
  - gate outcomes 1–5, read from their JSON files;
  - run counts and compute time.
- `eval/__main__.py`:
  - `mode=audit`: coverage and examples from `gate_recipes.parquet`, plus a **label audit** (report-only). The audit takes the mean |Δp| and the flip rate of A0's best checkpoint (run through its `trainer.predictor`) between each parent and its edit over the gate recipes, which shows whether edits erase diagnostic evidence.
  - `mode=freeze`: writes `freeze.json`.
  - `split=val`: reports, dashboard views and `results.md` from `predictions_val.parquet`.
  - `split=test`:
    1. `verify_freeze` + `claim_test_evaluation`.
    2. SYNTIDE loads fold 10 itself through the Phase 1 adapter.
    3. For each **frozen** run, it imports the run's `trainer.predictor` and writes `predictions_test.parquet`.
    4. It produces the same reports.

    The trainer never sees the test split.

### Tests (tests/eval and tests/test_seams.py)
1. **Paired arms cancel:** identical predictions give estimate 0 and a CI of exactly [0, 0].
2. **Known effect detected:** the CI excludes 0, and every seed's effect is reported.
3. **Pairing guard:** each of these raises `PairingError`: differing seeds, records or labels; a different trainer fingerprint; a different `init_sha256` within a (regime, seed) block.
4. **Degenerate class:** it is excluded, and the result is finite.
5. **Decision rules:**
   - CI [0.003, 0.015] sets both `positive_effect_supported` and `target_effect_ruled_out`;
   - a clean estimate of −0.005 with ci_low −0.02 does not support non-inferiority;
   - the gap gate is inclusive;
   - the fixed sequence stops at the first failure;
   - gap closed is withheld when the gate fails or the denominator ≤ 0.
6. **Freeze guard** (parametrized): a changed checkpoint, rule or calibration is rejected, and a second test claim is refused.
7. **Seam test:** a tiny plan → reference-trainer run of A0 and A3 produces run dirs that `load_run` and `paired_contrast` accept, and its ledger parents are all in `depleted`.
8. **Tracking backends** (`tests/eval/test_tracking.py`):
   - each backend accepts the same call sequence;
   - the `files` output is read back, and TensorBoard events are read with `EventAccumulator`;
   - `wandb` runs with `mode=offline` in a temp dir, and is skipped when wandb is not installed;
   - each `conf/tracker/*.yaml` loads into `TrackerConfig`.

**Done when:** the tests pass; `mode=audit` and `split=val` populate all five views on the configured dashboard, and `results.md`; the test-once flow works on a throwaway runs dir (a second `split=test` is refused); and the real `split=test` is run exactly once after freezing.

---

## Phase seams
| From → to | Contract |
|---|---|
| Phase 1 → 2 | Registry `physionet/ptb-xl@1.0.3`; `manifests/<protocol>/pool{10,25}.json`; `PtbXlAdapter.load → Samples`; `load_manifest`, `check_manifest`, `check_calibration_ids` |
| SYNTIDE → trainer | `runs/<protocol>/<regime>/<arm>/seed<n>/trial.json` (`TrialSpec`); `syntide.trial.prepare → Trial` (`batches()`, `val`, `tracker`); `syntide.eval.stats.val_scores` |
| Trainer → SYNTIDE | `Trial.finish(TrainerInfo, best, final, val_history, val_probs, init_sha256)`; the checkpoints; a `Predictor` named in `TrainerInfo.predictor` |
| Phase 2 → 3 | Run dirs (`trial.json`, `result.json`, `predictions_val.parquet`, `recipes.parquet`, `exposure.json`, checkpoints); calibration + gate JSON + `gate_recipes.parquet`; each run's `trainer.predictor`; `edit` |
| Phase 3 → presentation | `results.md`, `figures/*.png`, `comparisons.json` (UtilityReports), the dashboard (files, TensorBoard or W&B) |

## Schedule against the brief's 3-day budget, and deliverables
| Day | Build | Run / gate |
|---|---|---|
| 1 | Phase 1; Phase 2A; Phase 2C (reference trainer) | Gate 1; pilot + model gate (Gate 2); gap-gate runs A0/C1 × 3 seeds |
| 2 | Phase 2B; Phase 3 stats + views | Gap gate verdict (Gate 3; MVE fallback: retain 0% and retest once); Gate 4; remaining 66 trials (≈ 8–13 h on the M1 Max) |
| 3 | Phase 3 report | Freeze (Gate 5) → test once → `results.md` + figures → slides for the 45-minute discussion |

Deliverables:
- the code repo, with a README covering how to reproduce each gate;
- `results.md` and figures;
- slides covering the design (from the architecture and MVE docs) and the experiment;
- a notes section filled from the actual findings: what evidence would change the conclusion, and next week's plan with a larger cluster (from MVE §Plan).

## Out of scope (future plugins at these seams)
TimeNet exporter (provenance lives in the ledger until then), `electrodes_problems` artifact type, electrode-attribution descriptor, joint multi-control planning, validation-error slices in the gap detector, two-sample plausibility test, blinded gap discovery, NSTDB stress test, learned or text-conditioned generators, Plotly export, hydra-zen, Ray/Slurm launchers, and a second trainer (for example Lightning) on the same seam.

## Risks
- Hydra 1.3 crashes on Python 3.14, because its help object fails the stricter argparse. The project is therefore pinned to 3.13 (`.python-version`, `requires-python`).
- Retained artifact counts are small (≈ 37 / 93 split over 3 types), so the per-type quantile maps are noisy. A KDE floor `ε` and clipping to the envelope regularize them; the numbers are reported in the gap report.
- Descriptors measured on natural ECGs mix with physiological content (T waves in the wander band, QRS in the high-frequency band). The robust estimators limit this, and the coverage view shows the residual bias.
- TimeNet per-lead reads may be slow; measure, and add the `.npy` cache only if needed.
- MPS is not bitwise deterministic. Pairing holds at the RNG and data level. The device is part of the trainer fingerprint, which compared runs must share.
- A trainer could ignore `init_seed`, or use a different config per arm. Eval catches both, because `init_sha256` and the trainer fingerprint must match across compared runs.
- W&B ignores logged steps that go backwards within a run, so each trial and each eval split is its own W&B run.
- Disk: 1.84 GB download + about 3 GB extracted + the registry.

## Verification (end to end)
```bash
uv sync --all-packages && uv run ruff check && uv run pytest tests trainers/tests   # add --extra wandb for W&B
uv run python -m syntide.data                                    # registry + manifests (Gate 1)
TIMENET_REGISTRY=datasets/timenet-registry uv run timenet info physionet/ptb-xl
# Pilot: plan one A1 trial with a trial budget, then train it with the reference trainer
uv run python -m syntide.generate mode=plan runs_dir=runs/pilot select.arms=[A1] select.regimes=[scarcity25] \
  select.seeds=[0] protocol.regimes.scarcity25.max_steps=20000
uv run python -m syntide_trainers.torch_loop "runs/pilot/ptbxl-artifact-v2/scarcity25/A1/seed0"
# Gap-gate runs
uv run python -m syntide.generate mode=plan select.arms=[A0,C1] select.regimes=[scarcity25] select.seeds=[0,1,2]
uv run python -m syntide_trainers.torch_loop "runs/ptbxl-artifact-v2/scarcity25/*/seed*"
uv run python -m syntide.generate                                # mode=calibrate: calibration + Gate 4 + review sample
uv run python -m syntide.eval mode=audit                         # coverage, examples, label audit
uv run python -m syntide.generate mode=plan                      # full 72-trial plan
uv run python -m syntide_trainers.torch_loop "runs/ptbxl-artifact-v2/*/*/seed*"   # skips the 6 completed trials
uv run python -m syntide.eval split=val                          # tracker=files|tensorboard|wandb picks the dashboard
uv run python -m syntide.eval mode=freeze && uv run python -m syntide.eval split=test   # once
uv run tensorboard --logdir runs                                 # with the tensorboard backend
```
