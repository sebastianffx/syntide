# SYNTIDE — three-phase implementation plan

## Context

SYNTIDE (docs/SYNTIDE_Architecture.md, docs/SYNTIDE_MVE.md) asks: when artifact ECGs are scarce, does targeted synthetic editing beat standard augmentation on real artifact-bearing ECGs? The repo has only docs, a hello-world `main.py` and `timenet[cli,torch]==0.11.0` in `.venv`. Before any generator work, the engine needs a data plane, a controlled training harness and a credible evaluator.

This plan builds those three in order. Each phase stands alone: its own package, tests and file outputs. Each phase's outputs are the next phase's inputs. Once all three are done, the engine can run the arms that don't need the artifact editor (A0, A1, C1, C2). It can also check the MVE's leakage gate and gap gate, and support the model gate and the step-budget pilot. The editor (A2/A3), descriptors, gap detector, ledger and exporter come later and plug into the seams defined here.

Verified facts this plan depends on:
- `timenet.engine.run_pipeline(connector, root, *, cache_dir, keep_cache, force)` builds a connector straight from Python. `timenet-connectors==0.11.0` provides `BasePhysioNetConnector`, which reads format-16 `.dat` files with NumPy and parses `.hea` headers. Its ECG-QA CoT connector is a working PTB-XL template. It uses `ptbxl-{ecg_id}` record ids and a `split` annotation.
- TimeNet 0.11.0 has **no split API**, and `Record.subject_ids` is not stored. Patient and fold must be stored by us. `Record.metadata` (JSON) and annotations (bool/int/float/str) round-trip.
- Lead Signals come back **sorted by name**, not in clinical order. The adapter must reorder them.
- All deps resolve on Python 3.14: hydra-core 1.3.7, tensorboard 2.21, scikit-learn 1.9, matplotlib 3.11, timenet-connectors 0.11.0. `timenet-connectors` pulls `timenet[build]` (pyyaml, jsonschema), which is needed to load the dataset card.
- PTB-XL 1.0.3 zip is at `https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip` (1.84 GB, plain HTTPS, so no boto3).

## Cross-cutting decisions

| Decision | Choice | Why |
|---|---|---|
| Package | `src/syntide/` with uv build backend (`module-name = "syntide"`); delete `main.py` | One importable package, three subpackages = three phases |
| TimeNet boundary | Only `syntide.data` imports `timenet`. Pin `timenet[torch]==0.11.0`, `timenet-connectors==0.11.0` | timenet_syntide.md §7: isolate TimeNet so API drift is a one-package change |
| Config | Hydra 1.3 + one structured dataclass per entry point (registered in `ConfigStore`) + small YAML groups in `conf/` | training_loop doc §1/§3. Hydra stays at the entry points; library code receives plain dataclasses |
| Training loop | **Lightning `Trainer`** (`lightning==2.6.*`), `TensorBoardLogger` (its `.experiment` is a `torch.utils.tensorboard.SummaryWriter`) | Chosen by user; training doc §1. Used as a plain trainer only: no DataModule, no custom callbacks. Pairing and exposure stay in SYNTIDE code |
| Sampling rate | 100 Hz (`records100`, 1,000 samples × 12 leads) | PTB-XL benchmark convention; 5× cheaper than 500 Hz on the M1 |
| Labels | 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP), multi-label, from `scp_statements.csv` (`diagnostic == 1`) | MVE task; standard PTB-XL aggregation |
| No speculative abstractions | Concrete PTB-XL classes and functions; no plugin Protocols, registries or hydra-zen until a second dataset exists | User constraint |
| Dependencies | runtime: `timenet[cli,torch]==0.11.0`, `timenet-connectors==0.11.0`, `hydra-core>=1.3.7,<1.4`, `lightning==2.6.*`, `tensorboard`, `scikit-learn`, `matplotlib`; dev group: `pytest`, `ruff`, `wfdb` (test fixtures only) | Exact TimeNet pins; everything else at a minimum or minor-series pin |
| Tooling | `uv`, `ruff` (lint+format), `pytest` | |
| Git | Ignore `.timenet-registry/`, `.cache/`, `runs/`, `.DS_Store`. **Commit `manifests/`** (small JSON, part of the frozen protocol) | |

Final layout:
```
conf/  data.yaml  train.yaml  evaluate.yaml
       protocol/ptbxl_artifact_v2.yaml  arm/{A0,A1,C1,C2}.yaml
       regime/{scarcity10,scarcity25}.yaml  runtime/{mps,cpu}.yaml
src/syntide/
  data/      __main__.py  schema.py  adapter.py  challenge.py  leakage.py
             ptbxl/{__init__.py, connector.py, dataset.yaml}
  train/     __main__.py  config.py  model.py  mixer.py  trial.py
  eval/      __main__.py  config.py  stats.py  freeze.py  tensorboard.py
tests/  data/  train/  eval/  test_seams.py
manifests/ptbxl-artifact-v2/pool10.json, pool25.json
```

---

## Phase 1 — PTB-XL as a TimeNet dataset, plus manifests and the leakage gate

**Step 0 (scaffold):**
- Rewrite `pyproject.toml`: deps above, uv build backend with `module-name = "syntide"`, ruff and pytest config.
- Extend `.gitignore` and delete `main.py`.
- `uv sync`, then run the 3.14 smoke check (see Risks).
- `README.md` gets the commands from the Verification section and grows with each phase.

### Taken from timenet_syntide.md (basic items only)
- Isolate TimeNet imports in one integration package (§7).
- Typed, searchable facts go in annotations; non-typed provenance goes in namespaced `Record.metadata` (§7, row 1).
- Splits are assigned by SYNTIDE's frozen fold policy and stored on the Task, never inferred from the Record (§7, row 5).
- Pin the TimeNet version and add a round-trip contract test (§7).
- Global subject namespace `ptb-xl:<patient_id>`. Patient separation across the outer partitions. Explicit membership rules between retained, withheld and control manifests (§4A; training doc §5 "narrow the leakage assertion").
- **Deferred:** task-evidence compiler (§1), validation layers (§2), derivation graph and input-view policy (§4), coverage planning (§5), transfer ladder (§6), exporter corrections (§7, rows 2–4 and 6).

### Files
- `data/ptbxl/connector.py`: `PtbXlConnector(BasePhysioNetConnector[PtbXlSource])`, modelled on the ECG-QA CoT connector.
  - `download_async(cache_dir)`: `ensure_archive(PTBXL_URL, cache_dir)` → `find_dir_containing(..., "ptbxl_database.csv")`. Idempotent through the archive marker.
  - `convert(refs)`: one Record per `ecg_id`, mapped as in the table below. Uses the CSV's `filename_lr` column (no path arithmetic). Uses the inherited `_read_header` and `_lead_loader` (lazy, mV, float32). `CONNECTOR = PtbXlConnector`.
  - Small pure helpers in the same file: `diagnostic_superclasses(scp_codes, statements)` and `artifact_types(row)`. A noise column counts only if it is non-empty after `strip()`.
- `data/ptbxl/dataset.yaml`: `physionet/ptb-xl` 1.0.3, CC-BY-4.0, domains cardiology/health, tags ecg/physionet, source_url. The card schema forbids extra keys.
- `data/schema.py` (no timenet import, so Phase 2 can import it):
  - constants `LEADS = (I, II, III, aVR, aVL, aVF, V1…V6)` and `SUPERCLASSES`;
  - `subject_uid(patient_id) -> "ptb-xl:<id>"`;
  - frozen dataclass `Samples(record_id, subject_uid, is_artifact, labels[n,5] bool, x[n,12,T] float32)`.
- `data/adapter.py`: `PtbXlAdapter(registry, dataset="physionet/ptb-xl@1.0.3")`.
  - `index() -> pa.Table` builds one row per *labelled* record: `record_id, subject_uid, strat_fold, is_artifact, labels`. Built from `TimeFReader.annotation_table("Record")` + `target_table()`, columnar, with no signal reads.
  - `load(record_ids) -> Samples` uses `iter_records(ids)` and reorders leads by name into `LEADS`.
  - `source_sha256()` hashes the sorted `(path, checksum)` pairs of `manifest.files`. This is content-only; the hash of `manifest.json` itself changes with `build_env`.
- `data/challenge.py`: `ChallengeSpec` dataclass (also the Hydra schema node). `build_manifest(index, spec, pool_fraction) -> Manifest`. `save_manifest` / `load_manifest`; `load_manifest` recomputes the sha256 and rejects edited files.
- `data/leakage.py`: `LeakageError`, `check_manifest(manifest, subject_of)` (Gate 1).
- `data/__main__.py` + `conf/data.yaml`: `python -m syntide.data`. It builds the registry if missing (`run_pipeline(PtbXlConnector(), registry, cache_dir=download_cache, keep_cache=True)`), then writes and checks `pool10.json` and `pool25.json`. `hydra.output_subdir: null`.

### PTB-XL → TimeNet mapping
| PTB-XL | TimeNet |
|---|---|
| `ecg_id` | `Record(record_id="ptbxl-{ecg_id}")`, matching ECG-QA CoT ids for later cross-dataset leakage checks |
| 12 leads, `records100` | 12 lazy `Signal`s, spec `ecg` (mV), ids `ecg-{ecg_id}-{lead}`, `source_id="ptbxl-{ecg_id}"`, one `Source` |
| `patient_id`, `strat_fold` | int record annotations |
| `static_noise`, `burst_noise`, `baseline_drift`, `electrodes_problems` | str record annotations (stripped lead description), only when present (annotations can't be empty) |
| `scp_codes` (dict) | `Record.metadata["ptbxl.scp_codes"]` (dicts are not allowed as annotation values) |
| diagnostic superclasses | one `ClassificationTask(targets=sorted superclasses, target_schema="ptbxl.diagnostic_superclass", metadata={"split": train/validation/test})`. Only emitted when ≥1 superclass exists (~400 records have none; they stay as records without a task) |

### Challenge manifests (one JSON per pool fraction)
Fields: `challenge_id`, `spec` (folds, pool_fraction, retain, seed), `source` (dataset, version, source_sha256), `roles`, `sha256`.

How the roles are built:
- Outer partitions come from folds: `train` = folds 1–8, `val` = 9, `test` = 10. The fold mapping lives in config.
- `pool`: the first `⌈fraction × n_subjects⌉` train **subjects** after one seeded shuffle. All of a subject's records go together, and the pools are nested (10% ⊂ 25%). `unused` = train − pool.
- `retained_artifact` = `⌈retain × n_artifact_in_pool⌉` random artifact records. `withheld` = the remaining artifact records in the pool.
- `depleted` = pool − withheld. These are the parents for A0–A3.
- `natural_mix` = pool minus `|withheld|` records dropped uniformly at random (C1). `pool` itself is C2.

The leakage gate checks:
- the pool, val and test subjects are disjoint;
- every labelled record sits in exactly one outer partition;
- retained and withheld split the pool's artifact records exactly;
- `withheld ∩ depleted = ∅`;
- `natural_mix ⊆ pool` and `|natural_mix| = |depleted|`.

### Tests (tests/data)
A session fixture writes 4 synthetic format-16 WFDB records with `wfdb.wrsamp` (the pattern from the ECG-QA tests), plus `ptbxl_database.csv` and `scp_statements.csv`. Two records share a patient, one has `burst_noise`, one has only rhythm codes, one has a whitespace-only noise string, and the folds vary. The fixture then runs `convert` → `store_dataset` → `PtbXlAdapter`.
1. **Signal fidelity:** values come back in mV (within one ADC step) and in clinical lead order, even though TimeNet returns leads sorted by name.
2. **Label and task rules:** rhythm codes are ignored, superclasses are deduplicated, a record with no superclass has no task and is absent from `index()`, and the task split follows the fold rule.
3. **Index semantics:** `subject_uid` is namespaced, and `is_artifact` is true only for the record with a real noise string. A whitespace-only string counts as clean.
4. **Manifest construction** (synthetic index of about 1,500 records): no subject is split across the pool boundary, pool10 ⊆ pool25, the retained/withheld counts follow the ceil rule, depleted excludes withheld, and natural_mix matches depleted's size while keeping more artifact records.
5. **Determinism and integrity:** the same spec gives the same sha, a new seed gives a different pool, and a hand-edited manifest is rejected on load.
6. **Leakage gate** (parametrized over each violated invariant): each violation raises `LeakageError` with a message naming the offending subject or record.

**Done when:** `uv run pytest tests/data` passes. `uv run python -m syntide.data` builds the real registry and both manifests. The counts it logs are close to the MVE's (pool25 ≈ 928 artifact records, ≈ 93 retained; test artifact slice ≈ 513 ECGs / 477 patients). Loading the pool25 depleted set takes under about a minute; if it is slower, add an `.npy` cache inside `load()` keyed by `source_sha256`.

---

## Phase 2 — Training harness (Hydra + structured dataclasses + small YAML groups)

### Config (`train/config.py`, `conf/`)
Dataclasses use OmegaConf-safe types (primitives, `list`, `dict`, `Optional`, nested dataclasses):
- `ProtocolConfig`: id, registry, dataset, batch_size=64, val_every=250, transform_fraction=0.5, `ModelConfig` (width, blocks, kernel), `OptimConfig` (AdamW lr, weight_decay), `GenericAugConfig` (A1 severity envelope: gain range, shift_mv, noise_std_mv, wander_mv, wander_hz).
- `ArmConfig`: id, `source_role` (depleted | natural_mix | pool), `transform_family` (none | generic).
- `RegimeConfig`: id, manifest path, `max_steps: int = MISSING`.
- `RuntimeConfig`: accelerator (mps | cpu), deterministic, log_every_n_steps.
- `TrainConfig`: protocol, arm, regime, runtime, replicate_seed. Registered as `train_schema`.

Budgets live in the **regime** group, so every arm in a regime has the same budget by construction; no budget-equality check is needed.

`conf/train.yaml`:
- defaults: `[train_schema, protocol: ptbxl_artifact_v2, arm: A0, regime: scarcity25, runtime: mps, _self_]`;
- `hydra.run.dir` and the sweep `dir`/`subdir` both resolve to `runs/${protocol.id}/${regime.id}/${arm.id}/seed${replicate_seed}`, so a single run and a sweep land in the same directory;
- `hydra.sweeper.params`: `arm: A0,A1,C1,C2`, `regime: scarcity10,scarcity25`, `replicate_seed: 0,1,2,3,4,5`.

Regime YAMLs ship `max_steps: ???` until the pilot sets them.

### Modules
- `train/__main__.py` (the only Hydra code):
  1. `OmegaConf.to_object(cfg)` → `TrainConfig`. A missing value fails here.
  2. `load_manifest` + `check_manifest` (Phase 1, Gate 1).
  3. `validate(spec, manifest)`:
     - the arm's role exists;
     - any `transform_family != none` requires `source_role == depleted`;
     - `max_steps % val_every == 0`, so the last validation happens at the final update.
  4. `PtbXlAdapter.load(...)` for the arm role and `val`.
  5. `run_trial(...)`.
- `train/model.py`:
  - `ResNet1d(ModelConfig)`: about 1M params, 12 → 5 logits. `width` doubles for the model gate.
  - `EcgModule(L.LightningModule)` with `save_hyperparameters` of the model, optim and slice settings:
    - `training_step`: BCE-with-logits. Adds the batch size to the presentations count and Σmask to the transform-slots count. Logs `train/loss`, `budget/sample_presentations` and `budget/wall_time_seconds`.
    - `validation_step` collects probabilities. `on_validation_epoch_end` computes sklearn macro-AUROC overall/artifact/clean, logs `val/auroc_*` and `val/loss`, and appends `(global_step, auroc_overall)` to `val_history`.
    - `configure_optimizers`: AdamW.
  - `load_model(ckpt) -> ResNet1d` (via `EcgModule.load_from_checkpoint`) and `predict_proba(model, x, device)` are plain functions that Phase 3 reuses.
- `train/mixer.py` (the small custom paired stream from training doc §1 and §4A):
  - `rng(seed, stream, *keys)` = `np.random.default_rng([seed, stream, *keys])`. Separate streams for INIT, PARENTS, SLOTS and TRANSFORM, keyed by `(replicate_seed, step, …)` and **never the arm id**.
  - `batch_plan(n_parents, step, batch, seed, fraction) -> (indices, slot_mask)`: a new seeded permutation for each pass over the parents, and a slot mask drawn per step.
  - `generic_per_lead(x, rng, cfg)`: A1's gain, shift, Gaussian noise and sinusoidal wander, drawn for all slots and applied only to masked ones, so the draws don't depend on the mask. `none` = identity (A0's parents repeat in transform slots).
  - `PairedStream(IterableDataset)` yields `(x, y, mask)` for step 1, 2, … from `batch_plan` + transform, already batched. It is fed through `DataLoader(batch_size=None, num_workers=0)`. With 0 workers the batch order *is* the step order. Lightning's one-batch prefetch is harmless because each batch is a pure function of its step.
- `train/trial.py`:
  - `trial_id(spec, manifest_sha, source_sha)`: sha256 of the scientific inputs plus `syntide.__version__` and `runtime.accelerator`. It excludes logging settings and paths.
  - `run_trial(spec, train: Samples, val: Samples, out_dir) -> RunResult`:
    1. If `result.json` is already completed with the same id, return it (so gap-gate runs count toward the grid). If it holds a different id, refuse.
    2. `L.seed_everything` from the INIT stream before building the model (same initial weights across arms).
    3. Create `L.Trainer` with: `max_steps`; `val_check_interval=val_every`; `check_val_every_n_epoch=None`; `accumulate_grad_batches=1`; `num_sanity_val_steps=0`; `ModelCheckpoint(dirpath=out/checkpoints, filename="best", monitor="val/auroc_overall", mode="max", save_top_k=1, save_last=False)`; `TensorBoardLogger(save_dir=out_dir, name="", version="")` so the events sit in the run dir; and the runtime's accelerator and deterministic settings.
    4. `trainer.fit(module, train_dataloaders=stream_loader, val_dataloaders=val_loader)`.
    5. Then `trainer.save_checkpoint(out/checkpoints/final.ckpt)` explicitly. This is the true final update, because `save_last` is not guaranteed to be (training doc §4B).
    6. Read the best step from the best checkpoint's `global_step`.
    7. Run `predict_proba` with the best and final weights on val, and write the predictions.
- TensorBoard scalar names follow results_visualization.md, and Lightning's `global_step` (= optimizer updates) is the x-axis.

### Outputs per trial (Phase 3's input contract)
`runs/<protocol>/<regime>/<arm>/seed<k>/` holds:
- `.hydra/config.yaml`, written by Hydra;
- the TensorBoard event files;
- `checkpoints/{best,final}.ckpt`;
- `predictions_val.parquet`, in long format: `trial_id, arm, regime, seed, checkpoint, step, record_id, subject_uid, class_id, y_true, y_prob, is_artifact`;
- `result.json`, holding RunResult:
  - status, trial_id, arm, regime, seed;
  - best_step, best_val_auroc, final_step;
  - presentations, transform_slots, wall_time_s;
  - manifest_sha256, source_sha256, registry, dataset, manifest path;
  - checkpoint sha256s;
  - env (python, torch, lightning, accelerator, git rev).

No test-split access anywhere in Phase 2.

### Tests (tests/train), using synthetic `Samples`, CPU and a tiny model (no TimeNet)
1. **Pairing:** with the same seed, A0 and A1 get identical parent indices, slot masks and initial weights over 50 steps. A0's batches equal their parents. A1 differs from them only in masked slots. A new seed gives a new schedule.
2. **Budget and checkpoints** (a real Lightning run on CPU; 30 steps, val_every 10): presentations = steps × batch; transform_slots = Σmask for A1 and 0 for C1. `best_step` is the argmax of `val_history`, `final.ckpt` has `global_step == 30`, and the predictions cover both checkpoints × all val records × 5 classes.
3. **Trial identity and reuse:** the id ignores logging settings and the output dir, but changes with seed, max_steps, manifest sha and accelerator. Re-running a completed trial returns the stored result without training. A different trial in the same directory is refused.
4. **Config boundary** (Hydra `compose` against `conf/`, parametrized): a missing `regime.max_steps`, a transform arm on a non-depleted role, and a `max_steps` not divisible by `val_every` are each refused before any data loads.

**Done when:**
1. The tests pass.
2. The pilot runs: `python -m syntide.train arm=A1 regime=scarcity25 regime.max_steps=20000 hydra.run.dir=runs/pilot/A1-25`. Read the plateau in TensorBoard, then write the numeric `max_steps` into both regime YAMLs.
3. The gap-gate subset runs: `python -m syntide.train -m arm=A0,C1 regime=scarcity25 replicate_seed=0,1,2`.
4. Re-running that command skips all 6 completed trials.
5. Model gate: run the same subset with `protocol.model.width=128` and with a 2× `regime.max_steps`, each under its own `hydra.run.dir`.

---

## Phase 3 — Evaluation, test-once discipline, TensorBoard views

### Modules
- `eval/config.py` + `conf/evaluate.yaml`: `EvaluateConfig` with:
  - `runs_dir`, `split` (val | test), `checkpoint` (best | final, where final is the sensitivity check), `mode` (evaluate | freeze);
  - `n_boot=2000`, `ci_level=0.95`, `bootstrap_seed`;
  - `contrasts: [[A1,A0],[C1,A0],[C2,A0]]`. A3−A1, A3−A2 and A2−A1 are added when the editor exists. Labels follow timenet_syntide §3A, so A2−A1 is called "two recipes", not "realism".
  - `practical_target=0.02`, `noninferiority_margin=-0.01`;
  - `gap_gate: {contrast: [C1,A0], regime: scarcity25, min_effect: 0.02}`.
- `eval/stats.py` (pure, no I/O):
  - `macro_auroc(y, p, weights)`: sklearn per class, with patient weights. A class without both weighted labels is skipped, so the result never contains NaN.
  - `paired_contrast(preds, a, b, regime, slice) -> UtilityReport`:
    - Pairing guard: both arms must have the same seeds, the same record set and identical labels, or it raises `PairingError`.
    - Estimate = mean over seeds of the per-seed difference.
    - Crossed bootstrap: patients are resampled as multinomial multiplicities, which become record weights, and seed ids are resampled with replacement. The **same draw serves both arms**.
  - `UtilityReport`, a frozen dataclass: contrast, regime, split, slice, checkpoint, estimate, ci_low, ci_high, per-seed effects, n_patients, n_records, n_seeds, n_boot, thresholds, findings.
  - `findings(report_artifact, report_clean, cfg)`: three separate flags (timenet_syntide §3B; training doc §5):
    - `positive_effect_supported` (artifact ci_low > 0);
    - `target_effect_ruled_out` (artifact ci_high < 0.02);
    - `clean_noninferiority_supported` (clean **ci_low** > −0.01).
    - Plus `gap_gate_passed` (estimate ≥ 0.02) for the gap-gate contrast.
- `eval/freeze.py` (Gate 5):
  - `write_freeze(runs_dir, cfg)` writes `freeze.json`: completed trial ids, checkpoint sha256s, manifest shas, a hash of the decision rules, and the git rev.
  - `verify_freeze(...)` raises `FreezeError` naming each change.
  - `claim_test_evaluation(runs_dir)` creates a one-shot marker with `open(path, "x")`. A second test evaluation raises.
- `eval/tensorboard.py` (functions that take a `SummaryWriter`; untested presentation code):
  - `log_paired_effects`: a Matplotlib forest plot per regime with artifact and clean panels, a dot and CI per contrast, and seed dots. Reference lines at 0 and +0.02 on the artifact panel, and at 0 and −0.01 on the clean panel. Sent with `add_figure`.
  - `log_slice_differences`: per-class Δ with positive counts. Diagnostic only.
  - `log_hparams`: one row per trial (arm, regime, seed → slice AUROCs).
  - `log_findings`: a markdown table via `add_text`.
  - `log_signal_examples`: fixed development records showing original / A1-edited / residual, with consistent mV axes.
- `eval/__main__.py`:
  - `mode=freeze` writes `freeze.json`.
  - `split=val` globs `predictions_val.parquet`, computes the reports for each contrast × regime × slice, and writes `comparisons.json` and TensorBoard output to `runs_dir/evaluation/<split>-<checkpoint>/`.
  - `split=test` runs `verify_freeze` and `claim_test_evaluation`, then runs inference on fold 10 for the **frozen** trials only, using Phase 1's adapter and Phase 2's `load_model` / `predict_proba`. It writes `predictions_test.parquet` and then evaluates as above.

### Tests (tests/eval and tests/test_seams.py)
1. **Paired arms cancel:** identical predictions give estimate 0 with a CI of exactly [0, 0]. An unpaired bootstrap would give a non-zero width, so this proves the same draw serves both arms.
2. **Known effect detected:** near-perfect vs noisy predictions give a CI that excludes 0, with every seed's effect reported.
3. **Pairing guard:** a seed set, record set or labels that differ between arms raise `PairingError`.
4. **Degenerate class:** a class with no positives in the slice is excluded, and the result is finite.
5. **Decision rules:**
   - CI [0.003, 0.015] sets both `positive_effect_supported` and `target_effect_ruled_out`, the overlap case the docs flag.
   - A clean point estimate of −0.005 with ci_low −0.02 does not support non-inferiority.
   - The gap-gate threshold is inclusive.
6. **Freeze guard** (parametrized): a changed checkpoint or decision rule after freeze is rejected, and a second test claim is refused.
7. **Seam test:** Phase 2 outputs for A0 and A1 (tiny run) are accepted by Phase 3's `paired_contrast`, which proves pairing holds end to end.

**Done when:** the tests pass, and `python -m syntide.eval split=val` on the gap-gate runs shows the forest plot, the slice view, HParams and the gap-gate finding in `tensorboard --logdir runs`. The test-once flow is exercised on a throwaway runs dir: `mode=freeze` → `split=test`, and a second `split=test` is refused.

---

## Phase seams
| From → to | Contract |
|---|---|
| Phase 1 → 2 | TimeNet registry `physionet/ptb-xl@1.0.3`; `manifests/<protocol>/pool{10,25}.json` (roles + sha); `PtbXlAdapter.load(ids) -> Samples`; `load_manifest`, `check_manifest` |
| Phase 2 → 3 | Run dirs with `predictions_val.parquet` (long format above), `result.json`, `checkpoints/*.ckpt`; `load_model`, `predict_proba` |
| Phase 3 → later | `UtilityReport` / `comparisons.json`; contrast list ready for A2/A3; `log_signal_examples` reused for the editor audit |

## Out of scope (future plugins at these seams)
Electrode-space editor (A2/A3), descriptors, gap detector, control planner, validators, ledger (DuckDB/Parquet), TimeNet exporter, blinded gap discovery, NSTDB stress test, gap-closed ratio, Plotly export, hydra-zen, Ray/Slurm launchers.

## Risks
- Hydra 1.3 and Lightning 2.6 resolve on Python 3.14, but Lightning only declares support up to 3.13. The first implementation step runs a smoke check: import, Hydra compose, and a 2-step CPU `Trainer.fit`. If it fails, pin `.python-version` to 3.13.
- TimeNet per-lead reads may be slow for about 4k records. Measure in Phase 1 and add the `.npy` cache only if needed.
- MPS kernels are not bitwise deterministic. Pairing is guaranteed at the data/RNG level, and the device is recorded in the trial id.
- Disk: 1.84 GB download, about 3 GB extracted, plus the registry.

## Verification (end to end)
```bash
uv sync && uv run ruff check && uv run pytest
uv run python -m syntide.data                                   # registry + manifests (+ Gate 1)
TIMENET_REGISTRY=.timenet-registry uv run timenet info physionet/ptb-xl
uv run python -m syntide.train arm=A1 regime=scarcity25 regime.max_steps=20000 hydra.run.dir=runs/pilot/A1-25
uv run python -m syntide.train -m arm=A0,C1 regime=scarcity25 replicate_seed=0,1,2   # gap-gate runs
uv run python -m syntide.eval split=val                         # gap gate + forest plot
uv run tensorboard --logdir runs
```
