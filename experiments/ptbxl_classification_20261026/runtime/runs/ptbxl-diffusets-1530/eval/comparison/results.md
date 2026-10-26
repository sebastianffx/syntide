# DiffuSETS classification extension: 1,530 generated ECGs

All 27 old and new trials were checked with `load_run`; compared arms passed `check_pairing`, including trainer, initialization, exposure, held-out record, patient, label and quality-flag agreement. Original run files were read only.

G1 adds the 1,530 frozen accepted DiffuSETS ECGs to C2's 8,579 real ECGs. R1 adds the corresponding 1,530 real parent rows, retaining duplicate parents and matching G1's order and provisional label multiplicity. Each new arm has 10,109 rows.

All runs use seeds 0, 1 and 2, 6,500 optimizer steps, batch size 64, 416,000 presentations and validation every 100 steps. The best checkpoint is selected by validation overall macro-AUROC.

**Interpretation:** G1 − C2 is the requested exploratory comparison. G1 − R1 is the key matched-parent control; R1 − C2 describes the added parent information and reweighting. The full100 C2 arm is a descriptive real-data reference.

The 1,721-candidate snapshot includes 1,530 accepted and 191 rejected candidates. Accepted slots use 1,260 distinct conditioning ECGs from 1,239 patients. 717 accepted slots have parents outside C2's pool, adding 605 distinct source ECGs and 595 source patients. These parents remain in the training folds; no validation/test record or patient is added.

## Experimental setup

| Arm | Training rows | Unique real ECGs used as waveforms | Definition |
|---|---:|---:|---|
| C2 | 8,579 | 8,579 | Original 50% pool; all rows are real ECGs. |
| G1 | 10,109 | 8,579 | C2 plus 1,530 fully generated ECGs with provisional conditioning-parent labels. |
| R1 | 10,109 | 9,184 | C2 plus the same 1,530 parent slots as real ECGs, retaining duplicate parents. |
| C2@full100 | 17,084 | 17,084 | All real ECGs in the original full100 training pool. |

Unique real ECGs are counted by source record identity, not by bytewise waveform deduplication. G1's conditioning parents can supply information without their real waveforms appearing in G1. R1 includes those parent waveforms; its labels and appended slot order match G1 exactly.

The classifier is **ResNet1d**, width **64**, kernel size **7**, trained with **AdamW** (learning rate **0.001**, weight decay **0.01**) and **BCEWithLogitsLoss** binary cross-entropy for the five multilabel targets. All runs use the **MPS** device backend.

Each input contains **12 leads at 100 Hz for 10 seconds**, or 1,000 samples per lead, in mV. Validation contains 2,146 real ECGs and test contains 2,158. These held-out ECGs and labels are identical across arms.

## Test · selected

Macro-AUROC, mean ± sample SD across three seeds.

| Arm | All ECGs | No quality flag | Any quality flag |
|---|---:|---:|---:|
| A0 · depleted real | 0.9066 ± 0.0015 | 0.9090 ± 0.0016 | 0.8995 ± 0.0009 |
| A1 · standard augmentation | 0.9111 ± 0.0022 | 0.9127 ± 0.0022 | 0.9065 ± 0.0028 |
| A2 · matched artifact edits | 0.9102 ± 0.0006 | 0.9124 ± 0.0007 | 0.9040 ± 0.0032 |
| A3 · targeted artifact edits | 0.9123 ± 0.0008 | 0.9138 ± 0.0010 | 0.9080 ± 0.0004 |
| C1 · natural mix | 0.9092 ± 0.0019 | 0.9119 ± 0.0023 | 0.9011 ± 0.0020 |
| C2 · 8,579 real | 0.9141 ± 0.0019 | 0.9166 ± 0.0008 | 0.9066 ± 0.0067 |
| G1 · real + 1,530 generated | 0.9141 ± 0.0029 | 0.9164 ± 0.0037 | 0.9071 ± 0.0005 |
| R1 · real + 1,530 parent rows | 0.9140 ± 0.0018 | 0.9166 ± 0.0014 | 0.9054 ± 0.0036 |
| C2 full100 · all real | 0.9240 ± 0.0020 | 0.9250 ± 0.0026 | 0.9214 ± 0.0003 |

Paired differences: 2,000 crossed patient × seed bootstrap draws, 95% percentile intervals, bootstrap seed 0. Values are AUROC units.

| Contrast | All ECGs Δ [95% CI] | No quality flag Δ [95% CI] | Any quality flag Δ [95% CI] |
|---|---:|---:|---:|
| G1 − C2 | -0.0000 [-0.0055, +0.0054] | -0.0001 [-0.0055, +0.0054] | +0.0005 [-0.0098, +0.0109] |
| G1 − R1 | +0.0001 [-0.0050, +0.0052] | -0.0002 [-0.0059, +0.0053] | +0.0017 [-0.0067, +0.0103] |
| R1 − C2 | -0.0001 [-0.0034, +0.0035] | +0.0000 [-0.0033, +0.0039] | -0.0012 [-0.0095, +0.0081] |

## Validation · selected

Macro-AUROC, mean ± sample SD across three seeds.

| Arm | All ECGs | No quality flag | Any quality flag |
|---|---:|---:|---:|
| A0 · depleted real | 0.9140 ± 0.0016 | 0.9177 ± 0.0016 | 0.9047 ± 0.0019 |
| A1 · standard augmentation | 0.9155 ± 0.0006 | 0.9167 ± 0.0007 | 0.9133 ± 0.0006 |
| A2 · matched artifact edits | 0.9160 ± 0.0009 | 0.9184 ± 0.0019 | 0.9107 ± 0.0022 |
| A3 · targeted artifact edits | 0.9159 ± 0.0007 | 0.9173 ± 0.0005 | 0.9137 ± 0.0025 |
| C1 · natural mix | 0.9161 ± 0.0015 | 0.9187 ± 0.0027 | 0.9098 ± 0.0031 |
| C2 · 8,579 real | 0.9200 ± 0.0013 | 0.9224 ± 0.0005 | 0.9145 ± 0.0044 |
| G1 · real + 1,530 generated | 0.9189 ± 0.0011 | 0.9207 ± 0.0009 | 0.9149 ± 0.0026 |
| R1 · real + 1,530 parent rows | 0.9193 ± 0.0013 | 0.9218 ± 0.0010 | 0.9130 ± 0.0043 |
| C2 full100 · all real | 0.9289 ± 0.0024 | 0.9317 ± 0.0026 | 0.9225 ± 0.0029 |

Paired differences: 2,000 crossed patient × seed bootstrap draws, 95% percentile intervals, bootstrap seed 0. Values are AUROC units.

| Contrast | All ECGs Δ [95% CI] | No quality flag Δ [95% CI] | Any quality flag Δ [95% CI] |
|---|---:|---:|---:|
| G1 − C2 | -0.0011 [-0.0042, +0.0018] | -0.0017 [-0.0051, +0.0017] | +0.0004 [-0.0067, +0.0079] |
| G1 − R1 | -0.0005 [-0.0036, +0.0030] | -0.0012 [-0.0048, +0.0025] | +0.0020 [-0.0048, +0.0088] |
| R1 − C2 | -0.0007 [-0.0036, +0.0024] | -0.0005 [-0.0036, +0.0026] | -0.0015 [-0.0088, +0.0062] |

## Validation · final

Macro-AUROC, mean ± sample SD across three seeds.

| Arm | All ECGs | No quality flag | Any quality flag |
|---|---:|---:|---:|
| A0 · depleted real | 0.8867 ± 0.0094 | 0.8891 ± 0.0091 | 0.8818 ± 0.0107 |
| A1 · standard augmentation | 0.8813 ± 0.0035 | 0.8816 ± 0.0039 | 0.8832 ± 0.0032 |
| A2 · matched artifact edits | 0.8907 ± 0.0024 | 0.8904 ± 0.0027 | 0.8943 ± 0.0043 |
| A3 · targeted artifact edits | 0.8889 ± 0.0039 | 0.8894 ± 0.0056 | 0.8892 ± 0.0076 |
| C1 · natural mix | 0.8993 ± 0.0004 | 0.9015 ± 0.0016 | 0.8936 ± 0.0053 |
| C2 · 8,579 real | 0.8965 ± 0.0027 | 0.8987 ± 0.0023 | 0.8910 ± 0.0051 |
| G1 · real + 1,530 generated | 0.8938 ± 0.0049 | 0.8956 ± 0.0044 | 0.8895 ± 0.0060 |
| R1 · real + 1,530 parent rows | 0.8936 ± 0.0069 | 0.8946 ± 0.0075 | 0.8923 ± 0.0050 |
| C2 full100 · all real | 0.9059 ± 0.0030 | 0.9075 ± 0.0029 | 0.9022 ± 0.0042 |

Paired differences: 2,000 crossed patient × seed bootstrap draws, 95% percentile intervals, bootstrap seed 0. Values are AUROC units.

| Contrast | All ECGs Δ [95% CI] | No quality flag Δ [95% CI] | Any quality flag Δ [95% CI] |
|---|---:|---:|---:|
| G1 − C2 | -0.0027 [-0.0114, +0.0052] | -0.0031 [-0.0114, +0.0050] | -0.0015 [-0.0159, +0.0108] |
| G1 − R1 | +0.0002 [-0.0058, +0.0060] | +0.0010 [-0.0051, +0.0078] | -0.0028 [-0.0140, +0.0081] |
| R1 − C2 | -0.0030 [-0.0131, +0.0068] | -0.0041 [-0.0150, +0.0054] | +0.0013 [-0.0129, +0.0159] |

## Seed-level test differences

| Contrast | Slice | Seed 0 | Seed 1 | Seed 2 |
|---|---|---:|---:|---:|
| G1 − C2 | overall | -0.0034 | +0.0054 | -0.0021 |
| G1 − C2 | clean | -0.0030 | +0.0051 | -0.0025 |
| G1 − C2 | artifact | -0.0052 | +0.0085 | -0.0018 |
| G1 − R1 | overall | -0.0037 | +0.0042 | -0.0002 |
| G1 − R1 | clean | -0.0040 | +0.0048 | -0.0012 |
| G1 − R1 | artifact | -0.0024 | +0.0036 | +0.0040 |
| R1 − C2 | overall | +0.0003 | +0.0012 | -0.0020 |
| R1 − C2 | clean | +0.0010 | +0.0003 | -0.0012 |
| R1 − C2 | artifact | -0.0028 | +0.0049 | -0.0058 |

## Checkpoint selection

| Arm | Best steps, seeds 0 / 1 / 2 | Best step after 80% of budget |
|---|---|---:|
| A0 · depleted real | 1100 / 1000 / 900 | 0 / 3 |
| A1 · standard augmentation | 3800 / 3500 / 3000 | 0 / 3 |
| A2 · matched artifact edits | 1100 / 1000 / 1200 | 0 / 3 |
| A3 · targeted artifact edits | 1700 / 1600 / 1100 | 0 / 3 |
| C1 · natural mix | 800 / 1100 / 1100 | 0 / 3 |
| C2 · 8,579 real | 1200 / 1700 / 900 | 0 / 3 |
| G1 · real + 1,530 generated | 1800 / 2000 / 2500 | 0 / 3 |
| R1 · real + 1,530 parent rows | 1400 / 1200 / 1000 | 0 / 3 |
| C2 full100 · all real | 3200 / 3100 / 2100 | 0 / 3 |

Checkpoint timing is descriptive; an early best checkpoint does not prove convergence.

## Scope and limitations

- G1 uses provisional conditioning-parent diagnostic labels. Rhythm/shape acceptance does not validate generated NORM, MI, STTC, CD or HYP morphology.
- Some generated conditioning parents are outside C2's 50% training pool. G1 versus C2 therefore combines generated waveforms, additional parent information and changed sampling exposure.
- R1 repeats each accepted slot's real parent, including duplicates, in the same appended order and with the same label vector as G1. G1 versus R1 compares generated and parent waveforms under matched row multiplicities; label mismatch remains part of the generated-waveform effect.
- The 1,530 accepted signals are a frozen early prefix of the QA generation run, not a random sample of its final allocation. Acceptance and conditioning may change the sample distribution.
- This is an exploratory extension designed after the original test results were viewed. The unchanged test fold is reused; intervals are descriptive and are not multiplicity-adjusted confirmatory tests. Validation-best results are selection-biased.
- Published DiffuSETS pretraining is reported as MIMIC-IV-ECG, with no PTB-XL overlap declared. Checkpoint identity is pinned; checkpoint training membership was unavailable for independent holdout-overlap verification.
- Artifact and clean slices use the original real held-out ECG quality flags. No parent quality flag is treated as a verified generated-signal label. Full100 is a real-data reference, not full generation.
- Mean ± SD describes three training seeds; paired 95% intervals separately resample patients and training seeds. These quantify different sources of variation.

Exact frozen cohort counts, ancestry coverage, snapshot identities and label-policy provenance are preserved in `comparisons.json` → `data_provenance` and [data/provenance.json](/Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/data/provenance.json).

## Reproduction commands

The sequence below records collection, data freezing, training, held-out evaluation and comparison. Run it from a fresh experiment state with the same pinned source and credentials for collection. An existing completed experiment intentionally rejects replanning or training after its test claim.

```sh
cd /Users/sebastian/dev/TeeMoE/synthetic_data_engine
uv run python /Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/collect_snapshot.py prepare
uv run python /Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/collect_snapshot.py collect
uv run python /Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/prepare_data.py
uv run python /Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/run_experiment.py plan
uv run python /Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/run_experiment.py train
uv run python /Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/run_experiment.py evaluate
uv run python /Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/compare_results.py --output /Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/eval/comparison
```

## Figures

![main comparison](/Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/eval/comparison/main_comparison.png)

![all arms test](/Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/eval/comparison/all_arms_test.png)

![paired differences](/Users/sebastian/dev/TeeMoE/synthetic_data_engine/runs/ptbxl-diffusets-1530/eval/comparison/paired_differences.png)

Input identity: `5772b59cda64f8e4250746df85cc226f605e901232c4e32b2fbde30a6b78e7a1`.

All underlying predictions, trial definitions, results and checkpoint identities are listed with hashes in `comparisons.json`. SVG versions of the figures are included.
