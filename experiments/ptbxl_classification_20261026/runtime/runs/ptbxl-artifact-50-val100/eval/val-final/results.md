# ptbxl-artifact-50-val100: results on validation (final checkpoint)

Generated 2026-10-23T14:16:05+00:00 from 21 runs, seeds [0, 1, 2]. Decision rules `e6a66b593c2c`: rules frozen 2026-10-23T12:32:53+00:00.

## Primary question: A3 - A1 on the artifact slice (scarcity50)

**Useful negative: no artifact gain shown, and a gain of +0.02 is ruled out.**

- Artifact slice: Δ +0.006 [-0.008, +0.020] (528 ECGs from 494 patients, 3 seeds; per seed +0.005, -0.001, +0.015).
- Clean slice: Δ +0.008 [+0.001, +0.015].
- Some artifact improvement (CI low > 0): **no**.
- Practical target +0.02: **ruled out**.
- Clean-slice non-inferiority (CI low > -0.01): **yes**.

## Fixed sequence

Each step is tested only if every earlier step showed a benefit (artifact CI low > 0 and clean CI low > -0.01).

| Step | Contrast | Artifact Δ [CI] | Clean Δ [CI] | Status |
|---|---|---|---|---|
| 1 | A3 - A1 | +0.006 [-0.008, +0.020] | +0.008 [+0.001, +0.015] | failed |
| 2 | A3 - A2 | -0.005 [-0.017, +0.005] | -0.001 [-0.007, +0.005] | not tested |
| 3 | A2 - A1 | +0.011 [-0.001, +0.023] | +0.009 [+0.001, +0.016] | not tested |

## The results table

AUROC is mean ± SD over seeds. Δ is against the reference arm of the same regime (or of the primary regime when that regime does not run it).

| Arm | Seeds | Artifact AUROC | Clean AUROC | All records | Δ artifact vs A1 | Δ clean vs A1 | Gap closed: natural mix | Gap closed: 100% of patients |
|---|---|---|---|---|---|---|---|---|
| A0 | 3 | 0.882 ± 0.011 | 0.889 ± 0.009 | 0.887 ± 0.009 | -0.001 [-0.018, +0.013] | +0.007 [-0.002, +0.017] | - | 0% |
| A1 | 3 | 0.883 ± 0.003 | 0.882 ± 0.004 | 0.881 ± 0.004 | reference |  | - | 6% |
| A2 | 3 | 0.894 ± 0.004 | 0.890 ± 0.003 | 0.891 ± 0.002 | +0.011 [-0.001, +0.023] | +0.009 [+0.001, +0.016] | - | 61% |
| A3 | 3 | 0.889 ± 0.008 | 0.889 ± 0.006 | 0.889 ± 0.004 | +0.006 [-0.008, +0.020] | +0.008 [+0.001, +0.015] | - | 36% |
| C1 | 3 | 0.894 ± 0.005 | 0.902 ± 0.002 | 0.899 ± 0.000 | +0.010 [-0.003, +0.024] | +0.020 [+0.012, +0.028] | - | 58% |
| C2 | 3 | 0.891 ± 0.005 | 0.899 ± 0.002 | 0.897 ± 0.003 | +0.008 [-0.006, +0.022] | +0.017 [+0.010, +0.025] | - | 45% |
| C2@full100 | 3 | 0.902 ± 0.004 | 0.907 ± 0.003 | 0.906 ± 0.003 | +0.019 [+0.004, +0.034] | +0.026 [+0.018, +0.035] | - | 100% |

## Gap closed (descriptive)

- **natural mix**: A3 - A0 = +0.007 [-0.012, +0.028] over C1 - A0 = +0.012 [-0.002, +0.027]; ratio withheld: the gap gate did not pass on validation.
- **100% of patients**: A3 - A0 = +0.007 [-0.012, +0.028] over C2@full100 - A0 = +0.020 [+0.003, +0.037]; **36%** closed.

## Diagnostics (reported, never evidence)

| Contrast | Artifact Δ [CI] | Clean Δ [CI] | All records Δ [CI] | Artifact Δ per seed |
|---|---|---|---|---|
| A1 - A0 | +0.001 [-0.013, +0.018] | -0.007 [-0.017, +0.002] | -0.005 [-0.014, +0.003] | -0.005, -0.001, +0.010 |
| C1 - A0 | +0.012 [-0.002, +0.027] | +0.012 [+0.002, +0.022] | +0.013 [+0.003, +0.022] | +0.007, +0.004, +0.024 |
| C2 - A0 | +0.009 [-0.003, +0.024] | +0.010 [-0.001, +0.019] | +0.010 [+0.000, +0.019] | +0.000, +0.012, +0.015 |
| C2@full100 - A0 | +0.020 [+0.003, +0.037] | +0.018 [+0.006, +0.029] | +0.019 [+0.007, +0.030] | +0.006, +0.022, +0.033 |
| C2@full100 - C2 | +0.011 [-0.002, +0.023] | +0.009 [+0.003, +0.015] | +0.009 [+0.004, +0.015] | +0.005, +0.010, +0.018 |

## Generation and exposure (mean per run)

| Arm | Presentations | Transform slots | Accepted / candidates | Exhausted slots | Distinct recipes per accepted |
|---|---|---|---|---|---|
| A0 | 416,000 | 208,000 | - | 0 | - |
| A1 | 416,000 | 208,000 | 208,000 / 208,000 | 0 | 1.000 |
| A2 | 416,000 | 208,000 | 208,000 / 208,724 | 0 | 1.000 |
| A3 | 416,000 | 208,000 | 208,000 / 210,809 | 0 | 1.000 |
| C1 | 416,000 | 0 | - | 0 | - |
| C2 | 416,000 | 0 | - | 0 | - |
| C2@full100 | 416,000 | 0 | - | 0 | - |

## Gates

- Gate 1 (leakage): re-run on every manifest when a trial is prepared, and again before the test fold is loaded.
- Gate 3 (gap gate, validation artifact slice, scarcity50): C1 - A0 = +0.005 [-0.003, +0.014] against ≥ 0.02: **failed**.
- Gate 4 (generator, scarcity50): passed; coverage_matched 99.8% accepted, gap_targeted 98.7% accepted.
- Gate 5 (freeze): rules frozen 2026-10-23T12:32:53+00:00.

## Compute

21 finished runs, 1.6 h of training in total (4.7 min per run).

## Figures

![forest](figures/forest.png)
![per_class](figures/per_class.png)
![learning_curves](figures/learning_curves.png)

Δ is the paired difference in macro-AUROC, averaged over seeds. Brackets are 95% CIs from a crossed bootstrap (2000 draws) over patients and seeds; both arms are scored on the same draws. Artifact = ECGs with any PTB-XL signal-quality flag.
