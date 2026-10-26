# ptbxl-artifact-50-val100: results on test (best checkpoint)

Generated 2026-10-23T14:16:32+00:00 from 21 runs, seeds [0, 1, 2]. Decision rules `e6a66b593c2c`: rules frozen 2026-10-23T12:32:53+00:00; test fold claimed 2026-10-23T14:16:08+00:00.

## Primary question: A3 - A1 on the artifact slice (scarcity50)

**Useful negative: no artifact gain shown, and a gain of +0.02 is ruled out.**

- Artifact slice: Δ +0.002 [-0.006, +0.009] (513 ECGs from 477 patients, 3 seeds; per seed +0.004, +0.001, -0.001).
- Clean slice: Δ +0.001 [-0.003, +0.005].
- Some artifact improvement (CI low > 0): **no**.
- Practical target +0.02: **ruled out**.
- Clean-slice non-inferiority (CI low > -0.01): **yes**.

## Fixed sequence

Each step is tested only if every earlier step showed a benefit (artifact CI low > 0 and clean CI low > -0.01).

| Step | Contrast | Artifact Δ [CI] | Clean Δ [CI] | Status |
|---|---|---|---|---|
| 1 | A3 - A1 | +0.002 [-0.006, +0.009] | +0.001 [-0.003, +0.005] | failed |
| 2 | A3 - A2 | +0.004 [-0.003, +0.011] | +0.001 [-0.002, +0.005] | not tested |
| 3 | A2 - A1 | -0.002 [-0.013, +0.008] | -0.000 [-0.005, +0.004] | not tested |

## The results table

AUROC is mean ± SD over seeds. Δ is against the reference arm of the same regime (or of the primary regime when that regime does not run it).

| Arm | Seeds | Artifact AUROC | Clean AUROC | All records | Δ artifact vs A1 | Δ clean vs A1 | Gap closed: natural mix | Gap closed: 100% of patients |
|---|---|---|---|---|---|---|---|---|
| A0 | 3 | 0.899 ± 0.001 | 0.909 ± 0.002 | 0.907 ± 0.002 | -0.007 [-0.017, +0.003] | -0.004 [-0.009, +0.001] | - | 0% |
| A1 | 3 | 0.906 ± 0.003 | 0.913 ± 0.002 | 0.911 ± 0.002 | reference |  | - | 32% |
| A2 | 3 | 0.904 ± 0.003 | 0.912 ± 0.001 | 0.910 ± 0.001 | -0.002 [-0.013, +0.008] | -0.000 [-0.005, +0.004] | - | 21% |
| A3 | 3 | 0.908 ± 0.000 | 0.914 ± 0.001 | 0.912 ± 0.001 | +0.002 [-0.006, +0.009] | +0.001 [-0.003, +0.005] | - | 39% |
| C1 | 3 | 0.901 ± 0.002 | 0.912 ± 0.002 | 0.909 ± 0.002 | -0.005 [-0.015, +0.004] | -0.001 [-0.007, +0.005] | - | 7% |
| C2 | 3 | 0.907 ± 0.007 | 0.917 ± 0.001 | 0.914 ± 0.002 | +0.000 [-0.012, +0.011] | +0.004 [-0.001, +0.008] | - | 33% |
| C2@full100 | 3 | 0.921 ± 0.000 | 0.925 ± 0.003 | 0.924 ± 0.002 | +0.015 [+0.006, +0.024] | +0.012 [+0.006, +0.018] | - | 100% |

## Gap closed (descriptive)

- **natural mix**: A3 - A0 = +0.009 [+0.000, +0.016] over C1 - A0 = +0.002 [-0.008, +0.010]; ratio withheld: the gap gate did not pass on validation.
- **100% of patients**: A3 - A0 = +0.009 [+0.000, +0.016] over C2@full100 - A0 = +0.022 [+0.012, +0.031]; **39%** closed.

## Diagnostics (reported, never evidence)

| Contrast | Artifact Δ [CI] | Clean Δ [CI] | All records Δ [CI] | Artifact Δ per seed |
|---|---|---|---|---|
| A1 - A0 | +0.007 [-0.003, +0.017] | +0.004 [-0.001, +0.009] | +0.004 [-0.000, +0.010] | +0.004, +0.007, +0.011 |
| C1 - A0 | +0.002 [-0.008, +0.010] | +0.003 [-0.001, +0.007] | +0.003 [-0.001, +0.006] | -0.001, +0.003, +0.003 |
| C2 - A0 | +0.007 [-0.004, +0.018] | +0.008 [+0.003, +0.012] | +0.008 [+0.003, +0.012] | +0.012, -0.001, +0.010 |
| C2@full100 - A0 | +0.022 [+0.012, +0.031] | +0.016 [+0.011, +0.021] | +0.017 [+0.013, +0.022] | +0.021, +0.022, +0.023 |
| C2@full100 - C2 | +0.015 [+0.005, +0.026] | +0.008 [+0.004, +0.013] | +0.010 [+0.006, +0.014] | +0.009, +0.023, +0.013 |

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
- Gate 5 (freeze): rules frozen 2026-10-23T12:32:53+00:00; test fold claimed 2026-10-23T14:16:08+00:00.

## Compute

21 finished runs, 1.6 h of training in total (4.7 min per run).

## Figures

![forest](figures/forest.png)
![per_class](figures/per_class.png)
![learning_curves](figures/learning_curves.png)

Δ is the paired difference in macro-AUROC, averaged over seeds. Brackets are 95% CIs from a crossed bootstrap (2000 draws) over patients and seeds; both arms are scored on the same draws. Artifact = ECGs with any PTB-XL signal-quality flag.
