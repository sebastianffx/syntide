# ptbxl-artifact-50-val100: generator audit (report-only)

## scarcity50

- Provenance: 2000 of 2000 gate recipes regenerated from their ledger rows with matching ids; every parent is in `depleted`.
- Retained natural artifact ECGs per type (calibration): {'baseline_drift': 56, 'static_noise': 135, 'burst_noise': 29}.

| Policy (arm) | Accepted | Accepted by type | Rejections | Requested type mix |
|---|---|---|---|---|
| coverage_matched (A2) | 998 / 1000 | baseline_drift 100.0%, burst_noise 98.4%, static_noise 100.0% | alignment@1 2 | baseline_drift 24%, burst_noise 13%, static_noise 63% |
| gap_targeted (A3) | 987 / 1000 | baseline_drift 100.0%, burst_noise 97.4%, static_noise 100.0% | alignment@1 13 | baseline_drift 31%, burst_noise 50%, static_noise 19% |

### Label audit (model: runs/ptbxl-artifact-50-val100/scarcity50/A0/seed0)

Predictions of A0's best checkpoint on each parent vs its accepted edit. A flip is any class whose 0.5-thresholded label changes.

| Policy | Edits | Mean abs Δp | ECGs with a flipped class | Flip rate by type |
|---|---|---|---|---|
| coverage_matched | 998 | 0.0263 | 8.4% | baseline_drift 34.4%, burst_noise 0.0%, static_noise 0.2% |
| gap_targeted | 987 | 0.0749 | 21.3% | baseline_drift 61.5%, burst_noise 3.1%, static_noise 1.6% |

![coverage](figures/coverage_scarcity50.png)
![signal examples](figures/signal_examples_scarcity50.png)
