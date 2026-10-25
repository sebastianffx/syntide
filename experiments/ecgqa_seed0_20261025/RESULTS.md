# ECG-QA seven-arm comparison, seed 0

Reproduced from the committed full-precision saved metrics; no inference is run.

| Arm | Test accuracy | Clean accuracy | Noisy accuracy | Template macro-F1 | Invalid | Real minus zero |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 42.37% | 43.58% | 39.53% | 27.19% | 0.212% | +2.06 pp |
| Paraphrase | 40.68% | 41.03% | 39.87% | 27.75% | 0.458% | +1.53 pp |
| Grounded | 39.34% | 40.43% | 36.78% | 25.41% | 0.689% | +0.11 pp |
| Grounded + acquisition | 37.62% | 38.95% | 34.51% | 28.55% | 0.771% | +0.29 pp |
| Grounded + acquisition + DiffuSETS 3,030 | 38.04% | 37.23% | 39.92% | 27.33% | 0.302% | +0.13 pp |
| Grounded + acquisition + full DiffuSETS plan | 39.20% | 39.96% | 37.43% | 27.37% | 0.361% | -0.09 pp |
| Baseline + DiffuSETS 3,030; matched original exposure | 42.66% | 43.39% | 40.97% | 26.95% | 0.268% | +0.85 pp |

The exposure-matched arm uses fixed update 12,000: 192,000 original plus 24,000 generated draws, batch 18. Earlier arms use selection-chosen checkpoints and batch 16.
It also excludes grounded/acquisition data. Differences from the earlier stacks do not isolate original exposure alone. All arms use one seed; seed SD is null.
Validation and zero-waveform controls do not establish a reliable general improvement. See evidence/results.json for every validation/test slice and control.
