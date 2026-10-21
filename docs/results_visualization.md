> **Status (2026-10-21):** the dashboard is no longer TensorBoard-only. `syntide.eval.tracking` sends the same
> metric names and views to local files, TensorBoard or Weights & Biases, chosen by `conf/tracker/*.yaml`.
> The five views, the metric names and "save predictions, not just dashboard metrics" are unchanged.
> See `docs/implementation_plan_v2.md`, Phase 3.

**Yes—TensorBoard is a good starting point for SYNTIDE.** I would pair it with a small evaluation module that computes the comparisons you care about and saves the underlying predictions. That gives you live training diagnostics and a rigorous answer to “did this synthetic data improve the model?”

My suggested setup is:

- **TensorBoard:** learning curves, generator diagnostics, signal examples, and final comparison figures.
- **SYNTIDE evaluator:** metrics, paired comparisons, confidence intervals, and decision rules.
- **Your existing Parquet/DuckDB ledger:** predictions and metadata needed to reproduce those results.

PyTorch’s `SummaryWriter` already supports scalars, histograms, images, Matplotlib figures, and hyperparameter summaries, so most of the display layer is available. [PyTorch 2.14 documentation](https://docs.pytorch.org/docs/2.14/tensorboard.html?utm_source=chatgpt.com)

## 1. Start with five views

I would organize the dashboard around the questions you need to answer.

| View | What to show | What it tells you |
|---|---|---|
| **Training health** | Training loss; validation macro-AUROC for overall, artifact, and clean slices; selected checkpoint; training time | Whether the training run is stable and whether gains depend on checkpoint selection |
| **Downstream utility** | Paired differences between arms, with confidence intervals and individual seed results | Whether synthetic data helps, and which part of the recipe contributes |
| **Coverage and acceptance** | Descriptor distributions before generation, after generation, and after filtering; acceptance rates by severity and rejection reason | Whether generation fills the intended gaps and whether filtering removes the examples you needed |
| **Signal inspection** | Fixed examples showing original signal, edited signal, and residual; requested versus realized controls | What the generator actually changes |
| **Failure slices** | Per-class and per-artifact results, with sample counts and uncertainty | Where an aggregate improvement hides regressions |

For temporal data, I would give **signal inspection and coverage** as much space as the learning curves. A loss curve cannot reveal that an artifact generator consistently places disturbances at the same position or that filtering eliminates almost every severe example.

For the ECG viewer, keep the same small set of development examples across generator versions, use consistent time and mV axes, and shade the edited intervals. That makes visual comparisons much more informative than selecting new random examples each time.

## 2. Make the paired effect plot the main results figure

For your experiment, the most useful custom visualization would be **a forest plot: one dot and confidence interval per comparison**, with two adjacent panels:

- **Artifact slice:** change in macro-AUROC.
- **Clean slice:** change in macro-AUROC.

Use these rows:

| Comparison | Interpretation |
|---|---|
| **A3 − A1** | Benefit of the complete proposed recipe over standard augmentation |
| **A3 − A2** | Additional benefit from targeting within the electrode-space editor |
| **A2 − A1** | Difference between the editor recipe and standard augmentation; both transformation family and sampling policy differ |

On the artifact panel, draw reference lines at **0** and your **+0.02 practical target**. On the clean panel, draw lines at **0** and the **−0.01 non-inferiority margin**.

Overlay the six individual seed differences as smaller dots. This lets you see both the aggregate estimate and whether one seed dominates it.

The evaluator should calculate your mean of paired per-seed differences and the crossed patient/seed bootstrap. Use the same sampled patients and seed IDs across the compared arms, retaining all records from each sampled patient.

Report these as separate conclusions:

- Is an artifact-slice benefit supported?
- Is clean-slice non-inferiority established?
- Is an improvement of +0.02 still plausible, supported, or ruled out?

Keeping those separate handles a result that is reliably positive but smaller than your practical target.

**TensorBoard can display confidence intervals.** Its Custom Scalars dashboard includes margin charts; your evaluator supplies the estimate and bounds. For this particular categorical comparison, I would initially render the forest plot with Matplotlib and send it through `add_figure()`. [GitHub](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/custom_scalar/README.md?utm_source=chatgpt.com)

## 3. Keep the implementation small

I would add two reusable pieces to SYNTIDE:

**`UtilityReport`**

A structured output from the evaluation harness containing:

- Run and dataset identifiers.
- Metric, split, and slice.
- Estimate and confidence interval.
- Comparator and paired contrast.
- Number of patients, records, and seeds.
- Decision flags and the thresholds used.

**`TensorBoardReporter`**

A thin adapter that logs training metrics and renders `UtilityReport` and generator-audit outputs.

Use consistent metric names, for example:

```text
train/loss
val/auroc_overall
val/auroc_artifact
val/auroc_clean
budget/sample_presentations
budget/wall_time_seconds
generator/acceptance_rate
generator/control_error
evaluation/paired_effects
evaluation/slice_differences
examples/original_edit_residual
```

Name runs by regime, arm, and seed—for example, `runs/ptbxl-v2/pool25/A3/seed03/run-id`—then launch:

```bash
tensorboard --logdir runs
```

For training comparisons, use **optimizer updates or cumulative sample presentations** as the main horizontal axis. Epochs are less useful when training-pool sizes differ. Also keep wall-clock cost available so you can assess whether an improvement justifies the additional generation and training work.

During development, populate these views with **validation results**. Publish the final test comparison after freezing the recipe and checkpoint-selection rules.

## 4. Save predictions, not just dashboard metrics

The highest-value addition is a durable **per-example prediction table**. It lets you recompute paired comparisons, confidence intervals, calibration, and diagnostic slices without retraining.

A minimal record would contain:

```text
run_id
checkpoint_id
split
patient_id
record_id
class_id
y_true
y_prob
slice_labels
```

Join that to a run manifest containing the arm, seed, scarcity regime, dataset manifest, generator version, acceptance-policy version, code revision, and training budget.

I would maintain three outputs:

| Output | Purpose |
|---|---|
| **Run manifest** | Identifies exactly what was trained |
| **Prediction table** | Supports reproducible evaluation |
| **Comparison table** | Stores effects, intervals, counts, and decisions |

TensorBoard event logs then become a convenient view of this evidence.

For **TimeNet integration**, I would make the reporting contract independent of the particular task: each adapter provides its metrics, evaluation slices, and independent sampling unit. The dashboard can stay consistent while ECG classification, forecasting, or event detection use different evaluation functions.

## 5. Add Plotly only when interaction would help

Matplotlib figures inside TensorBoard are sufficient for the first version. If you want to zoom into waveforms, hover over examples, or share results with someone who is not running TensorBoard, add a **standalone Plotly HTML export**.

Plotly can export interactive figures with hover, zoom, and trace selection to a self-contained HTML file. That provides a useful sharing format without maintaining a separate dashboard application. [plotly.com](https://plotly.com/python/interactive-html-export/?utm_source=chatgpt.com)

The strongest uses here would be:

- An interactive original/edited/residual waveform viewer.
- A coverage plot where hovering reveals the generation recipe.
- A results plot where hovering reveals seed, sample counts, and confidence bounds.

**For the first implementation, I would prioritize training curves, the paired artifact/clean effect plot, and a fixed-example generator audit.** Those three views cover training behavior, downstream value, and generator behavior—the questions that matter most before expanding the dashboard.