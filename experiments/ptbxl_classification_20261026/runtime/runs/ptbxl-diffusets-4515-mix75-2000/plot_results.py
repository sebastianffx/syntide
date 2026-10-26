"""Export the completed 4,515-ECG trial and compare it with real-data scaling.

Run from any directory with synthetic_data_engine/.venv/bin/python. Reads native
SYNTIDE results and saved predictions only: no training or inference is performed.
Original runs are never changed. Every output is written under this run's figures/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
ENGINE = RUNS.parent
OUT = ROOT / "figures"
MIXED = ROOT / "pool50/mix75/seed0"
SHARES = (10, 20, 30, 50, 70, 100)
PARTS = ("overall", "artifact", "clean")
LABELS = {"overall": "All records", "artifact": "Noisy records", "clean": "Clean slice"}
COLORS = {"overall": "#2a78d6", "artifact": "#f26732", "clean": "#18b080"}
PURPLE, GREY, INK, SURFACE = "#8b46ba", "#898781", "#20201e", "#fcfcfb"
BUDGET = 2000


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def read_run(folder: Path, share: int, generated: int = 0) -> dict:
    result = read(folder / "result.json")
    if result.get("status") != "complete":
        raise ValueError(f"Incomplete run: {folder}")
    history = result["val_history"]
    if [row["step"] for row in history] != list(range(100, result["final_step"] + 1, 100)):
        raise ValueError(f"Unexpected validation grid: {folder}")
    best = max(history, key=lambda row: row["val/auroc_overall"])
    if best["step"] != result["best_step"] or not np.isclose(
        best["val/auroc_overall"], result["best_val_auroc"], atol=1e-12, rtol=1e-12
    ):
        raise ValueError(f"Best checkpoint does not match overall validation selection: {folder}")
    manifest = read(ENGINE / f"manifests/ptbxl-scaling/pool{share}.json")
    return {
        "folder": folder,
        "source": str(folder.relative_to(RUNS)),
        "real_patient_share_percent": share,
        "real_records": len(manifest["roles"]["pool"]),
        "generated_records": generated,
        "result": result,
        "history": history,
    }


def selected_row(run: dict, scenario: str, limit: int | None = None) -> dict:
    history = [row for row in run["history"] if limit is None or row["step"] <= limit]
    best = max(history, key=lambda row: row["val/auroc_overall"])
    smooth = np.convolve([row["val/auroc_overall"] for row in history], np.ones(5) / 5, "valid")
    peak_index = int(np.argmax(smooth))
    return {
        "scenario": scenario,
        "source": run["source"],
        "seed": run["result"]["replicate_seed"],
        "real_patient_share_percent": run["real_patient_share_percent"],
        "real_records": run["real_records"],
        "generated_records": run["generated_records"],
        "training_steps_completed": run["result"]["final_step"],
        "selection_budget_steps": history[-1]["step"],
        "best_step": best["step"],
        "smoothed_peak_step": history[peak_index + 2]["step"],
        "smoothed_peak_auroc": float(smooth[peak_index]),
        "smoothed_peak_at_observation_boundary": peak_index in (0, len(smooth) - 1),
        **{f"val_auroc_{part}": best[f"val/auroc_{part}"] for part in PARTS},
        "val_loss": best["val/loss"],
    }


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to export for {name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def history_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        history = run["history"]
        smooth = np.convolve([row["val/auroc_overall"] for row in history], np.ones(5) / 5, "valid")
        smooth_by_step = {history[i + 2]["step"]: float(value) for i, value in enumerate(smooth)}
        for entry in history:
            rows.append({
                "source": run["source"],
                "seed": run["result"]["replicate_seed"],
                "real_patient_share_percent": run["real_patient_share_percent"],
                "real_records": run["real_records"],
                "generated_records": run["generated_records"],
                "training_steps_completed": run["result"]["final_step"],
                "step": entry["step"],
                **{f"val_auroc_{part}": entry[f"val/auroc_{part}"] for part in PARTS},
                "val_loss": entry["val/loss"],
                "val_auroc_overall_centered_ma5": smooth_by_step.get(entry["step"], ""),
                "within_2000_step_budget": entry["step"] <= BUDGET,
            })
    return rows


def per_class_rows(run: dict, split: str) -> list[dict]:
    path = run["folder"] / f"predictions_{split}.parquet"
    if not path.is_file():
        return []
    table = pq.read_table(path).to_pydict()
    columns = {key: np.asarray(value) for key, value in table.items()}
    rows = []
    for name in sorted(set(columns["class_id"])):
        mask = (columns["checkpoint"] == "best") & (columns["class_id"] == name)
        if not mask.any():
            raise ValueError(f"No best-checkpoint predictions for {name}: {path}")
        y = columns["y_true"][mask].astype(bool)
        p = columns["y_prob"][mask].astype(float)
        steps = np.unique(columns["step"][mask])
        if steps.tolist() != [run["result"]["best_step"]] or not np.isfinite(p).all():
            raise ValueError(f"Invalid prediction step or probability: {path}")
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        rows.append({
            "source": run["source"],
            "split": split,
            "checkpoint": "best_by_validation_overall",
            "step": int(steps[0]),
            "class": str(name),
            "n_records": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int((~y).sum()),
            "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "binary_cross_entropy": float(-np.mean(np.where(y, np.log(clipped), np.log1p(-clipped)))),
        })
    return rows


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#696860",
        "ytick.color": "#696860",
        "axes.titlelocation": "left",
        "axes.titleweight": "semibold",
        "axes.titlesize": 14,
        "text.color": INK,
        "svg.fonttype": "none",
    })


def save_figure(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def saturation_plot(baselines: list[dict], mixed: dict, *, matched: bool) -> None:
    fig, (ax, steps) = plt.subplots(1, 2, figsize=(16, 6.6))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.76, wspace=0.22)
    title = "DiffuSETS trial in the context of real-data scaling"
    subtitle = "Original scaling: best of 12,000 steps  ·  DiffuSETS trial: best of 2,000 steps"
    if matched:
        title = "DiffuSETS versus real-data scaling at the same step budget"
        subtitle = "All checkpoint selections and smoothed peaks restricted to the first 2,000 steps"
    fig.suptitle(title, x=0.07, y=0.97, ha="left", fontsize=20, fontweight="semibold")
    fig.text(0.07, 0.903, "8,579 real ECGs + 4,515 accepted generated ECGs  ·  "
             "48 real + 16 generated per batch  ·  Seed 0", fontsize=12, color="#52514e")
    fig.text(0.07, 0.852, subtitle, fontsize=11, color="#696860")

    for part in PARTS:
        values = [row[f"val_auroc_{part}"] for row in baselines]
        ax.plot(SHARES, values, "o-", color=COLORS[part], lw=2.5, ms=7.5,
                mec=SURFACE, mew=1.6, label=LABELS[part])
        ax.scatter(50, mixed[f"val_auroc_{part}"], marker="D", s=110, color=COLORS[part],
                   edgecolor=PURPLE, linewidth=2, zorder=6)
    callout = ("50% real + 4,515 DiffuSETS\n"
               f"All: {mixed['val_auroc_overall']:.4f}\n"
               f"Noisy: {mixed['val_auroc_artifact']:.4f}  ·  Clean: {mixed['val_auroc_clean']:.4f}")
    ax.annotate(callout, (50, mixed["val_auroc_overall"]),
                xytext=(0.56, 0.40), textcoords="axes fraction", fontsize=10.5, color=PURPLE,
                arrowprops={"arrowstyle": "-", "color": PURPLE, "lw": 1.1},
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 3})
    ax.set(title="Best validation AUROC by training-data share", xlabel="Real training patients used (%)",
           ylabel="Macro-AUROC at the best overall checkpoint", xlim=(5, 105))
    all_values = [row[f"val_auroc_{part}"] for row in [*baselines, mixed] for part in PARTS]
    ax.set_ylim(min(all_values) - 0.004, max(all_values) + 0.006)
    ax.set_xticks(SHARES)
    ax.grid(color="#e1e0d9", lw=1)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="D", ls="", color=PURPLE, mfc=SURFACE, mew=2, ms=7)]
    labels += ["+4,515 DiffuSETS (same slice colors)"]
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=9)

    peaks = [row["smoothed_peak_step"] for row in baselines]
    steps.plot(SHARES, peaks, "o-", color=COLORS["overall"], lw=2.5, ms=7.5,
               mec=SURFACE, mew=1.6)
    for x, y in zip(SHARES, peaks, strict=True):
        steps.annotate(f"{y:,}", (x, y), xytext=(0, 10), textcoords="offset points", ha="center")
    mixed_peak = mixed["smoothed_peak_step"]
    steps.scatter(50, mixed_peak, marker="D", s=110, color=PURPLE,
                  edgecolor=SURFACE, linewidth=1.5, zorder=5)
    steps.annotate(f"+4,515 DiffuSETS: {mixed_peak:,}", (50, mixed_peak),
                   xytext=(15, 18 if matched else -26), textcoords="offset points", fontsize=11, color=PURPLE,
                   arrowprops={"arrowstyle": "-", "color": PURPLE})
    steps.set(title="Step where smoothed validation AUROC peaks", xlabel="Real training patients used (%)",
              ylabel="Optimizer steps", xlim=(5, 105), ylim=(0, max(*peaks, mixed_peak) * 1.23))
    steps.set_xticks(SHARES)
    steps.grid(color="#e1e0d9", lw=1)
    steps.text(0.03, 0.96, "Centered moving average over 5 validations", transform=steps.transAxes,
               va="top", fontsize=10, color="#696860")
    fig.text(0.07, 0.115, "All three AUROCs use the checkpoint chosen by overall validation AUROC. "
             "The x-axis counts real patients; generated ECGs do not change the 50% coordinate.",
             fontsize=9.5, color="#52514e")
    fig.text(0.07, 0.073, "The mixed trial has 20 validations (every 100 steps). Its smoothed peak is "
             "observed only within this 2,000-step window; it does not establish eventual convergence.",
             fontsize=9.5, color="#52514e")
    fig.text(0.07, 0.032, "Single seed; no error bars. 2,212 generated ECGs use conditioning records outside "
             "the real 50% pool; no held-out patient overlap.", fontsize=9, color=GREY)
    save_figure(fig, "scaling_saturation_budget_matched" if matched else "scaling_saturation")


def learning_plot(baselines: list[dict], mixed: dict) -> None:
    chosen = [run for run in baselines if run["real_patient_share_percent"] in (50, 100)]
    styles = [(chosen[0], GREY, "50% real only"),
              (chosen[1], COLORS["overall"], "100% real only"),
              (mixed, PURPLE, "50% real + 4,515 DiffuSETS")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.82, wspace=0.22, hspace=0.35)
    fig.suptitle("Validation learning curves through 2,000 steps", x=0.08, y=0.97,
                 ha="left", fontsize=21, fontweight="semibold")
    fig.text(0.08, 0.915, "Same ResNet1d and seed 0  ·  DiffuSETS batches: 48 real + 16 generated  ·  "
             "Validation every 100 steps", fontsize=11.5, color="#52514e")
    metrics = [("val/auroc_overall", "All validation records", "Macro-AUROC"),
               ("val/auroc_artifact", "Noisy validation records", "Macro-AUROC"),
               ("val/auroc_clean", "Clean validation slice", "Macro-AUROC"),
               ("val/loss", "Validation loss", "Binary cross-entropy")]
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics, strict=True):
        for run, color, label in styles:
            history = [row for row in run["history"] if row["step"] <= BUDGET]
            best = max(history, key=lambda row: row["val/auroc_overall"])
            ax.plot([row["step"] for row in history], [row[metric] for row in history],
                    color=color, label=label, lw=2.2, alpha=0.95)
            ax.scatter(best["step"], best[metric], marker="D" if run is mixed else "o",
                       s=50, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        ax.set(title=title, xlabel="Optimizer steps", ylabel=ylabel, xlim=(100, 2000))
        ax.set_xticks([100, 500, 1000, 1500, 2000])
        ax.grid(color="#e1e0d9", lw=0.9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.074, 0.884),
               ncol=3, frameon=False, fontsize=11)
    fig.text(0.08, 0.073, "Markers identify each run's best overall validation checkpoint within 2,000 steps; "
             "the same checkpoint is marked in every panel.", fontsize=10, color="#52514e")
    fig.text(0.08, 0.036, "Real-only curves are the first 2,000 steps of the historical 12,000-step runs. "
             "Curves show raw validation metrics; test data never selects checkpoints.", fontsize=10, color=GREY)
    save_figure(fig, "learning_curves")


def main() -> None:
    baselines = [read_run(RUNS / f"ptbxl-scaling/pool{share}/real/seed0", share) for share in SHARES]
    mixed = read_run(MIXED, 50, generated=4515)
    if mixed["result"]["final_step"] != BUDGET or mixed["result"]["replicate_seed"] != 0:
        raise ValueError("Expected the seed-0 trial completed at exactly 2,000 steps")
    spec = read(MIXED / "trial.json")
    sampling = read(MIXED / "sampling_audit.json")
    exposure = mixed["result"]["exposure"]
    if not (sampling["every_batch_has_exact_mix"] and sampling["all_rows_seen"]
            and sampling["real_per_batch"] == 48 and sampling["generated_per_batch"] == 16
            and exposure["real_presentations"] == 96000 and exposure["generated_presentations"] == 32000
            and exposure["distinct_real_records"] == 8579 and exposure["distinct_generated_records"] == 4515):
        raise ValueError("Recorded source composition or sample coverage differs from the requested trial")
    scores = read(MIXED / "test_scores.json")
    if scores["trial_id"] != mixed["result"]["trial_id"] or scores["step"] != mixed["result"]["best_step"]:
        raise ValueError("Test scores do not belong to the validation-selected mixed-trial checkpoint")
    full = [selected_row(run, "original_12000_step_context") for run in baselines]
    matched = [selected_row(run, "historical_first_2000_steps", BUDGET) for run in baselines]
    new = selected_row(mixed, "mixed_trial_2000_steps")
    comparisons = []
    for baseline in [*full, *matched]:
        comparisons.append({
            "baseline_scenario": baseline["scenario"],
            "baseline_source": baseline["source"],
            "baseline_real_patient_share_percent": baseline["real_patient_share_percent"],
            "baseline_real_records": baseline["real_records"],
            "baseline_selection_budget_steps": baseline["selection_budget_steps"],
            "baseline_best_step": baseline["best_step"],
            "mixed_source": mixed["source"],
            "mixed_best_step": new["best_step"],
            **{f"baseline_val_auroc_{part}": baseline[f"val_auroc_{part}"] for part in PARTS},
            **{f"mixed_val_auroc_{part}": new[f"val_auroc_{part}"] for part in PARTS},
            **{f"delta_val_auroc_{part}": new[f"val_auroc_{part}"] - baseline[f"val_auroc_{part}"]
               for part in PARTS},
        })
    per_class = [row for run in [*baselines, mixed] for row in per_class_rows(run, "val")]
    per_class += per_class_rows(mixed, "test")
    if len([row for row in per_class if row["source"] == mixed["source"] and row["split"] == "test"]) != 5:
        raise ValueError("Expected saved test predictions for all five mixed-trial classes")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("validation_history.csv", history_rows([*baselines, mixed]))
    write_csv("saturation_points.csv", [*full, *matched, new])
    write_csv("comparisons.csv", comparisons)
    write_csv("per_class_metrics.csv", per_class)
    summary = {
        "schema_version": "ptbxl-diffusets4515-plot-results@1",
        "mixed_trial": {
            **new,
            "trial_id": mixed["result"]["trial_id"],
            "label_policy": spec["transform"]["label_policy"],
            "batch_size": spec["batch_size"],
            "real_per_batch": sampling["real_per_batch"],
            "generated_per_batch": sampling["generated_per_batch"],
            "real_presentations": exposure["real_presentations"],
            "generated_presentations": exposure["generated_presentations"],
            "exposure_recorded": exposure,
            "val_best": {part: new[f"val_auroc_{part}"] for part in PARTS},
            "val_final": {part: mixed["history"][-1][f"val/auroc_{part}"] for part in PARTS},
            "test": {part: scores[f"test/auroc_{part}"] for part in PARTS},
            "test_loss": scores["test/loss"],
            "test_records": scores["n_test"],
            "trainer": mixed["result"]["trainer"],
            "wall_time_s": mixed["result"]["wall_time_s"],
        },
        "original_scaling_points": full,
        "budget_matched_scaling_points": matched,
        "comparisons": comparisons,
        "per_class_metrics": per_class,
        "comparability": {
            "trainer_fingerprint_matches_original_scaling": all(
                run["result"]["trainer"]["fingerprint"] == mixed["result"]["trainer"]["fingerprint"]
                for run in baselines),
            "initial_weights_match_original_scaling": all(
                run["result"]["trainer"]["init_sha256"] == mixed["result"]["trainer"]["init_sha256"]
                for run in baselines),
        },
        "notes": [
            "Every slice uses the first checkpoint maximizing overall validation macro-AUROC.",
            "The original figure uses all 12,000 historical steps; the new trial uses 2,000 steps.",
            "Budget-matched values reuse historical validations at steps <=2,000; no baseline was retrained.",
            "The x-axis measures real training patients: generated ECGs leave the mixed-trial point at 50%.",
            "Smoothed peak is argmax of the centered 5-validation moving average; endpoints are excluded.",
            "The mixed-trial peak is observed only within 2,000 steps and does not establish convergence.",
            "One seed per trial: no across-seed confidence intervals or significance claims.",
            "Test results describe only the mixed trial's validation-selected checkpoint; no test-based selection.",
            "Per-class historical metrics use saved global-best checkpoints, which may exceed 2,000 steps.",
        ],
        "files": ["validation_history.csv", "saturation_points.csv", "comparisons.csv",
                  "per_class_metrics.csv", "summary.json", "scaling_saturation.png", "scaling_saturation.svg",
                  "scaling_saturation_budget_matched.png", "scaling_saturation_budget_matched.svg",
                  "learning_curves.png", "learning_curves.svg"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    style()
    saturation_plot(full, new, matched=False)
    saturation_plot(matched, new, matched=True)
    learning_plot(baselines, mixed)
    print(json.dumps({"output_directory": str(OUT), "mixed_best_step": new["best_step"],
                      "validation": summary["mixed_trial"]["val_best"],
                      "test": summary["mixed_trial"]["test"],
                      "files": summary["files"]}, indent=2))


if __name__ == "__main__":
    main()
