"""Plot held-out test scores for the original seed-0 DiffuSETS trial.

Reads only figures/test_saturation_points.csv. No training, inference, or
checkpoint selection is performed; the CSV contains validation-selected scores.
Run with synthetic_data_engine/.venv/bin/python from any working directory.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "figures"
SHARES = (10, 20, 30, 50, 70, 100)
PARTS = ("overall", "artifact", "clean")
LABELS = {"overall": "All records", "artifact": "Noisy records", "clean": "Clean slice"}
COLORS = {"overall": "#2a78d6", "artifact": "#f26732", "clean": "#18b080"}
PURPLE, GREY, INK, SURFACE = "#8b46ba", "#898781", "#20201e", "#fcfcfb"
INT_FIELDS = (
    "real_patient_share_percent", "real_records", "generated_records", "seed",
    "training_steps_completed", "best_step", "n_test", "n_test_artifact",
)
FLOAT_FIELDS = (*[f"test_auroc_{part}" for part in PARTS], "test_loss")


def read_points() -> tuple[list[dict], dict]:
    with (OUT / "test_saturation_points.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 7:
        raise ValueError("Expected six real-data points and one DiffuSETS point")
    for row in rows:
        for field in INT_FIELDS:
            row[field] = int(row[field])
        for field in FLOAT_FIELDS:
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(f"Non-finite {field}: {row}")
        if not all(0 <= row[f"test_auroc_{part}"] <= 1 for part in PARTS):
            raise ValueError(f"Invalid AUROC: {row}")
        if (row["seed"], row["n_test"], row["n_test_artifact"]) != (0, 2158, 513):
            raise ValueError(f"Unexpected seed or held-out partition: {row}")
        if not 0 < row["best_step"] <= row["training_steps_completed"]:
            raise ValueError(f"Invalid selected checkpoint step: {row}")
    real = sorted(
        (row for row in rows if row["scenario"] == "real_scaling"),
        key=lambda row: row["real_patient_share_percent"],
    )
    mixed_rows = [row for row in rows if row["scenario"] == "diffusets"]
    if tuple(row["real_patient_share_percent"] for row in real) != SHARES or len(mixed_rows) != 1:
        raise ValueError("Unexpected scenarios or real-data shares")
    mixed = mixed_rows[0]
    if any(row["generated_records"] != 0 or row["training_steps_completed"] != 12000 for row in real):
        raise ValueError("Expected original 12,000-step real-data scaling runs")
    if tuple(mixed[field] for field in (
        "real_patient_share_percent", "real_records", "generated_records", "training_steps_completed"
    )) != (50, 8579, 4515, 2000):
        raise ValueError("Expected original 2,000-step DiffuSETS mixture trial")
    return real, mixed


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


def plot(real: list[dict], mixed: dict) -> None:
    fig, (ax, steps) = plt.subplots(1, 2, figsize=(16, 6.6))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.25, top=0.76, wspace=0.22)
    fig.suptitle("DiffuSETS trial on held-out test data", x=0.07, y=0.97,
                 ha="left", fontsize=20, fontweight="semibold")
    fig.text(0.07, 0.903, "8,579 real ECGs + 4,515 accepted generated ECGs  ·  "
             "48 real + 16 generated per batch  ·  Seed 0", fontsize=12, color="#52514e")
    fig.text(0.07, 0.852, "Validation-selected checkpoints: original scaling, best of 12,000 steps  ·  "
             "DiffuSETS trial, best of 2,000 steps", fontsize=11, color="#696860")

    for part in PARTS:
        ax.plot(SHARES, [row[f"test_auroc_{part}"] for row in real], "o-",
                color=COLORS[part], lw=2.5, ms=7.5, mec=SURFACE, mew=1.6, label=LABELS[part])
        ax.scatter(50, mixed[f"test_auroc_{part}"], marker="D", s=110,
                   color=COLORS[part], edgecolor=PURPLE, linewidth=2, zorder=6)
    callout = ("50% real + 4,515 DiffuSETS\n"
               f"All: {mixed['test_auroc_overall']:.4f}\n"
               f"Noisy: {mixed['test_auroc_artifact']:.4f}  ·  Clean: {mixed['test_auroc_clean']:.4f}")
    ax.annotate(callout, (50, mixed["test_auroc_overall"]),
                xytext=(0.56, 0.40), textcoords="axes fraction", fontsize=10.5, color=PURPLE,
                arrowprops={"arrowstyle": "-", "color": PURPLE, "lw": 1.1},
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 3})
    ax.set(title="Held-out test AUROC by training-data share", xlabel="Real training patients used (%)",
           ylabel="Test macro-AUROC", xlim=(5, 105))
    values = [row[f"test_auroc_{part}"] for row in [*real, mixed] for part in PARTS]
    ax.set_ylim(min(values) - 0.004, max(values) + 0.006)
    ax.set_xticks(SHARES)
    ax.grid(color="#e1e0d9", lw=1)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="D", ls="", color=PURPLE, mfc=SURFACE, mew=2, ms=7))
    labels.append("+4,515 DiffuSETS (same slice colors)")
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=9)

    selected_steps = [row["best_step"] for row in real]
    steps.plot(SHARES, selected_steps, "o-", color=COLORS["overall"], lw=2.5, ms=7.5,
               mec=SURFACE, mew=1.6)
    for share, step in zip(SHARES, selected_steps, strict=True):
        steps.annotate(f"{step:,}", (share, step), xytext=(0, 10),
                       textcoords="offset points", ha="center")
    mixed_step = mixed["best_step"]
    steps.scatter(50, mixed_step, marker="D", s=110, color=PURPLE,
                  edgecolor=SURFACE, linewidth=1.5, zorder=5)
    steps.annotate(f"+4,515 DiffuSETS: {mixed_step:,}", (50, mixed_step),
                   xytext=(15, 20), textcoords="offset points", fontsize=11, color=PURPLE,
                   arrowprops={"arrowstyle": "-", "color": PURPLE})
    steps.set(title="Checkpoint step selected on validation", xlabel="Real training patients used (%)",
              ylabel="Optimizer steps", xlim=(5, 105), ylim=(0, max(*selected_steps, mixed_step) * 1.23))
    steps.set_xticks(SHARES)
    steps.grid(color="#e1e0d9", lw=1)
    steps.text(0.03, 0.96, "Checkpoint used for held-out test evaluation", transform=steps.transAxes,
               va="top", fontsize=10, color="#696860")

    fig.text(0.07, 0.135, "Held-out PTB-XL test partition (fold 10): 2,158 ECGs  ·  "
             "513 noisy records  ·  1,645 clean records.", fontsize=9.5, color="#52514e")
    fig.text(0.07, 0.100, "All three test AUROCs use the same checkpoint selected by overall validation "
             "AUROC. Test data were not used for checkpoint selection.", fontsize=9.5, color="#52514e")
    fig.text(0.07, 0.065, "Training budgets differ: 12,000 real-only vs 2,000 mixed steps. "
             "The x-axis counts real patients; generated ECGs retain the 50% coordinate.",
             fontsize=9.5, color="#52514e")
    fig.text(0.07, 0.030, "Single seed; no error bars. 2,212 generated ECGs use conditioning records outside "
             "the real 50% pool; no held-out patient overlap.", fontsize=9, color=GREY)
    fig.savefig(OUT / "scaling_saturation_test.png", dpi=200)
    fig.savefig(OUT / "scaling_saturation_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    style()
    plot(*read_points())
