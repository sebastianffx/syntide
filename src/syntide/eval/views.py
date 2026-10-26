"""Create result and generator-audit figures for reports and dashboards.

Figures show paired treatment differences and confidence intervals, learning
curves, diagnostic-class changes, requested artifact strengths, and original
versus edited waveforms. Save each figure as an image and send it to the
configured progress tracker.
"""

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from syntide.data.schema import LEADS, SUPERCLASSES
from syntide.eval.runs import Run
from syntide.eval.tracking import Tracker

# Reference palette (dataviz skill): fixed categorical order, chart chrome and a
# blue <-> red diverging pair.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
DIVERGING = LinearSegmentedColormap.from_list(
    "delta", ["#e34948", "#f0efec", "#2a78d6"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)


def save(fig: Any, name: str, out_dir: Path, tracker: Tracker) -> Path:
    path = out_dir / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    tracker.log_figure(f"evaluation/{name}", fig, 0)
    plt.close(fig)
    return path


def forest(
    rows: Sequence[dict[str, Any]], *, target: float, margin: float, title: str
) -> Any:
    """Plot assessed noisy and clean comparisons from top to bottom."""
    height = 0.5 * len(rows) + 1.8
    fig, axes = plt.subplots(
        1, 2, figsize=(11, height), sharey=True, layout="constrained"
    )
    y = np.arange(len(rows))[::-1]
    for ax, name, line in (
        (axes[0], "artifact", target),
        (axes[1], "clean", margin),
    ):
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.axvline(line, color=MUTED, linewidth=1, linestyle="--")
        for row, yy in zip(rows, y, strict=True):
            report = row[name]
            color = MUTED if row["role"] == "diagnostic" else SERIES[0]
            ax.plot(
                [report["ci_low"], report["ci_high"]],
                [yy, yy],
                color=color,
                linewidth=2.5,
            )
            seeds = report["per_seed"]
            ax.scatter(
                seeds,
                np.full(len(seeds), yy + 0.22),
                s=12,
                color=MUTED,
                linewidth=0,
                zorder=2,
            )
            ax.scatter(
                [report["estimate"]],
                [yy],
                s=48,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
            )
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        reference = (
            f"+{target:g} target"
            if name == "artifact"
            else f"{margin:g} margin"
        )
        ax.set_title(f"{name} slice  (dashed: {reference})", loc="left")
        ax.set_xlabel("Δ macro-AUROC")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [_row_label(row) for row in rows])
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.supxlabel(
        (
            "Large dot: mean paired Δ over seeds. Bar: 95% patient x seed "
            "bootstrap CI. Small dots: each seed."
        ),
        color=MUTED,
        fontsize=7,
    )
    return fig


def _row_label(row: dict[str, Any]) -> str:
    if row["role"] == "diagnostic":
        return f"{row['label']}  (diagnostic)"
    return f"{row['label']}  ({row['role']}: {row.get('status', '')})"


LEARNING_PANELS = {
    "val/auroc_overall": "all records",
    "val/auroc_artifact": "artifact slice",
}


def learning_curves(
    runs: Sequence[Run], arm_order: Sequence[tuple[str, str]], labels: dict
) -> Any:
    """Plot validation means and ranges across seeds by training step."""
    groups: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        groups[run.key].append(run)
    keys = [key for key in arm_order if key in groups]
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 3.8), sharex=True, layout="constrained"
    )
    for ax, metric in zip(axes, LEARNING_PANELS, strict=True):
        for key in keys:
            histories = [run.result["val_history"] for run in groups[key]]
            steps = np.array([entry["step"] for entry in histories[0]])
            if any(
                [entry["step"] for entry in h] != list(steps) for h in histories
            ):
                continue  # seeds on different cadences cannot be averaged
            values = np.array(
                [[entry[metric] for entry in h] for h in histories]
            )
            color = SERIES[
                list(arm_order).index(key) % len(SERIES)
            ]  # color follows the arm
            ax.fill_between(
                steps,
                values.min(axis=0),
                values.max(axis=0),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                steps,
                values.mean(axis=0),
                color=color,
                linewidth=2,
                label=labels[key],
            )
            best = np.mean(
                [run.steps.get("best", np.nan) for run in groups[key]]
            )
            ax.scatter(
                [best],
                [np.interp(best, steps, values.mean(axis=0))],
                s=24,
                color=color,
                zorder=3,
            )
        ax.set_title(
            f"validation macro-AUROC, {LEARNING_PANELS[metric]}", loc="left"
        )
        ax.set_xlabel("optimizer updates")
    axes[1].legend(loc="lower right", ncols=2)
    fig.supxlabel(
        "Line: mean over seeds. Band: seed range. Dot: mean best step.",
        color=MUTED,
        fontsize=7,
    )
    return fig


def per_class(rows: Sequence[dict[str, Any]], title: str) -> Any:
    """Plot diagnostic-class differences on noisy recordings."""
    values = np.array(
        [
            [
                np.nan
                if row["artifact"]["per_class"][k] is None
                else row["artifact"]["per_class"][k]
                for k in SUPERCLASSES
            ]
            for row in rows
        ]
    )
    limit = (
        max(0.01, float(np.nanmax(np.abs(values))))
        if np.isfinite(values).any()
        else 0.01
    )
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.3))
    ax.imshow(values, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.grid(False)
    positives = rows[0]["artifact"]["n_positive"]
    ax.set_xticks(
        range(len(SUPERCLASSES)),
        [f"{k}\n(n+={positives[k]})" for k in SUPERCLASSES],
    )
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    for i, j in np.ndindex(values.shape):
        if np.isfinite(values[i, j]):
            ax.text(
                j,
                i,
                f"{values[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=INK,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left")
    return fig


POLICY_COLORS = {
    "coverage_matched": (SERIES[2], "A2 coverage-matched"),
    "gap_targeted": (SERIES[3], "A3 gap-targeted"),
}


def coverage(
    ledger: Sequence[dict[str, Any]], calibration: dict[str, Any], title: str
) -> Any:
    """Compare natural and requested artifact frequencies and strengths."""
    kinds = list(calibration["types"])
    fig, axes = plt.subplots(
        1,
        len(kinds) + 1,
        figsize=(4 * (len(kinds) + 1), 3.6),
        layout="constrained",
    )
    requested: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in ledger:
        controls = json.loads(row["controls_requested"])
        requested[row["policy"]][controls["artifact_type"]].append(
            controls["amplitude_mv"]
        )

    ax, width, x = axes[0], 0.27, np.arange(len(kinds))
    retained = [calibration["types"][kind]["frequency"] for kind in kinds]
    ax.bar(x - width, retained, width, color=MUTED, label="retained natural")
    for offset, (policy, (color, label)) in zip(
        (0, width), POLICY_COLORS.items(), strict=True
    ):
        total = sum(len(v) for v in requested[policy].values())
        ax.bar(
            x + offset,
            [len(requested[policy][k]) / total for k in kinds],
            width,
            color=color,
            label=label,
        )
    ax.set_xticks(x, [kind.replace("_", "\n") for kind in kinds])
    ax.set_title("artifact type mix", loc="left")
    ax.set_ylabel("share")
    ax.legend(loc="upper right")

    for ax, kind in zip(axes[1:], kinds, strict=True):
        info = calibration["types"][kind]
        low, high = calibration["envelope"][kind]["amplitude_mv"]
        bins = np.geomspace(low, high, 30)
        ax.hist(
            np.clip(info["severities"], low, high),
            bins=bins,
            density=True,
            color=MUTED,
            alpha=0.5,
            label=f"retained natural (n={info['n_retained']})",
        )
        for policy, (color, label) in POLICY_COLORS.items():
            ax.hist(
                requested[policy][kind],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=color,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_title(f"{kind}: severity", loc="left")
        ax.set_xlabel("mV (descriptor units, log scale)")
        ax.set_yticks([])
    axes[1].legend(loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    return fig


def signal_examples(
    examples: Sequence[tuple[dict[str, Any], np.ndarray, Any]],
    fs: int,
    title: str,
) -> Any:
    """Plot original and edited signals, followed by their difference.

    Each example contains a recipe record, an original signal array with 12
    leads, and the generated candidate.
    """
    fig, axes = plt.subplots(
        len(examples),
        2,
        figsize=(12, 2.4 * len(examples)),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for (row, parent, candidate), (left, right) in zip(
        examples, axes, strict=True
    ):
        lead = candidate.strongest_lead
        t = np.arange(parent.shape[-1]) / fs
        controls = json.loads(row["controls_requested"])
        left.plot(
            t, candidate.x[lead], color=SERIES[3], linewidth=1, label="edit"
        )
        left.plot(
            t, parent[lead], color=INK_2, linewidth=0.8, label="parent"
        )  # on top, so it stays visible
        right.plot(t, candidate.residual[lead], color=SERIES[3], linewidth=1)
        limit = 1.05 * max(
            np.abs(parent[lead]).max(), np.abs(candidate.x[lead]).max()
        )
        for ax in (left, right):
            ax.axvspan(
                *candidate.interval, color=SERIES[3], alpha=0.08, linewidth=0
            )
            ax.set_ylim(
                -limit, limit
            )  # one mV scale per example, so the residual reads against the ECG
            ax.set_ylabel("mV")
        details = ", ".join(
            _control(k, v)
            for k, v in controls.items()
            if k not in ("artifact_type", "electrode")
        )
        left.set_title(
            (
                f"{controls['artifact_type']} on {controls['electrode']}, "
                f"lead {LEADS[lead]} ({details})"
            ),
            loc="left",
            fontsize=9,
        )
        right.set_title("residual (edit - parent)", loc="left", fontsize=9)
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("seconds")
    axes[-1, 1].set_xlabel("seconds")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    return fig


def _control(name: str, value: float) -> str:
    for suffix, unit in (("_mv", "mV"), ("_hz", "Hz"), ("_s", "s")):
        if name.endswith(suffix):
            return f"{name.removesuffix(suffix)} {value:.3g} {unit}"
    return f"{name} {value:.3g}"
