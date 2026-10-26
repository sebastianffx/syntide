"""``python -m syntide.eval.aug_figure runs/ptbxl-aug-sweep [runs/<extension> ...] [--split val|test]
[--hide S4]``: what each amount and kind of augmentation buys, and what it costs.

Every number is a paired change over the baseline arm (A0, no augmentation): same seed = same
initialization, batch order and slots, so arm minus A0 isolates the augmentation. Reads each run's
``predictions_<split>.parquet`` at the checkpoint validation chose, so ``--split val`` works while trials
still train and ``--split test`` after ``syntide.eval.holdout`` scored them.

Further protocol directories add arms planned after the first protocol's test claim, such as
``ptbxl-aug-simvea``'s T6 and ``ptbxl-aug-simvea-whole``'s T6W; their runs pair with the first protocol's seed
by seed (``syntide.eval.across`` gives their CIs). ``--hide`` leaves arms out of the figures only; the
summary keeps every arm. Writes to the first ``<protocol_dir>``:

- ``figures/aug_dose_<split>.png``: change in macro-AUROC against the augmentation dose (share of the batch
  augmented x strength of A1's envelope) on all, noisy and clean records, with the gap to the 100% target,
  and the step each run peaked at (augmentation delays the peak, so it costs training steps).
- ``figures/aug_classes_<split>.png``: the change per arm on each slice and each diagnostic superclass. A
  technique that helps overall can still hurt one class (amplitude scaling and hypertrophy, for example).
- ``figures/aug_curves.png``: validation macro-AUROC over training steps for the dose ladder, the editor
  and the techniques, against A0 and the target: how high each arm peaks, when, and how fast it declines.
- ``aug_summary_<split>.md``: the table behind the first two figures.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from syntide.data.schema import SUPERCLASSES
from syntide.eval.arms_figure import (
    AXIS,
    GRID,
    INK,
    INK2,
    MINUS,
    MUTED,
    REFERENCE_COLOR,
    SURFACE,
    _style,
    shown,
)
from syntide.eval.stats import macro_auroc
from syntide.generate.generic import GenericAugConfig
from syntide.trial import load_spec

SLICES = ("overall", "artifact", "clean")
SLICE_NAMES = {"overall": "all records", "artifact": "noisy records", "clean": "clean records"}
COLUMNS = (*SLICES, *SUPERCLASSES)
COLUMN_NAMES = {"overall": "All", "artifact": "Noisy", "clean": "Clean"} | {c: c for c in SUPERCLASSES}
LABELS = {
    "A0": "no augmentation",
    "S1": "standard, 1/8 of batch, 1/4 strength",
    "S2": "standard, 1/4 of batch, 1/2 strength",
    "A1": "standard, 1/2 of batch",
    "S3": "standard, whole batch",
    "S4": "standard, whole batch, 2x strength",
    "A2": "artifact edits, matched, 1/2 of batch",
    "A3": "artifact edits, targeted, 1/2 of batch",
    "E1": "artifact edits, targeted, whole batch",
    "T1": "time masking",
    "T2": "lead dropout",
    "T3": "amplitude scaling",
    "T4": "time stretch",
    "T5": "mixup",
    "T6": "SIMVEA_h chest-lead mixing, 1/2 of batch",
    "T6W": "SIMVEA_h chest-lead mixing, whole batch",
    "C1": "natural mix",
    "C2": "every ECG of the pool",
}
# The two dose series, keyed by (transform family, sampling policy). Categorical slots 1 and 2 of the
# dataviz reference palette; the target keeps arms_figure's aqua.
SERIES = {
    ("generic", "fixed"): ("Standard augmentation (A1's recipe)", "#2a78d6"),
    ("editor", "gap_targeted"): ("Targeted artifact edits (A3's policy)", "#eb6834"),
}
SEED_OFFSET = 0.12  # seed dots sit left of their mean, in log2 dose units
# Diverging pair of the reference palette: red (worse) <-> gray midpoint <-> blue (better).
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")


def dose(spec) -> float:
    """Share of the batch augmented x strength, where A1's envelope (GenericAugConfig) has strength 1."""
    if spec.transform_family == "none":
        return 0.0
    strength = 1.0
    if spec.transform_family == "generic":
        unit, t = GenericAugConfig(), spec.transform
        strength = float(
            np.mean(
                [
                    (t["gain"][1] - 1) / (unit.gain[1] - 1),
                    t["max_shift_s"] / unit.max_shift_s,
                    t["noise_rms_mv"][1] / unit.noise_rms_mv[1],
                    t["wander_mv"][1] / unit.wander_mv[1],
                ]
            )
        )
    return spec.transform_fraction * strength


def _scores(path: Path, split: str) -> dict[str, float]:
    """Macro-AUROC per slice and AUROC per class (all records) at the best checkpoint."""
    table = pq.read_table(path).to_pandas()
    table = table[table["checkpoint"] == "best"]
    wide = table.pivot(index="record_id", columns="class_id", values=["y_true", "y_prob"]).sort_index()
    y = wide["y_true"][list(SUPERCLASSES)].to_numpy(dtype=bool)
    p = wide["y_prob"][list(SUPERCLASSES)].to_numpy(dtype=np.float64)
    noisy = table.groupby("record_id")["is_artifact"].first().sort_index().to_numpy(dtype=bool)
    out = {
        "overall": macro_auroc(y, p),
        "artifact": macro_auroc(y[noisy], p[noisy]),
        "clean": macro_auroc(y[~noisy], p[~noisy]),
    }
    for k, name in enumerate(SUPERCLASSES):
        out[name] = float(roc_auc_score(y[:, k], p[:, k]))
    return out


def collect(protocol_dirs: Sequence[str | Path], split: str) -> list[dict]:
    """One row per finished trial of every protocol: its arm, dose, best step and scores on ``split``."""
    rows, source = [], {}
    for protocol_dir in map(Path, protocol_dirs):
        for entry in json.loads((protocol_dir / "plan.json").read_text(encoding="utf-8"))["trials"]:
            run_dir = protocol_dir / entry["run_dir"]
            predictions = run_dir / f"predictions_{split}.parquet"
            if not (run_dir / "result.json").exists() or not predictions.exists():
                continue  # still training, or not scored on this split yet
            spec = load_spec(run_dir)
            key = (spec.regime, spec.arm, spec.replicate_seed)
            if key in source:
                raise SystemExit(f"{'/'.join(map(str, key))} is in both {source[key]} and {protocol_dir}")
            source[key] = protocol_dir
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            rows.append(
                {
                    "regime": spec.regime,
                    "arm": spec.arm,
                    "seed": spec.replicate_seed,
                    "source": str(protocol_dir),
                    "family": spec.transform_family,
                    "series": (spec.transform_family, spec.sampling_policy),
                    "share": spec.transform_fraction,
                    "dose": dose(spec),
                    "transform": spec.transform,
                    "best_step": result["best_step"],
                    "max_steps": spec.max_steps,
                    "history": [(h["step"], h["val/auroc_overall"]) for h in result["val_history"]],
                    "scores": _scores(predictions, split),
                }
            )
    if not rows:
        raise SystemExit(
            f"no finished run in {', '.join(map(str, protocol_dirs))} has {split} predictions yet"
        )
    return rows


def paired(rows: list[dict], key: tuple[str, str], baseline: tuple[str, str]) -> dict[str, list[float]]:
    """Per-seed changes ``key`` minus ``baseline`` on every column, over the seeds both finished."""
    a = {r["seed"]: r["scores"] for r in rows if (r["regime"], r["arm"]) == key}
    b = {r["seed"]: r["scores"] for r in rows if (r["regime"], r["arm"]) == baseline}
    seeds = sorted(set(a) & set(b))
    return {column: [a[s][column] - b[s][column] for s in seeds] for column in COLUMNS}


def _arm_rows(rows: list[dict], regime: str, baseline: str) -> list[tuple[str, str]]:
    """Every (regime, arm) except the baseline, in plan order, with other regimes (the target) last."""
    keys = list(dict.fromkeys((r["regime"], r["arm"]) for r in rows))
    own = [k for k in keys if k[0] == regime and k[1] != baseline]
    return own + [k for k in keys if k[0] != regime]


def _label(key: tuple[str, str], regime: str) -> str:
    if key[0] != regime:
        return f"{key[1]}@{key[0]}  100% of patients (target)"
    return f"{key[1]}  {LABELS.get(key[1], '')}"


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}".replace("-", MINUS)


def _summary_stats(rows, regime, baseline, reference):
    out = {}
    for key in _arm_rows(rows, regime, baseline):
        deltas = paired(rows, key, (regime, baseline))
        if not deltas["overall"]:
            continue
        arm_rows = [r for r in rows if (r["regime"], r["arm"]) == key]
        out[key] = {
            "deltas": deltas,
            "dose": arm_rows[0]["dose"],
            "family": arm_rows[0]["family"],
            "series": arm_rows[0]["series"],
            "best_step": [r["best_step"] for r in sorted(arm_rows, key=lambda r: r["seed"])],
            "max_steps": arm_rows[0]["max_steps"],
        }
    return out


def dose_figure(rows, regime, baseline, reference, split, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    stats = _summary_stats(rows, regime, baseline, reference)
    target = paired(rows, reference, (regime, baseline))
    base_steps = [r["best_step"] for r in rows if (r["regime"], r["arm"]) == (regime, baseline)]
    doses = sorted({s["dose"] for k, s in stats.items() if k[0] == regime and s["series"] in SERIES})
    if not doses:
        raise SystemExit("no dose arm (standard augmentation or targeted edits) has finished yet")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.9), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    delta_axes, step_ax = axes[:3], axes[3]
    jitter = dict(zip(SERIES, (-0.05, 0.05), strict=True))

    def series(name):
        keys = [k for k, s in stats.items() if k[0] == regime and s["series"] == name]
        return sorted(keys, key=lambda k: stats[k]["dose"])

    def marks(ax, x, values, color):
        ax.plot(
            [x - SEED_OFFSET] * len(values),
            values,
            "o",
            ms=4,
            color=color,
            alpha=0.5,
            mec=SURFACE,
            mew=0.8,
            zorder=4,
        )
        ax.plot([x], [_mean(values)], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5)

    for ax, part in zip(delta_axes, SLICES, strict=True):
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        if target["overall"]:
            t_mean, t_sd = (
                _mean(target[part]),
                float(np.std(target[part], ddof=1)) if len(target[part]) > 1 else 0,
            )
            ax.axhspan(t_mean - t_sd, t_mean + t_sd, color=REFERENCE_COLOR, alpha=0.10, lw=0, zorder=1)
            ax.axhline(t_mean, color=REFERENCE_COLOR, lw=2, zorder=2)
            ax.annotate(
                f"100% of patients {_signed(t_mean)}",
                (0.0, t_mean),
                xycoords=("axes fraction", "data"),
                xytext=(6, 5),
                textcoords="offset points",
                color=INK,
                fontsize=8.5,
            )
        for name, (_, color) in SERIES.items():
            keys = series(name)
            if not keys:
                continue
            x = np.log2([stats[k]["dose"] for k in keys]) + jitter[name]
            ax.plot(x, [_mean(stats[k]["deltas"][part]) for k in keys], color=color, lw=2, zorder=3)
            for xi, k in zip(x, keys, strict=True):
                values = stats[k]["deltas"][part]
                marks(ax, xi, values, color)
                if part == "overall":  # standard augmentation labelled below its marks, the editor above
                    below = name[0] == "generic"
                    ax.annotate(
                        k[1],
                        (xi, min(values) if below else max(values)),
                        xytext=(0, -13 if below else 6),
                        textcoords="offset points",
                        ha="center",
                        color=INK2,
                        fontsize=8,
                    )
        ax.set_title(f"Change vs A0, {SLICE_NAMES[part]}")
    _dose_axis(delta_axes, doses)
    low = min(min(ax.get_ylim()[0] for ax in delta_axes), 0)
    high = max(ax.get_ylim()[1] for ax in delta_axes)
    for ax in delta_axes:
        ax.set_ylim(low, high)
    for ax in delta_axes[1:]:
        ax.tick_params(labelleft=False)
    delta_axes[0].set_ylabel(f"Change in macro-AUROC ({split})")

    if base_steps:
        step_ax.axhline(_mean(base_steps), color=MUTED, lw=1, zorder=1)
        step_ax.annotate(
            "A0",
            (1.0, _mean(base_steps)),
            xycoords=("axes fraction", "data"),
            xytext=(-6, 4),
            textcoords="offset points",
            ha="right",
            color=MUTED,
            fontsize=8,
        )
    budget = max(s["max_steps"] for s in stats.values())
    step_ax.axhline(budget, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
    step_ax.annotate(
        "budget",
        (0.0, budget),
        xycoords=("axes fraction", "data"),
        xytext=(6, -12),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
    )
    for name, (_, color) in SERIES.items():
        keys = series(name)
        if not keys:
            continue
        x = np.log2([stats[k]["dose"] for k in keys]) + jitter[name]
        step_ax.plot(x, [_mean(stats[k]["best_step"]) for k in keys], color=color, lw=2, zorder=3)
        for xi, k in zip(x, keys, strict=True):
            marks(step_ax, xi, stats[k]["best_step"], color)
    _dose_axis([step_ax], doses)
    step_ax.set_ylim(0, budget * 1.08)
    step_ax.set_title("Step validation picked")
    step_ax.set_ylabel("Optimizer step")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", color=color, lw=2, ms=7, mec=SURFACE, mew=2, label=label)
        for name, (label, color) in SERIES.items()
        if series(name)
    ]
    handles += [
        Line2D(
            [], [], color=REFERENCE_COLOR, lw=2, label="Target: 100% of patients minus A0 (mean ± SD band)"
        ),
        Line2D([], [], marker="o", ls="", ms=4, color=MUTED, alpha=0.6, label="One seed"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.925),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    fig.suptitle(
        "How much augmentation helps, and when it starts to cost",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="semibold",
    )
    seeds = len({r["seed"] for r in rows})
    fig.text(
        0.01,
        0.015,
        f"Dose = share of the batch augmented x strength of A1's envelope (A1 = 1/2). Paired "
        f"change over A0, seed by seed; {seeds} seeds; {split} split, checkpoint validation chose.",
        color=MUTED,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.78, bottom=0.17, wspace=0.14)
    box = step_ax.get_position()  # room for the step panel's own y label
    step_ax.set_position([box.x0 + 0.025, box.y0, box.width - 0.025, box.height])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# Validation curves, three panels. Color follows the arm in every panel: A0 gray, A1 blue, the editor orange,
# the target aqua. The dose ladder takes an ordinal blue ramp; the techniques take categorical slots 4-8.
CURVE_PANELS = (
    ("How much standard augmentation", ("S1", "S2", "A1", "S3", "S4")),
    ("Targeted artifact edits: half vs every ECG", ("A3", "E1")),
    ("One technique each, on half the batch", ("A1", "T1", "T2", "T3", "T4", "T5")),
    ("SIMVEA_h on half vs every ECG, with A1 and time stretch", ("A1", "T4", "T6", "T6W")),
)
# The eight categorical slots are taken by then, so SIMVEA_h (T6 and T6W), the only new arms of its panel, is
# drawn in ink rather than a ninth hue.
CURVE_COLORS = {
    "S1": "#86b6ef", "S2": "#5598e7", "A1": "#2a78d6", "S3": "#1c5cab", "S4": "#104281",
    "A3": "#eb6834", "E1": "#eb6834",
    "T1": "#eda100", "T2": "#e87ba4", "T3": "#008300", "T4": "#6250d6", "T5": "#e34948",
    "T6": INK, "T6W": INK,
}  # fmt: skip
CURVE_DASHED = {"A3", "T6"}  # half the batch, against the whole batch (E1, T6W) in the same hue
CURVE_FLOOR, CURVE_CEILING = 0.87, 0.96


def curves_figure(rows, regime, baseline, reference, path: Path) -> Path:
    """Mean validation macro-AUROC on all records over steps, per arm, with A0 and the target."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()

    def curve(key):
        histories = [dict(r["history"]) for r in rows if (r["regime"], r["arm"]) == key]
        if not histories:
            return None
        steps = sorted(set.intersection(*(set(h) for h in histories)))
        return np.array(steps), np.array([np.mean([h[s] for h in histories]) for s in steps])

    def draw(ax, key, color, label, dashed=False, lw=2.0):
        found = curve(key)
        if found is None:
            return
        steps, mean = found
        ax.plot(steps, mean, color=color, lw=lw, ls="--" if dashed else "-", label=label, zorder=3)
        peak = int(np.argmax(mean))
        if mean[peak] >= CURVE_FLOOR:
            ax.plot(steps[peak], mean[peak], "o", ms=7, color=color, mec=SURFACE, mew=1.5, zorder=4)

    # A panel is drawn when it has an arm with runs that no earlier panel showed.
    panels, seen = [], set()
    for title, arms in CURVE_PANELS:
        found = [arm for arm in arms if curve((regime, arm)) is not None]
        if set(found) - seen:
            panels.append((title, found))
        seen |= set(found)
    if len(panels) == 4:  # two by two, so each panel keeps the width it has in a row of three
        fig, grid = plt.subplots(2, 2, figsize=(14, 9.6), sharey=True)
        axes, first_column = list(grid.flat), grid[:, 0]
    else:
        fig, row = plt.subplots(1, len(panels), figsize=(16, 5.0), sharey=True, squeeze=False)
        axes, first_column = list(row.flat), row[:, 0]
    for ax, (title, arms) in zip(axes, panels, strict=True):
        draw(ax, reference, REFERENCE_COLOR, "100% of patients (target)")
        draw(ax, (regime, baseline), MUTED, f"{baseline}  {LABELS[baseline]}", lw=1.6)
        for arm in arms:
            draw(ax, (regime, arm), CURVE_COLORS[arm], f"{arm}  {LABELS[arm]}", dashed=arm in CURVE_DASHED)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Optimizer step", color=INK2)
        budget = max(r["max_steps"] for r in rows)
        ax.set_xlim(0, budget)
        ticks = np.linspace(0, budget, 6)
        ax.set_xticks(ticks, [f"{t / 1000:g}k" if t else "0" for t in ticks])
        ax.set_ylim(CURVE_FLOOR, CURVE_CEILING)
        # Late steps above every curve are empty: every arm, and the target, has declined by then.
        ax.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    for ax in first_column:
        ax.set_ylabel("Validation macro-AUROC, all records", color=INK2)
    below = [
        arm
        for arm in dict.fromkeys(arm for _, arms in panels for arm in arms)
        if (found := curve((regime, arm))) is not None and np.mean(found[1] < CURVE_FLOOR) > 0.5
    ]
    note = f" The axis starts at {CURVE_FLOOR}, which cuts off the first steps of slower arms" + (
        f"; {', '.join(below)} stays below it for most of training." if below else "."
    )
    fig.suptitle(
        "When each arm peaks on validation, and how fast it overfits after",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        0.01,
        "Mean over seeds of validation macro-AUROC every 100 steps; dot = the mean curve's peak." + note,
        color=MUTED,
        fontsize=8,
    )
    if len(panels) == 4:
        fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.09, wspace=0.06, hspace=0.3)
    else:
        fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.14, wspace=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _dose_axis(axes, doses: Sequence[float]) -> None:
    from fractions import Fraction

    ticks = np.log2(doses)
    names = [str(Fraction(d).limit_denominator(64)) for d in doses]
    for ax in axes:
        ax.set_xticks(ticks, names, fontsize=8.5)
        ax.set_xlim(ticks.min() - 0.6, ticks.max() + 0.6)
        ax.set_xlabel("Augmentation dose (log scale)", color=INK2, fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.grid(True, axis="x", color=GRID, lw=0.6)


def classes_figure(rows, regime, baseline, reference, split, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    _style()
    stats = _summary_stats(rows, regime, baseline, reference)
    keys = list(stats)
    values = np.array([[_mean(stats[k]["deltas"][c]) for c in COLUMNS] for k in keys])
    agree = np.array(
        [[len({np.sign(v) for v in stats[k]["deltas"][c]}) == 1 for c in COLUMNS] for k in keys], dtype=bool
    )
    # Full color = the 100% target's overall gain, so each cell reads against more patients.
    # Larger losses saturate.
    target_gain = abs(_mean(stats[reference]["deltas"]["overall"])) if reference in stats else 0.0
    limit = target_gain or float(np.nanmax(np.abs(values))) or 0.01
    cmap = LinearSegmentedColormap.from_list("better_worse", DIVERGING)

    fig = plt.figure(figsize=(13, 0.42 * len(keys) + 2.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[len(COLUMNS), 2.2], wspace=0.04)
    ax, side = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(keys)):
        for j in range(len(COLUMNS)):
            # Gray text only on pale cells; on a saturated cell it would be unreadable.
            pale = abs(values[i, j]) < 0.5 * limit
            ax.text(
                j,
                i,
                _signed(values[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color=INK if agree[i, j] or not pale else MUTED,
                fontweight="semibold" if agree[i, j] else "normal",
            )
    # A 2px surface gap between cells, and a wider one between slices and classes.
    ax.set_xticks(np.arange(-0.5, len(COLUMNS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(keys)), minor=True)
    ax.grid(True, which="minor", color=SURFACE, lw=2)
    ax.axvline(len(SLICES) - 0.5, color=SURFACE, lw=6)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(len(COLUMNS)), [COLUMN_NAMES[c] for c in COLUMNS], fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(keys)), [_label(k, regime) for k in keys], fontsize=8.5)
    ax.text(
        1 / len(COLUMNS) * len(SLICES) / 2,
        1.09,
        "Macro-AUROC by slice",
        transform=ax.transAxes,
        ha="center",
        color=INK2,
        fontsize=9,
    )
    ax.text(
        (len(SLICES) + len(SUPERCLASSES) / 2) / len(COLUMNS),
        1.09,
        "AUROC by superclass, all records",
        transform=ax.transAxes,
        ha="center",
        color=INK2,
        fontsize=9,
    )

    side.set_xlim(0, 1)
    side.set_ylim(len(keys) - 0.5, -0.5)
    side.axis("off")
    side.text(0.05, -0.85, "best step", color=INK2, fontsize=8.5)
    for i, k in enumerate(keys):
        steps = stats[k]["best_step"]
        side.text(
            0.05,
            i,
            f"{_mean(steps):,.0f}",
            va="center",
            color=INK2,
            fontsize=8.5,
        )

    fig.suptitle(
        f"What each augmentation changes, by slice and by class ({split})",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
        fontweight="semibold",
    )
    fig.text(
        0.01,
        0.012,
        f"Paired change over A0 ({LABELS['A0']}), mean over seeds. Blue = better, red = worse; "
        "bold = every seed agrees on the sign; full color = the 100% target's overall gain.\n"
        "CD conduction disturbance, HYP hypertrophy, MI myocardial infarction, NORM normal, "
        "STTC ST/T change.",
        color=MUTED,
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.29, right=0.99, top=1 - 1.25 / fig.get_figheight(), bottom=0.6 / fig.get_figheight()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def summary(rows, regime, baseline, reference, split) -> str:
    stats = _summary_stats(rows, regime, baseline, reference)
    target = paired(rows, reference, (regime, baseline))
    target_gap = _mean(target["overall"])
    lines = [
        f"# Augmentation sweep on {regime}: paired change over {baseline} ({split})",
        "",
        f"Mean over seeds of arm minus {baseline}, seed by seed, at the checkpoint validation chose. "
        "`k/n` = seeds better than A0. Gap closed = change on all records / the 100% target's change "
        f"({_signed(target_gap, 4)}). Dose = share of the batch augmented x strength of A1's envelope.",
        "",
    ]
    sources = list(dict.fromkeys(r["source"] for r in rows if "source" in r))
    if len(sources) > 1:
        lines += [
            f"Runs from {', '.join(sources)}, paired seed by seed: arms of the later protocols were planned "
            f"after {sources[0]}'s test claim. Their CIs: `syntide.eval.across`.",
            "",
        ]
    lines += [
        "| Arm | What | Dose | All | Noisy | Clean | " + " | ".join(SUPERCLASSES) + " | Gap closed | "
        "Best step |",
        "|---|---|---|---|---|---|" + "---|" * len(SUPERCLASSES) + "---|---|",
    ]
    for key, s in stats.items():
        d = s["deltas"]

        def cell(column, d=d):
            better = sum(v > 0 for v in d[column])
            return f"{_mean(d[column]):+.4f} ({better}/{len(d[column])})"

        closed = _mean(d["overall"]) / target_gap if target_gap > 0 and key != reference else None
        dose_text = f"{s['dose']:.3g}" if key[0] == regime and s["family"] != "none" else "-"
        closed_text = "-" if closed is None else f"{closed:.0%}"
        name, what = _label(key, regime).split("  ", 1)
        lines.append(
            f"| {name} | {what} | "
            f"{dose_text} | {cell('overall')} | {cell('artifact')} | "
            f"{cell('clean')} | " + " | ".join(f"{_mean(d[c]):+.4f}" for c in SUPERCLASSES) + " | "
            f"{closed_text} | "
            f"{', '.join(f'{v:,}' for v in s['best_step'])} |"
        )
    late = [f"{k[1]}" for k, s in stats.items() if max(s["best_step"]) > 0.8 * s["max_steps"]]
    lines += [
        "",
        (
            f"Picked past 80% of the budget, so they may not have peaked: {', '.join(late)}."
            if late
            else "Every arm peaked before 80% of its budget."
        ),
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m syntide.eval.aug_figure",
        description="Plot the paired gain and cost of each augmentation dose and technique over A0.",
    )
    parser.add_argument(
        "protocol_dirs",
        nargs="+",
        help="protocol runs directories, e.g. runs/ptbxl-aug-sweep; outputs go to the first one",
    )
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--regime", default="scarcity50")
    parser.add_argument("--baseline", default="A0")
    parser.add_argument("--reference", default="full100/C2", help="REGIME/ARM of the 100% target")
    parser.add_argument("--hide", nargs="*", default=[], help="arms to leave out of the figures, e.g. S4")
    args = parser.parse_args(argv)
    protocol_dir = Path(args.protocol_dirs[0])
    reference = tuple(args.reference.split("/", 1))
    rows = collect(args.protocol_dirs, args.split)
    drawn = shown(rows, args.hide, args.regime, args.baseline, reference)
    figures = protocol_dir / "figures"
    print(
        dose_figure(
            drawn, args.regime, args.baseline, reference, args.split, figures / f"aug_dose_{args.split}.png"
        )
    )
    print(
        classes_figure(
            drawn,
            args.regime,
            args.baseline,
            reference,
            args.split,
            figures / f"aug_classes_{args.split}.png",
        )
    )
    print(curves_figure(drawn, args.regime, args.baseline, reference, figures / "aug_curves.png"))
    text = summary(rows, args.regime, args.baseline, reference, args.split)
    out = protocol_dir / f"aug_summary_{args.split}.md"
    out.write_text(text, encoding="utf-8")
    print(out)
    print(text)


if __name__ == "__main__":
    main()
