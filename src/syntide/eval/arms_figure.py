"""``python -m syntide.eval.arms_figure runs/<protocol> [runs/<extension> ...] [--hide S4]``: every arm next
to the 100%-data target.

Reads each run's ``result.json`` (validation scores at the best checkpoint, which validation chose) and
``test_scores.json`` (the held-out test fold, written once by ``syntide.eval.holdout``). Further protocol
directories add arms planned after the first protocol's test claim (``ptbxl-aug-simvea``'s T6), each scored
under its own claim. ``--hide`` leaves arms out of the figures only. Writes to the first protocol directory:

- ``figures/arms_overall.png``: macro-AUROC per arm on validation and on test. Each seed is a small dot,
  the mean ± SD a large one, and the reference arm (by default C2 on 100% of patients) a line with its
  ± SD band. The number above each arm is its gap to that line.
- ``figures/arms_slices.png``: the same for noisy records (any PTB-XL signal-quality flag) and clean ones.
- ``arms_summary.md``: the table behind both figures, each arm's paired change over the baseline arm
  seed by seed (same seed = same initialization and batch order), and a convergence check.

Validation picked each checkpoint, so its numbers lean optimistic; the test fold is the unbiased one.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from syntide.trial import load_spec

SPLITS = ("val", "test")
SLICES = ("overall", "artifact", "clean")
SLICE_NAMES = {"overall": "all records", "artifact": "noisy records", "clean": "clean records"}
ARM_NAMES = {
    "A0": "clean\nonly",
    "A1": "standard\naugmentation",
    "A2": "synthetic,\nmatched",
    "A3": "synthetic,\ntargeted",
    "C1": "natural\nmix",
    "C2": "every ECG\nof the pool",
    # The augmentation sweep (ptbxl_aug_sweep).
    "S1": "standard,\ndose 1/32",
    "S2": "standard,\ndose 1/8",
    "S3": "standard,\ndose 1",
    "S4": "standard,\ndose 2",
    "E1": "targeted,\nevery ECG",
    "T1": "time\nmasking",
    "T2": "lead\ndropout",
    "T3": "amplitude\nscaling",
    "T4": "time\nstretch",
    "T5": "mixup",
    "T6": "SIMVEA_h\n(chest leads)",
}
# Next to the sweep's dose ladder, A1 reads as one more dose.
SWEEP_NAMES = {"A1": "standard,\ndose 1/2"}
SWEEP_ARMS = {"S1", "S2", "S3", "S4"}
# Validated with the dataviz palette checks (light surface, all pairs): blue, orange, aqua.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GROUPS = {
    "clean": ("Clean ECGs only", MUTED),
    "augmented": ("Clean ECGs + augmentation or synthetic edits", "#2a78d6"),
    "real": ("Real noisy ECGs included", "#eb6834"),
}
REFERENCE_COLOR = "#1baf7a"
MINUS = "\u2212"  # a typographic minus for signed gaps
CONVERGENCE_SHARE = 0.8  # a best step past 80% of the budget may mean the run had not peaked yet


def collect(protocol_dirs: Sequence[str | Path]) -> list[dict]:
    """One row per planned trial of every protocol: arm, seed, budget, val and test AUROCs at the best
    checkpoint. A (regime, arm, seed) planned in two protocols is refused."""
    rows, source = [], {}
    for protocol_dir in map(Path, protocol_dirs):
        for row in _collect_one(protocol_dir):
            key = (row["regime"], row["arm"], row["seed"])
            if key in source:
                raise SystemExit(f"{'/'.join(map(str, key))} is in both {source[key]} and {protocol_dir}")
            source[key] = protocol_dir
            rows.append(row)
    return rows


def shown(rows: list[dict], hide: Sequence[str], regime: str, baseline: str, reference: tuple[str, str]):
    """The rows a figure draws: every arm of ``regime`` but the hidden ones. The baseline and the target stay,
    since every figure is drawn against them."""
    hidden = {(regime, arm) for arm in hide}
    if clash := hidden & {(regime, baseline), reference}:
        raise SystemExit(f"cannot hide {', '.join(arm for _, arm in sorted(clash))}: every figure needs it")
    return [r for r in rows if (r["regime"], r["arm"]) not in hidden]


def _collect_one(protocol_dir: Path) -> list[dict]:
    rows = []
    for entry in json.loads((protocol_dir / "plan.json").read_text(encoding="utf-8"))["trials"]:
        run_dir = protocol_dir / entry["run_dir"]
        spec = load_spec(run_dir)
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        scores_path = run_dir / "test_scores.json"
        if not scores_path.exists():
            raise SystemExit(f"{scores_path} is missing; run python -m syntide.eval.holdout {protocol_dir}")
        test = json.loads(scores_path.read_text(encoding="utf-8"))
        if test["trial_id"] != spec.trial_id or test["step"] != result["best_step"]:
            raise SystemExit(f"{scores_path} does not score this trial's best checkpoint")
        val = next(e for e in result["val_history"] if e["step"] == result["best_step"])
        rows.append(
            {
                "regime": spec.regime,
                "arm": spec.arm,
                "seed": spec.replicate_seed,
                "group": _group(spec.source_role, spec.transform_family),
                "best_step": result["best_step"],
                "max_steps": spec.max_steps,
                "val": {s: val[f"val/auroc_{s}"] for s in SLICES},
                "test": {s: test[f"test/auroc_{s}"] for s in SLICES},
            }
        )
    return rows


def _arm_name(arm: str, arms: Sequence[str]) -> str:
    names = ARM_NAMES | SWEEP_NAMES if SWEEP_ARMS & set(arms) else ARM_NAMES
    return names.get(arm, "")


def _group(source_role: str, family: str) -> str:
    if source_role != "depleted":
        return "real"
    return "clean" if family == "none" else "augmented"


def _values(rows: list[dict], regime: str, arm: str, split: str, part: str) -> dict[int, float]:
    return {r["seed"]: r[split][part] for r in rows if r["regime"] == regime and r["arm"] == arm}


def _mean_sd(values: dict[int, float]) -> tuple[float, float]:
    v = np.array(list(values.values()))
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": MUTED,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
        }
    )


def _panel(ax, rows, regime, arms, reference: tuple[str, str], split, part, title) -> None:
    ref_mean, ref_sd = _mean_sd(_values(rows, *reference, split, part))
    ax.axhspan(ref_mean - ref_sd, ref_mean + ref_sd, color=REFERENCE_COLOR, alpha=0.10, lw=0, zorder=1)
    ax.axhline(ref_mean, color=REFERENCE_COLOR, lw=2, zorder=2)
    ax.annotate(
        f"100% of patients {ref_mean:.3f}",
        (0.0, ref_mean),
        xycoords=("axes fraction", "data"),
        xytext=(6, 5),
        textcoords="offset points",
        ha="left",
        color=INK,
        fontsize=9,
    )
    for x, arm in enumerate(arms):
        values = _values(rows, regime, arm, split, part)
        group = next(r["group"] for r in rows if r["regime"] == regime and r["arm"] == arm)
        color = GROUPS[group][1]
        mean, sd = _mean_sd(values)
        ax.plot([x - 0.2] * len(values), list(values.values()), "o", ms=5, color=color, alpha=0.6,
                mec=SURFACE, mew=1, zorder=3)  # fmt: skip
        ax.plot([x, x], [mean - sd, mean + sd], color=color, lw=2, solid_capstyle="round", zorder=4)
        ax.plot([x], [mean], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5)
        top = max(mean + sd, *values.values())  # above every mark of this arm
        ax.annotate(f"{mean - ref_mean:+.3f}".replace("-", MINUS), (x, top), xytext=(0, 6),
                    textcoords="offset points", ha="center", color=INK2, fontsize=8.5)  # fmt: skip
    ax.set_xticks(range(len(arms)), [f"{arm}\n{_arm_name(arm, arms)}" for arm in arms], fontsize=8.5)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_title(title)
    ax.tick_params(axis="x", length=0)


def _share_y(axes, rows, regime, arms, reference: tuple[str, str], split_parts) -> None:
    values = []
    for split, part in split_parts:
        for arm in [*arms, None]:
            v = (
                _values(rows, *reference, split, part)
                if arm is None
                else _values(rows, regime, arm, split, part)
            )
            values += list(v.values())
    low, high = min(values), max(values)
    pad = 0.12 * (high - low or 0.01)
    for ax in axes:
        ax.set_ylim(low - pad, high + 2.2 * pad)


def _legend(fig, top: float, groups: set[str]) -> None:
    """Only the groups the figure shows, in GROUPS order."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color=color, mec=SURFACE, mew=2, label=name)
        for group, (name, color) in GROUPS.items()
        if group in groups
    ]
    handles += [
        Line2D([], [], color=REFERENCE_COLOR, lw=2, label="Target: 100% of patients (mean ± SD band)"),
        Line2D([], [], marker="o", ls="", ms=5, color=MUTED, alpha=0.6, label="One seed"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, top), ncol=5, frameon=False,
               fontsize=8.5, labelcolor=INK2, handletextpad=0.4, columnspacing=1.4)  # fmt: skip


def figures(rows: list[dict], regime: str, reference: tuple[str, str], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = list(dict.fromkeys(r["arm"] for r in rows if r["regime"] == regime))
    seeds = len({r["seed"] for r in rows})
    groups = {r["group"] for r in rows if r["regime"] == regime}
    split_titles = {"val": "Validation (fold 9, picks the checkpoint)", "test": "Test (fold 10, held out)"}
    caption = (
        f"Numbers above each arm: its gap to the 100% target. Arms train on {regime}; "
        f"{seeds} seeds each; macro-AUROC of the checkpoint validation chose."
    )
    paths = []

    width = max(13.0, 1.35 * len(arms))  # about 0.65 in per arm and panel
    fig, axes = plt.subplots(1, 2, figsize=(width, 4.8), sharey=True)
    for ax, split in zip(axes, SPLITS, strict=True):
        _panel(ax, rows, regime, arms, reference, split, "overall", split_titles[split])
    _share_y(axes, rows, regime, arms, reference, [(s, "overall") for s in SPLITS])
    axes[0].set_ylabel("Macro-AUROC, all records")
    fig.suptitle("Does augmenting 50% of the patients reach the performance of 100%?", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="semibold")  # fmt: skip
    _legend(fig, top=0.925, groups=groups)
    fig.text(0.01, 0.015, caption, color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.21, wspace=0.05)
    paths.append(out_dir / "arms_overall.png")
    fig.savefig(paths[-1], dpi=200)
    plt.close(fig)

    fig, grid = plt.subplots(2, 2, figsize=(width, 8.6), sharey="row")
    for row_axes, part in zip(grid, ("artifact", "clean"), strict=True):
        for ax, split in zip(row_axes, SPLITS, strict=True):
            _panel(
                ax, rows, regime, arms, reference, split, part, f"{split_titles[split]}: {SLICE_NAMES[part]}"
            )
        _share_y(row_axes, rows, regime, arms, reference, [(s, part) for s in SPLITS])
        row_axes[0].set_ylabel(f"Macro-AUROC, {SLICE_NAMES[part]}")
    fig.suptitle("The same comparison on noisy and on clean records", x=0.01, ha="left", color=INK,
                 fontsize=12, fontweight="semibold")  # fmt: skip
    _legend(fig, top=0.962, groups=groups)
    fig.text(0.01, 0.008, caption + " Noisy = any PTB-XL signal-quality flag.", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.885, bottom=0.105, wspace=0.05, hspace=0.42)
    paths.append(out_dir / "arms_slices.png")
    fig.savefig(paths[-1], dpi=200)
    plt.close(fig)
    return paths


def summary(rows: list[dict], regime: str, reference: tuple[str, str], baseline: str) -> str:
    """The markdown table behind the figures, paired changes over the baseline, and a convergence check."""
    arms = list(dict.fromkeys(r["arm"] for r in rows if r["regime"] == regime))
    lines = [
        f"# {regime}: every arm against {reference[1]} on {reference[0]}",
        "",
        "Macro-AUROC at the checkpoint validation chose, mean ± SD over seeds. Gap = arm mean minus the "
        "100% target's mean. Noisy = records with any PTB-XL signal-quality flag.",
        "",
        "| Arm | Val all | Test all | Test noisy | Test clean | Test gap, all | Test gap, noisy |",
        "|---|---|---|---|---|---|---|",
    ]
    ref = {(s, p): _mean_sd(_values(rows, *reference, s, p)) for s in SPLITS for p in SLICES}

    def cell(values):
        mean, sd = _mean_sd(values)
        return f"{mean:.4f} ± {sd:.4f}"

    for arm in arms:
        v = {(s, p): _values(rows, regime, arm, s, p) for s in SPLITS for p in SLICES}
        gap = {p: _mean_sd(v["test", p])[0] - ref["test", p][0] for p in ("overall", "artifact")}
        lines.append(
            f"| {arm} {_arm_name(arm, arms).replace(chr(10), ' ')} | {cell(v['val', 'overall'])} | "
            f"{cell(v['test', 'overall'])} | {cell(v['test', 'artifact'])} | {cell(v['test', 'clean'])} | "
            f"{gap['overall']:+.4f} | {gap['artifact']:+.4f} |"
        )
    r = {(s, p): _values(rows, *reference, s, p) for s in SPLITS for p in SLICES}
    lines.append(
        f"| **Target: {reference[1]}, {reference[0]}** | {cell(r['val', 'overall'])} | "
        f"{cell(r['test', 'overall'])} | {cell(r['test', 'artifact'])} | {cell(r['test', 'clean'])} | 0 | 0 |"
    )

    lines += [
        "",
        f"## Paired change over {baseline} on the test fold, seed by seed",
        "",
        f"Same seed = same initialization, batch order and transform slots, so arm minus {baseline} isolates "
        "what the arm changes. Seed values are listed in seed order.",
        "",
        "| Arm | All: mean | All: per seed | Noisy: mean | Noisy: per seed | Seeds better, all / noisy |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        if arm == baseline:
            continue
        out = []
        better = []
        for part in ("overall", "artifact"):
            base = _values(rows, regime, baseline, "test", part)
            values = _values(rows, regime, arm, "test", part)
            deltas = [values[seed] - base[seed] for seed in sorted(values) if seed in base]
            out += [f"{np.mean(deltas):+.4f}", ", ".join(f"{d:+.4f}" for d in deltas)]
            better.append(f"{sum(d > 0 for d in deltas)}/{len(deltas)}")
        lines.append(f"| {arm} | {' | '.join(out)} | {' / '.join(better)} |")

    late = [r for r in rows if r["best_step"] > CONVERGENCE_SHARE * r["max_steps"]]
    steps = defaultdict(list)
    for r in rows:
        steps[f"{r['regime']}/{r['arm']}"].append(r["best_step"])
    lines += [
        "",
        "## Convergence check",
        "",
        "Best step per seed (budget "
        + ", ".join(sorted({f"{r['max_steps']:,}" for r in rows}))
        + " steps): "
        + "; ".join(f"{key} {', '.join(f'{s:,}' for s in value)}" for key, value in steps.items())
        + ".",
        "",
        (
            f"{len(late)} runs picked a checkpoint past {CONVERGENCE_SHARE:.0%} of the budget; "
            "they may not have peaked."
            if late
            else f"Every run peaked before {CONVERGENCE_SHARE:.0%} of its budget, "
            "so the budget cut none of them short."
        ),
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m syntide.eval.arms_figure",
        description="Plot every arm against the 100%-data target on validation and on the test fold.",
    )
    parser.add_argument(
        "protocol_dirs",
        nargs="+",
        help="protocol runs directories, e.g. runs/ptbxl-artifact-50; outputs go to the first one",
    )
    parser.add_argument("--regime", default="scarcity50", help="the regime whose arms are compared")
    parser.add_argument("--reference", default="full100/C2", help="REGIME/ARM of the target line")
    parser.add_argument("--baseline", default="A0", help="the arm paired changes are measured from")
    parser.add_argument("--hide", nargs="*", default=[], help="arms to leave out of the figures, e.g. S4")
    args = parser.parse_args(argv)
    protocol_dir = Path(args.protocol_dirs[0])
    reference = tuple(args.reference.split("/", 1))
    rows = collect(args.protocol_dirs)
    drawn = shown(rows, args.hide, args.regime, args.baseline, reference)
    for path in figures(drawn, args.regime, reference, protocol_dir / "figures"):
        print(path)
    text = summary(rows, args.regime, reference, args.baseline)
    (protocol_dir / "arms_summary.md").write_text(text, encoding="utf-8")
    print(protocol_dir / "arms_summary.md")
    print(text)


if __name__ == "__main__":
    main()
