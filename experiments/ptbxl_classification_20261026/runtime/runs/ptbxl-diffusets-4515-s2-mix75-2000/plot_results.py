"""Plot three-seed S2 + DiffuSETS trials against the saved scaling and S2 runs.

Run with synthetic_data_engine/.venv/bin/python. This reads saved results and
predictions only; no training, inference, or historical artifact changes occur.
All figures and their full-precision data are written to this run's figures/.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RUNS, ENGINE = ROOT.parent, ROOT.parent.parent
OUT = ROOT / "figures"
SHARES = (10, 20, 30, 50, 70, 100)
PARTS = ("overall", "artifact", "clean")
PART_LABEL = {"overall": "All records", "artifact": "Noisy records", "clean": "Clean slice"}
COLORS = {"overall": "#2a78d6", "artifact": "#eb6834", "clean": "#1baf7a"}
PURPLE, BERRY, GREY, INK, SURFACE = "#8b46ba", "#c34367", "#898781", "#20201e", "#fcfcfb"
BUDGET = 2000
GROUPS = ("S2", "S2_generated")
GROUP_LABEL = {"S2": "S2", "S2_generated": "S2 + 4,515 generated"}
GROUP_COLOR = {"S2": PURPLE, "S2_generated": BERRY}
GROUP_MARKER = {"S2": "D", "S2_generated": "P"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_run(folder, group, share, real_records, generated=0):
    result, spec = read(folder / "result.json"), read(folder / "trial.json")
    if result["status"] != "complete" or result["trial_id"] != spec["trial_id"]:
        raise ValueError(f"Incomplete or mismatched trial: {folder}")
    history = result["val_history"]
    if [h["step"] for h in history] != list(range(100, result["final_step"] + 1, 100)):
        raise ValueError(f"Validation grid differs: {folder}")
    best = max(history, key=lambda h: h["val/auroc_overall"])
    if best["step"] != result["best_step"] or best["val/auroc_overall"] != result["best_val_auroc"]:
        raise ValueError(f"Validation checkpoint selection differs: {folder}")
    test_path = folder / "test_scores.json"
    test = read(test_path) if test_path.exists() else None
    if test is not None and (test["trial_id"] != spec["trial_id"] or test["step"] != best["step"]):
        raise ValueError(f"Test scores belong to another checkpoint: {folder}")
    return {"folder": folder, "source": str(folder.relative_to(RUNS)), "group": group,
            "share_percent": share, "real_records": real_records, "generated_records": generated,
            "seed": result["replicate_seed"], "result": result, "spec": spec,
            "history": history, "test": test}


def selected(run, scenario, limit=None):
    history = [h for h in run["history"] if limit is None or h["step"] <= limit]
    best = max(history, key=lambda h: h["val/auroc_overall"])
    smooth = np.convolve([h["val/auroc_overall"] for h in history], np.ones(5) / 5, "valid")
    peak = int(np.argmax(smooth))
    # A saved test score can only accompany its actual validation-selected checkpoint.
    test = run["test"] if run["test"] and run["test"]["step"] == best["step"] else None
    return {"scenario": scenario, "group": run["group"], "source": run["source"], "seed": run["seed"],
            "real_patient_share_percent": run["share_percent"], "real_records": run["real_records"],
            "generated_records": run["generated_records"], "training_steps": run["result"]["final_step"],
            "selection_budget_steps": history[-1]["step"], "best_step": best["step"],
            "smoothed_peak_step": history[peak + 2]["step"], "smoothed_peak_auroc": float(smooth[peak]),
            "smoothed_peak_at_boundary": peak in (0, len(smooth) - 1),
            **{f"val_{part}": best[f"val/auroc_{part}"] for part in PARTS},
            "val_loss": best["val/loss"],
            **{f"test_{part}": test[f"test/auroc_{part}"] if test else None for part in PARTS},
            "test_loss": test["test/loss"] if test else None,
            "test_checkpoint_matches_selection": test is not None,
            **{f"final_val_{part}": history[-1][f"val/auroc_{part}"] for part in PARTS}}


def stats(rows, metric):
    values = [r[metric] for r in rows if r[metric] is not None]
    return {"mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None, "n": len(values)}


def aggregate(rows):
    metrics = [f"{split}_{part}" for split in ("val", "test", "final_val") for part in PARTS]
    metrics += ["val_loss", "test_loss", "best_step", "smoothed_peak_step", "smoothed_peak_auroc"]
    return {group: {metric: stats([r for r in rows if r["group"] == group], metric)
                    for metric in metrics} for group in GROUPS}


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def histories(runs):
    rows = []
    for run in runs:
        for h in run["history"]:
            rows.append({"group": run["group"], "source": run["source"], "seed": run["seed"],
                         "real_patient_share_percent": run["share_percent"], "real_records": run["real_records"],
                         "generated_records": run["generated_records"], "step": h["step"],
                         **{f"val_{part}": h[f"val/auroc_{part}"] for part in PARTS},
                         "val_loss": h["val/loss"], "within_2000_steps": h["step"] <= BUDGET})
    return rows


def per_class(runs):
    rows = []
    for run in runs:
        for split in ("val", "test"):
            path = run["folder"] / f"predictions_{split}.parquet"
            if not path.exists():
                continue
            a = {k: np.asarray(v) for k, v in pq.read_table(path).to_pydict().items()}
            for cls in sorted(set(a["class_id"])):
                for part in PARTS:
                    mask = (a["checkpoint"] == "best") & (a["class_id"] == cls)
                    if part != "overall":
                        mask &= a["is_artifact"] if part == "artifact" else ~a["is_artifact"]
                    if set(a["step"][mask]) != {run["result"]["best_step"]}:
                        raise ValueError(f"Prediction checkpoint mismatch: {path}")
                    y, p = a["y_true"][mask].astype(bool), a["y_prob"][mask].astype(float)
                    rows.append({"group": run["group"], "source": run["source"], "seed": run["seed"],
                                 "split": split, "slice": part, "class": str(cls),
                                 "step": run["result"]["best_step"], "n_records": len(y),
                                 "n_positive": int(y.sum()),
                                 "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
                                 "selection": "saved global best overall validation checkpoint"})
    return rows


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                        "font.size": 10.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "axes.spines.top": False,
                        "axes.spines.right": False, "xtick.color": "#696860", "ytick.color": "#696860",
                        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 13,
                        "text.color": INK, "svg.fonttype": "none"})


def save(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=200)
    plt.close(fig)


def metric_table(ax, ag):
    ax.axis("off")
    ax.set_title("Validation and test · all records", pad=15)
    cells = []
    for group in GROUPS:
        cells.append([GROUP_LABEL[group]] + [f"{ag[group][f'{split}_overall']['mean']:.4f} ± "
                                             f"{ag[group][f'{split}_overall']['sd']:.4f}"
                                             if ag[group][f'{split}_overall']['n'] == 3 else "Not available"
                                             for split in ("val", "test")])
    table = ax.table(cellText=cells, colLabels=["", "Validation", "Test"],
                     colWidths=[0.43, 0.285, 0.285], cellLoc="center", bbox=[0, 0.38, 1, 0.53])
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(SURFACE)
        cell.set_edgecolor("#e1e0d9")
        cell.set_linewidth(0.6 if row == 0 else 0.0)
        if row:
            cell.get_text().set_color(GROUP_COLOR[GROUPS[row - 1]])
        else:
            cell.get_text().set_weight("bold")
    delta_val = ag["S2_generated"]["val_overall"]["mean"] - ag["S2"]["val_overall"]["mean"]
    delta_test = ag["S2_generated"]["test_overall"]["mean"] - ag["S2"]["test_overall"]["mean"]
    ax.text(0, 0.20, f"S2 + generated − S2:  validation {delta_val:+.4f}  ·  test {delta_test:+.4f}",
            transform=ax.transAxes, fontsize=10.8, color=INK)
    ax.text(0, 0.025, "Mean ± sample SD, 3 seeds; the real training sets differ.", transform=ax.transAxes,
            fontsize=10, color="#696860")


def main_figure(scaling, ag, *, full, matched):
    fig = plt.figure(figsize=(18, 10.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1, 0.8],
                           left=0.065, right=0.97, top=0.80, bottom=0.20, wspace=0.24, hspace=0.46)
    ax, peak, table = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    title = "S2 + generated ECGs in the context of real-data scaling"
    fig.suptitle(title, x=0.065, y=0.97, ha="left", fontsize=23, fontweight="bold")
    fig.text(0.065, 0.915, "New trial: 8,579 real + 4,515 generated  ·  48 real / 16 generated per batch  ·  "
             "S2 on 12 real + 4 generated slots  ·  3 seeds", fontsize=12.5, color="#52514e")
    context = ("All checkpoint selections and smoothed peaks restricted to the first 2,000 steps"
               if matched else "Training budgets: scaling 12,000 steps; S2 10,000 steps  ·  New trial: 2,000 steps")
    fig.text(0.065, 0.865, context, fontsize=12, color="#696860")
    # Both established filenames now show the requested overall-only comparison.
    parts = ("overall",)
    for part in parts:
        ys = [r[f"val_{part}"] for r in scaling]
        ax.plot(SHARES, ys, "o-", color=COLORS[part], lw=2.6, ms=8, mec=SURFACE, mew=1.5,
                label=f"Real scaling · {PART_LABEL[part].lower()}")
        ax.annotate(f"{ys[-1]:.4f}", (100, ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=COLORS[part])
    for i, group in enumerate(GROUPS):
        metric = ag[group]["val_overall"]
        ax.errorbar(50, metric["mean"], yerr=metric["sd"], fmt=GROUP_MARKER[group], color=GROUP_COLOR[group],
                    ms=10, elinewidth=1.8, capsize=6, mec=SURFACE, mew=1.2, zorder=7)
        ax.annotate(f"{GROUP_LABEL[group]}\n{metric['mean']:.4f} ± {metric['sd']:.4f}",
                    (50, metric["mean"]), xytext=(0.64 if i == 0 else 0.52, 0.50 - i * 0.19), textcoords="axes fraction",
                    color=GROUP_COLOR[group], fontsize=11,
                    arrowprops={"arrowstyle": "-", "color": GROUP_COLOR[group], "lw": 1.1},
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 3})
    vals = [r[f"val_{part}"] for r in scaling for part in parts]
    vals += [ag[g]["val_overall"]["mean"] + sign * ag[g]["val_overall"]["sd"] for g in GROUPS for sign in (-1, 1)]
    ax.set(title="Validation AUROC by real-data share", xlabel="Real training patient pool (%)",
           ylabel="Macro-AUROC at the best overall validation checkpoint", xlim=(5, 112),
           ylim=(min(vals) - 0.004, max(vals) + 0.008))
    ax.set_xticks(SHARES)
    ax.grid(color="#e1e0d9", lw=0.9)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker=GROUP_MARKER[g], ls="", color=GROUP_COLOR[g], ms=8) for g in GROUPS]
    labels += [f"{GROUP_LABEL[g]} · overall, mean ± SD" for g in GROUPS]
    ax.legend(handles, labels, loc="lower right", fontsize=9.3, frameon=False, labelspacing=0.8)
    ax.text(0.025, 0.965, "Real-data line: seed 0\nAugmented points: 3 seeds, both at 50%",
            transform=ax.transAxes, va="top", color="#696860", fontsize=10, linespacing=1.5)

    steps = [r["smoothed_peak_step"] for r in scaling]
    peak.plot(SHARES, steps, "o-", color=COLORS["overall"], lw=2.4, ms=7, mec=SURFACE, mew=1.2)
    for x, y in zip(SHARES, steps, strict=True):
        peak.annotate(f"{y:,}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5)
    upper = max(steps)
    peak_handles = []
    for group in GROUPS:
        m = ag[group]["smoothed_peak_step"]
        peak.errorbar(50, m["mean"], yerr=m["sd"], fmt=GROUP_MARKER[group], color=GROUP_COLOR[group],
                      ms=8.5, elinewidth=1.7, capsize=5, mec=SURFACE, mew=1.1, zorder=7)
        upper = max(upper, m["mean"] + m["sd"])
        peak_handles.append(Line2D([], [], marker=GROUP_MARKER[group], ls="", color=GROUP_COLOR[group],
                                  label=f"{GROUP_LABEL[group]}: {m['mean']:,.0f} ± {m['sd']:,.0f}"))
    peak.set(title="Step where smoothed validation AUROC peaks", xlabel="Real training patient pool (%)",
             ylabel="Optimizer steps", xlim=(5, 105), ylim=(0, upper * 1.4))
    peak.set_xticks(SHARES)
    peak.grid(color="#e1e0d9", lw=0.9)
    peak.legend(handles=peak_handles, loc="upper left", frameon=False, fontsize=9.8)
    peak.text(0.97, 0.96, "Centered 5-validation average", ha="right", va="top", transform=peak.transAxes,
              fontsize=9.3, color="#696860")
    metric_table(table, ag)
    fig.text(0.065, 0.127, "S2 uses 6,389 artifact-depleted real ECGs; the new trial uses all 8,579 real ECGs plus generated data. "
             "The difference does not isolate a generated-data effect.", fontsize=10, color="#52514e")
    fig.text(0.065, 0.094, "Each seed selects on overall validation AUROC; test evaluates that checkpoint. Error bars are sample SD, not confidence intervals. "
             "A 2,000-step peak does not establish convergence.", fontsize=10, color="#52514e")
    fig.text(0.065, 0.061, "Generated diagnoses are conditioning-parent proxies. 2,212 generated ECGs use parents outside the 50% pool, all within training folds; "
             "no held-out patient overlap.", fontsize=10, color="#696860")
    fig.text(0.065, 0.028, "See scaling_zoom for separated 50% points. S2's selected checkpoints are already within 2,000 steps, "
             "so its saved test scores apply in both budget views.", fontsize=9.5, color=GREY)
    stem = "scaling_saturation_with_s2_generated" if full else "scaling_saturation_overall"
    save(fig, stem + ("_budget_matched" if matched else ""))


def zoom_plot(scaling, rows, ag):
    real = next(r for r in scaling if r["real_patient_share_percent"] == 50)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.25, wspace=0.30)
    fig.suptitle("Zoom at the 50% real patient pool", x=0.08, y=0.97, ha="left", fontsize=21, fontweight="bold")
    fig.text(0.08, 0.89, "All selection windows restricted to 2,000 steps  ·  Augmented points: 3-seed mean ± SD; small points show each seed",
             fontsize=11.5, color="#52514e")
    names = ["Real only\n8,579 ECGs", "S2\n6,389 real ECGs", "S2 + generated\n8,579 real + 4,515 generated"]
    for ax, metric, title, ylabel in [(axes[0], "val_overall", "All-records validation AUROC", "Macro-AUROC"),
                                      (axes[1], "smoothed_peak_step", "Smoothed validation peak step", "Optimizer steps")]:
        points = [real[metric]]
        ax.plot(0, real[metric], marker="o", color=COLORS["overall"], ms=9)
        for x, group in enumerate(GROUPS, 1):
            m = ag[group][metric]
            seeds = [r[metric] for r in rows if r["group"] == group]
            ax.scatter(x + np.array([-0.10, 0, 0.10]), seeds, s=28, color=GROUP_COLOR[group], alpha=0.45, zorder=3)
            ax.errorbar(x, m["mean"], yerr=m["sd"], fmt=GROUP_MARKER[group], color=GROUP_COLOR[group],
                        ms=10, capsize=6, elinewidth=2, zorder=4, mec=SURFACE, mew=1)
            points += [m["mean"] - m["sd"], m["mean"] + m["sd"], *seeds]
        span = max(points) - min(points)
        pad = max(span * 0.3, 0.0008 if metric == "val_overall" else 100)
        ax.set(title=title, ylabel=ylabel, xlim=(-0.45, 2.45), ylim=(min(points) - pad, max(points) + pad * 1.6))
        ax.set_xticks(range(3), names, fontsize=10)
        ax.grid(axis="y", color="#e1e0d9")
        fmt = ".4f" if metric == "val_overall" else ",.0f"
        ax.text(0, real[metric] + pad * 0.45, format(real[metric], fmt), ha="center", color=COLORS["overall"], fontsize=10)
        for x, group in enumerate(GROUPS, 1):
            m = ag[group][metric]
            ax.text(x, m["mean"] + m["sd"] + pad * 0.45,
                    f"{m['mean']:{fmt}} ± {m['sd']:{fmt}}", ha="center", color=GROUP_COLOR[group], fontsize=10)
        if metric == "val_overall":
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    fig.text(0.08, 0.105, "The x-axis separates recipes for readability; all draw their real records from a nominal 50% patient pool. "
             "S2 uses the depleted subset.", fontsize=10, color="#52514e")
    fig.text(0.08, 0.055, "Real-only scaling has one seed; S2 comparisons have three. Parent-derived generated labels are provisional. "
             "Changes in real data, generation and sampling are combined.", fontsize=10, color=GREY)
    save(fig, "scaling_zoom")


def learning_plot(scaling_runs, groups):
    baseline = next(r for r in scaling_runs if r["share_percent"] == 50)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.80, bottom=0.15, wspace=0.25, hspace=0.35)
    fig.suptitle("S2 + generated ECGs · validation learning curves", x=0.075, y=0.97, ha="left", fontsize=21, fontweight="bold")
    fig.text(0.075, 0.91, "2,000 optimizer steps  ·  Curves and shading: mean ± sample SD over 3 seeds  ·  Real-only reference: seed 0",
             fontsize=11.5, color="#52514e")
    for ax, metric, title in zip(axes.flat,
                               ["val/auroc_overall", "val/auroc_artifact", "val/auroc_clean", "val/loss"],
                               ["All records", "Noisy records", "Clean slice", "Validation loss"], strict=True):
        bh = [r for r in baseline["history"] if r["step"] <= BUDGET]
        x = [r["step"] for r in bh]
        ax.plot(x, [r[metric] for r in bh], color=GREY, lw=1.8, label="Real only · 8,579 ECGs")
        for group in GROUPS:
            values = np.asarray([[r[metric] for r in run["history"] if r["step"] <= BUDGET] for run in groups[group]])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(x, mean, color=GROUP_COLOR[group], lw=2.2, label=GROUP_LABEL[group])
            ax.fill_between(x, mean - sd, mean + sd, color=GROUP_COLOR[group], alpha=0.12, lw=0)
        ax.set(title=title, xlabel="Optimizer steps", ylabel="Binary cross-entropy" if metric == "val/loss" else "Macro-AUROC",
               xlim=(100, 2000))
        ax.set_xticks([100, 500, 1000, 1500, 2000])
        ax.grid(color="#e1e0d9", lw=0.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.069, 0.87), ncol=3, frameon=False)
    fig.text(0.075, 0.073, "S2: 6,389 real ECGs. New recipe: 8,579 real + 4,515 generated, 48/16 source mix, S2 on 12 real + 4 generated slots.",
             fontsize=10.5, color="#52514e")
    fig.text(0.075, 0.034, "Reference curves are truncated without retraining; new trials complete at 2,000 steps. Each seed independently selects its best overall validation checkpoint.",
             fontsize=10, color=GREY)
    save(fig, "learning_curves")


def main():
    scaling = [load_run(RUNS / f"ptbxl-scaling/pool{share}/real/seed0", "real_scaling", share,
                        len(read(ENGINE / f"manifests/ptbxl-scaling/pool{share}.json")["roles"]["pool"])) for share in SHARES]
    groups = {
        "S2": [load_run(RUNS / f"ptbxl-aug-sweep/scarcity50/S2/seed{seed}", "S2", 50, 6389) for seed in range(3)],
        "S2_generated": [load_run(ROOT / f"pool50/mix75_s2/seed{seed}", "S2_generated", 50, 8579, 4515) for seed in range(3)],
    }
    s2_config = groups["S2"][0]["spec"]["transform"]
    for group, runs in groups.items():
        for seed, run in enumerate(runs):
            if run["seed"] != seed or run["spec"]["transform"] != s2_config or run["spec"]["transform_fraction"] != 0.25:
                raise ValueError(f"S2 recipe or seed differs: {run['source']}")
            if group == "S2_generated":
                if run["result"]["final_step"] != BUDGET:
                    raise ValueError("New trials must complete exactly 2,000 steps")
                e = run["result"]["exposure"]
                if (e["real_presentations"], e["generated_presentations"], e["transform_slots"]) != (96000, 32000, 32000):
                    raise ValueError("Source or augmentation exposure differs")
                if run["test"] is None:
                    raise ValueError("New trial test evaluation is missing")
    all_runs = [*scaling, *groups["S2"], *groups["S2_generated"]]
    rows = {scenario: [selected(run, scenario, limit) for run in all_runs]
            for scenario, limit in [("original_budget_context", None), ("first_2000_steps", BUDGET)]}
    ag = {scenario: aggregate(entries) for scenario, entries in rows.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("per_seed_metrics.csv", [r for entries in rows.values() for r in entries])
    write_csv("validation_history.csv", histories(all_runs))
    long_stats = [{"scenario": scenario, "group": group, "metric": metric, **value}
                  for scenario, groups_ in ag.items() for group, metrics in groups_.items() for metric, value in metrics.items()]
    write_csv("aggregate_metrics.csv", long_stats)
    comparisons = []
    for scenario, entries in rows.items():
        for metric in [f"{split}_{part}" for split in ("val", "test") for part in PARTS]:
            diffs = []
            for seed in range(3):
                a = next(r for r in entries if r["group"] == "S2" and r["seed"] == seed)
                b = next(r for r in entries if r["group"] == "S2_generated" and r["seed"] == seed)
                difference = None if a[metric] is None or b[metric] is None else b[metric] - a[metric]
                comparisons.append({"scenario": scenario, "metric": metric, "seed": seed,
                                    "historical_s2": a[metric], "s2_generated": b[metric], "difference": difference})
                if difference is not None:
                    diffs.append(difference)
            comparisons.append({"scenario": scenario, "metric": metric, "seed": "mean",
                                "historical_s2": ag[scenario]["S2"][metric]["mean"],
                                "s2_generated": ag[scenario]["S2_generated"][metric]["mean"],
                                "difference": float(np.mean(diffs)) if diffs else None,
                                "difference_sample_sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None})
    write_csv("comparisons.csv", comparisons)
    class_rows = per_class([*groups["S2"], *groups["S2_generated"]])
    write_csv("per_class_metrics.csv", class_rows)
    notes = [
        "Each seed selects the first maximum of overall validation macro-AUROC; all slices use that checkpoint.",
        "Original context: scaling seed0 runs 12000 steps, historical S2 seeds0/1/2 run10000; new trials run2000.",
        "Budget-matched plots restrict all historical histories to<=2000 without retraining.",
        "Historical S2 global-best checkpoints all fall within2000; saved test scores therefore match both selections.",
        "Historical S2 has6389 artifact-depleted real ECGs, versus8579 real ECGs in the new trial; this is not an isolated generated-data effect.",
        "S2 dose: quarter of each batch at half standard strength. New batches augment12real+4generated of48real+16generated.",
        "Generated diagnoses are provisional conditioning-parent labels.2212 generated ECGs use parents outside the50%pool, all in training folds; no heldout patient overlap.",
        "Smoothed peak is the center of the maximal5-validation moving average; a2000-step observation window does not establish convergence.",
        "Error bars are sample SD across three seeds, not confidence intervals. Real-data scaling has only seed0.",
        "The test fold was viewed in prior experiments; this extension is exploratory.",
    ]
    provenance_files = [run["folder"] / filename for run in all_runs for filename in
                        ("result.json", "trial.json", "test_scores.json") if (run["folder"] / filename).exists()]
    summary = {"schema_version": "ptbxl-s2-diffusets-three-seed-plots@1", "selection_budget": BUDGET,
               "aggregates": ag, "per_seed": rows, "comparisons": comparisons, "per_class_metrics": class_rows,
               "source_files_sha256": {str(p.relative_to(RUNS)): sha(p) for p in provenance_files}, "notes": notes}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    style()
    for scenario in rows:
        matched = scenario == "first_2000_steps"
        points = [r for r in rows[scenario] if r["group"] == "real_scaling"]
        for full in (False, True):
            main_figure(points, ag[scenario], full=full, matched=matched)
    zoom_plot([r for r in rows["first_2000_steps"] if r["group"] == "real_scaling"],
              rows["first_2000_steps"], ag["first_2000_steps"])
    learning_plot(scaling, groups)
    print(json.dumps({"output_directory": str(OUT), "new_trial": ag["first_2000_steps"]["S2_generated"],
                      "historical_s2": ag["first_2000_steps"]["S2"],
                      "files": sorted(p.name for p in OUT.iterdir() if p.is_file())}, indent=2))


if __name__ == "__main__":
    main()
