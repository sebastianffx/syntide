#!/usr/bin/env python3
"""Audit the completed 100-step cadence replication without modifying its inputs.

Run from synthetic_data_engine after training and test scoring have finished:
  uv run python runs/ptbxl-artifact-50-val100/compare_cadence.py

Writes cadence_comparison.json/.md and figures/scaling_comparison.png beside this
script. Missing inputs abort before writing outputs. Exact-equivalence failures
are recorded in the outputs and cause a nonzero exit status.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

SLICES = ("overall", "clean", "artifact")
SLICE_NAMES = {"overall": "All records", "clean": "Clean records", "artifact": "Noisy records"}
COMMON_STEPS = list(range(500, 6501, 500))
PROVENANCE_NOTE = (
    "This is a replication that changes validation cadence after the original test results were "
    "already viewed. The test records are the same fold 10 records; they are not a newly unseen "
    "confirmatory test set. Validation chooses each best checkpoint by raw overall macro-AUROC. "
    "The 100-step and 250-step checkpoint grids are not nested. Original artifacts are read-only."
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_rows(left: list[dict], right: list[dict], steps: list[int]) -> dict:
    left_by_step, right_by_step = ({r["step"]: r for r in history} for history in (left, right))
    differences = []
    for step in steps:
        a, b = left_by_step.get(step), right_by_step.get(step)
        if a != b:
            fields = sorted(set(a or {}) | set(b or {}))
            differences.append({
                "step": step,
                "fields": {key: {"left": (a or {}).get(key), "right": (b or {}).get(key)}
                           for key in fields if (a or {}).get(key) != (b or {}).get(key)},
                "missing_left": a is None,
                "missing_right": b is None,
            })
    return {"equal": not differences, "steps": steps, "rows_checked": len(steps),
            "differences": differences}


def compare_model_tensors(left: Path, right: Path) -> dict:
    import torch

    a = torch.load(left, map_location="cpu", weights_only=True)["model"]
    b = torch.load(right, map_location="cpu", weights_only=True)["model"]
    missing_left, missing_right = sorted(b.keys() - a.keys()), sorted(a.keys() - b.keys())
    differences = []
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key], b[key]
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            row = {"tensor": key, "left_shape": list(x.shape), "right_shape": list(y.shape),
                   "left_dtype": str(x.dtype), "right_dtype": str(y.dtype)}
            if x.shape == y.shape and x.numel():
                row["max_abs_difference"] = float((x.double() - y.double()).abs().max())
            differences.append(row)
    return {"equal": not (missing_left or missing_right or differences),
            "tensors_checked": len(a.keys() & b.keys()), "missing_left": missing_left,
            "missing_right": missing_right, "differences": differences,
            "note": "Compares model parameters and buffers, excluding trial-specific checkpoint metadata."}


def selected_scores(result: dict) -> dict:
    row = next(r for r in result["val_history"] if r["step"] == result["best_step"])
    return {part: row[f"val/auroc_{part}"] for part in SLICES}


def moments(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n_seeds": len(values), "values": values}


def summarize_groups(trials: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in trials:
        grouped[(row["regime"], row["arm"])].append(row)
    groups = []
    for (regime, arm), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r["seed"])
        group = {"regime": regime, "arm": arm, "seeds": [r["seed"] for r in rows]}
        for split in ("val", "test"):
            group[split] = {}
            for part in SLICES:
                old = moments([r["old"][split][part] for r in rows])
                new = moments([r["new"][split][part] for r in rows])
                deltas = [r["new"][split][part] - r["old"][split][part] for r in rows]
                group[split][part] = {"old": old, "new": new, "paired_delta": moments(deltas)}
        groups.append(group)
    return groups


def make_figure(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"old": "#dd7543", "new": "#2474b8", "scaling": "#159b79"}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "axes.titlelocation": "left",
                         "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(3)
    scaling = report["scaling_equivalence"]
    ref = next(g for g in report["groups"] if g["regime"] == "full100" and g["arm"] == "C2")
    ax = axes[0]
    for offset, key, label, marker in [
        (-0.07, "scaling_scores", "Scaling: seed 0, every 100 steps", "o"),
        (0.07, "new_scores", "Replication: seed 0, every 100 steps", "D"),
    ]:
        values = [scaling[key][part] for part in SLICES]
        color = colors["scaling" if key == "scaling_scores" else "new"]
        ax.plot(x + offset, values, marker, ms=8, color=color, label=label, zorder=3)
    for i, part in enumerate(SLICES):
        a, b = scaling["scaling_scores"][part], scaling["new_scores"][part]
        ax.plot([i - .07, i + .07], [a, b], color="#777777", lw=1)
        label = f"{a:.6f}" if a == b else f"{a:.6f} / {b:.6f}"
        ax.annotate(label, (i, max(a, b)), xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set_title("Same seed, same metric, same checkpoint cadence", pad=16)
    ax.set_ylabel("Validation macro-AUROC at selected checkpoint")
    ax.text(.02, .03, f"Selected steps: scaling {scaling['scaling_best_step']:,}; "
            f"replication {scaling['new_best_step']:,}\n"
            f"Exact first-65-row match: {'PASS' if scaling['history']['equal'] else 'FAIL'}",
            transform=ax.transAxes, fontsize=9, va="bottom", color="#454545")
    ax = axes[1]
    for offset, version, label in [(-.10, "old", "Original: every 250 steps, seeds 0–2"),
                                    (.10, "new", "Replication: every 100 steps, seeds 0–2")]:
        values = [ref["val"][part][version]["mean"] for part in SLICES]
        errors = [ref["val"][part][version]["sd"] for part in SLICES]
        ax.errorbar(x + offset, values, yerr=errors, fmt="o", capsize=5, ms=7,
                    color=colors[version], lw=1.7, label=label, zorder=3)
    ax.set_title("100% reference across three seeds: mean ± SD", pad=16)
    ax.set_ylabel("Validation macro-AUROC at selected checkpoints")
    for ax in axes:
        ax.set_xticks(x, [SLICE_NAMES[part] for part in SLICES])
        ax.set_xlim(-.45, 2.45)
        ax.grid(axis="y", color="#deded6", linewidth=.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), frameon=False, fontsize=9)
    values = [scaling[key][part] for key in ("scaling_scores", "new_scores") for part in SLICES]
    values += [ref["val"][part][version]["mean"] + direction * ref["val"][part][version]["sd"]
               for part in SLICES for version in ("old", "new") for direction in (-1, 1)]
    low, high = min(values), max(values)
    padding = max(.001, (high - low) * .30)
    for ax in axes:
        ax.set_ylim(low - padding, high + padding * 1.5)
    fig.suptitle("PTB-XL: separating checkpoint cadence, record slice, and seed aggregation",
                 x=.035, ha="left", fontsize=14, fontweight="semibold")
    fig.text(.035, .025, "All models train on the same full training pool. Validation is fold 9. "
             "Best checkpoint uses raw overall AUROC; no smoothing.\n"
             "SD describes seed variation, not a confidence interval. "
             "This replication follows an already viewed test evaluation.", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.07, right=.985, bottom=.27, top=.83, wspace=.26)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def markdown(report: dict) -> str:
    check = report["checks"]
    status = "PASS" if report["passed"] else "FAIL"
    lines = ["# PTB-XL validation-cadence replication", "",
             f"Generated {report['generated_at']}. **Audit: {status}.**", "", PROVENANCE_NOTE, "",
             "The original experiment validates every 250 optimizer steps; this replication validates "
             "every 100. Both train 21 trials for 6,500 steps, batch size 64, seeds 0–2. "
             "Both the 50% arms and the 100% C2 reference retain their original data and settings.", "",
             "| Exact audit | Result |", "|---|---|"]
    lines.extend(f"| {name.replace('_', ' ')} | {'PASS' if value else 'FAIL'} |"
                 for name, value in check.items())
    if report["failures"]:
        lines += ["", "Failures:", ""] + [f"- {item}" for item in report["failures"]]
    scale = report["scaling_equivalence"]
    lines += ["", "## Scaling check: one seed, validation", "",
              "The scaling comparison uses seed 0 on the identical full training pool. "
              "The original 0.933 label describes clean records; overall AUROC is about 0.931.", "",
              "| Slice | Scaling seed 0 | Replication seed 0 | Difference |", "|---|---:|---:|---:|"]
    for part in SLICES:
        a, b = scale["scaling_scores"][part], scale["new_scores"][part]
        lines.append(f"| {SLICE_NAMES[part]} | {a:.9f} | {b:.9f} | {b-a:+.9f} |")
    lines += ["", f"Selected steps: scaling **{scale['scaling_best_step']:,}**, replication "
              f"**{scale['new_best_step']:,}**. Compared all 65 history rows through step 6,500 "
              "and all tensors of the selected model. Scaling trained to 12,000 steps; its "
              "selected checkpoint lies within the replication's 6,500-step budget.", "",
              "![Cadence and scaling comparison](figures/scaling_comparison.png)", ""]
    for split in ("val", "test"):
        label = "Validation" if split == "val" else "Test (previously viewed fold 10)"
        lines += [f"## {label}: mean ± sample SD across seeds 0–2", "",
                  "| Regime / arm | All: old | All: new | Clean: old | Clean: new | Noisy: old | Noisy: new |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for group in report["groups"]:
            cells = [f"{group['regime']} / {group['arm']}"]
            for part in SLICES:
                for version in ("old", "new"):
                    s = group[split][part][version]
                    cells.append(f"{s['mean']:.6f} ± {s['sd']:.6f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## All 21 trials", "",
              "Scores below are overall macro-AUROC. Deltas are new minus old. "
              "Complete slice scores and exact comparisons are in cadence_comparison.json.", "",
              "| Trial | Best step old → new | Val old → new | Δ val | Test old → new | Δ test |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in report["trials"]:
        a, b = row["old"], row["new"]
        lines.append(f"| {row['run_dir']} | {a['best_step']} → {b['best_step']} | "
                     f"{a['val']['overall']:.6f} → {b['val']['overall']:.6f} | "
                     f"{b['val']['overall']-a['val']['overall']:+.6f} | "
                     f"{a['test']['overall']:.6f} → {b['test']['overall']:.6f} | "
                     f"{b['test']['overall']-a['test']['overall']:+.6f} |")
    lines += ["", "All shared checkpoints are checked at steps 500, 1,000, …, 6,500: "
              "13 complete metric rows per trial (273 rows total). Final checkpoint comparisons "
              "use model parameters and buffers, excluding trial-ID metadata. "
              "Exact equality is required; this audit applies no numerical tolerance.", ""]
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=root.parent / "ptbxl-artifact-50")
    parser.add_argument("--new", type=Path, default=root)
    parser.add_argument("--scaling", type=Path, default=root.parent / "ptbxl-scaling")
    args = parser.parse_args()
    old_dir, new_dir, scaling_dir = (p.resolve() for p in (args.old, args.new, args.scaling))
    if old_dir == new_dir or scaling_dir == new_dir:
        parser.error("the output replication directory must differ from both source directories")
    old_plan, new_plan = (read_json(p / "plan.json") for p in (old_dir, new_dir))
    old_entries = {r["run_dir"]: r for r in old_plan["trials"]}
    new_entries = {r["run_dir"]: r for r in new_plan["trials"]}
    if len(old_entries) != 21 or old_entries.keys() != new_entries.keys():
        parser.error("expected matching 21-trial plans; no output written")
    scaling_run = scaling_dir / "pool100/real/seed0"
    required = [directory / rel / name for directory in (old_dir, new_dir)
                for rel in old_entries for name in ("trial.json", "result.json", "test_scores.json", "final.pt")]
    required += [scaling_run / name for name in ("trial.json", "result.json", "best.pt")]
    required += [new_dir / "full100/C2/seed0/best.pt"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error("training/test scoring is incomplete; no output written:\n" + "\n".join(missing))
    failures, trials = [], []
    checks = {name: True for name in (
        "all_21_specs_change_only_protocol_cadence_and_id", "all_21_results_match_plan_and_best_test_checkpoint",
        "all_21_init_hashes_and_trainer_fingerprints_match", "all_21_history_grids_match_expected_cadence",
        "all_273_shared_history_rows_equal", "all_21_final_model_states_equal",
        "scaling_seed0_first_65_rows_equal", "scaling_seed0_selected_scores_equal",
        "scaling_seed0_best_model_state_equal", "scaling_seed0_init_and_trainer_match")}

    def check(name: str, ok: bool, detail: str) -> None:
        if not ok:
            checks[name] = False
            failures.append(detail)

    for index, rel in enumerate(sorted(old_entries), 1):
        print(f"[{index}/21] Checking {rel}", flush=True)
        dirs = {"old": old_dir / rel, "new": new_dir / rel}
        specs = {k: read_json(d / "trial.json") for k, d in dirs.items()}
        results = {k: read_json(d / "result.json") for k, d in dirs.items()}
        tests = {k: read_json(d / "test_scores.json") for k, d in dirs.items()}
        a, b = specs["old"], specs["new"]
        changed = {key: {"old": a.get(key), "new": b.get(key)}
                   for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)}
        ok = (set(changed) == {"protocol", "val_every", "trial_id"}
              and a["val_every"] == 250 and b["val_every"] == 100
              and a["protocol"] == old_plan["protocol"] and b["protocol"] == new_plan["protocol"])
        check("all_21_specs_change_only_protocol_cadence_and_id", ok, f"{rel}: unexpected spec differences: {changed}")
        for version, cadence, entries in (("old", 250, old_entries), ("new", 100, new_entries)):
            result, test, spec = results[version], tests[version], specs[version]
            expected_steps = list(range(cadence, 6501, cadence))
            check("all_21_history_grids_match_expected_cadence",
                  [r["step"] for r in result["val_history"]] == expected_steps,
                  f"{rel}/{version}: unexpected validation history grid")
            ok = (result["status"] == "complete" and result["final_step"] == 6500
                  and result["trial_id"] == spec["trial_id"] == entries[rel]["trial_id"] == test["trial_id"]
                  and test["step"] == result["best_step"] and test["checkpoint"] == "best"
                  and result["best_val_auroc"] == selected_scores(result)["overall"]
                  and result["best_val_auroc"] == max(r["val/auroc_overall"] for r in result["val_history"]))
            check("all_21_results_match_plan_and_best_test_checkpoint", ok,
                  f"{rel}/{version}: result, selected history, test checkpoint, or plan mismatch")
        init_equal = results["old"]["trainer"]["init_sha256"] == results["new"]["trainer"]["init_sha256"]
        fingerprint_equal = results["old"]["trainer"]["fingerprint"] == results["new"]["trainer"]["fingerprint"]
        check("all_21_init_hashes_and_trainer_fingerprints_match", init_equal and fingerprint_equal,
              f"{rel}: initialization or trainer differs")
        common = compare_rows(results["old"]["val_history"], results["new"]["val_history"], COMMON_STEPS)
        check("all_273_shared_history_rows_equal", common["equal"], f"{rel}: shared-checkpoint metrics differ")
        final = compare_model_tensors(dirs["old"] / "final.pt", dirs["new"] / "final.pt")
        check("all_21_final_model_states_equal", final["equal"], f"{rel}: final model tensors differ")
        row = {"run_dir": rel, "regime": b["regime"], "arm": b["arm"], "seed": b["replicate_seed"],
               "spec_changes": changed, "init_equal": init_equal, "fingerprint_equal": fingerprint_equal,
               "shared_history": common, "final_model": final}
        for version in ("old", "new"):
            result = results[version]
            row[version] = {"trial_id": result["trial_id"], "best_step": result["best_step"],
                            "val": selected_scores(result),
                            "test": {p: tests[version][f"test/auroc_{p}"] for p in SLICES},
                            "trainer": result["trainer"], "environment": result.get("env"),
                            "result_sha256": sha256(dirs[version] / "result.json"),
                            "test_scores_sha256": sha256(dirs[version] / "test_scores.json")}
        trials.append(row)
    scaling_result = read_json(scaling_run / "result.json")
    new_result = read_json(new_dir / "full100/C2/seed0/result.json")
    history = compare_rows(scaling_result["val_history"], new_result["val_history"], list(range(100, 6501, 100)))
    best_model = compare_model_tensors(scaling_run / "best.pt", new_dir / "full100/C2/seed0/best.pt")
    scale = {"history": history, "best_model": best_model,
             "scaling_best_step": scaling_result["best_step"], "new_best_step": new_result["best_step"],
             "scaling_scores": selected_scores(scaling_result), "new_scores": selected_scores(new_result),
             "scaling_result_sha256": sha256(scaling_run / "result.json")}
    check("scaling_seed0_first_65_rows_equal", history["equal"], "full100 seed0: scaling history differs")
    check("scaling_seed0_selected_scores_equal",
          scale["scaling_scores"] == scale["new_scores"] and scale["scaling_best_step"] == scale["new_best_step"],
          "full100 seed0: selected scaling scores or step differ")
    check("scaling_seed0_best_model_state_equal", best_model["equal"], "full100 seed0: selected scaling model differs")
    check("scaling_seed0_init_and_trainer_match",
          all(scaling_result["trainer"][k] == new_result["trainer"][k] for k in ("init_sha256", "fingerprint")),
          "full100 seed0: scaling initialization or trainer differs")
    report = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "passed": all(checks.values()), "checks": checks, "failures": failures,
              "provenance_note": PROVENANCE_NOTE,
              "inputs": {"old": str(old_dir), "new": str(new_dir), "scaling": str(scaling_dir),
                         "old_plan_sha256": sha256(old_dir / "plan.json"),
                         "new_plan_sha256": sha256(new_dir / "plan.json")},
              "scaling_equivalence": scale, "trials": trials, "groups": summarize_groups(trials)}
    (new_dir / "cadence_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (new_dir / "cadence_comparison.md").write_text(markdown(report), encoding="utf-8")
    make_figure(report, new_dir / "figures/scaling_comparison.png")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: wrote comparison JSON, Markdown, and figure to {new_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
