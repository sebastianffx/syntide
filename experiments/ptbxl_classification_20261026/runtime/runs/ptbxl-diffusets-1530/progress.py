#!/usr/bin/env python3
"""Read-only snapshot of the six training trials; ETA excludes evaluation/reporting.

Uses the trainer's measured budget/wall_time_seconds, not file modification times, for speed.
File age only indicates how recently the active metrics file changed. Partial trailing JSONL
records are ignored because training may be writing while this script reads.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent


def duration(seconds):
    if seconds is None:
        return "unavailable"
    seconds = round(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def snapshot(root=ROOT):
    plan = json.loads((root / "plan.json").read_text())
    now = time.time()
    rows, rates, measured_seconds = [], [], 0.0
    for entry in plan["trials"]:
        directory = root / entry["run_dir"]
        spec = json.loads((directory / "trial.json").read_text())
        row = {"trial": entry["run_dir"], "status": "pending", "max_steps": spec["max_steps"],
               "last_train_step": 0, "last_val_step": 0, "measured_training_seconds": 0.0,
               "metrics_age_seconds": None}
        result_path, metrics_path = directory / "result.json", directory / "metrics.jsonl"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("status") == "complete":
                row.update(status="complete", last_train_step=result["final_step"],
                           last_val_step=max(h["step"] for h in result["val_history"]),
                           measured_training_seconds=result["wall_time_s"])
                rates.append(result["wall_time_s"] / result["final_step"])
        elif metrics_path.exists():
            lines = metrics_path.read_text().splitlines()
            events = []
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines) - 1:
                        row["warning"] = "Malformed metrics row encountered"
            budgets = [e for e in events if "budget/wall_time_seconds" in e]
            train = [e["step"] for e in events if "train/loss" in e]
            val = [e["step"] for e in events if "val/auroc_overall" in e]
            row.update(status="in_progress", last_train_step=max(train, default=0),
                       last_val_step=max(val, default=0), metrics_age_seconds=max(0, now - metrics_path.stat().st_mtime))
            if budgets:
                budget = budgets[-1]
                row["measured_training_seconds"] = budget["budget/wall_time_seconds"]
                # The budget event is emitted before this step's optimizer update. Train/loss is
                # emitted afterward; use only completed logged train steps to show progress.
                if budget["step"] >= 500:
                    rates.append(budget["budget/wall_time_seconds"] / max(1, budget["step"] - 1))
            if row["metrics_age_seconds"] > 120:
                row["warning"] = "No recent metrics update; process liveness is not checked"
        measured_seconds += row["measured_training_seconds"]
        rows.append(row)
    completed = sum(r["status"] == "complete" for r in rows)
    remaining_steps = sum(r["max_steps"] - r["last_train_step"] for r in rows)
    seconds_per_step = statistics.median(rates) if rates else None
    # With stale logs, extrapolation from prior throughput would imply a false live ETA.
    stale = any(r["status"] == "in_progress" and r["metrics_age_seconds"] > 120 for r in rows)
    eta = remaining_steps * seconds_per_step if seconds_per_step is not None and not stale else None
    return {"observed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "completed_trials": completed, "planned_trials": len(rows), "trials": rows,
            "logged_optimizer_steps": sum(r["last_train_step"] for r in rows),
            "planned_optimizer_steps": sum(r["max_steps"] for r in rows),
            "measured_training_seconds": measured_seconds, "estimated_training_seconds_remaining": eta,
            "eta_basis": "median measured seconds/update; warm-up omitted until 500 updates; excludes evaluation and reports",
            "process_liveness_checked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the same snapshot as JSON")
    args = parser.parse_args()
    result = snapshot()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Training: {result['completed_trials']}/{result['planned_trials']} complete; "
          f"{result['logged_optimizer_steps']:,}/{result['planned_optimizer_steps']:,} logged updates")
    for row in result["trials"]:
        age = f"; updated {round(row['metrics_age_seconds'])}s ago" if row["metrics_age_seconds"] is not None else ""
        print(f"  {row['trial']:24s} {row['status']:11s} "
              f"train {row['last_train_step']:4d}/{row['max_steps']}  val {row['last_val_step']:4d}{age}")
        if row.get("warning"):
            print(f"    {row['warning']}")
    print(f"Measured training time: {duration(result['measured_training_seconds'])}; "
          f"estimated training remaining: {duration(result['estimated_training_seconds_remaining'])}")
    print("ETA excludes evaluation/reporting; this snapshot does not check process liveness.")


if __name__ == "__main__":
    main()
