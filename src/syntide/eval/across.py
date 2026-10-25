"""``python -m syntide.eval.across runs/<protocol> runs/<extension> --split val|test --contrast T6-A0 ...``:
paired contrasts between arms planned in different protocols.

A protocol's test fold is claimed once, for the trials it planned, so an arm added later gets a protocol of
its own (``ptbxl-aug-simvea`` adds T6 to ``ptbxl-aug-sweep``). Its trials can still be paired with the first
protocol's: SYNTIDE keys every random stream by seed, step and slot, never by arm or protocol. This module
compares them exactly as ``syntide.eval`` compares arms within one protocol:

- every run is loaded through ``syntide.eval.runs.load_runs``: the run contract holds, and on the test split
  each run must be in its own protocol's ``test_claim.json``;
- the first protocol's rules must be frozen, and its plan unchanged since. Its bootstrap settings (draws,
  seed, CI level) and its primary regime are used, so a contrast here is resampled with the same draws as
  that protocol's own diagnostics;
- ``syntide.eval.contrasts`` refuses arms that are not paired: other seeds, trainer, budget, initial weights,
  transform slots, records or labels.

Writes ``<out>/<split>/across.json`` and ``across.md`` (``--out`` defaults to the last protocol's ``eval``).
These contrasts are diagnostics, never evidence: no frozen rule names them.
"""

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from syntide.data.schema import SUPERCLASSES
from syntide.eval.analysis import Analysis, AnalysisConfig, arm_key
from syntide.eval.arms_figure import MINUS
from syntide.eval.contrasts import SLICES, PairingError, UtilityReport
from syntide.eval.freeze import RULES_FILE, FreezeError, frozen_rules, verify_frozen
from syntide.eval.runs import Run, RunError, load_runs

SLICE_NAMES = {"overall": "All records", "artifact": "Noisy", "clean": "Clean"}


class AcrossError(RuntimeError):
    """The protocols cannot be compared: rules not frozen, a trial unfinished, or a run listed twice."""


def load(protocol_dirs: Sequence[str | Path], split: str) -> tuple[AnalysisConfig, list[Run]]:
    """The first protocol's frozen rules, and every run of every protocol, once all of them are finished."""
    first = Path(protocol_dirs[0])
    path = first / RULES_FILE
    if not path.exists():
        raise AcrossError(f"{path} does not exist; the first protocol's rules set the bootstrap")
    rules = frozen_rules(json.loads(path.read_text(encoding="utf-8")))
    try:
        verify_frozen(first, rules, claimed=split == "test")
    except FreezeError as error:
        raise AcrossError(str(error)) from error
    runs, source = [], {}
    for protocol_dir in map(Path, protocol_dirs):
        found, pending = load_runs(protocol_dir, split)
        if pending:
            raise AcrossError(
                f"{protocol_dir}: {len(pending)} planned trials have no {split} predictions yet"
            )
        for run in found:
            key = (*run.key, run.seed)
            if key in source:
                raise AcrossError(f"{'/'.join(map(str, key))} is in both {source[key]} and {protocol_dir}")
            source[key] = protocol_dir
            runs.append(run)
    return rules, runs


def compare(
    runs: Sequence[Run], rules: AnalysisConfig, contrasts: Sequence[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Each contrast ``a - b`` on every slice, at the checkpoints validation chose."""
    analysis = Analysis(runs, rules, checkpoint="best")
    out = []
    for a_name, b_name in contrasts:
        a, b = arm_key(a_name, rules.regime), arm_key(b_name, rules.regime)
        missing = [name for name, key in ((a_name, a), (b_name, b)) if key not in analysis.groups]
        if missing:
            raise AcrossError(f"no finished run of {', '.join(missing)} in the given protocols")
        reports: dict[str, UtilityReport] = {}
        for name in SLICES:
            report = analysis.contrast(a, b, name)
            if report is None:
                raise AcrossError(f"{a_name} and {b_name} share no seed")
            reports[name] = report
        steps = {
            label: [group[seed].steps["best"] for seed in sorted(group)]
            for label, group in ((a_name, analysis.groups[a]), (b_name, analysis.groups[b]))
        }
        out.append(
            {
                "label": f"{a_name} - {b_name}",
                "best_step": steps,
                **{name: report.to_dict() for name, report in reports.items()},
            }
        )
    return out


def render(results: dict[str, Any]) -> str:
    def ci(report: dict[str, Any]) -> str:
        values = (report["estimate"], report["ci_low"], report["ci_high"])
        return "{:+.3f} [{:+.3f}, {:+.3f}]".format(*values).replace("-", MINUS)

    def signed(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.3f}".replace("-", MINUS)

    rules = results["rules"]
    lines = [
        f"# Paired contrasts across protocols ({results['split']})",
        "",
        f"Runs: {', '.join(results['protocol_dirs'])}. Checkpoint: the one validation chose. "
        f"Bootstrap: {rules['n_boot']:,} crossed draws over patients and seeds, "
        f"seed {rules['bootstrap_seed']}, {rules['ci_level']:.0%} CIs, from the first protocol's "
        f"frozen rules ({results['rules_sha256'][:12]}). "
        "Diagnostics: no frozen rule names these contrasts.",
        "",
        "| Contrast | " + " | ".join(SLICE_NAMES[name] for name in ("overall", "artifact", "clean")) + " | "
        "Seeds better, all | " + " | ".join(SUPERCLASSES) + " |",
        "|---|---|---|---|---|" + "---|" * len(SUPERCLASSES),
    ]
    for entry in results["contrasts"]:
        overall = entry["overall"]
        better = sum(value > 0 for value in overall["per_seed"])
        lines.append(
            f"| {entry['label']} | "
            + " | ".join(ci(entry[name]) for name in ("overall", "artifact", "clean"))
            + f" | {better}/{len(overall['per_seed'])} | "
            + " | ".join(signed(overall["per_class"][name]) for name in SUPERCLASSES)
            + " |"
        )
    lines += [
        "",
        "Per class: mean over seeds of the change in each superclass's AUROC on all records.",
        "",
        "| Contrast | Per seed, all records | Mean, all records (a / b) | Best step (a; b) |",
        "|---|---|---|---|",
    ]
    for entry in results["contrasts"]:
        overall = entry["overall"]
        steps = "; ".join(", ".join(f"{s:,}" for s in values) for values in entry["best_step"].values())
        lines.append(
            f"| {entry['label']} | {', '.join(signed(v) for v in overall['per_seed'])} | "
            f"{overall['mean_a']:.4f} / {overall['mean_b']:.4f} | {steps} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m syntide.eval.across",
        description="Paired contrasts between arms of protocols that share seeds, data, trainer and budget.",
    )
    parser.add_argument(
        "protocol_dirs", nargs="+", help="protocol runs directories; the first one's rules apply"
    )
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument(
        "--contrast", action="append", required=True, help="A-B, e.g. T6-A0 or C2@full100-A0; repeatable"
    )
    parser.add_argument(
        "--out", help="where to write <split>/across.{json,md} (default: <last protocol>/eval)"
    )
    args = parser.parse_args(argv)
    contrasts = []
    for text in args.contrast:
        a, sep, b = text.partition("-")
        if not (sep and a and b):
            parser.error(f"--contrast takes A-B, got {text!r}")
        contrasts.append((a, b))
    try:
        rules, runs = load(args.protocol_dirs, args.split)
        entries = compare(runs, rules, contrasts)
    except (AcrossError, RunError, PairingError) as error:
        raise SystemExit(f"no contrasts: {error}") from error
    results = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": args.split,
        "checkpoint": "best",
        "protocol_dirs": [str(Path(d)) for d in args.protocol_dirs],
        "rules": rules.to_dict(),
        "rules_sha256": rules.sha256,
        "runs": sorted(str(run.run_dir) for run in runs),
        "contrasts": entries,
    }
    out_dir = Path(args.out or Path(args.protocol_dirs[-1]) / "eval") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "across.json").write_text(
        json.dumps(results, indent=1, default=_json) + "\n", encoding="utf-8"
    )
    text = render(results)
    (out_dir / "across.md").write_text(text, encoding="utf-8")
    print(out_dir / "across.md")
    print(text)


def _json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
