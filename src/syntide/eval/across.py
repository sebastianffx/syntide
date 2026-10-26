"""Compare paired treatments from separate experiment protocols.

Treatments introduced after test evaluation use their own protocol. They can
still be compared with an earlier protocol when seeds, training budgets,
initial weights, and evaluation records match. Each test result must belong
to its protocol's recorded test-evaluation set.

Use the first protocol's fixed analysis rules and resampling settings.
Reject incompatible or duplicated runs. Save across.json and across.md below
the chosen output directory. These later comparisons are exploratory and do
not replace the experiment's predefined confirmatory comparisons.
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
from syntide.eval.freeze import (
    RULES_FILE,
    FreezeError,
    frozen_rules,
    verify_frozen,
)
from syntide.eval.runs import Run, RunError, load_runs

SLICE_NAMES = {"overall": "All records", "artifact": "Noisy", "clean": "Clean"}


class AcrossError(RuntimeError):
    """Separate protocols have incomplete or inconsistent run records."""


def load(
    protocol_dirs: Sequence[str | Path], split: str
) -> tuple[AnalysisConfig, list[Run]]:
    """Load fixed rules and all completed runs across protocols."""
    first = Path(protocol_dirs[0])
    path = first / RULES_FILE
    if not path.exists():
        raise AcrossError(
            f"{path} does not exist; the first protocol's rules set the "
            "bootstrap"
        )
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
                f"{protocol_dir}: {len(pending)} planned trials have no "
                f"{split} predictions yet"
            )
        for run in found:
            key = (*run.key, run.seed)
            if key in source:
                raise AcrossError(
                    f"{'/'.join(map(str, key))} is in both {source[key]} "
                    f"and {protocol_dir}"
                )
            source[key] = protocol_dir
            runs.append(run)
    return rules, runs


def compare(
    runs: Sequence[Run],
    rules: AnalysisConfig,
    contrasts: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Compare selected checkpoints on each data subset."""
    analysis = Analysis(runs, rules, checkpoint="best")
    out = []
    for a_name, b_name in contrasts:
        a, b = arm_key(a_name, rules.regime), arm_key(b_name, rules.regime)
        missing = [
            name
            for name, key in ((a_name, a), (b_name, b))
            if key not in analysis.groups
        ]
        if missing:
            raise AcrossError(
                f"no finished run of {', '.join(missing)} in the given "
                "protocols"
            )
        reports: dict[str, UtilityReport] = {}
        for name in SLICES:
            report = analysis.contrast(a, b, name)
            if report is None:
                raise AcrossError(f"{a_name} and {b_name} share no seed")
            reports[name] = report
        steps = {
            label: [group[seed].steps["best"] for seed in sorted(group)]
            for label, group in (
                (a_name, analysis.groups[a]),
                (b_name, analysis.groups[b]),
            )
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
        (
            f"Runs: {', '.join(results['protocol_dirs'])}. Checkpoint: the "
            f"one validation chose. Bootstrap: {rules['n_boot']:,} crossed "
            f"draws over patients and seeds, seed {rules['bootstrap_seed']}"
            f", {rules['ci_level']:.0%} CIs, from the first protocol's "
            f"frozen rules ({results['rules_sha256'][:12]}). Diagnostics: "
            "no frozen rule names these contrasts."
        ),
        "",
        "| Contrast | "
        + " | ".join(
            SLICE_NAMES[name] for name in ("overall", "artifact", "clean")
        )
        + " | "
        "Seeds better, all | " + " | ".join(SUPERCLASSES) + " |",
        "|---|---|---|---|---|" + "---|" * len(SUPERCLASSES),
    ]
    for entry in results["contrasts"]:
        overall = entry["overall"]
        better = sum(value > 0 for value in overall["per_seed"])
        lines.append(
            f"| {entry['label']} | "
            + " | ".join(
                ci(entry[name]) for name in ("overall", "artifact", "clean")
            )
            + f" | {better}/{len(overall['per_seed'])} | "
            + " | ".join(
                signed(overall["per_class"][name]) for name in SUPERCLASSES
            )
            + " |"
        )
    lines += [
        "",
        (
            "Per class: mean over seeds of the change in each "
            "superclass's AUROC on all records."
        ),
        "",
        (
            "| Contrast | Per seed, all records | Mean, all records (a / "
            "b) | Best step (a; b) |"
        ),
        "|---|---|---|---|",
    ]
    for entry in results["contrasts"]:
        overall = entry["overall"]
        steps = "; ".join(
            ", ".join(f"{s:,}" for s in values)
            for values in entry["best_step"].values()
        )
        lines.append(
            f"| {entry['label']} | "
            f"{', '.join(signed(v) for v in overall['per_seed'])} | "
            f"{overall['mean_a']:.4f} / {overall['mean_b']:.4f} | {steps} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m syntide.eval.across",
        description=(
            "Paired contrasts between arms of protocols that share seeds, "
            "data, trainer and budget."
        ),
    )
    parser.add_argument(
        "protocol_dirs",
        nargs="+",
        help="protocol runs directories; the first one's rules apply",
    )
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument(
        "--contrast",
        action="append",
        required=True,
        help="A-B, e.g. T6-A0 or C2@full100-A0; repeatable",
    )
    parser.add_argument(
        "--out",
        help=(
            "where to write <split>/across.{json,md} (default: <last "
            "protocol>/eval)"
        ),
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
    out_dir = (
        Path(args.out or Path(args.protocol_dirs[-1]) / "eval") / args.split
    )
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
