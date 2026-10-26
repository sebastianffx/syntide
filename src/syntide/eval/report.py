"""Write comparisons, figures, and a report for one data split.

Everything lands in ``runs/<protocol>/eval/<split>[-final]/``.
``comparisons.json`` holds every paired contrast (``UtilityReport``) and the
findings, so ``results.md`` and the figures can be rebuilt from it.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from syntide.eval import views
from syntide.eval.analysis import (
    AnalysisConfig,
    analyze,
    arm_label,
    gap_gate_outcome,
)
from syntide.eval.freeze import (
    CLAIM_FILE,
    RULES_FILE,
    FreezeError,
    verify_frozen,
)
from syntide.eval.runs import RunError, load_runs
from syntide.eval.tracking import TrackerConfig, make_tracker
from syntide.generate.plan import ProtocolConfig, regime_arms

SPLIT_NAMES = {"val": "validation", "test": "test"}


def write_report(
    protocol_dir: str | Path,
    protocol: ProtocolConfig,
    rules: AnalysisConfig,
    split: str,
    checkpoint: str,
    tracker: TrackerConfig,
) -> Path:
    protocol_dir = Path(protocol_dir)
    runs, pending = load_runs(protocol_dir, split)
    if not runs:
        raise RunError(
            f"no run of {protocol_dir} has finished with {split} "
            "predictions yet"
        )
    val_runs = runs if split == "val" else load_runs(protocol_dir, "val")[0]
    results = analyze(
        runs,
        protocol,
        rules,
        checkpoint=checkpoint,
        gate=gap_gate_outcome(val_runs, rules),
    )
    results = {
        "protocol": protocol.id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pending": pending,
        "freeze": _freeze_status(protocol_dir, rules),
        "generator_gates": _generator_gates(protocol),
        **results,
    }

    out_dir = (
        protocol_dir
        / "eval"
        / (split if checkpoint == "best" else f"{split}-{checkpoint}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "comparisons.json", results)

    order = [
        (regime, arm)
        for regime in protocol.regimes
        for arm in regime_arms(protocol, regime)
    ]
    labels = {key: arm_label(key, rules.regime) for key in order}
    board = make_tracker(
        tracker,
        out_dir / "dashboard",
        f"eval/{out_dir.name}",
        protocol.id,
        {"split": split, "checkpoint": checkpoint},
    )
    try:
        judged = [
            row
            for row in [*results["chain"], *results["diagnostics"]]
            if not row["missing"]
        ]
        figures = []
        if judged:
            title = (
                f"{protocol.id}: paired effects on {SPLIT_NAMES[split]} ("
                f"{checkpoint} checkpoint)"
            )
            fig = views.forest(
                judged,
                target=rules.practical_target,
                margin=rules.noninferiority_margin,
                title=title,
            )
            figures.append(views.save(fig, "forest", out_dir, board))
            fig = views.per_class(
                judged,
                (
                    "Per-class Δ macro-AUROC, artifact slice ("
                    f"{SPLIT_NAMES[split]})"
                ),
            )
            figures.append(views.save(fig, "per_class", out_dir, board))
        figures.append(
            views.save(
                views.learning_curves(runs, order, labels),
                "learning_curves",
                out_dir,
                board,
            )
        )
        markdown = render(results, rules, figures, out_dir)
        (out_dir / "results.md").write_text(markdown, encoding="utf-8")
        board.log_text("evaluation/results", markdown, 0)
        scalars = {}
        for row in judged:
            for name in ("artifact", "clean"):
                for field in ("estimate", "ci_low", "ci_high"):
                    scalars[f"evaluation/{row['label']}/{name}/{field}"] = row[
                        name
                    ][field]
        board.log_scalars(scalars, 0)
        primary = results["chain"][0]
        board.log_summary(
            {
                "protocol": protocol.id,
                "split": split,
                "checkpoint": checkpoint,
                "rules": rules.sha256[:12],
            },
            {
                "evaluation/primary_artifact_estimate": primary["artifact"][
                    "estimate"
                ]
            }
            if not primary["missing"]
            else {},
        )
    finally:
        board.close()
    return out_dir


def render(
    results: dict[str, Any],
    rules: AnalysisConfig,
    figures: list[Path],
    out_dir: Path,
) -> str:
    split, checkpoint = SPLIT_NAMES[results["split"]], results["checkpoint"]
    table, chain = results["table"], results["chain"]
    seeds = sorted({seed for row in table for seed in row["seeds"]})
    lines = [
        (
            f"# {results['protocol']}: results on {split} ({checkpoint} "
            "checkpoint)"
        ),
        "",
        (
            f"Generated {results['generated_at']} from "
            f"{len(results['runs'])} runs, seeds {seeds}. Decision rules `"
            f"{rules.sha256[:12]}`: {results['freeze']}."
        ),
    ]
    if results["pending"]:
        lines += [
            "",
            (
                f"**Partial report:** {len(results['pending'])} planned "
                "runs are not finished ("
                f"{', '.join(results['pending'][:6])}"
                f"{(', ...' if len(results['pending']) > 6 else '')})."
            ),
        ]

    primary = chain[0]
    lines += [
        "",
        (
            f"## Primary question: {primary['label']} on the artifact "
            f"slice ({rules.regime})"
        ),
        "",
    ]
    if primary["missing"]:
        lines.append(
            "Not run yet: no finished runs for "
            f"{', '.join(primary['missing'])}."
        )
    else:
        f, art, clean = (
            primary["findings"],
            primary["artifact"],
            primary["clean"],
        )
        lines += [
            f"**{f['verdict'][0].upper()}{f['verdict'][1:]}.**",
            "",
            (
                f"- Artifact slice: Δ {_ci(art)} ({art['n_records']} ECGs "
                f"from {art['n_patients']} patients, {len(art['seeds'])} "
                "seeds; per seed "
                f"{', '.join(f'{v:+.3f}' for v in art['per_seed'])})."
            ),
            f"- Clean slice: Δ {_ci(clean)}.",
            (
                "- Some artifact improvement (CI low > 0): **"
                f"{_yes(f['positive_effect_supported'])}**."
            ),
            (
                f"- Practical target +{rules.practical_target:g}: **"
                f"{f['target_effect']}**."
            ),
            (
                "- Clean-slice non-inferiority (CI low > "
                f"{rules.noninferiority_margin:g}): **"
                f"{_yes(f['clean_noninferiority_supported'])}**."
            ),
        ]

    lines += [
        "",
        "## Fixed sequence",
        "",
        (
            "Each step is tested only if every earlier step showed a "
            "benefit (artifact CI low > 0 and clean CI low > "
            f"{rules.noninferiority_margin:g})."
        ),
        "",
        "| Step | Contrast | Artifact Δ [CI] | Clean Δ [CI] | Status |",
        "|---|---|---|---|---|",
    ]
    for index, step in enumerate(chain, 1):
        if step["missing"]:
            missing = ", ".join(step["missing"])
            lines.append(
                f"| {index} | {step['label']} | | | {step['status']} (no "
                f"runs: {missing}) |"
            )
        else:
            cells = [
                str(index),
                step["label"],
                _ci(step["artifact"]),
                _ci(step["clean"]),
                step["status"],
            ]
            lines.append("| " + " | ".join(cells) + " |")

    lines += _results_table(table, rules)
    lines += _gap_closed(results)

    lines += [
        "",
        "## Diagnostics (reported, never evidence)",
        "",
        (
            "| Contrast | Artifact Δ [CI] | Clean Δ [CI] | All records Δ "
            "[CI] | Artifact Δ per seed |"
        ),
        "|---|---|---|---|---|",
    ]
    for row in results["diagnostics"]:
        if row["missing"]:
            lines.append(
                f"| {row['label']} | not run ({', '.join(row['missing'])}) "
                "| | | |"
            )
        else:
            per_seed = ", ".join(
                f"{v:+.3f}" for v in row["artifact"]["per_seed"]
            )
            cells = [
                row["label"],
                *(_ci(row[name]) for name in ("artifact", "clean", "overall")),
                per_seed,
            ]
            lines.append("| " + " | ".join(cells) + " |")

    lines += _generation(table)
    lines += _gates(results, rules)
    hours = sum(run["wall_time_s"] for run in results["runs"]) / 3600
    lines += [
        "",
        "## Compute",
        "",
        (
            f"{len(results['runs'])} finished runs, {hours:.1f} h of "
            "training in total ("
            f"{60 * hours / max(1, len(results['runs'])):.1f} min per run)."
        ),
        "",
        "## Figures",
        "",
    ]
    lines += [
        f"![{path.stem}]({os.path.relpath(path, out_dir)})" for path in figures
    ]
    lines += [
        "",
        (
            "Δ is the paired difference in macro-AUROC, averaged over "
            f"seeds. Brackets are {100 * rules.ci_level:g}% CIs from a "
            f"crossed bootstrap ({rules.n_boot} draws) over patients and "
            "seeds; both arms are scored on the same draws. Artifact = "
            "ECGs with any PTB-XL signal-quality flag."
        ),
        "",
    ]
    return "\n".join(lines)


def _results_table(
    table: list[dict[str, Any]], rules: AnalysisConfig
) -> list[str]:
    names = [entry.name for entry in rules.gap_closed]
    header = [
        "Arm",
        "Seeds",
        "Artifact AUROC",
        "Clean AUROC",
        "All records",
        f"Δ artifact vs {rules.reference}",
    ]
    header += [f"Δ clean vs {rules.reference}"] + [
        f"Gap closed: {name}" for name in names
    ]
    lines = [
        "",
        "## The results table",
        "",
        (
            "AUROC is mean ± SD over seeds. Δ is against the reference "
            "arm of the same regime (or of the primary regime when that "
            "regime does not run it)."
        ),
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in table:
        delta = row["delta"]
        if delta is None:
            deltas = ["reference" if row["arm"] == rules.reference else "", ""]
        elif "unpaired" in delta:
            deltas = [f"not paired: {delta['unpaired']}", ""]
        else:
            suffix = (
                ""
                if delta["against"] == rules.reference
                else f" (vs {delta['against']})"
            )
            deltas = [_ci(delta["artifact"]) + suffix, _ci(delta["clean"])]
        gaps = [_percent(row["gap_closed"][name]) for name in names]
        cells = [
            row["label"],
            str(len(row["seeds"])),
            *(_mean_sd(row[name]) for name in ("artifact", "clean", "overall")),
            *deltas,
            *gaps,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _gap_closed(results: dict[str, Any]) -> list[str]:
    lines = ["", "## Gap closed (descriptive)", ""]
    for entry in results["gap_closed"]:
        if "numerator" not in entry:
            lines.append(f"- **{entry['name']}**: {entry['withheld_because']}.")
            continue
        num, den = entry["numerator"], entry["denominator"]
        ratio = (
            f"**{100 * entry['ratio']:.0f}%** closed"
            if entry["ratio"] is not None
            else f"ratio withheld: {entry['withheld_because']}"
        )
        lines.append(
            f"- **{entry['name']}**: {num['a']} - {num['b']} = {_ci(num)} "
            f"over {den['a']} - {den['b']} = {_ci(den)}; {ratio}."
        )
    return lines


def _generation(table: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Generation and exposure (mean per run)",
        "",
        (
            "| Arm | Presentations | Transform slots | Accepted / "
            "candidates | Exhausted slots | Distinct recipes per accepted "
            "|"
        ),
        "|---|---|---|---|---|---|",
    ]
    for row in table:
        e = row["exposure"]
        accepted = (
            f"{e['accepted']:,.0f} / {e['candidates']:,.0f}"
            if e["candidates"]
            else "-"
        )
        distinct = (
            f"{e['distinct_recipes'] / e['accepted']:.3f}"
            if e["accepted"]
            else "-"
        )
        lines.append(
            f"| {row['label']} | {e['presentations']:,.0f} | "
            f"{e['transform_slots']:,.0f} | {accepted} | "
            f"{e['exhausted_slots']:,.0f} | {distinct} |"
        )
    return lines


def _gates(results: dict[str, Any], rules: AnalysisConfig) -> list[str]:
    gate = results["gap_gate"]
    if gate["status"] == "not run":
        gap = f"not run ({gate['label']} has no finished validation runs)"
    else:
        threshold = f"≥ {gate['min_effect']:g}"
        gap = (
            f"{gate['label']} = {_ci(gate['artifact'])} against {threshold}"
            f": **{gate['status']}**"
        )
    lines = [
        "",
        "## Gates",
        "",
        (
            "- Gate 1 (leakage): re-run on every manifest when a trial is "
            "prepared, and again before the test fold is loaded."
        ),
        (
            f"- Gate 3 (gap gate, validation artifact slice, {rules.regime}"
            f"): {gap}."
        ),
    ]
    for regime, outcome in results["generator_gates"].items():
        lines.append(f"- Gate 4 (generator, {regime}): {outcome}.")
    lines.append(f"- Gate 5 (freeze): {results['freeze']}.")
    return lines


def _freeze_status(protocol_dir: Path, rules: AnalysisConfig) -> str:
    rules_path, claim_path = (
        protocol_dir / RULES_FILE,
        protocol_dir / CLAIM_FILE,
    )
    if not rules_path.exists():
        status = "rules not frozen"
    else:
        frozen_at = json.loads(rules_path.read_text(encoding="utf-8"))[
            "frozen_at"
        ]
        try:
            verify_frozen(protocol_dir, rules, claimed=False)
            status = f"rules frozen {frozen_at}"
        except FreezeError as error:
            status = f"rules frozen {frozen_at}, but {error}"
    if claim_path.exists():
        status += (
            "; test fold claimed "
            f"{json.loads(claim_path.read_text(encoding='utf-8'))['claimed_at']}"
        )
    return status


def _generator_gates(protocol: ProtocolConfig) -> dict[str, str]:
    outcomes = {}
    for name, regime in protocol.regimes.items():
        if (
            not regime.generator_gate
            or not Path(regime.generator_gate).exists()
        ):
            continue
        gate = json.loads(
            Path(regime.generator_gate).read_text(encoding="utf-8")
        )
        rates = ", ".join(
            f"{policy} {100 * result['acceptance_rate']:.1f}% accepted"
            for policy, result in gate["policies"].items()
        )
        outcomes[name] = (
            f"{'passed' if gate['passed'] else '**failed**'}; {rates}"
        )
    return outcomes


def _ci(report: dict[str, Any]) -> str:
    return (
        f"{report['estimate']:+.3f} [{report['ci_low']:+.3f}, "
        f"{report['ci_high']:+.3f}]"
    )


def _mean_sd(summary: dict[str, Any]) -> str:
    return f"{summary['mean']:.3f}" + (
        f" ± {summary['sd']:.3f}" if summary["sd"] is not None else ""
    )


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _write_json(path: Path, payload: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(partial, path)
