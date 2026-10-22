"""``mode=audit``: what the generator made, before any training result is read (report-only).

For each regime with an editor calibration and a Gate 4 result, from the gate's 1,000 recipes per policy:

- **provenance**: every recipe is regenerated from its ledger row and must give back its ``recipe_id``;
  every parent must be in ``depleted`` (Gate 1's recipe rule);
- **coverage**: natural severities of the retained artifact ECGs vs what A2 (coverage-matched) and A3
  (gap-targeted) requested, per artifact type, and the type mix;
- **acceptance**: accepted share by policy and type, and the rejection reasons;
- **signal inspection**: one fixed accepted A3 recipe per type (the first in the gate's order): parent,
  edit and residual on the lead where the requested amplitude appears, with the edited interval shaded;
- **label audit**: A0's best checkpoint (lowest seed, primary regime), run through its trainer's predictor
  on each parent and its accepted edit. Mean |Δp| and the share of ECGs whose 0.5-thresholded label
  changes show whether edits erase diagnostic evidence. It never filters anything.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from syntide.data.schema import FS_HZ, SUPERCLASSES
from syntide.eval import views
from syntide.eval.analysis import AnalysisConfig, arm_key
from syntide.eval.runs import load_runs
from syntide.eval.tracking import TrackerConfig, make_tracker
from syntide.generate.editor import edit
from syntide.generate.ledger import Recipe
from syntide.generate.plan import ProtocolConfig
from syntide.generate.policy import POLICIES, load_calibration

POLICY_ARMS = {"coverage_matched": "A2", "gap_targeted": "A3"}


class AuditError(RuntimeError):
    """The audit's inputs are missing or a recipe does not regenerate."""


def write_audit(
    protocol_dir: str | Path,
    protocol: ProtocolConfig,
    rules: AnalysisConfig,
    tracker: TrackerConfig,
    *,
    device: str = "cpu",
    registry: str | None = None,
) -> Path:
    from syntide.data.adapter import PtbXlAdapter
    from syntide.data.challenge import load_manifest
    from syntide.data.leakage import check_recipe_parents
    from syntide.eval.holdout import import_predictor

    protocol_dir = Path(protocol_dir)
    out_dir = protocol_dir / "eval" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = PtbXlAdapter(registry or protocol.registry, protocol.dataset)
    model = _audit_model(protocol_dir, rules)
    board = make_tracker(tracker, out_dir / "dashboard", "eval/audit", protocol.id, {"mode": "audit"})
    results: dict[str, Any] = {
        "protocol": protocol.id,
        "label_audit_model": model and model["run_dir"],
        "regimes": {},
    }
    try:
        for name, regime in protocol.regimes.items():
            if not regime.calibration or not regime.generator_gate:
                continue
            ledger = pq.read_table(
                Path(regime.generator_gate).with_suffix("") / "recipes.parquet"
            ).to_pylist()
            calibration = load_calibration(regime.calibration)
            manifest = load_manifest(Path(regime.manifest))
            check_recipe_parents({row["parent_record_id"] for row in ledger}, manifest)
            parent_ids = sorted({row["parent_record_id"] for row in ledger})
            parents = adapter.load(parent_ids)
            position = {record: i for i, record in enumerate(parents.record_id)}
            edits = {}
            for row in ledger:
                recipe = Recipe(
                    row["parent_record_id"],
                    row["generator"],
                    row["version"],
                    json.loads(row["controls_requested"]),
                    row["seed_key"],
                    row["gap_report_id"],
                )
                if recipe.recipe_id != row["recipe_id"]:
                    raise AuditError(f"{name}: ledger row {row['rng_stream']} does not hash to its recipe_id")
                edits[row["recipe_id"]] = edit(parents.x[position[recipe.parent_id]], recipe, FS_HZ)

            regime_out: dict[str, Any] = {
                "recipes": len(ledger),
                "regenerated": len(edits),
                "parents_in_depleted": True,
                "acceptance": _acceptance(ledger),
                "requested_type_mix": {
                    policy: dict(
                        Counter(_controls(row)["artifact_type"] for row in ledger if row["policy"] == policy)
                    )
                    for policy in POLICIES
                },
                "retained": {kind: info["n_retained"] for kind, info in calibration["types"].items()},
            }
            fig = views.coverage(ledger, calibration, f"{protocol.id} {name}: what each policy asked for")
            views.save(fig, f"coverage_{name}", out_dir, board)
            examples = _examples(ledger)
            fig = views.signal_examples(
                [
                    (row, parents.x[position[row["parent_record_id"]]], edits[row["recipe_id"]])
                    for row in examples
                ],
                FS_HZ,
                f"{protocol.id} {name}: A3 edits (first accepted gate recipe per type)",
            )
            views.save(fig, f"signal_examples_{name}", out_dir, board)
            regime_out["examples"] = [row["recipe_id"] for row in examples]
            if model is not None:
                regime_out["label_audit"] = _label_audit(
                    ledger, parents, position, edits, model, device, import_predictor
                )
            results["regimes"][name] = regime_out

        markdown = render_audit(results)
        (out_dir / "audit.md").write_text(markdown, encoding="utf-8")
        (out_dir / "audit.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        board.log_text("evaluation/audit", markdown, 0)
    finally:
        board.close()
    return out_dir


def _controls(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["controls_requested"])


def _acceptance(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for policy in POLICIES:
        rows = [row for row in ledger if row["policy"] == policy]
        by_type = {}
        for kind in sorted({_controls(row)["artifact_type"] for row in rows}):
            typed = [row for row in rows if _controls(row)["artifact_type"] == kind]
            by_type[kind] = sum(row["decision"] == "accepted" for row in typed) / len(typed)
        out[policy] = {
            "candidates": len(rows),
            "accepted": sum(row["decision"] == "accepted" for row in rows),
            "by_type": by_type,
            "rejections": dict(Counter(row["reason"] for row in rows if row["decision"] != "accepted")),
        }
    return out


def _examples(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The first accepted gap-targeted recipe per artifact type, in the gate's order (fixed across runs)."""
    chosen: dict[str, dict[str, Any]] = {}
    for row in sorted(ledger, key=lambda row: row["step"]):
        kind = _controls(row)["artifact_type"]
        if row["policy"] == "gap_targeted" and row["decision"] == "accepted" and kind not in chosen:
            chosen[kind] = row
    return [chosen[kind] for kind in sorted(chosen)]


def _audit_model(protocol_dir: Path, rules: AnalysisConfig) -> dict[str, Any] | None:
    """A0's best checkpoint in the primary regime (lowest finished seed), or None if A0 has not finished."""
    runs, _ = load_runs(protocol_dir, "val")
    a0 = sorted((run for run in runs if run.key == arm_key("A0", rules.regime)), key=lambda run: run.seed)
    if not a0:
        return None
    run = a0[0]
    best = run.result["provenance"]["checkpoints"]["best"]
    return {
        "run_dir": str(run.run_dir),
        "checkpoint": run.run_dir / best["path"],
        "step": best["step"],
        "predictor": run.result["trainer"]["predictor"],
    }


def _label_audit(ledger, parents, position, edits, model, device, import_predictor) -> dict[str, Any]:
    from syntide.trial import Checkpoint

    predict = import_predictor(model["predictor"])
    checkpoint = Checkpoint(model["checkpoint"], model["step"])
    p_parent = np.asarray(predict(checkpoint, parents.x, device), dtype=np.float64)
    out = {}
    for policy in POLICIES:
        rows = [row for row in ledger if row["policy"] == policy and row["decision"] == "accepted"]
        x = np.stack([edits[row["recipe_id"]].x for row in rows])
        p_edit = np.asarray(predict(checkpoint, x, device), dtype=np.float64)
        p_before = p_parent[[position[row["parent_record_id"]] for row in rows]]
        delta = np.abs(p_edit - p_before)
        flips = (p_edit >= 0.5) != (p_before >= 0.5)
        types = np.array([_controls(row)["artifact_type"] for row in rows])
        out[policy] = {
            "edits": len(rows),
            "mean_abs_dp": float(delta.mean()),
            "flip_rate": float(flips.any(axis=1).mean()),
            "per_class_mean_abs_dp": dict(zip(SUPERCLASSES, map(float, delta.mean(axis=0)), strict=True)),
            "per_class_flip_rate": dict(zip(SUPERCLASSES, map(float, flips.mean(axis=0)), strict=True)),
            "by_type": {
                kind: {
                    "edits": int((types == kind).sum()),
                    "flip_rate": float(flips[types == kind].any(axis=1).mean()),
                }
                for kind in sorted(set(types))
            },
        }
    return out


def render_audit(results: dict[str, Any]) -> str:
    lines = [f"# {results['protocol']}: generator audit (report-only)", ""]
    model = results["label_audit_model"]
    for name, regime in results["regimes"].items():
        lines += [
            f"## {name}",
            "",
            f"- Provenance: {regime['regenerated']} of {regime['recipes']} gate recipes regenerated from "
            "their ledger rows with matching ids; every parent is in `depleted`.",
            f"- Retained natural artifact ECGs per type (calibration): {regime['retained']}.",
            "",
            "| Policy (arm) | Accepted | Accepted by type | Rejections | Requested type mix |",
            "|---|---|---|---|---|",
        ]
        for policy, acceptance in regime["acceptance"].items():
            by_type = ", ".join(f"{kind} {100 * rate:.1f}%" for kind, rate in acceptance["by_type"].items())
            rejections = (
                ", ".join(f"{reason} {n}" for reason, n in acceptance["rejections"].items()) or "none"
            )
            total = sum(regime["requested_type_mix"][policy].values())
            mix = ", ".join(
                f"{k} {100 * v / total:.0f}%" for k, v in sorted(regime["requested_type_mix"][policy].items())
            )
            accepted = f"{acceptance['accepted']} / {acceptance['candidates']}"
            cells = [f"{policy} ({POLICY_ARMS[policy]})", accepted, by_type, rejections, mix]
            lines.append("| " + " | ".join(cells) + " |")
        audit = regime.get("label_audit")
        lines += [
            "",
            f"### Label audit ({'model: ' + model if model else 'not run: A0 has not finished'})",
            "",
        ]
        if audit:
            lines += [
                "Predictions of A0's best checkpoint on each parent vs its accepted edit. A flip is any "
                "class whose 0.5-thresholded label changes.",
                "",
                "| Policy | Edits | Mean abs Δp | ECGs with a flipped class | Flip rate by type |",
                "|---|---|---|---|---|",
            ]
            for policy, row in audit.items():
                by_type = ", ".join(f"{k} {100 * v['flip_rate']:.1f}%" for k, v in row["by_type"].items())
                flips = f"{100 * row['flip_rate']:.1f}%"
                cells = [policy, str(row["edits"]), f"{row['mean_abs_dp']:.4f}", flips, by_type]
                lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            f"![coverage](figures/coverage_{name}.png)",
            f"![signal examples](figures/signal_examples_{name}.png)",
            "",
        ]
    return "\n".join(lines)
