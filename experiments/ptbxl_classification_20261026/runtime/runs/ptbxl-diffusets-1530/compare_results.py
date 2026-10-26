#!/usr/bin/env python3
# ruff: noqa: RUF001 -- Minus and multiplication signs are intentional plot/report typography.
"""Compare the completed DiffuSETS classification extension with frozen old runs.

Run only after all six extension trials have completed training and test scoring:
    uv run python runs/ptbxl-diffusets-1530/compare_results.py

No training, prediction, cloud access, or old-run writes occur here. All scores are
recomputed from contract-checked predictions; paired CIs use SYNTIDE's scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from syntide.data.challenge import load_manifest
from syntide.data.schema import SUPERCLASSES
from syntide.eval.contrasts import PairedScorer, check_pairing, class_aurocs, slice_mask
from syntide.eval.runs import Run, load_run
from syntide.eval.stats import macro_auroc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = REPO / "runs/ptbxl-artifact-50-val100"
SEEDS = (0, 1, 2)
SLICES = ("overall", "clean", "artifact")
ENDPOINTS = (("val", "best"), ("val", "final"), ("test", "best"))
NEW_ARMS = ("G1", "R1")
OLD_ARMS = ("A0", "A1", "A2", "A3", "C1", "C2")
REFERENCE = "C2@full100"
ORDER = (*OLD_ARMS, *NEW_ARMS, REFERENCE)
MAIN_ORDER = ("C2", "R1", "G1", REFERENCE)
CONTRASTS = (("G1", "C2"), ("G1", "R1"), ("R1", "C2"))
LABELS = {
    "A0": "A0 · depleted real",
    "A1": "A1 · standard augmentation",
    "A2": "A2 · matched artifact edits",
    "A3": "A3 · targeted artifact edits",
    "C1": "C1 · natural mix",
    "C2": "C2 · 8,579 real",
    "G1": "G1 · real + 1,530 generated",
    "R1": "R1 · real + 1,530 parent rows",
    REFERENCE: "C2 full100 · all real",
}
COLORS = {
    "A0": "#88929E",
    "A1": "#7293AC",
    "A2": "#A18DAD",
    "A3": "#A96F8A",
    "C1": "#B0976D",
    "C2": "#366F91",
    "G1": "#D26D47",
    "R1": "#745DA6",
    REFERENCE: "#328471",
}
ENDPOINT_LABELS = {
    "val_best": "Validation · selected",
    "val_final": "Validation · final",
    "test_best": "Test · selected",
}
SLICE_LABELS = {"overall": "All ECGs", "clean": "No quality flag", "artifact": "Any quality flag"}
CAVEATS = [
    "G1 uses provisional conditioning-parent diagnostic labels. Rhythm/shape acceptance does not "
    "validate generated NORM, MI, STTC, CD or HYP morphology.",
    "Some generated conditioning parents are outside C2's 50% training pool. G1 versus C2 therefore "
    "combines generated waveforms, additional parent information and changed sampling exposure.",
    "R1 repeats each accepted slot's real parent, including duplicates, in the same appended order "
    "and with the same label vector as G1. G1 versus R1 compares generated and parent waveforms "
    "under matched row multiplicities; label mismatch remains part of the generated-waveform effect.",
    "The 1,530 accepted signals are a frozen early prefix of the QA generation run, not a random "
    "sample of its final allocation. Acceptance and conditioning may change the sample distribution.",
    "This is an exploratory extension designed after the original test results were viewed. The "
    "unchanged test fold is reused; intervals are descriptive and are not multiplicity-adjusted "
    "confirmatory tests. Validation-best results are selection-biased.",
    "Published DiffuSETS pretraining is reported as MIMIC-IV-ECG, with no PTB-XL overlap declared. "
    "Checkpoint identity is pinned; checkpoint training membership was unavailable for independent "
    "holdout-overlap verification.",
    "Artifact and clean slices use the original real held-out ECG quality flags. No parent quality "
    "flag is treated as a verified generated-signal label. Full100 is a real-data reference, "
    "not full generation.",
    "Mean ± SD describes three training seeds; paired 95% intervals separately resample patients "
    "and training seeds. These quantify different sources of variation.",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def rules() -> tuple[dict, dict]:
    protocol = read(HERE / "protocol.json")
    require(
        protocol.get("label_policy_authorized") is True,
        "Explicit provisional-label authorization is required",
    )
    require(
        protocol.get("label_policy") == "provisional_conditioning_parent_superclasses",
        "Unexpected generated-label policy",
    )
    analysis = protocol["analysis"]
    expected = {
        "primary": ["G1", "C2"],
        "key_control": ["G1", "R1"],
        "diagnostic": ["R1", "C2"],
        "contrasts": [list(pair) for pair in CONTRASTS],
        "slices": list(SLICES),
        "n_boot": 2000,
        "ci_level": 0.95,
        "bootstrap_seed": 0,
    }
    for key, value in expected.items():
        require(analysis.get(key) == value, f"Frozen analysis.{key} differs: {analysis.get(key)!r}")
    return protocol, analysis


def frozen_inputs(protocol: dict) -> tuple[dict, dict]:
    freeze = read(HERE / "experiment_freeze.json")
    require(freeze["protocol_sha256"] == sha(HERE / "protocol.json"), "Frozen protocol changed")
    require(freeze["plan_sha256"] == sha(HERE / "plan.json"), "Frozen trial plan changed")
    require(freeze["runner_sha256"] == sha(HERE / "run_experiment.py"), "Frozen training runner changed")
    require(freeze["analysis"] == protocol["analysis"], "Frozen analysis differs")
    for category in ("input_files_sha256", "original_files_sha256"):
        for filename, expected in freeze[category].items():
            path = Path(filename)
            path = path if path.is_absolute() else REPO / path
            require(sha(path) == expected, f"Frozen input changed: {path}")
    manifest_ref = protocol["data_manifest"]
    manifest_path = HERE / manifest_ref["path"]
    require(sha(manifest_path) == manifest_ref["sha256"], "Frozen data manifest changed")
    manifest = read(manifest_path)
    provenance_path = HERE / manifest["provenance"]["path"]
    require(
        provenance_path.resolve() == (HERE / "data/provenance.json").resolve(),
        "Unexpected frozen provenance path",
    )
    require(sha(provenance_path) == manifest["provenance"]["sha256"], "Data provenance changed")
    for audit in ("training_audit.json", "evaluation_audit.json"):
        require(read(HERE / audit)["passed"] is True, f"{audit} did not pass")
    claim = read(HERE / "test_claim.json")
    require(
        datetime.fromisoformat(freeze["frozen_at"]) <= datetime.fromisoformat(claim["claimed_at"]),
        "Experiment was not frozen before test scoring",
    )
    return freeze, read(provenance_path)


def check_records(reference: Run, other: Run) -> None:
    require(reference.split == other.split, "Cannot align different evaluation splits")
    for name in ("record_id", "subject_uid", "labels", "is_artifact"):
        require(
            np.array_equal(getattr(reference, name), getattr(other, name)),
            f"{other.run_dir}: held-out {name} differs from original C2",
        )


def collect(split: str) -> dict[str, list[Run]]:
    groups = {}
    plans = {
        root: {entry["run_dir"]: entry for entry in read(root / "plan.json")["trials"]}
        for root in (HERE, ORIGINAL)
    }
    claims = {
        root: {entry["run_dir"]: entry for entry in read(root / "test_claim.json")["trials"]}
        for root in (HERE, ORIGINAL)
    }
    for arm in ORDER:
        root = HERE if arm in NEW_ARMS else ORIGINAL
        regime, actual_arm = ("full100", "C2") if arm == REFERENCE else ("scarcity50", arm)
        group = [load_run(root / regime / actual_arm / f"seed{seed}", split) for seed in SEEDS]
        require([run.seed for run in group] == list(SEEDS), f"{arm}: seeds differ")
        for run in group:
            relative = str(run.run_dir.relative_to(root))
            require(
                plans[root].get(relative, {}).get("trial_id") == run.spec.trial_id,
                f"{run.run_dir}: trial is not in the frozen plan",
            )
            claim = claims[root].get(relative, {})
            require(
                claim.get("trial_id") == run.spec.trial_id
                and claim.get("best_sha256") == run.result["provenance"]["checkpoints"]["best"]["sha256"],
                f"{run.run_dir}: selected checkpoint differs from test claim",
            )
            require(run.key == (regime, actual_arm), f"{run.run_dir}: wrong regime or arm")
            require(
                (run.spec.batch_size, run.spec.max_steps, run.spec.val_every) == (64, 6500, 100),
                f"{run.run_dir}: budget or validation cadence differs",
            )
            require(run.exposure["presentations"] == 416000, f"{run.run_dir}: incomplete exposure")
        groups[arm] = group
    reference = groups["C2"][0]
    for group in groups.values():
        check_pairing(group, groups["C2"])
        for run in group:
            check_records(reference, run)
    return groups


def summarize(group: list[Run], checkpoint: str, part: str) -> dict:
    mask = slice_mask(group[0].is_artifact, part)
    values, per_class = [], []
    for run in group:
        labels, probs = run.labels[mask], run.probs[checkpoint][mask]
        values.append(macro_auroc(labels, probs))
        per_class.append(class_aurocs(labels, probs, np.ones((1, int(mask.sum()))))[0])
    require(np.isfinite(values).all(), "Non-finite score")
    classes = np.asarray(per_class)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "seeds": list(SEEDS),
        "per_seed": [float(v) for v in values],
        "n_records": int(mask.sum()),
        "n_patients": len(np.unique(group[0].subject_uid[mask])),
        "per_class": {
            name: {"mean": float(classes[:, i].mean()), "sd": float(classes[:, i].std(ddof=1))}
            if np.isfinite(classes[:, i]).all()
            else None
            for i, name in enumerate(SUPERCLASSES)
        },
    }


def run_evidence(groups: dict[str, list[Run]]) -> list[dict]:
    entries = []
    for arm in ORDER:
        for run in groups[arm]:
            files = (
                "trial.json",
                "result.json",
                "predictions_val.parquet",
                "predictions_test.parquet",
                "test_scores.json",
            )
            entries.append(
                {
                    "arm": arm,
                    "seed": run.seed,
                    "run_dir": str(run.run_dir.resolve()),
                    "trial_id": run.spec.trial_id,
                    "trainer_fingerprint": run.fingerprint,
                    "init_sha256": run.init_sha256,
                    "exposure": run.exposure,
                    "best_step": run.steps["best"],
                    "final_step": run.steps["final"],
                    "checkpoint_provenance": run.result["provenance"]["checkpoints"],
                    "files_sha256": {name: sha(run.run_dir / name) for name in files},
                }
            )
    return entries


def compare(protocol: dict, analysis: dict) -> dict:
    # Refuse partial results. Nothing is written until every endpoint and contrast validates.
    freeze, provenance = frozen_inputs(protocol)
    groups = {split: collect(split) for split in ("val", "test")}
    reference_spec = groups["val"][REFERENCE][0].spec
    reference_manifest_path = REPO / reference_spec.manifest
    reference_manifest = load_manifest(reference_manifest_path)
    require(
        reference_manifest.sha256 == reference_spec.manifest_sha256,
        "Full100 manifest differs from the reference trial",
    )
    n_reference = len(reference_manifest.roles["pool"])
    require(
        len(set(reference_manifest.roles["pool"])) == n_reference,
        "Full100 pool contains duplicate ECG identities",
    )
    n_real, n_extra = provenance["original_pool_ecgs"], provenance["snapshot_accepted_candidates"]
    setup = {
        "trainer": groups["val"]["C2"][0].result["trainer"],
        "model": "ResNet1d",
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "signal": {
            "leads": len(provenance["lead_order"]),
            "sample_rate_hz": provenance["sample_rate_hz"],
            "duration_seconds": 10,
            "samples_per_lead": 1000,
            "units": provenance["units"],
        },
        "arms": {
            "C2": {
                "training_rows": n_real,
                "unique_real_ecgs": n_real,
                "definition": "Original 50% pool; all rows are real ECGs.",
            },
            "G1": {
                "training_rows": n_real + n_extra,
                "unique_real_ecgs": n_real,
                "definition": "C2 plus 1,530 fully generated ECGs with provisional "
                "conditioning-parent labels.",
            },
            "R1": {
                "training_rows": n_real + n_extra,
                "unique_real_ecgs": provenance["combined_distinct_real_root_ecgs"],
                "definition": "C2 plus the same 1,530 parent slots as real ECGs, "
                "retaining duplicate parents.",
            },
            REFERENCE: {
                "training_rows": n_reference,
                "unique_real_ecgs": n_reference,
                "definition": "All real ECGs in the original full100 training pool.",
            },
        },
    }
    summaries, contrasts = {}, {}
    for split, checkpoint in ENDPOINTS:
        name = f"{split}_{checkpoint}"
        print(f"Scoring {name}: all arms and paired patient × seed intervals", flush=True)
        arms = groups[split]
        summaries[name] = {
            arm: {part: summarize(arms[arm], checkpoint, part) for part in SLICES} for arm in ORDER
        }
        scorer = PairedScorer(
            arms["C2"][0], checkpoint=checkpoint, n_boot=analysis["n_boot"], seed=analysis["bootstrap_seed"]
        )
        contrasts[name] = []
        for a, b in CONTRASTS:
            role = dict(zip(CONTRASTS, ("primary", "key_control", "diagnostic"), strict=True))[(a, b)]
            for part in SLICES:
                report = scorer.contrast(
                    arms[a], arms[b], part, ci_level=analysis["ci_level"], labels=(a, b)
                ).to_dict()
                require(
                    all(np.isfinite(report[k]) for k in ("estimate", "ci_low", "ci_high")),
                    f"Invalid bootstrap interval for {name}/{a}-{b}/{part}",
                )
                contrasts[name].append({"role": role, **report})
    evidence = run_evidence(groups["val"])
    provenance_path = HERE / "data/provenance.json"
    inputs = {
        "protocol_sha256": sha(HERE / "protocol.json"),
        "analysis_sha256": digest(analysis),
        "data_provenance_sha256": sha(provenance_path),
        "script_sha256": sha(Path(__file__)),
        "runs": evidence,
        "freeze_sha256": sha(HERE / "experiment_freeze.json"),
        "full100_manifest_sha256": sha(reference_manifest_path),
        "run_roots_sha256": {
            str(root): {name: sha(root / name) for name in ("plan.json", "test_claim.json")}
            for root in (HERE, ORIGINAL)
        },
    }
    return {
        "schema_version": "syntide.diffusets-classification-comparison@1",
        "generated_at": datetime.now(UTC).isoformat(),
        "input_identity": digest(inputs),
        "inputs": inputs,
        "protocol": protocol,
        "analysis": analysis,
        "experiment_freeze": freeze,
        "data_provenance": provenance,
        "experimental_setup": setup,
        "caveats": CAVEATS,
        "metric": "macro_auroc",
        "superclasses": list(SUPERCLASSES),
        "arm_labels": LABELS,
        "arm_order": list(ORDER),
        "summaries": summaries,
        "contrasts": contrasts,
    }


def cell(value: dict) -> str:
    return f"{value['mean']:.4f} ± {value['sd']:.4f}"


def ci_cell(row: dict) -> str:
    return f"{row['estimate']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"


def markdown(result: dict, figure_files: list[str], output: Path) -> str:
    provenance = result["data_provenance"]
    coverage = provenance["conditioning_coverage"]
    setup = result["experimental_setup"]
    trainer, signal = setup["trainer"], setup["signal"]
    model, optim = trainer["config"]["model"], trainer["config"]["optim"]
    lines = [
        "# DiffuSETS classification extension: 1,530 generated ECGs",
        "",
        "All 27 old and new trials were checked with `load_run`; compared arms passed "
        "`check_pairing`, including trainer, initialization, exposure, held-out record, patient, "
        "label and quality-flag agreement. Original run files were read only.",
        "",
        "G1 adds the 1,530 frozen accepted DiffuSETS ECGs to C2's 8,579 real ECGs. R1 adds "
        "the corresponding 1,530 real parent rows, retaining duplicate parents and "
        "matching G1's order and provisional label multiplicity. Each new arm has 10,109 rows.",
        "",
        "All runs use seeds 0, 1 and 2, 6,500 optimizer steps, batch size 64, 416,000 presentations "
        "and validation every 100 steps. The best checkpoint is selected by validation overall macro-AUROC.",
        "",
        "**Interpretation:** G1 − C2 is the requested exploratory comparison. G1 − R1 is the "
        "key matched-parent control; R1 − C2 describes the added parent information and reweighting. "
        "The full100 C2 arm is a descriptive real-data reference.",
        "",
        f"The {provenance['snapshot_completed_candidates']:,}-candidate snapshot includes "
        f"{provenance['snapshot_accepted_candidates']:,} accepted and "
        f"{provenance['snapshot_rejected_candidates']:,} rejected candidates. Accepted slots use "
        f"{provenance['distinct_conditioning_ecgs']:,} distinct conditioning ECGs from "
        f"{provenance['distinct_conditioning_patients']:,} patients. "
        f"{coverage.get('outside_original_pool', 0):,} accepted slots have parents outside C2's pool, "
        f"adding {provenance['additional_distinct_real_root_ecgs']:,} distinct source ECGs and "
        f"{provenance['additional_distinct_real_root_patients']:,} source patients. "
        "These parents remain in the training folds; no validation/test record or patient is added.",
        "",
        "## Experimental setup",
        "",
        "| Arm | Training rows | Unique real ECGs used as waveforms | Definition |",
        "|---|---:|---:|---|",
    ]
    for arm in ("C2", "G1", "R1", REFERENCE):
        row = setup["arms"][arm]
        lines.append(
            f"| {arm} | {row['training_rows']:,} | {row['unique_real_ecgs']:,} | {row['definition']} |"
        )
    lines += [
        "",
        "Unique real ECGs are counted by source record identity, not by bytewise waveform deduplication. "
        "G1's conditioning parents can supply information without their real waveforms appearing in G1. "
        "R1 includes those parent waveforms; its labels and appended slot order match G1 exactly.",
        "",
        f"The classifier is **{setup['model']}**, width **{model['width']}**, kernel size "
        f"**{model['kernel_size']}**, trained with **{setup['optimizer']}** (learning rate "
        f"**{optim['lr']:g}**, weight decay **{optim['weight_decay']:g}**) and "
        f"**{setup['loss']}** binary cross-entropy for the five multilabel targets. "
        f"All runs use the **{trainer['device'].upper()}** device backend.",
        "",
        f"Each input contains **{signal['leads']} leads at {signal['sample_rate_hz']} Hz for "
        f"{signal['duration_seconds']} seconds**, or {signal['samples_per_lead']:,} samples per lead, "
        f"in {signal['units']}. Validation contains {provenance['validation_ecgs']:,} real ECGs and "
        f"test contains {provenance['test_ecgs']:,}. "
        "These held-out ECGs and labels are identical across arms.",
        "",
    ]
    for endpoint in ("test_best", "val_best", "val_final"):
        lines += [
            f"## {ENDPOINT_LABELS[endpoint]}",
            "",
            "Macro-AUROC, mean ± sample SD across three seeds.",
            "",
            "| Arm | All ECGs | No quality flag | Any quality flag |",
            "|---|---:|---:|---:|",
        ]
        for arm in ORDER:
            row = result["summaries"][endpoint][arm]
            lines.append(f"| {LABELS[arm]} | " + " | ".join(cell(row[p]) for p in SLICES) + " |")
        lines += [
            "",
            "Paired differences: 2,000 crossed patient × seed bootstrap draws, "
            "95% percentile intervals, bootstrap seed 0. Values are AUROC units.",
            "",
            "| Contrast | All ECGs Δ [95% CI] | No quality flag Δ [95% CI] | Any quality flag Δ [95% CI] |",
            "|---|---:|---:|---:|",
        ]
        rows = result["contrasts"][endpoint]
        for a, b in CONTRASTS:
            cells = [
                ci_cell(next(r for r in rows if (r["a"], r["b"], r["slice"]) == (a, b, part)))
                for part in SLICES
            ]
            lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
        lines += [""]
    lines += [
        "## Seed-level test differences",
        "",
        "| Contrast | Slice | Seed 0 | Seed 1 | Seed 2 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["contrasts"]["test_best"]:
        lines.append(
            f"| {row['a']} − {row['b']} | {row['slice']} | "
            + " | ".join(f"{v:+.4f}" for v in row["per_seed"])
            + " |"
        )
    lines += [
        "",
        "## Checkpoint selection",
        "",
        "| Arm | Best steps, seeds 0 / 1 / 2 | Best step after 80% of budget |",
        "|---|---|---:|",
    ]
    for arm in ORDER:
        runs = [r for r in result["inputs"]["runs"] if r["arm"] == arm]
        steps = [r["best_step"] for r in runs]
        lines.append(
            f"| {LABELS[arm]} | " + " / ".join(map(str, steps)) + f" | {sum(s > 5200 for s in steps)} / 3 |"
        )
    lines += [
        "",
        "Checkpoint timing is descriptive; an early best checkpoint does not prove convergence.",
        "",
        "## Scope and limitations",
        "",
        *[f"- {text}" for text in result["caveats"]],
        "",
        "Exact frozen cohort counts, ancestry coverage, snapshot identities and label-policy "
        "provenance are preserved in `comparisons.json` → `data_provenance` and "
        f"[data/provenance.json]({HERE / 'data/provenance.json'}).",
        "",
        "## Reproduction commands",
        "",
        "The sequence below records collection, data freezing, training, held-out evaluation and comparison. "
        "Run it from a fresh experiment state with the same pinned source and credentials for collection. "
        "An existing completed experiment intentionally rejects replanning or training after its test claim.",
        "",
        "```sh",
        f"cd {shlex.quote(str(REPO))}",
        f"uv run python {shlex.quote(str(HERE / 'collect_snapshot.py'))} prepare",
        f"uv run python {shlex.quote(str(HERE / 'collect_snapshot.py'))} collect",
        f"uv run python {shlex.quote(str(HERE / 'prepare_data.py'))}",
        f"uv run python {shlex.quote(str(HERE / 'run_experiment.py'))} plan",
        f"uv run python {shlex.quote(str(HERE / 'run_experiment.py'))} train",
        f"uv run python {shlex.quote(str(HERE / 'run_experiment.py'))} evaluate",
        f"uv run python {shlex.quote(str(HERE / 'compare_results.py'))} --output {shlex.quote(str(output))}",
        "```",
        "",
        "## Figures",
        "",
    ]
    for name in figure_files:
        if name.endswith(".png"):
            lines += [f"![{Path(name).stem.replace('_', ' ')}]({name})", ""]
    lines += [
        f"Input identity: `{result['input_identity']}`.",
        "",
        "All underlying predictions, trial definitions, results and checkpoint identities are "
        "listed with hashes in `comparisons.json`. SVG versions of the figures are included.",
        "",
    ]
    return "\n".join(lines)


def figures(result: dict, output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 10,
            "figure.facecolor": "#F8F9FB",
            "axes.facecolor": "#F8F9FB",
            "axes.edgecolor": "#CCD3DC",
            "axes.labelcolor": "#344455",
            "text.color": "#243244",
            "xtick.color": "#526476",
            "ytick.color": "#344455",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "grid.color": "#E0E5EC",
            "grid.linewidth": 0.7,
            "savefig.facecolor": "#F8F9FB",
            "svg.fonttype": "none",
        }
    )
    saved = []

    def save(fig, name):
        for suffix in ("png", "svg"):
            filename = f"{name}.{suffix}"
            fig.savefig(output / filename, dpi=190, bbox_inches="tight")
            saved.append(str((output / filename).resolve()))
        plt.close(fig)

    def score_panel(ax, endpoint, part, arms, limits):
        for y, arm in enumerate(arms):
            row = result["summaries"][endpoint][arm][part]
            ax.errorbar(
                row["mean"],
                y,
                xerr=row["sd"],
                fmt="o",
                markersize=7,
                capsize=4,
                elinewidth=1.8,
                color=COLORS[arm],
                zorder=3,
            )
            for offset, value in zip((-0.12, 0, 0.12), row["per_seed"], strict=True):
                ax.scatter(value, y + offset, s=12, color=COLORS[arm], alpha=0.4, zorder=2)
        ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms])
        ax.invert_yaxis()
        ax.set_xlim(*limits)
        ax.set_ylim(len(arms) - 0.55, -0.55)
        ax.xaxis.set_major_locator(MultipleLocator(0.02))
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0, pad=7)
        ax.set_xlabel("Macro-AUROC")

    def score_limits(endpoints, arms):
        rows = [result["summaries"][e][a][p] for e in endpoints for a in arms for p in SLICES]
        low = [min(min(r["per_seed"]), r["mean"] - r["sd"]) for r in rows]
        high = [max(max(r["per_seed"]), r["mean"] + r["sd"]) for r in rows]
        lo, hi = min(low), max(high)
        padding = max(0.006, (hi - lo) * 0.12)
        return max(0, lo - padding), min(1, hi + padding)

    endpoints = ("val_best", "test_best")
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True)
    limits = score_limits(endpoints, MAIN_ORDER)
    for row, endpoint in enumerate(endpoints):
        for col, part in enumerate(SLICES):
            ax = axes[row, col]
            score_panel(ax, endpoint, part, MAIN_ORDER, limits)
            ax.set_title(f"{ENDPOINT_LABELS[endpoint]}\n{SLICE_LABELS[part]}", loc="left", pad=16)
            if col:
                ax.set_yticklabels([])
    fig.suptitle("What did the generated ECGs add?", x=0.02, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.02,
        0.92,
        "Same model, seeds and training budget · dots: mean AUROC · bars: ±1 seed SD",
        color="#526476",
        fontsize=11,
    )
    fig.text(
        0.02,
        0.02,
        "G1: provisional conditioning labels. R1: matched real parents, including repeats. "
        "Previously viewed test fold; exploratory comparison.",
        fontsize=9,
        color="#526476",
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.82, bottom=0.13, wspace=0.18, hspace=0.58)
    save(fig, "main_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    limits = score_limits(("test_best",), ORDER)
    for col, part in enumerate(SLICES):
        score_panel(axes[col], "test_best", part, ORDER, limits)
        axes[col].set_title(SLICE_LABELS[part], loc="left", pad=14)
        if col:
            axes[col].set_yticklabels([])
    fig.suptitle("All arms · real test ECGs", x=0.02, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.02,
        0.92,
        "Validation-selected checkpoints · mean ±1 sample SD across seeds 0, 1 and 2",
        fontsize=11,
        color="#526476",
    )
    fig.text(
        0.02,
        0.025,
        "A2/A3 use artifact editing. G1 uses full DiffuSETS generation with provisional "
        "labels. Full100 uses all real training data.",
        fontsize=9,
        color="#526476",
    )
    fig.subplots_adjust(left=0.24, right=0.98, top=0.82, bottom=0.12, wspace=0.2)
    save(fig, "all_arms_test")

    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    all_contrasts = [r for rows in result["contrasts"].values() for r in rows]
    extent = max(max(abs(r["ci_low"]), abs(r["ci_high"])) for r in all_contrasts) * 1.13
    extent = max(extent, 0.005)
    for row, (split, checkpoint) in enumerate(ENDPOINTS):
        endpoint = f"{split}_{checkpoint}"
        for col, part in enumerate(SLICES):
            ax = axes[row, col]
            for y, (a, b) in enumerate(CONTRASTS):
                r = next(
                    r for r in result["contrasts"][endpoint] if (r["a"], r["b"], r["slice"]) == (a, b, part)
                )
                color = "#D26D47" if a == "G1" else "#745DA6"
                ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=color, linewidth=2)
                ax.scatter(r["estimate"], y, color=color, s=35, zorder=3)
            ax.axvline(0, color="#7E8B98", linewidth=0.9, linestyle="--")
            ax.set_xlim(-extent, extent)
            ax.set_ylim(2.55, -0.55)
            ax.set_yticks(range(3), [f"{a} − {b}" for a, b in CONTRASTS] if col == 0 else [""] * 3)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x")
            ax.set_xlabel("Paired Δ macro-AUROC")
            ax.set_title(f"{ENDPOINT_LABELS[endpoint]} · {SLICE_LABELS[part]}", loc="left", fontsize=11)
    fig.suptitle("Paired differences and uncertainty", x=0.02, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.02,
        0.93,
        "2,000 patient × seed bootstrap draws · 95% intervals · positive favors the first arm",
        fontsize=11,
        color="#526476",
    )
    fig.text(
        0.02,
        0.025,
        "Exploratory intervals, without multiplicity adjustment. G1 − C2 is primary; "
        "G1 − R1 is the matched-parent control.",
        fontsize=9,
        color="#526476",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.11, wspace=0.27, hspace=0.65)
    save(fig, "paired_differences")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "eval/comparison",
        help="Report directory, which must stay inside this new run",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(HERE) and output != HERE, "Output must be inside the new run directory")
    require(not output.is_relative_to(ORIGINAL), "Original runs are read-only")
    protocol, analysis = rules()
    result = compare(protocol, analysis)
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "comparisons.json"
    if existing.exists():
        require(
            read(existing)["input_identity"] == result["input_identity"],
            "Existing report has different inputs; choose another output directory",
        )
    figure_files = figures(result, output)
    result["figures"] = figure_files
    (output / "comparisons.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "results.md").write_text(markdown(result, figure_files, output), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "input_identity": result["input_identity"],
                "report": str(output / "results.md"),
                "figures": figure_files,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
