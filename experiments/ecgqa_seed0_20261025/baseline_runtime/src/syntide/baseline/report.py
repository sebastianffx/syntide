"""Verify and report saved RunPod baseline artifacts, without loading a model.

Example::

    python -m syntide.baseline.report --run-dir runs/ecgqa --data-dir runs/ecgqa-data \
        --output runs/ecgqa/report

The normal command requires complete seeds 0, 1, 2, full validation/test and
zero-signal predictions, and the convergence guard. ``--allow-partial`` writes an
explicitly incomplete diagnostic report without presenting missing results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from syntide.baseline.metrics import (
    PAPER_REFERENCE,
    fit_majority_answers,
    majority_predictions,
    score_predictions,
)

METRICS = (
    "accuracy",
    "weighted_template_macro_f1",
    "template_macro_f1",
    "template_macro_accuracy",
    "invalid_rate",
)
SCENARIOS = ("validation", "test", "validation_zero", "test_zero")
DEVIATIONS = [
    "Patient-disjoint PTB-XL folds 1-8/9/10 replace the released CoT train/validation/test split.",
    "Training uses the real-only 50% patient pool, with every training question/rationale row retained.",
    "Validation/test deduplicate (record_id, question), choosing the lowest task_id; the paper does not "
    "specify this deduplication.",
    "No TSQA/M4 curriculum warmup or released ECG-trained weights; the Gemma base model is frozen.",
    "A fixed optimizer-step budget and fixed validation subset select the checkpoint; the paper reports "
    "up to 200 epochs and early-stopping patience 5.",
    "Primary scoring requires a last case-insensitive Answer: marker and strips punctuation. The optional "
    "upstream parser accepts bare answers and requires a case-sensitive marker when present.",
    "F1 aggregation follows pinned upstream code: macro over all options within each template, then "
    "sample-weighted or unweighted across templates. The paper itself does not identify its averaging rule.",
    "Per-lead z-score normalization follows the upstream ECG loader; the paper describes [-1,1] inputs.",
    "The CUDA trainer corrects upstream joint-media marker and target-loss masking behavior; the saved "
    "prompt_version identifies this implementation.",
    "The upstream joint lead-frame Perceiver has no frame embeddings, so its waveform representation is "
    "invariant to lead permutation. This limitation is retained for baseline fidelity.",
    "Zero-signal results hold all text, clinical context, lead labels, and original lead mean/std values "
    "fixed and replace only normalized signal tensors. Their gap measures waveform contribution under "
    "this ablation, not clinical correctness of generated rationales.",
    "The saved max_length and max_new_tokens fix input/target truncation and greedy generation limits. "
    "These limits and any truncated training rationales must remain fixed for synthetic-data comparisons.",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _stats(values: list[float]) -> dict:
    return {
        "n_seeds": len(values),
        "mean": statistics.mean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def _read_predictions(path: Path, expected: dict) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank prediction row")
            rows.append(json.loads(line))
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate task IDs")
    if len(rows) != expected["questions"] or _json_hash(ids) != expected["task_ids_sha256"]:
        raise ValueError(f"{path}: prediction coverage/order does not match the full frozen split")
    return rows


def _aggregate(runs: list[dict]) -> dict:
    result = {}
    for scenario in SCENARIOS:
        result[scenario] = {}
        for group in ("all", "noisy", "clean"):
            result[scenario][group] = {}
            for metric in METRICS:
                values = [
                    run["evaluations"][scenario]["strict"]["groups"][group][metric]
                    for run in runs
                    if scenario in run["evaluations"]
                ]
                result[scenario][group][metric] = _stats([v for v in values if v is not None])
    return result


def build_report(
    run_dir: Path, manifest: dict, majority: dict[str, str], *, expected_seeds=(0, 1, 2), allow_partial=False
) -> dict:
    """Recompute all scores from verified predictions and aggregate available seeds.

    ``manifest`` and ``majority`` are explicitly supplied so pure-data tests never
    need a model, torch, a registry, or even a Parquet dependency.
    """
    run_dir = Path(run_dir)
    configs = (
        [run_dir / "config.json"]
        if (run_dir / "config.json").is_file()
        else sorted(run_dir.glob("seed*/config.json"))
    )
    runs, issues, seen, references = [], [], set(), {}
    comparison_config = None
    for config_path in configs:
        config = _read_json(config_path)
        seed, directory = config["seed"], config_path.parent
        if seed not in expected_seeds:
            continue
        if seed in seen:
            raise ValueError(f"Multiple runs for seed {seed}; select an unambiguous run directory")
        seen.add(seed)
        if config["manifest_sha256"] != manifest["sha256"]:
            raise ValueError(f"{directory}: training/export manifest mismatch")
        current_config = {k: v for k, v in config.items() if k not in {"seed", "init_sha256"}}
        if comparison_config is not None and comparison_config != current_config:
            raise ValueError(
                "Seed configurations differ beyond seed/initialization; do not aggregate experiments"
            )
        comparison_config = current_config
        result_path = directory / "train_result.json"
        if not result_path.exists():
            issues.append(f"Seed {seed}: training result not available")
            continue
        trained = _read_json(result_path)
        if trained.get("status") != "trained" or trained.get("seed") != seed:
            raise ValueError(f"{directory}: invalid training result")
        if trained["manifest_sha256"] != manifest["sha256"]:
            raise ValueError(f"{directory}: training result manifest mismatch")
        if trained["max_steps"] != config["max_steps"]:
            raise ValueError(f"{directory}: training budget mismatch")
        if trained.get("smoke_only"):
            issues.append(f"Seed {seed}: smoke-only training, not a completed baseline")
        if not trained["best_step"] < 0.8 * trained["max_steps"]:
            issues.append(f"Seed {seed}: best step is not before 80% of the training budget")
        run = {
            "seed": seed,
            "directory": str(directory.resolve()),
            "config": config,
            "training": trained,
            "evaluations": {},
            "signal_ablation_gap": {},
        }
        for scenario in SCENARIOS:
            split, zero = scenario.removesuffix("_zero"), scenario.endswith("_zero")
            predictions_path = directory / f"{scenario}_predictions.jsonl"
            metrics_path = directory / f"{scenario}_metrics.json"
            if not predictions_path.is_file() or not metrics_path.is_file():
                issues.append(f"Seed {seed}: {scenario} full evaluation unavailable")
                continue
            saved = _read_json(metrics_path)
            expected_metadata = {
                "split": split,
                "zero_signal": zero,
                "seed": seed,
                "checkpoint_step": trained["best_step"],
                "checkpoint_sha256": trained["best_checkpoint_sha256"],
                "manifest_sha256": manifest["sha256"],
                "prediction_sha256": _file_hash(predictions_path),
                "questions": manifest["splits"][split]["questions"],
            }
            for key, value in expected_metadata.items():
                if saved.get(key) != value:
                    raise ValueError(f"{metrics_path}: {key} does not match frozen run/predictions")
            rows = _read_predictions(predictions_path, manifest["splits"][split])
            strict = score_predictions(rows)
            upstream = score_predictions(rows, parser="upstream")
            if saved.get("strict") != strict or saved.get("upstream_parser") != upstream:
                raise ValueError(f"{metrics_path}: saved metrics do not reproduce from predictions")
            run["evaluations"][scenario] = {
                **expected_metadata,
                "elapsed_s": saved.get("elapsed_s"),
                "prediction_file": str(predictions_path.resolve()),
                "strict": strict,
                "upstream_parser": upstream,
            }
            if split not in references:
                references[split] = score_predictions(majority_predictions(rows, majority))
        for split in ("validation", "test"):
            if split in run["evaluations"] and f"{split}_zero" in run["evaluations"]:
                run["signal_ablation_gap"][split] = {}
                for group in ("all", "noisy", "clean"):
                    real = run["evaluations"][split]["strict"]["groups"][group]
                    zeros = run["evaluations"][f"{split}_zero"]["strict"]["groups"][group]
                    run["signal_ablation_gap"][split][group] = {
                        metric: real[metric] - zeros[metric] if real[metric] is not None else None
                        for metric in METRICS
                        if metric != "invalid_rate"
                    }
        runs.append(run)
    for seed in expected_seeds:
        if seed not in seen:
            issues.append(f"Seed {seed}: run unavailable")
    if not runs:
        issues.append("No completed training runs are available")
    if issues and not allow_partial:
        raise ValueError(
            "Baseline report is incomplete: " + "; ".join(issues) + ". Use --allow-partial for diagnostics."
        )
    ablation = {}
    for split in ("validation", "test"):
        ablation[split] = {}
        for group in ("all", "noisy", "clean"):
            ablation[split][group] = {
                metric: _stats(
                    [
                        run["signal_ablation_gap"][split][group][metric]
                        for run in runs
                        if split in run["signal_ablation_gap"]
                        and run["signal_ablation_gap"][split][group][metric] is not None
                    ]
                )
                for metric in METRICS
                if metric != "invalid_rate"
            }
    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete" if not issues else "incomplete",
        "issues": issues,
        "expected_seeds": list(expected_seeds),
        "available_seeds": [run["seed"] for run in runs],
        "manifest": manifest,
        "majority_answers": majority,
        "majority_reference": references,
        "paper_context": {
            **PAPER_REFERENCE,
            "directly_comparable": False,
            "reason": "Different patient split, training data/budget, curriculum, and parser; "
            "paper averaging is unspecified.",
        },
        "deviations": DEVIATIONS,
        "runs": runs,
        "aggregate": _aggregate(runs),
        "signal_ablation_gap": ablation,
        "uncertainty": "SD is sample standard deviation across independent seeds (ddof=1), "
        "not a confidence interval; one-seed SD is null.",
    }


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def _mean_sd(value: dict) -> str:
    if value["mean"] is None:
        return "unavailable"
    sd = "SD unavailable" if value["sd"] is None else f"SD {100 * value['sd']:.2f} pp"
    return f"{_percent(value['mean'])} ({sd}; n={value['n_seeds']})"


def render_report(report: dict) -> str:
    """Render an independently readable report, preserving unavailable results."""
    lines = ["# OpenTSLM ECG-QA-CoT RunPod baseline", "", f"Status: **{report['status']}**.", ""]
    if report["issues"]:
        lines += ["This is a partial diagnostic report. Outstanding items:", ""]
        lines += [f"- {issue}" for issue in report["issues"]]
        lines.append("")
    lines += [
        "Accuracy and F1 below are generated-answer scores. Primary F1 is macro over all declared "
        "answer options within each template, weighted by template question counts across templates. "
        "The unweighted template macro-F1 is also recorded. Invalid answers remain in every denominator.",
        "",
        "| Scenario | Slice | Accuracy mean ± SD | Weighted template macro-F1 mean ± SD | "
        "Invalid rate mean ± SD |",
        "|---|---|---|---|---|",
    ]
    for scenario in SCENARIOS:
        for group in ("all", "noisy", "clean"):
            item = report["aggregate"][scenario][group]
            lines.append(
                f"| {scenario} | {group} | {_mean_sd(item['accuracy'])} | "
                f"{_mean_sd(item['weighted_template_macro_f1'])} | {_mean_sd(item['invalid_rate'])} |"
            )
    lines += [
        "",
        "## Per-seed results",
        "",
        "| Seed | Scenario | Questions | Accuracy | Weighted F1 | Template macro-F1 | Invalid |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        for scenario, scores in run["evaluations"].items():
            item = scores["strict"]["groups"]["all"]
            lines.append(
                f"| {run['seed']} | {scenario} | {item['n']} | {_percent(item['accuracy'])} | "
                f"{_percent(item['weighted_template_macro_f1'])} | "
                f"{_percent(item['template_macro_f1'])} | {_percent(item['invalid_rate'])} |"
            )
    lines += [
        "",
        "## Training-pool majority and signal ablation",
        "",
        "The majority reference is fitted only on training questions, preserving training-row weighting. "
        "Ties use lexicographic order. It uses the same held-out questions and strict scoring.",
        "",
        "| Split | Slice | Majority accuracy | Majority weighted F1 | Real-minus-zero accuracy |",
        "|---|---|---:|---:|---|",
    ]
    for split, reference in report["majority_reference"].items():
        for group in ("all", "noisy", "clean"):
            item = reference["groups"][group]
            gap = report["signal_ablation_gap"][split][group]["accuracy"]
            lines.append(
                f"| {split} | {group} | {_percent(item['accuracy'])} | "
                f"{_percent(item['weighted_template_macro_f1'])} | {_mean_sd(gap)} |"
            )
    lines += [
        "",
        "## Dataset and provenance",
        "",
        f"Frozen export SHA-256: `{report['manifest']['sha256']}`.",
        "",
        "| Split | Questions | ECGs | Patients | Noisy questions |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, counts in report["manifest"]["splits"].items():
        lines.append(
            f"| {split} | {counts['questions']} | {counts.get('records', 'unavailable')} | "
            f"{counts.get('patients', 'unavailable')} | {counts.get('noisy_questions', 'unavailable')} |"
        )
    for run in report["runs"]:
        config, trained = run["config"], run["training"]
        lines += [
            "",
            f"Seed {run['seed']}: best step {trained['best_step']} of {trained['max_steps']}; "
            f"selection accuracy {_percent(trained.get('best_selection_accuracy'))}; "
            f"selection method `{config.get('selection_method', 'unavailable')}`. "
            f"Checkpoint SHA-256 `{trained['best_checkpoint_sha256']}`.",
            f"Model `{config.get('model', 'unavailable')}` "
            f"at `{config.get('base_revision', 'unavailable')}`; "
            f"OpenTSLM `{config.get('opentslm_commit', 'unavailable')}`; "
            f"prompt `{config.get('prompt_version', 'unavailable')}`.",
            f"Maximum sequence length {config.get('max_length', 'unavailable')}; "
            f"generation cap {config.get('max_new_tokens', 'unavailable')} tokens; "
            f"distinct truncated training rationales "
            f"{trained.get('distinct_truncated_rationales', 'unavailable')}. "
            f"GPU `{config.get('gpu', 'unavailable')}`; "
            f"training wall time {trained.get('wall_seconds', 'unavailable')} seconds; "
            f"peak allocated memory {trained.get('peak_memory_gb', 'unavailable')} GB.",
        ]
    lines += [
        "",
        "Each prediction file checksum, full split coverage, selected checkpoint, and dataset "
        "identity were verified; all saved metrics were recomputed. The JSON contains complete settings, "
        "noise-slice results, per-template/per-class counts, and secondary upstream-parser scores.",
        "",
        "## Paper context and deviations",
        "",
        "The supplied OpenTSLM paper, Table 2 (page 6), reports Gemma3-270M Flamingo F1 **32.71%** "
        "and accuracy **35.50%**. These are contextual references, not directly comparable targets for "
        "this patient split and training protocol.",
        "",
    ]
    lines += [f"- {deviation}" for deviation in report["deviations"]]
    lines += ["", report["uncertainty"], "", "Artifacts: `summary.json` and `per_template.csv`.", ""]
    return "\n".join(lines)


def write_report(report: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "report.md").write_text(render_report(report), encoding="utf-8")
    fields = ["seed", "scenario", "slice", "template_id", "n", "accuracy", "macro_f1", "invalid_rate"]
    with (output / "per_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        sources = [
            (run["seed"], scenario, scores["strict"])
            for run in report["runs"]
            for scenario, scores in run["evaluations"].items()
        ]
        sources += [("majority", split, scores) for split, scores in report["majority_reference"].items()]
        for seed, scenario, scores in sources:
            for group, values in scores["groups"].items():
                for template, value in values["per_template"].items():
                    writer.writerow(
                        {
                            "seed": seed,
                            "scenario": scenario,
                            "slice": group,
                            "template_id": template,
                            **{key: value[key] for key in ("n", "accuracy", "macro_f1", "invalid_rate")},
                        }
                    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", "--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not repeat a seed")
    manifest = _read_json(args.data_dir / "manifest.json")
    if _json_hash({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
        raise ValueError("Dataset export manifest checksum mismatch")
    train_path = args.data_dir / "train.parquet"
    if _file_hash(train_path) != manifest["files"]["train.parquet"]["sha256"]:
        raise ValueError("Training data checksum mismatch; refusing to fit majority reference")
    import pyarrow.parquet as pq

    majority = fit_majority_answers(pq.read_table(train_path).to_pylist())
    report = build_report(
        args.run_dir, manifest, majority, expected_seeds=tuple(args.seeds), allow_partial=args.allow_partial
    )
    write_report(report, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "available_seeds": report["available_seeds"],
                "output": str(args.output.resolve()),
                "issues": report["issues"],
            }
        )
    )


if __name__ == "__main__":
    main()
