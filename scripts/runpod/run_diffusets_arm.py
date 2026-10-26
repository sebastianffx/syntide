"""Train and evaluate the additional DiffuSETS question-answering experiment.

The saved protocol identifies the prepared data, original baseline, candidate
plan, checkpoint, and model configuration by hash. It also records the
evaluation time budget and training corpus order. Existing baseline outputs stay
unchanged.

An operational override may extend the available runtime only after its hash and
original protocol match. It preserves the checks against repeated training or
evaluation and writes a separate audit record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import importlib.util
import os
import re
import sys
import time
from pathlib import Path

ARM = "grounded_acq_gen"
STUDY_KIND = "selection_tuned_diffusets_extension"


def helper(name):
    spec = importlib.util.spec_from_file_location(
        f"_diffusets_{name}", Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = helper("run_qa_arms")
read, write, sha, freeze_once = (
    shared.read,
    shared.write,
    shared.sha,
    shared.freeze_once,
)


def validate_protocol(protocol):
    if protocol.get("study_kind") != STUDY_KIND or set(
        protocol.get("arms", {})
    ) != {ARM}:
        raise ValueError(
            "Protocol must declare the separate selection-tuned "
            "grounded_acq_gen extension"
        )
    if protocol.get("seeds") != [0] or protocol.get("candidate_count") != 5000:
        raise ValueError(
            "Extension requires seed 0 and exactly 5000 generation candidates"
        )
    composition = protocol.get("training_corpora", {ARM: []})
    if (
        not isinstance(composition, dict)
        or set(composition) != {ARM}
        or not isinstance(composition[ARM], list)
    ):
        raise ValueError(
            "training_corpora must map grounded_acq_gen to its "
            "component-name list"
        )
    batch = protocol.get("generation_batch_size")
    if type(batch) is not int or batch < 1:
        raise ValueError("generation_batch_size must be a positive integer")
    budget = protocol["evaluation_budget"]
    deadline = dt.datetime.fromisoformat(
        budget["deadline_utc"].replace("Z", "+00:00")
    )
    if deadline.tzinfo is None:
        raise ValueError("Evaluation deadline must include a timezone")
    for field, minimum in (
        ("seconds_per_question", 0.000001),
        ("safety_factor", 1),
        ("seconds_per_pass_overhead", 0),
        ("final_reserve_seconds", 0),
    ):
        value = budget.get(field)
        if type(value) not in (int, float) or not minimum <= value < float(
            "inf"
        ):
            raise ValueError(f"Invalid evaluation time budget: {field}")
    baseline = protocol["baseline"]
    for digest in (
        protocol.get("candidate_plan_sha256"),
        *(
            baseline.get(k)
            for k in ("config_sha256", "checkpoint_sha256", "manifest_sha256")
        ),
    ):
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(
                "Protocol must pin valid baseline and candidate-plan "
                "SHA256 fingerprints"
            )
    root = Path(protocol["run_dir"]).resolve()
    original = Path(baseline["original_study_dir"]).resolve()
    reference = Path(baseline["run_dir"]).resolve()
    if not reference.is_relative_to(original):
        raise ValueError(
            "Baseline reference must be inside its declared original "
            "study directory"
        )
    for protected in (
        original,
        Path(baseline["prepared"]).resolve(),
        Path(protocol["arms"][ARM]["prepared"]).resolve(),
    ):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError(
                "Extension run directory must be separate from the "
                "original study and prepared data"
            )
    return root


def verify_baseline(protocol):
    from syntide.qa_baseline_audit import audit_config, audit_run

    ref = protocol["baseline"]
    if (
        protocol.get("candidate_plan_path")
        and sha(protocol["candidate_plan_path"])
        != protocol["candidate_plan_sha256"]
    ):
        raise ValueError("Frozen candidate plan file differs from protocol")
    directory = Path(ref["run_dir"])
    config = read(directory / "config.json")
    if (
        sha(directory / "config.json") != ref["config_sha256"]
        or config["manifest_sha256"] != ref["manifest_sha256"]
    ):
        raise ValueError(
            "Baseline reference config or manifest fingerprint changed"
        )
    audit_config(config, config)
    shared.verify_runtime_sources(config)
    audit_run(
        config,
        config,
        read(directory / "train_result.json"),
        read(directory / "selection_history.json"),
        checkpoint_path=directory / "best.pt",
        frozen_checkpoint_sha256=ref["checkpoint_sha256"],
    )
    return config


def verify_data(protocol, baseline):
    from syntide.baseline.data import verify_prepared

    base = verify_prepared(
        protocol["baseline"]["prepared"], splits=("train", "selection")
    )
    new = verify_prepared(
        protocol["arms"][ARM]["prepared"], splits=("train", "selection")
    )
    for key, expected in baseline["data_files"].items():
        if base["files"][key]["sha256"] != expected:
            raise ValueError(f"Original baseline prepared file differs: {key}")
    if base["sha256"] != baseline["manifest_sha256"]:
        raise ValueError("Original baseline prepared manifest differs")
    for split in ("selection", "validation", "test"):
        if (
            new["files"][split + ".parquet"]
            != base["files"][split + ".parquet"]
        ):
            raise ValueError(f"Extension {split} bytes changed")
    return new


def preflight(protocol, output):
    root = validate_protocol(protocol)
    root.mkdir(parents=True, exist_ok=True)
    freeze_once(root / "protocol.json", protocol)
    baseline = verify_baseline(protocol)
    verify_data(protocol, baseline)
    module = helper("qa_preflight")
    # Private module instance in this CLI process; the original preflight source
    # is unchanged.
    module.ARMS = ("real_qa", ARM)
    inputs = {
        "seeds": [0],
        "arms": {
            "real_qa": {"prepared": protocol["baseline"]["prepared"]},
            ARM: protocol["arms"][ARM],
        },
    }
    report = module.preflight(inputs, baseline, output)
    report.update(
        extension_protocol_sha256=sha(root / "protocol.json"),
        baseline_config_sha256=protocol["baseline"]["config_sha256"],
        candidate_plan_sha256=protocol["candidate_plan_sha256"],
    )
    write(output, report)
    return report


def check_preflight(protocol, baseline, manifest):
    from syntide.qa_baseline_audit import audit_config

    root = Path(protocol["run_dir"])
    report = read(root / "preflight.json")
    if (
        report.get("status") != "passed"
        or report.get("extension_protocol_sha256")
        != sha(root / "protocol.json")
        or report.get("baseline_config_sha256")
        != protocol["baseline"]["config_sha256"]
        or report.get("candidate_plan_sha256")
        != protocol["candidate_plan_sha256"]
    ):
        raise ValueError(
            "A passed preflight bound to this exact extension protocol "
            "is required"
        )
    if set(report.get("arms", {})) != {"real_qa", ARM}:
        raise ValueError(
            "Preflight must scan exactly the baseline and extension "
            "training data"
        )
    for arm in ("real_qa", ARM):
        item = report["arms"][arm]
        audit_config(baseline, item["expected_config"])
        scan = item["full_training_token_scan"]
        if (
            scan.get("truncated_rationales") != 0
            or scan.get("questions", 0) <= 0
        ):
            raise ValueError(
                f"{arm}: preflight did not verify zero truncation for "
                f"training rows"
            )
    original = report["arms"]["real_qa"]["expected_config"]
    if (
        original["manifest_sha256"] != baseline["manifest_sha256"]
        or original["data_files"] != baseline["data_files"]
    ):
        raise ValueError(
            "Preflight baseline dataset differs from original baseline"
        )
    if (
        report["arms"][ARM]["full_training_token_scan"]["questions"]
        != manifest["splits"]["train"]["questions"]
    ):
        raise ValueError("Preflight did not scan every extension training row")
    expected = report["arms"][ARM]["expected_config"]
    if expected["manifest_sha256"] != manifest["sha256"] or any(
        manifest["files"][name]["sha256"] != digest
        for name, digest in expected["data_files"].items()
    ):
        raise ValueError("Prepared extension data changed after preflight")
    return report


def evaluation_budget(
    protocol, run, manifest, *, now=None, operational_override=None
):
    """Estimate all remaining complete passes without opening held-out rows.

    The initial A40 rate is a planning assumption, not a validated benchmark.
    Completed validation timings can increase it; test scores never tune it.
    """
    budget = protocol["evaluation_budget"]
    deadline = (
        operational_override.timestamp()
        if operational_override
        else dt.datetime.fromisoformat(
            budget["deadline_utc"].replace("Z", "+00:00")
        ).timestamp()
    )
    rates = {"initial_planning_assumption": budget["seconds_per_question"]}
    for label in ("validation", "validation_zero"):
        path = run / f"{label}_metrics.json"
        if path.exists():
            metric = read(path)
            if (
                metric.get("questions")
                != manifest["splits"]["validation"]["questions"]
            ):
                raise ValueError(
                    "Completed validation timing has an unexpected "
                    "question count"
                )
            rates[label] = metric["elapsed_s"] / metric["questions"]
            if not 0 < rates[label] < float("inf"):
                raise ValueError("Completed validation timing is invalid")
    pending = {
        split + ("_zero" if zero else ""): manifest["splits"][split][
            "questions"
        ]
        for split in ("validation", "test")
        for zero in (False, True)
        if not (
            run / f"{split}{'_zero' if zero else ''}_predictions.jsonl"
        ).exists()
    }
    seconds_per_question = max(rates.values()) * budget["safety_factor"]
    estimate = (
        sum(pending.values()) * seconds_per_question
        + len(pending) * budget["seconds_per_pass_overhead"]
        + budget["final_reserve_seconds"]
    )
    remaining = deadline - (time.time() if now is None else now)
    return {
        "status": "sufficient_estimated_time"
        if remaining >= estimate
        else "needs_time_budget_review",
        "deadline_utc": dt.datetime.fromtimestamp(deadline, dt.UTC).isoformat(),
        "scientific_deadline_utc": budget["deadline_utc"],
        **(
            {"operational_override": operational_override.audit()}
            if operational_override
            else {}
        ),
        "available_seconds": remaining,
        "required_seconds_estimate": estimate,
        "pending_pass_questions": pending,
        "rate_inputs_seconds_per_question": rates,
        "safety_factor": budget["safety_factor"],
        "initial_rate_is_measured": False,
        "limitation": (
            "Planning estimate with margin; runtime "
            "variability can exceed the estimate."
        ),
    }


def evaluate(
    protocol,
    run,
    lock,
    checkpoint_sha256,
    manifest,
    *,
    operational_override=None,
):
    root = Path(protocol["run_dir"])
    for split in ("validation", "test"):
        for zero in (False, True):
            label = split + ("_zero" if zero else "")
            claim = run / f"{label}_claim.json"
            predictions = run / f"{label}_predictions.jsonl"
            if sha(run / "best.pt") != checkpoint_sha256:
                raise ValueError("Extension checkpoint changed after freeze")
            if (
                claim.exists()
                and read(claim).get("batch_size")
                != protocol["generation_batch_size"]
            ):
                raise ValueError(
                    f"Existing {label} generation batch size differs "
                    f"from protocol"
                )
            if split == "test" and claim.exists() and not predictions.exists():
                raise ValueError(
                    f"Interrupted {label} claim: preserving artifacts "
                    f"and refusing repeated test generation"
                )
            if not claim.exists() and any(run.glob(f"{label}_predictions*")):
                raise ValueError(
                    f"Existing {label} predictions have no provenance claim"
                )
            if not predictions.exists():
                budget = evaluation_budget(
                    protocol,
                    run,
                    manifest,
                    operational_override=operational_override,
                )
                write(root / "evaluation_budget.json", budget)
                if budget["status"] != "sufficient_estimated_time":
                    write(
                        root / "status.json",
                        {
                            "status": "needs_time_budget_review",
                            "phase": label,
                            "arm": ARM,
                            "checkpoint_retained": True,
                            "test_opened": any(run.glob("test*_claim.json")),
                            "updated_unix": time.time(),
                            "budget": budget,
                        },
                    )
                    return False
            write(
                root / "status.json",
                {
                    "status": "running",
                    "phase": label,
                    "arm": ARM,
                    "updated_unix": time.time(),
                },
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "syntide_trainers.opentslm_loop",
                "evaluate",
                "--prepared",
                protocol["arms"][ARM]["prepared"],
                "--run-dir",
                str(run),
                "--split",
                split,
                "--batch-size",
                str(protocol["generation_batch_size"]),
            ]
            if zero:
                command.append("--zero-signal")
            # Published predictions are validated/rescored by the unchanged
            # evaluator, never regenerated.
            shared.execute(command, run.parent / f"{label}.log", lock)
    return True


def write_report(root, summary):
    lines = [
        "# Selection-tuned DiffuSETS extension, seed 0",
        "",
        "This exploratory extension was allocated using TRAIN and "
        "selection evidence after the original "
        "study began. It is reported separately; one seed provides no "
        "across-seed standard deviation.",
        "",
        f"Best selection accuracy: "
        f"{summary['training']['best_selection_accuracy']:.2%} "
        f"at step {summary['training']['best_step']:,}; distinct "
        f"training questions seen: "
        f"{summary['training']['distinct_seen_training_questions']:,}.",
        "",
        "| Split | Slice | Real accuracy | Zero ECG accuracy | Real "
        "template macro-F1 | Real invalid rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split, key in (
        ("validation", "full_validation_generation"),
        ("test", "test_generation"),
    ):
        for group in ("all", "clean", "noisy"):
            real = summary[key]["strict"]["groups"][group]
            zero = summary["waveform_ablation"][split]["strict"]["groups"][
                group
            ]
            lines.append(
                f"| {split} | {group} | {real['accuracy']:.2%} | "
                f"{zero['accuracy']:.2%} | "
                f"{real['template_macro_f1']:.2%} | "
                f"{real['invalid_rate']:.2%} |"
            )
    (root / "report.md").write_text("\n".join(lines) + "\n")


def run_study(protocol, lock, *, operational_override=None):
    import pyarrow.parquet as pq
    from syntide.baseline.metrics import (
        fit_majority_answers,
        majority_predictions,
        score_predictions,
    )

    from syntide.qa_baseline_audit import audit_config, audit_run

    root = validate_protocol(protocol)
    root.mkdir(parents=True, exist_ok=True)
    helper("run_diffusets_pipeline").record_override(root, operational_override)
    freeze_once(root / "protocol.json", protocol)
    baseline = verify_baseline(protocol)
    shared.verify_holdout_state(root)
    manifest = verify_data(protocol, baseline)
    check_preflight(protocol, baseline, manifest)
    prepared = Path(protocol["arms"][ARM]["prepared"])
    original = Path(protocol["baseline"]["prepared"])
    majority = fit_majority_answers(
        pq.read_table(original / "train.parquet").to_pylist()
    )
    selection = pq.read_table(original / "selection.parquet").to_pylist()
    freeze_once(
        root / "majority_selection.json",
        score_predictions(majority_predictions(selection, majority)),
    )
    run = root / ARM / "seed0"
    run.parent.mkdir(parents=True, exist_ok=True)
    if not (run / "train_result.json").exists():
        if (root / "freeze.json").exists() or (
            root / "holdout_claim.json"
        ).exists():
            raise ValueError(
                "Cannot start training after extension checkpoint "
                "freeze or holdout claim"
            )
        if run.exists() and any(run.iterdir()):
            raise ValueError(
                "Interrupted extension training requires explicit "
                "recovery; preserving files"
            )
        write(
            root / "status.json",
            {
                "status": "running",
                "phase": "training",
                "arm": ARM,
                "updated_unix": time.time(),
            },
        )
        shared.execute(
            shared.train_command(prepared, run, baseline),
            run.parent / "train.log",
            lock,
            baseline=baseline,
            config_path=run / "config.json",
        )
    config, result = read(run / "config.json"), read(run / "train_result.json")
    write(run.parent / "config_comparison.json", audit_config(baseline, config))
    if config["manifest_sha256"] != manifest["sha256"] or any(
        manifest["files"][name]["sha256"] != digest
        for name, digest in config["data_files"].items()
    ):
        raise ValueError(
            "Extension training dataset fingerprint differs from prepared data"
        )
    audit = audit_run(
        baseline,
        config,
        result,
        read(run / "selection_history.json"),
        checkpoint_path=run / "best.pt",
        require_convergence=False,
    )
    write(run.parent / "run_audit.json", audit)
    shared.make_summary(root, ARM, majority, complete=False)
    if not audit["test_eligible"]:
        write(
            root / "status.json",
            {
                "status": "needs_budget_review",
                "arm": ARM,
                "test_opened": False,
                "updated_unix": time.time(),
            },
        )
        return
    shared.verify_runtime_sources(baseline)
    frozen = {
        "checkpoint_sha256": result["best_checkpoint_sha256"],
        "config_sha256": sha(run / "config.json"),
        "manifest_sha256": manifest["sha256"],
        "best_step": result["best_step"],
    }
    freeze_once(
        root / "freeze.json",
        {
            "study_kind": STUDY_KIND,
            "protocol_sha256": sha(root / "protocol.json"),
            "preflight_sha256": sha(root / "preflight.json"),
            "candidate_plan_sha256": protocol["candidate_plan_sha256"],
            "baseline_reference": protocol["baseline"],
            "checkpoints": {ARM: frozen},
            "seeds": [0],
            "rule": "complete 15531 updates, identical baseline settings and "
            "initialization, "
            "zero truncation, first argmax before 80%; prospective "
            "extension checkpoint freeze",
        },
    )
    freeze_once(
        root / "holdout_claim.json",
        {
            "freeze_sha256": sha(root / "freeze.json"),
            "evaluations": "one real and one zero-waveform pass per split for "
            "grounded_acq_gen only",
        },
    )
    if not evaluate(
        protocol,
        run,
        lock,
        frozen["checkpoint_sha256"],
        manifest,
        operational_override=operational_override,
    ):
        return
    summary = shared.make_summary(root, ARM, majority, complete=True)
    summary.update(
        study_kind=STUDY_KIND,
        candidate_plan_sha256=protocol["candidate_plan_sha256"],
        interpretation=(
            "Exploratory extension allocated using "
            "selection results; separate from the "
            "original confirmatory study."
        ),
    )
    write(run.parent / "summary.json", summary)
    write(
        root / "extension_result.json",
        {
            "study_kind": STUDY_KIND,
            "seeds": [0],
            "seed_sd": None,
            "arms": {ARM: summary},
            "freeze_sha256": sha(root / "freeze.json"),
        },
    )
    write_report(root, summary)
    write(
        root / "status.json",
        {"status": "complete", "updated_unix": time.time()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--protocol", required=True, type=Path)
    operational = helper("run_diffusets_pipeline")
    parser.add_argument(
        "--operational-override",
        default=os.environ.get(operational.OVERRIDE_PATH_ENV),
    )
    parser.add_argument(
        "--operational-override-sha256",
        default=os.environ.get(operational.OVERRIDE_SHA_ENV),
    )
    args = parser.parse_args()
    protocol = read(args.protocol)
    override = operational.deadline_override(
        args.operational_override,
        args.operational_override_sha256,
        protocol=protocol,
    )
    root = validate_protocol(protocol)
    root.mkdir(parents=True, exist_ok=True)
    operational.record_override(root, override)
    with (root / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.action == "preflight":
                if (root / "freeze.json").exists() or (
                    root / ARM / "seed0"
                ).exists():
                    raise ValueError(
                        "Preflight must run before extension training "
                        "or checkpoint freeze"
                    )
                preflight(protocol, root / "preflight.json")
            else:
                run_study(protocol, lock, operational_override=override)
        except BaseException as error:
            write(
                root / "status.json",
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "updated_unix": time.time(),
                },
            )
            raise


if __name__ == "__main__":
    main()
