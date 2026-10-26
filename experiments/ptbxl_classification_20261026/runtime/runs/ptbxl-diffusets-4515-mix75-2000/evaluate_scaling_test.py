"""Evaluate the frozen scaling checkpoints and export the held-out figure data.

Run with the project .venv Python. Existing training runs stay unchanged. Only
the four missing test evaluations are inferred; exact equivalent checkpoints
provide the existing 50% and 100% scores. Every score is checked against saved
predictions and the common fold-10 labels before plotting.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from syntide.data.schema import SUPERCLASSES
from syntide.eval.holdout import FrozenRun, claim, frozen_runs, load_test, score_run
from syntide.eval.stats import val_scores
from syntide.trial import Checkpoint, load_spec

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parents[1]
OUT = ROOT / "heldout_scaling_evaluation"
FIGURES = ROOT / "figures"
SHARES = (10, 20, 30, 50, 70, 100)
REUSE = {
    50: ENGINE / "runs/ptbxl-artifact-50-val100/scarcity50/C2/seed0",
    100: ENGINE / "runs/ptbxl-artifact-50-val100/full100/C2/seed0",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def frozen_equivalent(path):
    spec = load_spec(path)
    result = read(path / "result.json")
    assert result["status"] == "complete" and result["trial_id"] == spec.trial_id
    best = result["provenance"]["checkpoints"]["best"]
    checkpoint = path / best["path"]
    assert sha(checkpoint) == best["sha256"]
    return FrozenRun(path, spec, result, Checkpoint(checkpoint, best["step"]), best["sha256"])


def check_equivalence(original, reused):
    a = torch.load(original.best.path, weights_only=True, map_location="cpu")
    b = torch.load(reused.best.path, weights_only=True, map_location="cpu")
    assert a["model_config"] == b["model_config"]
    assert a["model"].keys() == b["model"].keys()
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["step"] == b["step"] == original.best.step == reused.best.step
    assert original.result["trainer"] == reused.result["trainer"]
    ma, mb = read(Path(original.spec.manifest)), read(Path(reused.spec.manifest))
    assert ma["source"] == mb["source"] and ma["roles"] == mb["roles"]
    return {
        "scaling_checkpoint": str(original.best.path),
        "reused_checkpoint": str(reused.best.path),
        "scaling_checkpoint_sha256": original.best_sha256,
        "reused_checkpoint_sha256": reused.best_sha256,
        "all_model_tensors_exactly_equal": True,
        "model_tensors_compared": len(a["model"]),
        "model_config_trainer_manifest_source_and_all_roles_equal": True,
        "step": original.best.step,
    }


def verify_predictions(run, test):
    scores_path = run.run_dir / "test_scores.json"
    predictions_path = run.run_dir / "predictions_test.parquet"
    scores = read(scores_path)
    assert scores["trial_id"] == run.spec.trial_id
    assert scores["step"] == run.best.step and scores["checkpoint"] == "best"
    assert scores["n_test"] == len(test) and scores["n_test_artifact"] == test.is_artifact.sum()
    table = pq.read_table(predictions_path).to_pydict()
    assert len(table["record_id"]) == test.labels.size
    assert set(table["trial_id"]) == {run.spec.trial_id}
    assert set(table["step"]) == {run.best.step} and set(table["checkpoint"]) == {"best"}
    record_index = {r: i for i, r in enumerate(test.record_id)}
    class_index = {c: i for i, c in enumerate(SUPERCLASSES)}
    is_artifact = test.is_artifact
    probabilities = np.full(test.labels.shape, np.nan, dtype=np.float64)
    for row in zip(table["record_id"], table["class_id"], table["y_true"],
                   table["y_prob"], table["subject_uid"], table["is_artifact"], strict=True):
        record, cls, label, probability, patient, artifact = row
        i, j = record_index[record], class_index[cls]
        assert np.isnan(probabilities[i, j]), "Duplicate record/class prediction"
        assert label == test.labels[i, j] and patient == test.subject_uid[i]
        assert artifact == is_artifact[i]
        probabilities[i, j] = probability
    recomputed = {k.replace("val/", "test/", 1): v for k, v in val_scores(test, probabilities).items()}
    for key, value in recomputed.items():
        assert np.isclose(value, scores[key], rtol=0, atol=1e-12), (key, value, scores[key])
    return scores, {
        "scores_path": str(scores_path), "scores_sha256": sha(scores_path),
        "predictions_path": str(predictions_path), "predictions_sha256": sha(predictions_path),
        "all_prediction_records_labels_patients_and_artifact_flags_verified": True,
        "all_saved_metrics_recomputed_within_1e-12": True,
    }


def main():
    os.chdir(ENGINE)
    OUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    scaling = frozen_runs(ENGINE / "runs/ptbxl-scaling")
    by_share = {int(r.spec.regime.removeprefix("pool")): r for r in scaling}
    originals = [by_share[share] for share in SHARES] + frozen_runs(ROOT)
    assert len(originals) == 7
    assert len({r.result["trainer"]["fingerprint"] for r in originals}) == 1
    destinations, score_sources, records, equivalences = [], [], [], []
    for run in originals:
        share = int(run.spec.regime.removeprefix("pool"))
        mixed = run.spec.arm == "mix75"
        chosen = max(run.result["val_history"], key=lambda row: row["val/auroc_overall"])
        assert chosen["step"] == run.best.step == run.result["best_step"]
        target = OUT / ("diffusets50" if mixed else f"real{share}") / "seed0"
        target.mkdir(parents=True, exist_ok=True)
        destination = replace(run, run_dir=target)
        source = run if mixed else destination
        mode = "existing_original_test" if mixed else "new_frozen_checkpoint_evaluation"
        if not mixed and share in REUSE:
            source = frozen_equivalent(REUSE[share])
            equivalences.append(check_equivalence(run, source))
            mode = "existing_exactly_equivalent_checkpoint_test"
        destinations.append(destination)
        score_sources.append(source)
        records.append({
            "trial_id": run.spec.trial_id, "original_run": str(run.run_dir),
            "checkpoint_path": str(run.best.path), "checkpoint_sha256": run.best_sha256,
            "checkpoint_step": run.best.step, "selection_metric": "val/auroc_overall",
            "manifest": run.spec.manifest, "manifest_sha256": run.spec.manifest_sha256,
            "source_sha256": run.spec.source_sha256, "score_source": str(source.run_dir),
            "score_source_trial_id": source.spec.trial_id, "mode": mode,
        })
    freeze = {"split": "test", "fold": 10, "selection": "Best overall validation AUROC; no test selection",
              "trials": records, "checkpoint_equivalences": equivalences}
    frozen_path = OUT / "selected_checkpoints.json"
    if frozen_path.exists():
        assert read(frozen_path) == freeze, "Previously frozen comparison changed"
    else:
        write(frozen_path, freeze)
    claim(OUT, destinations)
    print("Frozen seven validation-selected checkpoints; loading common test fold.", flush=True)
    test = load_test(originals + [s for s in score_sources if s.spec.arm == "C2"])
    assert len(test) == 2158 and int(test.is_artifact.sum()) == 513
    rows, audits = [], []
    for original, source, record in zip(originals, score_sources, records, strict=True):
        if not (source.run_dir / "test_scores.json").exists():
            print(f"Evaluating {original.spec.regime} at step {original.best.step} ...", flush=True)
            score_run(source, test)
        scores, audit = verify_predictions(source, test)
        share = int(original.spec.regime.removeprefix("pool"))
        mixed = original.spec.arm == "mix75"
        rows.append({
            "scenario": "diffusets" if mixed else "real_scaling",
            "real_patient_share_percent": share,
            "real_records": len(read(Path(original.spec.manifest))["roles"]["pool"]),
            "generated_records": 4515 if mixed else 0, "seed": original.spec.replicate_seed,
            "training_steps_completed": original.result["final_step"], "best_step": original.best.step,
            **{f"test_auroc_{part}": scores[f"test/auroc_{part}"] for part in ("overall", "artifact", "clean")},
            "test_loss": scores["test/loss"], "n_test": len(test), "n_test_artifact": int(test.is_artifact.sum()),
            "source": str(original.run_dir), "test_scores_source": str(source.run_dir / "test_scores.json"),
        })
        audits.append({**record, **audit})
        print(f"{rows[-1]['scenario']} {share}%: {scores['test/auroc_overall']:.8f}", flush=True)
    with (FIGURES / "test_saturation_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write(OUT / "evaluation_audit.json", {
        "status": "complete", "n_test": len(test), "n_noisy": int(test.is_artifact.sum()),
        "n_clean": int((~test.is_artifact).sum()), "source_sha256": originals[0].spec.source_sha256,
        "frozen_checkpoint_selection": str(frozen_path), "test_claim": str(OUT / "test_claim.json"),
        "metrics_verified_from_predictions": True, "trials": audits,
        "evaluation_script_sha256": sha(Path(__file__)),
        "note": "Exploratory reporting: this held-out partition was also viewed in prior experiments. No test history or test-selected checkpoint is reported.",
    })


if __name__ == "__main__":
    main()
