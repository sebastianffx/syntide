#!/usr/bin/env python3
"""Freeze all accepted DiffuSETS ECGs with explicitly provisional diagnosis targets.

The generation acceptance policy measures rhythm and technical quality, not diagnostic
superclasses. The conditioning parent's PTB-XL target is the available weak label that
corresponds to the exact diagnostic report used by the generator. Original catalogs
remain untouched, including their pending diagnostic-review status.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from syntide.data.adapter import PtbXlAdapter
from syntide.data.challenge import load_manifest
from syntide.data.leakage import check_lineage, check_manifest
from syntide.data.schema import ARTIFACT_TYPES, LEADS, SUPERCLASSES

RUN = Path(__file__).resolve().parent
PROJECT = RUN.parents[1]
GENERATION = PROJECT / "outputs/qa-runs/diffusets-extension/study/generation"
MANIFEST = PROJECT / "manifests/ptbxl-scaling/pool50.json"
LABEL_POLICY = "provisional_conditioning_parent_superclasses"
GENERATION_ID = "bb68a35b6430053d2c9d2993a0c5ab45c2f6b775a78c53f773674fd98f1b522e"
SUMMARY_SHA256 = "e7dc8e22829eb40d088801bc33093f5a9ceb48d71f5631a1354bcd0e88dd3c38"
EXPECTED_ARTIFACTS = {
    "acceptance.json", "candidate.json", "catalog.jsonl", "ledger.jsonl",
    "run.json", "signals/00000000.npy",
}


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def freeze_json(path, value):
    if path.exists():
        require(read(path) == value, f"Existing frozen file differs: {path}")
    else:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


def freeze_npz(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            require(set(old.files) == set(arrays), "Existing array fields differ")
            for name, values in arrays.items():
                require(np.array_equal(old[name], values), f"Existing array differs: {name}")
        return
    partial = path.with_suffix(".partial")
    with partial.open("xb") as stream:
        np.savez(stream, **arrays)
    os.rename(partial, path)


def main():
    require(sha(GENERATION / "summary.json") == SUMMARY_SHA256, "Full generation summary differs")
    plan = read(GENERATION / "candidate-plan.json")
    identity = read(GENERATION / "identity.json")
    summary = read(GENERATION / "summary.json")
    require(identity["generation_id"] == summary["generation_id"] == GENERATION_ID,
            "Generation identity differs")
    require(summary["candidate_count"] == 5000 and summary["accepted"] == 4515
            and summary["rejected"] == 485, "Full generation counts differ")
    require(len(plan["candidates"]) == 5000, "Plan count differs")
    require([row["slot_id"] for row in summary["slots"]] == list(range(5000)),
            "Summary slot ordering differs")

    adapter = PtbXlAdapter(PROJECT / "datasets/timenet-registry")
    manifest = load_manifest(MANIFEST)
    index, source = adapter.index(), adapter.source()
    check_manifest(manifest, index, source)
    index_rows = {row["record_id"]: row for row in index.to_pylist()}
    pool = set(manifest.roles["pool"])
    require(len(pool) == 8579, "Expected all 8,579 real pool50 records")
    pool_subjects = {index_rows[r]["subject_uid"] for r in pool}
    heldout = set(manifest.roles["val"]) | set(manifest.roles["test"])
    heldout_subjects = {index_rows[r]["subject_uid"] for r in heldout}

    signals, rows, ancestry = [], [], []
    verified_files = 0
    for slot_item in summary["slots"]:
        slot = slot_item["slot_id"]
        folder = GENERATION / "candidates" / f"{slot:05d}"
        require(sha(folder / "result.json") == slot_item["result_sha256"], f"Result hash: {slot}")
        result = read(folder / "result.json")
        candidate = plan["candidates"][slot]
        require(candidate["slot_id"] == result["slot_id"] == slot, f"Slot identity: {slot}")
        require(result["generation_id"] == GENERATION_ID, f"Generation identity: {slot}")
        require(result["candidate_sha256"] == digest(candidate), f"Candidate identity: {slot}")
        require(type(result["accepted"]) is bool
                and result["accepted"] == (result["status"] == "verified"), f"Acceptance: {slot}")
        if not result["accepted"]:
            continue
        require(set(result["artifacts"]) == EXPECTED_ARTIFACTS, f"Bundle files: {slot}")
        for relative, expected in result["artifacts"].items():
            require(sha(folder / relative) == expected, f"Bundle hash: {slot}/{relative}")
            verified_files += 1
        require(read(folder / "candidate.json") == candidate, f"Frozen plan mismatch: {slot}")
        ledger = read(folder / "ledger.jsonl")
        catalog = read(folder / "catalog.jsonl")
        acceptance = read(folder / "acceptance.json")
        require(acceptance["status"] == "verified"
                and acceptance["clinical_validation"] == "not_performed", f"Review status: {slot}")
        require(catalog["labels"] == [] and catalog["metadata"]["label_status"] == "pending_review",
                f"Unexpected diagnostic labels: {slot}")
        require([lead.upper() for lead in catalog["channels"]] == list(LEADS)
                and catalog["sample_rate"] == 100 and catalog["units"] == "mV", f"Signal format: {slot}")
        require(catalog["signal_path"] == ledger["signal_path"] == "signals/00000000.npy",
                f"Signal path: {slot}")
        require(ledger["signal_sha256"] == result["artifacts"][ledger["signal_path"]], f"Signal ledger: {slot}")
        require(catalog["metadata"]["generation"] == ledger, f"Catalog ledger: {slot}")
        signal = np.load(folder / ledger["signal_path"], allow_pickle=False)
        require(signal.shape == (12, 1000) and signal.dtype == np.float32
                and np.isfinite(signal).all(), f"Waveform shape/values: {slot}")
        parent_id = candidate["source_record_id"]
        parent = index_rows[parent_id]
        require(ledger["source_record_ids"] == [parent_id]
                and parent["subject_uid"] == candidate["subject_uid"], f"Parent lineage: {slot}")
        require(parent["strat_fold"] in range(1, 9) and parent_id not in heldout
                and parent["subject_uid"] not in heldout_subjects, f"Heldout leakage: {slot}")
        require(bool(parent["labels"]), f"Unlabelled conditioning parent: {slot}")
        record_id = f"synthetic:{GENERATION_ID}:{slot:08d}"
        require(catalog["record_id"] == ledger["record_id"] == result["record_id"] == record_id,
                f"Generated record identity: {slot}")
        for field in ("report", "age", "sex", "heart_rate"):
            require(ledger["conditioning"][field] == candidate[field], f"Conditioning {field}: {slot}")
        ancestry.append((record_id, [parent_id], [parent["subject_uid"]]))
        signals.append(signal)
        rows.append({
            "slot_id": slot, "record_id": record_id, "conditioning_record_id": parent_id,
            "conditioning_subject_uid": parent["subject_uid"], "conditioning_fold": parent["strat_fold"],
            "conditioning_in_real_pool50": parent_id in pool, "conditioning_labels": parent["labels"],
            "label_policy": LABEL_POLICY, "diagnostic_label_validation": "not_performed",
            "generated_artifact_status": "unknown", "seed": candidate["seed"],
            "result_sha256": slot_item["result_sha256"], "signal_sha256": ledger["signal_sha256"],
            "acceptance_sha256": result["artifacts"]["acceptance.json"],
            "conditioning_report_sha256": candidate["report_sha256"],
        })
    require(len(rows) == 4515 and len({r["record_id"] for r in rows}) == 4515, "Accepted count differs")
    check_lineage(ancestry, manifest, index, allowed_roles=["pool", "unused"])
    parents = {r["conditioning_record_id"] for r in rows}
    subjects = {r["conditioning_subject_uid"] for r in rows}
    arrays = {
        "x": np.stack(signals),
        "labels": np.array([[c in r["conditioning_labels"] for c in SUPERCLASSES] for r in rows], bool),
        "record_id": np.array([r["record_id"] for r in rows], str),
        "subject_uid": np.array([r["conditioning_subject_uid"] for r in rows], str),
        "root_record_id": np.array([r["conditioning_record_id"] for r in rows], str),
        "root_subject_uid": np.array([r["conditioning_subject_uid"] for r in rows], str),
        "artifacts": np.zeros((4515, len(ARTIFACT_TYPES)), bool),
        "is_synthetic": np.ones(4515, bool),
        "slot_id": np.array([r["slot_id"] for r in rows], np.int64),
    }
    data = RUN / "data"
    data.mkdir(parents=True, exist_ok=True)
    freeze_npz(data / "generated.npz", arrays)
    freeze_json(data / "conditioning_labels.json", rows)
    provenance = {
        "schema_version": "ptbxl-diffusets-classification-provenance@2",
        "label_policy": LABEL_POLICY,
        "label_rationale": "Use the conditioning parent's five PTB-XL diagnostic superclasses as weak targets because its exact diagnostic report conditioned generation. Rhythm-only QA labels cannot establish these diagnoses. No generated diagnosis validation or model-based relabelling was performed.",
        "label_choice_authorization": "User requested the labels that make most sense for all 4,515 generated ECGs.",
        "class_order": list(SUPERCLASSES), "lead_order": list(LEADS), "sample_rate_hz": 100, "units": "mV",
        "generation_id": GENERATION_ID, "generation_path": str(GENERATION.relative_to(PROJECT)),
        "summary_sha256": sha(GENERATION / "summary.json"),
        "candidate_plan_sha256": sha(GENERATION / "candidate-plan.json"),
        "generation_identity_sha256": sha(GENERATION / "identity.json"),
        "attempted": 5000, "accepted": 4515, "rejected": 485,
        "accepted_slot_ids": [r["slot_id"] for r in rows],
        "verified_result_hashes": 5000, "verified_accepted_artifact_hashes": verified_files,
        "classification_manifest": str(MANIFEST.relative_to(PROJECT)),
        "classification_manifest_sha256": manifest.sha256, "source_sha256": source["sha256"],
        "real_pool_ecgs": len(pool), "real_pool_patients": len(pool_subjects),
        "distinct_conditioning_ecgs": len(parents), "distinct_conditioning_patients": len(subjects),
        "conditioning_coverage": dict(Counter("inside_real_pool50" if r["conditioning_in_real_pool50"] else "outside_real_pool50" for r in rows)),
        "additional_distinct_real_root_ecgs": len(parents - pool),
        "additional_distinct_real_root_patients": len(subjects - pool_subjects),
        "conditioning_class_counts": {c: int(arrays["labels"][:, i].sum()) for i, c in enumerate(SUPERCLASSES)},
        "conditioning_fold_counts": dict(Counter(str(r["conditioning_fold"]) for r in rows)),
        "validation_ecgs": len(manifest.roles["val"]), "test_ecgs": len(manifest.roles["test"]),
        "heldout_record_overlap": len(parents & heldout), "heldout_patient_overlap": len(subjects & heldout_subjects),
        "lineage_allowed_roles": ["pool", "unused"], "lineage_check": "PASS",
        "synthetic_artifact_flags": "Unknown. False placeholders must not be interpreted as clean labels or used to stratify generated ECGs.",
        "caveats": [
            "Synthetic diagnosis targets are conditioning-parent proxies, not verified generated diagnoses.",
            "The generator's training parents include records outside the 50% real classification pool, so this trial adds source information as well as generated waveforms.",
            "Acceptance verifies technical quality and rhythm measurements only; all 4,515 accepted ECGs have clinical_validation=not_performed.",
            "The test fold has been viewed in prior experiments; this is an exploratory trial.",
            "Generator checkpoint pretraining membership is unavailable for independent PTB-XL holdout overlap verification.",
        ],
    }
    freeze_json(data / "provenance.json", provenance)
    output_manifest = {
        "schema_version": "ptbxl-generated-training@1", "label_policy": LABEL_POLICY,
        "count": 4515, "class_order": list(SUPERCLASSES),
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        "files": {name: {"path": f"data/{name}", "sha256": sha(data / name)}
                  for name in ("generated.npz", "conditioning_labels.json", "provenance.json")},
    }
    freeze_json(data / "manifest.json", output_manifest)
    print(json.dumps({"status": "prepared", "count": len(rows), "manifest": str(data / "manifest.json"),
                      "manifest_sha256": sha(data / "manifest.json"), "files": output_manifest["files"],
                      "conditioning_coverage": provenance["conditioning_coverage"],
                      "heldout_record_overlap": 0, "heldout_patient_overlap": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
