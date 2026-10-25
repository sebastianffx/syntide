#!/usr/bin/env python3
"""Prepare measured QA from the immutable 3,030-accepted DiffuSETS snapshot.

Run in the original generation environment. This wrapper never generates ECGs,
changes the original 5,000-slot plan, or claims that the full plan completed.
The original measurement/build/export implementations remain unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
import fcntl
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile


COUNT, ACCEPTED = 3380, 3030
HEARTBEAT_SHA = "9c2c45aa690a2ba8f9b01b7fd3b5b00be929adcf65d71e9bf1a39037976fddec"
IDENTITY_SHA = "8333a3ad0c2138b64cbd210f0d68957d47bdbd5c13ec71267020b4a8f29f64f2"
PLAN_SHA = "2bc86083e7a684340bd5a10e8a4ea174671e665e3b1f5903e78a865ec3f009f4"
BASELINE_MANIFEST_SHA = "b47801817e9661919b71ed96049d6fb473c2d63430de69434e657a7fd6c00e8f"
EXPECTED_ARTIFACTS = {
    "acceptance.json", "candidate.json", "catalog.jsonl", "ledger.jsonl",
    "run.json", "signals/00000000.npy",
}
META_FILES = {"snapshot_manifest.json", "source_heartbeat.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def freeze(path, value):
    path = Path(path)
    require(not path.is_symlink(), f"Symlink refused: {path}")
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        require(path.read_bytes() == encoded, f"Existing frozen artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        require(path.read_bytes() == encoded, f"Concurrent frozen artifact differs: {path}")
    finally:
        temporary.unlink()


def progress(args, phase, **details):
    """Write only this derivative preparation's durable progress file."""
    path = args.audit_output.with_name("preparation-progress.json")
    require(not path.is_symlink(), "Progress output symlink refused")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump({"status": "running", "phase": phase,
                   "updated_utc": datetime.now(UTC).isoformat(), **details},
                  stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe_relative(value):
    require(isinstance(value, str) and bool(value), "Invalid artifact path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value,
            f"Unsafe artifact path: {value!r}")
    return path


def validate_inventory(inventory):
    require(inventory["schema"] == "syntide.diffusets-prefix-inventory@1",
            "Unexpected snapshot inventory schema")
    require(inventory["candidate_count"] == COUNT and inventory["accepted_count"] == ACCEPTED
            and inventory["rejected_count"] == COUNT - ACCEPTED
            and inventory["prefix_slot_ids"] == list(range(COUNT)), "Snapshot count/prefix mismatch")
    require(inventory["generation_identity_sha256"] == IDENTITY_SHA
            and inventory["original_plan_file_sha256"] == PLAN_SHA, "Original provenance changed")
    slots = inventory["slots"]
    require([row["slot_id"] for row in slots] == list(range(COUNT)), "Slot inventory changed")
    require(all(type(row["accepted"]) is bool for row in slots), "Acceptance flags must be Boolean")
    accepted = [row["slot_id"] for row in slots if row["accepted"]]
    require(accepted == inventory["accepted_slot_ids"] and len(accepted) == ACCEPTED,
            "Accepted snapshot IDs changed")
    expected = {"identity.json", "candidate-plan.json"}
    for row in slots:
        prefix = f"candidates/{row['slot_id']:05d}"
        expected.add(prefix + "/result.json")
        if row["accepted"]:
            expected.update(prefix + "/" + name for name in EXPECTED_ARTIFACTS)
    require(set(inventory["files"]) == expected, "Snapshot file set changed")
    for name, info in inventory["files"].items():
        safe_relative(name)
        require(type(info["bytes"]) is int and info["bytes"] >= 0, "Invalid file size")
        require(isinstance(info["sha256"], str) and len(info["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in info["sha256"]), "Invalid file hash")
    require(inventory["total_bytes"] == sum(i["bytes"] for i in inventory["files"].values()),
            "Inventory byte total changed")
    return inventory


def verify_snapshot(snapshot):
    """Verify pinned source provenance and every local byte; no model imports."""
    snapshot = Path(snapshot)
    require(snapshot.is_dir() and not snapshot.is_symlink(), "Invalid snapshot directory")
    manifest = read(snapshot / "snapshot_manifest.json")
    require(manifest["schema"] == "syntide.diffusets-qa-snapshot@1"
            and manifest["local_and_remote_hashes_verified"] is True,
            "Requires the verified QA snapshot")
    inventory = validate_inventory(manifest["remote_inventory"])
    require(canonical(inventory) == manifest["remote_inventory_sha256"], "Inventory digest changed")
    require(manifest["source_heartbeat_sha256"] == HEARTBEAT_SHA
            and sha(snapshot / "source_heartbeat.json") == HEARTBEAT_SHA, "Heartbeat changed")
    heartbeat = read(snapshot / "source_heartbeat.json")
    require(heartbeat["count"] == COUNT and heartbeat["accepted"] == ACCEPTED
            and heartbeat["rejected"] == COUNT - ACCEPTED and heartbeat["identity_verified"] is True,
            "Heartbeat snapshot scope differs")
    allowed = set(inventory["files"]) | META_FILES
    seen = set()
    for current, dirs, files in os.walk(snapshot, followlinks=False):
        for name in dirs + files:
            require(not (Path(current) / name).is_symlink(), "Snapshot symlink refused")
        for name in files:
            path = Path(current) / name
            relative = path.relative_to(snapshot).as_posix()
            require(stat.S_ISREG(path.stat().st_mode) and relative in allowed,
                    f"Unexpected snapshot file: {relative}")
            seen.add(relative)
    require(seen == allowed, "Missing snapshot file")
    for name, info in inventory["files"].items():
        path = snapshot / name
        require(path.stat().st_size == info["bytes"] and sha(path) == info["sha256"],
                f"Snapshot hash/size mismatch: {name}")
    require(sha(snapshot / "identity.json") == IDENTITY_SHA, "Generation identity changed")
    identity, plan = read(snapshot / "identity.json"), read(snapshot / "candidate-plan.json")
    require(canonical({k: v for k, v in identity.items() if k != "generation_id"})
            == identity["generation_id"] == inventory["generation_id"], "Generation ID changed")
    require(canonical(plan) == identity["plan_sha256"] == inventory["plan_canonical_sha256"],
            "Original full plan changed")
    require(len(plan["candidates"]) == 5000
            and [r["slot_id"] for r in plan["candidates"]] == list(range(5000)),
            "Original plan must retain all 5,000 ordered slots")
    return manifest, inventory, identity, plan


def import_driver(path, identity):
    require(sha(path) == identity["driver_sha256"], "Generation driver source changed")
    spec = importlib.util.spec_from_file_location("snapshot_original_diffusets_driver", path)
    driver = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = driver
    spec.loader.exec_module(driver)
    require(driver.implementation_fingerprint() == identity["implementation"],
            "Original measurement/build implementation or packages changed")
    require(identity["qa_policy"] == driver.QA_POLICY
            and identity["qa_policy_sha256"] == canonical(driver.QA_POLICY), "Measured QA policy changed")
    return driver


def verified_bundles(snapshot, inventory, identity, plan, driver):
    bundles = []
    for slot in inventory["slots"]:
        number = slot["slot_id"]
        row = plan["candidates"][number]
        path = snapshot / "candidates" / f"{number:05d}"
        result = read(path / "result.json")
        require(sha(path / "result.json") == slot["result_sha256"]
                and result["candidate_sha256"] == canonical(row) == slot["candidate_sha256"]
                and result["generation_id"] == identity["generation_id"]
                and result["slot_id"] == number and result["seed"] == row["seed"] == slot["seed"]
                and result["accepted"] is slot["accepted"] and result["status"] == slot["status"],
                f"Published slot mismatch: {number}")
        require(result["accepted"] == (result["status"] == "verified"), "Acceptance status mismatch")
        if not result["accepted"]:
            continue
        require(set(result["artifacts"]) == EXPECTED_ARTIFACTS, "Accepted artifact set changed")
        driver.check_published(path, identity["generation_id"], row)
        require(read(path / "candidate.json") == row and read(path / "acceptance.json")["status"] == "verified",
                f"Accepted candidate/review mismatch: {number}")
        bundles.append(path)
    require(len(bundles) == ACCEPTED, "Accepted bundle count differs")
    return bundles


def prepare(args):
    from syntide.data.challenge import load_manifest
    from syntide.generate.qa.build import build_corpus, identity as corpus_identity
    from syntide.generate.qa.corpus import verify_corpus_files
    from syntide.generate.qa.export import export_prepared, verify_prepared
    from syntide.generate.qa.sources import open_source

    snapshot = args.snapshot.resolve()
    progress(args, "snapshot_hash_verification")
    _, inventory, generation, plan = verify_snapshot(snapshot)
    progress(args, "original_measurement_implementation_verification")
    driver = import_driver(args.generation_driver.resolve(), generation)
    split_manifest = load_manifest(args.manifest)
    require(split_manifest.sha256 == driver.POOL_SHA, "Training split differs from original")
    if "manifest_content_sha256" in plan:
        require(plan["manifest_sha256"] == sha(args.manifest), "Training manifest file changed")
    driver.validate_plan(plan, set(split_manifest.roles["pool"]))
    bundles = verified_bundles(snapshot, inventory, generation, plan, driver)
    baseline = verify_prepared(args.baseline_prepared)
    require(baseline["sha256"] == BASELINE_MANIFEST_SHA, "Baseline prepared manifest changed")
    source = open_source(args.registry, args.manifest, "pool", manifest_sha256=driver.POOL_SHA)
    config = driver.qa_config(bundles)
    config.name = "ecgqa-cot-generated-snapshot3030-seed0"
    expected_identity = corpus_identity(config, source, None, None)
    intent = {
        "schema_version": "syntide.diffusets-prefix-qa-preparation@1",
        "candidate_count": COUNT, "accepted_waveforms": ACCEPTED, "rejected": COUNT - ACCEPTED,
        "source_plan_candidate_count": 5000,
        "snapshot_manifest_sha256": sha(snapshot / "snapshot_manifest.json"),
        "remote_inventory_sha256": canonical(inventory),
        "source_heartbeat_sha256": HEARTBEAT_SHA,
        "generation_driver_sha256": sha(args.generation_driver),
        "wrapper_sha256": sha(__file__),
        "original_plan_file_sha256": PLAN_SHA,
        "original_generation_id": generation["generation_id"],
        "accepted_slot_ids": inventory["accepted_slot_ids"],
        "qa_config": asdict(config), "qa_config_sha256": canonical(asdict(config)),
        "qa_identity_sha256": canonical(expected_identity),
        "baseline_manifest_file_sha256": sha(args.baseline_prepared / "manifest.json"),
        "grounded_corpus_file_sha256": sha(args.grounded / "corpus.json"),
        "acquisition_corpus_file_sha256": sha(args.acquisition / "corpus.json"),
        "label_policy": "Measured waveform rhythm answers only; no conditioning-parent diagnosis labels",
        "scope": "Exploratory fixed completed prefix; does not complete or replace the original 5,000-slot generation plan",
    }
    freeze(args.audit_output.with_suffix(".intent.json"), intent)
    progress(args, "qa_build", accepted_waveforms=ACCEPTED)
    if (args.qa_output / "corpus.json").exists():
        info = verify_corpus_files(args.qa_output)
        require(canonical(info["identity"]) == canonical(expected_identity), "Existing QA corpus identity differs")
    else:
        info = build_corpus(config, source, args.qa_output)
    require(info["counts"]["generated_waveforms"] == ACCEPTED,
            "QA remeasurement did not retain all 3,030 accepted waveforms")
    families = info["counts"]["by_family"]
    expected_families = {t.split("@")[0] for t in driver.QA_POLICY["templates"]}
    require(set(families) <= expected_families
            and all(families.get(name, 0) == ACCEPTED for name in expected_families - {"rhythm.rate_band"})
            and 3 * ACCEPTED <= info["counts"]["qa_examples"] <= 4 * ACCEPTED,
            "Generated question families/counts differ from frozen rhythm policy")
    progress(args, "export", generated_qa_examples=info["counts"]["qa_examples"])
    prepared = export_prepared([args.grounded, args.acquisition, args.qa_output],
                               args.baseline_prepared, args.prepared_output)
    progress(args, "evaluation_byte_verification")
    eval_hashes = {}
    for split in ("selection", "validation", "test"):
        name = split + ".parquet"
        require(prepared["files"][name] == baseline["files"][name]
                and sha(args.prepared_output / name) == sha(args.baseline_prepared / name),
                f"Evaluation bytes changed: {split}")
        eval_hashes[name] = prepared["files"][name]["sha256"]
    generated_report = prepared["qa_export"]["corpora"][-1]
    require(generated_report["corpus_id"] == info["corpus_id"]
            and generated_report["counts"].get("exported_tasks") == info["counts"]["qa_examples"]
            and not generated_report["excluded"], "Generated QA tasks were lost during export")
    require(prepared["splits"]["train"]["questions"] == 209722 + info["counts"]["qa_examples"],
            "Combined original/grounded/acquisition/generated row count differs")
    strata = Counter(plan["candidates"][i].get("stratum", "unspecified")
                     for i in inventory["accepted_slot_ids"])
    report = {
        "schema_version": intent["schema_version"], "status": "prepared_verified",
        "preparation_intent_sha256": sha(args.audit_output.with_suffix(".intent.json")),
        "snapshot_manifest_sha256": intent["snapshot_manifest_sha256"],
        "candidate_count": COUNT, "accepted_waveforms": ACCEPTED, "rejected": COUNT - ACCEPTED,
        "generated_qa_examples": info["counts"]["qa_examples"], "generated_qa_by_family": families,
        "generated_corpus_id": info["corpus_id"], "accepted_parent_strata": dict(sorted(strata.items())),
        "prepared_manifest_sha256": prepared["sha256"],
        "prepared_manifest_file_sha256": sha(args.prepared_output / "manifest.json"),
        "training_questions": prepared["splits"]["train"]["questions"],
        "evaluation_file_sha256": eval_hashes, "evaluation_files_byte_identical": True,
        "prepared": str(args.prepared_output.resolve()), "qa_output": str(args.qa_output.resolve()),
        "label_policy": intent["label_policy"], "heldout_labels_parsed": False,
    }
    freeze(args.audit_output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "generation-driver", "registry", "manifest", "baseline-prepared",
                 "grounded", "acquisition", "qa-output", "prepared-output", "audit-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_output.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = prepare(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
