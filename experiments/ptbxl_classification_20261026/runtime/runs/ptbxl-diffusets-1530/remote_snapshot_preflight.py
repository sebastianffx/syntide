#!/usr/bin/env python3
"""Read-only remote inventory of a fixed, already published DiffuSETS prefix.

This script uses only the standard library, never acquires a writer lock, never
imports generation code, and never writes a remote file. Its sole output is JSON.
It is sent over SSH stdin by collect_snapshot.py; do not run the job supervisors.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path("/workspace/diffusets-study/generation")
PLAN = Path("/workspace/diffusets-study/planning/candidate-plan.json")
PLAN_SHA = "2bc86083e7a684340bd5a10e8a4ea174671e665e3b1f5903e78a865ec3f009f4"
IDENTITY_SHA = "8333a3ad0c2138b64cbd210f0d68957d47bdbd5c13ec71267020b4a8f29f64f2"
COUNT, ACCEPTED = 1721, 1530
EXPECTED_ARTIFACTS = {
    "acceptance.json", "candidate.json", "catalog.jsonl", "ledger.jsonl",
    "run.json", "signals/00000000.npy",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def safe_relative(value):
    require(isinstance(value, str) and bool(value), "Invalid artifact path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value,
            f"Unsafe artifact path: {value!r}")
    return path


def check_path(path, root):
    require(not root.is_symlink() and root.is_dir(), f"Invalid root: {root}")
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f"Symlink refused: {current}")
    require(path.is_file(), f"Missing regular file: {path}")


def inspect(path, root):
    check_path(path, root)
    before = path.stat()
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    require(all(getattr(before, key) == getattr(after, key) for key in fields),
            f"File changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": checksum.hexdigest()}


def read_json(path, root):
    check_path(path, root)
    return json.loads(path.read_text())


def main():
    plan_info = inspect(PLAN, PLAN.parent)
    require(plan_info["sha256"] == PLAN_SHA, "Frozen planning file hash differs")
    plan = read_json(PLAN, PLAN.parent)
    rows = plan["candidates"]
    require(len(rows) == 5000 and [row["slot_id"] for row in rows] == list(range(5000)),
            "Expected the original ordered 5,000-slot plan")
    identity_info = inspect(ROOT / "identity.json", ROOT)
    require(identity_info["sha256"] == IDENTITY_SHA, "Frozen generation identity hash differs")
    identity = read_json(ROOT / "identity.json", ROOT)
    generation_id = identity["generation_id"]
    require(digest({k: v for k, v in identity.items() if k != "generation_id"}) == generation_id,
            "Generation ID does not match its canonical identity")
    require(identity["plan_sha256"] == digest(plan), "Generation plan identity differs")
    published_plan = read_json(ROOT / "candidate-plan.json", ROOT)
    require(published_plan == plan, "Published plan differs from original planning file")
    files = {"identity.json": identity_info,
             "candidate-plan.json": inspect(ROOT / "candidate-plan.json", ROOT)}
    slots = []
    for slot in range(COUNT):
        row = rows[slot]
        prefix = f"candidates/{slot:05d}"
        directory = ROOT / prefix
        result_path = directory / "result.json"
        result_info = inspect(result_path, ROOT)
        result = read_json(result_path, ROOT)
        require(result["slot_id"] == slot and result["seed"] == row["seed"],
                f"Slot or seed mismatch: {slot}")
        require(result["generation_id"] == generation_id and result["candidate_sha256"] == digest(row),
                f"Published candidate identity mismatch: {slot}")
        require(type(result["accepted"]) is bool and result["accepted"] == (result["status"] == "verified"),
                f"Inconsistent acceptance status: {slot}")
        files[prefix + "/result.json"] = result_info
        slots.append({"slot_id": slot, "accepted": result["accepted"], "status": result["status"],
                      "result_sha256": result_info["sha256"], "candidate_sha256": result["candidate_sha256"],
                      "seed": row["seed"], "source_record_id": row["source_record_id"],
                      "record_id": result.get("record_id")})
        # Rejected results preserve the evidence and denominator. Their payloads are not downloaded.
        if not result["accepted"]:
            continue
        artifacts = result["artifacts"]
        require(set(artifacts) == EXPECTED_ARTIFACTS, f"Unexpected accepted-bundle file set: {slot}")
        for relative, expected_sha in sorted(artifacts.items()):
            safe_relative(relative)
            name = prefix + "/" + relative
            info = inspect(ROOT / name, ROOT)
            require(info["sha256"] == expected_sha, f"Accepted artifact hash mismatch: {name}")
            files[name] = info
        require(read_json(directory / "candidate.json", ROOT) == row, f"Candidate contents differ: {slot}")
        acceptance = read_json(directory / "acceptance.json", ROOT)
        require(acceptance["status"] == "verified", f"Acceptance document differs: {slot}")
        ledger = json.loads((directory / "ledger.jsonl").read_text())
        require(ledger["source_record_ids"] == [row["source_record_id"]]
                and ledger["seed"] == row["seed"] and ledger["record_id"] == result["record_id"]
                and ledger["signal_path"] == "signals/00000000.npy"
                and ledger["signal_sha256"] == artifacts["signals/00000000.npy"],
                f"Accepted ledger lineage differs: {slot}")
    accepted = [row["slot_id"] for row in slots if row["accepted"]]
    require(len(accepted) == ACCEPTED, f"Expected {ACCEPTED} accepted in prefix; found {len(accepted)}")
    payload = {
        "schema": "syntide.diffusets-prefix-inventory@1", "remote_root": str(ROOT),
        "candidate_count": COUNT, "accepted_count": ACCEPTED, "rejected_count": COUNT - ACCEPTED,
        "prefix_slot_ids": list(range(COUNT)), "accepted_slot_ids": accepted,
        "generation_id": generation_id, "generation_identity_sha256": IDENTITY_SHA,
        "original_plan_file_sha256": PLAN_SHA, "plan_canonical_sha256": digest(plan),
        "selection_rule": "Completed ordered prefix 0..1720 from the 2026-10-23T22:02:51.597781+00:00 heartbeat; keep accepted slots only.",
        "label_note": "Accepted means technical/rhythm evidence checks passed, not validated diagnostic class labels.",
        "slots": slots, "files": dict(sorted(files.items())),
        "total_bytes": sum(item["bytes"] for item in files.values()),
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
