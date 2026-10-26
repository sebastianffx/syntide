#!/usr/bin/env python3
"""Freeze and collect only the 1,530 accepted ECGs in the known 1,721-slot prefix.

  uv run python runs/ptbxl-diffusets-1530/collect_snapshot.py prepare
  uv run python runs/ptbxl-diffusets-1530/collect_snapshot.py collect
  uv run python runs/ptbxl-diffusets-1530/collect_snapshot.py verify

prepare performs an authenticated read-only pod lookup and fixed-prefix inventory.
collect uses rsync --files-from, validates all hashes locally, inventories the same
remote prefix again, then atomically publishes the local snapshot directory.
No supervisor, job, GPU operation, lease, lock, or remote mutation is invoked.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT = REPO / "outputs/ptbxl-diffusets-classification-1530"
SNAPSHOT = OUTPUT / "snapshot"
PARTIAL = OUTPUT / "snapshot.partial"
TRANSFER = OUTPUT / "transfer_manifest.json"
FILE_LIST = OUTPUT / "transfer_files.txt"
RECEIPT = REPO / "outputs/qa-runs/diffusets-extension/pod-create.json"
HEARTBEAT = REPO / "outputs/qa-runs/diffusets-extension/heartbeat-20261023T2205.json"
HEARTBEAT_SHA = "0e13d385152319db8964e05aa2beac6f89f403a591048439288e34452126d66e"
EXPECTED_POD = "0il9463gvr3mmv"
REMOTE_ROOT = "/workspace/diffusets-study/generation"
IDENTITY_SHA = "8333a3ad0c2138b64cbd210f0d68957d47bdbd5c13ec71267020b4a8f29f64f2"
PLAN_SHA = "2bc86083e7a684340bd5a10e8a4ea174671e665e3b1f5903e78a865ec3f009f4"
LOCAL_METADATA = {"snapshot_manifest.json", "source_heartbeat.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, data):
    """Atomically publish exact bytes once; never replace an existing artifact."""
    path = Path(path)
    require(not path.is_symlink(), f"Symlink refused: {path}")
    if path.exists():
        require(path.read_bytes() == data, f"Existing frozen file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        require(path.read_bytes() == data, f"Concurrent frozen file differs: {path}")
    finally:
        temporary.unlink()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def check_relative(value):
    path = PurePosixPath(value)
    require(bool(value) and not path.is_absolute() and ".." not in path.parts and str(path) == value,
            f"Unsafe relative path: {value!r}")


def check_inventory(inventory):
    require(inventory["schema"] == "syntide.diffusets-prefix-inventory@1"
            and inventory["remote_root"] == REMOTE_ROOT, "Inventory schema/root differs")
    require(inventory["candidate_count"] == 1721 and inventory["accepted_count"] == 1530
            and inventory["rejected_count"] == 191 and inventory["prefix_slot_ids"] == list(range(1721)),
            "Snapshot scope differs from the 1,721 / 1,530 heartbeat")
    require(inventory["generation_identity_sha256"] == IDENTITY_SHA
            and inventory["original_plan_file_sha256"] == PLAN_SHA, "Frozen identity/plan differs")
    slots = inventory["slots"]
    require([row["slot_id"] for row in slots] == list(range(1721)), "Slot inventory is not the exact prefix")
    accepted = [row["slot_id"] for row in slots if row["accepted"]]
    require(accepted == inventory["accepted_slot_ids"] and len(accepted) == 1530, "Accepted IDs differ")
    files = inventory["files"]
    require(files["identity.json"]["sha256"] == IDENTITY_SHA, "Identity-file manifest hash differs")
    expected = {"identity.json", "candidate-plan.json"}
    for row in slots:
        prefix = f"candidates/{row['slot_id']:05d}"
        expected.add(prefix + "/result.json")
        if row["accepted"]:
            expected.update(prefix + "/" + name for name in
                            ("acceptance.json", "candidate.json", "catalog.jsonl", "ledger.jsonl",
                             "run.json", "signals/00000000.npy"))
    require(set(files) == expected, "Transfer manifest contains missing, later, or unrelated candidate files")
    for name, info in files.items():
        check_relative(name)
        require(isinstance(info["bytes"], int) and info["bytes"] >= 0
                and len(info["sha256"]) == 64 and all(c in "0123456789abcdef" for c in info["sha256"]),
                f"Invalid file metadata: {name}")
    require(inventory["total_bytes"] == sum(info["bytes"] for info in files.values()), "Byte totals differ")


def source_checks():
    require(sha(HEARTBEAT) == HEARTBEAT_SHA, "Local heartbeat changed")
    heartbeat = read(HEARTBEAT)
    require(heartbeat["count"] == 1721 and heartbeat["accepted"] == 1530
            and heartbeat["identity_verified"] is True, "Unexpected heartbeat scope")
    receipt = read(RECEIPT)
    require(receipt["pod"]["id"] == EXPECTED_POD and receipt["status"] == "created"
            and receipt["execute"] is True, "Unexpected pod receipt")
    return receipt


def ssh_endpoint():
    receipt = source_checks()
    sys.path.insert(0, str(REPO / "scripts/runpod"))
    import qa_pod
    # Only an authenticated read query; never call create/status/supervisor methods.
    pod = qa_pod.RunPod().query(qa_pod.POD_QUERY, {"input": {"podId": EXPECTED_POD}})["pod"]
    require(pod is not None and pod["id"] == EXPECTED_POD, "Owned snapshot pod is unavailable")
    argv = qa_pod.ssh_argv(pod, receipt["ssh_key_path"])
    require(bool(argv), "Owned pod has no current public SSH endpoint")
    return argv


def remote_inventory(argv):
    script = (HERE / "remote_snapshot_preflight.py").read_bytes()
    result = subprocess.run([*argv, "python3 -"], input=script, capture_output=True, timeout=180)
    require(result.returncode == 0,
            "Read-only remote inventory failed: " + result.stderr.decode(errors="replace")[-4000:])
    inventory = json.loads(result.stdout)
    check_inventory(inventory)
    return inventory


def prepare():
    inventory = remote_inventory(ssh_endpoint())
    payload = {"schema": "syntide.diffusets-snapshot-transfer@1", "pod_id": EXPECTED_POD,
               "heartbeat_sha256": HEARTBEAT_SHA, "source_heartbeat": str(HEARTBEAT.relative_to(REPO)),
               "receipt_sha256": sha(RECEIPT), "remote_inventory": inventory,
               "remote_inventory_sha256": canonical(inventory),
               "selection_note": "Exactly the immutable completed prefix observed in the pinned heartbeat; later generation is excluded.",
               "label_note": "No diagnostic class labels are established by this collection or the generation acceptance checks."}
    require(not OUTPUT.is_symlink(), f"Symlink refused: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    freeze(TRANSFER, json_bytes(payload))
    freeze(FILE_LIST, ("\n".join(sorted(inventory["files"])) + "\n").encode())
    print(json.dumps({"status": "prepared", "files": len(inventory["files"]),
                      "bytes": inventory["total_bytes"], "accepted": 1530,
                      "attempted_prefix": 1721, "manifest": str(TRANSFER)}, indent=2))


def load_transfer():
    source_checks()
    payload = read(TRANSFER)
    require(payload["pod_id"] == EXPECTED_POD and payload["heartbeat_sha256"] == HEARTBEAT_SHA
            and payload["receipt_sha256"] == sha(RECEIPT), "Transfer source binding differs")
    inventory = payload["remote_inventory"]
    check_inventory(inventory)
    require(payload["remote_inventory_sha256"] == canonical(inventory), "Transfer inventory digest differs")
    require(FILE_LIST.read_text() == "\n".join(sorted(inventory["files"])) + "\n", "Transfer file list changed")
    return payload


def inspect_local_tree(directory, expected, *, require_complete):
    require(directory.is_dir() and not directory.is_symlink(), f"Invalid local snapshot directory: {directory}")
    seen = set()
    for current, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            require(not path.is_symlink(), f"Symlink refused: {path}")
        for name in files:
            path = Path(current) / name
            require(stat.S_ISREG(path.stat().st_mode), f"Not a regular file: {path}")
            relative = path.relative_to(directory).as_posix()
            require(relative in expected or relative in LOCAL_METADATA, f"Unlisted snapshot file: {relative}")
            seen.add(relative)
    if require_complete:
        require(set(expected) <= seen, f"Missing snapshot files: {sorted(set(expected) - seen)[:10]}")


def verify_files(directory, inventory):
    inspect_local_tree(directory, inventory["files"], require_complete=True)
    for name, info in inventory["files"].items():
        path = directory / name
        require(path.stat().st_size == info["bytes"] and sha(path) == info["sha256"],
                f"Local file checksum/size mismatch: {name}")


def verify_snapshot(payload=None):
    payload = payload or load_transfer()
    inventory = payload["remote_inventory"]
    verify_files(SNAPSHOT, inventory)
    manifest = read(SNAPSHOT / "snapshot_manifest.json")
    require(manifest["transfer_manifest_sha256"] == sha(TRANSFER)
            and manifest["remote_inventory_sha256"] == canonical(inventory)
            and manifest["remote_inventory"] == inventory
            and manifest["source_heartbeat_sha256"] == HEARTBEAT_SHA
            and sha(SNAPSHOT / "source_heartbeat.json") == HEARTBEAT_SHA,
            "Published snapshot manifest/source binding differs")
    print(json.dumps({"status": "verified", "snapshot": str(SNAPSHOT),
                      "candidate_count": 1721, "accepted_count": 1530,
                      "files": len(inventory["files"]), "snapshot_manifest_sha256": sha(SNAPSHOT / "snapshot_manifest.json")}, indent=2))


def collect():
    payload = load_transfer()
    inventory = payload["remote_inventory"]
    if SNAPSHOT.exists():
        verify_snapshot(payload)
        return
    require(not PARTIAL.is_symlink(), f"Symlink refused: {PARTIAL}")
    PARTIAL.mkdir(parents=True, exist_ok=True)
    inspect_local_tree(PARTIAL, inventory["files"], require_complete=False)
    argv = ssh_endpoint()
    # --files-from is the entire transfer scope. No --delete, remote writes, or whole-tree recursion.
    command = ["rsync", "-a", "--no-owner", "--no-group", "--checksum", "--partial", "--timeout=120",
               "--files-from", str(FILE_LIST), "-e", shlex.join(argv[:-1]),
               argv[-1] + ":" + REMOTE_ROOT + "/", str(PARTIAL) + "/"]
    print(f"Collecting {len(inventory['files'])} fixed files ({inventory['total_bytes']:,} bytes).", flush=True)
    result = subprocess.run(command, capture_output=True, timeout=900)
    require(result.returncode == 0, "Fixed-list rsync failed: " + result.stderr.decode(errors="replace")[-4000:])
    verify_files(PARTIAL, inventory)
    require(remote_inventory(argv) == inventory, "Remote immutable prefix differs after transfer")
    freeze(PARTIAL / "source_heartbeat.json", HEARTBEAT.read_bytes())
    manifest = {"schema": "syntide.diffusets-classification-snapshot@1", "verified_utc": datetime.now(UTC).isoformat(),
                "pod_id": EXPECTED_POD, "source_heartbeat_sha256": HEARTBEAT_SHA,
                "transfer_manifest_sha256": sha(TRANSFER), "remote_inventory_sha256": canonical(inventory),
                "remote_inventory": inventory, "local_and_remote_hashes_verified": True,
                "diagnostic_labels": "not provided or validated; a separate explicit label policy is required",
                "collection_scope": "Only 1,530 accepted bundles and all 1,721 prefix result files; later candidates excluded.",
                "remote_operations": ["authenticated pod read query", "read-only prefix inventory twice", "rsync fixed-file pull"]}
    existing = PARTIAL / "snapshot_manifest.json"
    if existing.exists():
        previous = read(existing)
        manifest["verified_utc"] = previous["verified_utc"]
    freeze(existing, json_bytes(manifest))
    require(not SNAPSHOT.exists(), "Snapshot appeared concurrently; refusing replacement")
    # Same-filesystem publication only after every payload and its immutable manifest verifies.
    PARTIAL.rename(SNAPSHOT)
    verify_snapshot(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect", "verify"))
    args = parser.parse_args()
    try:
        {"prepare": prepare, "collect": collect, "verify": verify_snapshot}[args.command]()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
