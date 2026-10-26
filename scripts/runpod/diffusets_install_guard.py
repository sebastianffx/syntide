#!/usr/bin/env python3
"""Install or inspect the DiffuSETS instance's independent deadline monitor.

The monitor runs separately from the training process and never receives an
account credential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import diffusets_pod
import qa_pod

REMOTE = "/workspace/diffusets-extension"
INSTALL = r"""
import json, os, pathlib, subprocess, time

root = pathlib.Path("/workspace/diffusets-extension")
root.mkdir(exist_ok=True)
source_path = root / "diffusets_guard.py"
config_path = root / "guard-config.json"
state_path = root / "guard-state.json"
expected = payload["config"]
if config_path.exists() and json.loads(config_path.read_text()) != expected:
    raise RuntimeError(
        "Existing guard configuration differs; refusing to replace it"
    )
if state_path.exists():
    current = json.loads(state_path.read_text())
    process = (
        pathlib.Path("/proc") / str(current.get("guard_pid", 0)) / "cmdline"
    )
    if process.exists() and b"diffusets_guard.py" in process.read_bytes():
        if current.get("pod_id") != expected["pod_id"]:
            raise RuntimeError("Existing guard belongs to a different pod")
        print(json.dumps({"already_running": True, "state": current}))
        raise SystemExit(0)
source_path.write_text(payload["source"])
config_path.write_text(json.dumps(expected, indent=2) + "\n")
os.chmod(source_path, 0o700)
os.chmod(config_path, 0o600)
checked = subprocess.run(
    ["python3", str(source_path), str(config_path), "--check-only"],
    capture_output=True,
    text=True,
    timeout=45,
)
if checked.returncode:
    # No raw command output: CLI failures may include sensitive platform data.
    raise RuntimeError("Guard platform identity/authentication probe failed")
preflight = json.loads(checked.stdout)
with (root / "guard.log").open("ab") as log:
    child = subprocess.Popen(
        ["python3", "-u", str(source_path), str(config_path)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
for _ in range(20):
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if (
            state.get("guard_pid") == child.pid
            and state.get("status") == "armed"
        ):
            print(
                json.dumps(
                    {
                        "already_running": False,
                        "preflight": preflight,
                        "state": state,
                    }
                )
            )
            break
    if child.poll() is not None:
        raise RuntimeError(
            "Guard exited before arming; inspect redacted guard status"
        )
    time.sleep(0.5)
else:
    raise RuntimeError("Guard arming not confirmed; no workload authorized")
"""


def guard_config(receipt):
    if receipt.get("status") != "created" or not receipt.get("execute"):
        raise ValueError("Completed paid pod receipt required")
    if receipt["pod"]["id"] == diffusets_pod.OLD_POD_ID:
        raise ValueError("Original study pod must never receive this guard")
    if not receipt.get("hourly_rate_matches"):
        raise ValueError("Live pod price exceeded the approved cap")
    return {
        "pod_id": receipt["pod"]["id"],
        "created_utc": receipt["created_utc"],
        "deadline_utc": receipt["deadline_utc"],
        "hourly_rate_upper_bound_usd": receipt["cost_plan"][
            "hourly_rate_upper_bound_usd"
        ],
        "max_spend_usd": receipt["cost_plan"]["max_spend_usd"],
        "state_path": REMOTE + "/guard-state.json",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt", default=diffusets_pod.DIRECTORY + "/pod-create.json"
    )
    args = parser.parse_args()
    path = Path(args.receipt)
    receipt = json.loads(path.read_text())
    config = guard_config(receipt)
    status = diffusets_pod.status(args, qa_pod.RunPod())
    if not status["ssh_argv"]:
        raise RuntimeError("New pod SSH is not ready yet; no guard installed")
    source = Path(__file__).with_name("diffusets_guard.py").read_text()
    script = (
        "payload = "
        + repr({"source": source, "config": config})
        + "\n"
        + INSTALL
    )
    result = subprocess.run(
        [*status["ssh_argv"], "python3 -"],
        input=script,
        text=True,
        capture_output=True,
        timeout=70,
    )
    if result.returncode:
        raise RuntimeError(
            "Remote guard install failed; inspect its safe status "
            "before workload"
        )
    installed = json.loads(result.stdout)
    if (
        installed["state"]["pod_id"] != receipt["pod"]["id"]
        or installed["state"]["status"] != "armed"
    ):
        raise RuntimeError("Guard did not confirm own-pod armed state")
    evidence = {
        "checked_utc": qa_pod.iso(qa_pod.utc_now()),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "remote": installed,
        "ssh_argv": status["ssh_argv"],
        "account_key_transferred": False,
    }
    qa_pod.save(path.parent / "guard-install.json", evidence)
    receipt["guard"].update(
        status="armed",
        evidence_file="guard-install.json",
        guard_pid=installed["state"]["guard_pid"],
    )
    qa_pod.save(path, receipt)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
