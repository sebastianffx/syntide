#!/usr/bin/env python3
"""Collect DiffuSETS results after the runner reports a completed or failed job.

The remote deadline monitor remains independent of this local collector. Cost
estimates use this instance's bounded hourly rate. Finishing generation or
waiting for data preparation does not authorize collection or shutdown.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

import diffusets_guard
import diffusets_pod
import qa_pod
import watch_qa_study as base

ROOT = "/workspace/diffusets-study"
TERMINAL = base.TERMINAL | {"needs_time_budget_review"}
PROBE = r"""
import json, pathlib, time

root = pathlib.Path('/workspace/diffusets-study')
status = (
    json.loads((root / "status.json").read_text())
    if (root / "status.json").is_file()
    else None
)
guard_path = pathlib.Path('/workspace/diffusets-extension/guard-state.json')
guard = json.loads(guard_path.read_text()) if guard_path.exists() else None
guard_pid = (guard or {}).get("guard_pid", 0)
guard_process = pathlib.Path('/proc') / str(guard_pid)
try:
    guard_args = (
        (guard_process / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    )
    script_args = guard_args[1:]
    if script_args[:1] == [b"-u"]:
        script_args = script_args[1:]
    alive = (
        pathlib.Path(guard_args[0].decode()).name.startswith("python")
        and len(script_args) >= 2
        and script_args[0]
        in (
            b"/workspace/diffusets-extension/diffusets_guard.py",
            b"/workspace/diffusets-extension/diffusets_guard_v2.py",
        )
        and script_args[1]
        == b"/workspace/diffusets-extension/guard-config.json"
        and (guard_process / "stat").read_text().rsplit(")", 1)[1].split()[0]
        != "Z"
    )
except (FileNotFoundError, PermissionError):
    alive = False
processes = []
for proc in pathlib.Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        # Its operational override lives in the study tree, but the verified
        # guard is not workload and must remain alive through collection.
        if alive and int(proc.name) == guard_pid:
            continue
        command = (
            (proc / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
        if "/workspace/diffusets-study/" in command and (
            "python" in command or "bash" in command
        ):
            if (proc / "stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
                processes.append(
                    {"pid": int(proc.name), "command": command[:1000]}
                )
    except (FileNotFoundError, PermissionError):
        pass
age = time.time() - guard_path.stat().st_mtime if guard_path.exists() else None
healthy = bool(
    alive
    and age is not None
    and age < 180
    and (guard or {}).get("status") == "armed"
)
print(
    json.dumps(
        {
            "status": status,
            "processes": processes,
            "guard": guard,
            "guard_health": {
                "pid_alive": alive,
                "state_age_seconds": age,
                "healthy": healthy,
            },
        }
    )
)
"""
MANIFEST = r"""
import hashlib, json, pathlib

root = pathlib.Path("/workspace/diffusets-study")
if not root.is_dir():
    raise RuntimeError("Extension outputs absent")
files = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise RuntimeError("Refusing symlink in extension outputs")
    if path.is_file():
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        files[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": checksum,
        }
print(json.dumps({"root": str(root), "files": files}, sort_keys=True))
"""


class Supervisor(base.Supervisor):
    def __init__(self, args, client):
        self.args, self.client = args, client
        self.receipt = json.loads(Path(args.receipt).read_text())
        self.pod_id = self.receipt["pod"]["id"]
        if (
            self.pod_id == diffusets_pod.OLD_POD_ID
            or self.receipt.get("guard", {}).get("status") != "armed"
        ):
            raise ValueError(
                "Requires a distinct extension with its pod-local guard armed"
            )
        if self.receipt.get("status") != "created" or not self.receipt.get(
            "execute"
        ):
            raise ValueError("Completed creation receipt required")
        self.deadline = diffusets_guard.parse_utc(self.receipt["deadline_utc"])
        self.created = diffusets_guard.parse_utc(self.receipt["created_utc"])
        override_path = getattr(args, "operational_override", None)
        self.override = None
        if override_path:
            expected = getattr(args, "operational_override_sha256", None)
            if not expected or base.sha(override_path) != expected:
                raise ValueError("Operational override checksum differs")
            self.override = json.loads(Path(override_path).read_text())
            value = self.override
            if (
                value["pod_id"] != self.pod_id
                or value["created_utc"] != self.receipt["created_utc"]
                or value["receipt_sha256"] != base.sha(args.receipt)
                or value["max_spend_usd"]
                != self.receipt["cost_plan"]["max_spend_usd"]
                or value["hourly_rate_upper_bound_usd"]
                != self.receipt["cost_plan"]["hourly_rate_upper_bound_usd"]
            ):
                raise ValueError(
                    "Operational override changes owned pod or cost bounds"
                )
            diffusets_guard.validate_policy(value)
            self.deadline = diffusets_guard.parse_utc(value["deadline_utc"])
            if self.deadline <= diffusets_guard.parse_utc(
                self.receipt["deadline_utc"]
            ):
                raise ValueError(
                    "Operational override does not extend original deadline"
                )
        elif getattr(args, "operational_override_sha256", None):
            raise ValueError("Operational override checksum requires its file")
        self.state = (
            json.loads(Path(args.state).read_text())
            if Path(args.state).exists()
            else {}
        )
        if self.state and self.state.get("pod_id") != self.pod_id:
            raise ValueError("Monitor state belongs to another pod")
        self.state.update(
            pod_id=self.pod_id,
            receipt_sha256=base.sha(args.receipt),
            execute=args.execute,
            effective_deadline_utc=qa_pod.iso(self.deadline),
            operational_override_sha256=getattr(
                args, "operational_override_sha256", None
            ),
            storage_reserve_usd=(self.override or {}).get(
                "storage_reserve_usd", 0
            ),
            max_spend_usd=self.receipt["cost_plan"]["max_spend_usd"],
            spending_method="own_pod_elapsed_time_upper_bound; "
            "account_delta_not_attributed",
        )
        # These globals affect only this separate process, never the old
        # monitor.
        base.REMOTE_ROOT, base.SUPPORT = ROOT, {}
        base.PROBE_SCRIPT, base.MANIFEST_SCRIPT = PROBE, MANIFEST

    def poll(self):
        if (
            self.override
            and base.sha(self.args.operational_override)
            != self.args.operational_override_sha256
        ):
            raise ValueError(
                "Pinned operational override changed during supervision"
            )
        now = qa_pod.utc_now()
        remaining = (self.deadline - now).total_seconds()
        spend = (
            max(0, (now - self.created).total_seconds())
            / 3600
            * self.receipt["cost_plan"]["hourly_rate_upper_bound_usd"]
        )
        account = self.client.query(qa_pod.ACCOUNT_QUERY)["myself"]
        self.save(
            account=account,
            own_pod_spend_upper_bound_usd=spend,
            seconds_to_stop_deadline=remaining,
            last_error=None,
        )
        pod = self.own_pod()
        if pod is None:
            verified = bool(self.state.get("collection"))
            self.save(
                status="terminated_after_verified_collection"
                if verified
                else "pod_absent",
                action_required=not verified,
            )
            return True
        if pod["desiredStatus"] in {"EXITED", "STOPPED"}:
            self.save(
                status="pod_stopped_preserving_volume", action_required=True
            )
            return True
        reserve = (self.override or {}).get("storage_reserve_usd", 0)
        risk = (
            remaining < 1800
            or spend + reserve
            >= self.receipt["cost_plan"]["max_spend_usd"] - 0.5
        )
        argv = qa_pod.ssh_argv(pod, self.receipt["ssh_key_path"])
        if argv is None:
            if risk:
                return self.stop_preserving(
                    "Extension deadline or own-pod cost bound reached "
                    "without SSH"
                )
            self.save(status="waiting_for_ssh")
            return False
        probe = self.ssh(argv, PROBE)
        remote_status = (probe.get("status") or {}).get("status")
        had_running = self.state.get("saw_running_process", False) or bool(
            probe["processes"]
        )
        missing = (
            0
            if probe["processes"]
            else self.state.get("missing_process_polls", 0) + 1
        )
        guard_ok = probe.get("guard_health", {}).get("healthy", False)
        guard_ok = (
            guard_ok and (probe.get("guard") or {}).get("pod_id") == self.pod_id
        )
        if self.override:
            guard_ok = (
                guard_ok
                and (probe.get("guard") or {}).get(
                    "operational_override_sha256"
                )
                == self.args.operational_override_sha256
            )
        self.save(
            remote=probe,
            saw_running_process=had_running,
            missing_process_polls=missing,
            action_required=not guard_ok,
            guard_healthy=guard_ok,
        )
        terminal = remote_status in TERMINAL
        if terminal and not probe["processes"]:
            self.save(
                status="collecting", terminal_status=probe["status"]["status"]
            )
            timeout = min(
                self.args.collection_timeout_seconds, int(remaining - 960)
            )
            if timeout < 30:
                return self.stop_preserving(
                    "Too near pod-local stop deadline to collect safely"
                )
            return self.terminate_verified(self.collect(argv, timeout))
        if risk:
            return self.stop_preserving(
                "Extension deadline or own-pod cost bound reached"
            )
        if had_running and missing >= 2:
            if remote_status == "awaiting_preparation":
                self.save(
                    status="awaiting_continuation",
                    action_required=True,
                    reason="Generation finished; preparation/training "
                    "continuation is required",
                )
                return False
            self.save(status="unexpected_process_exit", action_required=True)
            return self.stop_preserving(
                "Extension runner disappeared on two polls; preserving "
                "all outputs"
            )
        self.save(status="watching")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt", default=diffusets_pod.DIRECTORY + "/pod-create.json"
    )
    parser.add_argument(
        "--state", default=diffusets_pod.DIRECTORY + "/supervisor.json"
    )
    parser.add_argument("--output", default=diffusets_pod.DIRECTORY + "/study")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--collection-timeout-seconds", type=int, default=1800)
    parser.add_argument("--operational-override")
    parser.add_argument("--operational-override-sha256")
    parser.add_argument(
        "--collection-lease-path",
        default="/workspace/diffusets-extension/collection-lease.json",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    Path(args.state).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.state).with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor = Supervisor(args, qa_pod.RunPod())
        while True:
            try:
                done = supervisor.poll()
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as error:
                supervisor.save(
                    status="supervisor_error",
                    last_error=f"{type(error).__name__}: {error}",
                )
                done = False
            print(json.dumps(supervisor.state, sort_keys=True), flush=True)
            if done or args.once:
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
