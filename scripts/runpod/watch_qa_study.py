#!/usr/bin/env python3
"""Collect and verify one training instance's outputs before releasing it.

Use --once for a single inspection or --execute to enable instance management.
Credentials stay in the local Keychain. The supervisor never changes other
instances. Stopping preserves the volume. Deletion requires two matching remote
file inventories and successful verification of every copied file.

The local supervisor enforces its deadline and spending limit. The provider's
original stopAfter setting is unverified. A deadline override changes only the
local deadline, leaving the creation receipt and provider schedule unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

import qa_pod

TERMINAL = {"complete", "failed", "needs_budget_review"}
REMOTE_ROOT = "/workspace/qa-study"
SUPPORT = {
    "/workspace/bootstrap.log": "_support/bootstrap.log",
    "/workspace/runtime-pip-freeze.txt": "_support/runtime-pip-freeze.txt",
}
STOP = """mutation($input: PodStopInput!) {
  podStop(input: $input) { id desiredStatus }
}"""
TERMINATE = (
    """mutation($input: PodTerminateInput!) { podTerminate(input: $input) }"""
)

PROBE_SCRIPT = r"""
import json, pathlib

root = pathlib.Path("/workspace/qa-study")
status_path = root / "status.json"
status = json.loads(status_path.read_text()) if status_path.is_file() else None
processes = []
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        command = (
            (proc / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
        if "run_qa_arms.py" in command or (
            "syntide_trainers.opentslm_loop" in command
            and "/workspace/qa-study/" in command
        ):
            state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if state != "Z":
                processes.append(
                    {"pid": int(proc.name), "command": command[:1000]}
                )
    except (FileNotFoundError, PermissionError):
        pass
print(json.dumps({"status": status, "processes": processes}))
"""

MANIFEST_SCRIPT = r"""
import hashlib, json, pathlib

root = pathlib.Path("/workspace/qa-study")
if not root.is_dir():
    raise RuntimeError("Study directory is absent")
files = {}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise RuntimeError("Study contains a symlink: " + str(path))
    if path.is_file():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
for remote, local in {
    "/workspace/bootstrap.log": "_support/bootstrap.log",
    "/workspace/runtime-pip-freeze.txt": "_support/runtime-pip-freeze.txt",
}.items():
    path = pathlib.Path(remote)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Missing support file: " + remote)
    if local in files:
        raise RuntimeError("Support destination conflicts with study content")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    files[local] = {"bytes": path.stat().st_size, "sha256": digest}
print(json.dumps({"root": str(root), "files": files}, sort_keys=True))
"""


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_deadline_utc(value):
    """Parse an explicit deadline in Coordinated Universal Time."""
    try:
        if "T" not in value or not value.endswith(("Z", "+00:00")):
            raise ValueError
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() != dt.timedelta(0):
            raise ValueError
        return parsed
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "Deadline must be an ISO UTC timestamp ending in Z or +00:00"
        ) from error


def verify_local(directory, manifest):
    directory = Path(directory).resolve()
    files = manifest.get("files", {})
    if not files:
        raise ValueError(
            "Empty artifact inventory cannot authorize termination"
        )
    for relative, expected in files.items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unsafe artifact inventory path")
        path = directory / name
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Local artifact path contains a symlink")
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha(path) != expected["sha256"]
        ):
            raise ValueError(
                f"Collected artifact does not match remote checksum: {relative}"
            )
    return {
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files.values()),
    }


class Supervisor:
    def __init__(self, args, client):
        self.args, self.client = args, client
        self.receipt = json.loads(Path(args.receipt).read_text())
        self.pod_id = self.receipt["pod"]["id"]
        if self.receipt.get("status") != "created" or not self.receipt.get(
            "execute"
        ):
            raise ValueError("Supervisor requires a completed create receipt")
        override = getattr(args, "deadline_utc", None)
        self.deadline = override or parse_deadline_utc(
            self.receipt["request"]["stopAfter"]
        )
        self.state = (
            json.loads(Path(args.state).read_text())
            if Path(args.state).exists()
            else {}
        )
        if self.state and self.state.get("pod_id") != self.pod_id:
            raise ValueError("Existing supervisor state belongs to another pod")
        self.state.update(
            pod_id=self.pod_id,
            receipt_sha256=sha(args.receipt),
            execute=args.execute,
            effective_deadline_utc=self.deadline.isoformat().replace(
                "+00:00", "Z"
            ),
            deadline_source="command_line_override"
            if override
            else "receipt_stopAfter",
            requested_api_stop_after_utc=self.receipt["request"]["stopAfter"],
            api_stop_after_enforcement="unverified",
            max_spend_usd=args.max_spend_usd,
        )

    def save(self, **fields):
        self.state.update(fields, checked_utc=qa_pod.iso(qa_pod.utc_now()))
        qa_pod.save(self.args.state, self.state)

    def own_pod(self):
        pod = self.client.query(
            qa_pod.POD_QUERY, {"input": {"podId": self.pod_id}}
        )["pod"]
        if pod and any(
            pod.get(key) != self.receipt["pod"].get(key)
            for key in ("id", "name", "imageName")
        ):
            raise ValueError(
                "Live pod identity differs from receipt; no action authorized"
            )
        return pod

    def ssh(self, argv, script, timeout=60):
        result = subprocess.run(
            [*argv, "python3 -"],
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"Remote collection/probe exited {result.returncode}: "
                f"{result.stderr[-2000:]}"
            )
        return json.loads(result.stdout)

    def collect(self, argv, timeout):
        lease_path = getattr(self.args, "collection_lease_path", None)
        if lease_path:
            allowed = {
                "/workspace/qa-backstop/collection-lease.json",
                "/workspace/diffusets-extension/collection-lease.json",
            }
            if lease_path not in allowed:
                raise ValueError("Unsupported own-pod collection lease path")
            lease = {
                "path": lease_path,
                "pod_id": self.pod_id,
                "seconds": min(1800, timeout),
            }
            script = (
                "payload = "
                + repr(lease)
                + "\n"
                + r"""
import datetime as dt, json, os, pathlib

now = dt.datetime.now(dt.UTC)
path = pathlib.Path(payload["path"])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".partial")
temporary.write_text(
    json.dumps(
        {
            "pod_id": payload["pod_id"],
            "issued_utc": now.isoformat(),
            "expires_utc": (
                now + dt.timedelta(seconds=payload["seconds"])
            ).isoformat(),
        }
    )
    + "\n"
)
os.replace(temporary, path)
print(
    json.dumps(
        {"collection_lease_written": True, "seconds": payload["seconds"]}
    )
)
"""
            )
            self.ssh(argv, script, timeout=30)
        expires = time.monotonic() + timeout

        def remaining():
            value = int(expires - time.monotonic())
            if value < 1:
                raise TimeoutError("Collection deadline reached")
            return value

        before = self.ssh(argv, MANIFEST_SCRIPT, timeout=min(120, remaining()))
        destination = Path(self.args.output).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        ssh_transport, host = shlex.join(argv[:-1]), argv[-1]
        transfers = [(REMOTE_ROOT + "/", str(destination) + "/")]
        support_dir = destination / "_support"
        support_dir.mkdir(exist_ok=True)
        transfers += [
            (remote, str(destination / local))
            for remote, local in SUPPORT.items()
        ]
        for source, target in transfers:
            result = subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--no-owner",
                    "--no-group",
                    "--checksum",
                    "--partial",
                    "--timeout=120",
                    "-e",
                    ssh_transport,
                    host + ":" + source,
                    target,
                ],
                capture_output=True,
                text=True,
                timeout=remaining(),
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"rsync exited {result.returncode}: {result.stderr[-2000:]}"
                )
        checked = verify_local(destination, before)
        after = self.ssh(argv, MANIFEST_SCRIPT, timeout=min(120, remaining()))
        if before != after:
            raise ValueError(
                "Remote artifacts changed during collection; "
                "termination refused"
            )
        probe = self.ssh(argv, PROBE_SCRIPT, timeout=min(60, remaining()))
        if probe["processes"]:
            raise ValueError(
                "Study process restarted during collection; termination refused"
            )
        manifest_path = (
            Path(self.args.state).parent / "collected-remote-manifest.json"
        )
        qa_pod.save(manifest_path, before)
        return {
            **checked,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha(manifest_path),
            "verified_utc": qa_pod.iso(qa_pod.utc_now()),
        }

    def stop_preserving(self, reason):
        self.save(status="stop_required", action_required=True, reason=reason)
        if not self.args.execute:
            return True
        pod = self.own_pod()
        if pod and pod["desiredStatus"] not in {"EXITED", "STOPPED"}:
            self.client.query(STOP, {"input": {"podId": self.pod_id}})
        self.save(
            status="stopped_preserving_volume",
            action_required=True,
            stopped_pod=self.own_pod(),
        )
        return True

    def terminate_verified(self, verification):
        if not verification or not verification.get("files"):
            raise ValueError("No verified collection; termination refused")
        self.save(status="collected_verified", collection=verification)
        if not self.args.execute:
            self.save(status="would_terminate_verified_pod")
            return True
        # Recheck the persisted collection immediately before irreversible
        # deletion.
        manifest_path = Path(verification["manifest_path"])
        if sha(manifest_path) != verification["manifest_sha256"]:
            raise ValueError("Saved inventory changed; termination refused")
        verify_local(self.args.output, json.loads(manifest_path.read_text()))
        if self.own_pod():
            self.client.query(TERMINATE, {"input": {"podId": self.pod_id}})
        self.save(status="termination_requested", action_required=False)
        pod = self.own_pod()
        account = self.client.query(qa_pod.ACCOUNT_QUERY)["myself"]
        if pod is not None:
            self.save(status="termination_pending", account_after=account)
            return False
        self.save(
            status="terminated_after_verified_collection", account_after=account
        )
        return True

    def poll(self):
        account = self.client.query(qa_pod.ACCOUNT_QUERY)["myself"]
        remaining = (self.deadline - qa_pod.utc_now()).total_seconds()
        balance_delta = (
            self.receipt["account_before"]["clientBalance"]
            - account["clientBalance"]
        )
        self.save(
            account=account,
            observed_account_balance_decrease=balance_delta,
            seconds_to_stop_deadline=remaining,
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
        risk = (
            remaining < self.args.deadline_margin_seconds
            or balance_delta > self.args.max_spend_usd
        )
        argv = qa_pod.ssh_argv(pod, self.receipt["ssh_key_path"])
        if argv is None:
            if risk:
                return self.stop_preserving(
                    "Deadline or spending guard reached without SSH"
                )
            self.save(status="waiting_for_ssh")
            return False
        try:
            probe = self.ssh(argv, PROBE_SCRIPT)
            remote_status = (probe["status"] or {}).get("status")
            had_running = self.state.get("saw_running_process", False) or bool(
                probe["processes"]
            )
            missing_count = (
                0
                if probe["processes"]
                else self.state.get("missing_process_polls", 0) + 1
            )
            self.save(
                remote=probe,
                saw_running_process=had_running,
                missing_process_polls=missing_count,
            )
            terminal = remote_status in TERMINAL
            exited = had_running and missing_count >= 2
            if not probe["processes"] and (terminal or exited):
                self.save(
                    status="collecting",
                    terminal_status=remote_status or "unexpected_process_exit",
                )
                timeout = min(
                    self.args.collection_timeout_seconds, int(remaining - 90)
                )
                if timeout < 30:
                    return self.stop_preserving(
                        "Insufficient time to collect before local "
                        "stop deadline"
                    )
                verification = self.collect(argv, timeout=timeout)
                return self.terminate_verified(verification)
            if risk:
                return self.stop_preserving(
                    "Deadline or spending guard reached while work is active"
                )
            self.save(status="watching")
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.TimeoutExpired,
        ) as error:
            self.save(
                status="retryable_collection_or_probe_error",
                last_error=f"{type(error).__name__}: {error}",
            )
            if (
                risk
                or (self.deadline - qa_pod.utc_now()).total_seconds()
                < self.args.deadline_margin_seconds
            ):
                return self.stop_preserving(
                    "Unable to collect safely before the cost/deadline guard"
                )
        return False


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--receipt", default="outputs/qa-runs/pod-create.json")
    result.add_argument("--state", default="outputs/qa-runs/supervisor.json")
    result.add_argument("--output", default="outputs/qa-runs/study")
    result.add_argument("--poll-seconds", type=int, default=60)
    result.add_argument(
        "--deadline-utc",
        type=parse_deadline_utc,
        help="Override only the local deadline with an ISO UTC timestamp; "
        "receipt and API remain unchanged",
    )
    result.add_argument("--deadline-margin-seconds", type=int, default=900)
    result.add_argument("--collection-timeout-seconds", type=int, default=1800)
    result.add_argument(
        "--collection-lease-path",
        help="Bounded pod-side lease while collecting verified artifacts",
    )
    result.add_argument("--max-spend-usd", type=float, default=50)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--once", action="store_true")
    return result


def main():
    args = parser().parse_args()
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
