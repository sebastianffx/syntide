#!/usr/bin/env python3
"""Stop a RunPod instance when its deadline or spending limit is reached.

Authentication uses the credential injected into that instance by RunPod. The
account credential stays on the local machine. Stopping preserves /workspace and
its storage charges; this script never deletes the instance.

An operational override must match the saved configuration's cryptographic hash.
It cannot change the instance, creation time, or spending limit. The full
allowed interval and reserved storage cost must fit that limit. Stopping a
completed job also requires matching execution records, free execution locks,
verified results when successful, and an expired collection grace period.
Processor utilization alone never establishes completion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, contextmanager
from pathlib import Path

STOP_MARGIN_SECONDS = 900
TERMINAL = {
    "complete",
    "failed",
    "needs_budget_review",
    "needs_time_budget_review",
    "idle",
}


def parse_utc(value):
    if (
        not isinstance(value, str)
        or "T" not in value
        or not value.endswith(("Z", "+00:00"))
    ):
        raise ValueError("Expected an explicit UTC deadline")
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() != dt.timedelta(0):
        raise ValueError("Deadline must be UTC")
    return result


def runtime_environment(pod_id, raw=None):
    if not re.fullmatch(r"[a-z0-9]{10,32}", pod_id):
        raise ValueError("Invalid pod identity")
    raw = raw if raw is not None else Path("/proc/1/environ").read_bytes()
    platform = dict(
        item.split("=", 1) for item in raw.decode().split("\0") if "=" in item
    )
    if platform.get("RUNPOD_POD_ID") != pod_id:
        raise ValueError(
            "Refusing guard: platform pod identity differs from receipt"
        )
    if not platform.get("RUNPOD_API_KEY"):
        raise ValueError("Platform-provisioned pod credential is unavailable")
    # The injected credential stays in memory and the child environment only.
    env = os.environ.copy()
    env["RUNPOD_API_KEY"] = platform["RUNPOD_API_KEY"]
    env["RUNPOD_API_URL"] = "https://api.runpod.io/graphql"
    return env


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def config_digest(config):
    """Hash a configuration supplied by a Python caller."""
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def number(value, name, *, minimum=0, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if value < minimum or (positive and value == minimum):
        raise ValueError(f"Invalid {name}")
    return value


def validate_policy(config):
    """Check the allowed runtime and reserved storage cost against the limit."""
    started, deadline = (
        parse_utc(config["created_utc"]),
        parse_utc(config["deadline_utc"]),
    )
    hours = (deadline - started).total_seconds() / 3600
    rate = number(
        config["hourly_rate_upper_bound_usd"], "hourly rate", positive=True
    )
    cap = number(config["max_spend_usd"], "spend cap", positive=True)
    reserve = number(config.get("storage_reserve_usd", 0), "storage reserve")
    if hours <= 0 or cap > 40 or hours * rate + reserve > cap:
        raise ValueError(
            "Deadline and retained-storage reserve exceed the "
            "unchanged $40 cost bound"
        )
    return config


def validate_completion(value):
    required = {
        "status_path",
        "state_path",
        "status_matches",
        "state_matches",
        "lock_paths",
        "required_complete_paths",
    }
    optional = {"grace_seconds", "collection_lease_path"}
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - required - optional
    ):
        raise ValueError(
            "Completion stop needs explicit status/state identity, "
            "locks and result paths"
        )
    for name in ("status_matches", "state_matches"):
        if not isinstance(value[name], dict) or not value[name]:
            raise ValueError(
                "Completion status and state need nonempty identity bindings"
            )
        if any(
            not isinstance(k, str) or not isinstance(v, str) or not v
            for k, v in value[name].items()
        ):
            raise ValueError(
                "Completion identity bindings must be nonempty strings"
            )
    for name in ("lock_paths", "required_complete_paths"):
        if not isinstance(value[name], list) or not value[name]:
            raise ValueError(
                "Completion stop needs execution locks and "
                "completed-result evidence"
            )
    paths = [
        value["status_path"],
        value["state_path"],
        *value["lock_paths"],
        *value["required_complete_paths"],
    ]
    if any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths):
        raise ValueError("Completion evidence paths must be absolute")
    if len(set(value["lock_paths"])) != len(value["lock_paths"]):
        raise ValueError("Completion execution locks must be distinct")
    if number(value.get("grace_seconds", 900), "completion grace") > 3600:
        raise ValueError("Completion grace cannot exceed one hour")
    lease = value.get("collection_lease_path")
    if lease is not None and (
        not isinstance(lease, str) or not Path(lease).is_absolute()
    ):
        raise ValueError("Collection lease path must be absolute")


def validate_override(base, value, base_sha256):
    required = {
        "schema_version",
        "pod_id",
        "created_utc",
        "base_config_sha256",
        "deadline_utc",
        "hourly_rate_upper_bound_usd",
        "max_spend_usd",
        "storage_reserve_usd",
        "reviewed_at_utc",
        "reason",
    }
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError("Unsupported operational override fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported operational override schema")
    if any(
        value[name] != base[name]
        for name in ("pod_id", "created_utc", "max_spend_usd")
    ):
        raise ValueError(
            "Operational override cannot change pod, creation time or spend cap"
        )
    if value["base_config_sha256"] != base_sha256:
        raise ValueError(
            "Operational override belongs to a different original guard config"
        )
    if (
        number(
            value["hourly_rate_upper_bound_usd"], "hourly rate", positive=True
        )
        < base["hourly_rate_upper_bound_usd"]
    ):
        raise ValueError(
            "Operational override cannot reduce the conservative hourly rate"
        )
    if parse_utc(value["reviewed_at_utc"]) < parse_utc(base["created_utc"]):
        raise ValueError("Operational review predates pod creation")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Operational override needs a review reason")
    if "completion_stop" in value:
        validate_completion(value["completion_stop"])
    if "work_deadline_utc" in value:
        work_deadline = parse_utc(value["work_deadline_utc"])
        latest = parse_utc(value["deadline_utc"]) - dt.timedelta(
            seconds=STOP_MARGIN_SECONDS
        )
        if not parse_utc(base["created_utc"]) < work_deadline <= latest:
            raise ValueError(
                "Work deadline must retain the 15-minute stop margin"
            )
    return validate_policy({**base, **value})


@contextmanager
def idle_execution(value):
    """Hold all cooperating execution locks while stopping this instance."""
    with ExitStack() as stack:
        try:
            for path in sorted(value["lock_paths"]):
                # Never create missing locks: that could be a typo or an
                # unstarted controller.
                stream = stack.enter_context(Path(path).open("r+"))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ValueError):
            yield False
            return
        yield True


def terminal_evidence(value):
    """Check completion evidence while all execution locks are held."""
    try:
        status = json.loads(Path(value["status_path"]).read_text())
        state = json.loads(Path(value["state_path"]).read_text())
        for document, bindings in (
            (status, value["status_matches"]),
            (state, value["state_matches"]),
        ):
            if not isinstance(document, dict) or any(
                document.get(k) != v for k, v in bindings.items()
            ):
                return None
        terminal = status.get("status")
        if terminal not in TERMINAL or state.get("terminal") != terminal:
            return None
        if terminal == "complete":
            # The controller publishes these before terminal status and releases
            # its lock last.
            for path in value["required_complete_paths"]:
                if not Path(path).is_file() or Path(path).stat().st_size == 0:
                    return None
        return terminal
    except (OSError, ValueError, TypeError):
        return None


def collection_lease(value, pod_id, now):
    """Delay completed-job shutdown while a valid collection lease remains."""
    try:
        lease = json.loads(Path(value["collection_lease_path"]).read_text())
        if lease.get("pod_id") != pod_id:
            return None
        issued, expires = (
            parse_utc(lease["issued_utc"]),
            parse_utc(lease["expires_utc"]),
        )
        if (
            not issued <= now < expires
            or not 0 < (expires - issued).total_seconds() <= 1800
        ):
            return None
        return {
            "issued_utc": issued.isoformat(),
            "expires_utc": expires.isoformat(),
        }
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def previous_observation(state_path, pod_id, override_sha256):
    try:
        state = json.loads(state_path.read_text())
        if (
            state["pod_id"] == pod_id
            and state.get("operational_override_sha256") == override_sha256
        ):
            return state.get("terminal_observation")
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def cli_commands(binary, run=subprocess.run):
    # The image ships v1; retain support for a later image with v2 command
    # order.
    for get_args, stop_args in (
        (["get", "pod"], ["stop", "pod"]),
        (["pod", "get"], ["pod", "stop"]),
    ):
        result = run(
            [binary, *stop_args, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and "stop" in result.stdout.lower():
            return get_args, stop_args
    raise RuntimeError(
        "Installed runpodctl has no supported own-pod stop command"
    )


def probe(pod_id, run=subprocess.run):
    env = runtime_environment(pod_id)
    binary = shutil.which("runpodctl")
    if not binary:
        raise RuntimeError("Platform runpodctl is unavailable")
    _, stop_args = cli_commands(binary, run)
    # Old runpodctl get lists the account, correctly forbidden by a pod-scoped
    # credential. Verify only this pod directly, without broadening its scope.
    request = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps(
            {
                "query": "query($input: PodFilter!) { pod(input: $input) { id "
                "desiredStatus } }",
                "variables": {"input": {"podId": pod_id}},
            }
        ).encode(),
        headers={
            "Authorization": "Bearer " + env["RUNPOD_API_KEY"],
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise RuntimeError(
            "Pod-scoped authenticated own-pod lookup failed"
        ) from None
    if (
        result.get("errors")
        or ((result.get("data") or {}).get("pod") or {}).get("id") != pod_id
    ):
        raise RuntimeError("Pod-scoped authenticated own-pod lookup failed")
    return binary, stop_args, env


def guard(
    config,
    *,
    once=False,
    check_only=False,
    operational_override=None,
    operational_override_sha256=None,
    base_config_sha256=None,
):
    validate_policy(config)
    pod_id = config["pod_id"]
    started = parse_utc(config["created_utc"])
    if operational_override and not re.fullmatch(
        r"[a-f0-9]{64}", operational_override_sha256 or ""
    ):
        raise ValueError(
            "Operational override requires its reviewed SHA256 pin"
        )
    if operational_override_sha256 and not operational_override:
        raise ValueError("An operational override hash requires its file")
    base_config_sha256 = base_config_sha256 or config_digest(config)
    policy = config
    applied_override_sha256 = None
    state_path = Path(config["state_path"])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binary, stop_args, env = probe(pod_id)
        observed = previous_observation(
            state_path, pod_id, operational_override_sha256
        )
        while True:
            override_error = None
            if operational_override:
                try:
                    raw = Path(operational_override).read_bytes()
                    if (
                        hashlib.sha256(raw).hexdigest()
                        != operational_override_sha256
                    ):
                        raise ValueError(
                            "Operational override differs from its "
                            "reviewed SHA256"
                        )
                    policy = validate_override(
                        config, json.loads(raw), base_config_sha256
                    )
                    applied_override_sha256 = operational_override_sha256
                except (OSError, ValueError, TypeError, KeyError) as error:
                    # A malformed/replaced override must never disable the
                    # original/last valid guard.
                    override_error = type(error).__name__
            deadline = parse_utc(policy["deadline_utc"])
            rate, cap = (
                policy["hourly_rate_upper_bound_usd"],
                policy["max_spend_usd"],
            )
            reserve = policy.get("storage_reserve_usd", 0)
            now = dt.datetime.now(dt.UTC)
            estimated = max(0, (now - started).total_seconds()) / 3600 * rate
            remaining = (deadline - now).total_seconds()
            state = {
                "pod_id": pod_id,
                "guard_pid": os.getpid(),
                "checked_utc": now.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "created_utc": started.isoformat(),
                "seconds_remaining": remaining,
                "estimated_pod_spend_upper_bound_usd": estimated,
                "max_spend_usd": cap,
                "hourly_rate_upper_bound_usd": rate,
                "storage_reserve_usd": reserve,
                "authorized_interval_cost_plus_reserve_usd": (
                    deadline - started
                ).total_seconds()
                / 3600
                * rate
                + reserve,
                "storage_note": (
                    "Volume retained. A finite reserve "
                    "cannot cap storage charges forever."
                ),
                "base_config_sha256": base_config_sha256,
                "operational_override_sha256": applied_override_sha256,
                "credential_source": "platform_injected_pod_scoped",
                "authenticated_own_pod_lookup": True,
                "status": "armed",
                "terminal_observation": observed,
            }
            if override_error:
                state["operational_override_error_type"] = override_error
                state["operational_override_rejected"] = True
            if check_only:
                return {**state, "stop_command_supported": True}
            work_due = "work_deadline_utc" in policy and now >= parse_utc(
                policy["work_deadline_utc"]
            )
            budget_due = (
                estimated + rate * STOP_MARGIN_SECONDS / 3600 + reserve >= cap
            )
            deadline_due = remaining <= STOP_MARGIN_SECONDS or work_due
            due = budget_due or deadline_due

            def request_stop(reason, state=state):
                state["stop_reason"] = reason
                state["status"] = "stop_requested_preserving_volume"
                save(state_path, state)
                # Flush completed output data/metadata before stopping with
                # /workspace retained.
                os.sync()
                try:
                    result = subprocess.run(
                        [binary, *stop_args, pod_id],
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    # A lost response may mean the pod stopped successfully.
                    # Never terminate.
                    state["last_stop_exit_code"] = result.returncode
                except (OSError, subprocess.TimeoutExpired) as error:
                    state["last_stop_error_type"] = type(error).__name__
                state["status"] = "stop_requested_retrying_until_shutdown"

            if due:
                request_stop("budget_bound" if budget_due else "deadline")
            elif completion := policy.get("completion_stop"):
                with idle_execution(completion) as idle:
                    terminal = terminal_evidence(completion) if idle else None
                    if terminal:
                        binding = config_digest(
                            {"completion": completion, "terminal": terminal}
                        )
                        try:
                            first_seen = parse_utc(observed["first_seen_utc"])
                            if (
                                observed["binding"] != binding
                                or first_seen > now
                            ):
                                raise ValueError(
                                    "Different terminal observation"
                                )
                        except (TypeError, KeyError, ValueError):
                            observed = {
                                "binding": binding,
                                "first_seen_utc": now.isoformat(),
                            }
                            first_seen = now
                        state["terminal_observation"] = observed
                        grace_remaining = (
                            completion.get("grace_seconds", 900)
                            - (now - first_seen).total_seconds()
                        )
                        lease = collection_lease(completion, pod_id, now)
                        if grace_remaining > 0:
                            state["completion_stop_waiting_for"] = (
                                "collection_grace"
                            )
                            state["completion_grace_remaining_seconds"] = (
                                grace_remaining
                            )
                        elif lease:
                            state["completion_stop_waiting_for"] = (
                                "collection_lease"
                            )
                            state["active_collection_lease"] = lease
                        else:
                            due = True
                            state["execution_locks_held_for_stop"] = completion[
                                "lock_paths"
                            ]
                            request_stop("terminal_" + terminal)
                    else:
                        observed = None
                        state["terminal_observation"] = None
                        state["completion_stop_waiting_for"] = (
                            "terminal_evidence" if idle else "execution_locks"
                        )
            save(state_path, state)
            if once:
                return state
            time.sleep(30 if due else 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--operational-override", type=Path)
    parser.add_argument("--operational-override-sha256")
    args = parser.parse_args()
    raw = Path(args.config).read_bytes()
    result = guard(
        json.loads(raw),
        check_only=args.check_only,
        base_config_sha256=hashlib.sha256(raw).hexdigest(),
        operational_override=args.operational_override,
        operational_override_sha256=args.operational_override_sha256,
    )
    if result:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
