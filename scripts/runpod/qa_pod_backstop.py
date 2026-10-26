#!/usr/bin/env python3
"""Enforce a deadline for the original question-answering training instance.

Run with --check-only before starting the independent monitor. Authentication
uses the platform-injected credential through diffusets_guard. An account
credential must never be saved in the configuration.

Budget fields describe an account snapshot that includes overlapping instances.
The hourly upper bound must exceed the observed total rate by at least $0.05.
This account estimate does not measure one instance's cost.

A collector can record a temporary collection lease in
``/workspace/qa-backstop/collection-lease.json``. Each lease lasts at most
1,800 seconds. Together
they may defer shutdown for at most 2,700 seconds after a terminal status and a
free execution lock were first observed. Deadline and spending limits always
take precedence.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import diffusets_guard

POD_ID = "y9z0q4fgr4zvn6"
ORIGINAL_DEADLINE = "2026-10-24T01:42:38.340Z"
STUDY = Path("/workspace/qa-study")
ROOT = Path("/workspace/qa-backstop")
TERMINAL = {
    "complete",
    "failed",
    "needs_budget_review",
    "needs_time_budget_review",
}
GRACE_SECONDS = 900
MAX_LEASE_SECONDS = 1800
BUDGET_MARGIN_USD = 0.5
RATE_MARGIN_USD = 0.05
CONFIG_FIELDS = {
    "pod_id",
    "deadline_utc",
    "budget_anchor_utc",
    "observed_account_delta_usd",
    "observed_total_hourly_spend_usd",
    "hourly_spend_upper_bound_usd",
    "max_spend_usd",
}


def utc_now():
    return dt.datetime.now(dt.UTC)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(config, now):
    if set(config) != CONFIG_FIELDS or config["pod_id"] != POD_ID:
        raise ValueError("Only the original receipt-owned QA pod is supported")
    deadline = diffusets_guard.parse_utc(config["deadline_utc"])
    if deadline > diffusets_guard.parse_utc(ORIGINAL_DEADLINE):
        raise ValueError("Original deadline cannot be extended")
    anchor = diffusets_guard.parse_utc(config["budget_anchor_utc"])
    if anchor > now + dt.timedelta(seconds=30):
        raise ValueError("Budget anchor is in the future")
    for key in CONFIG_FIELDS - {"pod_id", "deadline_utc", "budget_anchor_utc"}:
        if not number(config[key]) or config[key] < 0:
            raise ValueError("Budget fields must be finite nonnegative numbers")
    if not 0 < config["max_spend_usd"] <= 50:
        raise ValueError("Original account-delta cap cannot exceed $50")
    if (
        config["hourly_spend_upper_bound_usd"] + 1e-12
        < config["observed_total_hourly_spend_usd"] + RATE_MARGIN_USD
    ):
        raise ValueError(
            "Hourly bound lacks the required margin above observed "
            "total billing"
        )
    return deadline, anchor


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def terminal_execution_lock(study):
    """Hold the runner lock while requesting completed-job shutdown."""
    status_path, lock_path = study / "status.json", study / ".run.lock"
    try:
        status = json.loads(status_path.read_text()).get("status")
    except (OSError, ValueError, AttributeError):
        yield None, False
        return
    if status not in TERMINAL:
        yield status, False
        return
    try:
        stream = lock_path.open(
            "r+"
        )  # Missing lock is not evidence of safe completion.
    except OSError:
        yield status, False
        return
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield status, False
            return
        # Re-read while holding the lock in case a runner changed status
        # meanwhile.
        try:
            status = json.loads(status_path.read_text()).get("status")
        except (OSError, ValueError, AttributeError):
            status = None
        yield status, status in TERMINAL


def lease_deadline(path, now, eligible_since):
    if not path.exists():
        return None, "absent"
    try:
        lease = json.loads(path.read_text())
        if (
            set(lease) != {"pod_id", "issued_utc", "expires_utc"}
            or lease["pod_id"] != POD_ID
        ):
            raise ValueError
        issued = diffusets_guard.parse_utc(lease["issued_utc"])
        expires = diffusets_guard.parse_utc(lease["expires_utc"])
        if (
            issued > now + dt.timedelta(seconds=30)
            or not 0 < (expires - issued).total_seconds() <= 1800
        ):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        return None, "invalid_ignored"
    absolute_limit = eligible_since + dt.timedelta(
        seconds=GRACE_SECONDS + MAX_LEASE_SECONDS
    )
    bounded = min(expires, absolute_limit)
    return (bounded, "active") if bounded > now else (None, "expired")


def request_stop(state, state_path, authentication, *, run=subprocess.run):
    binary, stop_args, env = authentication
    state["status"] = "stop_requested_preserving_volume"
    try:
        save(state_path, state)
    except OSError as error:
        state["state_write_error_type"] = type(error).__name__
    try:
        # State is fsynced above; flush study data before requesting a
        # preserving stop.
        flushed = run(["sync"], capture_output=True, text=True, timeout=30)
        state["sync_exit_code"] = flushed.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        state["sync_error_type"] = type(error).__name__
    try:
        result = run(
            [binary, *stop_args, POD_ID],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        state["last_stop_exit_code"] = result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        # Never store exceptions, stdout, stderr, command environments, or
        # credentials.
        state["last_stop_error_type"] = type(error).__name__
    state["status"] = "stop_requested_retrying_until_shutdown"
    try:
        save(state_path, state)
    except OSError as error:
        state["state_write_error_type"] = type(error).__name__
    return state


def poll(
    config,
    prior,
    authentication,
    *,
    now=None,
    root=None,
    study=None,
    run=subprocess.run,
):
    now = now or utc_now()
    root, study = root or ROOT, study or STUDY
    deadline, anchor = validate(config, now)
    estimate = (
        config["observed_account_delta_usd"]
        + max(0, (now - anchor).total_seconds())
        / 3600
        * config["hourly_spend_upper_bound_usd"]
    )
    if not math.isfinite(estimate):
        estimate = config[
            "max_spend_usd"
        ]  # An overflowing bound requires a preserving stop.
    state_path = root / "state.json"
    state = {
        "pod_id": POD_ID,
        "guard_pid": os.getpid(),
        "checked_utc": now.isoformat(),
        "deadline_utc": deadline.isoformat(),
        "max_spend_usd": config["max_spend_usd"],
        "budget_anchor_utc": anchor.isoformat(),
        "estimated_account_delta_upper_bound_usd": estimate,
        "budget_includes_overlapping_pods": True,
        "credential_source": "platform_injected_pod_scoped",
        "authenticated_own_pod_lookup": True,
        "status": "armed",
        "eligible_terminal_utc": None,
    }
    risk = (
        "absolute_deadline"
        if now >= deadline
        else (
            "account_delta_budget_margin"
            if estimate >= config["max_spend_usd"] - BUDGET_MARGIN_USD
            else None
        )
    )
    if risk:
        state["stop_reason"] = risk
        return request_stop(state, state_path, authentication, run=run)
    with terminal_execution_lock(study) as (terminal, eligible):
        state.update(
            runner_status=terminal, terminal_execution_lock_free=eligible
        )
        if eligible:
            since = now
            if prior.get("runner_status") == terminal and prior.get(
                "eligible_terminal_utc"
            ):
                since = diffusets_guard.parse_utc(
                    prior["eligible_terminal_utc"]
                )
            state["eligible_terminal_utc"] = since.isoformat()
            grace = since + dt.timedelta(seconds=GRACE_SECONDS)
            lease, lease_status = lease_deadline(
                root / "collection-lease.json", now, since
            )
            state["collection_lease_status"] = lease_status
            stop_after = max(grace, lease or grace)
            state["terminal_stop_after_utc"] = stop_after.isoformat()
            if now >= stop_after:
                state["stop_reason"] = (
                    "terminal_execution_finished_collection_grace_elapsed"
                )
                return request_stop(state, state_path, authentication, run=run)
    save(state_path, state)
    return state


def guard(config, *, check_only=False, once=False):
    validate(config, utc_now())
    if check_only:
        diffusets_guard.probe(POD_ID)
        return {
            "pod_id": POD_ID,
            "status": "read_only_probe_passed",
            "account_key_transferred": False,
        }
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "guard.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        authentication = diffusets_guard.probe(POD_ID)
        path = ROOT / "state.json"
        prior = json.loads(path.read_text()) if path.exists() else {}
        if prior and prior.get("pod_id") != POD_ID:
            raise ValueError("Existing state belongs to another pod")
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        if prior.get("config_sha256") not in (None, config_hash):
            raise ValueError(
                "Existing backstop configuration differs; explicit "
                "review required"
            )
        while True:
            try:
                state = poll(config, prior, authentication)
                state["config_sha256"] = config_hash
                save(path, state)
            except OSError as error:
                # A transient status/storage failure must not permanently
                # disable the timer.
                state = {
                    **prior,
                    "status": "state_io_error",
                    "error_type": type(error).__name__,
                }
                print(
                    json.dumps(
                        {
                            "status": "state_io_error",
                            "error_type": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
            if once:
                return state
            prior = state
            time.sleep(
                30 if state["status"].startswith("stop_requested") else 60
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        result = guard(
            json.loads(args.config.read_text()),
            check_only=args.check_only,
            once=args.once,
        )
    except Exception as error:
        print(
            json.dumps(
                {"status": "backstop_error", "error_type": type(error).__name__}
            ),
            flush=True,
        )
        raise SystemExit(1) from None
    if result:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
