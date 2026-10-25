"""Run the frozen DiffuSETS extension stages under one inherited lock and deadline.

Usage: python scripts/runpod/run_diffusets_pipeline.py --config pipeline-config.json \
    --config-sha256 <sha256>

Commands are explicit argv arrays with separate cwd/env settings. Generation and preparation
always re-enter their own immutable recovery logic. Only a completed preflight with matching
input/output hashes is skipped. A failed or interrupted training/evaluation is never retried here.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

STAGES = (
    ("generate", "generating"),
    ("prepare", "preparing"),
    ("preflight", "preflight"),
    ("run", "training_or_evaluation"),
)
REVIEW = {"needs_budget_review", "needs_time_budget_review"}


class PipelineInterrupted(Exception):
    pass


class DeadlineReached(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def now_utc():
    return datetime.now(UTC).isoformat()


def deadline_timestamp(config):
    value = datetime.fromisoformat(config["deadline_utc"].replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("deadline_utc must include an explicit UTC offset")
    return value.timestamp()


def write(path, value):
    """Atomically update only this pipeline's mutable state/status files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def bind_files(paths):
    return {str(path): sha(path) for path in paths}


def verify_bindings(bindings):
    for path, expected in bindings.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise ValueError(f"Frozen input/source file differs: {path}")


def validate(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported pipeline config schema")
    deadline_timestamp(config)
    if (
        type(config.get("min_training_remaining_seconds")) not in (int, float)
        or config["min_training_remaining_seconds"] <= 0
    ):
        raise ValueError("A positive conservative training/evaluation time allowance is required")
    for name in ("run_dir", "runner_status_path", "runner_result_path"):
        if not isinstance(config.get(name), str) or not Path(config[name]).is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    if set(config.get("commands", {})) != {name for name, _ in STAGES}:
        raise ValueError("Exactly generate, prepare, preflight and run commands are required")
    for name, command in config["commands"].items():
        if (
            not isinstance(command.get("argv"), list)
            or not command["argv"]
            or any(not isinstance(v, str) for v in command["argv"])
        ):
            raise ValueError(f"{name}: argv must be a nonempty string array")
        if not Path(command["argv"][0]).is_absolute() or not Path(command["cwd"]).is_absolute():
            raise ValueError(f"{name}: executable and cwd must be absolute")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in command.get("env", {}).items()):
            raise ValueError(f"{name}: environment overrides must be strings")
        for path in [*command.get("inputs", []), *command.get("outputs", [])]:
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError(f"{name}: bound inputs/outputs must be absolute paths")
    if not config.get("bindings") or any(not Path(path).is_absolute() for path in config["bindings"]):
        raise ValueError("Frozen source/input bindings with absolute paths are required")
    if not config["commands"]["preflight"].get("inputs") or not config["commands"]["preflight"].get(
        "outputs"
    ):
        raise ValueError("Preflight needs explicit inputs and outputs for safe caching")
    verify_bindings(config["bindings"])


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pipeline or inherited child still holds the execution lock") from None
        try:
            yield stream.fileno()
        finally:
            # Closing our descriptor releases the lock only after every inherited child closes it.
            # Explicit LOCK_UN would release it while a surviving child still owns the descriptor.
            pass


def command_identity(command, bindings):
    return digest(
        {"command": command, "frozen_bindings": bindings, "inputs": bind_files(command.get("inputs", []))}
    )


def preflight_reusable(receipt, identity):
    if not receipt or receipt.get("command_identity") != identity or receipt.get("status") != "complete":
        return False
    try:
        verify_bindings(receipt["outputs"])
    except (ValueError, FileNotFoundError):
        return False
    return True


def stop_child(child):
    """Stop the entire current stage's process group and wait for evidence to flush."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def run_child(command, log_path, lock_fd, deadline, on_start, *, poll_seconds=1):
    environment = {**os.environ, **command.get("env", {})}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as output:
        child = subprocess.Popen(
            command["argv"],
            cwd=command["cwd"],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lock_fd,),
        )
        try:
            on_start(child.pid)
            while child.poll() is None:
                if time.time() >= deadline:
                    raise DeadlineReached("Absolute pod deadline reached during active stage")
                time.sleep(poll_seconds)
            return child.returncode
        except BaseException:
            stop_child(child)
            raise


def run_pipeline(config, config_sha256, *, child_runner=run_child):
    validate(config)
    root = Path(config["run_dir"])
    state_path, status_path = root / "pipeline-state.json", root / "status.json"
    identity = digest(
        {"config": config, "config_sha256": config_sha256, "pipeline_source_sha256": sha(__file__)}
    )
    deadline = deadline_timestamp(config)
    with lock(root / ".pipeline.lock") as lock_fd:
        if state_path.exists():
            state = read(state_path)
            if state["identity"] != identity:
                raise ValueError("Frozen pipeline config or source changed; refusing to resume")
        else:
            state = {
                "identity": identity,
                "config_sha256": config_sha256,
                "started_at": now_utc(),
                "completed": {},
            }
            frozen_config = root / "pipeline-config.frozen.json"
            if frozen_config.exists() and read(frozen_config) != config:
                raise ValueError("Existing frozen pipeline config differs")
            if not frozen_config.exists():
                write(frozen_config, config)
            write(state_path, state)

        def status(value, **details):
            payload = {
                "status": value,
                "pipeline_identity": identity,
                "updated_at": now_utc(),
                "deadline_utc": config["deadline_utc"],
                **details,
            }
            write(status_path, payload)
            print(json.dumps(payload), flush=True)
            return payload

        if state.get("terminal") == "complete":
            if state["completed"]["run"]["command_identity"] != command_identity(
                config["commands"]["run"], config["bindings"]
            ):
                raise ValueError("Completed pipeline training inputs changed")
            verify_bindings(state["completed"]["run"]["outputs"])
            runner_state = read(config["runner_status_path"])
            if runner_state.get("status") != "complete" or not Path(config["runner_result_path"]).is_file():
                raise ValueError("Completed pipeline no longer has its bound completed runner result")
            return status("complete", resumed=True)
        if state.get("run_started") and state.get("terminal") != "complete":
            # Never automatically invoke training/evaluation again after an uncertain attempt.
            # The child runner's own recovery is an additional guard, not permission to retry tests.
            prior = state.get("terminal", "failed")
            terminal = prior if prior in REVIEW else "failed"
            return status(
                terminal,
                reason="Training/evaluation was already dispatched; explicit recovery review is required",
            )
        phase = "initializing"
        try:
            for name, phase in STAGES:
                verify_bindings(config["bindings"])
                remaining = deadline - time.time()
                minimum = config["min_training_remaining_seconds"] if name == "run" else 0
                if remaining <= minimum:
                    state["terminal"] = "needs_time_budget_review"
                    write(state_path, state)
                    return status(
                        "needs_time_budget_review",
                        stage=phase,
                        remaining_seconds=max(0, remaining),
                        required_seconds=minimum,
                    )
                command = config["commands"][name]
                bound_identity = command_identity(command, config["bindings"])
                if name == "preflight" and preflight_reusable(state["completed"].get(name), bound_identity):
                    status("preflight", reused_completed_preflight=True)
                    continue
                status(phase)
                if name == "run":
                    state["run_started"] = {"at": now_utc(), "command_identity": bound_identity}
                    write(state_path, state)
                returncode = child_runner(
                    command,
                    root / "logs" / f"pipeline-{name}.log",
                    lock_fd,
                    deadline,
                    lambda pid, current_phase=phase: status(current_phase, child_pid=pid),
                )
                if returncode != 0:
                    raise RuntimeError(f"Stage {name} exited {returncode}; see pipeline-{name}.log")
                if name == "run":
                    runner_state = read(config["runner_status_path"])
                    runner_status = runner_state.get("status")
                    if runner_status in REVIEW:
                        state["terminal"] = runner_status
                        write(state_path, state)
                        return status(runner_status, runner_status=runner_state)
                    if runner_status != "complete" or not Path(config["runner_result_path"]).is_file():
                        raise RuntimeError("Runner exited without complete status and extension_result.json")
                outputs = bind_files(command.get("outputs", []))
                if name == "run":
                    outputs.update(bind_files([config["runner_status_path"], config["runner_result_path"]]))
                state["completed"][name] = {
                    "status": "complete",
                    "command_identity": bound_identity,
                    "outputs": outputs,
                }
                write(state_path, state)
            state["terminal"] = "complete"
            write(state_path, state)
            return status("complete")
        except DeadlineReached as error:
            terminal = "needs_time_budget_review"
            state["terminal"] = terminal
            write(state_path, state)
            return status(terminal, stage=phase, reason=str(error))
        except (Exception, KeyboardInterrupt) as error:
            state["terminal"] = "failed"
            write(state_path, state)
            return status("failed", stage=phase, reason=f"{type(error).__name__}: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("Pipeline config file checksum mismatch")

    def interrupted(signum, frame):
        raise PipelineInterrupted(f"Pipeline received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = run_pipeline(read(args.config), args.config_sha256)
    raise SystemExit(1 if result["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
