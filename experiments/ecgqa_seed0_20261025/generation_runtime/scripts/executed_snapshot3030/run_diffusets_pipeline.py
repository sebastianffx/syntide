"""Run the frozen DiffuSETS extension stages under one inherited lock and deadline.

Usage: python scripts/runpod/run_diffusets_pipeline.py --config pipeline-config.json \
    --config-sha256 <sha256>

Commands are explicit argv arrays with separate cwd/env settings. A completed preflight is
reused only when its input/output hashes match. With reuse_completed_generation=true, an
adopted complete generation receipt is also reused; failed or partial generation is never
redispatched in that mode. Training/evaluation is never retried after dispatch.

--operational-override PATH --operational-override-sha256 SHA accepts a separately bound
operational time extension while preserving the scientific config and $40 spending cap.
An external guard must enforce that cap; the pipeline enforces the work deadline.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
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


OVERRIDE_PATH_ENV = "SYNTIDE_OPERATIONAL_DEADLINE_OVERRIDE_PATH"
OVERRIDE_SHA_ENV = "SYNTIDE_OPERATIONAL_DEADLINE_OVERRIDE_SHA256"


def utc_timestamp(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an explicit UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field} must be an explicit UTC timestamp")
    return parsed.timestamp()


class OperationalDeadline:
    """An immutable, hash-pinned operational extension of a frozen scientific deadline.

    Revalidation reads only small JSON provenance files. The caller must separately enforce
    the unchanged spending cap; this object never authorizes retries or alters study identity.
    """

    def __init__(self, path, expected_sha256, *, config=None, config_sha256=None, protocol=None):
        if not path or not expected_sha256:
            raise ValueError("Operational override path and SHA256 are both required")
        self.path = Path(path)
        if not self.path.is_absolute() or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError("Operational override requires an absolute path and valid SHA256")
        self.expected_sha256 = expected_sha256
        self.config, self.config_sha256, self.protocol = config, config_sha256, protocol
        self.read()

    def read(self):
        if sha(self.path) != self.expected_sha256:
            raise ValueError("Operational override SHA256 changed")
        override = read(self.path)
        if (
            override.get("schema_version") != 1
            or override.get("kind", "operational_deadline_extension") != "operational_deadline_extension"
        ):
            raise ValueError("Unsupported operational deadline override")
        bound = {}
        for name, path_field, hash_field in (
            ("pipeline_config", "pipeline_config_path", "pipeline_config_sha256"),
            ("extension_protocol", "protocol_path", "protocol_sha256"),
            ("pod_receipt", "receipt_path", "receipt_sha256"),
            ("guard_config", "base_config_path", "base_config_sha256"),
        ):
            path, expected = override.get(path_field), override.get(hash_field)
            if (
                not isinstance(path, str)
                or not Path(path).is_absolute()
                or not isinstance(expected, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None
                or sha(path) != expected
            ):
                raise ValueError(f"Operational override {name} binding differs")
            bound[name] = read(path)
        config, protocol, receipt = (
            bound[k] for k in ("pipeline_config", "extension_protocol", "pod_receipt")
        )
        if self.config is not None:
            validate_operational_config(self.config, config, self)
            if self.config == config and override["pipeline_config_sha256"] != self.config_sha256:
                raise ValueError("Operational override belongs to a different frozen pipeline config")
        if self.protocol is not None and protocol != self.protocol:
            raise ValueError("Operational override belongs to a different frozen extension protocol")
        original = utc_timestamp(config.get("deadline_utc"), "pipeline deadline")
        if (
            utc_timestamp(protocol.get("evaluation_budget", {}).get("deadline_utc"), "protocol deadline")
            != original
        ):
            raise ValueError("Operational override original deadlines differ")
        extended = utc_timestamp(override.get("work_deadline_utc"), "work_deadline_utc")
        absolute = utc_timestamp(override.get("deadline_utc"), "deadline_utc")
        created = utc_timestamp(override.get("created_utc"), "created_utc")
        reviewed = utc_timestamp(override.get("reviewed_at_utc"), "reviewed_at_utc")
        if extended <= original or created >= extended or reviewed >= extended or absolute < extended:
            raise ValueError(
                "Operational override must extend the original deadline within the absolute stop"
            )
        if (
            not override.get("authorization")
            or not isinstance(override.get("reason"), str)
            or not override["reason"].strip()
        ):
            raise ValueError("Operational override requires user authorization and reason")
        if not isinstance(override.get("completion_stop"), dict) or not override["completion_stop"]:
            raise ValueError("Operational override must retain stop on completion")
        pod_id = override.get("pod_id")
        if (
            not isinstance(pod_id, str)
            or not pod_id
            or pod_id != receipt.get("pod", {}).get("id")
            or (os.environ.get("RUNPOD_POD_ID") and pod_id != os.environ["RUNPOD_POD_ID"])
        ):
            raise ValueError("Operational override pod identity differs from receipt or current pod")
        cost = receipt.get("cost_plan", {})
        cap, rate = override.get("max_spend_usd"), override.get("hourly_rate_upper_bound_usd")
        reserve = override.get("storage_reserve_usd")
        if type(cap) not in (int, float) or cap != 40 or cost.get("max_spend_usd") != cap:
            raise ValueError("Operational override must preserve the receipt's $40 spending cap")
        if (
            type(rate) not in (int, float)
            or not 0 < rate < float("inf")
            or rate != cost.get("hourly_rate_upper_bound_usd")
        ):
            raise ValueError("Operational override hourly rate differs from receipt")
        started = utc_timestamp(receipt.get("created_utc"), "receipt created_utc")
        if (
            type(reserve) not in (int, float)
            or not 0 <= reserve < cap
            or absolute <= started
            or (absolute - started) / 3600 * rate + reserve > cap + 1e-9
        ):
            raise ValueError(
                "Operational override projected cost plus storage exceeds unchanged spending cap"
            )
        prior = override.get("previous_pipeline_source_sha256")
        source = [
            path for path in config.get("bindings", {}) if Path(path).name == "run_diffusets_pipeline.py"
        ]
        if len(source) != 1 or config["bindings"][source[0]] != prior or sha(source[0]) != prior:
            raise ValueError("Operational override original pipeline source binding differs")
        return override

    def timestamp(self):
        return utc_timestamp(self.read()["work_deadline_utc"], "work_deadline_utc")

    def environment(self):
        self.read()
        return {OVERRIDE_PATH_ENV: str(self.path), OVERRIDE_SHA_ENV: self.expected_sha256}

    def audit(self):
        return {"path": str(self.path), "sha256": self.expected_sha256, "override": self.read()}


def validate_operational_config(current, original, override):
    """Permit only relocation of orchestration helpers and explicit override arguments.

    A wrapper must still audit and migrate the execution state; this check cannot authorize
    any replay or replacement of scientific inputs, generator, trainer, or evaluator.
    """
    permitted = {"bindings", "commands", "reuse_completed_generation"}
    if {k: v for k, v in current.items() if k not in permitted} != {
        k: v for k, v in original.items() if k not in permitted
    }:
        raise ValueError("Operational config changes frozen scientific or execution settings")
    if type(current.get("reuse_completed_generation", False)) is not bool:
        raise ValueError("reuse_completed_generation must be boolean")
    mutable = {"run_diffusets_pipeline.py", "run_diffusets_arm.py"}
    copied = {"run_qa_arms.py", "qa_preflight.py"}
    allowed = mutable | copied
    old_bindings, new_bindings = original["bindings"], current["bindings"]
    old_fixed = {p: h for p, h in old_bindings.items() if Path(p).name not in allowed}
    new_fixed = {
        p: h
        for p, h in new_bindings.items()
        if Path(p).name not in allowed
        and not (p not in old_bindings and Path(p).is_relative_to(override.path.parent))
    }
    if new_fixed != old_fixed:
        raise ValueError("Operational config changes scientific/source input bindings")
    if str(override.path) in new_bindings and new_bindings[str(override.path)] != override.expected_sha256:
        raise ValueError("Operational config override binding differs")
    for name in allowed:
        old = {p: h for p, h in old_bindings.items() if Path(p).name == name}
        new = {p: h for p, h in new_bindings.items() if Path(p).name == name}
        if len(old) != 1 or not 1 <= len(new) <= 2:
            raise ValueError("Operational helper binding is missing or ambiguous")
        for path, digest_value in new.items():
            if (
                (path in old and digest_value != old[path])
                or (path not in old and not Path(path).is_relative_to(override.path.parent))
                or (name in copied and digest_value != next(iter(old.values())))
            ):
                raise ValueError("Operational helper changes original source or a baseline helper")
    if set(current["commands"]) != set(original["commands"]):
        raise ValueError("Operational config changes stage set")
    for name, old in original["commands"].items():
        new = current["commands"][name]
        if {k: v for k, v in new.items() if k != "argv"} != {k: v for k, v in old.items() if k != "argv"}:
            raise ValueError(f"Operational config changes {name} environment, inputs, or outputs")
        argv = list(new["argv"])
        for option, expected in (
            ("--operational-override", str(override.path)),
            ("--operational-override-sha256", override.expected_sha256),
        ):
            if option in argv:
                index = argv.index(option)
                if (
                    argv.count(option) != 1
                    or argv[index + 1 : index + 2] != [expected]
                    or name not in {"preflight", "run"}
                ):
                    raise ValueError("Operational override command arguments differ")
                del argv[index : index + 2]
        if name in {"preflight", "run"}:
            for index, value in enumerate(old["argv"]):
                if Path(value).name == "run_diffusets_arm.py" and index < len(argv):
                    replacement = argv[index]
                    if Path(replacement).name != "run_diffusets_arm.py" or replacement not in new_bindings:
                        raise ValueError("Operational arm executable is not source-bound")
                    argv[index] = value
        if argv != old["argv"]:
            raise ValueError(f"Operational config changes frozen {name} command")


def deadline_override(path=None, expected_sha256=None, **context):
    if path is None and expected_sha256 is None:
        return None
    return OperationalDeadline(path, expected_sha256, **context)


def record_override(root, override):
    if override is None:
        return
    path = Path(root) / "operational-overrides" / (override.expected_sha256 + ".json")
    audit = override.audit()
    if path.exists():
        if read(path) != audit:
            raise ValueError("Existing operational override audit differs")
    else:
        write(path, audit)


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
                if time.time() >= (deadline() if callable(deadline) else deadline):
                    raise DeadlineReached("Absolute pod deadline reached during active stage")
                time.sleep(poll_seconds)
            return child.returncode
        except BaseException:
            stop_child(child)
            raise


def run_pipeline(config, config_sha256, *, child_runner=run_child, operational_override=None):
    validate(config)
    root = Path(config["run_dir"])
    state_path, status_path = root / "pipeline-state.json", root / "status.json"
    identity = digest(
        {"config": config, "config_sha256": config_sha256, "pipeline_source_sha256": sha(__file__)}
    )
    deadline = operational_override.timestamp if operational_override else lambda: deadline_timestamp(config)
    with lock(root / ".pipeline.lock") as lock_fd:
        record_override(root, operational_override)
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
                "deadline_utc": datetime.fromtimestamp(deadline(), UTC).isoformat(),
                "scientific_deadline_utc": config["deadline_utc"],
                **({"operational_override": operational_override.audit()} if operational_override else {}),
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
                remaining = deadline() - time.time()
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
                if name == "generate" and config.get("reuse_completed_generation", False):
                    receipt = state["completed"].get(name)
                    if receipt:
                        if not preflight_reusable(receipt, bound_identity):
                            raise ValueError(
                                "Completed generation receipt or outputs changed; refusing replay"
                            )
                        status("generating", reused_completed_generation=True)
                        continue
                    if state.get("generation_started"):
                        raise ValueError(
                            "Incomplete or failed generation requires explicit recovery; refusing replay"
                        )
                    state["generation_started"] = {"at": now_utc(), "command_identity": bound_identity}
                    write(state_path, state)
                if name == "preflight" and preflight_reusable(state["completed"].get(name), bound_identity):
                    status("preflight", reused_completed_preflight=True)
                    continue
                status(phase)
                if name == "run":
                    state["run_started"] = {"at": now_utc(), "command_identity": bound_identity}
                    write(state_path, state)
                dispatch = command
                if operational_override:
                    dispatch = {
                        **command,
                        "env": {**command.get("env", {}), **operational_override.environment()},
                    }
                returncode = child_runner(
                    dispatch,
                    root / "logs" / f"pipeline-{name}.log",
                    lock_fd,
                    deadline if operational_override else deadline(),
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
    parser.add_argument("--operational-override", default=os.environ.get(OVERRIDE_PATH_ENV))
    parser.add_argument("--operational-override-sha256", default=os.environ.get(OVERRIDE_SHA_ENV))
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("Pipeline config file checksum mismatch")

    def interrupted(signum, frame):
        raise PipelineInterrupted(f"Pipeline received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    config = read(args.config)
    override = deadline_override(
        args.operational_override,
        args.operational_override_sha256,
        config=config,
        config_sha256=args.config_sha256,
    )
    result = run_pipeline(config, args.config_sha256, operational_override=override)
    raise SystemExit(1 if result["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
