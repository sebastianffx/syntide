#!/usr/bin/env python3
"""Attach a controller to an existing DiffuSETS generation process.

The generation child keeps its execution lock while its old controller is
replaced. All 5,000 published candidates must pass verification before the
replacement pipeline proceeds. This command never starts generation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError("Existing adoption evidence differs")
    else:
        write(path, value)


def process(pid, proc_root=Path("/proc")):
    directory = proc_root / str(pid)
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": int(pid),
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
            "argv": [
                p.decode()
                for p in (directory / "cmdline").read_bytes().split(b"\0")
                if p
            ],
        }
    except FileNotFoundError:
        return None


def require_process(expected):
    current = process(expected["pid"])
    if current is None or any(
        current[k] != expected[k]
        for k in ("pid", "start_ticks", "argv", "pgid", "sid")
    ):
        raise ValueError("Process identity changed; refusing handoff")
    return current


def require_inherited_locks(pid, paths, proc_root=Path("/proc")):
    """Verify the child owns the execution locks before removing its parent."""
    for path in paths:
        expected = Path(path).stat()
        found = False
        for fd in (proc_root / str(pid) / "fd").iterdir():
            try:
                info = fd.stat()
            except (OSError, ValueError):
                continue
            if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
                found = True
                break
        if not found:
            raise ValueError(
                "Generator does not hold the expected inherited execution lock"
            )
        # NFS-backed /workspace omits flock ownership in /proc/fdinfo. Verify
        # the
        # exact open inode plus actual lock exclusion; the pinned parent's
        # source
        # proves pass_fds, and the deployment smoke tests survival on this
        # filesystem.
        with Path(path).open("a+") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            raise ValueError("Expected execution lock is not exclusively held")


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def verify_generation(root, plan, identity_sha256, expected_count=5000):
    root = Path(root)
    if sha(root / "identity.json") != identity_sha256:
        raise ValueError("Generator identity changed during adoption")
    identity, summary = (
        read(root / "identity.json"),
        read(root / "summary.json"),
    )
    rows = plan["candidates"]
    if len(rows) != expected_count or [r["slot_id"] for r in rows] != list(
        range(expected_count)
    ):
        raise ValueError("Candidate plan is not the exact fixed slot budget")
    if len({r["seed"] for r in rows}) != expected_count:
        raise ValueError("Candidate seeds are not unique")
    if (
        identity["plan_sha256"] != digest(plan)
        or identity["count"] != expected_count
    ):
        raise ValueError("Generation identity does not bind the plan")
    if (
        summary["generation_id"] != identity["generation_id"]
        or summary["candidate_count"] != expected_count
    ):
        raise ValueError("Generation summary identity/count mismatch")
    if len(summary["slots"]) != expected_count:
        raise ValueError("Generation summary is incomplete")
    if (
        sha(root / "measurement-summary.json")
        != summary["measurement_summary_sha256"]
    ):
        raise ValueError("Measurement summary checksum differs")
    if {p.name for p in (root / "candidates").iterdir()} != {
        f"{i:05d}" for i in range(expected_count)
    }:
        raise ValueError("Missing or additional candidate directories")
    accepted = 0
    for row, slot in zip(rows, summary["slots"], strict=True):
        directory = root / "candidates" / f"{row['slot_id']:05d}"
        if directory.is_symlink() or any(
            p.is_symlink() for p in directory.rglob("*")
        ):
            raise ValueError("Published candidate contains a symlink")
        result = read(directory / "result.json")
        if (
            slot["slot_id"] != row["slot_id"]
            or sha(directory / "result.json") != slot["result_sha256"]
        ):
            raise ValueError("Published result checksum differs")
        if any(result[k] != row[k] for k in ("slot_id", "seed")) or result[
            "candidate_sha256"
        ] != digest(row):
            raise ValueError("Published candidate identity differs")
        if result["generation_id"] != identity["generation_id"]:
            raise ValueError("Published candidate belongs to another run")
        actual_files = {
            str(p.relative_to(directory))
            for p in directory.rglob("*")
            if p.is_file()
        }
        if actual_files != set(result["artifacts"]) | {"result.json"}:
            raise ValueError(
                "Published candidate has missing or extra artifacts"
            )
        for relative, expected in result["artifacts"].items():
            path = directory / relative
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(directory.resolve())
                or sha(path) != expected
            ):
                raise ValueError("Published candidate artifact differs")
        accepted += bool(result["accepted"])
    if (
        accepted != summary["accepted"]
        or summary["rejected"] != expected_count - accepted
    ):
        raise ValueError("Acceptance counts differ")
    return {
        "candidate_count": expected_count,
        "accepted": accepted,
        "rejected": expected_count - accepted,
        "summary_sha256": sha(root / "summary.json"),
        "identity_sha256": identity_sha256,
    }


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location("adopted_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(config):
    root = Path(config["study_root"])
    operation = Path(config["operation_root"])
    state_path, status_path = (
        operation / "control-state.json",
        operation / "control-status.json",
    )
    identity = config["controller_identity"]
    for path, expected in config["bindings"].items():
        if sha(path) != expected:
            raise ValueError("Adoption input/source checksum differs")
    override = read(config["override_path"])
    if (
        sha(config["override_path"]) != config["override_sha256"]
        or override["pod_id"] != config["pod_id"]
    ):
        raise ValueError("Operational override differs")
    pipeline = load_pipeline(config["pipeline_source"])
    new_config = read(config["pipeline_config_path"])
    if (
        sha(config["pipeline_config_path"]) != config["pipeline_config_sha256"]
        or new_config.get("reuse_completed_generation") is not True
    ):
        raise ValueError(
            "Operational pipeline must pin its config and forbid "
            "generation replay"
        )
    pipeline.validate(new_config)
    policy = pipeline.deadline_override(
        config["override_path"],
        config["override_sha256"],
        config=new_config,
        config_sha256=config["pipeline_config_sha256"],
    )
    policy.timestamp()
    deadline = datetime.fromisoformat(
        override["work_deadline_utc"].replace("Z", "+00:00")
    ).timestamp()
    for path in override["completion_stop"]["lock_paths"]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch(exist_ok=True)
    with locked(root / ".continuation.lock") as controller_lock:
        if state_path.exists():
            raise ValueError(
                "Adoption already attempted; explicit recovery review required"
            )
        state = {
            "controller_identity": identity,
            "started_at": datetime.now(UTC).isoformat(),
            "prior_pipeline_identity": config["prior_pipeline_identity"],
        }

        def publish(status, **details):
            state.update(details)
            if status in {
                "complete",
                "failed",
                "needs_time_budget_review",
                "needs_budget_review",
            }:
                state["terminal"] = status
            write(state_path, state)
            write(
                status_path,
                {
                    "controller_identity": identity,
                    "status": status,
                    "updated_at": datetime.now(UTC).isoformat(),
                    **details,
                },
            )

        parent_paused = False
        parent_removed = False
        try:
            parent, child = (
                require_process(config["parent"]),
                require_process(config["child"]),
            )
            if (
                child["ppid"] != parent["pid"]
                or child["sid"] != child["pid"]
                or parent["sid"] == child["sid"]
            ):
                raise ValueError(
                    "Generator is not an independently-sessioned child "
                    "of this parent"
                )
            current_state = read(root / "pipeline-state.json")
            current_status = read(root / "status.json")
            if (
                current_state["identity"] != config["prior_pipeline_identity"]
                or current_state.get("run_started")
                or current_state.get("completed")
            ):
                raise ValueError(
                    "Pipeline progressed beyond the reviewed "
                    "generation-only handoff"
                )
            if (
                current_status.get("status") != "generating"
                or current_status.get("child_pid") != child["pid"]
            ):
                raise ValueError(
                    "Pipeline is no longer generating the expected child"
                )
            if (
                sha(root / "generation/identity.json")
                != config["generation_identity_sha256"]
            ):
                raise ValueError("Generation identity differs before handoff")
            require_inherited_locks(
                child["pid"],
                [root / ".pipeline.lock", root / "generation/.writer.lock"],
            )
            frozen(operation / "prior-pipeline-state.json", current_state)
            frozen(operation / "prior-pipeline-status.json", current_status)
            publish(
                "handoff_prepared",
                adopted_pid=child["pid"],
                adopted_start_ticks=child["start_ticks"],
            )
            os.kill(parent["pid"], signal.SIGSTOP)
            parent_paused = True
            require_process(config["parent"])
            require_process(config["child"])
            require_inherited_locks(
                child["pid"],
                [root / ".pipeline.lock", root / "generation/.writer.lock"],
            )
            # SIGTERM invokes the old parent's child-killing cleanup; never use
            # it here.
            os.kill(parent["pid"], signal.SIGKILL)
            parent_removed = True
            publish("generation_adopted", old_parent_removed=True)
            write(
                root / "status.json",
                {
                    **current_status,
                    "deadline_utc": override["work_deadline_utc"],
                    "scientific_deadline_utc": current_status["deadline_utc"],
                    "controller_identity": identity,
                    "controller_status": "generation_adopted",
                    "operational_override_sha256": config["override_sha256"],
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            while True:
                live = process(child["pid"])
                if live is None or live["start_ticks"] != child["start_ticks"]:
                    break
                if live["argv"] != child["argv"]:
                    raise ValueError(
                        "Adopted generator changed command identity"
                    )
                if time.time() >= deadline:
                    publish(
                        "needs_time_budget_review",
                        reason="Work deadline reached while adopted generator "
                        "remains active",
                    )
                    return
                time.sleep(config.get("poll_seconds", 30))
            with ExitStack() as stack:
                stack.enter_context(locked(root / ".pipeline.lock"))
                stack.enter_context(locked(root / "generation/.writer.lock"))
                checked = verify_generation(
                    root / "generation",
                    read(config["plan_path"]),
                    config["generation_identity_sha256"],
                )
                frozen(
                    operation / "generation-adoption-verification.json", checked
                )
                pipeline = load_pipeline(config["pipeline_source"])
                new_config = read(config["pipeline_config_path"])
                pipeline.validate(new_config)
                if (
                    sha(config["pipeline_config_path"])
                    != config["pipeline_config_sha256"]
                ):
                    raise ValueError("Operational pipeline config changed")
                new_identity = pipeline.digest(
                    {
                        "config": new_config,
                        "config_sha256": config["pipeline_config_sha256"],
                        "pipeline_source_sha256": sha(
                            config["pipeline_source"]
                        ),
                    }
                )
                command = new_config["commands"]["generate"]
                new_state = {
                    "identity": new_identity,
                    "config_sha256": config["pipeline_config_sha256"],
                    "started_at": current_state["started_at"],
                    "adoption_record": str(
                        operation / "generation-adoption-verification.json"
                    ),
                    "prior_identity": current_state["identity"],
                    "completed": {
                        "generate": {
                            "status": "complete",
                            "command_identity": pipeline.command_identity(
                                command, new_config["bindings"]
                            ),
                            "outputs": pipeline.bind_files(command["outputs"]),
                        }
                    },
                }
                write(root / "pipeline-state.json", new_state)
                publish(
                    "generation_verified",
                    pipeline_identity=new_identity,
                    generation_verification=checked,
                )
            argv = [
                config["pipeline_python"],
                "-u",
                config["pipeline_source"],
                "--config",
                config["pipeline_config_path"],
                "--config-sha256",
                config["pipeline_config_sha256"],
                "--operational-override",
                config["override_path"],
                "--operational-override-sha256",
                config["override_sha256"],
            ]
            with (operation / "pipeline.log").open("ab") as log:
                resumed = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    pass_fds=(controller_lock.fileno(),),
                )
                publish("pipeline_running", resumed_pipeline_pid=resumed.pid)
                code = resumed.wait()
            final = read(root / "status.json")
            status = final.get("status")
            if (
                status
                not in {
                    "complete",
                    "needs_time_budget_review",
                    "needs_budget_review",
                }
                or code
            ):
                raise RuntimeError(
                    "Operational pipeline failed; no automatic retry"
                )
            if (
                status == "complete"
                and not (root / "training/extension_result.json").is_file()
            ):
                raise ValueError("Completion has no published extension result")
            publish(status, pipeline_final_status=final)
            os.sync()
        except BaseException as error:
            if parent_paused and not parent_removed:
                try:
                    require_process(config["parent"])
                    os.kill(config["parent"]["pid"], signal.SIGCONT)
                except (ProcessLookupError, ValueError):
                    pass
            publish(
                "failed",
                reason=f"{type(error).__name__}: {error}",
                old_parent_removed=parent_removed,
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("Adoption config checksum differs")
    run(read(args.config))


if __name__ == "__main__":
    main()
