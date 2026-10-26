"""Pipeline stages preserve execution locks and checks against repeated runs."""

import copy
import importlib.util
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture
def pipeline():
    path = (
        Path(__file__).parents[1] / "scripts/runpod/run_diffusets_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_diffusets_pipeline", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(pipeline, tmp_path):
    source = tmp_path / "bound-source.py"
    source.write_text("frozen code")
    prepared = tmp_path / "prepared-manifest.json"
    config = {
        "schema_version": 1,
        "run_dir": str(tmp_path / "study"),
        "deadline_utc": datetime.fromtimestamp(
            time.time() + 3600, UTC
        ).isoformat(),
        "min_training_remaining_seconds": 600,
        "runner_status_path": str(tmp_path / "training/status.json"),
        "runner_result_path": str(tmp_path / "training/extension_result.json"),
        "bindings": {str(source): pipeline.sha(source)},
        "commands": {},
    }
    for name in ("generate", "prepare", "preflight", "run"):
        config["commands"][name] = {
            "argv": [sys.executable, name],
            "cwd": str(tmp_path),
            "env": {"PYTHONPATH": str(tmp_path)},
            "inputs": [],
            "outputs": [str(tmp_path / f"{name}-output.json")],
        }
    config["commands"]["prepare"]["outputs"].append(str(prepared))
    config["commands"]["preflight"]["inputs"] = [str(prepared)]
    config["commands"]["run"]["inputs"] = [str(prepared)]
    calls = []

    def child(command, log, fd, deadline, on_start):
        name = command["argv"][1]
        calls.append(name)
        on_start(123)
        for path in command["outputs"]:
            pipeline.write(path, {"stage": name})
        if name == "run":
            pipeline.write(config["runner_status_path"], {"status": "complete"})
            pipeline.write(
                config["runner_result_path"], {"scientific_result": True}
            )
        return 0

    return config, calls, child


def run(pipeline, setup, child=None):
    config, _, default = setup
    return pipeline.run_pipeline(
        config, pipeline.digest(config), child_runner=child or default
    )


def test_success_runs_all_stages_in_order_and_requires_runner_result(
    pipeline, setup
):
    result = run(pipeline, setup)
    assert result["status"] == "complete"
    assert setup[1] == ["generate", "prepare", "preflight", "run"]
    state = pipeline.read(Path(setup[0]["run_dir"]) / "pipeline-state.json")
    assert set(state["completed"]) == {
        "generate",
        "prepare",
        "preflight",
        "run",
    }
    setup[1].clear()
    assert run(pipeline, setup)["status"] == "complete"
    assert setup[1] == []


@pytest.mark.parametrize(
    "terminal", ["needs_budget_review", "needs_time_budget_review"]
)
def test_exit_zero_review_state_never_marks_pipeline_complete(
    pipeline, setup, terminal
):
    config, calls, original = setup

    def child(command, *args):
        if command["argv"][1] == "run":
            calls.append("run")
            pipeline.write(config["runner_status_path"], {"status": terminal})
            return 0
        return original(command, *args)

    assert run(pipeline, setup, child)["status"] == terminal
    assert not Path(config["runner_result_path"]).exists()
    calls.clear()
    assert run(pipeline, setup, child)["status"] == terminal
    assert calls == []


def test_deadline_prevents_new_training_and_unverified_preflight(
    pipeline, setup, monkeypatch
):
    config, calls, original = setup
    clock = [pipeline.deadline_timestamp(config) - 700]
    monkeypatch.setattr(pipeline.time, "time", lambda: clock[0])

    def child(command, *args):
        result = original(command, *args)
        if command["argv"][1] == "preflight":
            clock[0] += 200
        return result

    result = run(pipeline, setup, child)
    assert result["status"] == "needs_time_budget_review"
    assert calls == ["generate", "prepare", "preflight"]
    # Simulate a prior preflight receipt at a time when the original absolute
    # deadline still has room.
    # The pipeline config/deadline is never changed to make the retry pass.
    clock[0] -= 200
    calls.clear()
    assert run(pipeline, setup, original)["status"] == "complete"
    assert calls == ["generate", "prepare", "run"]


def test_changed_preflight_input_or_output_cannot_reuse_receipt(
    pipeline, setup
):
    config, _, _ = setup
    command = config["commands"]["preflight"]
    for p in command["inputs"] + command["outputs"]:
        pipeline.write(p, {"old": 1})
    identity = pipeline.command_identity(command, config["bindings"])
    receipt = {
        "status": "complete",
        "command_identity": identity,
        "outputs": pipeline.bind_files(command["outputs"]),
    }
    assert pipeline.preflight_reusable(receipt, identity)
    pipeline.write(command["inputs"][0], {"new": 2})
    assert not pipeline.preflight_reusable(
        receipt, pipeline.command_identity(command, config["bindings"])
    )
    pipeline.write(command["outputs"][0], {"new": 2})
    assert not pipeline.preflight_reusable(receipt, identity)


def test_interrupted_or_failed_training_is_not_automatically_redispatched(
    pipeline, setup
):
    _, calls, original = setup

    def child(command, *args):
        if command["argv"][1] == "run":
            calls.append("run")
            return 1
        return original(command, *args)

    assert run(pipeline, setup, child)["status"] == "failed"
    calls.clear()
    result = run(pipeline, setup, child)
    assert result["status"] == "failed" and "review" in result["reason"]
    assert calls == []


def test_generation_retry_preserves_pipeline_stages(pipeline, setup):
    _, calls, original = setup

    def failed(command, *args):
        calls.append(command["argv"][1])
        return 1

    assert run(pipeline, setup, failed)["status"] == "failed"
    assert calls == ["generate"]
    calls.clear()
    assert run(pipeline, setup, original)["status"] == "complete"
    assert calls == ["generate", "prepare", "preflight", "run"]


def test_zero_exit_without_complete_status_and_result_is_failure(
    pipeline, setup
):
    config, _, original = setup

    def child(command, *args):
        value = original(command, *args)
        if command["argv"][1] == "run":
            Path(config["runner_result_path"]).unlink()
        return value

    assert run(pipeline, setup, child)["status"] == "failed"


def test_source_or_config_change_blocks_resume(pipeline, setup):
    config, _, _ = setup
    assert run(pipeline, setup)["status"] == "complete"
    changed = copy.deepcopy(config)
    changed["min_training_remaining_seconds"] += 1
    with pytest.raises(ValueError, match="Frozen pipeline config"):
        pipeline.run_pipeline(changed, pipeline.digest(changed))
    next(iter(map(Path, config["bindings"]))).write_text("changed")
    with pytest.raises(ValueError, match="Frozen input/source"):
        run(pipeline, setup)


def test_inherited_child_keeps_lock_after_parent_descriptor_closes(
    pipeline, tmp_path
):
    path = tmp_path / "lock"
    child = None
    try:
        with pipeline.lock(path) as fd:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                pass_fds=(fd,),
            )
        with (
            pytest.raises(RuntimeError, match="inherited child"),
            pipeline.lock(path),
        ):
            pass
    finally:
        if child:
            child.terminate()
            child.wait(timeout=5)
    with pipeline.lock(path):
        pass


def test_live_stage_deadline_stops_entire_process_group(pipeline, tmp_path):
    command = {
        "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
        "cwd": str(tmp_path),
    }
    pids = []
    with (
        pipeline.lock(tmp_path / "lock") as fd,
        pytest.raises(pipeline.DeadlineReached),
    ):
        pipeline.run_child(
            command,
            tmp_path / "child.log",
            fd,
            time.time() - 1,
            pids.append,
            poll_seconds=0.01,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_completed_pipeline_refuses_changed_prepared_inputs(pipeline, setup):
    config, _, _ = setup
    assert run(pipeline, setup)["status"] == "complete"
    Path(config["commands"]["run"]["inputs"][0]).write_text(
        "changed prepared dataset"
    )
    with pytest.raises(ValueError, match="training inputs changed"):
        run(pipeline, setup)


@pytest.fixture
def override_bundle(pipeline, setup, tmp_path):
    config, _, _ = setup
    for name in (
        "run_diffusets_pipeline.py",
        "run_diffusets_arm.py",
        "run_qa_arms.py",
        "qa_preflight.py",
    ):
        source = tmp_path / "original-code" / name
        source.parent.mkdir(exist_ok=True)
        source.write_text("original " + name)
        config["bindings"][str(source)] = pipeline.sha(source)
    original_config = tmp_path / "pipeline-config.json"
    protocol = tmp_path / "protocol.json"
    receipt = tmp_path / "receipt.json"
    guard = tmp_path / "guard.json"
    pipeline.write(original_config, config)
    pipeline.write(
        protocol,
        {"evaluation_budget": {"deadline_utc": config["deadline_utc"]}},
    )
    pipeline.write(
        receipt,
        {
            "pod": {"id": "owned-pod"},
            "created_utc": datetime.fromtimestamp(
                time.time() - 3600, UTC
            ).isoformat(),
            "cost_plan": {
                "max_spend_usd": 40,
                "hourly_rate_upper_bound_usd": 0.62,
            },
        },
    )
    pipeline.write(guard, {"max_spend_usd": 40})
    payload = {
        "schema_version": 1,
        "pod_id": "owned-pod",
        "created_utc": datetime.now(UTC).isoformat(),
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "deadline_utc": datetime.fromtimestamp(
            time.time() + 10800, UTC
        ).isoformat(),
        "work_deadline_utc": datetime.fromtimestamp(
            time.time() + 9900, UTC
        ).isoformat(),
        "reason": (
            "Operational time extension; original "
            "scientific settings stay frozen"
        ),
        "authorization": (
            "User explicitly authorized deadline extension within $40"
        ),
        "completion_stop": {"stop_on_completion": True},
        "max_spend_usd": 40,
        "hourly_rate_upper_bound_usd": 0.62,
        "storage_reserve_usd": 2.5,
        "previous_pipeline_source_sha256": config["bindings"][
            str(tmp_path / "original-code/run_diffusets_pipeline.py")
        ],
    }
    for path_key, hash_key, path in (
        ("pipeline_config_path", "pipeline_config_sha256", original_config),
        ("protocol_path", "protocol_sha256", protocol),
        ("receipt_path", "receipt_sha256", receipt),
        ("base_config_path", "base_config_sha256", guard),
    ):
        payload.update({path_key: str(path), hash_key: pipeline.sha(path)})
    override_path = tmp_path / "operations/override.json"
    pipeline.write(override_path, payload)
    return override_path, payload, original_config


def build_override(pipeline, setup, override_bundle):
    path, _, original = override_bundle
    return pipeline.deadline_override(
        str(path),
        pipeline.sha(path),
        config=setup[0],
        config_sha256=pipeline.sha(original),
    )


def test_operational_deadline_extends_time_without_changing_frozen_identity(
    pipeline, setup, override_bundle, monkeypatch
):
    config, calls, original_child = setup
    override = build_override(pipeline, setup, override_bundle)
    original_bytes = override_bundle[2].read_bytes()
    monkeypatch.setattr(
        pipeline.time, "time", lambda: pipeline.deadline_timestamp(config) + 1
    )
    assert run(pipeline, setup)["status"] == "needs_time_budget_review"
    state_path = Path(config["run_dir"]) / "pipeline-state.json"
    identity = pipeline.read(state_path)["identity"]

    def child(command, log, fd, deadline, on_start):
        assert callable(deadline) and deadline() > pipeline.time.time()
        assert (
            command["env"][pipeline.OVERRIDE_SHA_ENV]
            == override.expected_sha256
        )
        assert command["env"][pipeline.OVERRIDE_PATH_ENV] == str(override.path)
        return original_child(command, log, fd, deadline, on_start)

    result = pipeline.run_pipeline(
        config,
        pipeline.digest(config),
        child_runner=child,
        operational_override=override,
    )
    assert result["status"] == "complete"
    assert calls == ["generate", "prepare", "preflight", "run"]
    assert pipeline.read(state_path)["identity"] == identity
    assert override_bundle[2].read_bytes() == original_bytes
    assert result["scientific_deadline_utc"] == config["deadline_utc"]
    assert result["operational_override"]["sha256"] == override.expected_sha256


@pytest.mark.parametrize(
    "field,value",
    [
        ("pod_id", "another-pod"),
        ("max_spend_usd", 41),
        ("hourly_rate_upper_bound_usd", 0.60),
        ("pipeline_config_sha256", "0" * 64),
        ("protocol_sha256", "0" * 64),
        ("receipt_sha256", "0" * 64),
        ("base_config_sha256", "0" * 64),
        ("previous_pipeline_source_sha256", "0" * 64),
        ("authorization", ""),
        ("completion_stop", {}),
    ],
)
def test_override_rejects_unbound_provenance_or_loosened_cost_guard(
    pipeline, setup, override_bundle, field, value
):
    path, payload, _ = override_bundle
    payload[field] = value
    pipeline.write(path, payload)
    with pytest.raises(ValueError):
        build_override(pipeline, setup, override_bundle)


def test_override_rejects_total_compute_plus_storage_over_40(
    pipeline, setup, override_bundle
):
    path, payload, _ = override_bundle
    receipt = pipeline.read(payload["receipt_path"])
    started = pipeline.utc_timestamp(receipt["created_utc"], "created")
    payload["deadline_utc"] = datetime.fromtimestamp(
        started + 60 * 3600, UTC
    ).isoformat()
    payload["work_deadline_utc"] = datetime.fromtimestamp(
        started + 60 * 3600 - 900, UTC
    ).isoformat()
    pipeline.write(path, payload)
    assert (
        build_override(pipeline, setup, override_bundle).timestamp() > started
    )
    payload["deadline_utc"] = datetime.fromtimestamp(
        started + 61 * 3600, UTC
    ).isoformat()
    pipeline.write(path, payload)
    with pytest.raises(ValueError, match="projected cost"):
        build_override(pipeline, setup, override_bundle)


def test_override_is_rechecked_during_live_child_and_tampering_stops_child(
    pipeline, setup, override_bundle, tmp_path
):
    override = build_override(pipeline, setup, override_bundle)
    command = {
        "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
        "cwd": str(tmp_path),
    }
    pids = []

    def started(pid):
        pids.append(pid)
        override.path.write_text("tampered")

    with (
        pipeline.lock(tmp_path / "override-test.lock") as fd,
        pytest.raises(ValueError, match="SHA256"),
    ):
        pipeline.run_child(
            command,
            tmp_path / "child.log",
            fd,
            override.timestamp,
            started,
            poll_seconds=0.01,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_operational_relocation_preserves_generator_and_baseline_helpers(
    pipeline, setup, override_bundle
):
    config, _, _ = setup
    path, _, _ = override_bundle
    override = build_override(pipeline, setup, override_bundle)
    changed = copy.deepcopy(config)
    changed["reuse_completed_generation"] = True
    changed["bindings"][str(path)] = pipeline.sha(path)
    for name in (
        "run_diffusets_pipeline.py",
        "run_diffusets_arm.py",
        "run_qa_arms.py",
        "qa_preflight.py",
    ):
        old_path = next(p for p in config["bindings"] if Path(p).name == name)
        new_path = path.parent / name
        new_path.write_bytes(
            Path(old_path).read_bytes()
            + (b" updated" if name.startswith("run_diffusets") else b"")
        )
        changed["bindings"][str(new_path)] = pipeline.sha(new_path)
    pipeline.validate_operational_config(changed, config, override)
    changed["commands"]["generate"]["argv"].append("--different-seed")
    with pytest.raises(ValueError, match="frozen generate command"):
        pipeline.validate_operational_config(changed, config, override)
    changed["commands"]["generate"]["argv"].pop()
    changed["bindings"][str(path.parent / "run_qa_arms.py")] = "0" * 64
    with pytest.raises(ValueError, match="baseline helper"):
        pipeline.validate_operational_config(changed, config, override)


def test_adoption_skips_only_verified_complete_generation(pipeline, setup):
    config, calls, child = setup
    config["reuse_completed_generation"] = True

    def fail_prepare(command, *args):
        if command["argv"][1] == "prepare":
            calls.append("prepare")
            return 1
        return child(command, *args)

    assert run(pipeline, setup, fail_prepare)["status"] == "failed"
    assert calls == ["generate", "prepare"]
    calls.clear()
    assert run(pipeline, setup)["status"] == "complete"
    assert calls == ["prepare", "preflight", "run"]


def test_adoption_refuses_failed_generation_replay(pipeline, setup):
    config, calls, _ = setup
    config["reuse_completed_generation"] = True

    def fail(command, *args):
        calls.append(command["argv"][1])
        return 1

    assert run(pipeline, setup, fail)["status"] == "failed"
    calls.clear()
    result = run(pipeline, setup)
    assert (
        result["status"] == "failed" and "refusing replay" in result["reason"]
    )
    assert calls == []


def test_adoption_refuses_changed_completed_generation_outputs(pipeline, setup):
    config, calls, child = setup
    config["reuse_completed_generation"] = True

    def fail_prepare(command, *args):
        return 1 if command["argv"][1] == "prepare" else child(command, *args)

    assert run(pipeline, setup, fail_prepare)["status"] == "failed"
    Path(config["commands"]["generate"]["outputs"][0]).write_text("changed")
    calls.clear()
    result = run(pipeline, setup)
    assert (
        result["status"] == "failed" and "refusing replay" in result["reason"]
    )
    assert calls == []
