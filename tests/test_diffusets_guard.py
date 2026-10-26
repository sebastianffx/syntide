"""Deadline changes and completed-job shutdown need no live cloud instance."""

import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1] / "scripts/runpod/diffusets_guard.py"
)
spec = importlib.util.spec_from_file_location("guard_under_test", SOURCE)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    now = [dt.datetime(2026, 10, 23, 13, tzinfo=dt.UTC)]

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0]

    monkeypatch.setattr(guard.dt, "datetime", Clock)
    monkeypatch.setattr(
        guard,
        "probe",
        lambda _: ("runpodctl", ["stop", "pod"], {"TEST": "only"}),
    )
    calls = []
    monkeypatch.setattr(guard.os, "sync", lambda: calls.append("sync"))
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda argv, **kwargs: (
            calls.append(argv) or SimpleNamespace(returncode=0)
        ),
    )
    base = {
        "pod_id": "abcdefghijkl",
        "created_utc": "2026-10-23T11:27:47Z",
        "deadline_utc": "2026-10-25T11:27:47Z",
        "hourly_rate_upper_bound_usd": 0.62,
        "max_spend_usd": 40,
        "state_path": str(tmp_path / "guard-state.json"),
    }
    override = {
        "schema_version": 1,
        "pod_id": base["pod_id"],
        "created_utc": base["created_utc"],
        "base_config_sha256": guard.config_digest(base),
        "deadline_utc": "2026-10-25T23:27:47Z",
        "work_deadline_utc": "2026-10-25T23:12:47Z",
        "hourly_rate_upper_bound_usd": 0.62,
        "max_spend_usd": 40,
        "storage_reserve_usd": 2.5,
        "reviewed_at_utc": "2026-10-23T12:00:00Z",
        "reason": "Reviewed time extension within the unchanged cost cap",
        "authorization": "User authorized completion within $40",
        "pipeline_config_sha256": "a" * 64,
    }
    override_path = tmp_path / "operational-override.json"

    def write_override():
        raw = json.dumps(override).encode()
        override_path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def run(**kwargs):
        return guard.guard(
            base,
            once=True,
            operational_override=override_path,
            operational_override_sha256=write_override(),
            **kwargs,
        )

    return SimpleNamespace(
        base=base,
        override=override,
        path=override_path,
        now=now,
        calls=calls,
        run=run,
        write=write_override,
        root=tmp_path,
    )


def completion(setup, terminal="complete"):
    root = setup.root
    status, state, result = (
        root / "control-status.json",
        root / "control-state.json",
        root / "result.json",
    )
    status.write_text(
        json.dumps({"status": terminal, "controller_identity": "controller-1"})
    )
    state.write_text(
        json.dumps(
            {"terminal": terminal, "controller_identity": "controller-1"}
        )
    )
    result.write_text('{"verified": true}')
    locks = [
        root / name
        for name in (
            "continuation.lock",
            ".pipeline.lock",
            "training.lock",
            "writer.lock",
        )
    ]
    for lock in locks:
        lock.touch()
    value = {
        "status_path": str(status),
        "state_path": str(state),
        "status_matches": {"controller_identity": "controller-1"},
        "state_matches": {"controller_identity": "controller-1"},
        "lock_paths": [str(p) for p in locks],
        "required_complete_paths": [str(result)],
        "grace_seconds": 0,
        "collection_lease_path": str(root / "collection-lease.json"),
    }
    setup.override["completion_stop"] = value
    return value


def test_reviewed_60_hours_is_bounded_by_same_cap_and_storage_reserve(setup):
    result = setup.run()
    assert result["status"] == "armed"
    assert result["deadline_utc"] == "2026-10-25T23:27:47+00:00"
    assert result["authorized_interval_cost_plus_reserve_usd"] == pytest.approx(
        39.7
    )
    assert result["max_spend_usd"] == 40
    assert result["storage_reserve_usd"] == 2.5
    assert "finite" in result["storage_note"]
    assert not setup.calls


@pytest.mark.parametrize(
    "name,value",
    [
        ("pod_id", "anotherpodid"),
        ("created_utc", "2026-10-24T11:27:47Z"),
        ("max_spend_usd", 41),
        ("max_spend_usd", 39),
        ("hourly_rate_upper_bound_usd", 0.60),
        ("hourly_rate_upper_bound_usd", float("nan")),
        ("storage_reserve_usd", float("inf")),
        ("storage_reserve_usd", -1),
        ("base_config_sha256", "b" * 64),
        ("deadline_utc", "2026-10-26T01:27:47Z"),
        ("work_deadline_utc", "2026-10-25T23:27:47Z"),
    ],
)
def test_override_cannot_weaken_identity_cost_or_stop_margin(
    setup, name, value
):
    setup.override[name] = value
    with pytest.raises(ValueError):
        guard.validate_override(
            setup.base, setup.override, guard.config_digest(setup.base)
        )


def test_changed_pinned_override_is_rejected_without_disabling_base_guard(
    setup,
):
    expected = setup.write()
    setup.path.write_text("{}")
    result = guard.guard(
        setup.base,
        once=True,
        operational_override=setup.path,
        operational_override_sha256=expected,
    )
    assert result["status"] == "armed"
    assert result["operational_override_rejected"]
    assert result["operational_override_sha256"] is None
    assert result["deadline_utc"] == "2026-10-25T11:27:47+00:00"
    assert not setup.calls


def test_bad_reload_retains_last_valid_policy_in_running_guard(
    setup, monkeypatch
):
    expected = setup.write()
    polls = []

    def sleep(_):
        polls.append(1)
        if len(polls) == 1:
            setup.path.write_text("{interrupted write")
        else:
            raise InterruptedError("end unit test")

    monkeypatch.setattr(guard.time, "sleep", sleep)
    with pytest.raises(InterruptedError):
        guard.guard(
            setup.base,
            operational_override=setup.path,
            operational_override_sha256=expected,
        )
    state = json.loads(Path(setup.base["state_path"]).read_text())
    assert state["deadline_utc"] == "2026-10-25T23:27:47+00:00"
    assert state["operational_override_sha256"] == expected
    assert state["operational_override_rejected"]


@pytest.mark.parametrize("terminal", sorted(guard.TERMINAL))
def test_terminal_stop_requires_matching_durable_state_and_holds_every_lock(
    setup, monkeypatch, terminal
):
    value = completion(setup, terminal)
    if terminal != "complete":
        Path(value["required_complete_paths"][0]).unlink()

    def stop(argv, **kwargs):
        assert setup.calls == ["sync"]
        assert argv == ["runpodctl", "stop", "pod", setup.base["pod_id"]]
        for path in value["lock_paths"]:
            with (
                Path(path).open("r+") as stream,
                pytest.raises(BlockingIOError),
            ):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(Path(setup.base["state_path"]).read_text())
        assert state["status"] == "stop_requested_preserving_volume"
        setup.calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(guard.subprocess, "run", stop)
    result = setup.run()
    assert result["stop_reason"] == "terminal_" + terminal
    assert result["status"] == "stop_requested_retrying_until_shutdown"


@pytest.mark.parametrize(
    "issue",
    [
        "busy_child",
        "busy_controller",
        "missing_lock",
        "missing_result",
        "state_disagrees",
        "wrong_identity",
        "malformed_status",
        "active_status",
    ],
)
def test_unsafe_terminal_evidence_never_triggers_early_stop(setup, issue):
    value = completion(setup)
    with ExitStack() as stack:
        if issue.startswith("busy_"):
            path = value["lock_paths"][0 if issue == "busy_controller" else -1]
            held = stack.enter_context(Path(path).open("r+"))
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif issue == "missing_lock":
            Path(value["lock_paths"][0]).unlink()
        elif issue == "missing_result":
            Path(value["required_complete_paths"][0]).unlink()
        elif issue == "state_disagrees":
            Path(value["state_path"]).write_text(
                '{"terminal":"failed","controller_identity":"controller-1"}'
            )
        else:
            status = {
                "status": "complete",
                "controller_identity": "controller-1",
            }
            if issue == "wrong_identity":
                status["controller_identity"] = "another-controller"
            if issue == "active_status":
                status["status"] = "training_or_evaluation"
            Path(value["status_path"]).write_text(
                "{" if issue == "malformed_status" else json.dumps(status)
            )
        assert setup.run()["status"] == "armed"
        assert not setup.calls


def test_completion_grace_survives_guard_restart(setup):
    value = completion(setup)
    value["grace_seconds"] = 900
    first = setup.run()
    assert first["completion_stop_waiting_for"] == "collection_grace"
    setup.now[0] += dt.timedelta(minutes=14)
    second = setup.run()
    assert second["completion_grace_remaining_seconds"] == 60
    assert not setup.calls
    setup.now[0] += dt.timedelta(minutes=2)
    assert setup.run()["stop_reason"] == "terminal_complete"


@pytest.mark.parametrize(
    "issue", [None, "expired", "future", "wrong_pod", "too_long", "malformed"]
)
def test_only_fresh_bounded_own_pod_collection_lease_delays_terminal_stop(
    setup, issue
):
    value = completion(setup)
    issued, expires = (
        setup.now[0] - dt.timedelta(seconds=60),
        setup.now[0] + dt.timedelta(seconds=600),
    )
    if issue == "expired":
        expires = setup.now[0] - dt.timedelta(seconds=1)
    if issue == "future":
        issued = setup.now[0] + dt.timedelta(seconds=1)
    if issue == "too_long":
        expires = issued + dt.timedelta(seconds=1801)
    lease = {
        "pod_id": "wrongpod1234"
        if issue == "wrong_pod"
        else setup.base["pod_id"],
        "issued_utc": issued.isoformat(),
        "expires_utc": expires.isoformat(),
    }
    Path(value["collection_lease_path"]).write_text(
        "{" if issue == "malformed" else json.dumps(lease)
    )
    result = setup.run()
    if issue is None:
        assert result["completion_stop_waiting_for"] == "collection_lease"
        assert not setup.calls
    else:
        assert result["stop_reason"] == "terminal_complete"


def test_absolute_stop_ignores_active_locks_collection_grace_and_lease(setup):
    value = completion(setup)
    value["grace_seconds"] = 900
    setup.now[0] = dt.datetime(2026, 10, 25, 23, 13, tzinfo=dt.UTC)
    Path(value["collection_lease_path"]).write_text(
        json.dumps(
            {
                "pod_id": setup.base["pod_id"],
                "issued_utc": setup.now[0].isoformat(),
                "expires_utc": (
                    setup.now[0] + dt.timedelta(minutes=20)
                ).isoformat(),
            }
        )
    )
    with Path(value["lock_paths"][0]).open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = setup.run()
    assert result["stop_reason"] == "deadline"
    assert setup.calls == [
        "sync",
        ["runpodctl", "stop", "pod", setup.base["pod_id"]],
    ]
