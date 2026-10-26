"""Training deadline checks work without live cloud resources."""

import datetime as dt
import fcntl
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/runpod"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location(
    "qa_pod_backstop", DIRECTORY / "qa_pod_backstop.py"
)
backstop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backstop)
NOW = dt.datetime(2026, 10, 23, 14, tzinfo=dt.UTC)


@pytest.fixture
def config():
    return {
        "pod_id": backstop.POD_ID,
        "deadline_utc": backstop.ORIGINAL_DEADLINE,
        "budget_anchor_utc": NOW.isoformat(),
        "observed_account_delta_usd": 30,
        "observed_total_hourly_spend_usd": 2.419,
        "hourly_spend_upper_bound_usd": 2.47,
        "max_spend_usd": 50,
    }


@pytest.fixture
def paths(tmp_path):
    study, root = tmp_path / "study", tmp_path / "guard"
    study.mkdir()
    root.mkdir()
    (study / ".run.lock").touch()
    (study / "status.json").write_text('{"status":"complete"}')
    return study, root


def authentication():
    return (
        "/usr/bin/runpodctl",
        ["stop", "pod"],
        {"RUNPOD_API_KEY": "TEST_SECRET_NOT_LOGGED"},
    )


def polling(config, paths, prior=None, now=NOW, run=None):
    study, root = paths
    return backstop.poll(
        config,
        prior or {},
        authentication(),
        now=now,
        study=study,
        root=root,
        run=run
        or (lambda *a, **k: pytest.fail("Unexpected stop/sync command")),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("pod_id", "0il9463gvr3mmv"),
        ("max_spend_usd", 50.01),
        ("max_spend_usd", True),
        ("max_spend_usd", 0),
        ("observed_account_delta_usd", -1),
        ("observed_account_delta_usd", float("nan")),
        ("hourly_spend_upper_bound_usd", float("inf")),
        ("hourly_spend_upper_bound_usd", 2.419),
        ("deadline_utc", "2026-10-24T01:42:38.341Z"),
        ("deadline_utc", "2026-10-24T01:42:38"),
        ("budget_anchor_utc", "2026-10-23T14:01:00Z"),
    ],
)
def test_bounds_cannot_be_extended(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        backstop.validate(config, NOW)


def test_overlap_rate_estimates_whole_account(config, paths):
    state = polling(config, paths, now=NOW + dt.timedelta(hours=2))
    assert state["estimated_account_delta_upper_bound_usd"] == pytest.approx(
        34.94
    )
    assert state["budget_includes_overlapping_pods"] is True


def test_terminal_grace_survives_restart_and_stops_only_owned_pod(
    config, paths
):
    first = polling(config, paths)
    later = polling(config, paths, first, NOW + dt.timedelta(seconds=899))
    assert later["eligible_terminal_utc"] == first["eligible_terminal_utc"]
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout="TEST_SECRET_NOT_LOGGED",
            stderr="TEST_SECRET_NOT_LOGGED",
        )

    stopped = polling(
        config, paths, later, NOW + dt.timedelta(seconds=900), run
    )
    assert calls == [
        ["sync"],
        ["/usr/bin/runpodctl", "stop", "pod", backstop.POD_ID],
    ]
    assert stopped["status"] == "stop_requested_retrying_until_shutdown"
    assert "TEST_SECRET_NOT_LOGGED" not in json.dumps(stopped)
    assert "TEST_SECRET_NOT_LOGGED" not in (paths[1] / "state.json").read_text()


def test_inherited_child_execution_lock_prevents_terminal_stop(config, paths):
    lock = (paths[0] / ".run.lock").open("r+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        pass_fds=(lock.fileno(),),
        start_new_session=True,
    )
    lock.close()  # The child alone retains the inherited open file description.
    try:
        state = polling(
            config,
            paths,
            {
                "runner_status": "complete",
                "eligible_terminal_utc": NOW.isoformat(),
            },
            NOW + dt.timedelta(seconds=4000),
        )
        assert state["terminal_execution_lock_free"] is False
        assert state["eligible_terminal_utc"] is None
    finally:
        child.communicate(timeout=10)
    assert polling(config, paths)["terminal_execution_lock_free"] is True


def test_missing_execution_lock_never_proves_terminal(config, paths):
    (paths[0] / ".run.lock").unlink()
    assert polling(config, paths)["terminal_execution_lock_free"] is False


def test_lease_cannot_renew_past_total_2700_seconds(config, paths):
    first = polling(config, paths)
    issued = NOW + dt.timedelta(seconds=1000)
    lease = {
        "pod_id": backstop.POD_ID,
        "issued_utc": issued.isoformat(),
        "expires_utc": (issued + dt.timedelta(seconds=1800)).isoformat(),
    }
    (paths[1] / "collection-lease.json").write_text(json.dumps(lease))
    state = polling(config, paths, first, NOW + dt.timedelta(seconds=2699))
    assert (
        state["terminal_stop_after_utc"]
        == (NOW + dt.timedelta(seconds=2700)).isoformat()
    )
    calls = []
    polling(
        config,
        paths,
        state,
        NOW + dt.timedelta(seconds=2700),
        lambda argv, **kw: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    assert calls[-1][-1] == backstop.POD_ID


@pytest.mark.parametrize("reason", ["budget", "deadline"])
def test_risk_stop_overrides_active_runner_and_lease(config, paths, reason):
    (paths[0] / "status.json").write_text('{"status":"running"}')
    (paths[1] / "collection-lease.json").write_text(
        json.dumps(
            {
                "pod_id": backstop.POD_ID,
                "issued_utc": NOW.isoformat(),
                "expires_utc": (NOW + dt.timedelta(seconds=1800)).isoformat(),
            }
        )
    )
    if reason == "budget":
        config["observed_account_delta_usd"] = 49.5
    else:
        config["deadline_utc"] = NOW.isoformat()
    calls = []
    state = polling(
        config,
        paths,
        run=lambda argv, **kw: (
            calls.append(argv) or SimpleNamespace(returncode=0)
        ),
    )
    assert state["stop_reason"] in {
        "absolute_deadline",
        "account_delta_budget_margin",
    }
    assert calls[-1] == ["/usr/bin/runpodctl", "stop", "pod", backstop.POD_ID]


def test_read_only_probe_does_not_create_state(config, monkeypatch, tmp_path):
    monkeypatch.setattr(backstop, "ROOT", tmp_path / "absent")
    monkeypatch.setattr(backstop, "utc_now", lambda: NOW)
    called = []
    monkeypatch.setattr(
        backstop.diffusets_guard, "probe", lambda pod: called.append(pod)
    )
    result = backstop.guard(config, check_only=True)
    assert called == [backstop.POD_ID]
    assert result["status"] == "read_only_probe_passed"
    assert not backstop.ROOT.exists()


def test_stop_does_not_log_sensitive_timeout_output(config, paths):
    config["observed_account_delta_usd"] = 50

    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(
            argv,
            30,
            output="TEST_SECRET_NOT_LOGGED",
            stderr="TEST_SECRET_NOT_LOGGED",
        )

    state = polling(config, paths, run=run)
    assert state["last_stop_error_type"] == "TimeoutExpired"
    assert "TEST_SECRET_NOT_LOGGED" not in json.dumps(state)
