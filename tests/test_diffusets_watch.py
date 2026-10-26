"""Generation status cannot trigger premature result collection."""

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts/runpod"


def module(name):
    spec = importlib.util.spec_from_file_location(
        name, DIRECTORY / (name + ".py")
    )
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


qa_pod = module("qa_pod")
module("diffusets_pod")
module("diffusets_guard")
module("watch_qa_study")
watch = module("diffusets_watch")


@pytest.mark.parametrize("guard_matches", [True, False])
def test_probe_excludes_only_verified_guard_with_study_override(
    tmp_path, capsys, guard_matches
):
    root = tmp_path / "study"
    root.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()
    guard_path = tmp_path / "guard-state.json"
    guard_path.write_text(json.dumps({"guard_pid": 10, "status": "armed"}))
    guard_script = "/workspace/diffusets-extension/diffusets_guard_v2.py"
    override = "/workspace/diffusets-study/operations/operational-override.json"
    commands = {
        10: [
            "python3",
            "-u",
            guard_script,
            "/workspace/diffusets-extension/guard-config.json",
            "--operational-override",
            override,
        ],
        # Merely mentioning the guard must not hide a workload process.
        20: [
            "python3",
            "/workspace/diffusets-study/runner.py",
            "--reference",
            guard_script,
        ],
    }
    if not guard_matches:
        commands[10].insert(2, "/workspace/diffusets-study/other.py")
    for pid, args in commands.items():
        folder = proc / str(pid)
        folder.mkdir()
        (folder / "cmdline").write_bytes(
            b"\0".join(a.encode() for a in args) + b"\0"
        )
        (folder / "stat").write_text(f"{pid} (python3) S 1 2 3")
    script = (
        watch.PROBE.replace(
            "pathlib.Path('/workspace/diffusets-study')",
            f"pathlib.Path({str(root)!r})",
        )
        .replace(
            "pathlib.Path('/workspace/diffusets-extension/guard-state.json')",
            f"pathlib.Path({str(guard_path)!r})",
        )
        .replace("pathlib.Path('/proc')", f"pathlib.Path({str(proc)!r})")
    )
    exec(script, {})
    result = json.loads(capsys.readouterr().out)
    assert sorted(p["pid"] for p in result["processes"]) == (
        [20] if guard_matches else [10, 20]
    )
    assert result["guard_health"]["healthy"] is guard_matches


@pytest.fixture
def supervisor(tmp_path, monkeypatch):
    now = dt.datetime.now(dt.UTC)
    own = {
        "id": "abcdefghijkl",
        "name": "extension",
        "imageName": "image",
        "desiredStatus": "RUNNING",
    }
    receipt = {
        "pod": own,
        "guard": {"status": "armed"},
        "status": "created",
        "execute": True,
        "created_utc": now.isoformat(),
        "deadline_utc": (now + dt.timedelta(hours=48)).isoformat(),
        "cost_plan": {"max_spend_usd": 40, "hourly_rate_upper_bound_usd": 0.62},
        "ssh_key_path": "/fake",
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    args = SimpleNamespace(
        receipt=path,
        state=tmp_path / "state.json",
        output=tmp_path / "study",
        execute=False,
        collection_timeout_seconds=1800,
    )

    class Client:
        def query(self, query):
            assert query == qa_pod.ACCOUNT_QUERY
            return {"myself": {"clientBalance": 0, "currentSpendPerHr": 50}}

    result = watch.Supervisor(args, Client())
    monkeypatch.setattr(result, "own_pod", lambda: own)
    monkeypatch.setattr(qa_pod, "ssh_argv", lambda *a: ["ssh", "own"])
    return result


@pytest.mark.parametrize(
    "status", [None, "running", "awaiting_preparation", "generation_complete"]
)
def test_nonterminal_idle_phase_never_collects_or_stops(
    supervisor, monkeypatch, status
):
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a: {"status": {"status": status}, "processes": []},
    )
    monkeypatch.setattr(
        supervisor,
        "collect",
        lambda *a, **k: pytest.fail("premature collection"),
    )
    monkeypatch.setattr(
        supervisor,
        "stop_preserving",
        lambda *a: pytest.fail("account-wide spend used"),
    )
    assert supervisor.poll() is False
    assert supervisor.state["status"] == "watching"
    assert supervisor.state["own_pod_spend_upper_bound_usd"] < 0.01


def test_terminal_status_with_active_process_does_not_collect(
    supervisor, monkeypatch
):
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a: {
            "status": {"status": "complete"},
            "processes": [{"pid": 5}],
        },
    )
    monkeypatch.setattr(
        supervisor, "collect", lambda *a, **k: pytest.fail("active process")
    )
    assert supervisor.poll() is False


def test_terminal_quiescent_status_collects_before_termination(
    supervisor, monkeypatch
):
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a: {"status": {"status": "complete"}, "processes": []},
    )
    events = []
    monkeypatch.setattr(
        supervisor,
        "collect",
        lambda *a, **k: events.append("verified") or {"files": 5},
    )
    monkeypatch.setattr(
        supervisor,
        "terminate_verified",
        lambda verified: events.append(verified) or True,
    )
    assert supervisor.poll() is True
    assert events == ["verified", {"files": 5}]


def test_time_budget_review_is_terminal(supervisor, monkeypatch):
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a: {
            "status": {"status": "needs_time_budget_review"},
            "processes": [],
        },
    )
    monkeypatch.setattr(supervisor, "collect", lambda *a, **k: {"files": 1})
    monkeypatch.setattr(supervisor, "terminate_verified", lambda v: True)
    assert supervisor.poll() is True


def test_lost_runner_stops_preserving_after_two_missing_polls(
    supervisor, monkeypatch
):
    supervisor.state["saw_running_process"] = True
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a: {"status": {"status": "running"}, "processes": []},
    )
    stops = []
    monkeypatch.setattr(
        supervisor, "stop_preserving", lambda why: stops.append(why) or True
    )
    monkeypatch.setattr(
        supervisor,
        "collect",
        lambda *a, **k: pytest.fail("absence is not terminal"),
    )
    assert supervisor.poll() is False
    assert supervisor.poll() is True
    assert len(stops) == 1


def test_preparation_gap_requests_continuation_without_stopping(
    supervisor, monkeypatch
):
    supervisor.state["saw_running_process"] = True
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a: {
            "status": {"status": "awaiting_preparation"},
            "processes": [],
        },
    )
    monkeypatch.setattr(
        supervisor,
        "stop_preserving",
        lambda *a: pytest.fail("intentional phase gap"),
    )
    assert supervisor.poll() is False
    assert supervisor.poll() is False
    assert supervisor.state["status"] == "awaiting_continuation"
    assert supervisor.state["action_required"] is True


def test_unhealthy_remote_guard_requires_action(supervisor, monkeypatch):
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a: {
            "status": {"status": "running"},
            "processes": [{"pid": 12}],
            "guard": {"pod_id": "abcdefghijkl"},
            "guard_health": {"healthy": False},
        },
    )
    assert supervisor.poll() is False
    assert supervisor.state["guard_healthy"] is False
    assert supervisor.state["action_required"] is True


def test_reviewed_sixty_hour_override_keeps_spend_cap(supervisor):
    receipt = supervisor.receipt
    created = dt.datetime.fromisoformat(receipt["created_utc"])
    path = Path(supervisor.args.state).parent / "override.json"
    value = {
        "pod_id": supervisor.pod_id,
        "created_utc": receipt["created_utc"],
        "receipt_sha256": watch.base.sha(supervisor.args.receipt),
        "max_spend_usd": 40,
        "hourly_rate_upper_bound_usd": 0.62,
        "storage_reserve_usd": 2.5,
        "deadline_utc": (created + dt.timedelta(hours=60)).isoformat(),
    }
    path.write_text(json.dumps(value))
    supervisor.args.operational_override = str(path)
    supervisor.args.operational_override_sha256 = watch.base.sha(path)
    revised = watch.Supervisor(supervisor.args, supervisor.client)
    assert revised.deadline == created + dt.timedelta(hours=60)
    assert revised.state["max_spend_usd"] == 40
    value["deadline_utc"] = (created + dt.timedelta(hours=62)).isoformat()
    path.write_text(json.dumps(value))
    supervisor.args.operational_override_sha256 = watch.base.sha(path)
    with pytest.raises(ValueError, match="cost bound"):
        watch.Supervisor(supervisor.args, supervisor.client)


def test_override_cannot_bypass_changed_receipt(supervisor):
    path = Path(supervisor.args.state).parent / "override.json"
    value = {
        "pod_id": supervisor.pod_id,
        "created_utc": supervisor.receipt["created_utc"],
        "receipt_sha256": "0" * 64,
        "max_spend_usd": 40,
        "hourly_rate_upper_bound_usd": 0.62,
    }
    path.write_text(json.dumps(value))
    supervisor.args.operational_override = str(path)
    supervisor.args.operational_override_sha256 = watch.base.sha(path)
    with pytest.raises(ValueError, match="owned pod or cost bounds"):
        watch.Supervisor(supervisor.args, supervisor.client)
