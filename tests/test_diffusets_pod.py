"""Paid-resource isolation and durable guard checks; never contact RunPod."""

import datetime as dt
import importlib.util
import io
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
pod = module("diffusets_pod")
guard = module("diffusets_guard")
installer = module("diffusets_install_guard")


@pytest.fixture
def inputs():
    old = {
        "status": "created",
        "pod": {"id": pod.OLD_POD_ID, "name": "original"},
        "account_before": {"clientBalance": 90},
    }
    account = {
        "pods": [
            {
                "id": pod.OLD_POD_ID,
                "name": "original",
                "desiredStatus": "RUNNING",
                "costPerHr": 1.79,
            }
        ],
        "currentSpendPerHr": 1.809,
        "clientBalance": 66,
    }
    offer = {"id": pod.GPU, "availability": "LOW", "price": {"secure": 0.59}}
    return account, offer, old


def validate(inputs, **kwargs):
    return pod.validate(
        *inputs,
        max_rate=0.59,
        hours=kwargs.get("hours", 48),
        cap=kwargs.get("cap", 40),
    )


def test_allows_only_original_owned_pod_and_reserves_its_budget(inputs):
    result = validate(inputs)
    assert result["original_study_reserved_usd"] == 26
    assert result["48h_or_requested_cost_upper_bound_usd"] < 30
    assert "account delta" in result["attribution"]


@pytest.mark.parametrize(
    "change",
    ["unknown", "renamed", "overspend", "no_capacity", "price", "balance"],
)
def test_rejects_unsafe_preflight(inputs, change):
    account, offer, _ = inputs
    if change == "unknown":
        account["pods"].append({"id": "other", "name": "other"})
    if change == "renamed":
        account["pods"][0]["name"] = "unrecognized"
    if change == "overspend":
        account["currentSpendPerHr"] = 5
    if change == "no_capacity":
        offer["availability"] = "NONE"
    if change == "price":
        offer["price"]["secure"] = 0.60
    if change == "balance":
        account["clientBalance"] = 1
    with pytest.raises(ValueError):
        validate(inputs)


@pytest.mark.parametrize(
    "hours,cap", [(49, 40), (0, 40), (48, 41), (48, 20), (float("nan"), 40)]
)
def test_bounds_cannot_be_bypassed(inputs, hours, cap):
    with pytest.raises(ValueError):
        validate(inputs, hours=hours, cap=cap)


def test_payload_is_exactly_one_a40_without_account_key_or_deadline_claim():
    payload = pod.build_input("ssh-ed25519 FAKE", "unique")
    assert payload["gpuTypeId"] == "NVIDIA A40"
    assert payload["gpuCount"] == 1
    assert payload["cloudType"] == "SECURE"
    assert payload["env"] == [
        {"key": "PUBLIC_KEY", "value": "ssh-ed25519 FAKE"}
    ]
    assert "stopAfter" not in payload and "terminateAfter" not in payload


def test_unknown_creation_keeps_intent_and_refuses_duplicate(
    inputs, monkeypatch, tmp_path
):
    account, offer, old = inputs
    oldpath = tmp_path / "old.json"
    oldpath.write_text(json.dumps(old))
    output = tmp_path / "receipt.json"
    args = SimpleNamespace(
        output=output,
        original_receipt=oldpath,
        max_gpu_rate=0.59,
        guard_hours=48,
        max_spend_usd=40,
        ssh_key="/fake",
        execute=True,
    )
    monkeypatch.setattr(pod, "catalog", lambda client: offer)
    monkeypatch.setattr(
        qa_pod, "read_public_key", lambda p: ("ssh-ed25519 FAKE", Path(p))
    )

    class Client:
        def __init__(self):
            self.creates = 0

        def query(self, query, variables=None):
            if query == qa_pod.ACCOUNT_QUERY:
                return {"myself": account}
            assert query == qa_pod.CREATE_MUTATION
            self.creates += 1
            raise RuntimeError("unknown outcome")

    client = Client()
    with pytest.raises(RuntimeError):
        pod.create(args, client)
    assert json.loads(output.read_text())["status"] == "create_pending"
    with pytest.raises(ValueError):
        pod.create(args, client)
    assert client.creates == 1


def test_platform_credential_must_match_own_pod():
    raw = b"RUNPOD_POD_ID=abcdefghijkl\0RUNPOD_API_KEY=synthetic-test-only\0"
    assert (
        guard.runtime_environment("abcdefghijkl", raw)["RUNPOD_API_KEY"]
        == "synthetic-test-only"
    )
    with pytest.raises(ValueError):
        guard.runtime_environment(pod.OLD_POD_ID, raw)
    with pytest.raises(ValueError):
        guard.runtime_environment(
            "abcdefghijkl", b"RUNPOD_POD_ID=abcdefghijkl\0"
        )


def test_guard_authentication_queries_only_its_own_pod(monkeypatch):
    monkeypatch.setattr(
        guard, "runtime_environment", lambda _: {"RUNPOD_API_KEY": "fake-test"}
    )
    monkeypatch.setattr(guard.shutil, "which", lambda _: "/usr/bin/runpodctl")
    monkeypatch.setattr(
        guard, "cli_commands", lambda *a: (["get", "pod"], ["stop", "pod"])
    )
    requests = []

    def response(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(b'{"data":{"pod":{"id":"abcdefghijkl"}}}')

    monkeypatch.setattr(guard.urllib.request, "urlopen", response)
    assert guard.probe("abcdefghijkl")[1] == ["stop", "pod"]
    assert requests[0]["variables"] == {"input": {"podId": "abcdefghijkl"}}
    assert (
        "myself" not in requests[0]["query"]
        and "mutation" not in requests[0]["query"]
    )


def test_installer_refuses_original_pod_before_any_ssh():
    with pytest.raises(ValueError, match="Original study"):
        installer.guard_config(
            {
                "status": "created",
                "execute": True,
                "pod": {"id": pod.OLD_POD_ID},
            }
        )


def test_deadline_requires_utc():
    assert guard.parse_utc("2026-10-25T12:00:00Z").tzinfo == dt.UTC
    for invalid in ("2026-10-25T12:00:00", "2026-10-25T12:00:00+02:00"):
        with pytest.raises(ValueError):
            guard.parse_utc(invalid)


def test_guard_armed_does_not_stop_early(monkeypatch, tmp_path):
    now = dt.datetime.now(dt.UTC)
    config = {
        "pod_id": "abcdefghijkl",
        "created_utc": now.isoformat(),
        "deadline_utc": (now + dt.timedelta(hours=48)).isoformat(),
        "hourly_rate_upper_bound_usd": 0.61,
        "max_spend_usd": 40,
        "state_path": str(tmp_path / "state.json"),
    }
    monkeypatch.setattr(
        guard, "probe", lambda _: ("runpodctl", ["stop", "pod"], {})
    )
    monkeypatch.setattr(
        guard.subprocess, "run", lambda *a, **k: pytest.fail("premature stop")
    )
    assert guard.guard(config, once=True)["status"] == "armed"


def test_guard_only_stops_own_pod_at_deadline_preserving_data(
    monkeypatch, tmp_path
):
    now = dt.datetime.now(dt.UTC)
    config = {
        "pod_id": "abcdefghijkl",
        "created_utc": (now - dt.timedelta(hours=47)).isoformat(),
        "deadline_utc": (now + dt.timedelta(minutes=5)).isoformat(),
        "hourly_rate_upper_bound_usd": 0.61,
        "max_spend_usd": 40,
        "state_path": str(tmp_path / "state.json"),
    }
    monkeypatch.setattr(
        guard, "probe", lambda _: ("runpodctl", ["stop", "pod"], {})
    )
    calls = []
    monkeypatch.setattr(
        guard.subprocess,
        "run",
        lambda argv, **kw: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    result = guard.guard(config, once=True)
    assert calls == [["runpodctl", "stop", "pod", "abcdefghijkl"]]
    assert result["status"] == "stop_requested_retrying_until_shutdown"
    assert pod.OLD_POD_ID not in str(calls)
