"""Safety checks for paid pod creation; no tests make network requests."""

import datetime as dt
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "qa_pod", Path(__file__).resolve().parents[1] / "scripts/runpod/qa_pod.py"
)
qa_pod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa_pod)


def test_payload_is_single_ondemand_gpu_with_stop_and_only_public_key():
    payload = qa_pod.build_input(
        "ssh-ed25519 TEST", now=dt.datetime(2026, 10, 22, tzinfo=dt.UTC)
    )
    assert payload["stopAfter"] == "2026-10-23T00:00:00Z"
    assert payload["cloudType"] == "COMMUNITY"
    assert payload["gpuCount"] == 1
    assert payload["ports"] == "22/tcp"
    assert payload["volumeInGb"] == 100
    assert "terminateAfter" not in payload
    assert "env" not in qa_pod.sanitized_input(payload)
    assert "TEST" not in str(qa_pod.sanitized_input(payload))


@pytest.mark.parametrize("hours", [0, -1, 24.01, float("nan")])
def test_guard_cannot_be_disabled_or_extended(hours):
    with pytest.raises(ValueError):
        qa_pod.build_input("ssh-ed25519 TEST", guard_hours=hours)


@pytest.mark.parametrize(
    "account,catalog",
    [
        (
            {
                "pods": [{"id": "existing"}],
                "currentSpendPerHr": 0,
                "clientBalance": 90,
            },
            {"availability": "LOW", "price": {"community": 1.39}},
        ),
        (
            {"pods": [], "currentSpendPerHr": 0.1, "clientBalance": 90},
            {"availability": "LOW", "price": {"community": 1.39}},
        ),
        (
            {"pods": [], "currentSpendPerHr": 0, "clientBalance": 90},
            {"availability": "NONE", "price": {"community": 1.39}},
        ),
        (
            {"pods": [], "currentSpendPerHr": 0, "clientBalance": 90},
            {"availability": "LOW", "price": {"community": 1.40}},
        ),
        (
            {"pods": [], "currentSpendPerHr": 0, "clientBalance": 2},
            {"availability": "LOW", "price": {"community": 1.39}},
        ),
    ],
)
def test_preflight_refuses_existing_cost_insufficient_balance_or_wrong_offer(
    account, catalog
):
    with pytest.raises(ValueError):
        qa_pod.validate_preflight(account, catalog, 1.39, 24)


def test_preview_never_creates_or_saves(monkeypatch, tmp_path):
    class Client:
        def query(self, query):
            assert query == qa_pod.ACCOUNT_QUERY
            return {
                "myself": {
                    "pods": [],
                    "currentSpendPerHr": 0,
                    "clientBalance": 90,
                }
            }

        def catalog(self, cloud):
            assert cloud == "COMMUNITY"
            return {"availability": "LOW", "price": {"community": 1.39}}

    monkeypatch.setattr(
        qa_pod, "read_public_key", lambda path: ("ssh-ed25519 TEST", Path(path))
    )
    args = SimpleNamespace(
        ssh_key="/fake/key",
        name="test",
        cloud="COMMUNITY",
        guard_hours=24,
        max_gpu_rate=1.39,
        execute=False,
        output=tmp_path / "receipt.json",
    )
    receipt = qa_pod.create(args, Client())
    assert receipt["execute"] is False
    assert receipt["cost_plan"]["compute_guard_usd"] == pytest.approx(33.36)
    assert not args.output.exists()


def test_secure_cloud_uses_explicit_rate_limit():
    account = {"pods": [], "currentSpendPerHr": 0, "clientBalance": 90}
    catalog = {
        "availability": "LOW",
        "price": {"community": 1.39, "secure": 1.79},
    }
    with pytest.raises(ValueError):
        qa_pod.validate_preflight(account, catalog, 1.39, 24, "SECURE")
    costs = qa_pod.validate_preflight(account, catalog, 1.79, 24, "SECURE")
    assert costs["compute_guard_usd"] == pytest.approx(42.96)
    assert (
        qa_pod.build_input("ssh-ed25519 TEST", cloud="SECURE")["cloudType"]
        == "SECURE"
    )


def test_ssh_uses_only_public_tcp_port_and_identity_file():
    pod = {
        "runtime": {
            "ports": [
                {
                    "privatePort": 22,
                    "publicPort": 123,
                    "ip": "10.0.0.1",
                    "type": "tcp",
                    "isIpPublic": False,
                },
                {
                    "privatePort": 22,
                    "publicPort": 456,
                    "ip": "192.0.2.1",
                    "type": "tcp",
                    "isIpPublic": True,
                },
            ]
        }
    }
    argv = qa_pod.ssh_argv(pod, "/fake/key")
    assert argv[-3:] == ["-p", "456", "root@192.0.2.1"]
    assert "StrictHostKeyChecking=accept-new" in argv
    assert qa_pod.ssh_argv({"runtime": None}, "/fake/key") is None
