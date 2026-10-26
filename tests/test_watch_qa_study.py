"""Deletion and spending checks work without live cloud resources."""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/runpod"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "watch_qa_study", SCRIPTS / "watch_qa_study.py"
)
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


def manifest_for(path):
    return {
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        }
    }


def test_collection_requires_every_byte_and_rejects_path_traversal(tmp_path):
    artifact = tmp_path / "best.pt"
    artifact.write_bytes(b"checkpoint")
    manifest = manifest_for(artifact)
    assert watch.verify_local(tmp_path, manifest)["files"] == 1
    artifact.write_bytes(b"CHECKPOINT")
    with pytest.raises(ValueError, match="checksum"):
        watch.verify_local(tmp_path, manifest)
    with pytest.raises(ValueError, match="Unsafe"):
        watch.verify_local(
            tmp_path, {"files": {"../best.pt": manifest["files"]["best.pt"]}}
        )
    with pytest.raises(ValueError, match="Empty"):
        watch.verify_local(tmp_path, {"files": {}})


def make_supervisor(tmp_path, *, deadline_hours=24, execute=True):
    pod = {
        "id": "ours",
        "name": "study",
        "imageName": "image",
        "desiredStatus": "RUNNING",
        "runtime": {
            "ports": [
                {
                    "privatePort": 22,
                    "publicPort": 123,
                    "isIpPublic": True,
                    "ip": "192.0.2.1",
                    "type": "tcp",
                }
            ]
        },
    }
    receipt = {
        "status": "created",
        "execute": True,
        "pod": pod,
        "ssh_key_path": "/fake/key",
        "request": {
            "stopAfter": (
                dt.datetime.now(dt.UTC) + dt.timedelta(hours=deadline_hours)
            ).isoformat()
        },
        "account_before": {"clientBalance": 90},
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    args = SimpleNamespace(
        receipt=receipt_path,
        state=tmp_path / "state.json",
        output=tmp_path / "study",
        execute=execute,
        deadline_margin_seconds=900,
        max_spend_usd=50,
        collection_timeout_seconds=1800,
    )

    class Client:
        def __init__(self):
            self.calls = []

        def query(self, query, variables=None):
            self.calls.append((query, variables))
            if query == watch.qa_pod.ACCOUNT_QUERY:
                return {
                    "myself": {
                        "clientBalance": 89,
                        "pods": [pod],
                        "currentSpendPerHr": 1.79,
                    }
                }
            if query == watch.qa_pod.POD_QUERY:
                assert variables == {"input": {"podId": "ours"}}
                return {"pod": pod}
            if query == watch.STOP:
                assert variables == {"input": {"podId": "ours"}}
                pod["desiredStatus"] = "EXITED"
                return {"podStop": pod}
            raise AssertionError("Unexpected mutation")

    client = Client()
    return watch.Supervisor(args, client), client


def test_collection_failure_never_terminates(monkeypatch, tmp_path):
    supervisor, client = make_supervisor(tmp_path)
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a, **kw: {"status": {"status": "complete"}, "processes": []},
    )
    monkeypatch.setattr(
        supervisor,
        "collect",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad checksum")),
    )
    assert supervisor.poll() is False
    assert all(
        query not in {watch.STOP, watch.TERMINATE} for query, _ in client.calls
    )
    assert supervisor.state["status"] == "retryable_collection_or_probe_error"


def test_active_work_near_deadline_is_stopped_not_deleted(
    monkeypatch, tmp_path
):
    supervisor, client = make_supervisor(tmp_path, deadline_hours=0.1)
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a, **kw: {"status": {"status": "running"}, "processes": [123]},
    )
    assert supervisor.poll() is True
    assert any(query == watch.STOP for query, _ in client.calls)
    assert all(query != watch.TERMINATE for query, _ in client.calls)
    assert supervisor.state["action_required"] is True


def test_read_only_watch_never_stops(monkeypatch, tmp_path):
    supervisor, client = make_supervisor(
        tmp_path, deadline_hours=0.1, execute=False
    )
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a, **kw: {"status": {"status": "running"}, "processes": [123]},
    )
    assert supervisor.poll() is True
    assert all(
        query not in {watch.STOP, watch.TERMINATE} for query, _ in client.calls
    )


def test_termination_without_verified_manifest_is_rejected(tmp_path):
    supervisor, client = make_supervisor(tmp_path)
    with pytest.raises(ValueError, match="No verified"):
        supervisor.terminate_verified(None)
    assert client.calls == []


def test_wrong_live_pod_identity_never_authorizes_action(tmp_path):
    supervisor, client = make_supervisor(tmp_path)
    supervisor.receipt["pod"]["name"] = "another"
    with pytest.raises(ValueError, match="identity"):
        supervisor.poll()
    assert all(
        query not in {watch.STOP, watch.TERMINATE} for query, _ in client.calls
    )


def test_tampered_local_copy_prevents_termination(tmp_path):
    supervisor, client = make_supervisor(tmp_path)
    supervisor.args.output.mkdir()
    checkpoint = supervisor.args.output / "best.pt"
    checkpoint.write_bytes(b"complete")
    inventory = tmp_path / "manifest.json"
    inventory.write_text(json.dumps(manifest_for(checkpoint)))
    verification = {
        "files": 1,
        "manifest_path": str(inventory),
        "manifest_sha256": watch.sha(inventory),
    }
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        supervisor.terminate_verified(verification)
    assert all(query != watch.TERMINATE for query, _ in client.calls)


def test_verified_collection_terminates_only_receipt_pod(monkeypatch, tmp_path):
    supervisor, client = make_supervisor(tmp_path)
    supervisor.args.output.mkdir()
    checkpoint = supervisor.args.output / "best.pt"
    checkpoint.write_bytes(b"complete checkpoint")
    inventory = tmp_path / "manifest.json"
    inventory.write_text(json.dumps(manifest_for(checkpoint)))
    verification = {
        "files": 1,
        "manifest_path": str(inventory),
        "manifest_sha256": watch.sha(inventory),
    }
    pods = [{"id": "ours"}, None]
    monkeypatch.setattr(supervisor, "own_pod", lambda: pods.pop(0))

    def query(operation, variables=None):
        client.calls.append((operation, variables))
        if operation == watch.TERMINATE:
            assert variables == {"input": {"podId": "ours"}}
            assert supervisor.state["status"] == "collected_verified"
            return {"podTerminate": None}
        assert operation == watch.qa_pod.ACCOUNT_QUERY
        return {
            "myself": {"clientBalance": 80, "currentSpendPerHr": 0, "pods": []}
        }

    monkeypatch.setattr(client, "query", query)
    assert supervisor.terminate_verified(verification) is True
    assert supervisor.state["status"] == "terminated_after_verified_collection"
    assert supervisor.state["action_required"] is False


@pytest.mark.parametrize(
    "value", ["2026-10-24T01:42:38.340Z", "2026-10-24T01:42:38.340+00:00"]
)
def test_deadline_cli_preserves_utc_instant_and_fraction(value):
    args = watch.parser().parse_args(["--deadline-utc", value])
    assert args.deadline_utc == dt.datetime(
        2026, 10, 24, 1, 42, 38, 340000, tzinfo=dt.UTC
    )
    assert args.max_spend_usd == 50


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-24",
        "2026-10-24T01:42:38",
        "2026-10-24T01:42:38+02:00",
        "2026-10-24T01:42:38-00:00",
        "2026-13-24T01:42:38Z",
        "tomorrow",
    ],
)
def test_deadline_rejects_implicit_or_non_utc_time(value):
    with pytest.raises(argparse.ArgumentTypeError):
        watch.parse_deadline_utc(value)


def test_default_deadline_uses_receipt_and_records_unverified_api_guard(
    tmp_path,
):
    supervisor, client = make_supervisor(tmp_path)
    assert supervisor.deadline == watch.parse_deadline_utc(
        supervisor.receipt["request"]["stopAfter"]
    )
    assert supervisor.state["deadline_source"] == "receipt_stopAfter"
    assert supervisor.state["api_stop_after_enforcement"] == "unverified"
    assert client.calls == []


def test_override_preserves_receipt_and_replaces_near_deadline(
    monkeypatch, tmp_path
):
    original, client = make_supervisor(tmp_path, deadline_hours=0.1)
    receipt_bytes = original.args.receipt.read_bytes()
    override = dt.datetime.now(dt.UTC) + dt.timedelta(hours=28)
    original.args.deadline_utc = override
    supervisor = watch.Supervisor(original.args, client)
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a, **kw: {"status": {"status": "running"}, "processes": [1]},
    )
    assert supervisor.poll() is False
    assert supervisor.state["status"] == "watching"
    saved = json.loads(supervisor.args.state.read_text())
    assert saved["deadline_source"] == "command_line_override"
    assert watch.parse_deadline_utc(saved["effective_deadline_utc"]) == override
    assert saved["max_spend_usd"] == 50
    assert original.args.receipt.read_bytes() == receipt_bytes
    assert all(
        query not in {watch.STOP, watch.TERMINATE} for query, _ in client.calls
    )


def test_deadline_override_does_not_relax_spending_cap(monkeypatch, tmp_path):
    original, client = make_supervisor(tmp_path)
    original.args.deadline_utc = dt.datetime.now(dt.UTC) + dt.timedelta(
        hours=28
    )
    supervisor = watch.Supervisor(original.args, client)
    original_query = client.query

    def query(operation, variables=None):
        result = original_query(operation, variables)
        if operation == watch.qa_pod.ACCOUNT_QUERY:
            result["myself"]["clientBalance"] = 39
        return result

    monkeypatch.setattr(client, "query", query)
    monkeypatch.setattr(
        supervisor,
        "ssh",
        lambda *a, **kw: {"status": {"status": "running"}, "processes": [1]},
    )
    assert supervisor.poll() is True
    assert any(operation == watch.STOP for operation, _ in client.calls)
    assert all(operation != watch.TERMINATE for operation, _ in client.calls)
