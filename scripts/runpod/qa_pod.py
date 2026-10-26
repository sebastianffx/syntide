#!/usr/bin/env python3
"""Preview, create, or inspect a question-answering training instance.

Examples:
    python3 scripts/runpod/qa_pod.py create
    python3 scripts/runpod/qa_pod.py create --execute
    python3 scripts/runpod/qa_pod.py status --pod-id <id>

Credentials are read from macOS Keychain into memory. They never appear in
command arguments or saved configuration files. Creation supplies only the
existing public login key and leaves account settings unchanged. The provider's
stopAfter setting cannot be verified here; the local supervisor enforces the
deadline and spending limit. Stopping preserves the volume. Collect outputs
before deleting the instance to end its storage charges.

Provider references:
    https://graphql-spec.runpod.io/
    https://github.com/runpod/containers/blob/main/container-template/start.sh
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
GPU = "NVIDIA A100-SXM4-80GB"
KEYCHAIN_SERVICE = "syntide-runpod-api-key"
ACCOUNT_QUERY = """query {
  myself { clientBalance currentSpendPerHr
    pods { id name desiredStatus costPerHr }
  }
}"""
CREATE_MUTATION = (
    "mutation($input: PodFindAndDeployOnDemandInput!) {\n"
    "  podFindAndDeployOnDemand(input: $input) {\n"
    "    id name desiredStatus costPerHr imageName gpuCount "
    "volumeInGb containerDiskInGb\n"
    "  }\n"
    "}"
)
POD_QUERY = (
    "query($input: PodFilter!) {\n"
    "  pod(input: $input) {\n"
    "    id name desiredStatus costPerHr imageName gpuCount\n"
    "    runtime { uptimeInSeconds ports { ip isIpPublic "
    "privatePort publicPort type } }\n"
    "  }\n"
    "}"
)


def utc_now():
    return dt.datetime.now(dt.UTC)


def iso(value):
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def read_public_key(key_path):
    key_path = Path(key_path).expanduser().resolve()
    public_path = Path(str(key_path) + ".pub")
    public = public_path.read_text().strip()
    fields = public.split()
    if (
        len(fields) < 2
        or "\n" in public
        or fields[0] not in {"ssh-ed25519", "ssh-rsa"}
    ):
        raise ValueError("Expected one existing Ed25519 or RSA public SSH key")
    # Verify that the supplied private/public pair really matches. Do not log
    # it.
    derived = subprocess.run(
        ["ssh-keygen", "-y", "-P", "", "-f", str(key_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if derived.returncode != 0 or derived.stdout.split()[:2] != fields[:2]:
        raise ValueError(
            "Existing SSH key pair is unavailable or does not match"
        )
    return " ".join(fields[:2]), key_path


def build_input(
    public_key,
    *,
    name="syntide-qa-arms-seed0",
    cloud="COMMUNITY",
    guard_hours=24,
    now=None,
):
    if cloud not in {"COMMUNITY", "SECURE"}:
        raise ValueError("Cloud must be COMMUNITY or SECURE")
    if not 0 < guard_hours <= 24:
        raise ValueError(
            "Guard must be greater than zero and no more than 24 hours"
        )
    now = now or utc_now()
    return {
        "cloudType": cloud,
        "gpuTypeId": GPU,
        "gpuCount": 1,
        "name": name,
        "imageName": IMAGE,
        "containerDiskInGb": 40,
        "volumeInGb": 100,
        "volumeMountPath": "/workspace",
        "minVcpuCount": 8,
        "minMemoryInGb": 64,
        "minCudaVersion": "12.8",
        "supportPublicIp": True,
        "ports": "22/tcp",
        "startSsh": True,
        "startJupyter": False,
        "dockerArgs": "",
        "env": [{"key": "PUBLIC_KEY", "value": public_key}],
        "stopAfter": iso(now + dt.timedelta(hours=guard_hours)),
    }


def sanitized_input(payload):
    safe = {k: v for k, v in payload.items() if k != "env"}
    safe["ssh_public_key_sha256"] = hashlib.sha256(
        payload["env"][0]["value"].encode()
    ).hexdigest()
    return safe


class RunPod:
    def __init__(self):
        result = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode or not result.stdout.strip():
            raise RuntimeError(
                "RunPod API credential is unavailable in macOS Keychain"
            )
        self._key = result.stdout.strip()

    def request(self, url, body=None):
        encoded = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "Authorization": "Bearer " + self._key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            # Never echo remote error bodies: APIs can echo request credentials.
            raise RuntimeError(
                f"RunPod API HTTP {error.code}; inspect state before retrying"
            ) from None
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(
                f"RunPod API {type(error).__name__}; outcome may be "
                f"unknown; inspect state before retrying"
            ) from None
        return result

    def query(self, query, variables=None):
        result = self.request(
            "https://api.runpod.io/graphql",
            {"query": query, "variables": variables or {}},
        )
        if result.get("errors"):
            messages = "; ".join(
                str(error.get("message", "API error"))
                for error in result["errors"]
            )
            raise RuntimeError(
                "RunPod GraphQL: " + messages.replace(self._key, "[REDACTED]")
            )
        return result["data"]

    def catalog(self, cloud="COMMUNITY"):
        query = urllib.parse.urlencode(
            {
                "count": 1,
                "include": "AVAILABILITY",
                "product": "POD",
                "cloud": cloud,
                "minCudaVersion": "12.8",
            }
        )
        return self.request(
            "https://api.runpod.io/v2/catalog/gpus/"
            + urllib.parse.quote(GPU, safe="")
            + "?"
            + query,
        )


def validate_preflight(
    account, catalog, max_gpu_rate, guard_hours, cloud="COMMUNITY"
):
    if account.get("pods"):
        raise ValueError(
            "Refusing creation: RunPod account already contains a pod"
        )
    if account.get("currentSpendPerHr") != 0:
        raise ValueError("Refusing creation: existing hourly spend is not zero")
    if catalog.get("availability") not in {"LOW", "MEDIUM", "HIGH"}:
        raise ValueError(
            f"Requested {cloud.lower()} A100 configuration currently "
            f"has no capacity"
        )
    rate = catalog.get("price", {}).get(cloud.lower())
    if not isinstance(rate, (int, float)) or not 0 < rate <= max_gpu_rate:
        raise ValueError(
            "Current GPU catalog price exceeds the requested hourly ceiling"
        )
    # $0.10/GB/month is a planning allowance, not a billing guarantee.
    allowance = (rate + 140 * 0.10 / 720) * guard_hours
    if account.get("clientBalance", 0) < allowance:
        raise ValueError(
            "Insufficient account balance for the full guard interval"
        )
    return {
        "gpu_hourly_rate": rate,
        "compute_guard_usd": rate * guard_hours,
        "guard_with_storage_allowance_usd": allowance,
    }


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def create(args, client):
    public, key_path = read_public_key(args.ssh_key)
    payload = build_input(
        public, name=args.name, cloud=args.cloud, guard_hours=args.guard_hours
    )
    account = client.query(ACCOUNT_QUERY)["myself"]
    catalog = client.catalog(args.cloud)
    costs = validate_preflight(
        account, catalog, args.max_gpu_rate, args.guard_hours, args.cloud
    )
    receipt = {
        "checked_utc": iso(utc_now()),
        "execute": args.execute,
        "account_before": account,
        "request": sanitized_input(payload),
        "ssh_key_path": str(key_path),
        "cost_plan": costs,
        "guard": "API stopAfter requested, enforcement unverified; local "
        "supervisor required",
    }
    output = Path(args.output)
    if not args.execute:
        return receipt
    if output.exists():
        raise ValueError(
            "Receipt already exists; inspect it and account state "
            "before another create"
        )
    # Write intent first. A network timeout must not trigger an automatic
    # duplicate.
    receipt["status"] = "create_pending"
    save(output, receipt)
    pod = client.query(CREATE_MUTATION, {"input": payload})[
        "podFindAndDeployOnDemand"
    ]
    receipt.update(
        status="created", pod=pod, created_receipt_utc=iso(utc_now())
    )
    receipt["hourly_rate_matches"] = pod["costPerHr"] <= args.max_gpu_rate
    save(output, receipt)
    return receipt


def ssh_argv(pod, key_path):
    ports = (pod.get("runtime") or {}).get("ports") or []
    for port in ports:
        if (
            port.get("privatePort") == 22
            and port.get("isIpPublic")
            and port.get("type") == "tcp"
        ):
            return [
                "ssh",
                "-i",
                str(Path(key_path).expanduser().resolve()),
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=15",
                "-p",
                str(port["publicPort"]),
                "root@" + port["ip"],
            ]
    return None


def status(args, client):
    account = client.query(ACCOUNT_QUERY)["myself"]
    result = {"checked_utc": iso(utc_now()), "account": account}
    if args.pod_id:
        pod = client.query(POD_QUERY, {"input": {"podId": args.pod_id}})["pod"]
        result["pod"] = pod
        argv = ssh_argv(pod, args.ssh_key) if pod else None
        result["ssh_argv"] = argv
        result["ssh_command"] = shlex.join(argv) if argv else None
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser(
        "create", help="preview unless --execute is supplied"
    )
    create_parser.add_argument("--execute", action="store_true")
    create_parser.add_argument("--name", default="syntide-qa-arms-seed0")
    create_parser.add_argument(
        "--cloud", choices=("COMMUNITY", "SECURE"), default="COMMUNITY"
    )
    create_parser.add_argument("--guard-hours", type=float, default=24)
    create_parser.add_argument("--max-gpu-rate", type=float, default=1.39)
    create_parser.add_argument(
        "--output", default="outputs/qa-runs/pod-create.json"
    )
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--pod-id")
    for child in (create_parser, status_parser):
        child.add_argument("--ssh-key", default="~/.ssh/id_runpod_syntide")
    return result


def main():
    args = parser().parse_args()
    try:
        result = {"create": create, "status": status}[args.command](
            args, RunPod()
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (RuntimeError, ValueError, OSError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
