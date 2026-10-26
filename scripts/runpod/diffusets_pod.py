#!/usr/bin/env python3
"""Preview or create one separate A40 instance for DiffuSETS.

Creation requires --execute. Only the recorded original question-answering
instance may coexist. A pending creation receipt prevents an automatic retry
when the provider's response is ambiguous. Cost estimates belong to this
instance; changes in the shared account balance do not measure its cost.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import urllib.parse
import uuid
from pathlib import Path

import qa_pod

GPU = "NVIDIA A40"
OLD_POD_ID = "y9z0q4fgr4zvn6"
DIRECTORY = "outputs/qa-runs/diffusets-extension"
SOURCES = [
    "https://docs.runpod.io/pods/manage-pods",
    "https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpodctl/SKILL.md",
    "https://github.com/runpod/runpodctl/releases/tag/v2.12.0",
    "https://raw.githubusercontent.com/runpod/runpodctl/v1.14.15/api/query.go",
]


def catalog(client):
    query = urllib.parse.urlencode(
        {
            "count": 1,
            "include": "AVAILABILITY",
            "product": "POD",
            "cloud": "SECURE",
            "minCudaVersion": "12.8",
        }
    )
    return client.request(
        "https://api.runpod.io/v2/catalog/gpus/"
        + urllib.parse.quote(GPU, safe="")
        + "?"
        + query
    )


def validate(account, offer, old_receipt, *, max_rate, hours, cap):
    if (
        not math.isfinite(hours)
        or not 0 < hours <= 48
        or not math.isfinite(cap)
        or not 0 < cap <= 40
    ):
        raise ValueError("Extension is limited to 48h and $40")
    if (
        old_receipt.get("pod", {}).get("id") != OLD_POD_ID
        or old_receipt.get("status") != "created"
    ):
        raise ValueError("Original owned pod receipt is invalid")
    old = old_receipt["pod"]
    for pod in account.get("pods", []):
        if pod.get("id") != OLD_POD_ID or pod.get("name") != old.get("name"):
            raise ValueError(
                "Unknown/existing extension pod found; inspect before creating"
            )
    known_compute = sum(
        pod.get("costPerHr", 0)
        for pod in account.get("pods", [])
        if pod.get("desiredStatus") == "RUNNING"
    )
    if account.get("currentSpendPerHr", 0) > known_compute + 0.05:
        raise ValueError("Account has unexplained existing hourly charges")
    rate = offer.get("price", {}).get("secure")
    if offer.get("id") != GPU or offer.get("availability") not in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }:
        raise ValueError("Secure A40 capacity unavailable")
    if not isinstance(rate, (int, float)) or not 0 < rate <= max_rate <= 0.59:
        raise ValueError("A40 rate exceeds authorized ceiling")
    # Round above the observed $0.020/h disk increment, rather than labelling a
    # monthly-rate approximation below billed rounding as an upper bound.
    hourly_upper = rate + 0.03
    bounded_cost = hourly_upper * hours
    if bounded_cost > cap:
        raise ValueError("Requested duration exceeds extension spending cap")
    old_spent = max(
        0,
        old_receipt["account_before"]["clientBalance"]
        - account["clientBalance"],
    )
    old_reserve = max(0, 50 - old_spent) if account.get("pods") else 0
    if account["clientBalance"] < bounded_cost + old_reserve:
        raise ValueError(
            "Insufficient balance for extension plus original study reserve"
        )
    return {
        "gpu_hourly_rate_usd": rate,
        "hourly_rate_upper_bound_usd": hourly_upper,
        "48h_or_requested_cost_upper_bound_usd": bounded_cost,
        "max_spend_usd": cap,
        "original_study_reserved_usd": old_reserve,
        "attribution": (
            "GPU rate plus conservative disk allowance times own-pod "
            "elapsed time; "
            "account delta includes overlapping original pod and is "
            "not extension spend"
        ),
    }


def build_input(public, name):
    # Reuse proven image/SSH/disk settings; never rely on unsupported stopAfter.
    result = qa_pod.build_input(public, name=name, cloud="SECURE")
    result.update(gpuTypeId=GPU, minVcpuCount=8, minMemoryInGb=48)
    result.pop("stopAfter")
    return result


def create(args, client):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise ValueError(
                "Create receipt already exists; inspect it, never "
                "blindly recreate"
            )
        old_receipt = json.loads(Path(args.original_receipt).read_text())
        account = client.query(qa_pod.ACCOUNT_QUERY)["myself"]
        offer = catalog(client)
        costs = validate(
            account,
            offer,
            old_receipt,
            max_rate=args.max_gpu_rate,
            hours=args.guard_hours,
            cap=args.max_spend_usd,
        )
        public, private = qa_pod.read_public_key(args.ssh_key)
        creation = qa_pod.utc_now()
        name = "syntide-diffusets-5000-" + uuid.uuid4().hex[:8]
        payload = build_input(public, name)
        receipt = {
            "status": "preview",
            "execute": args.execute,
            "checked_utc": qa_pod.iso(creation),
            "account_before": account,
            "catalog": offer,
            "request": qa_pod.sanitized_input(payload),
            "ssh_key_path": str(private),
            "cost_plan": costs,
            "allowed_existing_pod_id": OLD_POD_ID,
            "candidate_count": 5000,
            "training_seed": 0,
            "training_max_steps": 15531,
            "created_utc": qa_pod.iso(creation),
            "deadline_utc": qa_pod.iso(
                creation + dt.timedelta(hours=args.guard_hours)
            ),
            "guard": {
                "status": "must_arm_before_workload",
                "mechanism": "pod-local own-pod runpodctl stop using "
                "platform-injected credential",
                "account_key_copied": False,
                "stopAfter_supported": False,
            },
            "official_sources": SOURCES,
        }
        if not args.execute:
            return receipt
        receipt["status"] = "create_pending"
        qa_pod.save(output, receipt)
        # Intentionally no retry; timeout/unknown outcomes leave durable intent.
        pod = client.query(qa_pod.CREATE_MUTATION, {"input": payload})[
            "podFindAndDeployOnDemand"
        ]
        if not pod or not pod.get("id") or pod["id"] == OLD_POD_ID:
            raise RuntimeError(
                "Create response did not identify a new pod; inspect "
                "pending receipt"
            )
        receipt.update(
            status="created",
            pod=pod,
            created_receipt_utc=qa_pod.iso(qa_pod.utc_now()),
        )
        receipt["hourly_rate_matches"] = pod["costPerHr"] <= args.max_gpu_rate
        qa_pod.save(output, receipt)
        return receipt


def status(args, client):
    receipt = json.loads(Path(args.receipt).read_text())
    pod_id = receipt["pod"]["id"]
    if pod_id == OLD_POD_ID or receipt.get("status") != "created":
        raise ValueError("Not a completed extension receipt")
    pod = client.query(qa_pod.POD_QUERY, {"input": {"podId": pod_id}})["pod"]
    if pod and any(
        pod.get(key) != receipt["pod"].get(key)
        for key in ("id", "name", "imageName")
    ):
        raise ValueError("Pod identity differs from extension receipt")
    account = client.query(qa_pod.ACCOUNT_QUERY)["myself"]
    elapsed = max(
        0,
        (
            qa_pod.utc_now()
            - dt.datetime.fromisoformat(
                receipt["created_utc"].replace("Z", "+00:00")
            )
        ).total_seconds(),
    )
    return {
        "checked_utc": qa_pod.iso(qa_pod.utc_now()),
        "pod": pod,
        "account": account,
        "ssh_argv": qa_pod.ssh_argv(pod, receipt["ssh_key_path"])
        if pod
        else None,
        "own_pod_spend_conservative_estimate_usd": elapsed
        / 3600
        * receipt["cost_plan"]["hourly_rate_upper_bound_usd"],
        "account_delta_attribution": "aggregate_only_not_extension_cost",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    provision = commands.add_parser("create")
    provision.add_argument("--execute", action="store_true")
    provision.add_argument("--output", default=DIRECTORY + "/pod-create.json")
    provision.add_argument(
        "--original-receipt", default="outputs/qa-runs/pod-create.json"
    )
    provision.add_argument("--ssh-key", default="~/.ssh/id_runpod_syntide")
    provision.add_argument("--guard-hours", type=float, default=48)
    provision.add_argument("--max-gpu-rate", type=float, default=0.59)
    provision.add_argument("--max-spend-usd", type=float, default=40)
    inspect = commands.add_parser("status")
    inspect.add_argument("--receipt", default=DIRECTORY + "/pod-create.json")
    args = parser.parse_args()
    print(
        json.dumps(
            {"create": create, "status": status}[args.command](
                args, qa_pod.RunPod()
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
