import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "adopt", Path(__file__).parents[1] / "scripts/runpod/diffusets_adopt.py"
)
adopt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adopt)


@pytest.fixture
def bundle(tmp_path):
    plan = {"candidates": [{"slot_id": i, "seed": 100 + i} for i in range(3)]}
    identity = {
        "generation_id": "fixed",
        "count": 3,
        "plan_sha256": adopt.digest(plan),
    }
    adopt.write(tmp_path / "identity.json", identity)
    adopt.write(tmp_path / "measurement-summary.json", {"measured": 3})
    slots = []
    for row in plan["candidates"]:
        directory = tmp_path / "candidates" / f"{row['slot_id']:05d}"
        adopt.write(directory / "signal.json", {"signal": row["slot_id"]})
        result = {
            **row,
            "candidate_sha256": adopt.digest(row),
            "generation_id": "fixed",
            "accepted": row["slot_id"] != 2,
            "artifacts": {"signal.json": adopt.sha(directory / "signal.json")},
        }
        adopt.write(directory / "result.json", result)
        slots.append(
            {
                "slot_id": row["slot_id"],
                "result_sha256": adopt.sha(directory / "result.json"),
            }
        )
    adopt.write(
        tmp_path / "summary.json",
        {
            "generation_id": "fixed",
            "candidate_count": 3,
            "slots": slots,
            "accepted": 2,
            "rejected": 1,
            "measurement_summary_sha256": adopt.sha(
                tmp_path / "measurement-summary.json"
            ),
        },
    )
    return tmp_path, plan, adopt.sha(tmp_path / "identity.json")


def test_full_bundle_requires_exact_published_outputs(bundle):
    root, plan, identity = bundle
    assert adopt.verify_generation(root, plan, identity, 3)["accepted"] == 2
    (root / "candidates/00000/extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="extra"):
        adopt.verify_generation(root, plan, identity, 3)


def test_incomplete_budget_cannot_be_adopted(bundle):
    root, plan, identity = bundle
    with pytest.raises(ValueError, match="budget"):
        adopt.verify_generation(root, plan, identity)


def test_changed_waveform_refuses_continuation(bundle):
    root, plan, identity = bundle
    (root / "candidates/00001/signal.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact differs"):
        adopt.verify_generation(root, plan, identity, 3)


def test_published_symlink_refused(bundle):
    root, plan, identity = bundle
    (root / "candidates/00000/link").symlink_to(root / "identity.json")
    with pytest.raises(ValueError, match="symlink"):
        adopt.verify_generation(root, plan, identity, 3)


def test_pid_reuse_cannot_be_killed(monkeypatch):
    expected = {
        "pid": 9,
        "start_ticks": 4,
        "argv": ["generator"],
        "pgid": 9,
        "sid": 9,
    }
    monkeypatch.setattr(
        adopt, "process", lambda pid: {**expected, "start_ticks": 5}
    )
    with pytest.raises(ValueError, match="identity changed"):
        adopt.require_process(expected)


def test_inherited_lock_requires_same_inode_and_exclusion(tmp_path):
    target = tmp_path / "pipeline.lock"
    target.touch()
    proc = tmp_path / "proc/4"
    (proc / "fd").mkdir(parents=True)
    (proc / "fdinfo").mkdir()
    (proc / "fd/3").symlink_to(target)
    (proc / "fdinfo/3").write_text(
        "lock: 1: FLOCK ADVISORY WRITE 3 00:01:123 0 EOF\n"
    )
    with adopt.locked(target):
        adopt.require_inherited_locks(4, [target], tmp_path / "proc")
    with pytest.raises(ValueError, match="not exclusively held"):
        adopt.require_inherited_locks(4, [target], tmp_path / "proc")


def test_invalid_operational_policy_fails_before_process_mutation(
    tmp_path, monkeypatch
):
    override, config = tmp_path / "override.json", tmp_path / "config.json"
    adopt.write(override, {"pod_id": "owned"})
    adopt.write(config, {"reuse_completed_generation": True})

    def reject(*a, **kw):
        raise ValueError("policy rejected")

    monkeypatch.setattr(
        adopt,
        "load_pipeline",
        lambda p: SimpleNamespace(
            validate=lambda c: None, deadline_override=reject
        ),
    )
    monkeypatch.setattr(
        adopt.os,
        "kill",
        lambda *a: pytest.fail("process mutation before policy validation"),
    )
    with pytest.raises(ValueError, match="policy rejected"):
        adopt.run(
            {
                "study_root": str(tmp_path),
                "operation_root": str(tmp_path / "ops"),
                "controller_identity": "x",
                "bindings": {},
                "override_path": str(override),
                "override_sha256": adopt.sha(override),
                "pod_id": "owned",
                "pipeline_source": "fake",
                "pipeline_config_path": str(config),
                "pipeline_config_sha256": adopt.sha(config),
            }
        )
