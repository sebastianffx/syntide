"""Save analysis rules before evaluating held-out test data.

The create-only rules.json records the rules, their content hash, the plan
hash, and the repository revision. Reports require these rules to predate
the test-evaluation record and remain unchanged. The holdout module
separately checks trials and checkpoints before scoring them.
"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from syntide import __version__
from syntide.config import load_dataclass
from syntide.eval.analysis import AnalysisConfig

RULES_FILE = "eval/rules.json"
PLAN_FILE = "plan.json"
CLAIM_FILE = "test_claim.json"  # written by syntide.eval.holdout


class FreezeError(RuntimeError):
    """Rules changed or were not fixed before test evaluation."""


def freeze_rules(
    protocol_dir: str | Path, rules: AnalysisConfig
) -> tuple[Path, bool]:
    """Save fixed analysis rules without replacing existing different rules.

    Returns:
        A pair containing the output path and whether a new file was
        written.
    """
    protocol_dir = Path(protocol_dir)
    plan = protocol_dir / PLAN_FILE
    if not plan.exists():
        raise FreezeError(
            f"{plan} does not exist; plan the protocol before freezing its "
            "rules"
        )
    path = protocol_dir / RULES_FILE
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if _frozen_sha256(frozen) != rules.sha256:
            changes = _diff(frozen["rules"], rules.to_dict())
            raise FreezeError(
                f"{path} froze other rules on {frozen['frozen_at']}: "
                f"{changes}; frozen rules stay frozen"
            )
        return path, False
    if (protocol_dir / CLAIM_FILE).exists():
        raise FreezeError(
            f"the test fold of {protocol_dir} is already claimed; rules "
            "frozen now are not pre-registered"
        )
    payload = {
        "frozen_at": _now(),
        "protocol_dir": str(protocol_dir),
        "rules": rules.to_dict(),
        "rules_sha256": rules.sha256,
        "plan_sha256": _sha256(plan),
        "syntide_version": __version__,
        "git": _git_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    return path, True


def verify_frozen(
    protocol_dir: str | Path, rules: AnalysisConfig, *, claimed: bool
) -> dict[str, Any]:
    """Check that fixed rules and the recorded plan remain unchanged."""
    protocol_dir = Path(protocol_dir)
    path = protocol_dir / RULES_FILE
    if not path.exists():
        raise FreezeError(
            f"{path} does not exist; run mode=freeze before the test fold "
            "is claimed"
        )
    frozen = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if _frozen_sha256(frozen) != rules.sha256:
        problems.append(
            "the rules changed since they were frozen: "
            f"{_diff(frozen['rules'], rules.to_dict())}"
        )
    if frozen["plan_sha256"] != _sha256(protocol_dir / PLAN_FILE):
        problems.append("plan.json changed since the rules were frozen")
    claim_path = protocol_dir / CLAIM_FILE
    if claimed or claim_path.exists():
        if not claim_path.exists():
            problems.append(
                f"{claim_path} does not exist; the test fold is not scored yet"
            )
        else:
            claimed_at = json.loads(claim_path.read_text(encoding="utf-8"))[
                "claimed_at"
            ]
            if datetime.fromisoformat(claimed_at) < datetime.fromisoformat(
                frozen["frozen_at"]
            ):
                problems.append(
                    f"the test fold was claimed ({claimed_at}) before the "
                    "rules were frozen"
                )
    if problems:
        raise FreezeError("; ".join(problems))
    return frozen


def frozen_rules(frozen: dict[str, Any]) -> AnalysisConfig:
    """Load saved rules, applying defaults for newly added fields."""
    return load_dataclass(AnalysisConfig, frozen["rules"])


def _frozen_sha256(frozen: dict[str, Any]) -> str:
    return frozen_rules(frozen).sha256


def _diff(frozen: dict[str, Any], current: dict[str, Any]) -> str:
    changed = sorted(
        key
        for key in frozen.keys() | current.keys()
        if frozen.get(key) != current.get(key)
    )
    return (
        "changed " + ", ".join(changed)
        if changed
        else "same keys, different values"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_state() -> dict[str, Any] | None:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"rev": rev.strip(), "dirty": bool(status.stdout.strip())}
