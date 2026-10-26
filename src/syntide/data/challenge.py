"""Build patient-separated data pools for the scarcity experiment.

Each manifest stores ordered recording identifiers for all data roles.
Sampling always selects whole patients:

- ``pool`` and ``unused`` partition training patients. Smaller pools are
  subsets of larger pools because all fractions use the same patient
  ordering.
- ``retained_artifact`` contains noisy recordings used to calibrate the
  editor. ``withheld`` removes all recordings of the other selected noisy
  patients.
- ``depleted`` contains the remaining pool used by the baseline and
  synthetic data treatments.
- ``natural_mix`` removes random whole patients to match the depleted pool's
  size while retaining the natural prevalence of signal artifacts.
- ``val`` and ``test`` contain the untouched validation and test folds.

A content hash detects changes after a manifest is saved. Existing manifests
cannot be replaced with different contents.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa

ROLES = (
    "pool",
    "unused",
    "retained_artifact",
    "withheld",
    "depleted",
    "natural_mix",
    "val",
    "test",
)


@dataclass
class ChallengeSpec:
    """Settings for patient sampling and withholding noisy recordings."""

    id: str = "ptbxl-artifact-v2"
    train_folds: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8]
    )
    val_folds: list[int] = field(default_factory=lambda: [9])
    test_folds: list[int] = field(default_factory=lambda: [10])
    pool_fractions: list[float] = field(default_factory=lambda: [0.10, 0.25])
    retain: float = 0.10
    seed: int = 0


@dataclass(frozen=True)
class Manifest:
    challenge_id: str
    spec: dict
    source: dict
    roles: dict[str, tuple[str, ...]]
    sha256: str


class ManifestIntegrityError(ValueError):
    """The manifest's content no longer matches its recorded content hash."""


def pool_name(pool_fraction: float) -> str:
    return f"pool{round(pool_fraction * 100)}"


def build_manifest(
    index: pa.Table, spec: ChallengeSpec, pool_fraction: float, source: dict
) -> Manifest:
    """Build data roles for one training-pool fraction.

    The same index, specification, and pool fraction produce the same
    manifest.
    """
    record_id = np.asarray(index["record_id"].to_pylist())
    subject = np.asarray(index["subject_uid"].to_pylist())
    fold = np.asarray(index["strat_fold"].to_pylist())
    is_artifact = np.asarray(index["is_artifact"].to_pylist(), dtype=bool)

    train = np.isin(fold, spec.train_folds)
    train_subjects = np.unique(subject[train])
    # One permutation per seed, whatever the fraction: pools nest.
    order = np.random.default_rng([spec.seed, 0]).permutation(train_subjects)
    pool = train & np.isin(
        subject, order[: math.ceil(pool_fraction * len(train_subjects))]
    )

    stream = round(pool_fraction * 1_000_000)
    artifact_subjects = np.unique(subject[pool & is_artifact])
    retained_subjects = np.random.default_rng([spec.seed, 1, stream]).choice(
        artifact_subjects,
        math.ceil(spec.retain * len(artifact_subjects)),
        replace=False,
    )
    retained = pool & is_artifact & np.isin(subject, retained_subjects)
    withheld = (
        pool
        & np.isin(subject, artifact_subjects)
        & ~np.isin(subject, retained_subjects)
    )

    dropped_subjects = _subjects_totalling(
        subject[pool],
        int(withheld.sum()),
        np.random.default_rng([spec.seed, 2, stream]),
    )

    masks = {
        "pool": pool,
        "unused": train & ~pool,
        "retained_artifact": retained,
        "withheld": withheld,
        "depleted": pool & ~withheld,
        "natural_mix": pool & ~np.isin(subject, dropped_subjects),
        "val": np.isin(fold, spec.val_folds),
        "test": np.isin(fold, spec.test_folds),
    }
    roles = {role: tuple(record_id[masks[role]].tolist()) for role in ROLES}
    content = {
        "challenge_id": f"{spec.id}/{pool_name(pool_fraction)}",
        "spec": {**asdict(spec), "pool_fraction": pool_fraction},
        "source": dict(source),
        "roles": roles,
    }
    return Manifest(**content, sha256=_sha256(content))


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Save a manifest without replacing different existing contents."""
    path = Path(path)
    if path.exists():
        existing = load_manifest(path)
        if existing.sha256 != manifest.sha256:
            raise FileExistsError(
                f"{path} already holds a different manifest (sha256 "
                f"{existing.sha256[:12]}). Manifests are frozen: set a new "
                "challenge.id, or delete the file on purpose to rebuild it."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), indent=1) + "\n", encoding="utf-8"
    )


def load_manifest(path: Path) -> Manifest:
    """Load a manifest and verify its content hash."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = payload.pop("sha256")
    if _sha256(payload) != recorded:
        raise ManifestIntegrityError(
            f"{path}: content does not match its sha256; rebuild it "
            "instead of editing it"
        )
    payload["roles"] = {
        role: tuple(ids) for role, ids in payload["roles"].items()
    }
    return Manifest(**payload, sha256=recorded)


def _subjects_totalling(
    subjects: np.ndarray, n_records: int, rng: np.random.Generator
) -> list[str]:
    """Select whole patients with exactly the requested recording count.

    Visit patients in a seeded random order and skip those that would exceed
    the target. PTB-XL has many patients with one recording, so the exact
    target is reachable in practice.
    """
    names, counts = np.unique(subjects, return_counts=True)
    chosen, remaining = [], n_records
    for k in rng.permutation(len(names)):
        if remaining == 0:
            break
        if counts[k] <= remaining:
            chosen.append(str(names[k]))
            remaining -= int(counts[k])
    if remaining:
        raise ValueError(
            f"no set of whole subjects holds exactly {n_records} records"
        )
    return chosen


def _sha256(content: dict) -> str:
    canonical = json.dumps(
        content, sort_keys=True, separators=(",", ":"), default=list
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
