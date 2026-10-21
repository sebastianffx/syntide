"""The scarcity challenge: which records each experimental arm may train on.

One manifest per pool fraction holds every role as an explicit, ordered list of record ids:

- ``pool`` / ``unused``: a seeded fraction of training *subjects* (all their records) and the rest.
  Pools for different fractions share one subject permutation, so smaller pools nest in larger ones.
- ``retained_artifact`` / ``withheld``: the pool's artifact-bearing records, split by ``retain``.
- ``depleted`` = pool - withheld: parents for A0-A3.
- ``natural_mix``: the pool minus as many *random* records as were withheld (C1, matched N).
- ``val`` / ``test``: the outer validation and test folds, untouched.

The manifest's sha256 covers everything else in it, so an edited file is rejected on load.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa

ROLES = ("pool", "unused", "retained_artifact", "withheld", "depleted", "natural_mix", "val", "test")


@dataclass
class ChallengeSpec:
    """How the scarcity worlds are built. Also the Hydra schema node for ``challenge``."""

    id: str = "ptbxl-artifact-v2"
    train_folds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
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
    """The manifest's content no longer matches its recorded sha256."""


def pool_name(pool_fraction: float) -> str:
    return f"pool{round(pool_fraction * 100)}"


def build_manifest(index: pa.Table, spec: ChallengeSpec, pool_fraction: float, source: dict) -> Manifest:
    """Build every role for one pool fraction. Deterministic in (index, spec, pool_fraction)."""
    record_id = np.asarray(index["record_id"].to_pylist())
    subject = np.asarray(index["subject_uid"].to_pylist())
    fold = np.asarray(index["strat_fold"].to_pylist())
    is_artifact = np.asarray(index["is_artifact"].to_pylist(), dtype=bool)

    train = np.isin(fold, spec.train_folds)
    train_subjects = np.unique(subject[train])
    # One permutation per seed, whatever the fraction: pools nest.
    order = np.random.default_rng([spec.seed, 0]).permutation(train_subjects)
    pool_subjects = order[: math.ceil(pool_fraction * len(train_subjects))]
    pool = train & np.isin(subject, pool_subjects)

    artifact_in_pool = np.flatnonzero(pool & is_artifact)
    n_retained = math.ceil(spec.retain * len(artifact_in_pool))
    stream = round(pool_fraction * 1_000_000)
    retained_idx = np.random.default_rng([spec.seed, 1, stream]).choice(
        artifact_in_pool, n_retained, replace=False
    )
    retained = _mask(len(index), retained_idx)
    withheld = pool & is_artifact & ~retained

    pool_idx = np.flatnonzero(pool)
    dropped_idx = np.random.default_rng([spec.seed, 2, stream]).choice(
        pool_idx, withheld.sum(), replace=False
    )
    natural_mix = pool & ~_mask(len(index), dropped_idx)

    masks = {
        "pool": pool,
        "unused": train & ~pool,
        "retained_artifact": retained,
        "withheld": withheld,
        "depleted": pool & ~withheld,
        "natural_mix": natural_mix,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=1) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> Manifest:
    """Load a manifest and verify its content hash."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = payload.pop("sha256")
    if _sha256(payload) != recorded:
        raise ManifestIntegrityError(
            f"{path}: content does not match its sha256; rebuild it instead of editing it"
        )
    payload["roles"] = {role: tuple(ids) for role, ids in payload["roles"].items()}
    return Manifest(**payload, sha256=recorded)


def _mask(n: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    return mask


def _sha256(content: dict) -> str:
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(canonical.encode()).hexdigest()
