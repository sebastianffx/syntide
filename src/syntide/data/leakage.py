"""Gate 1: the leakage guard.

Arms deliberately overlap (A0-A3 share parents; natural_mix and depleted share most records), so the
guard does not demand that every role be disjoint. It enforces the relations the experiment relies on
(training_loop doc, "narrow the leakage assertion"):

- the outer partitions (pool, unused, val, test) cover every labelled record exactly once, and no
  subject appears in more than one of train (pool + unused), val and test;
- retained_artifact and withheld split the pool's artifact records exactly;
- depleted is the pool minus withheld, so no withheld record reaches A0-A3;
- natural_mix is drawn from the pool and has the same size as depleted;
- generator calibration reads only retained_artifact records.
"""

from collections.abc import Iterable

import pyarrow as pa

from syntide.data.challenge import Manifest

_OUTER = ("pool", "unused", "val", "test")


class LeakageError(Exception):
    """A manifest or calibration input breaks an isolation rule; the run must stop."""


def check_manifest(manifest: Manifest, index: pa.Table) -> None:
    """Raise LeakageError, naming the offending records or subjects, if any invariant fails."""
    roles = {role: set(ids) for role, ids in manifest.roles.items()}
    subject_of = dict(zip(index["record_id"].to_pylist(), index["subject_uid"].to_pylist(), strict=True))
    artifact = {
        r
        for r, flag in zip(index["record_id"].to_pylist(), index["is_artifact"].to_pylist(), strict=True)
        if flag
    }

    unknown = set().union(*roles.values()) - subject_of.keys()
    _fail_if(unknown, "records are not labelled PTB-XL records")

    for i, a in enumerate(_OUTER):
        for b in _OUTER[i + 1 :]:
            _fail_if(roles[a] & roles[b], f"records are in both {a} and {b}")
    _fail_if(
        subject_of.keys() - set().union(*(roles[r] for r in _OUTER)), "labelled records are in no partition"
    )

    partitions = {"train": roles["pool"] | roles["unused"], "val": roles["val"], "test": roles["test"]}
    subjects = {name: {subject_of[r] for r in ids} for name, ids in partitions.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        _fail_if(subjects[a] & subjects[b], f"subjects are in both {a} and {b}")

    pool_artifacts = roles["pool"] & artifact
    _fail_if(roles["retained_artifact"] & roles["withheld"], "records are both retained and withheld")
    _fail_if(
        pool_artifacts ^ (roles["retained_artifact"] | roles["withheld"]),
        "records break retained + withheld == the pool's artifact records",
    )
    _fail_if(roles["depleted"] & roles["withheld"], "withheld records are in depleted")
    _fail_if(
        roles["depleted"] ^ (roles["pool"] - roles["withheld"]), "records break depleted == pool - withheld"
    )
    _fail_if(roles["natural_mix"] - roles["pool"], "natural_mix records are outside the pool")
    if len(roles["natural_mix"]) != len(roles["depleted"]):
        raise LeakageError(
            f"{manifest.challenge_id}: natural_mix has {len(roles['natural_mix'])} records "
            f"but depleted has {len(roles['depleted'])}; the matched-N control needs equal sizes"
        )


def check_calibration_ids(record_ids: Iterable[str], manifest: Manifest) -> None:
    """Generator calibration may only see retained artifact records (never withheld, val or test)."""
    _fail_if(
        set(record_ids) - set(manifest.roles["retained_artifact"]),
        "calibration records are outside retained_artifact",
    )


def _fail_if(offenders: set[str], message: str) -> None:
    if offenders:
        examples = ", ".join(sorted(offenders)[:5])
        raise LeakageError(f"{len(offenders)} {message}: {examples}")
