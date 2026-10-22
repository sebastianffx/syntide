"""Gate 1: the leakage guard.

Arms deliberately overlap (A0-A3 share parents; natural_mix and depleted share most records), so the
guard does not demand that every role be disjoint. It enforces the relations the experiment relies on
(MVE Gate 1; training_loop doc, "narrow the leakage assertion"):

- the manifest was built from this exact data (same source fingerprint);
- the outer partitions (pool, unused, val, test) are disjoint and are exactly the spec's folds;
- no patient is in more than one of train, val and test, nor in both the pool and unused;
- retained_artifact and withheld cover the pool's artifact ECGs, depleted is the pool minus withheld,
  and no withheld patient keeps an ECG in depleted;
- natural_mix is the pool minus whole patients, with the same size as depleted;
- generator calibration reads only retained_artifact records, and every recipe parent is in depleted.
"""

from collections.abc import Iterable, Mapping

import pyarrow as pa

from syntide.data.challenge import Manifest

_OUTER = ("pool", "unused", "val", "test")


class LeakageError(Exception):
    """A manifest or generation input breaks an isolation rule; the run must stop."""


def check_manifest(manifest: Manifest, index: pa.Table, source: Mapping[str, str]) -> None:
    """Raise LeakageError, naming the offending records or subjects, if any invariant fails."""
    if dict(manifest.source) != dict(source):
        raise LeakageError(
            f"{manifest.challenge_id} was built from {manifest.source}, not from the local data "
            f"{dict(source)}; rebuild the manifest or the dataset"
        )
    roles = {role: set(ids) for role, ids in manifest.roles.items()}
    columns = index.select(["record_id", "subject_uid", "strat_fold", "is_artifact"]).to_pydict()
    subject_of = dict(zip(columns["record_id"], columns["subject_uid"], strict=True))
    fold_of = dict(zip(columns["record_id"], columns["strat_fold"], strict=True))
    artifact = {r for r, flag in zip(columns["record_id"], columns["is_artifact"], strict=True) if flag}

    _fail_if(set().union(*roles.values()) - subject_of.keys(), "records are not labelled PTB-XL records")

    for i, a in enumerate(_OUTER):
        for b in _OUTER[i + 1 :]:
            _fail_if(roles[a] & roles[b], f"records are in both {a} and {b}")
    train = roles["pool"] | roles["unused"]
    for name, ids in (("train", train), ("val", roles["val"]), ("test", roles["test"])):
        folds = manifest.spec[f"{name}_folds"]
        expected = {r for r, fold in fold_of.items() if fold in folds}
        _fail_if(ids ^ expected, f"records break {name} == folds {folds}")

    for role in ("retained_artifact", "withheld", "depleted", "natural_mix"):
        _fail_if(roles[role] - roles["pool"], f"{role} records are outside the pool")
    _fail_if(roles["retained_artifact"] & roles["withheld"], "records are both retained and withheld")
    _fail_if(
        (roles["pool"] & artifact) ^ (roles["retained_artifact"] | (roles["withheld"] & artifact)),
        "records break retained + withheld == the pool's artifact records",
    )
    _fail_if(roles["depleted"] & roles["withheld"], "withheld records are in depleted")
    _fail_if(
        roles["depleted"] ^ (roles["pool"] - roles["withheld"]), "records break depleted == pool - withheld"
    )

    def subjects(ids: set[str]) -> set[str]:
        return {subject_of[r] for r in ids}

    for a, a_ids, b, b_ids in (
        ("pool", roles["pool"], "unused", roles["unused"]),
        ("train", train, "val", roles["val"]),
        ("train", train, "test", roles["test"]),
        ("val", roles["val"], "test", roles["test"]),
        ("withheld", roles["withheld"], "depleted", roles["depleted"]),
        ("natural_mix", roles["natural_mix"], "the records C1 drops", roles["pool"] - roles["natural_mix"]),
    ):
        _fail_if(subjects(a_ids) & subjects(b_ids), f"subjects are in both {a} and {b}")

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


def check_recipe_parents(parent_ids: Iterable[str], manifest: Manifest) -> None:
    """Every recipe (artifact edit or standard augmentation) must start from a depleted-pool record."""
    _fail_if(set(parent_ids) - set(manifest.roles["depleted"]), "recipe parents are outside depleted")


def check_lineage(
    ancestry: Iterable[tuple[str, Iterable[str], Iterable[str]]],
    manifest: Manifest,
    index: pa.Table,
    allowed_roles: Iterable[str],
) -> None:
    """Every task input must descend only from records of ``allowed_roles``, and from their patients only.

    ``ancestry`` holds one ``(input id, root record ids, root subject uids)`` per task input: the real
    recordings a waveform descends from (itself, for a real recording). A generated or edited waveform's
    own id proves nothing; its roots are what this checks:

    - every root is a record of the index, and its declared patient is the index's patient;
    - every root is in one of ``allowed_roles``;
    - no root patient has any record outside ``allowed_roles`` (so a val or test patient, or a withheld
      patient for a depleted arm, can never come back through a synthetic descendant).

    An input without any root is refused: an ancestry that cannot be resolved does not pass.
    """
    allowed_roles = tuple(allowed_roles)
    unknown_roles = set(allowed_roles) - set(manifest.roles)
    if unknown_roles:
        raise LeakageError(f"{manifest.challenge_id} has no roles {sorted(unknown_roles)}")
    allowed = set().union(*(manifest.roles[role] for role in allowed_roles))
    columns = index.select(["record_id", "subject_uid"]).to_pydict()
    subject_of = dict(zip(columns["record_id"], columns["subject_uid"], strict=True))
    managed = set().union(*manifest.roles.values())
    # A partial index would silently shrink the forbidden patients, so the subject rule could never fire.
    _fail_if(managed - subject_of.keys(), "manifest records are missing from the index; pass the full index")
    forbidden_subjects = {subject_of[r] for r in managed - allowed}
    orphans, unknown, mismatched, outside, crossing = set(), set(), set(), set(), set()
    for input_id, roots, subjects in ancestry:
        roots, subjects = tuple(roots), tuple(subjects)
        if not roots or len(roots) != len(subjects):
            orphans.add(input_id)
            continue
        for root, subject in zip(roots, subjects, strict=True):
            if root not in subject_of:
                unknown.add(f"{input_id}<-{root}")
            elif subject_of[root] != subject:
                mismatched.add(f"{input_id}<-{root}")
            elif root not in allowed:
                outside.add(f"{input_id}<-{root}")
            if subject in forbidden_subjects:
                crossing.add(f"{input_id}<-{subject}")
    _fail_if(orphans, "task inputs have no resolvable real ancestry")
    _fail_if(unknown, "task inputs descend from records outside the index")
    _fail_if(mismatched, "task inputs declare another patient than their root record's")
    _fail_if(outside, f"task inputs descend from records outside roles {list(allowed_roles)}")
    _fail_if(crossing, f"task inputs descend from patients with records outside roles {list(allowed_roles)}")


def check_disjoint_subjects(named: dict[str, Iterable[str]]) -> None:
    """No patient may appear in two of the named populations (training corpus, selection, test, external)."""
    sets = {name: set(subjects) for name, subjects in named.items()}
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            _fail_if(sets[a] & sets[b], f"patients are in both {a} and {b}")


def _fail_if(offenders: set[str], message: str) -> None:
    if offenders:
        examples = ", ".join(sorted(offenders)[:5])
        raise LeakageError(f"{len(offenders)} {message}: {examples}")
