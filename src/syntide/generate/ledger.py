"""Record every generated candidate, including rejected candidates.

Each recipe stores its source recording, generator name and version, settings,
random seed, and calibration report. A content checksum identifies the recipe.
These fields allow its signal to be reproduced exactly.

Completed trials write their records to ``recipes.parquet`` for later analysis.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

LEDGER_SCHEMA = pa.schema(
    [
        ("recipe_id", pa.string()),
        ("step", pa.int64()),
        ("slot", pa.int64()),
        ("attempt", pa.int64()),
        (
            "rng_stream",
            pa.string(),
            # Treatment, seed, step, batch position, and retry attempt.
        ),
        ("parent_record_id", pa.string()),
        ("parent_subject_uid", pa.string()),
        ("generator", pa.string()),
        ("version", pa.string()),
        ("controls_requested", pa.string()),  # JSON
        (
            "controls_realized",
            pa.string(),
        ),  # JSON; null when the generator does not measure them
        ("seed_key", pa.int64()),
        ("gap_report_id", pa.string()),
        (
            "validations",
            pa.string(),
        ),  # JSON list of {name, version, value, pass}
        ("decision", pa.string()),  # accepted | rejected
        ("reason", pa.string()),
        (
            "label_effect",
            pa.string(),
        ),  # JSON {task family: preserves | determines | invalidates}
    ]
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def recipe_id(
    parent_id: str,
    generator: str,
    version: str,
    controls: Mapping[str, Any],
    seed_key: int,
    gap_report_id: str | None,
) -> str:
    content = {
        "parent_id": parent_id,
        "generator": generator,
        "version": version,
        "controls": dict(controls),
        "seed_key": int(seed_key),
        "gap_report_id": gap_report_id,
    }
    return hashlib.sha256(canonical_json(content).encode()).hexdigest()


@dataclass(frozen=True)
class Recipe:
    """Settings and sources needed to reproduce one generated input."""

    parent_id: str
    generator: str
    version: str
    controls: dict[str, Any]
    seed_key: int
    gap_report_id: str | None

    @property
    def recipe_id(self) -> str:
        return recipe_id(
            self.parent_id,
            self.generator,
            self.version,
            self.controls,
            self.seed_key,
            self.gap_report_id,
        )


def ledger_row(
    recipe: Recipe,
    *,
    step: int,
    slot: int,
    attempt: int,
    rng_stream: str,
    parent_subject_uid: str,
    label_effect: Mapping[str, str],
    controls_realized: Mapping[str, Any] | None = None,
    validations: Sequence[Mapping[str, Any]] = (),
    decision: str = "accepted",
    reason: str = "",
) -> dict[str, Any]:
    return {
        "recipe_id": recipe.recipe_id,
        "step": step,
        "slot": slot,
        "attempt": attempt,
        "rng_stream": rng_stream,
        "parent_record_id": recipe.parent_id,
        "parent_subject_uid": parent_subject_uid,
        "generator": recipe.generator,
        "version": recipe.version,
        "controls_requested": canonical_json(recipe.controls),
        "controls_realized": None
        if controls_realized is None
        else canonical_json(dict(controls_realized)),
        "seed_key": recipe.seed_key,
        "gap_report_id": recipe.gap_report_id,
        "validations": canonical_json([dict(v) for v in validations]),
        "decision": decision,
        "reason": reason,
        "label_effect": canonical_json(dict(label_effect)),
    }


class LedgerWriter:
    """Collect ledger rows and stream them to Parquet in row groups.

    Rows go to ``<path>.partial`` while the trial runs; ``close()`` moves the
    file to ``path``, so a
    ``recipes.parquet`` only exists for a stream that was closed. Nothing
    touches the disk before the
    first flush.
    """

    def __init__(self, path: Path, flush_every: int = 50_000) -> None:
        self.path = Path(path)
        self._partial = self.path.with_name(self.path.name + ".partial")
        self._flush_every = flush_every
        self._columns: dict[str, list] = {
            name: [] for name in LEDGER_SCHEMA.names
        }
        self._writer: pq.ParquetWriter | None = None
        self.rows = 0

    def append(self, row: Mapping[str, Any]) -> None:
        for name, values in self._columns.items():
            values.append(row[name])
        self.rows += 1
        if len(self._columns["recipe_id"]) >= self._flush_every:
            self._flush()

    def _flush(self) -> None:
        table = pa.table(self._columns, schema=LEDGER_SCHEMA)
        if self._writer is None:
            self._partial.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(
                self._partial, LEDGER_SCHEMA, compression="zstd"
            )
        self._writer.write_table(table)
        self._columns = {name: [] for name in LEDGER_SCHEMA.names}

    def close(self) -> Path:
        # Write the schema even for trials without generated examples.
        self._flush()
        self._writer.close()
        self._partial.replace(self.path)
        return self.path

    def discard(self) -> None:
        """Remove the temporary records of an unfinished training stream."""
        if self._writer is not None:
            self._writer.close()
        self._partial.unlink(missing_ok=True)
