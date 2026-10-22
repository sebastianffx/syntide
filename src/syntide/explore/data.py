"""Training-only recording catalogs for embedding-based generation discovery.

Catalog construction does not decode waveforms. QA examples attach to recordings as ``texts``;
they never create additional recording points. The PTB-XL adapter and private ECG-QA registry view
provide experiment membership and compatible fingerprints. Generic JSONL supports new tuple datasets
without adding another task-specific adapter.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from syntide.data.adapter import PtbXlAdapter
from syntide.data.challenge import load_manifest
from syntide.data.leakage import LeakageError
from syntide.data.schema import FS_HZ, LEADS

_SUPERCLASSES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}
_FORBIDDEN_ROLES = {"val", "validation", "test", "withheld", "heldout", "held_out", "unused"}
_SPLITS = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


class RecordingSource:
    """A complete selected population and an ordered, restricted waveform loader."""

    def __init__(self, rows: list[dict], provenance: dict, loader: Callable[[list[str]], list[dict]]):
        self.rows = rows
        self.provenance = provenance
        self._by_id = {row["record_id"]: row for row in rows}
        self._loader = loader

    def load(self, record_ids: Sequence[str]) -> list[dict]:
        """Load only selected population members, preserving the caller's order."""
        ids = list(record_ids)
        missing = sorted(set(ids) - self._by_id.keys())
        if missing:
            raise ValueError(f"Waveform request is outside the discovery population: {missing[:5]}")
        if not ids:
            return []
        loaded = self._loader(ids)
        if [row["record_id"] for row in loaded] != ids:
            raise RuntimeError("Waveform loader did not preserve requested recording order")
        for row in loaded:
            signal = np.asarray(row["signal"])
            if signal.ndim != 2 or not all(signal.shape):
                raise ValueError(f"{row['record_id']}: expected non-empty signal [channels, time]")
            if len(row["channels"]) != signal.shape[0]:
                raise ValueError(f"{row['record_id']}: channels do not match signal shape {signal.shape}")
            if not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal):
                raise ValueError(f"{row['record_id']}: signal must contain real numeric values")
            if not np.isfinite(signal).all():
                raise ValueError(f"{row['record_id']}: signal contains non-finite values")
            if np.max(np.abs(signal.astype(np.float64, copy=False))) > np.finfo(np.float32).max:
                raise ValueError(f"{row['record_id']}: signal exceeds the float32 numeric range")
            row["signal"] = signal.astype(np.float32, copy=False)
        return loaded


def open_source(config: dict[str, Any]) -> RecordingSource:
    """Open ``ptbxl``, ``ecgqa`` or ``jsonl`` with whole-record training membership.

    Shared keys: ``dataset``, ``manifest`` and ``role`` (defaults to ``depleted`` when a
    manifest is present, otherwise ``train``). Registry adapters also accept ``registry``,
    ``dataset_ref``, ``signals_ref`` (ECG-QA), ``scp_statements`` and ``ptbxl_root``.
    JSONL requires ``input``; paths inside it are relative to the JSONL file. Every row needs
    ``record_id``, ``signal_path``, ``sample_rate``, ``channels`` and an explicit ``split``.
    Repeated rows may supply different text/QA tuples for the same recording.
    """
    kind = str(config.get("dataset", "ptbxl")).lower()
    role = str(config.get("role") or ("depleted" if config.get("manifest") else "train"))
    if role in _FORBIDDEN_ROLES:
        raise ValueError(f"Role {role!r} cannot be used for generation discovery; use training records")
    if not config.get("manifest") and role != "train":
        raise ValueError("A named experiment role requires --manifest; without a manifest use role=train")
    if kind == "jsonl":
        return _open_jsonl(config, role)
    if kind in {"ptbxl", "ecgqa"}:
        return _open_registry(config, kind, role)
    raise ValueError(f"Unsupported discovery dataset {kind!r}; choose ptbxl, ecgqa, or jsonl")


def _open_registry(config: dict, kind: str, role: str) -> RecordingSource:
    registry = Path(config.get("registry", "datasets/timenet-registry")).expanduser().resolve()
    if kind == "ecgqa":
        from syntide.explore.qa import DATASET, SIGNALS, EcgQaView, check_qa_tasks

        adapter = EcgQaView(
            registry, config.get("dataset_ref") or DATASET, config.get("signals_ref") or SIGNALS
        )
        signal_adapter = adapter.signals
    else:
        adapter = PtbXlAdapter(registry, config.get("dataset_ref") or "physionet/ptb-xl@1.0.3")
        signal_adapter = adapter
    index = adapter.index()
    index_rows = index.to_pylist()
    source = adapter.source()
    manifest = None
    if config.get("manifest"):
        from syntide.data.leakage import check_manifest

        manifest = load_manifest(Path(config["manifest"]))
        check_manifest(manifest, index, source)
        if kind == "ecgqa":
            check_qa_tasks(adapter.tasks(), manifest, index)
        if role not in manifest.roles:
            raise ValueError(f"Manifest has no role {role!r}; choose depleted or retained_artifact")
        selected_ids = set(manifest.roles[role])
        forbidden = set().union(*(manifest.roles[name] for name in ("withheld", "val", "test")))
        forbidden_subjects = {row["subject_uid"] for row in index_rows if row["record_id"] in forbidden}
        unsafe = [
            row["record_id"]
            for row in index_rows
            if row["record_id"] in selected_ids
            and (row["record_id"] in forbidden or row["subject_uid"] in forbidden_subjects)
        ]
        if unsafe:
            raise LeakageError(
                f"Role {role!r} includes withheld or held-out records/patients; use depleted "
                f"for generation discovery (e.g. {unsafe[:3]})"
            )
    else:
        selected_ids = {row["record_id"] for row in index_rows if 1 <= row["strat_fold"] <= 8}
    # Apply the ordinary patient boundary even without an experiment manifest.
    _check_subject_splits([{**row, "split": _fold_split(row["strat_fold"])} for row in index_rows])
    chosen = [row for row in index_rows if row["record_id"] in selected_ids]
    if any(_fold_split(row["strat_fold"]) != "train" for row in chosen):
        raise LeakageError("Discovery selection includes records outside PTB-XL training folds 1-8")

    statements_path, database_path = _ptbxl_files(config, registry)
    definitions = _label_definitions(statements_path)
    raw_metadata = _clinical_metadata(database_path, selected_ids)
    # iter_records hydrates structure only; to_numpy is called only inside the loader below.
    record_metadata = {
        record.record_id: dict(record.metadata or {})
        for record in signal_adapter._reader.iter_records(
            [row["record_id"] for row in chosen], with_annotations=False
        )
    }
    texts: dict[str, list[dict]] = defaultdict(list)
    if kind == "ecgqa":
        for task in adapter.tasks().to_pylist():
            if task["record_id"] not in selected_ids:
                continue
            question, answer = task["question"], task["answer"]
            texts[task["record_id"]].append(
                {
                    "text_id": task["task_id"],
                    "text": "",
                    "question": question,
                    "answer": answer,
                    "rationale": task["rationale"],
                    "metadata": {
                        key: task[key]
                        for key in (
                            "template_id",
                            "question_type",
                            "answer_options",
                            "clinical_context",
                            "cot_split",
                            "has_rationale",
                        )
                    },
                }
            )
    rows = []
    for row in chosen:
        record_id = row["record_id"]
        metadata = record_metadata[record_id]
        codes = metadata.get("ptbxl.scp_codes", {})
        labels = sorted(
            {
                *(f"superclass:{value}" for value in row["labels"]),
                *(f"scp:{value}" for value in codes),
                *(f"artifact:{value}" for value in row["artifact_types"]),
            }
        )
        rows.append(
            {
                "record_id": record_id,
                "subject_uid": row["subject_uid"],
                "split": "train",
                "labels": labels,
                "artifact_types": list(row["artifact_types"]),
                "metadata": {
                    **metadata,
                    **raw_metadata.get(record_id, {}),
                    "strat_fold": row["strat_fold"],
                    "is_artifact": row["is_artifact"],
                    "label_descriptions": {
                        label: definitions.get(label, {}).get("description", label.split(":", 1)[-1])
                        for label in labels
                    },
                },
                "texts": texts.get(record_id, []),
            }
        )
    by_id = {row["record_id"]: row for row in rows}

    def load(ids: list[str]) -> list[dict]:
        samples = adapter.load(ids)
        return [
            {
                **by_id[record_id],
                "signal": samples.x[k],
                "sample_rate": float(FS_HZ),
                "channels": list(LEADS),
                "units": "mV",
            }
            for k, record_id in enumerate(ids)
        ]

    provenance = {
        "dataset": kind,
        "dataset_ref": f"{adapter.dataset_id}@{adapter.dataset_version}",
        "source": source,
        "source_sha256": source["sha256"],
        "registry": str(registry),
        "role": role,
        "manifest": str(Path(config["manifest"]).expanduser().resolve()) if manifest else None,
        "manifest_sha256": manifest.sha256 if manifest else None,
        "label_definitions": definitions,
        "label_namespaces": ["superclass", "scp", "artifact"],
        "population_scope": "PTB-XL records with a diagnostic superclass"
        + (" and at least one ECG-QA question" if kind == "ecgqa" else ""),
        "split_policy": "PTB-XL patient folds 1-8; CoT release split is descriptive only",
        "label_effect_status": "unreviewed; discovery does not approve labels after generation",
        "metadata_files": {
            str(path): _file_sha256(path) for path in (statements_path, database_path) if path
        },
    }
    _finish_provenance(provenance, rows)
    return RecordingSource(rows, provenance, load)


def _open_jsonl(config: dict, role: str) -> RecordingSource:
    if config.get("manifest"):
        raise ValueError(
            "JSONL uses explicit patient-level splits; PTB-XL challenge manifests are unsupported"
        )
    if not config.get("input"):
        raise ValueError("dataset=jsonl requires an input JSONL path")
    path = Path(config["input"]).expanduser().resolve()
    records: dict[str, dict] = {}
    signal_specs: dict[str, dict] = {}
    text_ids: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            record_id = _required_string(item, "record_id")
            raw_split = _required_string(item, "split")
            if raw_split not in _SPLITS:
                raise ValueError("split must be train, validation, or test")
            split = _SPLITS[raw_split]
            signal_path = Path(_required_string(item, "signal_path")).expanduser()
            signal_path = (path.parent / signal_path).resolve()
            if signal_path.suffix.lower() not in {".npy", ".npz"}:
                raise ValueError("signal_path must name a .npy or .npz file")
            sample_rate = float(item["sample_rate"])
            if not np.isfinite(sample_rate) or sample_rate <= 0:
                raise ValueError("sample_rate must be a finite positive number")
            channels = item["channels"]
            if (
                not isinstance(channels, list)
                or not channels
                or any(not isinstance(channel, str) or not channel for channel in channels)
                or len(set(channels)) != len(channels)
            ):
                raise ValueError("channels must be a non-empty list of unique names")
            spec = {
                "path": str(signal_path),
                "sample_rate": sample_rate,
                "channels": channels,
                "units": str(item.get("units", "unknown")),
                "signal_key": item.get("signal_key"),
            }
            row = {
                "record_id": record_id,
                "subject_uid": str(item.get("subject_uid") or record_id),
                "split": split,
                "labels": _strings(item.get("labels", []), "labels"),
                "metadata": dict(item.get("metadata") or {}),
                "texts": [],
                "artifact_types": _strings(item.get("artifact_types", []), "artifact_types"),
            }
            if record_id in records:
                previous = records[record_id]
                if signal_specs[record_id] != spec or any(
                    previous[key] != row[key]
                    for key in ("subject_uid", "split", "metadata", "artifact_types")
                ):
                    raise ValueError(f"conflicting recording metadata for repeated record {record_id!r}")
                previous["labels"] = sorted(set(previous["labels"]) | set(row["labels"]))
            else:
                records[record_id] = row
                signal_specs[record_id] = spec
            for text in _jsonl_texts(item, record_id, line_number):
                text_id = text["text_id"]
                identity = {"record_id": record_id, **text}
                if text_id in text_ids and text_ids[text_id] != identity:
                    raise ValueError(f"text_id {text_id!r} is reused for different text or recording")
                if text_id not in text_ids:
                    records[record_id]["texts"].append(text)
                    text_ids[text_id] = identity
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    all_rows = list(records.values())
    _check_subject_splits(all_rows)
    rows = [row for row in all_rows if row["split"] == "train"]
    # Hash selected signal bytes once per path without loading arrays or inspecting holdout signals.
    hashes = {signal_specs[row["record_id"]]["path"]: "" for row in rows}
    for signal_path in hashes:
        hashes[signal_path] = _file_sha256(Path(signal_path))
    selected_specs = {row["record_id"]: signal_specs[row["record_id"]] for row in rows}
    source = {
        "dataset": "jsonl",
        "input_sha256": _file_sha256(path),
        "signals": [
            {"record_id": record_id, **spec, "sha256": hashes[spec["path"]]}
            for record_id, spec in selected_specs.items()
        ],
    }
    # Content hashes, not machine-dependent absolute paths, determine source identity.
    source_sha256 = _sha256(
        {
            "input_sha256": source["input_sha256"],
            "signals": [
                {key: value for key, value in spec.items() if key != "path"} for spec in source["signals"]
            ],
        }
    )

    def load(ids: list[str]) -> list[dict]:
        loaded = []
        for record_id in ids:
            spec = selected_specs[record_id]
            # Detect mutation between analysis cataloging and waveform materialization.
            if _file_sha256(Path(spec["path"])) != hashes[spec["path"]]:
                raise ValueError(f"Signal file changed after cataloging: {spec['path']}")
            signal = np.load(spec["path"], allow_pickle=False)
            if isinstance(signal, np.lib.npyio.NpzFile):
                with signal as archive:
                    key = spec["signal_key"] or ("signal" if "signal" in archive else None)
                    if key is None and len(archive.files) == 1:
                        key = archive.files[0]
                    if key is None:
                        raise ValueError(f"{spec['path']}: ambiguous NPZ; set signal_key")
                    signal = archive[key]
            loaded.append(
                {
                    **records[record_id],
                    "signal": signal,
                    "sample_rate": spec["sample_rate"],
                    "channels": spec["channels"],
                    "units": spec["units"],
                }
            )
        return loaded

    provenance = {
        "dataset": "jsonl",
        "dataset_ref": config.get("dataset_ref") or "jsonl",
        "input": str(path),
        "source": source,
        "source_sha256": source_sha256,
        "role": role,
        "manifest": None,
        "manifest_sha256": None,
        "label_namespaces": ["user supplied"],
        "label_definitions": {},
        "population_scope": "Explicit train split; one point per distinct record_id",
        "split_policy": "Explicit splits with cross-split subject overlap rejected",
        "label_effect_status": "unreviewed; discovery does not approve labels after generation",
    }
    _finish_provenance(provenance, rows)
    return RecordingSource(rows, provenance, load)


def _jsonl_texts(item: dict, record_id: str, line_number: int) -> list[dict]:
    values = list(item.get("texts") or [])
    if any(item.get(key) is not None for key in ("text", "question", "answer")):
        values.insert(0, item)
    out = []
    for k, value in enumerate(values):
        question, answer = str(value.get("question") or ""), str(value.get("answer") or "")
        text = str(value.get("text") or "")
        if not any(part.strip() for part in (text, question, answer)):
            continue
        out.append(
            {
                "text_id": str(
                    value.get("text_id") or value.get("task_id") or f"{record_id}:line{line_number}:text{k}"
                ),
                "text": text,
                "question": question,
                "answer": answer,
                "rationale": str(value.get("rationale") or ""),
                "metadata": dict(
                    value.get("text_metadata") or (value.get("metadata") if value is not item else {}) or {}
                ),
            }
        )
    return out


def _check_subject_splits(rows: list[dict]) -> None:
    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits[row["subject_uid"]].add(row["split"])
    overlap = sorted(subject for subject, memberships in splits.items() if len(memberships) > 1)
    if overlap:
        raise LeakageError(f"Subjects occur in multiple dataset splits: {overlap[:5]}")


def _fold_split(fold: int) -> str:
    if 1 <= fold <= 8:
        return "train"
    if fold in (9, 10):
        return {9: "validation", 10: "test"}[fold]
    raise ValueError(f"Invalid PTB-XL strat_fold {fold!r}")


def _ptbxl_files(config: dict, registry: Path) -> tuple[Path | None, Path | None]:
    root = Path(config["ptbxl_root"]).expanduser().resolve() if config.get("ptbxl_root") else None
    statements = (
        Path(config["scp_statements"]).expanduser().resolve() if config.get("scp_statements") else None
    )
    if statements and not statements.is_file():
        raise FileNotFoundError(statements)
    if root is None:
        candidates = sorted((registry.parent / "raw" / "ptb-xl").glob("**/ptbxl_database.csv"))
        if len(candidates) == 1:
            root = candidates[0].parent
    if root and not root.is_dir():
        raise FileNotFoundError(root)
    if statements is None and root and (root / "scp_statements.csv").is_file():
        statements = root / "scp_statements.csv"
    database = root / "ptbxl_database.csv" if root else None
    return statements, database if database and database.is_file() else None


def _label_definitions(path: Path | None) -> dict[str, dict]:
    definitions = {
        f"superclass:{key}": {"description": value, "namespace": "superclass"}
        for key, value in _SUPERCLASSES.items()
    }
    for flag in ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems"):
        definitions[f"artifact:{flag}"] = {"description": flag.replace("_", " "), "namespace": "artifact"}
    if path:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                code = row.get("") or row.get("code") or row.get("scp_code")
                if code:
                    definitions[f"scp:{code}"] = {
                        "description": row.get("description")
                        or row.get("SCP-ECG Statement Description")
                        or code,
                        "namespace": "scp",
                        "categories": [
                            name for name in ("diagnostic", "form", "rhythm") if row.get(name) in {"1", "1.0"}
                        ],
                        "diagnostic_class": row.get("diagnostic_class") or None,
                        "diagnostic_subclass": row.get("diagnostic_subclass") or None,
                    }
    return definitions


def _clinical_metadata(path: Path | None, record_ids: set[str]) -> dict[str, dict]:
    if path is None:
        return {}
    # Context for meaningful slices; patient IDs and split remain adapter-controlled.
    fields = (
        "age",
        "sex",
        "height",
        "weight",
        "site",
        "device",
        "heart_axis",
        "infarction_stadium1",
        "infarction_stadium2",
        "report",
        "validated_by_human",
        "pacemaker",
        "extra_beats",
    )
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            record_id = f"ptbxl-{row['ecg_id']}"
            if record_id in record_ids:
                values = {field: row[field] for field in fields if row.get(field)}
                for field in ("age", "sex", "height", "weight"):
                    if field in values:
                        try:
                            number = float(values[field])
                            if np.isfinite(number):
                                values[field] = number
                        except ValueError:
                            pass
                out[record_id] = values
    return out


def _finish_provenance(provenance: dict, rows: list[dict]) -> None:
    provenance.update(
        {
            "split": "train",
            "catalog_sha256": _sha256(rows),
            "population_records": len(rows),
            "population_subjects": len({row["subject_uid"] for row in rows}),
            "population_texts": sum(len(row["texts"]) for row in rows),
            "population_distinct_questions": len(
                {
                    (row["record_id"], text["question"])
                    for row in rows
                    for text in row["texts"]
                    if text["question"]
                }
            ),
        }
    )


def _required_string(item: dict, key: str) -> str:
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _strings(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return sorted(set(value))


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
