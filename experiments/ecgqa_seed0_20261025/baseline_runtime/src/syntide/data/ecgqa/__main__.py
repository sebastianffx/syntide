"""``python -m syntide.data.ecgqa``: build ECG-QA-CoT into TimeNet, then write and check its manifests.

The roles are PTB-XL's (``syntide.data.challenge``), built on the records that have questions, with
whole patients, PTB-XL folds and the same seed. A run's questions are the questions on its role's
records. Each manifest passes Gate 1 and the question-level check before it is written, and
``counts.json`` beside the manifests reports ECGs, patients, questions and noisy shares per role,
plus how far the CoT release's own split disagrees with the folds.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from timenet.engine import run_pipeline

from syntide.data.challenge import ChallengeSpec, Manifest, build_manifest, pool_name, save_manifest
from syntide.data.ecgqa.adapter import DATASET, SIGNALS, EcgQaAdapter
from syntide.data.ecgqa.connector import SOURCES, EcgQaCotConnector
from syntide.data.ecgqa.signals import signal_agreement
from syntide.data.leakage import check_manifest, check_tasks

log = logging.getLogger(__name__)


@dataclass
class EcgQaDataConfig:
    registry: str = "datasets/timenet-registry"
    ptbxl_cache: str = "datasets/raw/ptb-xl"  # the PTB-XL extraction python -m syntide.data made
    download_cache: str = "datasets/raw/ecg-qa-cot"  # template answers and the CoT archive
    dataset: str = DATASET
    signals: str = SIGNALS
    manifests_dir: str = "manifests/${challenge.id}"
    signal_check_records: int = 8  # how many ECGs compare ecg-qa-cot's 500 Hz leads with PTB-XL's 100 Hz
    challenge: ChallengeSpec = field(
        default_factory=lambda: ChallengeSpec(id="ecgqa-artifact-50", pool_fractions=[0.5, 1.0])
    )


ConfigStore.instance().store(name="ecgqa_data_schema", node=EcgQaDataConfig)


@hydra.main(version_base=None, config_path="../../../../conf", config_name="data_ecgqa")
def main(cfg: DictConfig) -> None:
    config: EcgQaDataConfig = OmegaConf.to_object(cfg)
    version_dir = run_pipeline(
        EcgQaCotConnector(config.ptbxl_cache),
        Path(config.registry),
        cache_dir=Path(config.download_cache),
        keep_cache=True,
    )
    log.info("TimeNet dataset ready at %s (sources %s)", version_dir, SOURCES)

    adapter = EcgQaAdapter(config.registry, config.dataset, config.signals)
    tasks, index, source = adapter.tasks(), adapter.index(), adapter.source()
    log.info(
        "%d questions on %d labelled PTB-XL ECGs; dropped %s; source sha256 %s",
        tasks.num_rows,
        index.num_rows,
        adapter.dropped(),
        source["sha256"][:12],
    )

    checked = _spread(index["record_id"].to_pylist(), config.signal_check_records)
    agreement = signal_agreement(adapter, checked)
    log.info("500 Hz (ecg-qa-cot, decimated) vs 100 Hz (ptb-xl): %s", agreement["summary"])

    counts = {"source": source, "dropped": adapter.dropped(), "signal_agreement": agreement, "manifests": {}}
    for fraction in config.challenge.pool_fractions:
        manifest = build_manifest(index, config.challenge, fraction, source)
        check_manifest(manifest, index, source)  # Gate 1, by record
        check_tasks(tasks, manifest, index)  # Gate 1, by question
        path = Path(config.manifests_dir) / f"{pool_name(fraction)}.json"
        save_manifest(manifest, path)
        counts["manifests"][path.name] = role_counts(manifest, tasks)
        table = _table(counts["manifests"][path.name])
        log.info("%s passed the leakage gate -> %s\n%s", manifest.challenge_id, path, table)
    counts["cot_split_vs_folds"] = split_disagreement(tasks, config.challenge)
    log.info("CoT split vs PTB-XL folds: %s", counts["cot_split_vs_folds"])
    out = Path(config.manifests_dir) / "counts.json"
    out.write_text(json.dumps(counts, indent=1) + "\n", encoding="utf-8")
    log.info("counts -> %s", out)


def role_counts(manifest: Manifest, tasks) -> dict[str, dict[str, float]]:
    """ECGs, patients, questions and noisy shares per role."""
    columns = tasks.select(["record_id", "subject_uid", "is_artifact"]).to_pydict()
    questions = Counter(columns["record_id"])
    subject = dict(zip(columns["record_id"], columns["subject_uid"], strict=True))
    noisy = {r for r, flag in zip(columns["record_id"], columns["is_artifact"], strict=True) if flag}
    out = {}
    for role, ids in manifest.roles.items():
        n_questions = sum(questions[r] for r in ids)
        n_noisy = sum(questions[r] for r in ids if r in noisy)
        out[role] = {
            "ecgs": len(ids),
            "patients": len({subject[r] for r in ids}),
            "questions": n_questions,
            "noisy_ecg_share": round(sum(r in noisy for r in ids) / max(1, len(ids)), 4),
            "noisy_question_share": round(n_noisy / max(1, n_questions), 4),
        }
    return out


def split_disagreement(tasks, spec: ChallengeSpec) -> dict:
    """How the CoT release's split maps onto the PTB-XL folds, in questions, ECGs and patients."""
    columns = tasks.select(["record_id", "subject_uid", "strat_fold", "cot_split"]).to_pydict()
    partition = {
        **dict.fromkeys(spec.train_folds, "train"),
        **dict.fromkeys(spec.val_folds, "validation"),
        **dict.fromkeys(spec.test_folds, "test"),
    }
    questions, ecgs, patients = Counter(), {}, {}
    for record_id, subject, fold, cot in zip(*columns.values(), strict=True):
        key = f"cot {cot} -> folds {partition[fold]}"
        questions[key] += 1
        ecgs.setdefault(key, set()).add(record_id)
        patients.setdefault(key, set()).add(subject)
    return {
        key: {"questions": questions[key], "ecgs": len(ecgs[key]), "patients": len(patients[key])}
        for key in sorted(questions)
    }


def _spread(ids: list[str], n: int) -> list[str]:
    """``n`` ids spread evenly over the list (deterministic)."""
    if n <= 0 or not ids:
        return []
    return [ids[i] for i in np.linspace(0, len(ids) - 1, min(n, len(ids))).round().astype(int)]


def _table(counts: dict[str, dict[str, float]]) -> str:
    lines = [f"  {'role':18s} {'ECGs':>6s} {'patients':>8s} {'questions':>9s} {'noisy ECGs':>10s}"]
    for role, c in counts.items():
        lines.append(
            f"  {role:18s} {c['ecgs']:6d} {c['patients']:8d} {c['questions']:9d} {c['noisy_ecg_share']:10.1%}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
