"""A small synthetic PTB-XL + ECG-QA-CoT registry built through the production TimeNet pipeline.

Twenty-four 10 s, 100 Hz records over ten folds, with known heart rates (``tests/qa/fixtures.py``).
They are test fixtures, not clinically validated ECGs. No real PTB-XL or ECG-QA bytes are used.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import wfdb
from tests.qa.fixtures import fixture_ecg
from timenet.engine import run_pipeline
from timenet_connectors.datasets.physionet.ecg_qa_cot.connector import EcgQaCotConnector, EcgQaCotSource

from syntide.data.challenge import ChallengeSpec, build_manifest, save_manifest
from syntide.data.ptbxl.connector import PtbXlConnector, PtbXlSource
from syntide.data.schema import LEADS
from syntide.explore.qa import EcgQaView

RATES = (55, 72, 88, 110, 64, 95, 130, 48)


@dataclass(frozen=True)
class QARegistry:
    registry: Path
    manifest: Path
    view: EcgQaView
    rates: dict[str, float]


def _record_rows() -> list[dict]:
    rows = []
    for ecg_id in range(1, 25):
        patient = 500 + (ecg_id + 1) // 2  # two ECGs per patient
        fold = (patient % 10) + 1
        drift = ecg_id % 4 == 0
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient": patient,
                "fold": fold,
                "drift": drift,
                "codes": {"NORM": 100.0} if ecg_id % 3 else {"IMI": 100.0, "NDT": 50.0},
                "rate": RATES[ecg_id % len(RATES)],
            }
        )
    return rows


def _write_ptbxl(root: Path) -> dict[str, float]:
    bucket = root / "records100" / "00000"
    bucket.mkdir(parents=True)
    rates = {}
    for row in _record_rows():
        x = fixture_ecg(row["rate"], seed=row["ecg_id"], wander=0.8 if row["drift"] else 0.05)
        wfdb.wrsamp(
            f"{row['ecg_id']:05d}_lr",
            fs=100,
            units=["mV"] * 12,
            sig_name=list(LEADS),
            p_signal=x.T.astype(np.float64),
            fmt=["16"] * 12,
            write_dir=str(bucket),
        )
        rates[f"ptbxl-{row['ecg_id']}"] = row["rate"]
    columns = [
        "ecg_id",
        "patient_id",
        "scp_codes",
        "baseline_drift",
        "static_noise",
        "burst_noise",
        "electrodes_problems",
        "strat_fold",
        "filename_lr",
    ]
    with (root / "ptbxl_database.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in _record_rows():
            writer.writerow(
                {
                    "ecg_id": row["ecg_id"],
                    "patient_id": f"{row['patient']}.0",
                    "scp_codes": repr(row["codes"]),
                    "baseline_drift": "alles" if row["drift"] else "",
                    "static_noise": "",
                    "burst_noise": "",
                    "electrodes_problems": "",
                    "strat_fold": row["fold"],
                    "filename_lr": f"records100/00000/{row['ecg_id']:05d}_lr",
                }
            )
    with (root / "scp_statements.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "diagnostic", "diagnostic_class"])
        writer.writerows([("NORM", "1.0", "NORM"), ("IMI", "1.0", "MI"), ("NDT", "1.0", "STTC")])
    return rates


def _write_ecgqa(root: Path) -> tuple[Path, Path, list[tuple[str, Path]]]:
    bucket = root / "records500" / "00000"
    bucket.mkdir(parents=True)
    for row in _record_rows():
        x = fixture_ecg(row["rate"], seed=row["ecg_id"], fs=500.0)
        wfdb.wrsamp(
            f"{row['ecg_id']:05d}_hr",
            fs=500,
            units=["mV"] * 12,
            sig_name=list(LEADS),
            p_signal=x.T.astype(np.float64),
            fmt=["16"] * 12,
            write_dir=str(bucket),
        )
    answers = root / "answers.csv"
    with answers.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["template_id", "classes"])
        writer.writerows([(4, "['none', 'myocardial infarction']"), (27, "['no', 'yes']")])
    fields = ["ecg_id", "template_id", "question_type", "question", "answer", "rationale", "clinical_context"]
    path = root / "train.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in _record_rows():
            quality = "baseline drift noted" if row["drift"] else "no noise"
            context = f"{40 + row['ecg_id']}-year-old patient. 12-lead ECG. Signal quality: {quality}."
            drift = "yes" if row["drift"] else "no"
            mi = "myocardial infarction" if "IMI" in row["codes"] else "none"
            writer.writerow(
                {
                    "ecg_id": f"[{row['ecg_id']}]",
                    "template_id": 27,
                    "question_type": "single-verify",
                    "question": "Does this ECG show baseline drift?",
                    "answer": drift,
                    "rationale": f"The baseline looks {'unstable' if drift == 'yes' else 'stable'}."
                    f"\nAnswer: {drift}",
                    "clinical_context": context,
                }
            )
            writer.writerow(
                {
                    "ecg_id": f"[{row['ecg_id']}]",
                    "template_id": 4,
                    "question_type": "single-choose",
                    "question": "Which diagnostic symptom does this ECG show?",
                    "answer": mi,
                    "rationale": f"Considering the waveform.\nAnswer: {mi}",
                    "clinical_context": context,
                }
            )
    return root / "records500", answers, [("train", path)]


@pytest.fixture(scope="session")
def qa_registry(tmp_path_factory) -> QARegistry:
    base = tmp_path_factory.mktemp("qa-registry")
    tree = base / "ptb-xl-1.0.3"
    tree.mkdir()
    rates = _write_ptbxl(tree)
    records500, answers, cot = _write_ecgqa(base / "cot")
    registry = base / "registry"

    async def ptbxl_download(self, cache_dir):
        return [PtbXlSource(root=tree)]

    async def cot_download(self, cache_dir):
        return [EcgQaCotSource(records_root=records500, answers_path=answers, cot_csvs=tuple(cot))]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PtbXlConnector, "download_async", ptbxl_download)
        patch.setattr(EcgQaCotConnector, "download_async", cot_download)
        run_pipeline(PtbXlConnector(), registry, cache_dir=base / "cache1", keep_cache=True)
        run_pipeline(EcgQaCotConnector(), registry, cache_dir=base / "cache2", keep_cache=True)
    view = EcgQaView(registry)
    manifest = build_manifest(
        view.index(), ChallengeSpec(id="qa-test", pool_fractions=[1.0]), 1.0, view.source()
    )
    path = base / "manifests" / "pool100.json"
    save_manifest(manifest, path)
    return QARegistry(registry, path, view, rates)
