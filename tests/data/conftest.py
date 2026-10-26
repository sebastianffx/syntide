"""Build small synthetic PTB-XL datasets through the TimeNet data pipeline."""

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import wfdb
from timenet.engine import run_pipeline

from syntide.data.adapter import PtbXlAdapter
from syntide.data.ptbxl.connector import PtbXlConnector, PtbXlSource
from syntide.data.schema import LEADS

FS = 100
N_SAMPLES = 100

# ecg_id -> (patient_id, fold, scp_codes, noise columns). Chosen to hit every
# labelling rule:
# 1 and 2 share a patient; 2 has two MI codes (deduplicated) plus an STTC code;
# 3 has only a rhythm
# code (no diagnostic task); 4 has a whitespace-only noise value, which must
# count as clean.
RECORDS = {
    1: (101, 1, {"NORM": 100.0, "SR": 0.0}, {"static_noise": " , I-V1,  "}),
    2: (
        101,
        1,
        {"IMI": 50.0, "ASMI": 100.0, "NDT": 0.0},
        {"burst_noise": "alles"},
    ),
    3: (202, 9, {"SR": 0.0}, {}),
    4: (303, 10, {"LVH": 100.0, "CLBBB": 100.0}, {"baseline_drift": "   "}),
}
STATEMENTS = [  # (code, diagnostic, diagnostic_class)
    ("NORM", "1.0", "NORM"),
    ("IMI", "1.0", "MI"),
    ("ASMI", "1.0", "MI"),
    ("NDT", "1.0", "STTC"),
    ("LVH", "1.0", "HYP"),
    ("CLBBB", "1.0", "CD"),
    ("SR", "", ""),
]
NOISE_COLUMNS = (
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
)


@dataclass(frozen=True)
class BuiltPtbXl:
    adapter: PtbXlAdapter
    registry: Path
    tree: Path  # the synthetic PTB-XL directory the registry was built from
    signals: dict[
        int, np.ndarray
    ]  # ecg_id -> (N_SAMPLES, 12) physical values as written, mV
    adc_step: dict[int, float]  # ecg_id -> largest ADC step across leads, mV


def _write_tree(root: Path) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    t = np.arange(N_SAMPLES) / FS
    signals, steps = {}, {}
    for ecg_id in RECORDS:
        # Lead k carries a distinct offset, so any lead-order mix-up changes the
        # values.
        signal = np.stack(
            [
                0.1 * k + 0.5 * np.sin(2 * np.pi * 1.2 * t + k + ecg_id)
                for k in range(12)
            ],
            axis=1,
        )
        bucket = root / "records100" / "00000"
        bucket.mkdir(parents=True, exist_ok=True)
        name = f"{ecg_id:05d}_lr"
        wfdb.wrsamp(
            name,
            fs=FS,
            units=["mV"] * 12,
            sig_name=list(LEADS),
            p_signal=signal,
            fmt=["16"] * 12,
            write_dir=str(bucket),
        )
        signals[ecg_id] = signal
        steps[ecg_id] = float(
            1.0 / np.min(wfdb.rdheader(str(bucket / name)).adc_gain)
        )

    with (root / "ptbxl_database.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = [
            "ecg_id",
            "patient_id",
            "scp_codes",
            *NOISE_COLUMNS,
            "strat_fold",
            "filename_lr",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for ecg_id, (patient, fold, codes, noise) in RECORDS.items():
            writer.writerow(
                {
                    "ecg_id": ecg_id,
                    "patient_id": f"{patient}.0",
                    "scp_codes": repr(codes),
                    **{
                        column: noise.get(column, "")
                        for column in NOISE_COLUMNS
                    },
                    "strat_fold": fold,
                    "filename_lr": f"records100/00000/{ecg_id:05d}_lr",
                }
            )
    with (root / "scp_statements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "diagnostic", "diagnostic_class"])
        writer.writerows(STATEMENTS)
    return signals, steps


@pytest.fixture(scope="session")
def build_ptbxl_registry(tmp_path_factory) -> Callable[[Path], Path]:
    """Build a synthetic PTB-XL registry without network access."""

    def build(tree: Path) -> Path:
        async def local_download(self, cache_dir):
            return [PtbXlSource(root=tree)]

        registry = tmp_path_factory.mktemp("registry")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(PtbXlConnector, "download_async", local_download)
            run_pipeline(
                PtbXlConnector(),
                registry,
                cache_dir=tmp_path_factory.mktemp("cache"),
                keep_cache=True,
            )
        return registry

    return build


@pytest.fixture(scope="session")
def built_ptbxl(tmp_path_factory, build_ptbxl_registry) -> BuiltPtbXl:
    tree = tmp_path_factory.mktemp("ptbxl") / "ptb-xl-1.0.3"
    signals, steps = _write_tree(tree)
    registry = build_ptbxl_registry(tree)
    return BuiltPtbXl(
        adapter=PtbXlAdapter(registry),
        registry=registry,
        tree=tree,
        signals=signals,
        adc_step=steps,
    )
