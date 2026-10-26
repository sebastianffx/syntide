"""Expose PTB-XL version 1.0.3 as a TimeNet dataset.

Each recording is a ten-second, twelve-lead electrocardiogram sampled at 100
hertz in millivolts. Store patient, fold, and four signal-quality flags as
recording annotations, and retain original diagnostic codes in metadata.
Recordings with a diagnostic class receive a multi-label classification
task. Training uses folds one through eight, validation uses fold nine, and
testing uses fold ten. Patient identifiers and split membership are stored
explicitly because TimeNet version 0.11 does not preserve dedicated fields
for them.
"""

import ast
import csv
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.types import Annotation, ClassificationTask, TimeSeriesSpec, ureg
from timenet_connectors.bases.physionet import BasePhysioNetConnector
from timenet_connectors.download import ensure_archive, find_dir_containing

from syntide.data.schema import ARTIFACT_TYPES

# PhysioNet's open bucket over plain HTTPS (no boto3 needed).
PTBXL_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"
TARGET_SCHEMA = "ptbxl.diagnostic_superclass"
OFFICIAL_SPLIT = {
    **dict.fromkeys(range(1, 9), "train"),
    9: "validation",
    10: "test",
}

_ECG = TimeSeriesSpec(
    spec_type="ecg", name="12-lead ECG", unit_value=ureg.millivolt
)


@dataclass(frozen=True)
class PtbXlSource:
    """The extracted directory containing ptbxl_database.csv."""

    root: Path


def load_superclass_map(statements_csv: Path) -> dict[str, str]:
    """Map diagnostic codes to classes using scp_statements.csv."""
    with statements_csv.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        # The code column is unnamed in the official file.
        return {
            row[""]: row["diagnostic_class"]
            for row in rows
            if row["diagnostic"] == "1.0"
        }


def diagnostic_superclasses(
    scp_codes: Mapping[str, float], code_to_class: Mapping[str, str]
) -> tuple[str, ...]:
    """Return the sorted superclasses of a record's diagnostic codes.

    Every diagnostic code counts regardless of its likelihood value (the
    standard PTB-XL aggregation); form and rhythm codes are not diagnostic
    and are ignored.
    """
    return tuple(
        sorted(
            {code_to_class[code] for code in scp_codes if code in code_to_class}
        )
    )


def artifact_flags(row: Mapping[str, str]) -> dict[str, str]:
    """Return present signal-quality flags and their lead descriptions.

    PTB-XL leaves a column empty when the flag is absent; whitespace-only
    values count as absent.
    """
    return {
        column: value
        for column in ARTIFACT_TYPES
        if (value := (row.get(column) or "").strip())
    }


class PtbXlConnector(BasePhysioNetConnector[PtbXlSource]):
    """Connector for PTB-XL 1.0.3 (100 Hz records)."""

    async def download_async(self, cache_dir: Path) -> list[PtbXlSource]:
        root = await ensure_archive(PTBXL_URL, cache_dir)
        return [
            PtbXlSource(root=find_dir_containing(root, "ptbxl_database.csv"))
        ]

    def convert(self, raw_refs: list[PtbXlSource]) -> TimeFDataset:
        root = raw_refs[0].root
        code_to_class = load_superclass_map(root / "scp_statements.csv")
        dataset = TimeFDataset(metadata=self.metadata())
        tasks = []
        with (root / "ptbxl_database.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                ecg_id = int(row["ecg_id"])
                fold = int(row["strat_fold"])
                scp_codes = ast.literal_eval(row["scp_codes"])
                record = dataset.add_record(
                    record=self._record(root, row, ecg_id, scp_codes)
                )
                record.add_annotations(
                    [
                        Annotation(
                            key="patient_id",
                            value=int(float(row["patient_id"])),
                            id=f"ptbxl-{ecg_id}-patient",
                        ),
                        Annotation(
                            key="strat_fold",
                            value=fold,
                            id=f"ptbxl-{ecg_id}-fold",
                        ),
                        *(
                            Annotation(
                                key=column,
                                value=leads,
                                id=f"ptbxl-{ecg_id}-{column}",
                            )
                            for column, leads in artifact_flags(row).items()
                        ),
                    ]
                )
                superclasses = diagnostic_superclasses(scp_codes, code_to_class)
                if superclasses:
                    tasks.append(
                        ClassificationTask(
                            id=f"ptbxl-{ecg_id}-dx",
                            inputs=(record,),
                            targets=superclasses,
                            target_schema=TARGET_SCHEMA,
                            metadata={"split": OFFICIAL_SPLIT[fold]},
                        )
                    )
        dataset.add_tasks(tasks=tasks)
        return dataset

    def _record(
        self, root: Path, row: Mapping[str, str], ecg_id: int, scp_codes: dict
    ) -> Record:
        record_base = root / row["filename_lr"]
        header = self._read_header(record_base)
        axis = RegularAxis.from_rate_hz(Fraction(str(header.fs)))
        signals = tuple(
            Signal.from_loader(
                spec=_ECG,
                name=name,
                time_axis=axis,
                loader=self._lead_loader(record_base, header, lead_idx),
                id=f"ecg-{ecg_id}-{name}",
                n_values=header.sig_len,
                source_id=f"ptbxl-{ecg_id}",
            )
            for lead_idx, name in enumerate(header.sig_name)
        )
        return Record(
            record_id=f"ptbxl-{ecg_id}",
            sources=(
                Source(
                    id=f"ptbxl-{ecg_id}-source",
                    name="PTB-XL ECG",
                    signals=signals,
                ),
            ),
            metadata={"ptbxl.scp_codes": scp_codes},
        )
