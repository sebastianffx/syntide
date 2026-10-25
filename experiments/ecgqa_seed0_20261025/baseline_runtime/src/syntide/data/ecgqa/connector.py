"""ECG-QA-CoT as a TimeNet dataset, built from the PTB-XL archive SYNTIDE already holds.

The dataset is timenet-connectors' own ``physionet/ecg-qa-cot@1.0.0`` (same card, same records, same
tasks). Only the download differs: upstream fetches PTB-XL again from ``s3://``, while this subclass
reuses the extracted archive under ``datasets/raw/ptb-xl`` (fetched over HTTPS by ``python -m
syntide.data``). The template answers and the OpenTSLM CoT archive go to their own cache directory.
"""

import asyncio
import inspect
from pathlib import Path
from typing import ClassVar

from timenet_connectors.datasets.physionet.ecg_qa_cot import connector as upstream
from timenet_connectors.download import Artifact, download_files, ensure_archive, find_dir_containing

from syntide.data.ptbxl.connector import PTBXL_URL

DATASET = "physionet/ecg-qa-cot@1.0.0"
# The source URLs this build reads, recorded in provenance.
SOURCES = {
    "ptbxl": PTBXL_URL,
    "template_answers": upstream.ECG_QA_TEMPLATE_ANSWERS_URL,
    "cot": upstream.ECG_QA_COT_URL,
}


class EcgQaCotConnector(upstream.EcgQaCotConnector):
    """Upstream's connector, reading PTB-XL from an existing extraction instead of S3."""

    CARD: ClassVar[Path] = Path(inspect.getfile(upstream.EcgQaCotConnector)).with_name("dataset.yaml")

    def __init__(self, ptbxl_cache: str | Path = "datasets/raw/ptb-xl") -> None:
        super().__init__()
        self.ptbxl_cache = Path(ptbxl_cache)

    async def download_async(self, cache_dir: Path) -> list[upstream.EcgQaCotSource]:
        ptbxl = await ensure_archive(PTBXL_URL, self.ptbxl_cache)
        ptbxl_root = find_dir_containing(ptbxl, "ptbxl_database.csv")
        answers_path = cache_dir / "answers_for_each_template.csv"
        _, cot_root = await asyncio.gather(
            download_files([Artifact(upstream.ECG_QA_TEMPLATE_ANSWERS_URL, answers_path)]),
            ensure_archive(upstream.ECG_QA_COT_URL, cache_dir),
        )
        cot_csvs = tuple(
            (split, find_dir_containing(cot_root, name) / name) for split, name in self._COT_CSVS
        )
        return [
            upstream.EcgQaCotSource(
                records_root=ptbxl_root / "records500", answers_path=answers_path, cot_csvs=cot_csvs
            )
        ]
