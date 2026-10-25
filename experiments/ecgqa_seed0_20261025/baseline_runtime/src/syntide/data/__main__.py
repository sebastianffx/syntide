"""``python -m syntide.data``: build PTB-XL into TimeNet, then write and check the challenge manifests.

The build is skipped when the registry already holds the version; the raw download is cached in
``download_cache`` so a rebuild does not fetch 1.8 GB again. Manifests are written under
``manifests/<challenge.id>/``; an existing manifest with different content is never overwritten.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from timenet.engine import run_pipeline

from syntide.data.adapter import PtbXlAdapter
from syntide.data.challenge import ChallengeSpec, build_manifest, pool_name, save_manifest
from syntide.data.leakage import check_manifest
from syntide.data.ptbxl import PtbXlConnector

log = logging.getLogger(__name__)


@dataclass
class DataConfig:
    registry: str = "datasets/timenet-registry"
    download_cache: str = "datasets/raw/ptb-xl"
    dataset: str = "physionet/ptb-xl@1.0.3"
    manifests_dir: str = "manifests/${challenge.id}"
    challenge: ChallengeSpec = field(default_factory=ChallengeSpec)


ConfigStore.instance().store(name="data_schema", node=DataConfig)


@hydra.main(version_base=None, config_path="../../../conf", config_name="data")
def main(cfg: DictConfig) -> None:
    config: DataConfig = OmegaConf.to_object(cfg)
    version_dir = run_pipeline(
        PtbXlConnector(), Path(config.registry), cache_dir=Path(config.download_cache), keep_cache=True
    )
    log.info("TimeNet dataset ready at %s", version_dir)

    adapter = PtbXlAdapter(config.registry, config.dataset)
    index = adapter.index()
    source = adapter.source()
    log.info("%d labelled records, source sha256 %s", index.num_rows, source["sha256"][:12])

    for fraction in config.challenge.pool_fractions:
        manifest = build_manifest(index, config.challenge, fraction, source)
        check_manifest(manifest, index, source)  # Gate 1
        path = Path(config.manifests_dir) / f"{pool_name(fraction)}.json"
        save_manifest(manifest, path)
        counts = ", ".join(f"{role}={len(ids)}" for role, ids in manifest.roles.items())
        log.info("%s passed the leakage gate -> %s\n  %s", manifest.challenge_id, path, counts)


if __name__ == "__main__":
    main()
