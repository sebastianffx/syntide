"""Typed bridges from OmegaConf to SYNTIDE's config dataclasses.

``OmegaConf.to_object`` returns ``dict | list | str | Any | None``, so a type checker cannot tell that a
structured config builds the dataclass it was declared with. These helpers build it and check it.
"""

from typing import Any

from omegaconf import OmegaConf


def as_dataclass[T](cls: type[T], config: Any) -> T:
    """``OmegaConf.to_object(config)``, refused unless it builds a ``cls`` (e.g. Hydra's ``cfg``)."""
    built = OmegaConf.to_object(config)
    if not isinstance(built, cls):
        raise TypeError(f"the config builds a {type(built).__name__}, not a {cls.__name__}")
    return built


def load_dataclass[T](cls: type[T], *sources: Any) -> T:
    """``sources`` (dicts, loaded YAML, configs) merged in order over the schema of ``cls``."""
    return as_dataclass(cls, OmegaConf.merge(OmegaConf.structured(cls), *sources))
