"""Convert configuration values into checked Syntide dataclass instances.

OmegaConf conversion has a broad return type. These helpers verify that a
structured configuration produces the requested dataclass.
"""

from typing import Any

from omegaconf import OmegaConf


def as_dataclass[T](cls: type[T], config: Any) -> T:
    """Convert a structured configuration to the requested dataclass.

    Args:
        cls: Expected dataclass type.
        config: Structured OmegaConf configuration to convert.

    Returns:
        The converted configuration as an instance of cls.

    Raises:
        TypeError: The converted value is not an instance of cls.
    """
    built = OmegaConf.to_object(config)
    if not isinstance(built, cls):
        raise TypeError(
            f"the config builds a {type(built).__name__}, not a {cls.__name__}"
        )
    return built


def load_dataclass[T](cls: type[T], *sources: Any) -> T:
    """Merge configuration sources over a dataclass's defaults.

    Args:
        cls: Dataclass type that defines the configuration fields and
            defaults.
        *sources: Configuration objects or mappings, applied from left to
            right.

    Returns:
        A checked instance of cls containing the merged values.
    """
    return as_dataclass(
        cls, OmegaConf.merge(OmegaConf.structured(cls), *sources)
    )
