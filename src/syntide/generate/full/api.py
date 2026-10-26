"""Define requests and results for complete time-series generation."""

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from syntide.explore.plans import content_id


def _name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{field} must be a non-empty string without surrounding whitespace"
        )
    return value


def json_object(value: Any, field: str) -> dict:
    """Validate and copy a finite JSON object for content checksumming."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    try:
        content_id(value)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{field} must contain finite, JSON-compatible values: {error}"
        ) from error
    return value


@dataclass(frozen=True)
class SignalSpec:
    modality: str
    channels: tuple[str, ...]
    sample_rate: float
    num_samples: int
    units: str

    def __post_init__(self) -> None:
        _name(self.modality, "modality")
        _name(self.units, "units")
        if not isinstance(self.channels, list | tuple) or not self.channels:
            raise ValueError("channels must be a non-empty list or tuple")
        for channel in self.channels:
            _name(channel, "channel")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must have unique names")
        object.__setattr__(self, "channels", tuple(self.channels))
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int | float)
            or not np.isfinite(self.sample_rate)
            or self.sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a finite positive number")
        object.__setattr__(self, "sample_rate", float(self.sample_rate))
        if (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, int)
            or self.num_samples <= 0
        ):
            raise ValueError("num_samples must be a positive integer")
        json_object(self.to_dict(), "signal specification")

    @classmethod
    def from_dict(cls, value: dict) -> "SignalSpec":
        json_object(value, "signal")
        expected = {
            "modality",
            "channels",
            "sample_rate",
            "num_samples",
            "units",
        }
        unknown, missing = set(value) - expected, expected - set(value)
        if unknown or missing:
            raise ValueError(
                f"signal fields: unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}"
            )
        return cls(**value)

    def to_dict(self) -> dict:
        return {**asdict(self), "channels": list(self.channels)}


@dataclass(frozen=True)
class GenerationRequest:
    seed: int
    signal: SignalSpec
    conditioning: dict

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not isinstance(self.signal, SignalSpec):
            raise ValueError("signal must be a SignalSpec")
        json_object(self.conditioning, "conditioning")


@dataclass(frozen=True)
class GeneratedSignal:
    signal: np.ndarray
    spec: SignalSpec
    metadata: dict


class GeneratorBackend(Protocol):
    """Signal generator interface with validation before model loading."""

    def validate_request(self, request: GenerationRequest) -> None: ...

    def generate(self, request: GenerationRequest) -> GeneratedSignal: ...

    def provenance(self) -> dict: ...
