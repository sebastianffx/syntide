"""Generate complete recordings with optional model implementations."""

from syntide.generate.full.api import (
    GeneratedSignal,
    GenerationRequest,
    SignalSpec,
)
from syntide.generate.full.registry import create_generator, register_generator

__all__ = [
    "GeneratedSignal",
    "GenerationRequest",
    "SignalSpec",
    "create_generator",
    "register_generator",
]
