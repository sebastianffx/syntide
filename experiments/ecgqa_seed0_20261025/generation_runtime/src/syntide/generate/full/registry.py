"""Versioned generator factories; importing this module never loads model libraries."""

import re
from collections.abc import Callable
from importlib.metadata import entry_points

from syntide.generate.full.api import GeneratorBackend

ENTRY_POINT_GROUP = "syntide.full_generators"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*@[A-Za-z0-9][A-Za-z0-9_.+-]*")
_REGISTERED: dict[str, Callable[[dict], GeneratorBackend]] = {}


def _diffusets(config: dict) -> GeneratorBackend:
    from syntide.generate.full.diffusets import DiffuSETSBackend

    return DiffuSETSBackend(config)


_BUILTINS = {"diffusets@1": _diffusets}


def register_generator(name: str, factory: Callable[[dict], GeneratorBackend]) -> None:
    """Register a local factory explicitly, without executing code from configuration files."""
    if not isinstance(name, str) or _ID.fullmatch(name) is None:
        raise ValueError("Generator IDs must be versioned, for example diffusets@1")
    if name in _BUILTINS or name in _REGISTERED:
        raise ValueError(f"Generator {name!r} is already registered")
    if not callable(factory):
        raise ValueError("Generator factory must be callable")
    _REGISTERED[name] = factory


def available_generators() -> list[str]:
    return sorted({*_BUILTINS, *_REGISTERED, *(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))})


def create_generator(name: str, config: dict) -> GeneratorBackend:
    if not isinstance(name, str) or _ID.fullmatch(name) is None:
        raise ValueError("Generator IDs must be versioned, for example diffusets@1")
    candidates = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == name]
    factory = _BUILTINS.get(name) or _REGISTERED.get(name)
    if len(candidates) > 1 or (factory is not None and candidates):
        raise ValueError(f"Multiple installed factories claim generator {name!r}")
    if factory is None and candidates:
        factory = candidates[0].load()
    if factory is None:
        raise ValueError(f"Unknown generator {name!r}; available: {', '.join(available_generators())}")
    backend = factory(config)
    for method in ("validate_request", "generate", "provenance"):
        if not callable(getattr(backend, method, None)):
            raise TypeError(f"Generator {name!r} does not implement {method}()")
    return backend
