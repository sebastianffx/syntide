"""Every check CI runs, as one command: ``uvx nox`` runs all of them, ``uvx nox -s <session>`` one.

Each session has its own environment under ``.nox/``, built from ``uv.lock``:

- ``lint``: ``ruff check`` and ``ruff format --check``, with the ruff version locked in ``uv.lock``.
- ``typecheck``: pyright on the paths ``[tool.pyright]`` includes, with every extra installed so that
  optional imports (wandb, diffusers, ...) resolve.
- ``test_core``: pytest on the core install only (plain ``uv sync``). The core artifact experiment must
  not need an extra; tests that need one skip themselves.
- ``test_all``: pytest with every extra installed.

Arguments after ``--`` go to pytest or pyright: ``uvx nox -s test_all -- tests/eval -x``.
"""

import tomllib
from pathlib import Path

import nox

nox.needs_version = ">=2026.8.17"
nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "yes"
nox.options.sessions = ["lint", "typecheck", "test_core", "test_all"]

PYTHON = "3.13"
LINTED = ["src", "tests", "trainers", "noxfile.py"]


def _locked_version(package: str) -> str:
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    return next(entry["version"] for entry in lock["package"] if entry["name"] == package)


def _sync(session: nox.Session, *args: str) -> None:
    """Install the locked project into the session's environment. ``--locked`` fails if ``uv.lock`` no
    longer matches ``pyproject.toml``."""
    session.run_install(
        "uv",
        "sync",
        "--locked",
        *args,
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
        external=True,
    )


@nox.session(python=PYTHON)
def lint(session: nox.Session) -> None:
    session.install(f"ruff=={_locked_version('ruff')}")
    session.run("ruff", "check", *LINTED)
    session.run("ruff", "format", "--check", *LINTED)


@nox.session(python=PYTHON)
def typecheck(session: nox.Session) -> None:
    _sync(session, "--all-extras")
    session.run("pyright", *session.posargs)


@nox.session(python=PYTHON)
def test_core(session: nox.Session) -> None:
    _sync(session)
    session.run("pytest", *session.posargs)


@nox.session(python=PYTHON)
def test_all(session: nox.Session) -> None:
    _sync(session, "--all-extras")
    session.run("pytest", *session.posargs)
