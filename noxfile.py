"""Run formatting, type checks, and tests in reproducible environments.

Run ``uvx nox`` for all checks or select one with ``uvx nox -s lint``.
Dependencies come from ``uv.lock``. The core test session checks the base
installation; the full session also installs optional dependencies.
"""

import tomllib
from pathlib import Path

import nox

nox.needs_version = ">=2026.8.17"
nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "yes"
nox.options.sessions = ["lint", "typecheck", "test_core", "test_all"]

PYTHON = "3.13"
LINTED = ["src", "tests", "trainers", "scripts", "noxfile.py"]


def _locked_version(package: str) -> str:
    """Read a dependency version from the lock file."""
    lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
    return next(
        entry["version"]
        for entry in lock["package"]
        if entry["name"] == package
    )


def _sync(session: nox.Session, *args: str) -> None:
    """Install the project using its locked dependency versions."""
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
    """Check maintained source formatting and documentation style."""
    session.install(f"ruff=={_locked_version('ruff')}")
    session.run("ruff", "check", *LINTED)
    session.run("ruff", "format", "--check", *LINTED)
    session.run("python", "scripts/check_docstrings.py")
    for archive in (
        "experiments/ecgqa_seed0_20261025",
        "experiments/ptbxl_classification_20261026",
    ):
        session.run("python", f"{archive}/reproduce.py", "verify")


@nox.session(python=PYTHON)
def typecheck(session: nox.Session) -> None:
    """Check Python types with optional dependencies installed."""
    _sync(session, "--all-extras")
    session.run("pyright", *session.posargs)


@nox.session(python=PYTHON)
def test_core(session: nox.Session) -> None:
    """Run tests with only the core dependencies installed."""
    _sync(session)
    session.run("pytest", *session.posargs)


@nox.session(python=PYTHON)
def test_all(session: nox.Session) -> None:
    """Run tests with every optional dependency installed."""
    _sync(session, "--all-extras")
    session.run("pytest", *session.posargs)
