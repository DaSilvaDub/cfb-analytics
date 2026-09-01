"""Filesystem layout.

One module owns every path the package writes to, so nothing scatters
``Path(__file__).parent / ".."`` around the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# Overridable so tests never touch the real store.
_DATA_DIR_ENV = "CFB_DATA_DIR"


def data_dir() -> Path:
    override = os.getenv(_DATA_DIR_ENV)
    return Path(override) if override else PROJECT_ROOT / "data"


def database_path() -> Path:
    return data_dir() / "cfb.sqlite3"


def cache_dir() -> Path:
    return data_dir() / "cache"


def report_dir(slate_date: str) -> Path:
    return data_dir() / "reports" / slate_date


def ensure_dirs() -> None:
    for path in (data_dir(), cache_dir(), ARTIFACT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def outlier_session_dir() -> Path:
    """Where the authenticated Outlier browser session lives.

    This repo shares no code with the ``outlier`` project, but it does read that
    project's saved session rather than duplicating a login flow. The location is
    configurable so the coupling is one path, not an import.
    """
    override = os.getenv("OUTLIER_SESSION_DIR")
    if override:
        return Path(override)
    return Path.home() / "Dev" / "GitHub" / "outlier" / "config" / ".outlier_session"
