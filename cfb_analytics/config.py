"""Settings and credential access.

Credentials are read from the environment only. No value is ever logged,
written to an artifact, or included in an exception message.
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

from cfb_analytics import paths
from cfb_analytics.errors import ConfigError, MissingCredentialError

CFBD_ENV_VAR = "CFBD_API_KEY"
CFBD_PURPOSE = "CollegeFootballData API: fundamentals, ratings, and historical backfill"
CFBD_HOW = "Get a free key at https://collegefootballdata.com/key"


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overwriting real env vars.

    Deliberately minimal: no interpolation, no export syntax, no quoting rules
    beyond stripping one matching pair. Anything more belongs in a real env
    manager, not here.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def load_env() -> None:
    _load_dotenv(paths.PROJECT_ROOT / ".env")


def cfbd_api_key(*, required: bool = True) -> str | None:
    """Return the CFBD key, or raise a message naming the variable to set."""
    load_env()
    key = (os.getenv(CFBD_ENV_VAR) or "").strip()
    if key:
        return key
    if required:
        raise MissingCredentialError(CFBD_ENV_VAR, CFBD_PURPOSE, CFBD_HOW)
    return None


def has_cfbd_key() -> bool:
    return cfbd_api_key(required=False) is not None


@cache
def _read_json(name: str) -> dict[str, Any]:
    path = paths.CONFIG_DIR / name
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Expected a JSON object in {path}")
    return payload


def settings() -> dict[str, Any]:
    return _read_json("settings.json")


def sources() -> dict[str, Any]:
    return _read_json("sources.json")


def promotion() -> dict[str, Any]:
    return _read_json("promotion.json")


def is_shadow_mode() -> bool:
    """True until the backtest clears the promotion gate.

    While shadow, no CORE tier is emitted and every artifact is stamped
    UNPROMOTED. Defaults to True on any doubt — a missing or unreadable
    promotion file must never be interpreted as 'promoted'.
    """
    try:
        return str(promotion().get("status", "shadow")).lower() != "promoted"
    except ConfigError:
        return True


SHADOW_STAMP = "UNPROMOTED - shadow output, not decision-grade"
