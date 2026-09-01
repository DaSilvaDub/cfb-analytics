"""Read the saved Outlier browser session and build authenticated headers.

This repo does not implement a login flow. It reads the ``storage_state.json``
that the ``outlier`` project's Playwright login already produces, extracts a
bearer token and cookie header from it, and stops there. If the session is
stale the correct fix is to refresh it in that project, not to re-authenticate
from here.

No token or cookie value is ever logged or included in an exception message.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from cfb_analytics import paths
from cfb_analytics.errors import AuthRequiredError

APP_ORIGIN = "https://app.outlier.bet"
TOKEN_MARKERS = (
    "accesstoken",
    "authtoken",
    "idtoken",
    "bearer",
    "authorization",
)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_MIN_TOKEN_LENGTH = 20


def load_storage_state(session_dir: Path | None = None) -> dict[str, Any]:
    directory = session_dir or paths.outlier_session_dir()
    candidate = directory / "storage_state.json"
    if not candidate.exists() or candidate.stat().st_size == 0:
        raise AuthRequiredError(
            f"No saved Outlier session at {candidate}. Refresh it in the outlier project, "
            "or point OUTLIER_SESSION_DIR at the directory holding storage_state.json."
        )
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuthRequiredError(
            f"Saved Outlier session at {candidate} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AuthRequiredError(f"Saved Outlier session at {candidate} is not a JSON object")
    return payload


def _normalise_token(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.lower().startswith("bearer "):
        text = text.split(" ", 1)[1].strip()
    if len(text) < _MIN_TOKEN_LENGTH or any(ch.isspace() for ch in text):
        return None
    return text


def _token_from_value(value: Any) -> str | None:
    """Recursively hunt a token out of a nested/JSON-encoded localStorage value."""
    if isinstance(value, dict):
        for key, inner in value.items():
            compact = re.sub(r"[^a-z0-9]+", "", str(key).lower())
            if any(marker in compact for marker in TOKEN_MARKERS):
                token = _normalise_token(inner)
                if token:
                    return token
            token = _token_from_value(inner)
            if token:
                return token
        return None
    if isinstance(value, list):
        for item in value:
            token = _token_from_value(item)
            if token:
                return token
        return None
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "{[":
            try:
                return _token_from_value(json.loads(text))
            except json.JSONDecodeError:
                return _normalise_token(text)
        return _normalise_token(text)
    return None


def extract_bearer_token(storage_state: dict[str, Any]) -> str | None:
    for origin in storage_state.get("origins", []):
        if not isinstance(origin, dict) or origin.get("origin") != APP_ORIGIN:
            continue
        for item in origin.get("localStorage", []):
            if not isinstance(item, dict):
                continue
            compact = re.sub(r"[^a-z0-9]+", "", str(item.get("name") or "").lower())
            if not any(marker in compact for marker in TOKEN_MARKERS):
                continue
            token = _token_from_value(item.get("value"))
            if token:
                return token
    return None


def build_cookie_header(
    storage_state: dict[str, Any],
    hosts: tuple[str, ...] = ("api.outlier.bet", "app.outlier.bet"),
) -> str:
    now = time.time()
    pairs: list[str] = []
    for cookie in storage_state.get("cookies", []):
        if not isinstance(cookie, dict):
            continue
        name = str(cookie.get("name") or "").strip()
        value = str(cookie.get("value") or "").strip()
        domain = str(cookie.get("domain") or "").strip().lstrip(".").lower()
        if not name or not value or not domain:
            continue
        expires = cookie.get("expires")
        if isinstance(expires, (int, float)) and 0 < expires < now:
            continue
        if not any(
            host == domain or host.endswith(f".{domain}") or domain.endswith(f".{host}")
            for host in hosts
        ):
            continue
        pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def build_headers(storage_state: dict[str, Any]) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
        "Origin": APP_ORIGIN,
        "Referer": f"{APP_ORIGIN}/",
    }
    cookie = build_cookie_header(storage_state)
    if cookie:
        headers["Cookie"] = cookie
    token = extract_bearer_token(storage_state)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Header dict safe to log: credential values replaced by a length marker."""
    safe = {}
    for key, value in headers.items():
        if key.lower() in ("authorization", "cookie"):
            safe[key] = f"<redacted len={len(value)}>"
        else:
            safe[key] = value
    return safe
