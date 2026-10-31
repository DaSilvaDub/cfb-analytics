"""Cached HTTP GET with retry, backoff, and an offline replay mode.

Every response is written to ``data/cache`` keyed by a hash of the URL. Three
modes:

* ``live``   - fetch, honouring TTL, and write to cache (default)
* ``replay`` - serve only from cache; a miss raises. Used by tests so the whole
  suite runs offline against recorded fixtures.
* ``refresh``- ignore TTL and refetch everything

Error messages never include request headers, so a token cannot leak into a log
or a traceback.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, TypeAlias

from cfb_analytics import paths
from cfb_analytics.errors import AuthRequiredError, SourceError, UnknownLeagueError

MODE_ENV_VAR = "CFB_HTTP_MODE"
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# A 429's Retry-After is waited out, but never longer than this. Three attempts
# therefore cannot sleep for more than two minutes on one URL.
MAX_RETRY_AFTER_SECONDS = 60.0
# Outlier answers an unrecognised league token with 502 rather than 404, so a
# 502 is ambiguous: it can mean "unknown league" or a genuine upstream blip.
# The Outlier client disambiguates by probing a known-good token; see
# outlier.OutlierClient.fetch_schedule.
_UNKNOWN_LEAGUE_STATUS = frozenset({404, 502})
JsonPayload: TypeAlias = dict[str, Any] | list[Any]


def current_mode() -> str:
    return (os.getenv(MODE_ENV_VAR) or "live").strip().lower()


def _header_value(exc: urllib.error.HTTPError, name: str) -> str | None:
    headers = getattr(exc, "headers", None)
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    value = getter(name)
    if value is None:
        value = getter(name.lower())
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def retry_after_seconds(
    exc: urllib.error.HTTPError, *, now: datetime | None = None
) -> float | None:
    """Seconds a 429 asked us to wait, or None when the header is absent or unusable.

    ``Retry-After`` is either a delay in seconds or an HTTP date. A date in the
    past yields a negative number; the caller clamps it.
    """
    raw = _header_value(exc, "Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (parsed - moment.astimezone(UTC)).total_seconds()


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


@dataclass
class HttpClient:
    """Minimal cached GET client. One instance per source."""

    name: str
    timeout_seconds: int = 30
    max_retries: int = 3
    cache_ttl_seconds: int = 300
    headers: dict[str, str] = field(default_factory=dict)
    auth_error_hint: str = "the saved session is missing, expired, or lacks access"
    cache_root: Path | None = None
    _sleep: Any = time.sleep

    def _cache_path(self, url: str) -> Path:
        root = self.cache_root or paths.cache_dir()
        return root / self.name / f"{cache_key(url)}.json"

    def _read_cache(self, url: str, *, ignore_ttl: bool = False) -> JsonPayload | None:
        path = self._cache_path(url)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not ignore_ttl and self.cache_ttl_seconds > 0:
            age = time.time() - float(envelope.get("fetched_at", 0.0))
            if age > self.cache_ttl_seconds:
                return None
        payload = envelope.get("payload")
        return payload if isinstance(payload, (dict, list)) else None

    def _write_cache(self, url: str, payload: JsonPayload) -> None:
        path = self._cache_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"url": url, "fetched_at": time.time(), "payload": payload}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(envelope), encoding="utf-8")
        tmp.replace(path)

    def _backoff(self, attempt: int) -> float:
        return min(8.0, 0.5 * (2 ** (attempt - 1))) * (0.5 + random.random() / 2)

    def _retry_delay(self, attempt: int, exc: urllib.error.HTTPError | None) -> float:
        """Wait at least ``Retry-After`` on HTTP 429, and never longer than the cap.

        Other retryable statuses keep the short exponential backoff. A 429
        without the header does too. When the header is present the delay is
        the larger of that backoff and the header, then clamped to
        ``MAX_RETRY_AFTER_SECONDS``.
        """
        delay = self._backoff(attempt)
        if exc is not None and exc.code == 429:
            retry_after = retry_after_seconds(exc)
            if retry_after is not None:
                delay = max(delay, retry_after)
        return min(MAX_RETRY_AFTER_SECONDS, max(0.0, delay))

    def get_payload(self, url: str, *, ignore_ttl: bool = False) -> JsonPayload:
        """Return a cached JSON object or array.

        Outlier uses object envelopes while CollegeFootballData returns
        top-level arrays. Scalars and null are refused because accepting them
        would turn an upstream schema change into an apparently valid fetch.
        """
        mode = current_mode()

        if mode == "replay":
            cached = self._read_cache(url, ignore_ttl=True)
            if cached is None:
                raise SourceError(
                    f"[{self.name}] replay mode cache miss for {url}. "
                    "Record it in live mode first."
                )
            return cached

        if mode != "refresh" and not ignore_ttl:
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        last_detail = ""
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(url, headers=self.headers)
            retry_exc: urllib.error.HTTPError | None = None
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                if body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, (dict, list)):
                    raise SourceError(
                        f"[{self.name}] expected a JSON object or array from {url}"
                    )
                self._write_cache(url, payload)
                return payload
            except urllib.error.HTTPError as exc:
                last_detail = f"HTTP {exc.code}"
                retry_exc = exc
                if exc.code in (401, 403):
                    raise AuthRequiredError(
                        f"[{self.name}] {last_detail} for {url} - {self.auth_error_hint} "
                        "to this endpoint."
                    ) from exc
                if exc.code in _UNKNOWN_LEAGUE_STATUS and attempt >= self.max_retries:
                    raise UnknownLeagueError(f"[{self.name}] {last_detail} for {url}") from exc
                if exc.code not in RETRYABLE_STATUS or attempt >= self.max_retries:
                    raise SourceError(f"[{self.name}] {last_detail} for {url}") from exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_detail = f"{type(exc).__name__}"
                if attempt >= self.max_retries:
                    raise SourceError(
                        f"[{self.name}] network error for {url}: {last_detail}"
                    ) from exc
            except json.JSONDecodeError as exc:
                raise SourceError(f"[{self.name}] malformed JSON from {url}") from exc
            self._sleep(self._retry_delay(attempt, retry_exc))

        raise SourceError(f"[{self.name}] failed to fetch {url}: {last_detail}")

    def get_json(self, url: str) -> dict[str, Any]:
        payload = self.get_payload(url)
        if not isinstance(payload, dict):
            raise SourceError(f"[{self.name}] expected a JSON object from {url}")
        return payload
