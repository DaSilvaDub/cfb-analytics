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
from pathlib import Path
from typing import Any

from cfb_analytics import paths
from cfb_analytics.errors import AuthRequiredError, SourceError, UnknownLeagueError

MODE_ENV_VAR = "CFB_HTTP_MODE"
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# Outlier answers an unrecognised league token with 502 rather than 404, so a
# 502 is ambiguous: it can mean "unknown league" or a genuine upstream blip.
# The Outlier client disambiguates by probing a known-good token; see
# outlier.OutlierClient.fetch_schedule.
_UNKNOWN_LEAGUE_STATUS = frozenset({404, 502})


def current_mode() -> str:
    return (os.getenv(MODE_ENV_VAR) or "live").strip().lower()


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
    cache_root: Path | None = None
    _sleep: Any = time.sleep

    def _cache_path(self, url: str) -> Path:
        root = self.cache_root or paths.cache_dir()
        return root / self.name / f"{cache_key(url)}.json"

    def _read_cache(self, url: str, *, ignore_ttl: bool = False) -> dict[str, Any] | None:
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
        return payload if isinstance(payload, dict) else None

    def _write_cache(self, url: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"url": url, "fetched_at": time.time(), "payload": payload}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(envelope), encoding="utf-8")
        tmp.replace(path)

    def _backoff(self, attempt: int) -> float:
        return min(8.0, 0.5 * (2 ** (attempt - 1))) * (0.5 + random.random() / 2)

    def get_json(self, url: str) -> dict[str, Any]:
        mode = current_mode()

        if mode == "replay":
            cached = self._read_cache(url, ignore_ttl=True)
            if cached is None:
                raise SourceError(
                    f"[{self.name}] replay mode cache miss for {url}. "
                    "Record it in live mode first."
                )
            return cached

        if mode != "refresh":
            cached = self._read_cache(url)
            if cached is not None:
                return cached

        last_detail = ""
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(url, headers=self.headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                if body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise SourceError(f"[{self.name}] expected a JSON object from {url}")
                self._write_cache(url, payload)
                return payload
            except urllib.error.HTTPError as exc:
                last_detail = f"HTTP {exc.code}"
                if exc.code in (401, 403):
                    raise AuthRequiredError(
                        f"[{self.name}] {last_detail} for {url} - the saved session is "
                        "missing, expired, or lacks access to this endpoint."
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
            self._sleep(self._backoff(attempt))

        raise SourceError(f"[{self.name}] failed to fetch {url}: {last_detail}")
