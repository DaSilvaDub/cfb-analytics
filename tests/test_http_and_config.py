from __future__ import annotations

import json
import urllib.error

import pytest

from cfb_analytics import config
from cfb_analytics.errors import AuthRequiredError, MissingCredentialError, SourceError
from cfb_analytics.sources.http import MAX_RETRY_AFTER_SECONDS, HttpClient


def _client(tmp_path, **kwargs):
    params = {"name": "test", "cache_root": tmp_path / "cache", "_sleep": lambda _: None}
    params.update(kwargs)
    return HttpClient(**params)


def _seed_cache(client, url, payload, fetched_at=None):
    import time

    path = client._cache_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    # `or` would treat a deliberate 0.0 (epoch = maximally stale) as "unset".
    stamp = time.time() if fetched_at is None else fetched_at
    envelope = {"url": url, "fetched_at": stamp, "payload": payload}
    path.write_text(json.dumps(envelope), encoding="utf-8")


class TestReplayMode:
    def test_serves_from_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "replay")
        client = _client(tmp_path)
        _seed_cache(client, "https://x/y", {"ok": True})
        assert client.get_json("https://x/y") == {"ok": True}

    def test_miss_raises_rather_than_hitting_the_network(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "replay")
        with pytest.raises(SourceError, match="replay mode cache miss"):
            _client(tmp_path).get_json("https://x/never-recorded")

    def test_ignores_ttl(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "replay")
        client = _client(tmp_path, cache_ttl_seconds=1)
        _seed_cache(client, "https://x/y", {"ok": True}, fetched_at=0.0)
        assert client.get_json("https://x/y") == {"ok": True}

    def test_payload_mode_serves_cached_arrays(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "replay")
        client = _client(tmp_path)
        _seed_cache(client, "https://x/list", [{"ok": True}])
        assert client.get_payload("https://x/list") == [{"ok": True}]


class TestLiveMode:
    def test_expired_cache_is_refetched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "live")
        client = _client(tmp_path, cache_ttl_seconds=1)
        _seed_cache(client, "https://x/y", {"stale": True}, fetched_at=0.0)

        calls = []

        class FakeResponse:
            def read(self):
                return json.dumps({"fresh": True}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        assert client.get_json("https://x/y") == {"fresh": True}
        assert len(calls) == 1

    def test_retries_then_succeeds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "live")
        client = _client(tmp_path, max_retries=3)
        attempts = []

        class FakeResponse:
            def read(self):
                return b'{"ok": true}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def flaky(req, timeout=None):
            attempts.append(1)
            if len(attempts) < 3:
                raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", flaky)
        assert client.get_json("https://x/y") == {"ok": True}
        assert len(attempts) == 3

    def test_401_raises_auth_error_without_retrying(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "live")
        client = _client(tmp_path, max_retries=3)
        attempts = []

        def unauthorised(req, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError(req.full_url, 401, "no", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", unauthorised)
        with pytest.raises(AuthRequiredError, match="session is"):
            client.get_json("https://x/y")
        assert len(attempts) == 1, "auth failures must not be retried"

    def test_error_message_never_contains_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "live")
        secret = "supersecrettokenvalue1234567890"
        client = _client(tmp_path, max_retries=1)
        client.headers = {"Authorization": f"Bearer {secret}", "Cookie": f"sid={secret}"}

        def boom(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 500, "err", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with pytest.raises(SourceError) as excinfo:
            client.get_json("https://x/y")
        assert secret not in str(excinfo.value)

    def test_429_retries_are_bounded_and_honor_retry_after(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "live")
        sleeps: list[float] = []
        client = _client(tmp_path, max_retries=3, _sleep=sleeps.append)
        attempts: list[int] = []

        def limited(req, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError(
                req.full_url, 429, "rate", {"Retry-After": "30"}, None
            )

        monkeypatch.setattr("urllib.request.urlopen", limited)
        with pytest.raises(SourceError, match="HTTP 429"):
            client.get_json("https://x/y")
        assert len(attempts) == 3
        assert sleeps == [30.0, 30.0]

    def test_429_retry_after_is_capped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "live")
        sleeps: list[float] = []
        client = _client(tmp_path, max_retries=3, _sleep=sleeps.append)

        def limited(req, timeout=None):
            raise urllib.error.HTTPError(
                req.full_url, 429, "rate", {"Retry-After": "3600"}, None
            )

        monkeypatch.setattr("urllib.request.urlopen", limited)
        with pytest.raises(SourceError, match="HTTP 429"):
            client.get_json("https://x/y")
        assert sleeps == [MAX_RETRY_AFTER_SECONDS, MAX_RETRY_AFTER_SECONDS]
        assert all(delay <= MAX_RETRY_AFTER_SECONDS for delay in sleeps)

    def test_429_succeeds_after_one_retry_after_wait(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "live")
        sleeps: list[float] = []
        client = _client(tmp_path, max_retries=3, _sleep=sleeps.append)
        attempts: list[int] = []

        class FakeResponse:
            def read(self):
                return b'{"ok": true}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def flaky(req, timeout=None):
            attempts.append(1)
            if len(attempts) == 1:
                raise urllib.error.HTTPError(
                    req.full_url, 429, "rate", {"Retry-After": "12"}, None
                )
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", flaky)
        assert client.get_json("https://x/y") == {"ok": True}
        assert attempts == [1, 1]
        assert sleeps == [12.0]

    def test_429_without_retry_after_uses_the_short_backoff(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CFB_HTTP_MODE", "live")
        sleeps: list[float] = []
        client = _client(tmp_path, max_retries=3, _sleep=sleeps.append)
        attempts: list[int] = []

        def limited(req, timeout=None):
            attempts.append(1)
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {}, None)

        monkeypatch.setattr("urllib.request.urlopen", limited)
        with pytest.raises(SourceError, match="HTTP 429"):
            client.get_json("https://x/y")
        assert len(attempts) == 3
        assert len(sleeps) == 2
        assert all(0 <= delay <= 8.0 for delay in sleeps)

    def test_retry_after_http_date_is_seconds_from_now(self):
        from datetime import UTC, datetime, timedelta
        from email.utils import format_datetime

        from cfb_analytics.sources.http import retry_after_seconds

        moment = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
        stamp = format_datetime(moment + timedelta(seconds=45), usegmt=True)
        exc = urllib.error.HTTPError(
            "https://x/y", 429, "rate", {"Retry-After": stamp}, None
        )
        assert retry_after_seconds(exc, now=moment) == pytest.approx(45.0)


class TestConfig:
    def test_missing_cfbd_key_names_the_variable(self, monkeypatch):
        with pytest.raises(MissingCredentialError) as excinfo:
            config.cfbd_api_key()
        message = str(excinfo.value)
        assert "CFBD_API_KEY" in message
        assert "collegefootballdata.com/key" in message

    def test_has_cfbd_key_is_false_when_unset(self, monkeypatch):
        assert config.has_cfbd_key() is False

    def test_shadow_mode_is_the_default(self):
        assert config.is_shadow_mode() is True

    def test_settings_load(self):
        assert config.settings()["parlay"]["max_legs"] == 10

    def test_market_weight_floor_keeps_model_side_capped(self):
        floor = config.settings()["blend"]["market_weight_floor"]
        assert 0.0 < 1.0 - floor <= 0.25
