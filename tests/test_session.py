from __future__ import annotations

import json
import time

import pytest

from cfb_analytics.errors import AuthRequiredError
from cfb_analytics.sources import session

TOKEN = "abcdefghijklmnopqrstuvwxyz0123456789"


def _state(local_storage=None, cookies=None):
    return {
        "origins": [{"origin": session.APP_ORIGIN, "localStorage": local_storage or []}],
        "cookies": cookies or [],
    }


class TestExtractBearerToken:
    def test_finds_a_plain_token(self):
        state = _state([{"name": "access_token", "value": TOKEN}])
        assert session.extract_bearer_token(state) == TOKEN

    def test_finds_a_token_nested_in_json_encoded_value(self):
        state = _state([{"name": "sb-auth-token", "value": json.dumps({"accessToken": TOKEN})}])
        assert session.extract_bearer_token(state) == TOKEN

    def test_strips_bearer_prefix(self):
        state = _state([{"name": "authorization", "value": f"Bearer {TOKEN}"}])
        assert session.extract_bearer_token(state) == TOKEN

    def test_rejects_short_values(self):
        state = _state([{"name": "access_token", "value": "abc"}])
        assert session.extract_bearer_token(state) is None

    def test_rejects_values_containing_whitespace(self):
        state = _state([{"name": "access_token", "value": "abcdefghij klmnopqrstuvwxyz0123"}])
        assert session.extract_bearer_token(state) is None

    def test_ignores_other_origins(self):
        state = {"origins": [{"origin": "https://evil.example",
                              "localStorage": [{"name": "access_token", "value": TOKEN}]}]}
        assert session.extract_bearer_token(state) is None

    def test_returns_none_when_absent(self):
        assert session.extract_bearer_token(_state([{"name": "theme", "value": "dark"}])) is None


class TestCookieHeader:
    def test_includes_matching_domain(self):
        state = _state(cookies=[{"name": "sid", "value": "1", "domain": ".outlier.bet"}])
        assert session.build_cookie_header(state) == "sid=1"

    def test_excludes_foreign_domain(self):
        state = _state(cookies=[{"name": "sid", "value": "1", "domain": "example.com"}])
        assert session.build_cookie_header(state) == ""

    def test_excludes_expired_cookie(self):
        state = _state(cookies=[
            {"name": "old", "value": "1", "domain": "api.outlier.bet",
             "expires": time.time() - 60},
            {"name": "new", "value": "2", "domain": "api.outlier.bet",
             "expires": time.time() + 600},
        ])
        assert session.build_cookie_header(state) == "new=2"

    def test_keeps_session_cookie_with_negative_expiry(self):
        state = _state(cookies=[
            {"name": "s", "value": "1", "domain": "api.outlier.bet", "expires": -1}])
        assert session.build_cookie_header(state) == "s=1"


class TestBuildHeaders:
    def test_sets_authorization_when_token_present(self):
        headers = session.build_headers(_state([{"name": "access_token", "value": TOKEN}]))
        assert headers["Authorization"] == f"Bearer {TOKEN}"

    def test_omits_authorization_when_token_absent(self):
        assert "Authorization" not in session.build_headers(_state())

    def test_redaction_hides_credentials(self):
        headers = session.build_headers(_state(
            [{"name": "access_token", "value": TOKEN}],
            [{"name": "sid", "value": "secret", "domain": "api.outlier.bet"}],
        ))
        safe = session.redact_headers(headers)
        assert TOKEN not in json_dump(safe)
        assert "secret" not in json_dump(safe)
        assert safe["Origin"] == session.APP_ORIGIN


class TestLoadStorageState:
    def test_missing_file_raises_actionable_error(self, tmp_path):
        with pytest.raises(AuthRequiredError, match="storage_state.json"):
            session.load_storage_state(tmp_path)

    def test_malformed_json_raises(self, tmp_path):
        (tmp_path / "storage_state.json").write_text("{nope", encoding="utf-8")
        with pytest.raises(AuthRequiredError, match="not valid JSON"):
            session.load_storage_state(tmp_path)

    def test_reads_valid_state(self, tmp_path):
        (tmp_path / "storage_state.json").write_text(json.dumps(_state()), encoding="utf-8")
        assert "origins" in session.load_storage_state(tmp_path)


def json_dump(value) -> str:
    return json.dumps(value)
