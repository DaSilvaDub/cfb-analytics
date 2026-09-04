from __future__ import annotations

import pytest

from cfb_analytics import db


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Never touch the real store or cache during tests.

    Credentials are blanked too. Several tests previously passed only because
    the developer happened to have no CFBD key; once a key existed in `.env`
    they started failing. Setting the variable to empty (rather than deleting
    it) also stops `config.load_env` re-reading it out of `.env`, so the suite
    behaves identically on every machine and in CI.
    """
    monkeypatch.setenv("CFB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CFB_HTTP_MODE", "replay")
    monkeypatch.setenv("CFBD_API_KEY", "")
    yield tmp_path


@pytest.fixture
def conn(tmp_path):
    with db.open_db(tmp_path / "test.sqlite3") as connection:
        yield connection


@pytest.fixture
def moneyline_market():
    """A GAMELINE MONEYLINE market shaped like the live payload.

    The important detail is deliberate: ``outcomes[].books`` is in a DIFFERENT
    order from ``outcomes[].odds``, exactly as the live API returns it.
    """
    return {
        "marketId": "mkt-1",
        "proposition": "MONEYLINE",
        "books": ["FLIFF", "FANATICS", "BETRIVERS"],
        "outcomes": [
            {
                "position": "AWAY",
                "label": "OKST",
                "line": 0.0,
                "primary": True,
                "books": ["FLIFF", "FANATICS", "BETRIVERS"],
                "odds": [
                    {"american": "-260", "decimal": 1.3846153846153846, "book": "FANATICS"},
                    {"american": "-265", "decimal": 1.38, "book": "BETRIVERS"},
                    {"american": "-250", "decimal": 1.40, "book": "FLIFF"},
                ],
            },
            {
                "position": "HOME",
                "label": "TLSA",
                "line": 0.0,
                "primary": True,
                "books": ["FLIFF", "FANATICS", "BETRIVERS"],
                "odds": [
                    {"american": "+190", "decimal": 2.9, "book": "FANATICS"},
                    {"american": "+190", "decimal": 2.9, "book": "BETRIVERS"},
                ],
            },
        ],
    }


@pytest.fixture
def schedule_event():
    return {
        "eventId": "evt-1",
        # The live feed emits UTC with a "+00:00" suffix and a numeric dayOfWeek
        # (Mon=0 ... Sun=6). 23:30Z is 7:30pm ET Saturday.
        "scheduledTime": "2026-09-05T23:30:00+00:00",
        "dayOfWeek": 5,
        "season": 2026,
        "status": "pregame",
        "network": "ESPN",
        "venue": "Some Stadium",
        "home": {"teamId": "t-home", "name": "Tulsa", "alias": "TLSA", "market": "Tulsa"},
        "away": {"teamId": "t-away", "name": "Oklahoma State", "alias": "OKST",
                 "market": "Oklahoma State"},
    }
