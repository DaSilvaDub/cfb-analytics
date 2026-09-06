from __future__ import annotations

import pytest

from cfb_analytics import db
from cfb_analytics.ingest import store


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


# ---------------------------------------------------------------------------
# Canonical (CFBD) seeding helpers.
#
# CFBD is the canonical id space -- see ingest/identity.py. The Outlier ingest
# resolves onto these rows rather than minting its own, so any test exercising
# that ingest has to stand a canonical store up first, exactly as the real
# `daily` job does by bootstrapping the season schedule before the Outlier leg.
# ---------------------------------------------------------------------------


def seed_canonical_team(conn, cfbd_id, school, alias, *, aliases=()):
    """One CFBD team, plus any alternate names it publishes."""
    team_id = f"cfbd:{cfbd_id}"
    store.upsert_team(conn, {
        "team_id": team_id, "cfbd_id": cfbd_id, "school": school,
        "alias": alias, "market": school,
    })
    if aliases:
        store.insert_team_aliases(conn, [
            {"team_id": team_id, "source": "cfbd", "alias": name,
             "alias_type": "alternate_name"}
            for name in aliases
        ])
    return team_id


def seed_canonical_game(conn, game_id, home, away, *,
                        football_date="2026-09-05",
                        kickoff="2026-09-05T23:30:00+00:00"):
    store.upsert_cfbd_game(conn, {
        "game_id": game_id, "season": 2026, "week": 2, "season_type": "regular",
        "kickoff_utc": kickoff, "football_date": football_date,
        "neutral_site": 0, "conference_game": 0,
        "home_team_id": home, "away_team_id": away,
        "venue_name": "Some Stadium", "venue_id": None, "status": "scheduled",
        "home_points": None, "away_points": None, "completed": 0, "source": "cfbd",
    })
    return game_id


@pytest.fixture
def canonical_slate(conn):
    """The canonical rows behind the `schedule_event` / `two_events` fixtures."""
    tulsa = seed_canonical_team(conn, 202, "Tulsa", "TLSA")
    oklahoma_state = seed_canonical_team(conn, 197, "Oklahoma State", "OKST")
    duke = seed_canonical_team(conn, 150, "Duke", "DUKE")
    tulane = seed_canonical_team(conn, 2655, "Tulane", "TULN")
    games = {
        "evt-1": seed_canonical_game(conn, "cfbd:1001", tulsa, oklahoma_state),
        "evt-2": seed_canonical_game(conn, "cfbd:1002", duke, tulane),
    }
    conn.commit()
    return {"games": games, "tulsa": tulsa, "oklahoma_state": oklahoma_state,
            "duke": duke, "tulane": tulane}
