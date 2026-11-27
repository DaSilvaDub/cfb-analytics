"""Loaders must not read a row from at or after kickoff.

On master, a TBD midnight placeholder dropped an afternoon quote, the
situational spread took the newest row (a post-kickoff -10.5), weather took
the newest row and turned a null temperature into 20C, and the mispriced
margin used the latest Elo fit even when that fit was after kickoff.
"""

from __future__ import annotations

from cfb_analytics.features.asof import reader_for_game
from cfb_analytics.features.build_market import build_market_for_slate
from cfb_analytics.features.mispriced import _load_model_margins
from cfb_analytics.ingest import store
from cfb_analytics.reasoning.context import load_situational_context_for_game
from cfb_analytics.sources.outlier import OddsRow

KICKOFF = "2026-09-05T23:30:00+00:00"
SLATE = "2026-09-05"
PRE = "2026-09-05T16:00:00+00:00"
AT_KICK = KICKOFF
AFTER = "2026-09-06T01:00:00+00:00"
# Midnight ET is 05:00 UTC. 14:00 ET is 19:00 UTC, after that placeholder.
PLACEHOLDER = "2026-11-07T05:00:00+00:00"
AFTERNOON = "2026-11-07T19:00:00+00:00"
TBD_SLATE = "2026-11-07"


def _teams(conn) -> tuple[str, str]:
    store.upsert_team(conn, {"team_id": "h", "school": "Home U", "alias": "HOME", "market": "H"})
    store.upsert_team(conn, {"team_id": "a", "school": "Away U", "alias": "AWAY", "market": "A"})
    return "h", "a"


def _game(conn, *, kickoff: str = KICKOFF, slate: str = SLATE, game_id: str = "g1") -> str:
    store.upsert_game(
        conn,
        {
            "game_id": game_id,
            "season": 2026,
            "kickoff_utc": kickoff,
            "football_date": slate,
            "day_of_week": 5,
            "home_team_id": "h",
            "away_team_id": "a",
            "venue_name": None,
            "network": None,
            "status": "pregame",
        },
    )
    return game_id


def _spread(conn, line: float, as_of: str, *, n_books: int, game_id: str = "g1") -> None:
    for side, signed in (("HOME", line), ("AWAY", -line)):
        conn.execute(
            """INSERT INTO market_consensus
               (game_id, market, line, side, as_of_utc, n_books, consensus_price,
                best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
                prob_power, prob_spread, flags)
               VALUES (?, 'SPREAD', ?, ?, ?, ?, -110, -110, 'DK', 0.04, 'all_books',
                       0.5, 0.5, 0.5, 0.0, '[]')""",
            (game_id, signed, side, as_of, n_books),
        )


def _weather(conn, as_of: str, temp_c: float | None, *, wind_kph: float = 0.0) -> None:
    conn.execute(
        """INSERT INTO weather
           (game_id, as_of_utc, hours_to_kick, temp_c, wind_kph, wind_gust_kph,
            precip_mm, precip_prob, humidity, is_forecast, is_indoor, source)
           VALUES ('g1', ?, 1, ?, ?, ?, 0, 0, 50, 1, 0, 'open-meteo')""",
        (as_of, temp_c, wind_kph, wind_kph),
    )


def _elo(conn, as_of: str, home_rating: float, away_rating: float) -> None:
    for team, rating in (("h", home_rating), ("a", away_rating)):
        conn.execute(
            """INSERT INTO internal_elo_ratings
               (season, as_of_utc, team_id, model, rating, team_games,
                n_games_in_fit, generated_utc)
               VALUES (2026, ?, ?, 'internal_elo', ?, 4, 40, ?)""",
            (as_of, team, rating, as_of),
        )


def _odds(book: str, side: str, captured: str) -> OddsRow:
    return OddsRow(
        game_id="g1",
        market_id="m",
        book=book,
        market="ML",
        side=side,
        line=0.0,
        price_american=-150 if side == "HOME" else 130,
        price_decimal=None,
        is_primary=True,
        captured_utc=captured,
    )


def test_tbd_placeholder_does_not_drop_an_afternoon_quote() -> None:
    """reader_for_game must not treat a TBD midnight clock as kickoff."""
    reader = reader_for_game(
        {"game_id": "g1", "kickoff_utc": PLACEHOLDER, "kickoff_tbd": 1, "season": 2026}
    )
    kept = reader.admissible(
        [{"captured_utc": AFTERNOON, "tag": "14:00 ET"}],
        what="odds_snapshots",
        as_of_field="captured_utc",
    )
    assert [row["tag"] for row in kept] == ["14:00 ET"]


def test_a_real_midnight_kickoff_still_drops_the_afternoon_quote() -> None:
    reader = reader_for_game(
        {"game_id": "g1", "kickoff_utc": PLACEHOLDER, "kickoff_tbd": 0, "season": 2026}
    )
    kept = reader.admissible(
        [{"captured_utc": AFTERNOON}],
        what="odds_snapshots",
        as_of_field="captured_utc",
    )
    assert kept == []


def test_market_build_keeps_the_afternoon_quote_when_kickoff_is_tbd(conn) -> None:
    _teams(conn)
    _game(conn, kickoff=PLACEHOLDER, slate=TBD_SLATE)
    conn.execute("UPDATE games SET kickoff_tbd = 1 WHERE game_id = 'g1'")
    store.insert_odds(
        conn,
        [
            _odds(book, side, AFTERNOON)
            for book in ("A", "B", "C")
            for side in ("HOME", "AWAY")
        ],
    )
    summary = build_market_for_slate(conn, TBD_SLATE)
    assert summary.leaked_rows_dropped == 0
    stamps = {
        row["as_of_utc"]
        for row in conn.execute("SELECT as_of_utc FROM market_consensus")
    }
    assert stamps == {AFTERNOON}


def test_situational_spread_is_the_pre_kickoff_primary_line(conn) -> None:
    """Post-kickoff -10.5 must not replace the last pre-kickoff primary, -3.5.

    An older line with more books is not the current primary either.
    """
    _teams(conn)
    _game(conn)
    _spread(conn, -7.5, "2026-09-04T12:00:00+00:00", n_books=20)
    _spread(conn, -3.5, PRE, n_books=4)
    _spread(conn, -10.5, AFTER, n_books=30)
    ctx = load_situational_context_for_game(conn, "g1", SLATE)
    assert ctx.market_spread_home == -3.5


def test_weather_ignores_rows_at_or_after_kickoff(conn) -> None:
    _teams(conn)
    _game(conn)
    _weather(conn, PRE, 2.0)
    _weather(conn, AT_KICK, 28.0)
    _weather(conn, AFTER, 35.0)
    ctx = load_situational_context_for_game(conn, "g1", SLATE)
    assert ctx.weather.temperature_c == 2.0


def test_null_temperature_stays_unknown(conn) -> None:
    _teams(conn)
    _game(conn)
    _weather(conn, PRE, None, wind_kph=40.0)
    ctx = load_situational_context_for_game(conn, "g1", SLATE)
    assert ctx.weather.temperature_c is None
    assert ctx.weather.wind_kph == 40.0


def test_margin_uses_the_elo_fit_strictly_before_each_kickoff(conn) -> None:
    """A fit at or after kickoff is not the pregame margin.

    The 22:00 fit is before the late game and after the early one.
    """
    _teams(conn)
    early = "2026-09-05T20:00:00+00:00"
    late = KICKOFF
    _game(conn, kickoff=early, game_id="early")
    _game(conn, kickoff=late, game_id="late")
    _elo(conn, "2026-09-04T00:00:00+00:00", 1500.0, 1700.0)
    _elo(conn, "2026-09-05T22:00:00+00:00", 1800.0, 1500.0)
    _elo(conn, AT_KICK, 1900.0, 1400.0)
    _elo(conn, AFTER, 2000.0, 1300.0)
    margins = _load_model_margins(conn, SLATE)
    assert margins["early"] == -5.5
    assert margins["late"] == 14.5
