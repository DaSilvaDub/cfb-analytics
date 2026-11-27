"""Daily lines follow the schedule refresh, and January stays on the prior season.

Reproduced on master 9fa4204. A kickoff that moves to a new Eastern date is
still stored on the old date when lines are matched, so the game is not priced.
A January run passes the new calendar year into every season-scoped leg.
"""

from __future__ import annotations

from datetime import UTC, datetime

from cfb_analytics import daily
from cfb_analytics.utils import FOOTBALL_TZ
from tests.conftest import seed_canonical_game, seed_canonical_team

OLD_DATE = "2026-09-12"
NEW_KICKOFF = "2026-09-19T23:30:00.000Z"
NEW_DATE = "2026-09-19"
NOW = datetime(2026, 9, 18, 12, tzinfo=FOOTBALL_TZ)


class _MarketSummary:
    consensus_rows = 0
    movement_rows = 0
    games = 0


def _silence_other_legs(monkeypatch) -> None:
    def _quiet(*_args, **_kwargs) -> None:
        return None

    for name in (
        "_run_cfbd_rankings",
        "_run_cfbd_elo",
        "_run_team_boxes",
        "_run_cfbd_advanced",
        "_run_team_advanced_guard",
        "_freeze_over_boards",
        "_run_slate_reports",
    ):
        monkeypatch.setattr(daily, name, _quiet)
    monkeypatch.setattr(
        "cfb_analytics.features.build_market.build_market_for_slate",
        lambda *_args, **_kwargs: _MarketSummary(),
    )


def test_rescheduled_game_is_priced_after_the_kickoff_refresh(conn, monkeypatch) -> None:
    home = seed_canonical_team(conn, 202, "Tulsa", "TLSA")
    away = seed_canonical_team(conn, 197, "Oklahoma State", "OKST")
    seed_canonical_game(
        conn,
        "cfbd:900",
        home,
        away,
        football_date=OLD_DATE,
        kickoff="2026-09-12T23:30:00+00:00",
        week=3,
    )
    seen: dict[str, str] = {}

    class Client:
        def fetch_games(self, season, *, season_type="both", classification="fbs", refresh=False):
            return [
                {
                    "id": 900,
                    "season": 2026,
                    "week": 3,
                    "seasonType": "regular",
                    "startDate": NEW_KICKOFF,
                    "startTimeTBD": False,
                    "neutralSite": False,
                    "conferenceGame": False,
                    "completed": False,
                    "homeId": 202,
                    "homeTeam": "Tulsa",
                    "awayId": 197,
                    "awayTeam": "Oklahoma State",
                }
            ]

        def fetch_lines(self, year, *, week=None, season_type="regular"):
            row = conn.execute(
                "SELECT football_date FROM games WHERE game_id = 'cfbd:900'"
            ).fetchone()
            seen["date"] = str(row["football_date"])
            return [
                {
                    "id": 900,
                    "season": year,
                    "week": 3,
                    "startDate": NEW_KICKOFF,
                    "homeTeam": "Tulsa",
                    "awayTeam": "Oklahoma State",
                    "lines": [
                        {
                            "provider": "Bovada",
                            "spread": -6.5,
                            "spreadOpen": -3.0,
                            "overUnder": 55.5,
                            "overUnderOpen": 52.5,
                            "homeMoneyline": -260,
                            "awayMoneyline": 210,
                        }
                    ],
                }
            ]

    monkeypatch.setattr(daily.config, "has_cfbd_key", lambda: True)
    monkeypatch.setattr("cfb_analytics.sources.cfbd.CFBDClient", Client)
    _silence_other_legs(monkeypatch)

    daily.run_daily(
        conn,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        now=NOW,
    )

    assert seen["date"] == NEW_DATE
    priced = conn.execute(
        "SELECT COUNT(*) AS n FROM odds_snapshots WHERE game_id = 'cfbd:900'"
    ).fetchone()
    assert priced["n"] > 0


def _seasons_passed(conn, monkeypatch, now: datetime, *, season: int | None = None) -> set[int]:
    seen: list[int] = []

    def _record(*args, **_kwargs) -> None:
        seen.append(int(args[2]))

    for name in (
        "_run_cfbd_lines",
        "_run_cfbd_rankings",
        "_run_cfbd_elo",
        "_run_refresh_scores",
        "_run_team_boxes",
        "_run_cfbd_advanced",
    ):
        monkeypatch.setattr(daily, name, _record)

    daily.run_daily(
        conn,
        season=season,
        with_outlier=False,
        with_weather=False,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        bootstrap=False,
        now=now,
    )
    return set(seen)


def test_january_run_uses_the_season_that_started_in_august(conn, monkeypatch) -> None:
    seasons = _seasons_passed(conn, monkeypatch, datetime(2027, 1, 11, 7, tzinfo=FOOTBALL_TZ))
    assert seasons == {2026}


def test_new_year_utc_still_inside_december_eastern_keeps_that_season(conn, monkeypatch) -> None:
    # 2027-01-01 04:30 UTC is 2026-12-31 23:30 ET.
    seasons = _seasons_passed(conn, monkeypatch, datetime(2027, 1, 1, 4, 30, tzinfo=UTC))
    assert seasons == {2026}


def test_august_run_uses_the_new_season(conn, monkeypatch) -> None:
    seasons = _seasons_passed(conn, monkeypatch, datetime(2026, 8, 1, 0, tzinfo=FOOTBALL_TZ))
    assert seasons == {2026}


def test_july_run_stays_on_the_season_that_started_last_august(conn, monkeypatch) -> None:
    seasons = _seasons_passed(conn, monkeypatch, datetime(2026, 7, 31, 23, tzinfo=FOOTBALL_TZ))
    assert seasons == {2025}


def test_august_utc_that_is_still_july_eastern_stays_on_the_prior_season(
    conn, monkeypatch
) -> None:
    # 2026-08-01 03:30 UTC is 2026-07-31 23:30 ET.
    seasons = _seasons_passed(conn, monkeypatch, datetime(2026, 8, 1, 3, 30, tzinfo=UTC))
    assert seasons == {2025}


def test_explicit_season_overrides_the_calendar(conn, monkeypatch) -> None:
    seasons = _seasons_passed(
        conn,
        monkeypatch,
        datetime(2027, 1, 11, 7, tzinfo=FOOTBALL_TZ),
        season=2024,
    )
    assert seasons == {2024}
