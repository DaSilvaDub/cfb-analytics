"""Tests for the CFBD fundamentals backfill, focused on the weekly-Elo path.

This module had zero test coverage before this file: the entire weekly-Elo
write path silently inserted nothing for its whole existence (a CHECK
constraint on ``team_ratings`` rejected every row via ``INSERT OR IGNORE``,
with no error surfaced anywhere) and nothing caught it. See
``parse_elo_rating``'s docstring in ``sources/cfbd.py`` and
``insert_team_ratings``'s in ``ingest/store.py`` for the two-part fix.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_fundamentals import backfill_elo
from cfb_analytics.sources.cfbd import parse_elo_rating, parse_sp_rating
from cfb_analytics.utils import FOOTBALL_TZ


def _team_row(cfbd_id, school):
    return {
        "id": cfbd_id,
        "school": school,
        "mascot": "Mascots",
        "abbreviation": school[:3].upper(),
        "alternateNames": [school],
        "conference": "Test Conference",
        "division": None,
        "classification": "fbs",
        "location": None,
    }


class FakeCfbdClient:
    def __init__(self, *, teams, elo_by_week):
        self.teams = teams
        self.elo_by_week = elo_by_week

    def fetch_fbs_teams(self, year: int):
        return self.teams

    def fetch_elo(self, year: int, week: int, *, season_type: str = "regular"):
        return self.elo_by_week.get(week, [])


class TestParseEloRating:
    def test_period_embeds_season_type_and_week(self):
        row = parse_elo_rating(
            {"year": 2023, "team": "Georgia", "elo": 2100},
            week=3, as_of_utc="2023-09-15T00:00:00+00:00", season_type="regular",
        )
        assert row["period"] == "regular:week:03"
        assert row["season_type"] == "regular"
        assert row["snapshot_scope"] == "weekly"

    def test_defaults_season_type_to_regular(self):
        row = parse_elo_rating(
            {"year": 2023, "team": "Georgia", "elo": 2100},
            week=3, as_of_utc="2023-09-15T00:00:00+00:00",
        )
        assert row["season_type"] == "regular"


class TestInsertTeamRatingsWeeklyRow:
    """Regression guard for the exact bug found: a weekly row must actually
    land in the table, not vanish under INSERT OR IGNORE."""

    def test_a_weekly_elo_row_is_actually_persisted(self, conn):
        store.upsert_team(conn, {"team_id": "cfbd:1", "school": "Georgia",
                                  "alias": None, "market": None})
        row = parse_elo_rating(
            {"year": 2023, "team": "Georgia", "elo": 2100},
            week=3, as_of_utc="2023-09-15T00:00:00+00:00",
        )
        row = {**row, "team_id": "cfbd:1", "provenance_mode": "reconstructed"}
        del row["team_name"]

        written = store.insert_team_ratings(conn, [row])
        assert written == 1

        stored = conn.execute(
            "SELECT * FROM team_ratings WHERE source = 'elo_cfbd'"
        ).fetchone()
        assert stored is not None
        assert stored["period"] == "regular:week:03"
        assert stored["season_type"] == "regular"
        assert stored["rating"] == 2100

    def test_a_season_final_row_is_unaffected_by_the_season_type_column(self, conn):
        """SP+/SRS rows never set season_type; adding the column to the
        insert must not change their (already-correct) behavior."""
        store.upsert_team(conn, {"team_id": "cfbd:1", "school": "Georgia",
                                  "alias": None, "market": None})
        row = parse_sp_rating(
            {"year": 2023, "team": "Georgia", "rating": 31.2, "offense": {"rating": 41.5},
             "defense": {"rating": 12.2}, "specialTeams": {"rating": 1.9}},
            as_of_utc="2024-01-10T00:00:00+00:00",
        )
        row = {**row, "team_id": "cfbd:1", "provenance_mode": "reconstructed"}
        del row["team_name"]

        written = store.insert_team_ratings(conn, [row])
        assert written == 1
        stored = conn.execute(
            "SELECT season_type FROM team_ratings WHERE source = 'sp'"
        ).fetchone()
        assert stored["season_type"] is None


def _seed_regular_season_game(conn, *, season, week, kickoff_utc):
    """Team ids match the numeric CFBD ids ``_team_row`` produces (cfbd:1 /
    cfbd:2): in real usage every team_id comes from the same parse_team()
    function against a real CFBD numeric id, so the games table and the
    fbs-teams-endpoint resolver always agree. Using arbitrary ids here would
    make the resolver produce a team_id the games table never heard of --
    a foreign-key failure that would be a fixture bug, not a real one."""
    store.upsert_team(conn, {"team_id": "cfbd:1", "school": "Home U",
                              "alias": None, "market": None})
    store.upsert_team(conn, {"team_id": "cfbd:2", "school": "Away U",
                              "alias": None, "market": None})
    store.upsert_cfbd_game(conn, {
        "game_id": f"g-{season}-{week}", "season": season, "week": week,
        "season_type": "regular", "kickoff_utc": kickoff_utc,
        "football_date": kickoff_utc[:10], "neutral_site": 0, "conference_game": 0,
        "home_team_id": "cfbd:1", "away_team_id": "cfbd:2",
        "venue_name": None, "venue_id": None, "status": "final",
        "home_points": 30, "away_points": 10, "completed": 1, "source": "cfbd",
    })


class TestBackfillElo:
    def test_writes_a_weekly_row_per_team_per_week(self, conn):
        _seed_regular_season_game(
            conn, season=2023, week=1, kickoff_utc="2023-09-02T00:00:00+00:00")
        client = FakeCfbdClient(
            teams=[_team_row(1, "Home U"), _team_row(2, "Away U")],
            elo_by_week={1: [
                {"year": 2023, "team": "Home U", "elo": 1600},
                {"year": 2023, "team": "Away U", "elo": 1500},
            ]},
        )

        summary = backfill_elo(conn, client, start_year=2023, end_year=2023)

        assert summary.ratings == 2
        stored = conn.execute(
            "SELECT team_id, rating FROM team_ratings WHERE source = 'elo_cfbd' ORDER BY team_id"
        ).fetchall()
        assert [dict(r) for r in stored] == [
            {"team_id": "cfbd:1", "rating": 1600.0},
            {"team_id": "cfbd:2", "rating": 1500.0},
        ]

    def test_an_unresolvable_team_is_filtered_not_fatal(self, conn):
        _seed_regular_season_game(
            conn, season=2023, week=1, kickoff_utc="2023-09-02T00:00:00+00:00")
        client = FakeCfbdClient(
            teams=[_team_row(1, "Home U"), _team_row(2, "Away U")],
            elo_by_week={1: [
                {"year": 2023, "team": "Home U", "elo": 1600},
                {"year": 2023, "team": "Some FCS School", "elo": 1200},
            ]},
        )

        summary = backfill_elo(conn, client, start_year=2023, end_year=2023)

        assert summary.ratings == 1
        assert summary.filtered == 1

    def test_start_year_after_end_year_raises(self, conn):
        with pytest.raises(SchemaError):
            backfill_elo(conn, FakeCfbdClient(teams=[], elo_by_week={}),
                         start_year=2024, end_year=2023)

    def test_no_stored_games_for_the_season_raises(self, conn):
        with pytest.raises(SchemaError):
            backfill_elo(conn, FakeCfbdClient(teams=[], elo_by_week={}),
                         start_year=2023, end_year=2023)


# Dates copied from the published 2026 regular-season schedule (one game per
# week). Weeks 10, 11, 12, 13, and 15 are entirely after 2026-10-31.
_SEASON_2026_WEEKS = {
    1: "2026-08-29",
    2: "2026-09-12",
    3: "2026-09-19",
    4: "2026-09-26",
    5: "2026-10-03",
    6: "2026-10-10",
    7: "2026-10-17",
    8: "2026-10-24",
    9: "2026-10-31",
    10: "2026-11-07",
    11: "2026-11-14",
    12: "2026-11-21",
    13: "2026-11-28",
    15: "2026-12-12",
}
_AS_OF_2026_10_31 = datetime(2026, 10, 31, 12, tzinfo=FOOTBALL_TZ)


class _CountingEloClient(FakeCfbdClient):
    def __init__(self):
        super().__init__(
            teams=[_team_row(1, "Home U"), _team_row(2, "Away U")],
            elo_by_week={},
        )
        self.elo_weeks: list[int] = []
        self.fbs_calls = 0

    def fetch_fbs_teams(self, year: int):
        self.fbs_calls += 1
        return self.teams

    def fetch_elo(self, year: int, week: int, *, season_type: str = "regular"):
        self.elo_weeks.append(week)
        if week == 1:
            return [{"year": year, "team": "Home U", "elo": 1500}]
        return []


class TestEloWeeksSkipTheFuture:
    """backfill_elo used to request every stored week, including weeks that
    have not started. The 2026-10-25 health log recorded weeks 1-13 and 15
    (14 Elo GETs). Weeks 10+ are after 2026-10-31."""

    def test_future_weeks_are_not_requested_and_the_call_count_drops(self, conn):
        for week, day in _SEASON_2026_WEEKS.items():
            _seed_regular_season_game(
                conn, season=2026, week=week, kickoff_utc=f"{day}T23:00:00+00:00"
            )
        client = _CountingEloClient()

        summary = backfill_elo(
            conn, client, start_year=2026, end_year=2026, now=_AS_OF_2026_10_31
        )

        scheduled = len(_SEASON_2026_WEEKS)
        assert client.elo_weeks == list(range(1, 10))
        dropped = scheduled - len(client.elo_weeks)
        assert dropped == 5
        assert client.fbs_calls == 1
        # teams/fbs plus one Elo GET per stored week would have been 15 calls.
        # The filter leaves 10. The drop is the five future weeks.
        assert client.fbs_calls + scheduled == 15
        assert client.fbs_calls + len(client.elo_weeks) == 10
        assert summary.endpoints == 10
        assert {10, 11, 12, 13, 15}.isdisjoint(client.elo_weeks)

        # The as-of stamp is still latest kickoff + 1 day. The week filter
        # must not move it.
        kickoff = datetime.fromisoformat("2026-08-29T23:00:00+00:00")
        stored = conn.execute(
            "SELECT as_of_utc FROM team_ratings WHERE source = 'elo_cfbd' AND week = 1"
        ).fetchone()
        assert stored["as_of_utc"] == (kickoff + timedelta(days=1)).isoformat()

    def test_a_completed_past_season_still_requests_every_stored_week(self, conn):
        for week, kickoff in (
            (1, "2019-08-31T23:00:00+00:00"),
            (2, "2019-09-07T23:00:00+00:00"),
        ):
            _seed_regular_season_game(conn, season=2019, week=week, kickoff_utc=kickoff)
        client = _CountingEloClient()

        backfill_elo(conn, client, start_year=2019, end_year=2019, now=_AS_OF_2026_10_31)

        assert client.elo_weeks == [1, 2]
        assert len({1, 2}) - len(client.elo_weeks) == 0

    def test_only_future_weeks_make_no_elo_request(self, conn):
        _seed_regular_season_game(
            conn, season=2026, week=15, kickoff_utc="2026-12-12T23:00:00+00:00"
        )
        client = _CountingEloClient()

        summary = backfill_elo(
            conn, client, start_year=2026, end_year=2026, now=_AS_OF_2026_10_31
        )

        assert client.elo_weeks == []
        assert client.fbs_calls == 1
        assert summary.ratings == 0
        assert summary.endpoints == 1

    def test_the_week_that_has_started_is_kept_and_the_next_week_is_not(self, conn):
        """On 2026-10-27 Elo fetched week 10 and then 429'd on week 11.

        Week 9's first slate date in the published schedule is 2026-10-27, so
        that week is current. Week 10 starts 2026-11-03 and must not be asked.
        """
        _seed_regular_season_game(
            conn, season=2026, week=9, kickoff_utc="2026-10-27T23:00:00+00:00"
        )
        _seed_regular_season_game(
            conn, season=2026, week=10, kickoff_utc="2026-11-07T23:00:00+00:00"
        )
        _seed_regular_season_game(
            conn, season=2026, week=11, kickoff_utc="2026-11-14T23:00:00+00:00"
        )
        client = _CountingEloClient()
        now = datetime(2026, 10, 27, 12, tzinfo=FOOTBALL_TZ)

        backfill_elo(conn, client, start_year=2026, end_year=2026, now=now)

        assert client.elo_weeks == [9]
        assert 10 not in client.elo_weeks
        assert 11 not in client.elo_weeks
