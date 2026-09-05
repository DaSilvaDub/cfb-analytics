"""Tests for the CFBD fundamentals backfill, focused on the weekly-Elo path.

This module had zero test coverage before this file: the entire weekly-Elo
write path silently inserted nothing for its whole existence (a CHECK
constraint on ``team_ratings`` rejected every row via ``INSERT OR IGNORE``,
with no error surfaced anywhere) and nothing caught it. See
``parse_elo_rating``'s docstring in ``sources/cfbd.py`` and
``insert_team_ratings``'s in ``ingest/store.py`` for the two-part fix.
"""

from __future__ import annotations

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_fundamentals import backfill_elo
from cfb_analytics.sources.cfbd import parse_elo_rating, parse_sp_rating


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
