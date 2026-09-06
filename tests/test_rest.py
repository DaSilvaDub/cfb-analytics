"""Tests for the rest-days lookup (plan section 5's T5 ``rest_diff``)."""

from __future__ import annotations

import pytest

from cfb_analytics.features.rest import DEFAULT_REST_DAYS, RestLookup
from cfb_analytics.ingest import store


def _seed_team(conn, team_id, school):
    store.upsert_team(conn, {"team_id": team_id, "school": school, "alias": None, "market": None})


def _seed_game(conn, game_id, *, kickoff_utc, home, away, season=2019, week=1):
    store.upsert_cfbd_game(conn, {
        "game_id": game_id, "season": season, "week": week, "season_type": "regular",
        "kickoff_utc": kickoff_utc, "football_date": kickoff_utc[:10],
        "neutral_site": 0, "conference_game": 0,
        "home_team_id": home, "away_team_id": away,
        "venue_name": None, "venue_id": None, "status": "final",
        "home_points": 24, "away_points": 10, "completed": 1, "source": "cfbd",
    })


class TestRestLookup:
    def test_no_prior_game_uses_the_default(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        lookup = RestLookup(conn)
        assert lookup.rest_days("h", "2019-09-01T00:00:00+00:00") == DEFAULT_REST_DAYS

    def test_counts_days_since_the_most_recent_prior_game(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_team(conn, "b", "Third U")
        _seed_game(conn, "g1", kickoff_utc="2019-09-01T00:00:00+00:00", home="h", away="b",
                   week=1)
        lookup = RestLookup(conn)
        assert lookup.rest_days("h", "2019-09-08T00:00:00+00:00") == pytest.approx(7.0)

    def test_counts_a_team_regardless_of_home_or_away_side(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_team(conn, "b", "Third U")
        _seed_game(conn, "g1", kickoff_utc="2019-09-01T00:00:00+00:00", home="b", away="h",
                   week=1)
        lookup = RestLookup(conn)
        assert lookup.rest_days("h", "2019-09-06T00:00:00+00:00") == pytest.approx(5.0)

    def test_only_a_game_strictly_before_as_of_utc_counts(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_team(conn, "b", "Third U")
        _seed_game(conn, "g1", kickoff_utc="2019-09-01T00:00:00+00:00", home="h", away="b",
                   week=1)
        _seed_game(conn, "g2", kickoff_utc="2019-09-08T00:00:00+00:00", home="h", away="b",
                   week=2)
        lookup = RestLookup(conn)
        # As of week 2's own kickoff, week 2's row must not count as "prior".
        assert lookup.rest_days("h", "2019-09-08T00:00:00+00:00") == pytest.approx(7.0)

    def test_a_team_with_no_games_at_all_uses_the_default(self, conn):
        _seed_team(conn, "h", "Home U")
        lookup = RestLookup(conn)
        assert lookup.rest_days("h", "2019-09-01T00:00:00+00:00") == DEFAULT_REST_DAYS
