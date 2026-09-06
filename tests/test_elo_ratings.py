"""Tests for the leakage-safe CFBD weekly-Elo lookup.

The subtlety this guards: a stored elo_cfbd row for "week W" reflects the
rating AFTER week W's games (see the module docstring on
features/elo_ratings.py), so predicting week W's games must resolve to week
W-1's row, never week W's own row.
"""

from __future__ import annotations

from cfb_analytics.features.elo_ratings import elo_rating_as_of
from cfb_analytics.ingest import store


def _elo_row(team_id, season, week, as_of_utc, rating, season_type="regular"):
    return {
        "season": season, "period": f"{season_type}:week:{week:02d}", "week": week,
        "season_type": season_type, "team_id": team_id, "source": "elo_cfbd",
        "snapshot_scope": "weekly", "provenance_mode": "reconstructed",
        "as_of_utc": as_of_utc, "rating": rating, "ranking": None,
        "off_rating": None, "def_rating": None, "st_rating": None,
        "sos": None, "second_order_wins": None,
    }


def _seed_teams(conn, *team_ids):
    for team_id in team_ids:
        store.upsert_team(conn, {"team_id": team_id, "school": team_id,
                                  "alias": None, "market": None})


class TestEloRatingAsOf:
    def test_returns_none_with_no_stored_history(self, conn):
        assert elo_rating_as_of(conn, "cfbd:1", 2023, "2023-09-08T00:00:00+00:00") is None

    def test_resolves_to_the_prior_weeks_rating_not_this_weeks(self, conn):
        """A row for week 2 (as_of 2023-09-15) must not be visible when
        predicting week 2's own games (cutoff strictly before that)."""
        _seed_teams(conn, "cfbd:1")
        store.insert_team_ratings(conn, [
            _elo_row("cfbd:1", 2023, 1, "2023-09-08T00:00:00+00:00", 1600.0),
            _elo_row("cfbd:1", 2023, 2, "2023-09-15T00:00:00+00:00", 1650.0),
        ])
        # Cutoff = week 2's own start (its earliest kickoff), strictly before
        # week 2's own row's as_of_utc but strictly after week 1's.
        rating = elo_rating_as_of(conn, "cfbd:1", 2023, "2023-09-09T00:00:00+00:00")
        assert rating == 1600.0

    def test_advances_once_the_next_weeks_row_becomes_admissible(self, conn):
        _seed_teams(conn, "cfbd:1")
        store.insert_team_ratings(conn, [
            _elo_row("cfbd:1", 2023, 1, "2023-09-08T00:00:00+00:00", 1600.0),
            _elo_row("cfbd:1", 2023, 2, "2023-09-15T00:00:00+00:00", 1650.0),
        ])
        rating = elo_rating_as_of(conn, "cfbd:1", 2023, "2023-09-16T00:00:00+00:00")
        assert rating == 1650.0

    def test_a_cutoff_before_any_row_returns_none(self, conn):
        _seed_teams(conn, "cfbd:1")
        store.insert_team_ratings(conn, [
            _elo_row("cfbd:1", 2023, 1, "2023-09-08T00:00:00+00:00", 1600.0),
        ])
        assert elo_rating_as_of(conn, "cfbd:1", 2023, "2023-09-01T00:00:00+00:00") is None

    def test_does_not_mix_up_a_different_team_or_season(self, conn):
        _seed_teams(conn, "cfbd:1", "cfbd:2")
        store.insert_team_ratings(conn, [
            _elo_row("cfbd:1", 2023, 1, "2023-09-08T00:00:00+00:00", 1600.0),
            _elo_row("cfbd:2", 2023, 1, "2023-09-08T00:00:00+00:00", 1400.0),
            _elo_row("cfbd:1", 2022, 1, "2022-09-08T00:00:00+00:00", 1900.0),
        ])
        assert elo_rating_as_of(conn, "cfbd:1", 2023, "2023-12-01T00:00:00+00:00") == 1600.0
        assert elo_rating_as_of(conn, "cfbd:2", 2023, "2023-12-01T00:00:00+00:00") == 1400.0
