"""Tests for the leakage-safe ridge integration layer.

The pure math (margin recovery, gauge invariance, shrinkage) is already
covered by ``tests/test_ridge.py``. This file is about the layer on top: does
``fit_ratings_as_of`` actually enforce the as-of cutoff and the season/source
scoping described in its docstring, and does the store round-trip a fit.
"""

from __future__ import annotations

from cfb_analytics.features.team_ratings import fit_ratings_as_of
from cfb_analytics.ingest import store
from cfb_analytics.models.ridge import DEFAULT_MIN_GAMES


def _seed_team(conn, team_id, school):
    store.upsert_team(conn, {"team_id": team_id, "school": school, "alias": None, "market": None})


def _seed_game(
    conn, game_id, *, season, kickoff_utc, home, away, home_points=30, away_points=10,
    completed=1, source="cfbd", neutral_site=0,
):
    store.upsert_cfbd_game(conn, {
        "game_id": game_id, "season": season, "week": 1, "season_type": "regular",
        "kickoff_utc": kickoff_utc, "football_date": kickoff_utc[:10],
        "neutral_site": neutral_site, "conference_game": 0,
        "home_team_id": home, "away_team_id": away,
        "venue_name": None, "venue_id": None, "status": "final",
        "home_points": home_points, "away_points": away_points, "completed": completed,
        "source": source,
    })


class TestFitRatingsAsOfLeakageGuard:
    def test_a_game_at_or_after_the_cutoff_is_excluded(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, kickoff_utc="2026-09-01T00:00:00+00:00",
                   home="h", away="a")
        _seed_game(conn, "cfbd:2", season=2026, kickoff_utc="2026-09-10T00:00:00+00:00",
                   home="h", away="a")

        before_both = fit_ratings_as_of(conn, 2026, "2026-09-05T00:00:00+00:00", min_games=1)
        after_both = fit_ratings_as_of(conn, 2026, "2026-09-15T00:00:00+00:00", min_games=1)

        assert before_both.n_games == 1
        assert after_both.n_games == 2

    def test_a_different_season_is_excluded_even_if_earlier(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:old", season=2025, kickoff_utc="2025-09-01T00:00:00+00:00",
                   home="h", away="a")
        _seed_game(conn, "cfbd:new", season=2026, kickoff_utc="2026-09-01T00:00:00+00:00",
                   home="h", away="a")

        result = fit_ratings_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)
        assert result.n_games == 1

    def test_outlier_sourced_rows_are_not_read(self, conn):
        """Outlier-sourced ``games`` rows carry no final score and are never
        the ridge fit's input -- only completed CFBD rows are."""
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, kickoff_utc="2026-09-01T00:00:00+00:00",
                   home="h", away="a")
        _seed_game(conn, "outlier:1", season=2026, kickoff_utc="2026-09-01T00:00:00+00:00",
                   home="h", away="a", source="outlier", completed=0,
                   home_points=None, away_points=None)

        result = fit_ratings_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)
        assert result.n_games == 1

    def test_below_the_default_min_games_reports_insufficient_history(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, kickoff_utc="2026-09-01T00:00:00+00:00",
                   home="h", away="a")

        result = fit_ratings_as_of(conn, 2026, "2026-12-01T00:00:00+00:00")
        assert result.status == "insufficient_history"
        assert result.n_games < DEFAULT_MIN_GAMES


class TestUpsertInternalTeamRatings:
    def _active_fit(self, conn):
        teams = ["a", "b", "c", "d"]
        for team_id in teams:
            _seed_team(conn, team_id, team_id.upper())
        game_id = 0
        for _round in range(3):
            for i, home in enumerate(teams):
                for away in teams[i + 1:]:
                    game_id += 1
                    _seed_game(
                        conn, f"cfbd:{game_id}", season=2026,
                        kickoff_utc=f"2026-09-{1 + game_id % 27:02d}T00:00:00+00:00",
                        home=home, away=away,
                    )
        return fit_ratings_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)

    def test_writes_one_row_per_team_when_active(self, conn):
        ratings = self._active_fit(conn)
        assert ratings.status == "active"

        written = store.upsert_internal_team_ratings(
            conn, ratings, season=2026, as_of_utc="2026-12-01T00:00:00+00:00")
        assert written == len(ratings.teams)

        stored = conn.execute(
            "SELECT COUNT(*) AS n FROM internal_team_ratings WHERE season = 2026"
        ).fetchone()["n"]
        assert stored == len(ratings.teams)

    def test_refitting_the_same_as_of_replaces_rather_than_duplicates(self, conn):
        ratings = self._active_fit(conn)
        as_of = "2026-12-01T00:00:00+00:00"
        store.upsert_internal_team_ratings(conn, ratings, season=2026, as_of_utc=as_of)
        store.upsert_internal_team_ratings(conn, ratings, season=2026, as_of_utc=as_of)

        stored = conn.execute(
            "SELECT COUNT(*) AS n FROM internal_team_ratings WHERE season = 2026"
        ).fetchone()["n"]
        assert stored == len(ratings.teams)

    def test_insufficient_history_writes_nothing(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, kickoff_utc="2026-09-01T00:00:00+00:00",
                   home="h", away="a")
        result = fit_ratings_as_of(conn, 2026, "2026-12-01T00:00:00+00:00")

        written = store.upsert_internal_team_ratings(
            conn, result, season=2026, as_of_utc="2026-12-01T00:00:00+00:00")
        assert written == 0
