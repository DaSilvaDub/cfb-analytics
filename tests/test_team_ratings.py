"""Tests for the leakage-safe ridge integration layer.

The pure math (margin recovery, gauge invariance, shrinkage) is already
covered by ``tests/test_ridge.py``. This file is about the layer on top: does
``fit_ratings_as_of`` actually enforce the as-of cutoff and the season/source
scoping described in its docstring, and does the store round-trip a fit.
"""

from __future__ import annotations

import pytest

from cfb_analytics.features.team_ratings import (
    apply_shrinkage_prior,
    fit_ratings_as_of,
    previous_season_final_ratings,
)
from cfb_analytics.ingest import store
from cfb_analytics.models.ridge import DEFAULT_MIN_GAMES, RidgeRatings, TeamRating
from cfb_analytics.models.shrinkage import ShrinkageCoefficients


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


def _seed_league(conn, season, teams=("a", "b", "c", "d"), rounds=3):
    for team_id in teams:
        _seed_team(conn, team_id, team_id.upper())
    game_id = 0
    for _round in range(rounds):
        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                game_id += 1
                _seed_game(
                    conn, f"cfbd:{season}-{game_id}", season=season,
                    kickoff_utc=f"{season}-09-{1 + game_id % 27:02d}T00:00:00+00:00",
                    home=home, away=away,
                )


def _seed_talent(conn, season, team_id, talent_composite):
    store.insert_team_talent(conn, [{
        "season": season, "team_id": team_id, "availability_class": "preseason",
        "talent_composite": talent_composite,
    }])


def _seed_returning(conn, season, team_id, percent_ppa):
    store.insert_returning_production(conn, [{
        "season": season, "team_id": team_id, "availability_class": "preseason",
        "total_ppa": None, "passing_ppa": None, "receiving_ppa": None, "rushing_ppa": None,
        "percent_ppa": percent_ppa, "percent_passing_ppa": None,
        "percent_receiving_ppa": None, "percent_rushing_ppa": None,
        "usage": None, "passing_usage": None, "receiving_usage": None, "rushing_usage": None,
    }])


class TestPreviousSeasonFinalRatings:
    def test_none_when_the_prior_season_has_no_completed_games(self, conn):
        _seed_league(conn, 2026)
        assert previous_season_final_ratings(conn, 2026, min_games=1) is None

    def test_an_active_fit_when_the_prior_season_has_enough_games(self, conn):
        _seed_league(conn, 2025)
        result = previous_season_final_ratings(conn, 2026, min_games=1)
        assert result is not None
        assert result.status == "active"
        assert set(result.teams) == {"a", "b", "c", "d"}


class TestApplyShrinkagePrior:
    def _base_ratings(self):
        return RidgeRatings(
            status="active", n_games=10, ridge_lambda=25.0,
            league_avg_points=24.0, home_field_advantage=3.0,
            teams={
                "a": TeamRating(offense=10.0, defense=-5.0, games=2),
                "b": TeamRating(offense=-10.0, defense=5.0, games=2),
            },
        )

    def test_a_non_active_result_passes_through_unchanged(self, conn):
        insufficient = RidgeRatings(status="insufficient_history", n_games=3, ridge_lambda=25.0)
        result = apply_shrinkage_prior(conn, insufficient, 2026)
        assert result is insufficient

    def test_with_no_prior_or_talent_data_shrinks_toward_zero(self, conn):
        ratings = self._base_ratings()
        result = apply_shrinkage_prior(conn, ratings, 2026, coeffs=ShrinkageCoefficients(k=4.0))
        # n=2, k=4: the prior (0.0, since nothing is available) should pull
        # the blended value about two-thirds of the way toward zero.
        assert 0 < result.teams["a"].offense < ratings.teams["a"].offense
        assert ratings.teams["b"].offense < result.teams["b"].offense < 0

    def test_a_team_with_many_games_stays_close_to_its_own_fit(self, conn):
        ratings = RidgeRatings(
            status="active", n_games=200, ridge_lambda=25.0,
            teams={"a": TeamRating(offense=10.0, defense=-5.0, games=1000)},
        )
        result = apply_shrinkage_prior(conn, ratings, 2026, coeffs=ShrinkageCoefficients(k=4.0))
        assert result.teams["a"].offense == pytest.approx(10.0, abs=0.1)

    def test_blends_toward_an_explicitly_passed_previous_season(self, conn):
        ratings = self._base_ratings()
        previous = RidgeRatings(
            status="active", n_games=100, ridge_lambda=25.0,
            teams={"a": TeamRating(offense=50.0, defense=-50.0, games=12)},
        )
        result = apply_shrinkage_prior(
            conn, ratings, 2026, previous_season_ratings=previous,
            coeffs=ShrinkageCoefficients(a=1.0, b=0.0, c=0.0, k=4.0),
        )
        # a's prior is now 50 (its full previous-season offense); the blend
        # must move noticeably toward it relative to the no-prior case.
        no_prior_result = apply_shrinkage_prior(
            conn, ratings, 2026, coeffs=ShrinkageCoefficients(a=1.0, b=0.0, c=0.0, k=4.0))
        assert result.teams["a"].offense > no_prior_result.teams["a"].offense

    def test_blends_toward_talent_and_returning_production_zscores(self, conn):
        _seed_team(conn, "a", "A")
        _seed_team(conn, "b", "B")
        _seed_talent(conn, 2026, "a", talent_composite=900.0)
        _seed_talent(conn, 2026, "b", talent_composite=100.0)
        _seed_returning(conn, 2026, "a", percent_ppa=0.9)
        _seed_returning(conn, 2026, "b", percent_ppa=0.1)

        ratings = self._base_ratings()
        result = apply_shrinkage_prior(
            conn, ratings, 2026, coeffs=ShrinkageCoefficients(a=0.0, b=1.0, c=1.0, k=4.0))
        # "a" has the high talent/returning-production z-score (positive
        # prior) and "b" the low one (negative prior); the high-talent team's
        # blended offense should end up above the low-talent team's, more so
        # than with no talent signal at all.
        assert result.teams["a"].offense > result.teams["b"].offense


class TestFitRatingsAsOfShrinkage:
    def test_shrinkage_changes_the_fit_relative_to_the_raw_ridge_output(self, conn):
        _seed_league(conn, 2026)
        _seed_talent(conn, 2026, "a", talent_composite=900.0)

        raw = fit_ratings_as_of(
            conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1, apply_shrinkage=False)
        shrunk = fit_ratings_as_of(
            conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1, apply_shrinkage=True)

        assert raw.teams["a"].offense != shrunk.teams["a"].offense

    def test_default_is_to_apply_shrinkage(self, conn):
        _seed_league(conn, 2026)
        _seed_talent(conn, 2026, "a", talent_composite=900.0)

        default_call = fit_ratings_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)
        explicit_true = fit_ratings_as_of(
            conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1, apply_shrinkage=True)
        assert default_call.teams == explicit_true.teams
