"""Tests for the walk-forward harness's own contract: correct week ordering,
season isolation, and honest accounting of what it could not predict.

The ridge math itself (margin recovery, gauge invariance) is covered by
tests/test_ridge.py; fit_ratings_as_of's leakage guard is covered by
tests/test_team_ratings.py. This file is about the harness layer on top:
does it refit once per week from strictly-earlier games, and does it count
(rather than silently drop) games it could not predict.
"""

from __future__ import annotations

import pytest

from cfb_analytics.backtest.harness import run_walk_forward
from cfb_analytics.ingest import store
from cfb_analytics.models.ridge import fit_ratings


def _seed_team(conn, team_id, school):
    store.upsert_team(conn, {"team_id": team_id, "school": school, "alias": None, "market": None})


def _seed_game(
    conn, game_id, *, season, week, kickoff_utc, home, away, home_points=30, away_points=10,
    season_type="regular", neutral_site=0,
):
    store.upsert_cfbd_game(conn, {
        "game_id": game_id, "season": season, "week": week, "season_type": season_type,
        "kickoff_utc": kickoff_utc, "football_date": kickoff_utc[:10],
        "neutral_site": neutral_site, "conference_game": 0,
        "home_team_id": home, "away_team_id": away,
        "venue_name": None, "venue_id": None, "status": "final",
        "home_points": home_points, "away_points": away_points, "completed": 1,
        "source": "cfbd",
    })


class TestWeekOrderingAndHistory:
    def test_the_first_week_of_a_season_is_always_skipped(self, conn):
        """Even with min_games=1, week 1 has zero strictly-earlier games."""
        _seed_team(conn, "a", "A")
        _seed_team(conn, "b", "B")
        _seed_game(conn, "g1", season=2020, week=1,
                   kickoff_utc="2020-09-05T00:00:00+00:00", home="a", away="b")

        run = run_walk_forward(conn, [2020], min_games=1)
        assert run.predictions == []
        assert run.skipped_insufficient_history == 1

    def test_a_later_week_is_predicted_from_the_earlier_weeks_history(self, conn):
        _seed_team(conn, "a", "A")
        _seed_team(conn, "b", "B")
        _seed_game(conn, "g1", season=2020, week=1,
                   kickoff_utc="2020-09-05T00:00:00+00:00", home="a", away="b")
        _seed_game(conn, "g2", season=2020, week=2,
                   kickoff_utc="2020-09-12T00:00:00+00:00", home="b", away="a")

        run = run_walk_forward(conn, [2020], min_games=1)
        assert len(run.predictions) == 1
        assert run.predictions[0].game_id == "g2"
        assert run.skipped_insufficient_history == 1  # week 1's one game

    def test_a_game_in_the_same_week_never_sees_another_game_in_that_week(self, conn):
        """Two games sharing a week (a Thursday and a Saturday game) must not
        leak into each other's fit -- the harness cuts off at the week's
        EARLIEST kickoff, not each individual game's own kickoff."""
        _seed_team(conn, "a", "A")
        _seed_team(conn, "b", "B")
        _seed_team(conn, "c", "C")
        _seed_team(conn, "d", "D")
        _seed_game(conn, "g1", season=2020, week=1,
                   kickoff_utc="2020-09-01T00:00:00+00:00", home="a", away="b")
        # Same week, later in the week -- must be treated as "no history yet"
        # exactly like g1, not as if g1 already happened.
        _seed_game(conn, "g2", season=2020, week=1,
                   kickoff_utc="2020-09-05T00:00:00+00:00", home="c", away="d")

        run = run_walk_forward(conn, [2020], min_games=1)
        assert run.predictions == []
        assert run.skipped_insufficient_history == 2


class TestSeasonIsolation:
    def test_one_seasons_history_never_feeds_another_seasons_fit(self, conn):
        _seed_team(conn, "a", "A")
        _seed_team(conn, "b", "B")
        # A full 2019 season's worth of history for A vs B...
        _seed_game(conn, "g2019-1", season=2019, week=1,
                   kickoff_utc="2019-09-01T00:00:00+00:00", home="a", away="b")
        _seed_game(conn, "g2019-2", season=2019, week=2,
                   kickoff_utc="2019-09-08T00:00:00+00:00", home="b", away="a")
        # ...must not count as history for 2020 week 1, which has none of its own.
        _seed_game(conn, "g2020-1", season=2020, week=1,
                   kickoff_utc="2020-09-05T00:00:00+00:00", home="a", away="b")

        run = run_walk_forward(conn, [2019, 2020], min_games=1)
        assert run.skipped_insufficient_history == 2  # 2019 wk1 + 2020 wk1
        assert len(run.predictions) == 1
        assert run.predictions[0].game_id == "g2019-2"


class TestPostseasonExclusion:
    def test_postseason_games_are_not_walked_forward(self, conn):
        _seed_team(conn, "a", "A")
        _seed_team(conn, "b", "B")
        _seed_game(conn, "g1", season=2020, week=1,
                   kickoff_utc="2020-09-05T00:00:00+00:00", home="a", away="b")
        _seed_game(conn, "g2", season=2020, week=2,
                   kickoff_utc="2020-09-12T00:00:00+00:00", home="b", away="a")
        _seed_game(conn, "bowl", season=2020, week=1, season_type="postseason",
                   kickoff_utc="2021-01-01T00:00:00+00:00", home="a", away="b")

        run = run_walk_forward(conn, [2020], min_games=1)
        assert all(p.game_id != "bowl" for p in run.predictions)


class TestUnratedTeam:
    def test_a_team_with_no_prior_history_this_season_is_counted_not_dropped(self, conn):
        """An FCS opponent (or any team with zero prior games this season)
        gives `RidgeRatings.margin` nothing to rate it with; the harness must
        count that rather than silently skip it."""
        _seed_team(conn, "a", "A")
        _seed_team(conn, "b", "B")
        _seed_team(conn, "fcs", "FCS School")
        _seed_game(conn, "g1", season=2020, week=1,
                   kickoff_utc="2020-09-05T00:00:00+00:00", home="a", away="b")
        # week 2: A plays an opponent that has never appeared in this season's history.
        _seed_game(conn, "g2", season=2020, week=2,
                   kickoff_utc="2020-09-12T00:00:00+00:00", home="a", away="fcs")

        run = run_walk_forward(conn, [2020], min_games=1)
        assert run.skipped_unrated_team == 1
        assert all(p.game_id != "g2" for p in run.predictions)


class TestNeutralSite:
    def test_neutral_site_flag_removes_exactly_the_fitted_home_field_edge(self, conn):
        """Two games in the SAME week, same pair of teams, same fit -- only
        ``neutral_site`` differs -- so the two predicted margins must differ
        by exactly the home-field term that fit produced, not approximately."""
        teams = ["a", "b", "c", "d"]
        for t in teams:
            _seed_team(conn, t, t.upper())
        game_id = 0
        for week in range(1, 4):
            for i, home in enumerate(teams):
                for away in teams[i + 1:]:
                    game_id += 1
                    _seed_game(
                        conn, f"g{game_id}", season=2020, week=week,
                        kickoff_utc=f"2020-09-{week:02d}T00:00:00+00:00",
                        home=home, away=away,
                    )
        _seed_game(conn, "week4-home", season=2020, week=4,
                   kickoff_utc="2020-09-28T00:00:00+00:00", home="a", away="b")
        _seed_game(conn, "week4-neutral", season=2020, week=4,
                   kickoff_utc="2020-09-28T00:00:00+00:00", home="a", away="b", neutral_site=1)

        run = run_walk_forward(conn, [2020], min_games=1)
        home_pred = next(p for p in run.predictions if p.game_id == "week4-home")
        neutral_pred = next(p for p in run.predictions if p.game_id == "week4-neutral")

        history = conn.execute(
            """SELECT home_team_id, away_team_id, home_points, away_points, neutral_site
               FROM games WHERE season = 2020 AND week < 4"""
        ).fetchall()
        expected = fit_ratings([dict(row) for row in history], min_games=1)

        assert home_pred.predicted_margin - neutral_pred.predicted_margin == pytest.approx(
            expected.home_field_advantage
        )
