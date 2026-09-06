"""Tests for the leakage-safe P_logit integration layer."""

from __future__ import annotations

from cfb_analytics.features.ensemble import fit_logistic_as_of, game_features
from cfb_analytics.ingest import store


def _seed_team(conn, team_id, school):
    store.upsert_team(conn, {"team_id": team_id, "school": school, "alias": None, "market": None})


def _seed_game(
    conn, game_id, *, season, kickoff_utc, home, away, home_points=30, away_points=10,
    week=1, neutral_site=0,
):
    store.upsert_cfbd_game(conn, {
        "game_id": game_id, "season": season, "week": week, "season_type": "regular",
        "kickoff_utc": kickoff_utc, "football_date": kickoff_utc[:10],
        "neutral_site": neutral_site, "conference_game": 0,
        "home_team_id": home, "away_team_id": away,
        "venue_name": None, "venue_id": None, "status": "final",
        "home_points": home_points, "away_points": away_points, "completed": 1,
        "source": "cfbd",
    })


class TestGameFeatures:
    def test_missing_zscores_contribute_zero_not_a_crash(self):
        features = game_features("h", "a", neutral_site=False, talent_z={}, returning_z={})
        assert features == [0.0, 0.0, 1.0]

    def test_talent_diff_is_home_minus_away(self):
        features = game_features(
            "h", "a", neutral_site=False, talent_z={"h": 1.5, "a": -0.5}, returning_z={})
        assert features[0] == 2.0

    def test_neutral_site_gives_a_zero_home_field_indicator(self):
        features = game_features("h", "a", neutral_site=True, talent_z={}, returning_z={})
        assert features[2] == 0.0

    def test_home_game_gives_a_one_home_field_indicator(self):
        features = game_features("h", "a", neutral_site=False, talent_z={}, returning_z={})
        assert features[2] == 1.0


class TestFitLogisticAsOfLeakageGuard:
    def test_a_game_at_or_after_the_cutoff_is_excluded(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, kickoff_utc="2026-09-01T00:00:00+00:00",
                   home="h", away="a")
        _seed_game(conn, "cfbd:2", season=2026, kickoff_utc="2026-09-10T00:00:00+00:00",
                   home="h", away="a")

        before_both = fit_logistic_as_of(conn, "2026-09-05T00:00:00+00:00", min_n=1)
        after_both = fit_logistic_as_of(conn, "2026-09-15T00:00:00+00:00", min_n=1)
        assert before_both.n == 1
        assert after_both.n == 2

    def test_pools_games_across_multiple_seasons_not_just_the_current_one(self, conn):
        """Unlike ridge/Elo, P_logit's training set is NOT season-scoped --
        this is the deliberate behavior the module docstring describes."""
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:2020", season=2020, kickoff_utc="2020-09-01T00:00:00+00:00",
                   home="h", away="a")
        _seed_game(conn, "cfbd:2021", season=2021, kickoff_utc="2021-09-01T00:00:00+00:00",
                   home="h", away="a")

        result = fit_logistic_as_of(conn, "2026-01-01T00:00:00+00:00", min_n=1)
        assert result.n == 2

    def test_a_tied_score_is_dropped_not_fabricated_a_winner(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, kickoff_utc="2026-09-01T00:00:00+00:00",
                   home="h", away="a", home_points=20, away_points=20)

        result = fit_logistic_as_of(conn, "2026-12-01T00:00:00+00:00", min_n=1)
        assert result.status == "insufficient_data"

    def test_below_min_n_reports_insufficient_data(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, kickoff_utc="2026-09-01T00:00:00+00:00",
                   home="h", away="a")

        result = fit_logistic_as_of(conn, "2026-12-01T00:00:00+00:00", min_n=30)
        assert result.status == "insufficient_data"

    def test_an_active_fit_can_predict_a_new_games_probability(self, conn):
        teams = ["a", "b", "c", "d"]
        for team_id in teams:
            _seed_team(conn, team_id, team_id.upper())
        game_id = 0
        for _round in range(10):
            for i, home in enumerate(teams):
                for away in teams[i + 1:]:
                    game_id += 1
                    _seed_game(
                        conn, f"cfbd:{game_id}", season=2020,
                        kickoff_utc=f"2020-09-{1 + game_id % 27:02d}T00:00:00+00:00",
                        home=home, away=away,
                    )

        result = fit_logistic_as_of(conn, "2026-01-01T00:00:00+00:00", min_n=1)
        assert result.status == "active"
        prob = result.probability([0.0, 0.0, 1.0])
        assert prob is not None
        assert 0.0 < prob < 1.0
