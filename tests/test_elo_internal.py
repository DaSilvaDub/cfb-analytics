"""Tests for the leakage-safe internal Elo integration layer."""

from __future__ import annotations

from cfb_analytics.features.elo_internal import (
    fit_internal_elo_as_of,
    previous_season_final_elo,
)
from cfb_analytics.ingest import store
from cfb_analytics.models.elo import POOL_TEAM_ID


def _seed_team(conn, team_id, school, classification="fbs"):
    store.upsert_team(conn, {
        "team_id": team_id, "school": school, "alias": None, "market": None,
        "classification": classification,
    })


def _seed_game(
    conn, game_id, *, season, week, kickoff_utc, home, away, home_points=30, away_points=10,
    completed=1, source="cfbd", neutral_site=0,
):
    store.upsert_cfbd_game(conn, {
        "game_id": game_id, "season": season, "week": week, "season_type": "regular",
        "kickoff_utc": kickoff_utc, "football_date": kickoff_utc[:10],
        "neutral_site": neutral_site, "conference_game": 0,
        "home_team_id": home, "away_team_id": away,
        "venue_name": None, "venue_id": None, "status": "final",
        "home_points": home_points, "away_points": away_points, "completed": completed,
        "source": source,
    })


class TestFitInternalEloAsOfLeakageGuard:
    def test_a_game_at_or_after_the_cutoff_is_excluded(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, week=1,
                   kickoff_utc="2026-09-01T00:00:00+00:00", home="h", away="a")
        _seed_game(conn, "cfbd:2", season=2026, week=2,
                   kickoff_utc="2026-09-10T00:00:00+00:00", home="h", away="a")

        before_both = fit_internal_elo_as_of(
            conn, 2026, "2026-09-05T00:00:00+00:00", min_games=1)
        after_both = fit_internal_elo_as_of(
            conn, 2026, "2026-09-15T00:00:00+00:00", min_games=1)

        assert before_both.n_games == 1
        assert after_both.n_games == 2

    def test_a_different_season_is_excluded_even_if_earlier(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:old", season=2025, week=1,
                   kickoff_utc="2025-09-01T00:00:00+00:00", home="h", away="a")
        _seed_game(conn, "cfbd:new", season=2026, week=1,
                   kickoff_utc="2026-09-01T00:00:00+00:00", home="h", away="a")

        result = fit_internal_elo_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)
        assert result.n_games == 1

    def test_below_min_games_reports_insufficient_history(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, week=1,
                   kickoff_utc="2026-09-01T00:00:00+00:00", home="h", away="a")

        result = fit_internal_elo_as_of(conn, 2026, "2026-12-01T00:00:00+00:00")
        assert result.status == "insufficient_history"


class TestFcsPooling:
    def test_a_non_fbs_opponent_is_remapped_to_the_shared_pool(self, conn):
        _seed_team(conn, "h", "Home U", classification="fbs")
        _seed_team(conn, "fcs1", "Some FCS School", classification="fcs")
        _seed_game(conn, "cfbd:1", season=2026, week=1,
                   kickoff_utc="2026-09-01T00:00:00+00:00", home="h", away="fcs1")

        result = fit_internal_elo_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)
        assert "fcs1" not in result.teams
        assert POOL_TEAM_ID in result.teams

    def test_an_unclassified_team_is_also_pooled_not_trusted_as_fbs(self, conn):
        """Never guess: a team with no confirmed classification is treated
        as not-FBS, the conservative default, not individually tracked."""
        _seed_team(conn, "h", "Home U", classification="fbs")
        _seed_team(conn, "mystery", "Unknown Program", classification=None)
        _seed_game(conn, "cfbd:1", season=2026, week=1,
                   kickoff_utc="2026-09-01T00:00:00+00:00", home="h", away="mystery")

        result = fit_internal_elo_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)
        assert "mystery" not in result.teams
        assert POOL_TEAM_ID in result.teams

    def test_multiple_fcs_opponents_share_one_pooled_rating(self, conn):
        _seed_team(conn, "h", "Home U", classification="fbs")
        _seed_team(conn, "h2", "Home U 2", classification="fbs")
        _seed_team(conn, "fcs1", "FCS One", classification="fcs")
        _seed_team(conn, "fcs2", "FCS Two", classification="fcs")
        _seed_game(conn, "cfbd:1", season=2026, week=1,
                   kickoff_utc="2026-09-01T00:00:00+00:00", home="h", away="fcs1")
        _seed_game(conn, "cfbd:2", season=2026, week=2,
                   kickoff_utc="2026-09-08T00:00:00+00:00", home="h2", away="fcs2")

        result = fit_internal_elo_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)
        assert result.teams[POOL_TEAM_ID].games == 2


class TestPreviousSeasonFinalElo:
    def test_none_at_the_stores_earliest_season(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2020, week=1,
                   kickoff_utc="2020-09-01T00:00:00+00:00", home="h", away="a")

        assert previous_season_final_elo(conn, 2020, min_games=1) is None

    def test_an_active_fit_when_the_prior_season_has_games(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2020, week=1,
                   kickoff_utc="2020-09-01T00:00:00+00:00", home="h", away="a")
        _seed_game(conn, "cfbd:2", season=2021, week=1,
                   kickoff_utc="2021-09-01T00:00:00+00:00", home="h", away="a")

        result = previous_season_final_elo(conn, 2021, min_games=1)
        assert result is not None
        assert result.status == "active"
        # Every FBS team is seeded (and so appears) whether or not it
        # actually played in this fit -- see fit_elo's own "a seeded but
        # never-playing team still appears at its seed" contract -- so the
        # pool identifier legitimately shows up here too, at 0 games,
        # alongside the two teams that actually played each other.
        assert {"h", "a"}.issubset(result.teams)
        assert result.teams["h"].games == 1
        assert result.teams["a"].games == 1

    def test_a_gap_season_with_no_games_does_not_carry_a_stale_chain_forward(self, conn):
        """2020 has games, 2021 has none, 2022 has games again -- 2022's
        preseason blend must not silently inherit 2020's final ratings
        across the gap."""
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2020, week=1,
                   kickoff_utc="2020-09-01T00:00:00+00:00", home="h", away="a",
                   home_points=70, away_points=0)

        result = previous_season_final_elo(conn, 2022, min_games=1)
        assert result is None


class TestPreseasonChaining:
    def test_a_dominant_previous_season_raises_this_seasons_starting_ratings(self, conn):
        """"h" crushed "a" all of last season. This season, "h" plays a
        brand-new team "c" (no prior-season history at all, so "c" starts
        at the flat generic baseline) in a game close enough (10 points)
        that a single game's own update cannot explain a large gap -- any
        real separation has to be coming from the carried-forward preseason
        blend, not from this season's own (thin) evidence."""
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_team(conn, "c", "Brand New To FBS")
        for week, (hp, ap) in enumerate([(50, 0), (50, 0), (50, 0)], start=1):
            _seed_game(
                conn, f"cfbd:2020-{week}", season=2020, week=week,
                kickoff_utc=f"2020-09-{week:02d}T00:00:00+00:00",
                home="h", away="a", home_points=hp, away_points=ap,
            )
        _seed_game(conn, "cfbd:2021-1", season=2021, week=1,
                   kickoff_utc="2021-09-01T00:00:00+00:00", home="h", away="c",
                   home_points=24, away_points=14)

        result = fit_internal_elo_as_of(conn, 2021, "2021-09-02T00:00:00+00:00", min_games=1)
        assert result.status == "active"
        assert result.teams["h"].rating > result.teams["c"].rating + 50

    def test_a_weak_previous_season_lowers_this_seasons_starting_rating(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_team(conn, "c", "Brand New To FBS")
        for week, (hp, ap) in enumerate([(0, 50), (0, 50), (0, 50)], start=1):
            _seed_game(
                conn, f"cfbd:2020-{week}", season=2020, week=week,
                kickoff_utc=f"2020-09-{week:02d}T00:00:00+00:00",
                home="h", away="a", home_points=hp, away_points=ap,
            )
        _seed_game(conn, "cfbd:2021-1", season=2021, week=1,
                   kickoff_utc="2021-09-01T00:00:00+00:00", home="h", away="c",
                   home_points=14, away_points=24)

        result = fit_internal_elo_as_of(conn, 2021, "2021-09-02T00:00:00+00:00", min_games=1)
        assert result.status == "active"
        assert result.teams["h"].rating < result.teams["c"].rating - 50


class TestUpsertInternalEloRatings:
    def test_writes_one_row_per_team_including_the_pool(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "fcs1", "Some FCS School", classification="fcs")
        _seed_game(conn, "cfbd:1", season=2026, week=1,
                   kickoff_utc="2026-09-01T00:00:00+00:00", home="h", away="fcs1")

        ratings = fit_internal_elo_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)
        assert ratings.status == "active"

        written = store.upsert_internal_elo_ratings(
            conn, ratings, season=2026, as_of_utc="2026-12-01T00:00:00+00:00")
        assert written == len(ratings.teams)

        stored = conn.execute(
            "SELECT team_id FROM internal_elo_ratings WHERE season = 2026"
        ).fetchall()
        stored_ids = {row["team_id"] for row in stored}
        assert "h" in stored_ids
        assert POOL_TEAM_ID in stored_ids  # no FK to teams(team_id) -- see the migration

    def test_refitting_the_same_as_of_replaces_rather_than_duplicates(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, week=1,
                   kickoff_utc="2026-09-01T00:00:00+00:00", home="h", away="a")
        ratings = fit_internal_elo_as_of(conn, 2026, "2026-12-01T00:00:00+00:00", min_games=1)
        as_of = "2026-12-01T00:00:00+00:00"

        store.upsert_internal_elo_ratings(conn, ratings, season=2026, as_of_utc=as_of)
        store.upsert_internal_elo_ratings(conn, ratings, season=2026, as_of_utc=as_of)

        stored = conn.execute(
            "SELECT COUNT(*) AS n FROM internal_elo_ratings WHERE season = 2026"
        ).fetchone()["n"]
        assert stored == len(ratings.teams)

    def test_insufficient_history_writes_nothing(self, conn):
        _seed_team(conn, "h", "Home U")
        _seed_team(conn, "a", "Away U")
        _seed_game(conn, "cfbd:1", season=2026, week=1,
                   kickoff_utc="2026-09-01T00:00:00+00:00", home="h", away="a")
        result = fit_internal_elo_as_of(conn, 2026, "2026-12-01T00:00:00+00:00")

        written = store.upsert_internal_elo_ratings(
            conn, result, season=2026, as_of_utc="2026-12-01T00:00:00+00:00")
        assert written == 0
