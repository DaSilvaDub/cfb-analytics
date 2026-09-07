"""Tests for Candidate Play Scoring and Team Props Pipeline Integration."""

from __future__ import annotations

from datetime import datetime
from inspect import signature

import pytest

from cfb_analytics.daily import run_daily
from cfb_analytics.errors import SchemaError
from cfb_analytics.ingest import store
from cfb_analytics.models.team_props import (
    TeamPropsInputs,
    default_team_props_inputs,
    is_player_prop,
    is_supported_team_prop,
    normalize_team_prop_market,
    project_team_points,
    project_team_production,
    validate_market_type,
)
from cfb_analytics.scoring import (
    TeamPropCandidate,
    assign_play_tier,
    build_team_props_inputs_from_db,
    filter_team_prop_candidates,
    load_team_props_inputs_for_slate,
    resolve_devigged_fair_prob,
    score_candidate,
    score_candidates,
    score_confirming_insights,
    score_daily_slates,
    score_edge,
    score_hit_rate,
    score_line_movement,
    score_price_value,
    score_sample_size,
    score_slate_team_props,
)
from cfb_analytics.utils import FOOTBALL_TZ


@pytest.fixture
def sample_inputs() -> TeamPropsInputs:
    return TeamPropsInputs(
        pace=70.0,
        expected_possession_count=12.0,
        offensive_success_rate=0.45,
        explosiveness=1.5,
        expected_pass_attempts=35.0,
        expected_rushing_attempts=35.0,
        completion_probability=0.6,
        yards_per_completion=12.0,
        yards_before_contact=2.5,
        yards_after_contact=2.0,
    )


class TestStrictPlayerPropFiltering:
    """Proves player props are strictly filtered and rejected."""

    def test_validate_market_type_rejects_player_props(self):
        with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
            validate_market_type("PLAYER_PROP")
        with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
            validate_market_type("player_prop")
        with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
            validate_market_type("player_passing_yards")
        with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
            validate_market_type("pass_yards_player")
        with pytest.raises(SchemaError, match="Player props are strictly prohibited"):
            validate_market_type("PLAYER")

    def test_validate_market_type_allows_team_props(self):
        validate_market_type("TEAM_PROP")
        with pytest.raises(SchemaError, match="requires market_type='TEAM_PROP'"):
            validate_market_type("GAMELINE")

    def test_is_player_prop(self):
        assert is_player_prop("player_passing_yards") is True
        assert is_player_prop("PLAYER_RUSHING_YARDS") is True
        assert is_player_prop("pass_attempts") is True
        assert is_player_prop("rush_yards") is True
        assert is_player_prop("rec_yards") is True
        assert is_player_prop("team_offensive_yards") is False
        assert is_player_prop("team_rushing_yards") is False
        assert is_player_prop("team_receiving_yards") is False
        assert is_player_prop("team_total_points") is False

    def test_is_supported_team_prop(self):
        assert is_supported_team_prop("team_offensive_yards") is True
        assert is_supported_team_prop("team_rushing_yards") is True
        assert is_supported_team_prop("team_receiving_yards") is True
        assert is_supported_team_prop("team_total_points") is True
        assert is_supported_team_prop("offensive_yards") is True
        assert is_supported_team_prop("rushing_yards") is True
        assert is_supported_team_prop("receiving_yards") is True
        assert is_supported_team_prop("points") is True
        assert is_supported_team_prop("team_total") is True

        # Player props and unsupported markets are False
        assert is_supported_team_prop("player_passing_yards") is False
        assert is_supported_team_prop("player_prop") is False
        assert is_supported_team_prop("coin_toss") is False
        assert is_supported_team_prop("halftime_show") is False

    def test_normalize_team_prop_market(self):
        assert normalize_team_prop_market("team_offensive_yards") == "team_offensive_yards"
        assert normalize_team_prop_market("OFFENSIVE_YARDS") == "team_offensive_yards"
        assert normalize_team_prop_market("rushing_yards") == "team_rushing_yards"
        assert normalize_team_prop_market("receiving_yards") == "team_receiving_yards"
        assert normalize_team_prop_market("points") == "team_total_points"

        with pytest.raises(SchemaError):
            normalize_team_prop_market("player_passing_yards")
        with pytest.raises(SchemaError):
            normalize_team_prop_market("unsupported_prop")

    def test_filter_team_prop_candidates(self):
        candidates = [
            {
                "game_id": "g1",
                "team_id": "t1",
                "market_type": "TEAM_PROP",
                "market": "team_offensive_yards",
                "side": "OVER",
                "line": 400.5,
            },
            {
                "game_id": "g1",
                "team_id": "t1",
                "market_type": "PLAYER_PROP",
                "market": "passing_yards",
                "side": "OVER",
                "line": 275.5,
            },
            {
                "game_id": "g1",
                "team_id": "t1",
                "market_type": "PLAYER_PROP",
                "market": "PLAYER_PROP",
                "side": "UNDER",
                "line": 1.5,
            },
            {
                "game_id": "g1",
                "team_id": "t1",
                "market_type": "TEAM_PROP",
                "market": "team_rushing_yards",
                "side": "OVER",
                "line": 150.5,
            },
            {
                "game_id": "g1",
                "team_id": "t1",
                "market_type": "GAME_PROP",
                "market": "coin_toss",
                "side": "HEADS",
                "line": 0.5,
            },
        ]
        filtered = filter_team_prop_candidates(candidates)
        assert len(filtered) == 2
        markets = {c.market for c in filtered}
        assert markets == {"team_offensive_yards", "team_rushing_yards"}

    def test_score_candidate_rejects_player_prop_strictly(self, sample_inputs):
        cand = TeamPropCandidate(
            game_id="g1",
            team_id="t1",
            market="passing_yards",
            side="OVER",
            line=250.5,
            market_type="PLAYER_PROP",
        )
        with pytest.raises(SchemaError, match="requires market_type='TEAM_PROP'"):
            score_candidate(cand, sample_inputs, strict=True)

        # In non-strict mode, returns rejected PASS score
        res = score_candidate(cand, sample_inputs, strict=False)
        assert res.rejection_reason == "player_prop_strictly_prohibited"
        assert res.tier == "PASS"
        assert res.is_actionable is False


class TestModel3Projections:
    """Proves Model 3 cascading logic with points projection."""

    def test_project_team_production_includes_points(self, sample_inputs):
        proj = project_team_production("TEAM_PROP", sample_inputs)
        assert proj.expected_offensive_plays == 70.0
        assert proj.expected_yards_per_play == 5.85
        assert proj.projected_team_offensive_yards == 409.5
        assert proj.projected_team_receiving_yards == 252.0
        assert proj.projected_team_rushing_yards == 157.5
        assert proj.projected_team_total_points > 20.0

    def test_project_team_points_zero_yards(self):
        zero_inputs = TeamPropsInputs(
            pace=0,
            expected_possession_count=0,
            offensive_success_rate=0,
            explosiveness=0,
            expected_pass_attempts=0,
            expected_rushing_attempts=0,
            completion_probability=0,
            yards_per_completion=0,
            yards_before_contact=0,
            yards_after_contact=0,
        )
        assert project_team_points(zero_inputs) == 0.0

    def test_default_team_props_inputs(self):
        inputs = default_team_props_inputs()
        assert inputs.pace == 70.0
        assert inputs.expected_pass_attempts == 32.0
        assert inputs.expected_rushing_attempts == 36.0
        proj = project_team_production("TEAM_PROP", inputs)
        assert proj.projected_team_offensive_yards > 300.0


class TestCandidatePlayScoringComponents:
    """Proves 0-100 scoring components adhere to the architecture."""

    def test_hit_rate_scoring(self):
        assert score_hit_rate(0.85) == 25.0
        assert score_hit_rate(0.78) == 22.0
        assert score_hit_rate(0.72) == 19.0
        assert score_hit_rate(0.67) == 15.0
        assert score_hit_rate(0.62) == 10.0
        assert score_hit_rate(0.56) == 5.0
        assert score_hit_rate(0.50) == 0.0
        assert score_hit_rate(None) == 0.0

    def test_edge_scoring(self):
        # Yardage markets
        assert score_edge("team_offensive_yards", 40.0) == 20.0
        assert score_edge("team_offensive_yards", 22.0) == 16.0
        assert score_edge("team_rushing_yards", 12.0) == 12.0
        assert score_edge("team_receiving_yards", 4.0) == 7.0
        assert score_edge("team_offensive_yards", 1.0) == 4.0
        assert score_edge("team_offensive_yards", -5.0) == 0.0

        # Points markets
        assert score_edge("team_total_points", 8.0) == 20.0
        assert score_edge("team_total_points", 5.0) == 16.0
        assert score_edge("team_total_points", 3.0) == 12.0
        assert score_edge("team_total_points", 1.5) == 7.0
        assert score_edge("team_total_points", 0.5) == 4.0
        assert score_edge("team_total_points", -2.0) == 0.0

    def test_line_movement_scoring(self):
        assert score_line_movement("OVER", None, None) == 4.0
        assert score_line_movement("OVER", 42.5, 40.0) == 14.0  # moved down = favorable for OVER
        assert score_line_movement("OVER", 40.0, 39.5) == 10.0
        assert score_line_movement("OVER", 40.0, 40.0) == 4.0
        assert score_line_movement("OVER", 40.0, 41.0) == 2.0
        assert score_line_movement("OVER", 40.0, 43.0) == 1.0
        assert score_line_movement("OVER", 40.0, 40.0, rlm_flag=True) == 15.0

    def test_price_value_scoring(self):
        # Best price -105 (implied 0.5122) vs fair prob 0.57 -> price_edge = 0.0578
        assert score_price_value(-105, -115, 0.57) == 15.0
        # Best price -110 (implied 0.5238) vs fair prob 0.56 -> price_edge = 0.0362 >= 0.03
        assert score_price_value(-110, -110, 0.56) == 12.0
        # Best price -110 (implied 0.5238) vs fair prob 0.54 -> price_edge = 0.0162 >= 0.01
        assert score_price_value(-110, -110, 0.54) == 9.0
        assert score_price_value(None, None, None) == 0.0

    def test_sample_size_scoring(self):
        assert score_sample_size(60) == 10.0
        assert score_sample_size(35) == 8.0
        assert score_sample_size(25) == 6.0
        assert score_sample_size(15) == 4.0
        assert score_sample_size(7) == 2.0
        assert score_sample_size(2) == 0.0

    def test_confirming_insights_scoring(self):
        assert score_confirming_insights(5) == 15.0
        assert score_confirming_insights(3) == 12.0
        assert score_confirming_insights(2) == 8.0
        assert score_confirming_insights(1) == 4.0
        assert score_confirming_insights(0) == 0.0

    def test_play_tier_assignment(self):
        assert assign_play_tier(95.0) == "ELITE"
        assert assign_play_tier(85.0) == "STRONG"
        assert assign_play_tier(72.0) == "QUALIFIED"
        assert assign_play_tier(64.0) == "LEAN"
        assert assign_play_tier(52.0) == "PASS"


class TestCandidateEndToEndScoring:
    """Proves end-to-end candidate scoring connecting Model 3 and devigged fair odds."""

    def test_score_qualified_candidate(self, sample_inputs):
        # projected offensive yards = 409.5
        cand = TeamPropCandidate(
            game_id="g1",
            team_id="t1",
            market="team_offensive_yards",
            side="OVER",
            line=370.0,  # edge = +39.5 yards
            consensus_price=-110,
            best_price=-105,
            best_book="DraftKings",
            n_books=3,
            devigged_fair_prob=0.54,
            historical_hit_rate=0.78,  # hit_rate_score = 22.0
            sample_size=35,  # sample_size_score = 8.0
            confirming_insights_count=3,  # confirming_score = 12.0
            open_line=375.0,
            current_line=370.0,  # movement_score = 14.0
        )
        score = score_candidate(cand, sample_inputs)

        assert score.edge == 39.5
        assert score.projected_value == 409.5
        assert score.hit_rate_score == 22.0
        assert score.edge_score == 20.0
        assert score.movement_score == 14.0
        assert score.sample_size_score == 8.0
        assert score.confirming_score == 12.0
        assert score.play_score >= 80.0
        assert score.tier in ("STRONG", "ELITE")
        assert score.is_actionable is True

    def test_score_candidates_ranking(self, sample_inputs):
        cands = [
            TeamPropCandidate(
                game_id="g1",
                team_id="t1",
                market="team_offensive_yards",
                side="OVER",
                line=450.0,  # negative edge
                historical_hit_rate=0.50,
            ),
            TeamPropCandidate(
                game_id="g1",
                team_id="t1",
                market="team_rushing_yards",
                side="OVER",
                line=120.0,  # projected = 157.5, edge = +37.5
                historical_hit_rate=0.80,
                sample_size=40,
                confirming_insights_count=3,
            ),
        ]
        scores = score_candidates(cands, {"t1": sample_inputs})
        assert len(scores) == 2
        # Best play score ranked first
        assert scores[0].market == "team_rushing_yards"
        assert scores[0].play_score > scores[1].play_score


class TestDailyPipelineIntegration:
    """Proves candidate scoring is integrated directly into daily.py and DailyReport."""

    def test_score_daily_slates_empty_db(self, conn):
        res = score_daily_slates(conn, ["2026-09-05"], as_of_utc="2026-09-05T12:00:00Z")
        assert res == []

    def test_daily_scoring_is_opt_in(self):
        assert signature(run_daily).parameters["with_scoring"].default is False

    def test_score_daily_slates_no_market_data_reports_zero(self, conn, sample_inputs):
        """market_consensus never carries team-prop rows: build_market_for_slate only
        ever writes markets=("ML", "SPREAD", "TOTAL") (features/build_market.py), and
        Outlier's NCAAFB API does not offer team/player props at all (R1 discovery,
        docs/probes/2026-09-05-ncaafb-props-discovery.md). A slate with games but no
        team-prop consensus must report zero candidates, never fabricated ones."""
        store.upsert_team(conn, {"team_id": "h", "school": "Home", "alias": "H", "market": "H"})
        store.upsert_team(conn, {"team_id": "a", "school": "Away", "alias": "A", "market": "A"})
        store.upsert_game(
            conn,
            {
                "game_id": "g1",
                "season": 2026,
                "kickoff_utc": "2026-09-05T20:00:00+00:00",
                "football_date": "2026-09-05",
                "day_of_week": 5,
                "home_team_id": "h",
                "away_team_id": "a",
                "venue_name": "Stadium",
                "network": "ESPN",
                "status": "pregame",
            },
        )

        scores = score_daily_slates(conn, ["2026-09-05"], as_of_utc="2026-09-05T12:00:00Z")
        assert scores == []

    def test_score_daily_slates_scores_real_consensus_rows(self, conn, sample_inputs):
        """When a real team-prop consensus row does exist, it must be scored --
        this is the counterpart to the zero-fabrication test above."""
        store.upsert_team(conn, {"team_id": "h", "school": "Home", "alias": "H", "market": "H"})
        store.upsert_team(conn, {"team_id": "a", "school": "Away", "alias": "A", "market": "A"})
        store.upsert_game(
            conn,
            {
                "game_id": "g1",
                "season": 2026,
                "kickoff_utc": "2026-09-05T20:00:00+00:00",
                "football_date": "2026-09-05",
                "day_of_week": 5,
                "home_team_id": "h",
                "away_team_id": "a",
                "venue_name": "Stadium",
                "network": "ESPN",
                "status": "pregame",
            },
        )
        conn.execute(
            """INSERT INTO team_prop_consensus
               (game_id, team_id, market, line, side, as_of_utc, n_books,
                consensus_price, best_price, best_book, hold,
                prob_multiplicative, prob_shin)
               VALUES ('g1', 'h', 'team_total_points', 24.5, 'OVER',
                       '2026-09-05T12:00:00+00:00', 3, -110, -105,
                       'FanDuel', 0.045, 0.52, 0.52)"""
        )

        scores = score_daily_slates(
            conn,
            ["2026-09-05"],
            inputs_by_team={"h": sample_inputs},
            as_of_utc="2026-09-05T12:30:00Z",
        )
        assert len(scores) > 0
        assert all(is_supported_team_prop(s.market) for s in scores)
        assert all(not is_player_prop(s.market) for s in scores)

    def test_run_daily_wires_scoring(self, conn, monkeypatch):
        """run_daily calls into scoring and reports the outcome even when there is
        no team-prop market data to score -- 0 is a valid, truthful result, not a
        failure. See test_score_daily_slates_with_games_but_no_market_data_reports_zero
        for why an empty slate must not fabricate candidates."""
        monkeypatch.setattr("cfb_analytics.config.has_cfbd_key", lambda: False)
        store.upsert_team(conn, {"team_id": "h", "school": "Home", "alias": "H", "market": "H"})
        store.upsert_team(conn, {"team_id": "a", "school": "Away", "alias": "A", "market": "A"})
        store.upsert_game(
            conn,
            {
                "game_id": "g1",
                "season": 2026,
                "kickoff_utc": "2026-09-05T20:00:00+00:00",
                "football_date": "2026-09-05",
                "day_of_week": 5,
                "home_team_id": "h",
                "away_team_id": "a",
                "venue_name": "Stadium",
                "network": "ESPN",
                "status": "pregame",
            },
        )

        report = run_daily(
            conn,
            with_outlier=False,
            with_weather=False,
            with_player_passing=False,
            with_internal_ratings=False,
            with_internal_elo=False,
            with_scoring=True,
            now=datetime(2026, 9, 5, 12, tzinfo=FOOTBALL_TZ),
        )

        assert report.candidates_scored == 0
        assert report.candidate_scores == []
        scoring_outcome = next(o for o in report.outcomes if o.name == "scoring")
        assert scoring_outcome.status == "ok"
        assert "0 team prop candidates evaluated" in scoring_outcome.detail
        text = report.as_text()
        assert "candidates scored:" in text


class TestMinimumQualificationGates:
    """Proves candidates cannot qualify if edge is non-positive or prices are missing."""

    def test_non_positive_edge_cannot_qualify(self, sample_inputs):
        # High hit rate, movement, price value, sample-size, and confirmation scores.
        # Total play score without edge would be 79.0 >= 70, but edge is <= 0
        cand = TeamPropCandidate(
            game_id="g1",
            team_id="t1",
            market="team_offensive_yards",
            side="OVER",
            line=450.0,  # projected is 409.5, edge = -40.5
            consensus_price=-110,
            best_price=-105,
            devigged_fair_prob=0.55,
            historical_hit_rate=0.85,
            sample_size=60,
            confirming_insights_count=4,
            open_line=455.0,
            current_line=450.0,
        )
        score = score_candidate(cand, sample_inputs)
        assert score.edge < 0.0
        assert score.tier == "PASS"
        assert score.is_actionable is False

    def test_missing_prices_not_actionable(self, sample_inputs):
        cand = TeamPropCandidate(
            game_id="g1",
            team_id="t1",
            market="team_offensive_yards",
            side="OVER",
            line=350.0,  # projected is 409.5, edge = +59.5
            consensus_price=None,
            best_price=None,
            historical_hit_rate=0.85,
            sample_size=60,
            confirming_insights_count=4,
        )
        score = score_candidate(cand, sample_inputs)
        assert score.edge > 0.0
        assert score.play_score >= 70.0
        # Price is unavailable -> not actionable per Minimum Qualification Gates
        assert score.is_actionable is False


class TestDeviggingIntegration:
    """Proves devigging from cfb_analytics.models.devig is directly wired into scoring."""

    def test_two_sided_consensus_price_devig(self, sample_inputs):
        # OVER -115, UNDER -105 -> Shin devig fair prob for OVER should be ~0.509
        cand = TeamPropCandidate(
            game_id="g1",
            team_id="t1",
            market="team_total_points",
            side="OVER",
            line=28.5,
            consensus_price=-115,
            opposite_consensus_price=-105,
            best_price=-110,
            n_books=3,
        )
        fair = resolve_devigged_fair_prob(cand, "OVER")
        assert fair is not None
        assert 0.50 <= fair <= 0.52

    def test_multi_book_quotes_devig(self, sample_inputs):
        # Three books with two-sided quotes
        cand = TeamPropCandidate(
            game_id="g1",
            team_id="t1",
            market="team_total_points",
            side="OVER",
            line=28.5,
            book_quotes={
                "DraftKings": [-115, -105],
                "FanDuel": [-110, -110],
                "Caesars": [-112, -108],
            },
            n_books=3,
        )
        fair = resolve_devigged_fair_prob(cand, "OVER")
        assert fair is not None
        assert 0.50 <= fair <= 0.53

    def test_one_sided_price_does_not_fabricate_fair_probability(self, sample_inputs):
        cand = TeamPropCandidate(
            game_id="g1",
            team_id="t1",
            market="team_total_points",
            side="OVER",
            line=28.5,
            consensus_price=-110,
            hold=0.045,
            n_books=3,
        )
        fair = resolve_devigged_fair_prob(cand, "OVER")
        assert fair is None


class TestDatabaseDataWiring:
    """Proves database statistics populate real team-specific TeamPropsInputs."""

    def test_build_team_props_inputs_from_db(self, conn):
        # Insert prerequisite teams and game for foreign key constraints
        store.upsert_team(
            conn, {"team_id": "team_sec", "school": "SEC Team", "alias": "SEC", "market": "SEC"}
        )
        store.upsert_team(
            conn, {"team_id": "team_opp", "school": "Opp Team", "alias": "OPP", "market": "OPP"}
        )
        store.upsert_game(
            conn,
            {
                "game_id": "g1",
                "season": 2026,
                "kickoff_utc": "2026-09-01T20:00:00+00:00",
                "football_date": "2026-09-01",
                "day_of_week": 1,
                "home_team_id": "team_sec",
                "away_team_id": "team_opp",
                "venue_name": "Stadium",
                "network": "ESPN",
                "status": "final",
            },
        )

        # Insert advanced stats and player passing stats
        conn.execute(
            """INSERT INTO team_season_advanced
               (snapshot_id, season, week, team_id, side, as_of_utc, ingested_utc,
                garbage_excluded, success_rate, explosiveness, line_yards, plays, drives)
               VALUES ('snap1', 2026, 1, 'team_sec', 'off', '2026-09-01T00:00:00+00:00',
                       '2026-09-01T00:00:00+00:00', 1, 0.48, 1.45, 3.2, 75, 11)"""
        )
        conn.execute(
            """INSERT INTO players (player_id, name, first_seen_utc, last_seen_utc)
               VALUES ('p1', 'QB1', '2026-01-01', '2026-09-01')"""
        )
        conn.execute(
            """INSERT INTO player_game_passing
               (game_id, team_id, player_id, season, week, completions, attempts,
                yards, source, ingested_utc)
               VALUES ('g1', 'team_sec', 'p1', 2026, 1, 24, 32, 310, 'cfbd', '2026-09-01')"""
        )

        inputs = build_team_props_inputs_from_db(
            conn,
            "team_sec",
            season=2026,
            as_of_utc="2026-09-02T00:00:00Z",
        )
        assert inputs.offensive_success_rate == 0.48
        assert inputs.explosiveness == 1.45
        assert inputs.yards_before_contact == 3.2
        assert inputs.completion_probability == 0.75  # 24 / 32
        assert inputs.yards_per_completion == pytest.approx(12.92, rel=0.01)

    def test_build_team_props_inputs_unpopulated_fallback(self, conn):
        inputs = build_team_props_inputs_from_db(
            conn,
            "unknown_team",
            season=2026,
            as_of_utc="2026-09-02T00:00:00Z",
        )
        baseline = default_team_props_inputs()
        assert inputs.pace == baseline.pace
        assert inputs.offensive_success_rate == baseline.offensive_success_rate
        assert inputs.explosiveness == baseline.explosiveness
        assert inputs.completion_probability == baseline.completion_probability

        assert build_team_props_inputs_from_db(conn, "") == baseline

    def test_load_team_props_inputs_for_slate(self, conn):
        store.upsert_team(conn, {"team_id": "t1", "school": "Team1", "alias": "T1", "market": "T1"})
        store.upsert_team(conn, {"team_id": "t2", "school": "Team2", "alias": "T2", "market": "T2"})
        store.upsert_game(
            conn,
            {
                "game_id": "g1",
                "season": 2026,
                "kickoff_utc": "2026-09-05T20:00:00+00:00",
                "football_date": "2026-09-05",
                "day_of_week": 5,
                "home_team_id": "t1",
                "away_team_id": "t2",
                "venue_name": "Stadium",
                "network": "ESPN",
                "status": "pregame",
            },
        )

        slate_inputs = load_team_props_inputs_for_slate(
            conn,
            "2026-09-05",
            season=2026,
            as_of_utc="2026-09-05T12:00:00Z",
        )
        assert "t1" in slate_inputs
        assert "t2" in slate_inputs
        assert slate_inputs["t1"].pace > 0
        assert slate_inputs["t2"].pace > 0

    def test_score_slate_team_props_home_away_attribution(self, conn):
        store.upsert_team(
            conn, {"team_id": "home_team", "school": "Home", "alias": "HT", "market": "HT"}
        )
        store.upsert_team(
            conn, {"team_id": "away_team", "school": "Away", "alias": "AT", "market": "AT"}
        )
        store.upsert_game(
            conn,
            {
                "game_id": "g1",
                "season": 2026,
                "kickoff_utc": "2026-09-05T20:00:00+00:00",
                "football_date": "2026-09-05",
                "day_of_week": 5,
                "home_team_id": "home_team",
                "away_team_id": "away_team",
                "venue_name": "Stadium",
                "network": "ESPN",
                "status": "pregame",
            },
        )

        # Insert consensus rows for away team total points OVER
        conn.execute(
            """INSERT INTO team_prop_consensus
               (game_id, team_id, market, line, side, as_of_utc, n_books,
                consensus_price, best_price, best_book, hold,
                prob_multiplicative, prob_shin)
               VALUES ('g1', 'away_team', 'team_total_points', 24.5, 'OVER',
                       '2026-09-05T12:00:00+00:00', 3, -110, -105,
                       'FanDuel', 0.045, 0.52, 0.52)"""
        )

        scores = score_slate_team_props(
            conn,
            "2026-09-05",
            inputs_by_team={"away_team": default_team_props_inputs(data_quality_score=100)},
            as_of_utc="2026-09-05T12:30:00Z",
        )
        assert len(scores) > 0
        away_score = next((s for s in scores if s.team_id == "away_team"), None)
        assert away_score is not None
        assert away_score.side == "OVER"
        assert away_score.line == 24.5


class TestPointInTimeAndQualificationRegressions:
    def test_passing_inputs_exclude_games_after_cutoff(self, conn):
        for team_id in ("team", "opp"):
            store.upsert_team(
                conn,
                {"team_id": team_id, "school": team_id, "alias": team_id, "market": team_id},
            )
        for game_id, kickoff, week in (
            ("past", "2026-09-01T20:00:00+00:00", 1),
            ("future", "2026-09-15T20:00:00+00:00", 3),
        ):
            store.upsert_game(
                conn,
                {
                    "game_id": game_id,
                    "season": 2026,
                    "week": week,
                    "kickoff_utc": kickoff,
                    "football_date": kickoff[:10],
                    "day_of_week": 1,
                    "home_team_id": "team",
                    "away_team_id": "opp",
                    "venue_name": "Stadium",
                    "network": "ESPN",
                    "status": "final",
                },
            )
        conn.executemany(
            """INSERT INTO players (player_id, name, first_seen_utc, last_seen_utc)
               VALUES (?, ?, '2026-01-01', '2026-12-01')""",
            (("past_qb", "Past QB"), ("future_qb", "Future QB")),
        )
        conn.executemany(
            """INSERT INTO player_game_passing
               (game_id, team_id, player_id, season, week, completions, attempts,
                yards, source, ingested_utc)
               VALUES (?, 'team', ?, 2026, ?, ?, ?, ?, 'cfbd', '2026-09-20')""",
            (
                ("past", "past_qb", 1, 10, 20, 100),
                ("future", "future_qb", 3, 30, 30, 600),
            ),
        )

        inputs = build_team_props_inputs_from_db(
            conn,
            "team",
            season=2026,
            as_of_utc="2026-09-10T00:00:00Z",
        )

        assert inputs.completion_probability == 0.5
        assert inputs.yards_per_completion == 10.0
        assert inputs.expected_pass_attempts == 20.0

    def test_scoring_selects_latest_snapshot_before_cutoff(self, conn, sample_inputs):
        for team_id in ("home", "away"):
            store.upsert_team(
                conn,
                {"team_id": team_id, "school": team_id, "alias": team_id, "market": team_id},
            )
        store.upsert_game(
            conn,
            {
                "game_id": "g1",
                "season": 2026,
                "kickoff_utc": "2026-09-20T20:00:00+00:00",
                "football_date": "2026-09-20",
                "day_of_week": 6,
                "home_team_id": "home",
                "away_team_id": "away",
                "venue_name": "Stadium",
                "network": "ESPN",
                "status": "pregame",
            },
        )
        conn.executemany(
            """INSERT INTO team_prop_consensus
               (game_id, team_id, market, line, side, as_of_utc, n_books,
                consensus_price, best_price, best_book, prob_multiplicative, prob_shin)
               VALUES ('g1', 'away', 'team_total_points', 24.5, 'OVER', ?,
                       3, -110, ?, 'Book', ?, ?)""",
            (
                ("2026-09-05T12:00:00+00:00", -110, 0.50, 0.50),
                ("2026-09-10T12:00:00+00:00", -105, 0.53, 0.53),
                ("2026-09-15T12:00:00+00:00", 100, 0.56, 0.56),
            ),
        )

        scores = score_slate_team_props(
            conn,
            "2026-09-20",
            inputs_by_team={"away": sample_inputs},
            as_of_utc="2026-09-12T00:00:00Z",
        )

        assert len(scores) == 1
        assert scores[0].team_id == "away"
        assert scores[0].as_of_utc == "2026-09-10T12:00:00+00:00"
        assert scores[0].best_price == -105

    def test_thin_market_and_missing_fair_probability_fail_closed(self, sample_inputs):
        candidate = TeamPropCandidate(
            game_id="g1",
            team_id="team",
            market="team_offensive_yards",
            side="OVER",
            line=350.0,
            best_price=-105,
            n_books=1,
            historical_hit_rate=0.85,
            sample_size=60,
            confirming_insights_count=4,
        )

        score = score_candidate(candidate, sample_inputs)

        assert score.tier == "PASS"
        assert score.qualification_status == "INSUFFICIENT_DATA"
        assert score.devigged_fair_prob is None
        assert score.is_actionable is False

    def test_negative_model_ev_cannot_qualify(self, sample_inputs):
        candidate = TeamPropCandidate(
            game_id="g1",
            team_id="team",
            market="team_offensive_yards",
            side="OVER",
            line=400.0,
            consensus_price=-1000,
            best_price=-1000,
            n_books=3,
            devigged_fair_prob=0.91,
            historical_hit_rate=0.85,
            sample_size=60,
            confirming_insights_count=4,
        )

        score = score_candidate(candidate, sample_inputs)

        assert score.edge > 0
        assert score.expected_value is not None and score.expected_value < 0
        assert score.tier == "PASS"
        assert score.qualification_status == "PASS"
        assert score.is_actionable is False
