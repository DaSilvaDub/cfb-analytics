"""Tests for cfb_analytics.features.mispriced – Mispriced Line Scanner."""

from __future__ import annotations

import pytest

from cfb_analytics.features.mispriced import (
    MispricedCandidate,
    _assign_edge_tier,
    _model_prob_from_edge,
    build_mispriced_board,
    render_mispriced_board,
)
from cfb_analytics.ingest import store


# ── helpers ──────────────────────────────────────────────────────────────────


def _seed_teams(conn) -> tuple[str, str]:
    """Create a pair of teams and return (home_id, away_id)."""
    store.upsert_team(conn, {"team_id": "h", "school": "Home U", "alias": "HOME", "market": "H"})
    store.upsert_team(conn, {"team_id": "a", "school": "Away U", "alias": "AWAY", "market": "A"})
    return "h", "a"


def _seed_game(conn, home: str, away: str, *, game_id: str = "g1") -> str:
    store.upsert_game(
        conn,
        {
            "game_id": game_id,
            "season": 2026,
            "kickoff_utc": "2026-09-05T23:30:00+00:00",
            "football_date": "2026-09-05",
            "day_of_week": 5,
            "home_team_id": home,
            "away_team_id": away,
            "venue_name": "Stadium",
            "network": "ESPN",
            "status": "pregame",
        },
    )
    return game_id


def _insert_spread_consensus(
    conn,
    game_id: str,
    home_line: float,
    *,
    prob_shin: float = 0.52,
    n_books: int = 5,
) -> None:
    """Insert HOME and AWAY spread consensus rows."""
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'SPREAD', ?, 'HOME', '2026-09-05T12:00:00+00:00', ?, -110,
                   -110, 'DK', 0.04, 'all_books', ?, ?, ?, 0.01, '[]')""",
        (game_id, home_line, n_books, prob_shin, prob_shin, prob_shin),
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'SPREAD', ?, 'AWAY', '2026-09-05T12:00:00+00:00', ?, -110,
                   -110, 'DK', 0.04, 'all_books', ?, ?, ?, 0.01, '[]')""",
        (game_id, -home_line, n_books, 1.0 - prob_shin, 1.0 - prob_shin, 1.0 - prob_shin),
    )


def _insert_total_consensus(
    conn,
    game_id: str,
    total_line: float,
    *,
    prob_shin_over: float = 0.50,
    n_books: int = 5,
) -> None:
    """Insert OVER and UNDER total consensus rows."""
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'TOTAL', ?, 'OVER', '2026-09-05T12:00:00+00:00', ?, -110,
                   -110, 'FD', 0.04, 'all_books', ?, ?, ?, 0.01, '[]')""",
        (game_id, total_line, n_books, prob_shin_over, prob_shin_over, prob_shin_over),
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'TOTAL', ?, 'UNDER', '2026-09-05T12:00:00+00:00', ?, -110,
                   -110, 'FD', 0.04, 'all_books', ?, ?, ?, 0.01, '[]')""",
        (
            game_id,
            total_line,
            n_books,
            1.0 - prob_shin_over,
            1.0 - prob_shin_over,
            1.0 - prob_shin_over,
        ),
    )


def _insert_ridge_ratings(
    conn,
    home_id: str,
    away_id: str,
    home_off: float,
    home_def: float,
    away_off: float,
    away_def: float,
) -> None:
    """Insert Ridge power ratings so the model can project spreads."""
    for team_id, off, defe in [(home_id, home_off, home_def), (away_id, away_off, away_def)]:
        conn.execute(
            """INSERT INTO internal_team_ratings
               (season, as_of_utc, team_id, model, offense, defense,
                team_games, ridge_lambda, home_field_advantage,
                n_games_in_fit, generated_utc)
               VALUES (2026, '2026-09-04T00:00:00+00:00', ?, 'internal_ridge',
                       ?, ?, 3, 100.0, 3.0, 50, '2026-09-04T00:00:00+00:00')""",
            (team_id, off, defe),
        )


# ── pure function tests ─────────────────────────────────────────────────────


class TestModelProbFromEdge:
    def test_projected_equals_line_returns_fifty_pct(self):
        p = _model_prob_from_edge(45.0, 45.0, sigma=8.0)
        assert p == 0.50

    def test_projected_above_line_returns_above_fifty(self):
        p = _model_prob_from_edge(50.0, 45.0, sigma=8.0)
        assert p > 0.50

    def test_projected_below_line_returns_below_fifty(self):
        p = _model_prob_from_edge(40.0, 45.0, sigma=8.0)
        assert p < 0.50

    def test_clamps_to_01_and_99(self):
        p_hi = _model_prob_from_edge(100.0, 0.0, sigma=1.0)
        p_lo = _model_prob_from_edge(0.0, 100.0, sigma=1.0)
        assert p_hi == 0.99
        assert p_lo == 0.01

    def test_larger_sigma_reduces_confidence(self):
        tight = _model_prob_from_edge(50.0, 45.0, sigma=5.0)
        wide = _model_prob_from_edge(50.0, 45.0, sigma=20.0)
        assert tight > wide


class TestAssignEdgeTier:
    def test_strong_tier(self):
        assert _assign_edge_tier(0.07) == "STRONG"

    def test_moderate_tier(self):
        assert _assign_edge_tier(0.04) == "MODERATE"

    def test_marginal_tier(self):
        assert _assign_edge_tier(0.02) == "MARGINAL"

    def test_boundary_strong(self):
        assert _assign_edge_tier(0.06) == "STRONG"

    def test_boundary_moderate(self):
        assert _assign_edge_tier(0.03) == "MODERATE"


# ── DB-backed integration tests ─────────────────────────────────────────────


class TestBuildMispricedBoard:
    def test_empty_db_returns_empty(self, conn):
        result = build_mispriced_board(conn, "2026-09-05")
        assert result == []

    def test_no_consensus_returns_empty(self, conn):
        home, away = _seed_teams(conn)
        _seed_game(conn, home, away)
        conn.commit()
        result = build_mispriced_board(conn, "2026-09-05")
        assert result == []

    def test_no_model_ratings_returns_empty(self, conn):
        """Games + consensus but no model ratings → no candidates."""
        home, away = _seed_teams(conn)
        _seed_game(conn, home, away)
        _insert_spread_consensus(conn, "g1", -7.0)
        conn.commit()
        result = build_mispriced_board(conn, "2026-09-05")
        assert result == []

    def test_spread_mispricing_detected(self, conn):
        """When model projects a large discrepancy vs consensus, a candidate appears."""
        home, away = _seed_teams(conn)
        _seed_game(conn, home, away)
        # Consensus: Home -7.0, prob_shin HOME = 0.52
        _insert_spread_consensus(conn, "g1", -7.0, prob_shin=0.52)
        # Ridge says Home is much stronger (offense 30, defense 5 = 35 power)
        # vs Away (offense 10, defense -5 = 5 power). Diff=30 + HFA → big edge
        _insert_ridge_ratings(conn, home, away, 30.0, 5.0, 10.0, -5.0)
        conn.commit()

        result = build_mispriced_board(conn, "g1"[0:0] or "2026-09-05", min_edge=0.01)
        # Should find at least one side with edge >= 1%
        assert len(result) >= 1
        assert all(isinstance(c, MispricedCandidate) for c in result)
        assert all(c.market == "SPREAD" for c in result)

    def test_candidates_sorted_by_descending_edge(self, conn):
        """Multiple candidates are ranked by absolute edge, highest first."""
        home, away = _seed_teams(conn)
        _seed_game(conn, home, away)
        _insert_spread_consensus(conn, "g1", -7.0, prob_shin=0.52)
        _insert_ridge_ratings(conn, home, away, 30.0, 5.0, 10.0, -5.0)
        conn.commit()

        result = build_mispriced_board(conn, "2026-09-05", min_edge=0.01)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert abs(result[i].edge_pct) >= abs(result[i + 1].edge_pct)

    def test_ranks_assigned_sequentially(self, conn):
        home, away = _seed_teams(conn)
        _seed_game(conn, home, away)
        _insert_spread_consensus(conn, "g1", -7.0, prob_shin=0.52)
        _insert_ridge_ratings(conn, home, away, 30.0, 5.0, 10.0, -5.0)
        conn.commit()

        result = build_mispriced_board(conn, "2026-09-05", min_edge=0.01)
        for i, c in enumerate(result, 1):
            assert c.rank == i

    def test_min_edge_filters_small_edges(self, conn):
        """Raising min_edge eliminates marginal candidates."""
        home, away = _seed_teams(conn)
        _seed_game(conn, home, away)
        # Small edge scenario: consensus close to model
        _insert_spread_consensus(conn, "g1", -3.0, prob_shin=0.52)
        _insert_ridge_ratings(conn, home, away, 15.0, 0.0, 10.0, 0.0)
        conn.commit()

        result_loose = build_mispriced_board(conn, "2026-09-05", min_edge=0.001)
        result_tight = build_mispriced_board(conn, "2026-09-05", min_edge=0.50)
        assert len(result_tight) <= len(result_loose)


class TestRenderMispricedBoard:
    def test_empty_candidates_returns_message(self):
        output = render_mispriced_board("2026-09-05", [])
        assert "No mispriced opportunities" in output
        assert "2026-09-05" in output

    def test_renders_candidates(self):
        candidates = [
            MispricedCandidate(
                rank=1,
                game_id="g1",
                game_label="Away U at Home U",
                kickoff_et="7:30 p.m.",
                market="SPREAD",
                side="HOME",
                line=-7.0,
                model_projected=-10.5,
                model_prob=0.62,
                consensus_fair_prob=0.52,
                edge_pct=0.10,
                method_spread=0.01,
                best_price=-110,
                best_book="DK",
                n_books=5,
                hold=0.04,
                anchor="all_books",
                tier="STRONG",
                flags=(),
            ),
        ]
        output = render_mispriced_board("2026-09-05", candidates)
        assert "MISPRICED BOARD" in output
        assert "Away U at Home U" in output
        assert "SPREAD" in output
        assert "HOME" in output
        assert "STRONG" in output
        assert "1 mispriced opportunities detected" in output

    def test_renders_multiple_candidates_with_total(self):
        candidates = [
            MispricedCandidate(
                rank=i,
                game_id=f"g{i}",
                game_label=f"Team{i} at Host{i}",
                kickoff_et="3:30 p.m.",
                market="SPREAD" if i == 1 else "TOTAL",
                side="HOME" if i == 1 else "OVER",
                line=-7.0 if i == 1 else 52.5,
                model_projected=-10.0 if i == 1 else 56.0,
                model_prob=0.60,
                consensus_fair_prob=0.50,
                edge_pct=0.10 - (i * 0.02),
                method_spread=0.01,
                best_price=-110,
                best_book="FD",
                n_books=4,
                hold=0.04,
                anchor="all_books",
                tier="STRONG" if i == 1 else "MODERATE",
                flags=(),
            )
            for i in range(1, 4)
        ]
        output = render_mispriced_board("2026-09-05", candidates)
        assert "3 mispriced opportunities detected" in output

    def test_candidate_fields_frozen(self):
        c = MispricedCandidate(
            rank=1,
            game_id="g1",
            game_label="A at B",
            kickoff_et="12:00 p.m.",
            market="SPREAD",
            side="HOME",
            line=-3.0,
            model_projected=-5.0,
            model_prob=0.55,
            consensus_fair_prob=0.50,
            edge_pct=0.05,
            method_spread=0.01,
            best_price=-110,
            best_book="DK",
            n_books=3,
            hold=0.04,
            anchor="all_books",
            tier="MODERATE",
            flags=(),
        )
        with pytest.raises(AttributeError):
            c.rank = 2  # type: ignore[misc]
