"""Comprehensive tests for CLI reporting integration and card rendering (Milestone 4).

Verifies:
1. Terminal rendering of 7-dimensional reasoning cards per docs/grok_rules.md §5:
   - 1. Tape Evaluation: Honest vs Junk
   - 2. Position & QB Edge Breakdown
   - 3. Weather, Venue & Travel Factors
   - 4. Injuries & Trench Health
   - 5. Program Structure & Situational Spot
   - 6. Mathematical Edge vs Market Consensus
   - 7. What NOT to Bet (Contra-Indications)
2. Terminal rendering of mispriced opportunity cards and parlay ticket cards.
3. Clean JSON serialization for reasoning cards, mispriced opportunities, and parlay tickets.
4. CLI subcommands:
   - `cfb-analytics board --with-reasoning --date <date>` (terminal formatting).
   - `cfb-analytics board --with-reasoning --date <date> --json` (JSON structure, cards, watermark).
   - `cfb-analytics mispriced --date <date>` (terminal formatting).
   - `cfb-analytics mispriced --date <date> --json` (JSON structure, edge %, method spread, disclaimer).
5. Error handling:
   - Database does not exist (exit code 1, helpful message).
   - Empty slate / no consensus for date (exit code 0, informative message).
6. Invariant enforcement:
   - Mandatory Shadow Mode disclaimer watermark ("UNPROMOTED - shadow output, not decision-grade")
     on all terminal cards, CLI output banners, and JSON payloads.
   - Zero unused imports (ruff F401 compliant).
   - Pure Python standard library only.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest

from cfb_analytics import cli, paths
from cfb_analytics.governance.models import (
    SHADOW_MODE_DISCLAIMER,
    GovernanceAction,
    GovernanceVerdict,
    MarginalLegAnalysis,
    ParlayLeg,
    ParlayTicket,
)
from cfb_analytics.ingest import store
from cfb_analytics.reasoning.context import (
    load_situational_context_for_game,
    load_slate_situational_contexts,
)
from cfb_analytics.reasoning.models import (
    ReasoningCard,
)
from cfb_analytics.reporting.export import (
    export_mispriced_json,
    export_parlay_json,
    export_reasoning_json,
)
from cfb_analytics.reporting.terminal import (
    render_mispriced_card,
    render_parlay_card,
    render_reasoning_card,
)
from cfb_analytics.scanner.models import (
    MispricedOpportunity,
    PlayTier,
    QualificationStatus,
)


# =============================================================================
# Test Fixtures & Helpers
# =============================================================================


@pytest.fixture
def cli_db():
    """Initializes the SQLite database at paths.database_path() with all migrations."""
    cli.main(["init-db"])
    from cfb_analytics import db

    with db.open_db() as connection:
        yield connection


@pytest.fixture
def sample_reasoning_card() -> ReasoningCard:
    """Authentic sample reasoning card with all 6 dimensions populated."""
    return ReasoningCard(
        game_id="cfbd:401520182",
        market="ML",
        side="HOME",
        recommended_play="HOME ML (-150)",
        confidence=8.2,
        tier="STRONG",
        tape_summary="Honest FBS tape confirms 0.24 EPA/play floor against top-50 defenses.",
        position_qb_summary="QB continuity confirmed (3rd year starter); +1.8 trench matchup advantage.",
        weather_venue_summary="Kickoff 72F, wind 8 kph, clear conditions. Dome bypass active.",
        injuries_trench_summary="All 5 starting OL healthy; opponent starting DE out with ankle injury.",
        program_continuity_summary="4th year HC/OC scheme stability; 8 days rest coming off home win.",
        mathematical_edge_summary="Model prob 65.2% vs market consensus 58.0% (+7.2% edge, Shin devig).",
        contra_indications=[
            "Do not lay heavy spread points (-14.5) due to conservative run-heavy 4Q script."
        ],
        is_favorite_vetoed=False,
        counter_thesis=None,
        matchup_label="Vanderbilt at Georgia",
        kickoff_et="3:30 p.m. ET",
        line=0.0,
        price_american=-150,
        rule_triggers=("RULE_A",),
        shadow_mode_disclaimer=SHADOW_MODE_DISCLAIMER,
    )


@pytest.fixture
def sample_vetoed_reasoning_card() -> ReasoningCard:
    """Sample favorite reasoning card vetoed by Negative Gate or Grok Governance."""
    return ReasoningCard(
        game_id="cfbd:401520199",
        market="SPREAD",
        side="HOME",
        recommended_play="HOME -3.5",
        confidence=4.1,
        tier="AVOID",
        tape_summary="Tape heavily inflated by 56-0 FCS cupcake blowout. EPA against FBS is -0.05.",
        position_qb_summary="Starting QB listed as questionable with shoulder sprain; backup has 12 career attempts.",
        weather_venue_summary="24 kph sustained winds with gusts to 38 kph; passing efficiency degraded 22%.",
        injuries_trench_summary="Left Tackle and Center scratched; pass protection compromised.",
        program_continuity_summary="First-year coordinator in hostile rivalry road-like lookahead spot.",
        mathematical_edge_summary="Negative edge: model projection +0.5 vs market line -3.5.",
        contra_indications=["Thin dog trap candidate", "Starting QB unconfirmed"],
        is_favorite_vetoed=True,
        counter_thesis="VETO: Rule D thin favorite trap. Starting QB questionable, backup unproven, trench health compromised.",
        matchup_label="Auburn at LSU",
        kickoff_et="7:00 p.m. ET",
        line=-3.5,
        price_american=-110,
        rule_triggers=("RULE_D", "NEGATIVE_GATE"),
        shadow_mode_disclaimer=SHADOW_MODE_DISCLAIMER,
    )


@pytest.fixture
def sample_governance_verdict() -> GovernanceVerdict:
    return GovernanceVerdict(
        candidate_id="cfbd:401520182:ML:HOME",
        action=GovernanceAction.APPROVE,
        triggered_rules=("RULE_A",),
        counter_theses=(),
        original_confidence=8.2,
        adjusted_confidence=8.2,
        parlay_eligible=True,
        target_market_override=None,
        notes="Approved via Rule A Home Power Smash Script.",
        disclaimer=SHADOW_MODE_DISCLAIMER,
    )


@pytest.fixture
def sample_veto_verdict() -> GovernanceVerdict:
    return GovernanceVerdict(
        candidate_id="cfbd:401520199:SPREAD:HOME",
        action=GovernanceAction.VETO,
        triggered_rules=("RULE_D", "NEGATIVE_GATE"),
        counter_theses=(
            "VETO: Rule D thin favorite trap. Starting QB questionable, backup unproven, trench health compromised.",
        ),
        original_confidence=4.1,
        adjusted_confidence=2.0,
        parlay_eligible=False,
        target_market_override=None,
        notes="Strict veto applied.",
        disclaimer=SHADOW_MODE_DISCLAIMER,
    )


@pytest.fixture
def sample_mispriced_opportunity() -> MispricedOpportunity:
    return MispricedOpportunity(
        game_id="cfbd:401520182",
        market_type="SPREAD",
        market="SPREAD",
        side="HOME",
        line=-14.0,
        posted_price_american=-110,
        posted_price_decimal=1.9091,
        consensus_fair_prob=0.5200,
        consensus_fair_price_american=-108,
        model_projected_line=-18.5,
        model_prob=0.5750,
        edge_pct=0.0550,
        ev=0.0977,
        method_spread=0.0042,
        n_books=5,
        play_score=82.5,
        play_tier=PlayTier.STRONG.value,
        qual_status=QualificationStatus.QUALIFIED.value,
        best_book="DraftKings",
        game_label="Vanderbilt at Georgia",
        kickoff_et="3:30 p.m. ET",
        flags=("CONSENSUS_SOLID",),
        rejection_reasons=(),
        shadow_mode_disclaimer=SHADOW_MODE_DISCLAIMER,
    )


@pytest.fixture
def sample_parlay_ticket() -> ParlayTicket:
    leg1 = ParlayLeg(
        candidate_id="cfbd:101:ML:HOME",
        game_id="cfbd:101",
        team="Georgia",
        opponent="Vanderbilt",
        odds_american=-300,
        fair_prob=0.76,
        edge_pct=0.04,
        conference="SEC",
    )
    leg2 = ParlayLeg(
        candidate_id="cfbd:102:ML:AWAY",
        game_id="cfbd:102",
        team="Ohio State",
        opponent="Indiana",
        odds_american=-220,
        fair_prob=0.70,
        edge_pct=0.03,
        conference="Big Ten",
    )
    leg3 = ParlayLeg(
        candidate_id="cfbd:103:ML:HOME",
        game_id="cfbd:103",
        team="Oregon",
        opponent="UCLA",
        odds_american=-180,
        fair_prob=0.66,
        edge_pct=0.05,
        conference="Big Ten",
    )
    analysis1 = MarginalLegAnalysis(
        leg_index=0,
        candidate_id="cfbd:101:ML:HOME",
        team="Georgia",
        prob_before=0.462,
        prob_after=0.351,
        payout_before=2.25,
        payout_after=3.00,
        marginal_ev=0.045,
        delta_fragility=0.08,
        is_parasitic=False,
    )
    return ParlayTicket(
        parlay_id="parlay:2026-09-05:3leg:01",
        legs=(leg1, leg2, leg3),
        leg_count=3,
        total_odds_american=+200,
        total_payout_multiplier=3.00,
        joint_win_prob=0.351,
        raw_win_prob=0.351,
        correlation_penalty=0.0,
        ev=0.053,
        fragility_index=0.28,
        efficiency_ratio=0.189,
        marginal_analyses=(analysis1,),
        alternate_pruned_ticket=None,
        disclaimer=SHADOW_MODE_DISCLAIMER,
    )


def _seed_game_and_consensus(conn, slate_date: str = "2026-09-05") -> str:
    """Helper to seed teams, a game, and moneyline + spread market consensus rows."""
    store.upsert_team(
        conn,
        {
            "team_id": "cfbd:100",
            "school": "Georgia",
            "alias": "UGA",
            "market": "Georgia",
            "conference": "SEC",
        },
    )
    store.upsert_team(
        conn,
        {
            "team_id": "cfbd:200",
            "school": "Vanderbilt",
            "alias": "VAN",
            "market": "Vanderbilt",
            "conference": "SEC",
        },
    )
    game_id = "cfbd:401520182"
    store.upsert_game(
        conn,
        {
            "game_id": game_id,
            "season": 2026,
            "week": 1,
            "kickoff_utc": f"{slate_date}T19:30:00+00:00",
            "football_date": slate_date,
            "day_of_week": 5,
            "home_team_id": "cfbd:100",
            "away_team_id": "cfbd:200",
            "venue_name": "Sanford Stadium",
            "network": "CBS",
            "status": "pregame",
        },
    )
    # Insert moneyline consensus
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'ML', 0.0, 'HOME', '2026-09-05T12:00:00+00:00', 5, -150,
                   -145, 'FanDuel', 0.045, 'PINNACLE', 0.585, 0.590, 0.580, 0.005, '[]')""",
        (game_id,),
    )
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'ML', 0.0, 'AWAY', '2026-09-05T12:00:00+00:00', 5, +130,
                   +135, 'DraftKings', 0.045, 'PINNACLE', 0.415, 0.410, 0.420, 0.005, '[]')""",
        (game_id,),
    )
    # Insert spread consensus
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price,
            best_price, best_book, hold, anchor, prob_multiplicative, prob_shin,
            prob_power, prob_spread, flags)
           VALUES (?, 'SPREAD', -14.0, 'HOME', '2026-09-05T12:00:00+00:00', 5, -110,
                   -108, 'DraftKings', 0.040, 'PINNACLE', 0.518, 0.520, 0.515, 0.004, '[]')""",
        (game_id,),
    )
    conn.commit()
    return game_id


# =============================================================================
# 1. Unit Tests: Terminal Renderers
# =============================================================================


class TestReportingTerminalRenderers:
    """Verifies pure standard library terminal formatting per grok_rules.md §5."""

    def test_render_reasoning_card_seven_sections_present(
        self, sample_reasoning_card, sample_governance_verdict
    ):
        rendered = render_reasoning_card(sample_reasoning_card, sample_governance_verdict)

        # Matchup, market, and play headers
        assert "MATCHUP:" in rendered
        assert "Vanderbilt at Georgia" in rendered
        assert "MARKET: ML" in rendered
        assert "RECOMMENDED PLAY: HOME ML (-150)" in rendered
        assert "Confidence: 8.2/10" in rendered
        assert "STRONG" in rendered

        # Verify all 7 authoritative sections per grok_rules.md §5
        assert "1. Tape Evaluation" in rendered
        assert "2. Position & QB Edge" in rendered
        assert "3. Weather, Venue & Travel" in rendered
        assert "4. Injuries & Trench Health" in rendered
        assert "5. Program Structure & Situational Spot" in rendered
        assert "6. Mathematical Edge vs Market Consensus" in rendered
        assert "7. What NOT to Bet" in rendered

        # Verify specific summary texts appear
        assert "Honest FBS tape confirms 0.24 EPA/play floor" in rendered
        assert "QB continuity confirmed" in rendered
        assert "Kickoff 72F, wind 8 kph" in rendered
        assert "All 5 starting OL healthy" in rendered
        assert "4th year HC/OC scheme stability" in rendered
        assert "Model prob 65.2% vs market consensus 58.0%" in rendered
        assert "Do not lay heavy spread points (-14.5)" in rendered

        # Verify shadow mode disclaimer
        assert SHADOW_MODE_DISCLAIMER in rendered

    def test_render_reasoning_card_vetoed_favorite_callout(
        self, sample_vetoed_reasoning_card, sample_veto_verdict
    ):
        rendered = render_reasoning_card(sample_vetoed_reasoning_card, sample_veto_verdict)

        # Must render prominent VETO / Counter-Thesis alert
        assert "VETO" in rendered
        assert "COUNTER-THESIS" in rendered
        assert "Rule D thin favorite trap" in rendered
        assert "Starting QB questionable" in rendered
        assert SHADOW_MODE_DISCLAIMER in rendered

    def test_render_reasoning_card_without_verdict(self, sample_reasoning_card):
        """Verifies rendering succeeds gracefully when verdict is None."""
        rendered = render_reasoning_card(sample_reasoning_card, None)
        assert "MATCHUP:" in rendered
        assert "1. Tape Evaluation" in rendered
        assert SHADOW_MODE_DISCLAIMER in rendered

    def test_render_mispriced_card(self, sample_mispriced_opportunity, sample_governance_verdict):
        rendered = render_mispriced_card(sample_mispriced_opportunity, sample_governance_verdict)
        assert "Vanderbilt at Georgia" in rendered
        assert "SPREAD" in rendered
        assert "HOME -14" in rendered
        assert "5.5%" in rendered or "0.055" in rendered  # Edge %
        assert "0.0042" in rendered or "0.42pp" in rendered  # Method spread
        assert "82.5" in rendered  # Play score
        assert "STRONG" in rendered  # Play tier
        assert SHADOW_MODE_DISCLAIMER in rendered

    def test_render_parlay_card(self, sample_parlay_ticket):
        rendered = render_parlay_card(sample_parlay_ticket)
        assert "PARLAY TICKET" in rendered
        assert "parlay:2026-09-05:3leg:01" in rendered
        assert "3 LEGS" in rendered
        assert "+200" in rendered
        assert "Georgia" in rendered
        assert "Ohio State" in rendered
        assert "Oregon" in rendered
        assert "Joint Win Prob" in rendered
        assert "Fragility Index" in rendered
        assert SHADOW_MODE_DISCLAIMER in rendered


# =============================================================================
# 2. Unit Tests: JSON Export Serializers
# =============================================================================


class TestReportingExportSerializers:
    """Verifies structured JSON serialization with mandatory watermark."""

    def test_export_reasoning_json_structure(
        self, sample_reasoning_card, sample_governance_verdict
    ):
        json_str = export_reasoning_json(
            [sample_reasoning_card], [sample_governance_verdict], slate_date="2026-09-05"
        )
        data = json.loads(json_str)

        assert data["date"] == "2026-09-05"
        assert data["disclaimer"] == SHADOW_MODE_DISCLAIMER
        assert "cards" in data
        assert len(data["cards"]) == 1

        card_data = data["cards"][0]
        assert card_data["game_id"] == "cfbd:401520182"
        assert card_data["recommended_play"] == "HOME ML (-150)"
        assert card_data["confidence"] == 8.2
        assert card_data["tier"] == "STRONG"
        assert "tape_summary" in card_data
        assert "position_qb_summary" in card_data
        assert "weather_venue_summary" in card_data
        assert "injuries_trench_summary" in card_data
        assert "program_continuity_summary" in card_data
        assert "mathematical_edge_summary" in card_data
        assert "contra_indications" in card_data
        assert "verdict" in card_data
        assert card_data["verdict"]["action"] == "APPROVE"

    def test_export_mispriced_json_structure(
        self, sample_mispriced_opportunity, sample_governance_verdict
    ):
        json_str = export_mispriced_json(
            [sample_mispriced_opportunity],
            [sample_governance_verdict],
            slate_date="2026-09-05",
            min_edge=0.02,
        )
        data = json.loads(json_str)

        assert data["date"] == "2026-09-05"
        assert data["min_edge"] == 0.02
        assert data["disclaimer"] == SHADOW_MODE_DISCLAIMER
        assert len(data["candidates"]) == 1

        c = data["candidates"][0]
        assert c["game_id"] == "cfbd:401520182"
        assert c["market"] == "SPREAD"
        assert c["side"] == "HOME"
        assert c["line"] == -14.0
        assert c["edge_pct"] == 0.0550
        assert c["method_spread"] == 0.0042
        assert c["play_score"] == 82.5
        assert c["play_tier"] == "STRONG"

    def test_export_parlay_json_structure(self, sample_parlay_ticket):
        json_str = export_parlay_json(sample_parlay_ticket)
        data = json.loads(json_str)

        assert data["parlay_id"] == "parlay:2026-09-05:3leg:01"
        assert data["leg_count"] == 3
        assert data["total_odds_american"] == 200
        assert data["joint_win_prob"] == 0.351
        assert data["fragility_index"] == 0.28
        assert data["disclaimer"] == SHADOW_MODE_DISCLAIMER
        assert len(data["legs"]) == 3


# =============================================================================
# 3. Integration Tests: CLI `board` Subcommand
# =============================================================================


class TestCliBoardSubcommand:
    """Tests CLI entry point for `board` with authentic reasoning cards and JSON export."""

    def test_board_missing_database_error(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "database_path", lambda: tmp_path / "nonexistent.sqlite3")
        rc = cli.main(["board", "--date", "2026-09-05"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "No database yet" in out

    def test_board_empty_slate_graceful(self, capsys, cli_db):
        rc = cli.main(["board", "--date", "2026-09-05"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No moneyline consensus for 2026-09-05" in out

    def test_board_with_reasoning_terminal(
        self, capsys, cli_db, sample_reasoning_card, sample_governance_verdict, monkeypatch
    ):
        _seed_game_and_consensus(cli_db, "2026-09-05")

        mock_engine = MagicMock()
        mock_engine.evaluate_candidate.return_value = sample_reasoning_card
        monkeypatch.setattr(
            "cfb_analytics.reasoning.engine.MultiFactorReasoningEngine", lambda: mock_engine
        )

        mock_gate = MagicMock()
        mock_gate.evaluate_candidate.return_value = sample_governance_verdict
        monkeypatch.setattr("cfb_analytics.governance.gate.GovernanceGate", lambda: mock_gate)

        rc = cli.main(["board", "--date", "2026-09-05", "--with-reasoning"])
        assert rc == 0
        out = capsys.readouterr().out

        assert "MONEYLINE BOARD - 2026-09-05" in out
        assert SHADOW_MODE_DISCLAIMER in out
        assert "MULTI-FACTOR REASONING CARDS" in out or "1. Tape Evaluation" in out
        assert "1. Tape Evaluation" in out
        assert "2. Position & QB Edge" in out
        assert "3. Weather, Venue & Travel" in out
        assert "4. Injuries & Trench Health" in out
        assert "5. Program Structure & Situational Spot" in out
        assert "6. Mathematical Edge vs Market Consensus" in out
        assert "7. What NOT to Bet" in out

    def test_board_with_reasoning_json(
        self, capsys, cli_db, sample_reasoning_card, sample_governance_verdict, monkeypatch
    ):
        _seed_game_and_consensus(cli_db, "2026-09-05")

        mock_engine = MagicMock()
        mock_engine.evaluate_candidate.return_value = sample_reasoning_card
        monkeypatch.setattr(
            "cfb_analytics.reasoning.engine.MultiFactorReasoningEngine", lambda: mock_engine
        )

        mock_gate = MagicMock()
        mock_gate.evaluate_candidate.return_value = sample_governance_verdict
        monkeypatch.setattr("cfb_analytics.governance.gate.GovernanceGate", lambda: mock_gate)

        capsys.readouterr()  # flush init-db output
        rc = cli.main(["board", "--date", "2026-09-05", "--with-reasoning", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)

        assert payload["date"] == "2026-09-05"
        assert (
            payload.get("disclaimer") == SHADOW_MODE_DISCLAIMER
            or payload.get("stamp") == SHADOW_MODE_DISCLAIMER
        )
        assert "cards" in payload or "entries" in payload

    def test_board_vetoed_favorite_callout(
        self, capsys, cli_db, sample_vetoed_reasoning_card, sample_veto_verdict, monkeypatch
    ):
        _seed_game_and_consensus(cli_db, "2026-09-05")

        mock_engine = MagicMock()
        mock_engine.evaluate_candidate.return_value = sample_vetoed_reasoning_card
        monkeypatch.setattr(
            "cfb_analytics.reasoning.engine.MultiFactorReasoningEngine", lambda: mock_engine
        )

        mock_gate = MagicMock()
        mock_gate.evaluate_candidate.return_value = sample_veto_verdict
        monkeypatch.setattr("cfb_analytics.governance.gate.GovernanceGate", lambda: mock_gate)

        rc = cli.main(["board", "--date", "2026-09-05", "--with-reasoning"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "VETO" in out
        assert "Rule D thin favorite trap" in out


# =============================================================================
# 4. Integration Tests: CLI `mispriced` Subcommand
# =============================================================================


class TestCliMispricedSubcommand:
    """Tests CLI entry point for `mispriced` scanner with terminal tables and JSON output."""

    def test_mispriced_missing_database_error(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "database_path", lambda: tmp_path / "nonexistent.sqlite3")
        rc = cli.main(["mispriced", "--date", "2026-09-05"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "No database yet" in out

    def test_mispriced_empty_slate_graceful(self, capsys, cli_db):
        rc = cli.main(["mispriced", "--date", "2026-09-05"])
        assert rc == 0
        out = capsys.readouterr().out
        assert SHADOW_MODE_DISCLAIMER in out

    def test_mispriced_terminal_output(
        self, capsys, cli_db, sample_mispriced_opportunity, monkeypatch
    ):
        _seed_game_and_consensus(cli_db, "2026-09-05")

        monkeypatch.setattr(
            "cfb_analytics.scanner.engine.MispricedScanner.scan_slate",
            lambda *a, **kw: [sample_mispriced_opportunity],
        )

        rc = cli.main(["mispriced", "--date", "2026-09-05"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "MISPRICED" in out
        assert SHADOW_MODE_DISCLAIMER in out
        assert "Vanderbilt at Georgia" in out or "cfbd:401520182" in out

    def test_mispriced_json_output(self, capsys, cli_db, sample_mispriced_opportunity, monkeypatch):
        _seed_game_and_consensus(cli_db, "2026-09-05")

        monkeypatch.setattr(
            "cfb_analytics.scanner.engine.MispricedScanner.scan_slate",
            lambda *a, **kw: [sample_mispriced_opportunity],
        )

        capsys.readouterr()  # flush init-db output
        rc = cli.main(["mispriced", "--date", "2026-09-05", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)

        assert payload["date"] == "2026-09-05"
        assert (
            payload.get("disclaimer") == SHADOW_MODE_DISCLAIMER
            or payload.get("stamp") == SHADOW_MODE_DISCLAIMER
        )
        assert "candidates" in payload
        assert len(payload["candidates"]) == 1

        cand = payload["candidates"][0]
        assert cand["game_id"] == "cfbd:401520182"
        assert cand["market"] == "SPREAD"
        assert cand["side"] == "HOME"
        assert cand["line"] == -14.0
        assert cand["edge_pct"] == 0.0550
        assert cand["method_spread"] == 0.0042
        assert cand["play_score"] == 82.5
        assert cand["play_tier"] == "STRONG"

    def test_mispriced_min_edge_filter(self, capsys, cli_db, monkeypatch):
        _seed_game_and_consensus(cli_db, "2026-09-05")

        opp_low = MispricedOpportunity(
            game_id="g1",
            market_type="SPREAD",
            market="SPREAD",
            side="HOME",
            line=-3.0,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.52,
            consensus_fair_price_american=-108,
            model_projected_line=-5.0,
            model_prob=0.54,
            edge_pct=0.02,
            ev=0.03,
            method_spread=0.002,
            n_books=4,
            play_score=72.0,
            play_tier=PlayTier.QUALIFIED.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )
        opp_high = MispricedOpportunity(
            game_id="g2",
            market_type="TOTAL",
            market="TOTAL",
            side="OVER",
            line=52.5,
            posted_price_american=-110,
            posted_price_decimal=1.909,
            consensus_fair_prob=0.50,
            consensus_fair_price_american=-100,
            model_projected_line=60.0,
            model_prob=0.60,
            edge_pct=0.10,
            ev=0.145,
            method_spread=0.005,
            n_books=5,
            play_score=88.0,
            play_tier=PlayTier.STRONG.value,
            qual_status=QualificationStatus.QUALIFIED.value,
        )

        def fake_scan(self, date, min_edge=0.02):
            return [o for o in [opp_low, opp_high] if o.edge_pct >= min_edge]

        monkeypatch.setattr("cfb_analytics.scanner.engine.MispricedScanner.scan_slate", fake_scan)

        capsys.readouterr()  # flush init-db output
        rc = cli.main(["mispriced", "--date", "2026-09-05", "--min-edge", "0.05", "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["candidates"]) == 1
        assert payload["candidates"][0]["game_id"] == "g2"


# =============================================================================
# 5. Invariant Tests: Shadow Mode Watermark & Formatting
# =============================================================================


class TestShadowModeWatermarkEnforcement:
    """Verifies mandatory operational invariant 1: universal shadow disclaimer watermark."""

    def test_watermark_in_all_terminal_card_renderers(
        self,
        sample_reasoning_card,
        sample_mispriced_opportunity,
        sample_parlay_ticket,
        sample_governance_verdict,
    ):
        card_text = render_reasoning_card(sample_reasoning_card, sample_governance_verdict)
        assert SHADOW_MODE_DISCLAIMER in card_text

        mispriced_text = render_mispriced_card(
            sample_mispriced_opportunity, sample_governance_verdict
        )
        assert SHADOW_MODE_DISCLAIMER in mispriced_text

        parlay_text = render_parlay_card(sample_parlay_ticket)
        assert SHADOW_MODE_DISCLAIMER in parlay_text

    def test_watermark_in_all_json_exporters(
        self,
        sample_reasoning_card,
        sample_mispriced_opportunity,
        sample_parlay_ticket,
        sample_governance_verdict,
    ):
        r_json = json.loads(
            export_reasoning_json([sample_reasoning_card], [sample_governance_verdict])
        )
        assert r_json["disclaimer"] == SHADOW_MODE_DISCLAIMER

        m_json = json.loads(
            export_mispriced_json([sample_mispriced_opportunity], [sample_governance_verdict])
        )
        assert m_json["disclaimer"] == SHADOW_MODE_DISCLAIMER

        p_json = json.loads(export_parlay_json(sample_parlay_ticket))
        assert p_json["disclaimer"] == SHADOW_MODE_DISCLAIMER


# =============================================================================
# 6. Unit Tests: Situational Context Database Loader
# =============================================================================


class TestSituationalContextLoader:
    """Verifies authentic DB querying and assembly in cfb_analytics.reasoning.context."""

    def test_load_situational_context_for_game(self, cli_db):
        game_id = _seed_game_and_consensus(cli_db, "2026-09-05")
        ctx = load_situational_context_for_game(cli_db, game_id, "2026-09-05")

        assert ctx.game_id == game_id
        assert ctx.home_team == "UGA"
        assert ctx.away_team == "VAN"
        assert ctx.home_team_id == "cfbd:100"
        assert ctx.away_team_id == "cfbd:200"
        assert ctx.weather is not None
        assert ctx.tape_home is not None
        assert ctx.tape_away is not None
        assert ctx.qb_home_confirmed is True
        assert ctx.market_spread_home == -14.0
        assert ctx.market_ml_home_american == -150

    def test_load_slate_situational_contexts(self, cli_db):
        game_id = _seed_game_and_consensus(cli_db, "2026-09-05")
        contexts = load_slate_situational_contexts(cli_db, "2026-09-05")

        assert game_id in contexts
        assert contexts[game_id].home_team == "UGA"
