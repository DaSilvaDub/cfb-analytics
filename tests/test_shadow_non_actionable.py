"""Shadow promotion status labels published boards as non-actionable.

promotion.json status is shadow and beat_market is false. The nightly board
is still frozen. This does not call GovernanceGate. It reads the promotion
file and stamps the report and the snapshot.
"""

from __future__ import annotations

from cfb_analytics import config
from cfb_analytics.features.over_confidence import (
    OverConfidencePick,
    persist_over_board,
    render_over_confidence_board,
)
from cfb_analytics.reporting.slate import build_slate_report, render_slate_report, slate_report_json
from tests.conftest import seed_canonical_game, seed_canonical_team


def _pick() -> OverConfidencePick:
    return OverConfidencePick(
        rank=1,
        game_id="cfbd:iu",
        game_label="WKU at IU",
        kickoff_et="4:00 p.m.",
        market="team_rushing_yards",
        pick="IU rush",
        team_id="cfbd:84",
        side="OVER",
        line=170.5,
        projected=200.0,
        over_prob=0.7,
        confidence=7.6,
        tier="HIGH",
        inclusion_reasons=("ap_top_25",),
        flags=("inferred_mark",),
        family="favorite_team_rush",
        kickoff_utc="2026-09-19T20:00:00+00:00",
    )


def _seed(conn) -> None:
    home = seed_canonical_team(conn, 84, "Indiana", "IU")
    away = seed_canonical_team(conn, 98, "Western Kentucky", "WKU")
    seed_canonical_game(
        conn,
        "cfbd:iu",
        home,
        away,
        football_date="2026-09-19",
        kickoff="2026-09-19T20:00:00+00:00",
        week=3,
    )


def test_shadow_promotion_labels_board_slate_and_snapshot(conn, monkeypatch) -> None:
    monkeypatch.setattr("cfb_analytics.config.promotion", lambda: {"status": "shadow"})
    assert config.is_shadow_mode() is True
    assert config.published_output_is_actionable() is False

    board = render_over_confidence_board("2026-09-19", [], min_prob=0.50)
    assert config.NON_ACTIONABLE_LABEL in board
    slate = render_slate_report(build_slate_report(conn, "2026-09-19"))
    assert config.NON_ACTIONABLE_LABEL in slate
    payload = slate_report_json(build_slate_report(conn, "2026-09-19"))
    assert payload["actionable"] is False

    _seed(conn)
    persist_over_board(
        conn, "2026-09-19", [_pick()], as_of_utc="2026-09-19T12:00:00+00:00"
    )
    stored = conn.execute(
        "SELECT actionable FROM over_board_snapshots WHERE game_id = 'cfbd:iu'"
    ).fetchone()
    assert stored["actionable"] == 0


def test_promoted_promotion_omits_the_non_actionable_label(conn, monkeypatch) -> None:
    monkeypatch.setattr(
        "cfb_analytics.config.promotion",
        lambda: {"status": "promoted", "evidence": {"beat_market": True}},
    )
    assert config.is_shadow_mode() is False
    assert config.published_output_is_actionable() is True

    board = render_over_confidence_board("2026-09-19", [], min_prob=0.50)
    assert config.NON_ACTIONABLE_LABEL not in board
    slate = render_slate_report(build_slate_report(conn, "2026-09-19"))
    assert config.NON_ACTIONABLE_LABEL not in slate
    payload = slate_report_json(build_slate_report(conn, "2026-09-19"))
    assert payload["actionable"] is True

    _seed(conn)
    persist_over_board(
        conn, "2026-09-19", [_pick()], as_of_utc="2026-09-19T12:00:00+00:00"
    )
    stored = conn.execute(
        "SELECT actionable FROM over_board_snapshots WHERE game_id = 'cfbd:iu'"
    ).fetchone()
    assert stored["actionable"] == 1
