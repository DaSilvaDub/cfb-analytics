from __future__ import annotations

from types import SimpleNamespace

import pytest

from cfb_analytics.features.over_confidence import OverConfidencePick
from cfb_analytics.models.team_props import GameTotalsProjection
from cfb_analytics.reporting import slate


@pytest.fixture
def report_inputs(conn, canonical_slate, monkeypatch):
    game_id = canonical_slate["games"]["evt-1"]
    rows = [
        {
            "game_id": game_id,
            "market": market,
            "side": side,
            "line": line,
            "as_of_utc": "2026-09-05T12:00:00+00:00",
            "n_books": 3,
            "best_price": -110,
            "best_book": "Book A",
            "prob_shin": fair,
            "flags": "[]",
        }
        for market, side, line, fair in (
            ("ML", "HOME", 0, 0.55),
            ("ML", "AWAY", 0, 0.45),
            ("SPREAD", "HOME", -3.5, 0.50),
            ("SPREAD", "AWAY", 3.5, 0.50),
            ("TOTAL", "OVER", 45.5, 0.51),
            ("TOTAL", "UNDER", 45.5, 0.49),
        )
    ]
    monkeypatch.setattr(slate, "load_current_market_rows", lambda *_: rows)
    monkeypatch.setattr(slate, "_load_model_margins", lambda *_: {game_id: 7.0})
    monkeypatch.setattr(
        slate, "project_slate_game_totals",
        lambda *_, **kwargs: {game_id: GameTotalsProjection(28.0, 21.0, 49.0)},
    )
    monkeypatch.setattr(slate, "load_published_board", lambda *_: [])
    return SimpleNamespace(conn=conn, game_id=game_id, rows=rows)


def test_all_game_line_sections_keep_quotes_and_missing_games(report_inputs):
    report = slate.build_slate_report(report_inputs.conn, "2026-09-05")
    for section in ("moneylines", "spreads", "game_totals"):
        assert len(report[section]) == 3  # paired first game and missing second game
        assert report[section][-1]["quote_status"] == "missing"
        assert "missing_quote" in report[section][-1]["flags"]
        assert all(row["actionable"] is False for row in report[section])
    assert {item["market"] for item in report["missing_coverage"]} == {
        "ML", "SPREAD", "TOTAL", "YARDAGE"
    }
    rendered = slate.render_slate_report(report)
    for heading in ("Moneylines", "Spreads", "Game totals", "Yardage props"):
        assert f"## {heading}" in rendered
    assert "Duke" in rendered
    assert "Market fair P" in rendered
    assert "Capture UTC" in rendered
    assert "7:30 p.m." in rendered
    assert "UNPROMOTED - shadow output, not decision-grade" in rendered


def test_market_fair_probability_is_not_model_probability(report_inputs):
    report = slate.build_slate_report(report_inputs.conn, "2026-09-05")
    home, away = report["spreads"][:2]
    assert home["market_fair_probability"] == 0.5
    assert home["model_probability"] > 0.5
    assert away["model_probability"] < 0.5
    assert home["model_probability"] + away["model_probability"] == pytest.approx(1)
    over, under = report["game_totals"][:2]
    assert over["projected_game_total"] == 49.0
    assert over["market_fair_probability"] == 0.51
    assert over["model_probability"] > 0.51
    assert over["model_probability"] + under["model_probability"] == pytest.approx(1)


def test_yardage_uses_frozen_snapshot_and_exposes_inferred_source(report_inputs, monkeypatch):
    pick = OverConfidencePick(
        rank=1, game_id=report_inputs.game_id, game_label="Oklahoma State at Tulsa",
        kickoff_et="7:30 p.m.", market="team_rushing_yards", pick="Tulsa rushing",
        team_id="cfbd:202", side="OVER", line=100.5, projected=140, over_prob=0.7,
        confidence=7.0, tier="MEDIUM", inclusion_reasons=("conference_top_two",),
        flags=("inferred_mark", "cupcake_tape"), family="favorite_team_rush",
        kickoff_utc="2026-09-05T23:30:00+00:00",
    )
    monkeypatch.setattr(slate, "load_published_board", lambda *_: [pick])
    report = slate.build_slate_report(report_inputs.conn, "2026-09-05")
    row = report["yardage_props"][0]
    assert row["line_source"] == "inferred"
    assert row["inferred_line"] is True
    assert row["market_fair_probability"] is None
    assert row["model_probability"] == 0.7
    assert "cupcake_tape" in row["flags"]
    assert row["actionable"] is False
    assert "inferred lines are model marks" in slate.render_slate_report(report)
    count = report_inputs.conn.execute("SELECT count(*) FROM over_board_snapshots").fetchone()[0]
    assert count == 0


def test_missing_model_is_explicit_and_nonfinite_probability_rejected(report_inputs, monkeypatch):
    monkeypatch.setattr(slate, "_load_model_margins", lambda *_: {})
    report_inputs.rows[0]["prob_shin"] = float("nan")
    report = slate.build_slate_report(report_inputs.conn, "2026-09-05")
    home = report["moneylines"][0]
    assert home["market_fair_probability"] is None
    assert home["model_probability"] is None
    assert "model_probability_unavailable" in home["flags"]


def test_total_projection_receives_weather_and_asof_cutoff(report_inputs, monkeypatch):
    calls = []

    def project(*args, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(slate, "project_slate_game_totals", project)
    report = slate.build_slate_report(report_inputs.conn, "2026-09-05")
    assert calls == [{"as_of_utc": report["generated_utc"], "with_weather": True}]
