"""Daily weather opt-in propagation and non-actionable game totals."""

from dataclasses import asdict
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cfb_analytics.daily import run_daily
from cfb_analytics.ingest import store


@pytest.fixture
def daily_weather_db(conn, monkeypatch):
    for team_id in ("home", "away"):
        store.upsert_team(conn, {"team_id": team_id, "school": team_id})
    conn.execute(
        """INSERT INTO games
           (game_id, season, kickoff_utc, football_date, home_team_id,
            away_team_id, source, ingested_utc)
           VALUES ('game', 2026, '2026-09-05T20:00:00+00:00', '2026-09-05',
                   'home', 'away', 'cfbd', '2026-09-01T12:00:00+00:00')"""
    )
    conn.execute(
        """INSERT INTO weather
           (game_id, as_of_utc, hours_to_kick, temp_c, wind_kph, wind_gust_kph,
            precip_mm, precip_prob, humidity, is_forecast, is_indoor, source)
           VALUES ('game', '2026-09-05T12:00:00+00:00', 8, 3, 35, 45,
                   4, 85, 80, 1, 0, 'open-meteo')"""
    )
    conn.execute(
        """INSERT INTO team_prop_consensus
           (game_id, team_id, market, line, side, as_of_utc, n_books,
            consensus_price, best_price, best_book, hold,
            prob_multiplicative, prob_shin)
           VALUES ('game', 'home', 'team_total_points', 28.5, 'OVER',
                   '2026-09-05T12:00:00+00:00', 3, -110, -105,
                   'FanDuel', 0.045, 0.52, 0.52)"""
    )
    monkeypatch.setattr("cfb_analytics.daily._run_cfbd_lines", Mock())
    weather_ingest = Mock()
    monkeypatch.setattr("cfb_analytics.daily._run_weather", weather_ingest)
    monkeypatch.setattr(
        "cfb_analytics.features.build_market.build_market_for_slate",
        Mock(return_value=SimpleNamespace(consensus_rows=0, movement_rows=0, games=1)),
    )
    return conn, weather_ingest


def _daily(
    conn, *, with_weather=True, with_scoring=True, now=datetime(2026, 9, 5, 13, tzinfo=UTC)
):
    return run_daily(
        conn,
        with_outlier=False,
        with_weather=with_weather,
        with_player_passing=False,
        with_internal_ratings=False,
        with_internal_elo=False,
        with_scoring=with_scoring,
        bootstrap=False,
        now=now,
    )


def test_daily_weather_attenuates_candidates_and_research_totals(daily_weather_db):
    conn, weather_ingest = daily_weather_db
    baseline = _daily(conn, with_weather=False)
    weather_ingest.assert_not_called()
    adjusted = _daily(conn)
    weather_ingest.assert_called_once()

    assert baseline.candidates_scored == adjusted.candidates_scored == 1
    assert (
        adjusted.candidate_scores[0].projected_value
        < baseline.candidate_scores[0].projected_value
    )
    baseline_total = baseline.research_game_totals["game"]
    adjusted_total = adjusted.research_game_totals["game"]
    assert (
        adjusted_total.projection.projected_game_total
        < baseline_total.projection.projected_game_total
    )
    assert adjusted_total.status == "uncalibrated_shadow"
    assert adjusted_total.actionable is False
    serialized = asdict(adjusted)["research_game_totals"]["game"]
    assert serialized["actionable"] is False
    assert serialized["status"] == "uncalibrated_shadow"
    assert "uncalibrated_shadow; actionable=false" in adjusted.as_text()
    projection = adjusted_total.projection
    assert f"game: total={projection.projected_game_total:.2f}" in adjusted.as_text()
    assert f"home={projection.home_projected_points:.2f}" in adjusted.as_text()
    assert f"away={projection.away_projected_points:.2f}" in adjusted.as_text()


def test_daily_scoring_disabled_has_no_research_totals(daily_weather_db):
    conn, _ = daily_weather_db
    report = _daily(conn, with_weather=False, with_scoring=False)
    assert report.research_game_totals == {}
    assert report.candidate_scores == []
    assert all(outcome.name not in {"scoring", "game_totals"} for outcome in report.outcomes)


@pytest.mark.parametrize("historical", [False, True])
def test_daily_new_forecast_uses_scoring_clock_unless_cutoff_explicit(
    daily_weather_db, monkeypatch, historical
):
    conn, weather_ingest = daily_weather_db
    start = datetime(2026, 9, 5, 13, tzinfo=UTC)
    end = datetime(2026, 9, 5, 13, 2, tzinfo=UTC)
    baseline = _daily(conn, with_weather=False, now=start)
    conn.execute("DELETE FROM weather")

    def ingest_during_run(*args):
        conn.execute(
            """INSERT INTO weather
               (game_id, as_of_utc, hours_to_kick, temp_c, wind_kph, wind_gust_kph,
                precip_mm, precip_prob, humidity, is_forecast, is_indoor, source)
               VALUES ('game', '2026-09-05T13:01:00+00:00', 6.98, 3, 35, 45,
                       4, 85, 80, 1, 0, 'open-meteo')"""
        )

    weather_ingest.side_effect = ingest_during_run
    clock = Mock(wraps=datetime)
    clock.now.side_effect = [start, end]
    monkeypatch.setattr("cfb_analytics.daily.datetime", clock)
    report = _daily(conn, now=start if historical else None)
    forecast_total = report.research_game_totals["game"]
    baseline_total = baseline.research_game_totals["game"].projection.projected_game_total
    baseline_val = baseline.candidate_scores[0].projected_value
    actual_val = report.candidate_scores[0].projected_value
    if historical:
        assert actual_val == baseline_val
        assert forecast_total.projection.projected_game_total == baseline_total
        assert forecast_total.as_of_utc == start.isoformat()
        clock.now.assert_not_called()
    else:
        assert actual_val < baseline_val
        assert forecast_total.projection.projected_game_total < baseline_total
        assert forecast_total.as_of_utc == end.isoformat()
        assert clock.now.call_count == 2
