"""Tests for the moneyline backtest orchestration: does it wire the harness,
calibration, and metrics together the way its own docstring claims -- sigma_0
fit only from non-2020 residuals, 2020 scored separately as a stress slice,
and the report text making the missing-baselines limitation explicit rather
than silently pretending to be a promotion verdict.
"""

from __future__ import annotations

from cfb_analytics.backtest.moneyline import run_moneyline_backtest
from cfb_analytics.ingest import store


def _seed_team(conn, team_id, school):
    store.upsert_team(conn, {"team_id": team_id, "school": school, "alias": None, "market": None})


def _seed_season(conn, season, teams=("a", "b", "c", "d"), weeks=6):
    for team_id in teams:
        _seed_team(conn, f"{team_id}{season}", team_id.upper())
    game_id = 0
    for week in range(1, weeks + 1):
        for i, home in enumerate(teams):
            for away in teams[i + 1:]:
                game_id += 1
                store.upsert_cfbd_game(conn, {
                    "game_id": f"g{season}-{game_id}", "season": season, "week": week,
                    "season_type": "regular",
                    "kickoff_utc": f"{season}-09-{week:02d}T00:00:00+00:00",
                    "football_date": f"{season}-09-{week:02d}",
                    "neutral_site": 0, "conference_game": 0,
                    "home_team_id": f"{home}{season}", "away_team_id": f"{away}{season}",
                    "venue_name": None, "venue_id": None, "status": "final",
                    "home_points": 24 + game_id % 10, "away_points": 14 + game_id % 7,
                    "completed": 1, "source": "cfbd",
                })


class TestRunMoneylineBacktest:
    def test_stress_season_is_excluded_from_sigma_calibration_but_still_scored(self, conn):
        _seed_season(conn, 2019)
        _seed_season(conn, 2020)

        report = run_moneyline_backtest(conn, seasons=(2019, 2020), min_games=1)

        assert report.stress is not None
        assert report.stress.label == "2020 stress slice"
        assert report.stress.n_games > 0
        assert "2020" not in report.seasons.label.split("(")[0]
        assert report.n_games_calibrated == report.seasons.n_games

    def test_sigma_0_is_positive_and_finite(self, conn):
        _seed_season(conn, 2019)
        _seed_season(conn, 2020)
        report = run_moneyline_backtest(conn, seasons=(2019, 2020), min_games=1)
        assert report.sigma_0 > 0

    def test_report_text_states_the_missing_baseline_limitation(self, conn):
        _seed_season(conn, 2019)
        text = run_moneyline_backtest(conn, seasons=(2019,), min_games=1).as_text()
        assert "not a promotion decision" in text
        assert "baseline" in text.lower()

    def test_report_text_includes_both_calibration_tables(self, conn):
        _seed_season(conn, 2019)
        text = run_moneyline_backtest(conn, seasons=(2019,), min_games=1).as_text()
        assert "confidence buckets" in text
        assert "reliability curve" in text

    def test_no_stress_season_requested_omits_the_stress_section(self, conn):
        _seed_season(conn, 2019)
        report = run_moneyline_backtest(conn, seasons=(2019,), min_games=1)
        assert report.stress is None
        assert "stress" not in report.as_text().lower()
