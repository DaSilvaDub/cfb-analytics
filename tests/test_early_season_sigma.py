"""Early-season ridge sigma scale (weeks 1–4 skill-gap spike)."""

from __future__ import annotations

import pytest

from cfb_analytics.backtest.harness import GamePrediction
from cfb_analytics.backtest.moneyline import (
    EARLY_SEASON_MAX_WEEK,
    EARLY_SEASON_RIDGE_SIGMA_SCALE,
    _member_probs,
    _ridge_prob,
    ridge_sigma_for_week,
)
from cfb_analytics.backtest.calibration import margin_to_prob


def _pred(week: int, margin: float = 14.0) -> GamePrediction:
    return GamePrediction(
        game_id=f"g-w{week}",
        season=2024,
        week=week,
        home_team_id="h",
        away_team_id="a",
        neutral_site=False,
        predicted_margin=margin,
        actual_margin=7.0,
        elo_home_rating=None,
        elo_away_rating=None,
        internal_elo_win_prob=0.62,
        logit_win_prob=0.55,
    )


class TestRidgeSigmaForWeek:
    def test_weeks_1_through_4_are_inflated(self):
        for week in range(1, EARLY_SEASON_MAX_WEEK + 1):
            assert ridge_sigma_for_week(20.0, week) == pytest.approx(
                20.0 * EARLY_SEASON_RIDGE_SIGMA_SCALE
            )

    def test_week_5_plus_unchanged(self):
        assert ridge_sigma_for_week(20.0, 5) == 20.0
        assert ridge_sigma_for_week(20.0, 12) == 20.0

    def test_non_positive_sigma_raises(self):
        with pytest.raises(ValueError):
            ridge_sigma_for_week(0.0, 1)


class TestEarlySeasonRidgeProb:
    def test_early_week_ridge_is_less_extreme_than_late_week(self):
        sigma = 20.0
        early = _ridge_prob(_pred(2, margin=21.0), sigma)
        late = _ridge_prob(_pred(8, margin=21.0), sigma)
        # Same margin; inflated early sigma pulls probability toward 0.5.
        assert abs(early - 0.5) < abs(late - 0.5)
        assert early == pytest.approx(
            margin_to_prob(21.0, sigma * EARLY_SEASON_RIDGE_SIGMA_SCALE)
        )
        assert late == pytest.approx(margin_to_prob(21.0, sigma))

    def test_member_probs_uses_week_aware_ridge_sigma(self):
        sigma = 20.0
        members = _member_probs(_pred(3, margin=14.0), sigma)
        assert members["ridge"] == pytest.approx(_ridge_prob(_pred(3, margin=14.0), sigma))
        assert members["internal_elo"] == 0.62
        assert members["logit"] == 0.55
