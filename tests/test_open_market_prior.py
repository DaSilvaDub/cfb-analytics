"""Opening-line market prior (weeks 1–2 early-season spike)."""

from __future__ import annotations

import math

import pytest

from cfb_analytics.backtest.calibration import margin_to_prob
from cfb_analytics.backtest.harness import GamePrediction
from cfb_analytics.backtest.moneyline import (
    _blend_open_prior,
    _ensemble_raw_prob,
)
from cfb_analytics.features.open_market_prior import (
    OPEN_MARKET_BLEND_MAX_WEEK,
    OPEN_MARKET_BLEND_WEIGHT,
    OPEN_MARKET_PRIOR_MAX_WEEK,
    OPEN_MARKET_PRIOR_WEIGHT,
    open_prior_weight_for_week,
    open_spread_to_home_prob,
)


def _pred(week: int, margin: float = 7.0, gid: str = "g1") -> GamePrediction:
    return GamePrediction(
        game_id=gid,
        season=2024,
        week=week,
        home_team_id="h",
        away_team_id="a",
        neutral_site=False,
        predicted_margin=margin,
        actual_margin=3.0,
        elo_home_rating=None,
        elo_away_rating=None,
        internal_elo_win_prob=0.55,
        logit_win_prob=None,
    )


class TestOpenSpreadToProb:
    def test_home_favorite_above_half(self):
        # HOME -14 => margin +14 => P(home) > 0.5
        p = open_spread_to_home_prob(-14.0, 20.0)
        assert p == pytest.approx(margin_to_prob(14.0, 20.0))
        assert p > 0.5

    def test_home_dog_below_half(self):
        p = open_spread_to_home_prob(7.0, 20.0)
        assert p < 0.5


class TestBlendOpenPrior:
    def test_weight_one_is_replace(self):
        assert _blend_open_prior(0.4, 0.7, 1.0) == pytest.approx(0.7)

    def test_weight_zero_keeps_model(self):
        assert _blend_open_prior(0.4, 0.7, 0.0) == pytest.approx(0.4)

    def test_partial_is_between(self):
        p = _blend_open_prior(0.4, 0.7, 0.5)
        assert min(0.4, 0.7) < p < max(0.4, 0.7)


class TestEnsembleRawOpenPrior:
    def test_week2_replaces_with_open(self):
        weights = {"ridge": 0.15, "internal_elo": 0.85}
        # Open heavily favors home; model is near coin-flip.
        p = _ensemble_raw_prob(
            _pred(2, margin=0.0),
            20.0,
            weights,
            open_spreads={"g1": -21.0},
        )
        assert p == pytest.approx(open_spread_to_home_prob(-21.0, 20.0))

    def test_week5_ignores_open(self):
        weights = {"ridge": 0.15, "internal_elo": 0.85}
        p_with = _ensemble_raw_prob(
            _pred(5, margin=0.0),
            20.0,
            weights,
            open_spreads={"g1": -21.0},
        )
        p_without = _ensemble_raw_prob(
            _pred(5, margin=0.0), 20.0, weights, open_spreads=None
        )
        assert p_with == pytest.approx(p_without)

    def test_missing_open_keeps_model(self):
        weights = {"ridge": 1.0}
        p = _ensemble_raw_prob(
            _pred(1, margin=10.0),
            20.0,
            weights,
            open_spreads={},
        )
        assert p == pytest.approx(margin_to_prob(10.0, 20.0 * 1.5))  # early sigma

    def test_constants_match_spike(self):
        assert OPEN_MARKET_PRIOR_MAX_WEEK == 2
        assert OPEN_MARKET_PRIOR_WEIGHT == 1.0
        assert math.isfinite(OPEN_MARKET_PRIOR_WEIGHT)
        assert OPEN_MARKET_BLEND_MAX_WEEK == 4
        assert OPEN_MARKET_BLEND_WEIGHT == 0.75
        assert 0.0 < OPEN_MARKET_BLEND_WEIGHT < 1.0

    def test_week3_soft_blend_when_weight_set(self):
        weights = {"ridge": 0.15, "internal_elo": 0.85}
        p_base = _ensemble_raw_prob(
            _pred(3, margin=0.0),
            20.0,
            weights,
            open_spreads={"g1": -21.0},
            open_blend_weight=0.0,
        )
        p_blend = _ensemble_raw_prob(
            _pred(3, margin=0.0),
            20.0,
            weights,
            open_spreads={"g1": -21.0},
            open_blend_weight=0.5,
        )
        p_open = open_spread_to_home_prob(-21.0, 20.0)
        assert p_base != pytest.approx(p_open)
        assert min(p_base, p_open) < p_blend < max(p_base, p_open)

    def test_week2_still_full_replace_with_blend_on(self):
        weights = {"ridge": 0.15, "internal_elo": 0.85}
        p = _ensemble_raw_prob(
            _pred(2, margin=0.0),
            20.0,
            weights,
            open_spreads={"g1": -21.0},
            open_blend_weight=0.5,
        )
        assert p == pytest.approx(open_spread_to_home_prob(-21.0, 20.0))

    def test_week5_ignores_blend(self):
        weights = {"ridge": 0.15, "internal_elo": 0.85}
        p_with = _ensemble_raw_prob(
            _pred(5, margin=0.0),
            20.0,
            weights,
            open_spreads={"g1": -21.0},
            open_blend_weight=0.75,
        )
        p_without = _ensemble_raw_prob(
            _pred(5, margin=0.0), 20.0, weights, open_spreads=None
        )
        assert p_with == pytest.approx(p_without)


class TestOpenPriorWeightForWeek:
    def test_w12_replace_w34_soft_later_zero(self):
        assert open_prior_weight_for_week(1) == 1.0
        assert open_prior_weight_for_week(2) == 1.0
        assert open_prior_weight_for_week(3, blend_weight=0.5) == 0.5
        assert open_prior_weight_for_week(4, blend_weight=0.5) == 0.5
        assert open_prior_weight_for_week(5, blend_weight=0.5) == 0.0

    def test_default_blend_weight(self):
        assert open_prior_weight_for_week(3) == OPEN_MARKET_BLEND_WEIGHT
        assert open_prior_weight_for_week(4) == OPEN_MARKET_BLEND_WEIGHT
