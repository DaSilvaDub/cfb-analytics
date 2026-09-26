"""Market blend math + walk-forward weight fit (floor discipline)."""

from __future__ import annotations

import pytest

from cfb_analytics.backtest.harness import GamePrediction
from cfb_analytics.backtest.market_blend_fit import (
    BlendRow,
    fit_market_weight,
    score_fixed_weight,
    walk_forward_market_blend,
)
from cfb_analytics.backtest.metrics import log_loss
from cfb_analytics.backtest.moneyline import _build_same_game_set
from cfb_analytics.models.market_blend import (
    blend_with_market,
    clamp_market_weight,
    market_weight_floor_from_settings,
)


class TestBlendMath:
    def test_blend_at_w1_is_market(self):
        assert blend_with_market(0.70, 0.40, 1.0, floor=0.75) == pytest.approx(0.70)

    def test_blend_at_floor_caps_model_share(self):
        # w requested 0.0 but floor 0.75 → w=0.75
        p = blend_with_market(0.80, 0.20, 0.0, floor=0.75)
        assert p == pytest.approx(0.75 * 0.80 + 0.25 * 0.20)

    def test_clamp_rejects_bad_floor(self):
        with pytest.raises(ValueError):
            clamp_market_weight(0.9, floor=1.5)

    def test_settings_reader_default(self):
        assert market_weight_floor_from_settings({}) == 0.75
        assert market_weight_floor_from_settings({"blend": {"market_weight_floor": 0.8}}) == 0.8


class TestWalkForwardFit:
    def test_fit_prefers_market_when_model_is_noise(self):
        # Market sharp+correct; model near 0.5. Best w should be 1.0.
        rows = [
            BlendRow(season=2023, p_market=0.90, p_model=0.55, won=True),
            BlendRow(season=2023, p_market=0.10, p_model=0.45, won=False),
            BlendRow(season=2023, p_market=0.85, p_model=0.50, won=True),
            BlendRow(season=2023, p_market=0.15, p_model=0.50, won=False),
        ]
        w = fit_market_weight(rows, floor=0.75, grid_step=0.05)
        assert w == pytest.approx(1.0)

    def test_walk_forward_uses_prior_season_only(self):
        # 2023: model worse → fitted w for 2024 should be high (near 1).
        # 2024: model better than market → but fit is on 2023 only, so w stays high.
        rows = [
            BlendRow(season=2023, p_market=0.90, p_model=0.55, won=True, key="a"),
            BlendRow(season=2023, p_market=0.10, p_model=0.45, won=False, key="b"),
            BlendRow(season=2024, p_market=0.55, p_model=0.90, won=True, key="c"),
            BlendRow(season=2024, p_market=0.45, p_model=0.10, won=False, key="d"),
        ]
        wf = walk_forward_market_blend(rows, floor=0.75, grid_step=0.05)
        assert wf.weight_by_season[2023] == pytest.approx(0.75)  # no prior → floor
        assert wf.weight_by_season[2024] == pytest.approx(1.0)  # prior favors market
        assert len(wf.blended) == 4

    def test_pure_model_bracket_ignores_floor(self):
        rows = [BlendRow(season=2024, p_market=0.70, p_model=0.40, won=True)]
        preds = score_fixed_weight(rows, 0.0, floor=0.0)
        assert preds[0][0] == pytest.approx(0.40)


def _pred(game_id: str, *, season: int, margin: float, actual: float, elo: float, logit: float):
    return GamePrediction(
        game_id=game_id,
        season=season,
        week=2,
        home_team_id="h",
        away_team_id="a",
        neutral_site=False,
        predicted_margin=margin,
        actual_margin=actual,
        elo_home_rating=1600.0,
        elo_away_rating=1500.0,
        internal_elo_win_prob=elo,
        logit_win_prob=logit,
    )


class TestSameGameSetBlendEvidence:
    def test_blend_brackets_and_beat_market_on_pure_model(self):
        # Market sharp; ensemble near coin-flip → beat_market False;
        # walk-forward blend (high w) should be non-worse than market.
        preds = [
            _pred("g1", season=2023, margin=1.0, actual=14.0, elo=0.52, logit=0.51),
            _pred("g2", season=2023, margin=1.0, actual=7.0, elo=0.52, logit=0.51),
            _pred("g3", season=2024, margin=1.0, actual=10.0, elo=0.52, logit=0.51),
            _pred("g4", season=2024, margin=-1.0, actual=-8.0, elo=0.48, logit=0.49),
        ]
        market = {"g1": 0.90, "g2": 0.85, "g3": 0.88, "g4": 0.12}
        weights = {"ridge": 0.0, "internal_elo": 1.0, "logit": 0.0}
        sgs = _build_same_game_set(
            fit_predictions=preds,
            market_home=market,
            sigma_0=14.0,
            ensemble_weights=weights,
            market_min_books=1,
            market_weight_floor=0.75,
        )
        assert sgs is not None
        assert sgs.beat_market is False
        assert sgs.market_weight_floor == pytest.approx(0.75)
        assert sgs.blend_fitted is not None
        assert sgs.blend_pure_market is not None
        assert sgs.blend_pure_model is not None
        assert sgs.blend_at_floor is not None
        assert sgs.blend_weight_by_season is not None
        assert sgs.blend_weight_by_season[2023] == pytest.approx(0.75)
        assert sgs.blend_weight_by_season[2024] == pytest.approx(1.0)
        # Floor on the no-prior season pulls some model noise in, so overall
        # blend can be slightly worse than pure market — that is honest, not
        # a gate pass. Fitted blend must still beat pure model when market wins.
        assert sgs.blend_fitted.log_loss < sgs.blend_pure_model.log_loss
        assert sgs.blend_pure_market.log_loss == pytest.approx(sgs.market.log_loss)
        assert sgs.blend_pure_model.log_loss == pytest.approx(sgs.ensemble.log_loss)
        assert sgs.blend_non_worse_than_market is not None
        assert sgs.blend_gate_interpretation.startswith("pure_model_must_beat")
