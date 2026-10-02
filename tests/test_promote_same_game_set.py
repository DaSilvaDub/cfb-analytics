"""Same-game-set beat-market + median model-vs-close CLV (promotion evidence)."""

from __future__ import annotations

import json

import pytest

from cfb_analytics.backtest.harness import GamePrediction
from cfb_analytics.backtest.moneyline import (
    SameGameSetEvidence,
    _build_same_game_set,
    median_model_vs_close_clv,
    model_vs_close_clv,
)
from cfb_analytics.backtest.promote import (
    PromotionDecision,
    evaluate_promotion,
    write_promotion_result,
)


def _pred(
    game_id: str,
    *,
    season: int = 2024,
    margin: float = 7.0,
    actual: float = 10.0,
    elo_home: float | None = 1600.0,
    elo_away: float | None = 1500.0,
    internal_elo: float | None = 0.62,
    logit: float | None = 0.58,
) -> GamePrediction:
    return GamePrediction(
        game_id=game_id,
        season=season,
        week=2,
        home_team_id="h",
        away_team_id="a",
        neutral_site=False,
        predicted_margin=margin,
        actual_margin=actual,
        elo_home_rating=elo_home,
        elo_away_rating=elo_away,
        internal_elo_win_prob=internal_elo,
        logit_win_prob=logit,
    )


class TestModelVsCloseClv:
    def test_home_favorite_close_moves_toward_bet_is_positive(self):
        # Model favors home at 0.60; close 0.65 => +0.05
        assert model_vs_close_clv(0.60, 0.65) == pytest.approx(0.05)

    def test_away_favorite_close_moves_toward_bet_is_positive(self):
        # Model favors away (p_home=0.40); close p_home=0.35 => away close higher
        assert model_vs_close_clv(0.40, 0.35) == pytest.approx(0.05)

    def test_overconfident_home_favorite_is_negative(self):
        assert model_vs_close_clv(0.80, 0.70) == pytest.approx(-0.10)

    def test_median_sign_positive(self):
        pairs = [(0.55, 0.60), (0.55, 0.58), (0.40, 0.35)]
        med = median_model_vs_close_clv(pairs)
        assert med is not None and med > 0

    def test_median_sign_negative(self):
        pairs = [(0.80, 0.70), (0.75, 0.65), (0.20, 0.30)]
        med = median_model_vs_close_clv(pairs)
        assert med is not None and med < 0

    def test_empty_median_is_none(self):
        assert median_model_vs_close_clv([]) is None


class TestSameGameSetLoglossCompare:
    def test_identical_set_beat_market_when_ensemble_sharper(self):
        # Three games: market flat 0.5; ridge/ensemble more confident and correct.
        preds = [
            _pred("g1", margin=20.0, actual=14.0, internal_elo=0.85, logit=0.80),
            _pred("g2", margin=20.0, actual=21.0, internal_elo=0.85, logit=0.80),
            _pred("g3", margin=-20.0, actual=-14.0, internal_elo=0.15, logit=0.20),
        ]
        market = {"g1": 0.50, "g2": 0.50, "g3": 0.50}
        # Force ensemble to lean on internal_elo (sharp + correct).
        weights = {"ridge": 0.0, "internal_elo": 1.0, "logit": 0.0}
        sgs = _build_same_game_set(
            fit_predictions=preds,
            market_home=market,
            sigma_0=14.0,
            ensemble_weights=weights,
            market_min_books=1,
        )
        assert sgs is not None
        assert sgs.n_full == 3
        assert sgs.n_overlap == 3
        assert sgs.n_skipped == 0
        assert sgs.market is not None and sgs.ensemble is not None
        assert sgs.market.n_games == sgs.ensemble.n_games == 3
        assert sgs.beat_market is True
        assert sgs.ensemble.log_loss < sgs.market.log_loss

    def test_identical_set_does_not_beat_when_market_sharper(self):
        preds = [
            _pred("g1", margin=1.0, actual=14.0, internal_elo=0.52, logit=0.51),
            _pred("g2", margin=1.0, actual=7.0, internal_elo=0.52, logit=0.51),
            _pred("g3", margin=-1.0, actual=-3.0, internal_elo=0.48, logit=0.49),
        ]
        # Market is confidently correct; ensemble near coin-flip.
        market = {"g1": 0.90, "g2": 0.90, "g3": 0.10}
        weights = {"ridge": 0.0, "internal_elo": 1.0, "logit": 0.0}
        sgs = _build_same_game_set(
            fit_predictions=preds,
            market_home=market,
            sigma_0=14.0,
            ensemble_weights=weights,
            market_min_books=1,
        )
        assert sgs is not None
        assert sgs.beat_market is False
        assert sgs.ensemble.log_loss > sgs.market.log_loss

    def test_skipped_games_are_counted_not_silent(self):
        preds = [_pred("g1"), _pred("g2"), _pred("g3")]
        market = {"g1": 0.55}  # only one priced
        weights = {"ridge": 1.0, "internal_elo": 0.0, "logit": 0.0}
        sgs = _build_same_game_set(
            fit_predictions=preds,
            market_home=market,
            sigma_0=14.0,
            ensemble_weights=weights,
            market_min_books=1,
        )
        assert sgs is not None
        assert sgs.n_full == 3
        assert sgs.n_overlap == 1
        assert sgs.n_skipped == 2

    def test_empty_overlap_still_reports_counts(self):
        preds = [_pred("g1"), _pred("g2")]
        sgs = _build_same_game_set(
            fit_predictions=preds,
            market_home={},
            sigma_0=14.0,
            ensemble_weights={"ridge": 1.0, "internal_elo": 0.0, "logit": 0.0},
            market_min_books=1,
        )
        assert sgs is not None
        assert sgs.n_overlap == 0
        assert sgs.n_skipped == 2
        assert sgs.beat_market is None
        assert sgs.median_clv is None


class TestPromoteGatesFailClosed:
    def _minimal_report_with_sgs(self, sgs: SameGameSetEvidence):
        from cfb_analytics.backtest.moneyline import MoneylineBacktestReport, SliceMetrics

        # n_games on seasons must be >0 for evidence n_full path; use sgs.n_full
        seasons = SliceMetrics(
            label="seasons",
            n_games=sgs.n_full,
            brier=0.2,
            log_loss=0.5,
            confidence_buckets=[],
            reliability=sgs.ensemble.reliability if sgs.ensemble else [],
        )
        return MoneylineBacktestReport(
            sigma_0=14.0,
            n_games_calibrated=sgs.n_full,
            seasons=seasons,
            stress=None,
            skipped_insufficient_history=0,
            skipped_unrated_team=0,
            elo_seasons=None,
            elo_stress=None,
            skipped_elo_unrated=0,
            internal_elo_seasons=None,
            internal_elo_stress=None,
            skipped_internal_elo_unrated=0,
            logit_seasons=None,
            logit_stress=None,
            skipped_logit_unrated=0,
            ensemble_weights={"ridge": 1.0, "internal_elo": 0.0, "logit": 0.0},
            ensemble_seasons=sgs.ensemble,
            ensemble_stress=None,
            market_seasons=sgs.market,
            same_game_set=sgs,
        )

    def test_small_n_fails_closed(self):
        preds = [
            _pred("g1", margin=20.0, actual=14.0, internal_elo=0.85, logit=0.80),
            _pred("g2", margin=20.0, actual=21.0, internal_elo=0.85, logit=0.80),
        ]
        market = {"g1": 0.5, "g2": 0.5}
        sgs = _build_same_game_set(
            fit_predictions=preds,
            market_home=market,
            sigma_0=14.0,
            ensemble_weights={"ridge": 0.0, "internal_elo": 1.0, "logit": 0.0},
            market_min_books=1,
        )
        report = self._minimal_report_with_sgs(sgs)
        cfg = {
            "min_settled_games": 1500,
            "min_seasons_backtested": 3,
            "max_abs_calibration_gap": 0.03,
            "min_bucket_n_for_gap_test": 100,
            "require_oos_logloss_beat_market": True,
            "require_median_clv_non_negative": True,
        }
        decision = evaluate_promotion(report, seasons=(2024,), promotion_cfg=cfg)
        assert decision.passed is False
        assert decision.status == "shadow"
        assert any("n_overlap" in f for f in decision.failures)

    def test_write_promotion_keeps_shadow_on_fail(self, tmp_path, monkeypatch):
        from cfb_analytics import config as cfg_mod
        from cfb_analytics import paths as paths_mod

        promo_path = tmp_path / "promotion.json"
        promo_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "shadow",
                    "changed_utc": None,
                    "min_settled_games": 1500,
                    "min_seasons_backtested": 3,
                    "max_abs_calibration_gap": 0.03,
                    "min_bucket_n_for_gap_test": 100,
                    "require_oos_logloss_beat_market": True,
                    "require_median_clv_non_negative": True,
                    "evidence": None,
                    "_comment": "test",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(paths_mod, "CONFIG_DIR", tmp_path)
        cfg_mod._read_json.cache_clear()

        decision = PromotionDecision(
            passed=False,
            status="shadow",
            failures=("n_overlap too small",),
            evidence={"gate_passed": False, "n_overlap": 2},
        )
        write_promotion_result(decision)
        written = json.loads(promo_path.read_text(encoding="utf-8"))
        assert written["status"] == "shadow"
        assert written["evidence"]["n_overlap"] == 2
        cfg_mod._read_json.cache_clear()
