from __future__ import annotations

import pytest

from cfb_analytics.backtest.metrics import (
    brier_score,
    bucketed_win_rate,
    favorite_side,
    log_loss,
    reliability_curve,
    wilson_interval,
)


class TestBrierScore:
    def test_perfect_predictions_score_zero(self):
        assert brier_score([(1.0, True), (0.0, False)]) == pytest.approx(0.0)

    def test_maximally_wrong_predictions_score_one(self):
        assert brier_score([(1.0, False), (0.0, True)]) == pytest.approx(1.0)

    def test_a_coin_flip_on_a_coin_flip_scores_a_quarter(self):
        assert brier_score([(0.5, True), (0.5, False)]) == pytest.approx(0.25)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            brier_score([])


class TestLogLoss:
    def test_confident_and_correct_scores_near_zero(self):
        assert log_loss([(0.99, True)] * 10) < 0.02

    def test_confident_and_wrong_is_large_but_finite(self):
        # Without clipping this would be math.log(0) = -inf.
        import math
        assert math.isfinite(log_loss([(1.0, False)]))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            log_loss([])


class TestFavoriteSide:
    def test_home_favorite_passes_through_unchanged(self):
        assert favorite_side(0.7, True) == (0.7, True)
        assert favorite_side(0.7, False) == (0.7, False)

    def test_away_favorite_is_flipped_to_the_favored_perspective(self):
        prob, won = favorite_side(0.3, home_won=False)  # away favored, away won
        assert prob == pytest.approx(0.7)
        assert won is True

    def test_away_favorite_that_loses_flips_to_a_favorite_loss(self):
        prob, won = favorite_side(0.3, home_won=True)  # away favored, home upset
        assert prob == pytest.approx(0.7)
        assert won is False


class TestWilsonInterval:
    def test_interval_contains_the_point_estimate(self):
        low, high = wilson_interval(80, 100)
        assert low < 0.8 < high

    def test_more_data_narrows_the_interval(self):
        narrow_low, narrow_high = wilson_interval(800, 1000)
        wide_low, wide_high = wilson_interval(8, 10)
        assert (narrow_high - narrow_low) < (wide_high - wide_low)

    def test_zero_n_raises(self):
        with pytest.raises(ValueError):
            wilson_interval(0, 0)


class TestBucketedWinRate:
    def test_games_are_sorted_into_the_matching_confidence_band(self):
        predictions = [(0.82, True), (0.82, False), (0.96, True)]
        results = {b.label: b for b in bucketed_win_rate(predictions)}
        assert results["80-84.9%"].n == 2
        assert results["95%+"].n == 1

    def test_an_empty_bucket_reports_n_zero_not_a_crash(self):
        results = {b.label: b for b in bucketed_win_rate([(0.96, True)])}
        assert results["80-84.9%"].n == 0
        assert results["80-84.9%"].win_rate is None

    def test_win_rate_and_wilson_bounds_are_consistent(self):
        predictions = [(0.9, True), (0.9, True), (0.9, False), (0.9, True)]
        bucket = next(b for b in bucketed_win_rate(predictions) if b.label == "90-92.4%")
        assert bucket.n == 4
        assert bucket.win_rate == pytest.approx(0.75)
        assert bucket.wilson_low is not None and bucket.wilson_high is not None
        assert bucket.wilson_low < 0.75 < bucket.wilson_high


class TestReliabilityCurve:
    def test_covers_the_full_probability_range_in_deciles(self):
        curve = reliability_curve([(0.05, True), (0.95, True)])
        assert len(curve) == 10
        assert curve[0].n == 1  # the 0-10% bucket
        assert curve[-1].n == 1  # the 90-100% bucket

    def test_a_probability_of_exactly_one_lands_in_the_last_bucket(self):
        curve = reliability_curve([(1.0, True)])
        assert curve[-1].n == 1
