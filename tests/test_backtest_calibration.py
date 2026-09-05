from __future__ import annotations

import math

import pytest

from cfb_analytics.backtest.calibration import calibrate_sigma, margin_to_prob


class TestMarginToProb:
    def test_zero_margin_is_a_coin_flip(self):
        assert margin_to_prob(0.0, sigma=14.0) == pytest.approx(0.5)

    def test_a_large_positive_margin_approaches_certainty(self):
        assert margin_to_prob(100.0, sigma=14.0) > 0.999

    def test_a_large_negative_margin_approaches_zero(self):
        assert margin_to_prob(-100.0, sigma=14.0) < 0.001

    def test_symmetric_margins_are_complementary_probabilities(self):
        p_pos = margin_to_prob(7.0, sigma=14.0)
        p_neg = margin_to_prob(-7.0, sigma=14.0)
        assert p_pos + p_neg == pytest.approx(1.0)

    def test_a_wider_sigma_pulls_probability_toward_a_coin_flip(self):
        tight = margin_to_prob(7.0, sigma=7.0)
        wide = margin_to_prob(7.0, sigma=28.0)
        assert 0.5 < wide < tight

    def test_matches_the_hand_computed_standard_normal_cdf(self):
        # Phi(1) ~= 0.8413 (a one-sigma margin), a well-known reference value.
        assert margin_to_prob(14.0, sigma=14.0) == pytest.approx(0.8413, abs=1e-4)

    def test_nonpositive_sigma_raises(self):
        with pytest.raises(ValueError):
            margin_to_prob(3.0, sigma=0.0)
        with pytest.raises(ValueError):
            margin_to_prob(3.0, sigma=-1.0)


class TestCalibrateSigma:
    def test_recovers_the_known_stdev_of_synthetic_residuals(self):
        # A symmetric residual set with a known population stdev of exactly 2.0.
        residuals = [-2.0, -2.0, 2.0, 2.0]
        assert calibrate_sigma(residuals) == pytest.approx(math.sqrt(16 / 3), abs=1e-9)

    def test_zero_residuals_give_zero_sigma_not_a_crash(self):
        assert calibrate_sigma([0.0, 0.0, 0.0]) == 0.0

    def test_fewer_than_two_residuals_raises(self):
        with pytest.raises(ValueError):
            calibrate_sigma([1.0])
        with pytest.raises(ValueError):
            calibrate_sigma([])
