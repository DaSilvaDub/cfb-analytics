from __future__ import annotations

import pytest

from cfb_analytics.models.shrinkage import (
    ShrinkageCoefficients,
    blend_toward_prior,
    prior_rating,
    zscore,
)


class TestPriorRating:
    def test_combines_all_three_terms(self):
        coeffs = ShrinkageCoefficients(a=1.0, b=2.0, c=3.0)
        assert prior_rating(10.0, 1.0, 1.0, coeffs=coeffs) == pytest.approx(15.0)

    def test_missing_previous_season_contributes_zero_not_a_crash(self):
        coeffs = ShrinkageCoefficients(a=1.0, b=2.0, c=3.0)
        assert prior_rating(None, 1.0, 1.0, coeffs=coeffs) == pytest.approx(5.0)

    def test_missing_talent_and_returning_contributes_zero(self):
        coeffs = ShrinkageCoefficients(a=1.0, b=2.0, c=3.0)
        assert prior_rating(10.0, None, None, coeffs=coeffs) == pytest.approx(10.0)

    def test_all_inputs_missing_gives_a_zero_prior(self):
        assert prior_rating(None, None, None) == 0.0

    def test_default_coefficients_are_used_when_none_given(self):
        assert prior_rating(0.0, 0.0, 0.0) == 0.0


class TestBlendTowardPrior:
    def test_zero_games_this_season_is_entirely_the_prior(self):
        assert blend_toward_prior(fit_value=99.0, prior_value=5.0, n_games=0, k=4.0) == 5.0

    def test_many_games_this_season_is_almost_entirely_the_fit(self):
        blended = blend_toward_prior(fit_value=10.0, prior_value=0.0, n_games=1000, k=4.0)
        assert blended == pytest.approx(10.0, abs=0.05)

    def test_equal_n_and_k_averages_them_evenly(self):
        blended = blend_toward_prior(fit_value=10.0, prior_value=0.0, n_games=4, k=4.0)
        assert blended == pytest.approx(5.0)

    def test_negative_n_games_raises(self):
        with pytest.raises(ValueError):
            blend_toward_prior(fit_value=1.0, prior_value=1.0, n_games=-1, k=4.0)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            blend_toward_prior(fit_value=1.0, prior_value=1.0, n_games=1, k=-1.0)


class TestZscore:
    def test_recovers_a_known_zscore(self):
        # Values 10, 20, 30: mean 20, population stdev ~8.165.
        result = zscore({"a": 10.0, "b": 20.0, "c": 30.0})
        assert result["a"] == pytest.approx(-1.2247, abs=1e-3)
        assert result["b"] == pytest.approx(0.0, abs=1e-9)
        assert result["c"] == pytest.approx(1.2247, abs=1e-3)

    def test_a_team_absent_from_the_input_is_absent_from_the_output(self):
        result = zscore({"a": 10.0, "b": 20.0})
        assert "c" not in result

    def test_fewer_than_two_values_returns_zero_for_everyone(self):
        assert zscore({"a": 10.0}) == {"a": 0.0}
        assert zscore({}) == {}

    def test_zero_variance_returns_zero_for_everyone_not_a_divide_by_zero(self):
        assert zscore({"a": 5.0, "b": 5.0, "c": 5.0}) == {"a": 0.0, "b": 0.0, "c": 0.0}
