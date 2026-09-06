from __future__ import annotations

import math
import random

import pytest

from cfb_analytics.models.logistic import DEFAULT_L2_LAMBDA, fit_logistic, logit, sigmoid


class TestSigmoid:
    def test_zero_is_a_coin_flip(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_approaches_one(self):
        assert sigmoid(50.0) > 0.999999

    def test_large_negative_approaches_zero(self):
        assert sigmoid(-50.0) < 0.000001

    def test_does_not_overflow_on_extreme_input(self):
        # A naive 1/(1+exp(-x)) would OverflowError on exp(-(-1000)).
        assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-9)
        assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-9)

    def test_is_the_inverse_of_logit(self):
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            assert sigmoid(logit(p)) == pytest.approx(p)


class TestLogit:
    def test_a_coin_flip_is_zero(self):
        assert logit(0.5) == pytest.approx(0.0)

    def test_rejects_zero_and_one(self):
        with pytest.raises(ValueError):
            logit(0.0)
        with pytest.raises(ValueError):
            logit(1.0)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            logit(1.5)


def _synthetic_dataset(true_intercept, true_weights, n=400, seed=0):
    """Generate data exactly from the model's own equation -- the same
    parameter-recovery methodology test_ridge.py uses, and for the same
    reason: this is the correct way to check a regression fit recovers a
    known answer, not an approximation of one."""
    rng = random.Random(seed)
    rows = []
    outcomes = []
    for _ in range(n):
        row = [rng.uniform(-2.0, 2.0) for _ in true_weights]
        z = true_intercept + sum(w * x for w, x in zip(true_weights, row, strict=True))
        p = sigmoid(z)
        rows.append(row)
        outcomes.append(rng.random() < p)
    return rows, outcomes


class TestFitLogistic:
    def test_below_min_n_returns_insufficient_data(self):
        rows, outcomes = _synthetic_dataset(0.0, [1.0], n=10)
        result = fit_logistic(rows, outcomes, min_n=30)
        assert result.status == "insufficient_data"
        assert result.weights == []

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            fit_logistic([[1.0], [2.0]], [True], min_n=1)

    def test_inconsistent_row_widths_raise(self):
        with pytest.raises(ValueError):
            fit_logistic([[1.0, 2.0], [1.0]], [True, False], min_n=1)

    def test_recovers_a_known_single_feature_coefficient(self):
        rows, outcomes = _synthetic_dataset(0.2, [1.5], n=2000, seed=1)
        result = fit_logistic(rows, outcomes, l2_lambda=0.01, min_n=1)
        assert result.status == "active"
        assert result.weights[0] == pytest.approx(0.2, abs=0.2)  # intercept
        assert result.weights[1] == pytest.approx(1.5, abs=0.2)  # feature weight

    def test_recovers_known_multi_feature_coefficients(self):
        rows, outcomes = _synthetic_dataset(-0.3, [1.2, -0.8, 0.4], n=3000, seed=2)
        result = fit_logistic(rows, outcomes, l2_lambda=0.01, min_n=1)
        assert result.status == "active"
        assert result.weights[1] == pytest.approx(1.2, abs=0.25)
        assert result.weights[2] == pytest.approx(-0.8, abs=0.25)
        assert result.weights[3] == pytest.approx(0.4, abs=0.25)

    def test_a_feature_with_zero_true_effect_recovers_near_zero(self):
        rows, outcomes = _synthetic_dataset(0.0, [2.0, 0.0], n=3000, seed=3)
        result = fit_logistic(rows, outcomes, l2_lambda=0.01, min_n=1)
        assert result.weights[2] == pytest.approx(0.0, abs=0.2)

    def test_larger_l2_lambda_shrinks_weights_toward_zero(self):
        rows, outcomes = _synthetic_dataset(0.0, [2.0], n=2000, seed=4)
        loose = fit_logistic(rows, outcomes, l2_lambda=0.01, min_n=1)
        tight = fit_logistic(rows, outcomes, l2_lambda=500.0, min_n=1)
        assert abs(tight.weights[1]) < abs(loose.weights[1])

    def test_l2_lambda_does_not_shrink_the_intercept(self):
        """A dataset with NO features at all and a skewed outcome rate: the
        fitted intercept should reflect that rate almost exactly regardless
        of how large l2_lambda is, since the penalty never touches index 0."""
        rng = random.Random(5)
        outcomes = [rng.random() < 0.8 for _ in range(2000)]
        rows = [[] for _ in outcomes]
        result = fit_logistic(rows, outcomes, l2_lambda=1000.0, min_n=1)
        assert sigmoid(result.weights[0]) == pytest.approx(0.8, abs=0.05)


class TestLogisticFitProbability:
    def test_returns_none_when_not_active(self):
        from cfb_analytics.models.logistic import LogisticFit

        result = LogisticFit(status="insufficient_data", n=5)
        assert result.probability([1.0]) is None

    def test_returns_none_on_a_feature_count_mismatch(self):
        rows, outcomes = _synthetic_dataset(0.0, [1.0, 1.0], n=200, seed=6)
        result = fit_logistic(rows, outcomes, min_n=1)
        assert result.probability([1.0]) is None

    def test_matches_sigmoid_of_the_linear_combination(self):
        rows, outcomes = _synthetic_dataset(0.1, [1.0, -0.5], n=2000, seed=7)
        result = fit_logistic(rows, outcomes, l2_lambda=0.01, min_n=1)
        features = [1.0, 1.0]
        expected = sigmoid(
            result.weights[0] + result.weights[1] * features[0] + result.weights[2] * features[1]
        )
        assert result.probability(features) == pytest.approx(expected)


class TestDefaultLambda:
    def test_default_is_a_positive_finite_constant(self):
        assert DEFAULT_L2_LAMBDA > 0
        assert math.isfinite(DEFAULT_L2_LAMBDA)
