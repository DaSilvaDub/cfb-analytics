"""Unit tests for walk-forward ensemble probability calibration."""

from __future__ import annotations

import math

import pytest

from cfb_analytics.backtest.prob_calibrate import (
    IDENTITY,
    ProbCalibrator,
    TimedProb,
    fit_platt,
    fit_temperature,
    walk_forward_calibrate,
)
from cfb_analytics.models.logistic import logit, sigmoid


class TestIdentityNearPerfect:
    def test_identity_leaves_probs_unchanged(self):
        cal = IDENTITY
        for p in (0.1, 0.5, 0.9):
            assert cal.apply(p) == pytest.approx(p, abs=1e-9)

    def test_temperature_one_is_identity(self):
        cal = ProbCalibrator(method="temperature", temperature=1.0, n_fit=10)
        for p in (0.2, 0.55, 0.8):
            assert cal.apply(p) == pytest.approx(p, abs=1e-9)

    def test_platt_identity_params(self):
        cal = ProbCalibrator(method="platt", platt_a=1.0, platt_b=0.0, n_fit=10)
        for p in (0.2, 0.55, 0.8):
            assert cal.apply(p) == pytest.approx(p, abs=1e-9)

    def test_fit_temperature_on_perfectly_calibrated_stays_near_one(self):
        # Synthetic: p itself is the Bernoulli parameter; large n => T~1.
        probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 40
        # Deterministic outcomes matching rates approximately via threshold.
        outcomes = []
        for i, p in enumerate(probs):
            # Alternate so empirical rate tracks p in aggregate bins.
            outcomes.append((i % 10) < int(round(p * 10)))
        cal = fit_temperature(probs, outcomes)
        assert cal.method == "temperature"
        assert 0.7 <= cal.temperature <= 1.4


class TestReducesSyntheticMiscalibration:
    def test_temperature_softens_overconfident_probs(self):
        # Model claims extremes; reality is closer to 0.5.
        raw = [0.9] * 100 + [0.1] * 100
        # Outcomes ~55% / 45% instead of 90%/10%.
        outcomes = [True] * 55 + [False] * 45 + [True] * 45 + [False] * 55
        cal = fit_temperature(raw, outcomes)
        assert cal.temperature > 1.0  # softens
        assert abs(cal.apply(0.9) - 0.5) < abs(0.9 - 0.5)

    def test_platt_reduces_systematic_bias(self):
        # Raw probs systematically too high by ~0.1 in logit space.
        true_ps = [0.3, 0.4, 0.5, 0.6, 0.7] * 50
        raw = [sigmoid(logit(p) + 0.8) for p in true_ps]
        outcomes = []
        for i, p in enumerate(true_ps):
            outcomes.append((i % 10) < int(round(p * 10)))
        cal = fit_platt(raw, outcomes)
        assert cal.method == "platt"
        # Calibrated mid should move toward true.
        mid_raw = sigmoid(logit(0.5) + 0.8)
        mid_cal = cal.apply(mid_raw)
        assert abs(mid_cal - 0.5) < abs(mid_raw - 0.5)


class TestNoFutureLabelsInFit:
    def test_walk_forward_earlier_fold_cannot_see_later_labels(self):
        # Season 1 week 1: all 0.9, all win -> if peeked, T would sharpen;
        # week 2 should still be identity if min_fit_n is huge.
        rows = [
            TimedProb(2023, 1, 0.9, True, key=f"a{i}") for i in range(5)
        ] + [
            TimedProb(2023, 2, 0.9, False, key=f"b{i}") for i in range(5)
        ]
        out = walk_forward_calibrate(rows, method="temperature", min_fit_n=1000, fold="week")
        # Both folds identity because prior never reaches min_fit_n before week 2
        # (only 5 priors before week 2).
        for raw, cal, used in out:
            assert used.method == "identity"
            assert cal == pytest.approx(raw)

    def test_later_fold_fits_only_on_prior_games(self):
        # Build prior fold with overconfident wrong extremes so T>1 when fit.
        prior = (
            [TimedProb(2023, 1, 0.95, True, key=f"w{i}") for i in range(30)]
            + [TimedProb(2023, 1, 0.95, False, key=f"l{i}") for i in range(70)]
        )
        later = [TimedProb(2023, 2, 0.95, True, key="x")]
        out = walk_forward_calibrate(
            prior + later, method="temperature", min_fit_n=50, fold="week"
        )
        # First 100 rows: identity (no prior yet for week 1).
        for raw, cal, used in out[:100]:
            assert used.method == "identity"
        # Week 2 uses a fit from week 1 only.
        raw, cal, used = out[100]
        assert used.method == "temperature"
        assert used.n_fit == 100
        assert used.temperature > 1.0
        assert cal < raw  # softened

    def test_future_season_not_in_earlier_fit_n(self):
        rows = (
            [TimedProb(2023, 1, 0.6, True, key=f"a{i}") for i in range(100)]
            + [TimedProb(2024, 1, 0.6, False, key=f"b{i}") for i in range(10)]
        )
        out = walk_forward_calibrate(rows, method="temperature", min_fit_n=50, fold="week")
        # 2024 fold must have n_fit == 100 (only 2023), never 110.
        for _raw, _cal, used in out[100:]:
            assert used.n_fit == 100


    def test_season_fold_excludes_same_season(self):
        rows = (
            [TimedProb(2023, w, 0.9, False, key=f"a{w}-{i}") for w in range(1, 5) for i in range(30)]
            + [TimedProb(2024, 1, 0.9, True, key="b")]
        )
        out = walk_forward_calibrate(rows, method="temperature", min_fit_n=50, fold="season")
        # All 2023 rows share identity (no prior season).
        for raw, cal, used in out[:-1]:
            assert used.method == "identity"
        raw, cal, used = out[-1]
        assert used.method == "temperature"
        assert used.n_fit == 120  # all of 2023, not partial weeks
