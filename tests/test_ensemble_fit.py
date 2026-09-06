from __future__ import annotations

import pytest

from cfb_analytics.backtest.ensemble_fit import _simplex_grid, fit_ensemble_weights


class TestSimplexGrid:
    def test_every_combination_sums_to_one(self):
        for combo in _simplex_grid(3, 0.1):
            assert combo[0] + combo[1] + combo[2] == pytest.approx(1.0)

    def test_every_value_is_a_nonnegative_multiple_of_step(self):
        for combo in _simplex_grid(3, 0.25):
            for value in combo:
                assert value >= 0.0

    def test_combination_count_matches_the_stars_and_bars_formula(self):
        # C(steps + n - 1, n - 1) combinations for n members at this step.
        import math
        n, step = 3, 0.2
        steps = round(1 / step)
        expected = math.comb(steps + n - 1, n - 1)
        assert len(list(_simplex_grid(n, step))) == expected

    def test_two_members_includes_both_pure_endpoints(self):
        combos = list(_simplex_grid(2, 0.5))
        assert (1.0, 0.0) in combos
        assert (0.0, 1.0) in combos


class TestFitEnsembleWeights:
    def test_a_perfect_member_gets_all_the_weight(self):
        """"perfect" always predicts the true outcome with high confidence;
        "noise" predicts a coin flip every time -- the fit should put
        (near-)all weight on "perfect"."""
        probs = [{"perfect": 0.95 if outcome else 0.05, "noise": 0.5} for outcome in
                 [True, False, True, True, False, True, False, False, True, True] * 5]
        outcomes = [p["perfect"] > 0.5 for p in probs]
        weights = fit_ensemble_weights(probs, outcomes, ("perfect", "noise"), grid_step=0.1)
        assert weights["perfect"] >= 0.9

    def test_weights_sum_to_one(self):
        probs = [{"a": 0.6, "b": 0.4} for _ in range(20)]
        outcomes = [True] * 10 + [False] * 10
        weights = fit_ensemble_weights(probs, outcomes, ("a", "b"), grid_step=0.2)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_a_game_missing_one_member_still_contributes(self):
        probs = [{"a": 0.9, "b": 0.9}] * 10 + [{"a": 0.9}] * 10  # "b" absent for half
        outcomes = [True] * 20
        weights = fit_ensemble_weights(probs, outcomes, ("a", "b"), grid_step=0.25)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            fit_ensemble_weights([{"a": 0.5}], [True, False], ("a",))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            fit_ensemble_weights([], [], ("a", "b"))
