from __future__ import annotations

import pytest

from cfb_analytics.models.ensemble import pool_probabilities


class TestPoolProbabilities:
    def test_a_single_member_passes_through_unchanged(self):
        result = pool_probabilities({"ridge": 0.7}, {"ridge": 1.0})
        assert result == pytest.approx(0.7)

    def test_equal_weights_on_two_agreeing_members_reproduces_their_value(self):
        result = pool_probabilities({"ridge": 0.7, "elo": 0.7}, {"ridge": 0.5, "elo": 0.5})
        assert result == pytest.approx(0.7)

    def test_two_members_disagreeing_symmetrically_around_a_coin_flip_gives_a_coin_flip(self):
        # logit(0.7) and logit(0.3) are exact negatives of each other.
        result = pool_probabilities({"a": 0.7, "b": 0.3}, {"a": 0.5, "b": 0.5})
        assert result == pytest.approx(0.5)

    def test_a_higher_weight_pulls_the_blend_toward_that_member(self):
        heavy_a = pool_probabilities({"a": 0.9, "b": 0.5}, {"a": 0.9, "b": 0.1})
        heavy_b = pool_probabilities({"a": 0.9, "b": 0.5}, {"a": 0.1, "b": 0.9})
        assert heavy_a > heavy_b

    def test_dropping_an_absent_member_renormalizes_rather_than_diluting(self):
        """Weights (0.5, 0.5) fitted for a 2-member ensemble, but only one
        member has a prediction for this game -- the present member's
        weight must effectively become 1.0, not stay diluted at 0.5."""
        only_a_available = {"a": 0.5}  # caller filters the fitted {a:0.5,b:0.5} to what's available
        result = pool_probabilities({"a": 0.8}, only_a_available)
        assert result == pytest.approx(0.8)
        # Sanity: this matches what a would-be single-member call gives.
        assert result == pool_probabilities({"a": 0.8}, {"a": 1.0})

    def test_empty_weights_raises(self):
        with pytest.raises(ValueError):
            pool_probabilities({"a": 0.7}, {})

    def test_a_weighted_member_missing_from_probs_raises(self):
        with pytest.raises(ValueError):
            pool_probabilities({"a": 0.7}, {"a": 0.5, "b": 0.5})

    def test_zero_total_weight_raises(self):
        with pytest.raises(ValueError):
            pool_probabilities({"a": 0.7, "b": 0.3}, {"a": 0.0, "b": 0.0})

    def test_negative_weight_summing_to_zero_or_less_raises(self):
        with pytest.raises(ValueError):
            pool_probabilities({"a": 0.7, "b": 0.3}, {"a": 1.0, "b": -1.0})
