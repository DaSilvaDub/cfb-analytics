from __future__ import annotations

import pytest

from cfb_analytics.backtest.elo_baseline import HFA_ELO_POINTS, elo_win_probability


class TestEloWinProbability:
    def test_equal_ratings_at_a_neutral_site_is_a_coin_flip(self):
        assert elo_win_probability(1500, 1500, neutral_site=True) == pytest.approx(0.5)

    def test_equal_ratings_at_home_favors_the_home_team(self):
        """Home-field is a real, positive edge even between otherwise-equal teams."""
        assert elo_win_probability(1500, 1500, neutral_site=False) > 0.5

    def test_neutral_site_removes_the_home_field_term(self):
        home = elo_win_probability(1500, 1500, neutral_site=False)
        neutral = elo_win_probability(1500, 1500, neutral_site=True)
        assert home > neutral

    def test_a_large_rating_gap_dominates_home_field(self):
        assert elo_win_probability(1200, 1900, neutral_site=False) < 0.1

    def test_symmetric_ratings_at_neutral_site_are_complementary(self):
        p_home = elo_win_probability(1600, 1500, neutral_site=True)
        p_away = elo_win_probability(1500, 1600, neutral_site=True)
        assert p_home + p_away == pytest.approx(1.0)

    def test_default_hfa_matches_the_plan_constant(self):
        """Plan section 6.2: HFA = 60 Elo points. Not re-derived here -- this
        guards against silently drifting from the documented constant."""
        assert HFA_ELO_POINTS == 60.0

    def test_custom_hfa_is_respected(self):
        default_prob = elo_win_probability(1500, 1500)
        bigger_hfa_prob = elo_win_probability(1500, 1500, hfa=200.0)
        assert bigger_hfa_prob > default_prob
