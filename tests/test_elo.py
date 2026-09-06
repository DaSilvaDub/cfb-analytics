"""Tests for the sequential internal Elo model (plan section 6.2)."""

from __future__ import annotations

import pytest

from cfb_analytics.models.elo import (
    DEFAULT_INITIAL_RATING,
    DEFAULT_K,
    EloRatings,
    TeamEloState,
    expected_score,
    fit_elo,
    margin_of_victory_multiplier,
    preseason_rating,
    update_ratings,
)


def game(home, away, home_points, away_points, kickoff_utc, neutral_site=False):
    return {
        "home_team_id": home, "away_team_id": away,
        "home_points": home_points, "away_points": away_points,
        "kickoff_utc": kickoff_utc, "neutral_site": neutral_site,
    }


class TestExpectedScore:
    def test_equal_ratings_with_no_hfa_is_a_coin_flip(self):
        assert expected_score(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_is_favored(self):
        assert expected_score(1700, 1500) > 0.5

    def test_hfa_helps_the_first_argument(self):
        assert expected_score(1500, 1500, hfa=60) > 0.5

    def test_symmetric_with_the_arguments_swapped(self):
        p_a = expected_score(1600, 1500)
        p_b = expected_score(1500, 1600)
        assert p_a + p_b == pytest.approx(1.0)

    def test_matches_a_hand_computed_value(self):
        # A 400-point gap gives exactly 1/(1+10^-1) = 10/11.
        assert expected_score(1900, 1500) == pytest.approx(10 / 11, abs=1e-9)


class TestMarginOfVictoryMultiplier:
    def test_larger_margin_gives_a_larger_multiplier(self):
        small = margin_of_victory_multiplier(25.0, 3, 1600, 1500)
        large = margin_of_victory_multiplier(25.0, 40, 1600, 1500)
        assert large > small

    def test_zero_margin_gives_zero_since_ln_one_is_zero(self):
        assert margin_of_victory_multiplier(25.0, 0, 1600, 1500) == 0.0

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError):
            margin_of_victory_multiplier(25.0, -3, 1600, 1500)

    def test_an_upset_gets_a_larger_multiplier_than_a_similarly_sized_favorite_win(self):
        """The winner was rated LOWER than the loser (an upset) -- the
        denominator shrinks, amplifying the update relative to the same
        margin between equally-rated teams."""
        upset = margin_of_victory_multiplier(25.0, 14, winner_rating=1400, loser_rating=1700)
        even = margin_of_victory_multiplier(25.0, 14, winner_rating=1550, loser_rating=1550)
        assert upset > even

    def test_a_huge_favorite_blowing_out_a_huge_underdog_is_dampened(self):
        expected_blowout = margin_of_victory_multiplier(
            25.0, 40, winner_rating=1900, loser_rating=1300)
        even_blowout = margin_of_victory_multiplier(
            25.0, 40, winner_rating=1600, loser_rating=1600)
        assert expected_blowout < even_blowout

    def test_an_extreme_rating_gap_does_not_blow_up_or_flip_sign(self):
        # Nowhere near a real CFB gap, but the function must stay sane.
        result = margin_of_victory_multiplier(25.0, 10, winner_rating=100, loser_rating=5000)
        assert result > 0


class TestUpdateRatings:
    def test_is_zero_sum(self):
        new_home, new_away = update_ratings(
            1500, 1500, home_won=True, margin=10, k=25.0, hfa=60.0)
        assert (new_home - 1500) == pytest.approx(-(new_away - 1500))

    def test_the_winner_gains_and_the_loser_loses(self):
        new_home, new_away = update_ratings(
            1500, 1500, home_won=True, margin=10, k=25.0, hfa=0.0, neutral_site=True)
        assert new_home > 1500
        assert new_away < 1500

    def test_neutral_site_removes_the_hfa_term(self):
        home_field, _ = update_ratings(
            1500, 1500, home_won=True, margin=10, k=25.0, hfa=60.0, neutral_site=False)
        neutral, _ = update_ratings(
            1500, 1500, home_won=True, margin=10, k=25.0, hfa=60.0, neutral_site=True)
        # The home team's HFA-boosted win was "more expected", so it should
        # gain LESS than the identical result at a neutral site.
        assert (home_field - 1500) < (neutral - 1500)

    def test_a_massive_favorite_winning_barely_still_updates_correctly_signed(self):
        new_home, new_away = update_ratings(
            2000, 1200, home_won=True, margin=1, k=25.0, hfa=0.0, neutral_site=True)
        assert new_home > 2000  # still gains, just a little
        assert new_away < 1200


class TestFitElo:
    def test_below_min_games_returns_insufficient_history(self):
        games = [game("a", "b", 20, 10, "2023-09-02T00:00:00+00:00")]
        result = fit_elo(games, min_games=5)
        assert result.status == "insufficient_history"
        assert result.teams == {}

    def test_negative_min_games_raises(self):
        with pytest.raises(ValueError):
            fit_elo([], min_games=0)

    def test_games_missing_a_score_are_not_counted(self):
        games = [
            game("a", "b", 20, 10, "2023-09-02T00:00:00+00:00"),
            {"home_team_id": "c", "away_team_id": "d", "home_points": None,
             "away_points": 3, "kickoff_utc": "2023-09-03T00:00:00+00:00"},
        ]
        result = fit_elo(games, min_games=1)
        assert result.n_games == 1

    def test_a_tied_score_is_dropped_not_fabricated_a_winner(self):
        games = [game("a", "b", 20, 20, "2023-09-02T00:00:00+00:00")]
        result = fit_elo(games, min_games=1)
        assert result.status == "insufficient_history"

    def test_a_game_missing_kickoff_utc_is_dropped(self):
        games = [{"home_team_id": "a", "away_team_id": "b", "home_points": 20,
                   "away_points": 10, "kickoff_utc": None}]
        result = fit_elo(games, min_games=1)
        assert result.n_games == 0

    def test_new_teams_start_at_the_generic_default_rating(self):
        games = [game("a", "b", 20, 10, "2023-09-02T00:00:00+00:00")]
        result = fit_elo(games, min_games=1, hfa=0.0)
        # Symmetric matchup at the same starting rating and neutral hfa=0:
        # winner gains exactly what the loser loses, both starting at 1500.
        assert result.teams["a"].rating > DEFAULT_INITIAL_RATING
        assert result.teams["b"].rating < DEFAULT_INITIAL_RATING
        assert (result.teams["a"].rating - DEFAULT_INITIAL_RATING) == pytest.approx(
            -(result.teams["b"].rating - DEFAULT_INITIAL_RATING)
        )

    def test_initial_ratings_seed_a_teams_starting_point(self):
        games = [game("a", "b", 20, 10, "2023-09-02T00:00:00+00:00")]
        result = fit_elo(games, min_games=1, initial_ratings={"a": 2000.0, "b": 1000.0})
        # "a" was already a huge favorite (2000 vs 1000): winning by 10 is
        # the expected result and should barely move its rating.
        assert result.teams["a"].rating == pytest.approx(2000.0, abs=5.0)

    def test_games_are_processed_in_chronological_order_not_input_order(self):
        """Feed the games out of order; the final ratings must be identical
        to feeding them in order, since Elo depends on sequence, not on
        however the caller happened to list them."""
        in_order = [
            game("a", "b", 30, 10, "2023-09-01T00:00:00+00:00"),
            game("a", "b", 10, 30, "2023-09-08T00:00:00+00:00"),
            game("a", "b", 20, 20 - 1, "2023-09-15T00:00:00+00:00"),
        ]
        shuffled = [in_order[2], in_order[0], in_order[1]]
        assert fit_elo(in_order, min_games=1).teams == fit_elo(shuffled, min_games=1).teams

    def test_a_seeded_but_never_playing_team_still_appears_at_its_seed(self):
        """A team given a preseason rating but with no games yet this fit
        (e.g. its own week 1 hasn't happened) must still be predictable."""
        games = [game("a", "b", 20, 10, "2023-09-02T00:00:00+00:00")]
        result = fit_elo(games, min_games=1, initial_ratings={"c": 1650.0})
        assert result.teams["c"].rating == 1650.0
        assert result.teams["c"].games == 0

    def test_a_team_playing_many_games_moves_further_from_its_seed(self):
        rounds = [
            game("a", "b", 30, 10, f"2023-09-{i:02d}T00:00:00+00:00")
            for i in range(1, 10)
        ]
        result = fit_elo(rounds, min_games=1, initial_ratings={"a": 1500.0, "b": 1500.0})
        assert result.teams["a"].rating > 1600  # nine straight blowout wins


class TestEloRatingsProbability:
    def test_returns_none_when_not_active(self):
        result = EloRatings(status="insufficient_history", n_games=1, k=DEFAULT_K, hfa=60.0)
        assert result.probability("a", "b") is None

    def test_returns_none_for_an_unrated_team(self):
        result = EloRatings(
            status="active", n_games=10, k=DEFAULT_K, hfa=60.0,
            teams={"a": TeamEloState(rating=1600.0, games=5)},
        )
        assert result.probability("a", "never_played") is None

    def test_matches_expected_score_directly(self):
        result = EloRatings(
            status="active", n_games=10, k=DEFAULT_K, hfa=60.0,
            teams={
                "a": TeamEloState(rating=1600.0, games=5),
                "b": TeamEloState(rating=1500.0, games=5),
            },
        )
        assert result.probability("a", "b") == pytest.approx(
            expected_score(1600.0, 1500.0, hfa=60.0)
        )

    def test_neutral_site_drops_the_hfa_term(self):
        result = EloRatings(
            status="active", n_games=10, k=DEFAULT_K, hfa=60.0,
            teams={
                "a": TeamEloState(rating=1500.0, games=5),
                "b": TeamEloState(rating=1500.0, games=5),
            },
        )
        assert result.probability("a", "b", neutral_site=True) == pytest.approx(0.5)
        assert result.probability("a", "b", neutral_site=False) > 0.5


class TestPreseasonRating:
    def test_no_previous_season_collapses_to_the_generic_baseline(self):
        """0.75*1500 + 0.25*1500 == 1500 exactly when there is nothing to
        carry over -- the honest "nothing known yet" answer."""
        result = preseason_rating(None, None, None)
        assert result == pytest.approx(DEFAULT_INITIAL_RATING)

    def test_a_strong_previous_season_pulls_the_prior_up(self):
        result = preseason_rating(1900.0, None, None)
        assert result > DEFAULT_INITIAL_RATING

    def test_a_weak_previous_season_pulls_the_prior_down(self):
        result = preseason_rating(1100.0, None, None)
        assert result < DEFAULT_INITIAL_RATING

    def test_matches_the_plans_75_25_weighting_by_hand(self):
        result = preseason_rating(2000.0, None, None)
        assert result == pytest.approx(0.75 * 2000.0 + 0.25 * 1500.0)

    def test_positive_talent_zscore_raises_the_prior(self):
        baseline = preseason_rating(None, None, None)
        with_talent = preseason_rating(None, 1.0, None)
        assert with_talent > baseline

    def test_negative_returning_production_zscore_lowers_the_prior(self):
        baseline = preseason_rating(None, None, None)
        with_returning = preseason_rating(None, None, -1.0)
        assert with_returning < baseline

    def test_custom_coefficients_are_respected(self):
        result = preseason_rating(
            2000.0, 1.0, 1.0,
            previous_weight=0.5, baseline_weight=0.5, talent_coeff=10.0, returning_coeff=5.0,
        )
        assert result == pytest.approx(0.5 * 2000.0 + 0.5 * 1500.0 + 10.0 + 5.0)
