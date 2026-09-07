"""Comprehensive tests for Model 4: Season Futures, NIL & Roster Valuation."""

from __future__ import annotations

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.features.futures import (
    load_team_roster_inputs,
    load_team_schedule,
    project_team_futures_from_db,
)
from cfb_analytics.ingest import store
from cfb_analytics.models.futures import (
    NILTier,
    PortalComposite,
    QBContinuity,
    QBTier,
    ReturningProduction,
    RosterTalentInputs,
    ScheduledOpponent,
    calculate_true_talent_composite,
    calculate_true_talent_rating,
    calculate_win_total_distribution,
    classify_nil_tier,
    estimate_cfp_appearance_prob,
    estimate_conference_championship_prob,
    evaluate_win_total_line,
    expected_wins,
    nil_rating_adjustment,
    portal_rating_adjustment,
    project_game_win_probability,
    project_schedule,
    project_season_futures,
    qb_rating_adjustment,
    returning_production_adjustment,
    simulate_season_monte_carlo,
    win_total_variance,
)

# ===========================================================================
# 1. NIL Tier Classification & Adjustments
# ===========================================================================


class TestNILTierAndBudget:
    def test_classify_nil_tier(self) -> None:
        assert classify_nil_tier(22.0) == NILTier.TIER_1_ELITE
        assert classify_nil_tier(18.0) == NILTier.TIER_1_ELITE
        assert classify_nil_tier(15.0) == NILTier.TIER_2_UPPER_P4
        assert classify_nil_tier(12.0) == NILTier.TIER_2_UPPER_P4
        assert classify_nil_tier(8.5) == NILTier.TIER_3_MID_P4
        assert classify_nil_tier(6.0) == NILTier.TIER_3_MID_P4
        assert classify_nil_tier(4.0) == NILTier.TIER_4_LOWER_P4_HIGH_G5
        assert classify_nil_tier(3.0) == NILTier.TIER_4_LOWER_P4_HIGH_G5
        assert classify_nil_tier(1.5) == NILTier.TIER_5_G5_BASELINE
        assert classify_nil_tier(0.0) == NILTier.TIER_5_G5_BASELINE

    def test_classify_negative_budget_raises(self) -> None:
        with pytest.raises(SchemaError, match="NIL budget cannot be negative"):
            classify_nil_tier(-1.0)
        with pytest.raises(SchemaError, match="NIL budget cannot be negative"):
            NILTier.from_budget(-2.0)

    def test_nonfinite_budget_raises(self) -> None:
        with pytest.raises(SchemaError, match="NIL budget must be a finite number"):
            classify_nil_tier(float("nan"))

    def test_nil_rating_adjustments_by_tier(self) -> None:
        assert nil_rating_adjustment(NILTier.TIER_1_ELITE) == 3.5
        assert nil_rating_adjustment(NILTier.TIER_2_UPPER_P4) == 2.0
        assert nil_rating_adjustment(NILTier.TIER_3_MID_P4) == 0.0
        assert nil_rating_adjustment(NILTier.TIER_4_LOWER_P4_HIGH_G5) == -2.0
        assert nil_rating_adjustment(NILTier.TIER_5_G5_BASELINE) == -4.0

    def test_nil_rating_adjustment_from_string(self) -> None:
        assert nil_rating_adjustment("tier_1_elite") == 3.5
        assert nil_rating_adjustment("TIER_2_UPPER_P4") == 2.0

    def test_nil_invalid_string_raises(self) -> None:
        with pytest.raises(SchemaError, match="Invalid NIL tier"):
            nil_rating_adjustment("non_existent_tier")

    def test_continuous_budget_adjustment(self) -> None:
        # $9.0M is the zero anchor
        assert nil_rating_adjustment(NILTier.TIER_3_MID_P4, budget_millions=9.0) == 0.0
        # $19.0M -> (19 - 9) * 0.35 = +3.5
        assert nil_rating_adjustment(NILTier.TIER_1_ELITE, budget_millions=19.0) == 3.5
        # Clamped bounds check
        assert nil_rating_adjustment(NILTier.TIER_1_ELITE, budget_millions=50.0) == 5.0
        assert nil_rating_adjustment(NILTier.TIER_5_G5_BASELINE, budget_millions=0.0) == -3.15


# ===========================================================================
# 2. Transfer Portal Composite Calculations
# ===========================================================================


class TestTransferPortal:
    def test_portal_composite_auto_net(self) -> None:
        portal = PortalComposite(additions_score=45.0, departures_score=15.0)
        assert portal.net_composite == 30.0
        assert portal.is_net_positive is True

    def test_portal_composite_net_negative(self) -> None:
        portal = PortalComposite(additions_score=10.0, departures_score=25.0)
        assert portal.net_composite == -15.0
        assert portal.is_net_positive is False

    def test_portal_counts_negative_raises(self) -> None:
        with pytest.raises(SchemaError, match="Portal counts cannot be negative"):
            PortalComposite(additions_score=10.0, departures_score=5.0, additions_count=-1)

    def test_portal_rating_adjustment(self) -> None:
        portal = PortalComposite(additions_score=30.0, departures_score=10.0)  # net +20
        # 20 * 0.15 = +3.0
        assert portal_rating_adjustment(portal) == 3.0

        # Numeric input support
        assert portal_rating_adjustment(-20.0) == -3.0

        # Clamping at -6.0 and +6.0
        assert portal_rating_adjustment(80.0) == 6.0
        assert portal_rating_adjustment(-80.0) == -6.0

    def test_invalid_portal_type_raises(self) -> None:
        with pytest.raises(SchemaError, match="Expected PortalComposite"):
            portal_rating_adjustment("invalid_type")  # type: ignore[arg-type]


# ===========================================================================
# 2b. True Talent Composite Calculations
# ===========================================================================


class TestTrueTalentComposite:
    def test_calculate_true_talent_composite(self) -> None:
        portal = PortalComposite(additions_score=30.0, departures_score=10.0)  # net +20
        # 800 + 20 * 2.75 = 800 + 55 = 855.0
        comp = calculate_true_talent_composite(800.0, portal)
        assert comp == 855.0

    def test_true_talent_composite_negative_recruiting_raises(self) -> None:
        with pytest.raises(SchemaError, match="Recruiting composite cannot be negative"):
            calculate_true_talent_composite(-50.0, 0.0)

    def test_roster_inputs_true_talent_composite_auto_populated(self) -> None:
        inputs = RosterTalentInputs(
            team_id="t-osu",
            recruiting_composite=900.0,
            portal_composite=10.0,
            nil_tier=NILTier.TIER_1_ELITE,
            returning_production=0.70,
        )
        # 900 + 10 * 2.75 = 927.5
        assert inputs.true_talent_composite == 927.5

        # Also verified on AdjustedPowerRating
        rating = calculate_true_talent_rating(inputs)
        assert rating.true_talent_composite == 927.5
        assert rating.true_talent_rating == rating.adjusted_power_rating


# ===========================================================================
# 3. Returning Production Adjustments
# ===========================================================================


class TestReturningProduction:
    def test_returning_production_auto_overall(self) -> None:
        ret = ReturningProduction(percent_ppa_offense=0.70, percent_ppa_defense=0.50)
        # 0.53 * 0.70 + 0.47 * 0.50 = 0.371 + 0.235 = 0.606
        assert ret.percent_overall == pytest.approx(0.606, abs=1e-4)

    def test_returning_production_adjustment(self) -> None:
        # Exactly at FBS baseline 0.60
        ret_baseline = ReturningProduction(percent_ppa_offense=0.60, percent_ppa_defense=0.60)
        assert returning_production_adjustment(ret_baseline) == 0.0

        # High continuity (+0.20 off, +0.20 def)
        # 4.0 * 0.20 + 3.0 * 0.20 = 0.8 + 0.6 = +1.40
        ret_high = ReturningProduction(percent_ppa_offense=0.80, percent_ppa_defense=0.80)
        assert returning_production_adjustment(ret_high) == 1.40

        # Severe attrition (-0.30 off, -0.30 def)
        # 4.0 * -0.30 + 3.0 * -0.30 = -1.2 - 0.9 = -2.10
        ret_low = ReturningProduction(percent_ppa_offense=0.30, percent_ppa_defense=0.30)
        assert returning_production_adjustment(ret_low) == -2.10

    def test_returning_production_bounds_check(self) -> None:
        with pytest.raises(SchemaError, match="Returning offense production must be in"):
            ReturningProduction(percent_ppa_offense=1.5, percent_ppa_defense=0.5)
        with pytest.raises(SchemaError, match="Returning defense production must be in"):
            ReturningProduction(percent_ppa_offense=0.5, percent_ppa_defense=-0.2)

    def test_scalar_matches_equal_offense_and_defense(self) -> None:
        structured = ReturningProduction(percent_ppa_offense=0.80, percent_ppa_defense=0.80)
        assert returning_production_adjustment(0.80) == returning_production_adjustment(structured)


# ===========================================================================
# 4. Quarterback Tier & Continuity
# ===========================================================================


class TestQuarterbackAdjustments:
    def test_elite_returning_starter(self) -> None:
        # Tier 1 (+4.5) + Returning Multi-Year (+1.0) = +5.5
        adj = qb_rating_adjustment(QBTier.TIER_1_ELITE, QBContinuity.RETURNING_MULTI_YEAR)
        assert adj == 5.5

    def test_developing_first_year_starter(self) -> None:
        # Tier 4 (-2.5) + First Year in System (-1.0) = -3.5
        adj = qb_rating_adjustment(
            QBTier.TIER_4_DEVELOPING_UNPROVEN, QBContinuity.FIRST_YEAR_STARTER_IN_SYSTEM
        )
        assert adj == -3.5

    def test_unresolved_battle(self) -> None:
        # Tier 3 (0.0) + Unresolved Battle (-2.5) = -2.5
        adj = qb_rating_adjustment(QBTier.TIER_3_AVERAGE, QBContinuity.UNRESOLVED_BATTLE)
        assert adj == -2.5

    def test_string_inputs_and_invalid_handling(self) -> None:
        adj = qb_rating_adjustment("tier_2_quality_starter", "returning_starter_same_system")
        assert adj == 2.5

        with pytest.raises(SchemaError, match="Invalid QB tier"):
            qb_rating_adjustment("fake_qb_tier", "returning_starter_same_system")

        with pytest.raises(SchemaError, match="Invalid QB continuity"):
            qb_rating_adjustment("tier_2_quality_starter", "fake_continuity")


# ===========================================================================
# 5. True Talent Rating & Power Rating Decomposition
# ===========================================================================


class TestTrueTalentRating:
    def test_calculate_true_talent_rating_from_recruiting(self) -> None:
        inputs = RosterTalentInputs(
            team_id="t-georgia",
            recruiting_composite=980.0,  # Elite recruiting
            portal_composite=PortalComposite(
                additions_score=30.0, departures_score=10.0
            ),  # net +20 -> +3.0
            nil_tier=NILTier.TIER_1_ELITE,  # +3.5
            returning_production=ReturningProduction(
                percent_ppa_offense=0.75, percent_ppa_defense=0.70
            ),
            qb_tier=QBTier.TIER_1_ELITE,  # +4.5
            qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,  # +0.5
            conference="SEC",
        )
        rating = calculate_true_talent_rating(inputs)

        # Base: (980 - 650) * 0.055 = 330 * 0.055 = 18.15
        assert rating.base_talent_rating == pytest.approx(18.15, abs=0.01)
        assert rating.portal_adjustment == 3.0
        assert rating.nil_adjustment == 3.5
        assert rating.qb_adjustment == 5.0
        # Ret prod: 4.0 * 0.15 + 3.0 * 0.10 = 0.6 + 0.3 = 0.90
        assert rating.returning_production_adjustment == 0.90
        # Total adjusted: 18.15 + 3.0 + 3.5 + 0.9 + 5.0 = 30.55
        assert rating.adjusted_power_rating == pytest.approx(30.55, abs=0.05)
        # Elo: 1500 + 30.55 * 25 = 2263.8
        assert rating.elo_equivalent == pytest.approx(2263.8, abs=1.5)

    def test_explicit_base_power_rating_override(self) -> None:
        inputs = RosterTalentInputs(
            team_id="t-custom",
            recruiting_composite=700.0,
            portal_composite=0.0,
            nil_tier=NILTier.TIER_3_MID_P4,  # 0.0
            returning_production=0.60,  # 0.0
            qb_tier=QBTier.TIER_3_AVERAGE,  # 0.0
            qb_continuity=QBContinuity.RETURNING_STARTER_NEW_OC,  # 0.0
            base_power_rating=10.0,  # Explicitly set
        )
        rating = calculate_true_talent_rating(inputs)
        assert rating.base_talent_rating == 10.0
        assert rating.adjusted_power_rating == 10.0
        assert rating.elo_equivalent == 1500.0 + 10.0 * 25.0


# ===========================================================================
# 6. Game-by-Game Projections & Home Field Advantage
# ===========================================================================


class TestGameProjections:
    def test_neutral_site_even_matchup(self) -> None:
        spread, prob = project_game_win_probability(10.0, 10.0, is_neutral=True)
        assert spread == 0.0
        assert prob == 0.50

    def test_home_field_advantage_applied(self) -> None:
        # Team +10 at Home vs +10 Away. HFA = 2.5
        # Margin = +2.5. Spread = -2.5. Win prob > 0.50
        spread_home, prob_home = project_game_win_probability(10.0, 10.0, is_home=True, hfa=2.5)
        assert spread_home == -2.5
        assert prob_home > 0.50

        # Team +10 Away vs +10 Home. HFA = 2.5
        # Margin = -2.5. Spread = +2.5. Win prob < 0.50
        spread_away, prob_away = project_game_win_probability(10.0, 10.0, is_home=False, hfa=2.5)
        assert spread_away == 2.5
        assert prob_away < 0.50

        # Symmetry test
        assert prob_home + prob_away == pytest.approx(1.0, abs=1e-4)

    def test_heavy_favorite_and_underdog(self) -> None:
        spread_fav, prob_fav = project_game_win_probability(25.0, 0.0, is_home=True, hfa=2.5)
        assert spread_fav == -27.5
        assert prob_fav > 0.90

        spread_dog, prob_dog = project_game_win_probability(0.0, 25.0, is_home=False, hfa=2.5)
        assert spread_dog == 27.5
        assert prob_dog < 0.10

    def test_project_schedule(self) -> None:
        schedule = [
            ScheduledOpponent("opp-1", opponent_power_rating=0.0, is_home=True),
            ScheduledOpponent(
                "opp-2", opponent_power_rating=10.0, is_home=False, is_conference=True
            ),
        ]
        projections = project_schedule(15.0, schedule)
        assert len(projections) == 2
        assert projections[0].opponent_id == "opp-1"
        assert projections[0].is_conference is False
        assert projections[1].opponent_id == "opp-2"
        assert projections[1].is_conference is True

    def test_invalid_sigma_and_nonfinite_ratings_raise(self) -> None:
        with pytest.raises(SchemaError, match="Margin sigma must be greater"):
            project_game_win_probability(10.0, 10.0, sigma=0.0)
        with pytest.raises(SchemaError, match="team rating must be a finite number"):
            project_game_win_probability(float("nan"), 10.0)


# ===========================================================================
# 7. Exact Poisson Binomial Distribution & Expected Wins
# ===========================================================================


class TestPoissonBinomialDistribution:
    def test_identical_coin_flips_matches_binomial(self) -> None:
        # 4 games with p = 0.5 each -> standard Binomial(4, 0.5)
        # PMF: [1/16, 4/16, 6/16, 4/16, 1/16] = [0.0625, 0.25, 0.375, 0.25, 0.0625]
        probs = [0.5, 0.5, 0.5, 0.5]
        pmf = calculate_win_total_distribution(probs)

        assert len(pmf) == 5
        assert pmf[0] == pytest.approx(0.0625, abs=1e-6)
        assert pmf[1] == pytest.approx(0.2500, abs=1e-6)
        assert pmf[2] == pytest.approx(0.3750, abs=1e-6)
        assert pmf[3] == pytest.approx(0.2500, abs=1e-6)
        assert pmf[4] == pytest.approx(0.0625, abs=1e-6)
        assert sum(pmf) == pytest.approx(1.0, abs=1e-6)

        assert expected_wins(probs) == 2.0
        assert win_total_variance(probs) == 4 * 0.25  # 1.0

    def test_boundary_all_certain_wins(self) -> None:
        probs = [1.0] * 12
        pmf = calculate_win_total_distribution(probs)
        assert pmf[12] == pytest.approx(1.0, abs=1e-6)
        assert sum(pmf[:12]) == pytest.approx(0.0, abs=1e-6)
        assert expected_wins(probs) == 12.0
        assert win_total_variance(probs) == 0.0

    def test_boundary_all_certain_losses(self) -> None:
        probs = [0.0] * 12
        pmf = calculate_win_total_distribution(probs)
        assert pmf[0] == pytest.approx(1.0, abs=1e-6)
        assert expected_wins(probs) == 0.0
        assert win_total_variance(probs) == 0.0

    def test_heterogeneous_probabilities_invariants(self) -> None:
        # Realistic 12-game college football schedule
        probs = [0.95, 0.88, 0.82, 0.75, 0.68, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.20]
        pmf = calculate_win_total_distribution(probs)

        assert len(pmf) == 13
        assert sum(pmf) == pytest.approx(1.0, abs=1e-6)
        # Expected value formula verification: sum(k * P(W=k)) == sum(p_i)
        calculated_ev = sum(k * p for k, p in enumerate(pmf))
        assert calculated_ev == pytest.approx(expected_wins(probs), abs=1e-4)

        # Variance formula verification: E[W^2] - (E[W])^2 == sum(p_i * (1 - p_i))
        calculated_var = sum((k - calculated_ev) ** 2 * p for k, p in enumerate(pmf))
        assert calculated_var == pytest.approx(win_total_variance(probs), abs=1e-4)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
    def test_invalid_probabilities_raise(self, value: float) -> None:
        with pytest.raises(SchemaError, match="win probabilities"):
            calculate_win_total_distribution([value])
        with pytest.raises(SchemaError, match="win probabilities"):
            expected_wins([value])


# ===========================================================================
# 8. Win Total Line Evaluation (Over / Under / Push)
# ===========================================================================


class TestWinTotalLineEvaluation:
    def test_half_game_line_evaluation(self) -> None:
        # Distribution centered at 8.0 wins
        probs = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]  # 10 games, p=0.8, mean=8.0
        pmf = calculate_win_total_distribution(probs)

        eval_8_5 = evaluate_win_total_line(pmf, 8.5)
        assert eval_8_5.prob_push == 0.0
        # Over 8.5 means winning 9 or 10 games
        assert eval_8_5.prob_over == pytest.approx(pmf[9] + pmf[10], abs=1e-4)
        # Under 8.5 means winning 0 to 8 games
        assert eval_8_5.prob_under == pytest.approx(sum(pmf[:9]), abs=1e-4)
        assert eval_8_5.prob_over + eval_8_5.prob_under == pytest.approx(1.0, abs=1e-4)
        assert eval_8_5.fair_over_american is not None
        assert eval_8_5.fair_under_american is not None

    def test_integer_line_with_pushes(self) -> None:
        probs = [0.5] * 10  # 10 games, p=0.5, mean=5.0
        pmf = calculate_win_total_distribution(probs)

        eval_5_0 = evaluate_win_total_line(pmf, 5.0)
        # Push is winning exactly 5 games
        assert eval_5_0.prob_push == pytest.approx(pmf[5], abs=1e-4)
        # Over is winning 6+
        assert eval_5_0.prob_over == pytest.approx(sum(pmf[6:]), abs=1e-4)
        # Under is winning <= 4
        assert eval_5_0.prob_under == pytest.approx(sum(pmf[:5]), abs=1e-4)
        assert eval_5_0.prob_over + eval_5_0.prob_under + eval_5_0.prob_push == pytest.approx(
            1.0, abs=1e-4
        )

        # By symmetry of Binomial(10, 0.5), fair odds for Over and Under should be equal
        assert eval_5_0.fair_over_american == eval_5_0.fair_under_american

    def test_market_edge_and_recommendation(self) -> None:
        # Skewed distribution toward 9 wins
        probs = [0.85] * 10  # mean 8.5 wins
        pmf = calculate_win_total_distribution(probs)

        # Posted line is 7.5. Model thinks Over 7.5 is heavy favorite (>80%)
        # Market quotes Over -110, Under -110 (50% devigged)
        res = evaluate_win_total_line(pmf, 7.5, market_over_price=-110, market_under_price=-110)
        assert res.recommended_side == "OVER"
        assert res.edge > 0.05
        assert res.expected_value is not None and res.expected_value > 0.0
        assert res.model_status == "uncalibrated_shadow"
        assert res.is_actionable is False

    def test_integer_line_ev_push_accounting(self) -> None:
        # 10 games, p=0.5 -> symmetric distribution around 5.0 wins
        probs = [0.5] * 10
        pmf = calculate_win_total_distribution(probs)
        # Even market at -110 on 5.0 wins
        res = evaluate_win_total_line(pmf, 5.0, market_over_price=-110, market_under_price=-110)
        assert res.prob_push > 0.20
        # Over and under fair probabilities are both 0.50 (excluding push)
        assert res.edge == 0.0
        assert res.recommended_side == "PASS"

        # Now test with edge on over
        res_edge = evaluate_win_total_line(
            pmf, 4.0, market_over_price=-110, market_under_price=-110
        )
        assert res_edge.recommended_side == "OVER"
        # EV should be bounded below what raw non-push multiplication would claim
        assert res_edge.expected_value is not None
        assert res_edge.expected_value > 0.0

    def test_out_of_bounds_line_raises(self) -> None:
        pmf = calculate_win_total_distribution([0.5] * 12)
        with pytest.raises(SchemaError, match="outside possible win range"):
            evaluate_win_total_line(pmf, 15.0)

    def test_positive_devig_edge_cannot_recommend_negative_ev(self) -> None:
        result = evaluate_win_total_line(
            [0.4, 0.6],
            0.5,
            market_over_price=-200,
            market_under_price=-200,
        )
        assert result.recommended_side == "PASS"
        assert result.expected_value is None


# ===========================================================================
# 9. Conference Championship & CFP Appearance Estimations
# ===========================================================================


class TestConferenceAndCFPProjections:
    def test_conference_championship_unbeaten_sec(self) -> None:
        # Team winning all 8 conference games with high probability
        conf_probs = [0.95] * 8
        conf_pmf = calculate_win_total_distribution(conf_probs)

        proj = estimate_conference_championship_prob(conf_pmf, "SEC", team_rating=22.0)
        assert proj.prob_reach_ccg > 0.85
        assert proj.prob_win_ccg > 0.40
        assert proj.expected_conference_wins > 7.0

    def test_ccg_win_prob_varies_with_team_strength(self) -> None:
        conf_pmf = calculate_win_total_distribution([0.95] * 8)
        elite_proj = estimate_conference_championship_prob(conf_pmf, "SEC", team_rating=28.0)
        underdog_proj = estimate_conference_championship_prob(conf_pmf, "SEC", team_rating=8.0)
        # Elite team must have strictly higher win prob than underdog team
        assert elite_proj.prob_win_ccg > underdog_proj.prob_win_ccg
        p_elite_given_reach = elite_proj.prob_win_ccg / elite_proj.prob_reach_ccg
        p_dog_given_reach = underdog_proj.prob_win_ccg / underdog_proj.prob_reach_ccg
        assert p_elite_given_reach > 0.65
        assert p_dog_given_reach < 0.30

    def test_ccg_short_schedule_supported(self) -> None:
        # Short schedule: 2 conference games
        conf_pmf = calculate_win_total_distribution([0.95, 0.95])
        proj = estimate_conference_championship_prob(conf_pmf, "Big Ten", team_rating=20.0)
        assert proj.prob_reach_ccg > 0.80

    def test_conference_championship_independent_has_zero(self) -> None:
        conf_pmf = calculate_win_total_distribution([0.9] * 8)
        proj = estimate_conference_championship_prob(conf_pmf, "Independent", team_rating=20.0)
        assert proj.prob_reach_ccg == 0.0
        assert proj.prob_win_ccg == 0.0

    def test_cfp_appearance_unbeaten_p4(self) -> None:
        # 12-0 P4 team with high rating
        overall_pmf = calculate_win_total_distribution([0.98] * 12)
        p_cfp = estimate_cfp_appearance_prob(
            overall_pmf, "Big Ten", team_rating=25.0, sos=8.0, prob_win_ccg=0.60
        )
        assert p_cfp > 0.90

    def test_cfp_undefeated_short_schedule(self) -> None:
        # 10-0 schedule in P4
        pmf_10 = calculate_win_total_distribution([0.98] * 10)
        p_cfp = estimate_cfp_appearance_prob(
            pmf_10, "Big Ten", team_rating=25.0, sos=6.0, prob_win_ccg=0.50
        )
        assert p_cfp > 0.85

    def test_cfp_appearance_low_win_team(self) -> None:
        # 6-6 P4 team
        overall_pmf = calculate_win_total_distribution([0.50] * 12)
        p_cfp = estimate_cfp_appearance_prob(overall_pmf, "ACC", team_rating=2.0, sos=0.0)
        assert p_cfp < 0.02

    def test_g5_cfp_auto_bid_path(self) -> None:
        # 12-0 G5 team winning conference championship
        overall_pmf = calculate_win_total_distribution([0.95] * 12)
        # With CCG win, top G5 contender has legitimate CFP path
        p_cfp_with_ccg = estimate_cfp_appearance_prob(
            overall_pmf, "American", team_rating=10.0, sos=-2.0, prob_win_ccg=0.80
        )
        assert p_cfp_with_ccg > 0.30

        # Without CCG win, G5 team has near zero at-large path
        p_cfp_no_ccg = estimate_cfp_appearance_prob(
            overall_pmf, "American", team_rating=10.0, sos=-2.0, prob_win_ccg=0.0
        )
        assert p_cfp_no_ccg < 0.05

    def test_cfp_appearance_empty_and_degenerate_pmf(self) -> None:
        assert estimate_cfp_appearance_prob([], "Big Ten", team_rating=20.0) == 0.0
        assert estimate_cfp_appearance_prob([1.0], "Big Ten", team_rating=20.0) == 0.0

    def test_cfp_appearance_g5_elite_at_large(self) -> None:
        # Exceptional G5 team (e.g. 2021 Cincinnati style rating >= 20.0)
        overall_pmf = calculate_win_total_distribution([0.98] * 12)
        p_cfp_elite_g5_no_ccg = estimate_cfp_appearance_prob(
            overall_pmf, "American", team_rating=22.0, sos=2.0, prob_win_ccg=0.0
        )
        assert p_cfp_elite_g5_no_ccg > 0.05

    def test_cfp_appearance_notre_dame(self) -> None:
        # 11-1 Notre Dame with top-10 rating
        overall_pmf = calculate_win_total_distribution([0.92] * 12)
        p_cfp_nd = estimate_cfp_appearance_prob(
            overall_pmf, "Independent", team_rating=22.0, sos=6.0, prob_win_ccg=0.0
        )
        assert p_cfp_nd > 0.50

    def test_cfp_appearance_negative_rating(self) -> None:
        # Bottom-dwelling P4 team
        overall_pmf = calculate_win_total_distribution([0.30] * 12)
        p_cfp_bad = estimate_cfp_appearance_prob(
            overall_pmf, "SEC", team_rating=-5.0, sos=4.0, prob_win_ccg=0.0
        )
        assert p_cfp_bad < 0.01


# ===========================================================================
# 10. End-to-End High-Level Futures Engine
# ===========================================================================


class TestProjectSeasonFutures:
    def test_end_to_end_championship_contender(self) -> None:
        inputs = RosterTalentInputs(
            team_id="t-contender",
            recruiting_composite=950.0,
            portal_composite=PortalComposite(additions_score=25.0, departures_score=5.0),
            nil_tier=NILTier.TIER_1_ELITE,
            returning_production=ReturningProduction(
                percent_ppa_offense=0.80, percent_ppa_defense=0.75
            ),
            qb_tier=QBTier.TIER_1_ELITE,
            qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
            conference="Big Ten",
            strength_of_schedule=6.0,
        )

        schedule = [
            ScheduledOpponent(
                f"opp-{i}",
                opponent_power_rating=float(i * 2),
                is_home=(i % 2 == 0),
                is_conference=(i >= 4),
            )
            for i in range(12)
        ]

        projection = project_season_futures(inputs, schedule, posted_lines=[9.5, 10.0, 10.5])

        assert projection.team_id == "t-contender"
        assert projection.conference == "Big Ten"
        assert projection.adjusted_rating.adjusted_power_rating > 20.0
        assert projection.expected_wins > 9.0
        assert len(projection.schedule_projections) == 12
        assert len(projection.win_distribution) == 13
        assert projection.prob_cfp_appearance > 0.50
        assert projection.model_status == "uncalibrated_shadow"
        assert 10.0 in projection.win_total_evaluations
        assert projection.win_total_evaluations[10.0].prob_over > 0.0

    def test_posted_lines_preservation(self) -> None:
        inputs = RosterTalentInputs(
            team_id="t-preserved",
            recruiting_composite=750.0,
            portal_composite=0.0,
            nil_tier=NILTier.TIER_3_MID_P4,
            returning_production=0.60,
        )
        schedule = [ScheduledOpponent(f"opp-{i}", opponent_power_rating=0.0) for i in range(10)]
        # Request line 10.0 on a 10-game schedule (which should not be clamped down to 9.5)
        proj = project_season_futures(inputs, schedule, posted_lines=[10.0])
        assert 10.0 in proj.win_total_evaluations
        assert proj.win_total_evaluations[10.0].line == 10.0


# ===========================================================================
# 11. Monte Carlo Simulation Engine
# ===========================================================================


class TestMonteCarloSimulation:
    def test_simulation_reproducibility_and_convergence(self) -> None:
        probs = [0.8, 0.7, 0.6, 0.9, 0.85, 0.5, 0.4, 0.95, 0.75, 0.65, 0.55, 0.35]
        analytical_ev = expected_wins(probs)  # 8.00

        # Run with fixed seed
        sim1 = simulate_season_monte_carlo(
            probs, posted_lines=[7.5, 8.5], n_simulations=10000, seed=12345
        )
        sim2 = simulate_season_monte_carlo(
            probs, posted_lines=[7.5, 8.5], n_simulations=10000, seed=12345
        )

        # Identical results given identical seeds
        assert sim1.mean_wins == sim2.mean_wins
        assert sim1.simulated_win_probabilities == sim2.simulated_win_probabilities

        # Convergence to analytical expected wins within MC standard error ~ 0.03
        assert abs(sim1.mean_wins - analytical_ev) < 0.05
        assert sim1.median_wins in (7.0, 8.0, 9.0)
        assert sim1.p10_wins <= sim1.p25_wins <= sim1.median_wins <= sim1.p75_wins <= sim1.p90_wins

    def test_simulation_invalid_sim_count(self) -> None:
        with pytest.raises(SchemaError, match="n_simulations must be >= 1"):
            simulate_season_monte_carlo([0.5], n_simulations=0)

    def test_simulation_rejects_nonfinite_probability(self) -> None:
        with pytest.raises(SchemaError, match="schedule win probabilities"):
            simulate_season_monte_carlo([float("nan")])


# ===========================================================================
# 12. Database Feature Loading & End-to-End Store Integration
# ===========================================================================


class TestFuturesFeatureLayerIntegration:
    @staticmethod
    def _roster_overrides() -> dict[str, object]:
        return {
            "portal_composite": 0.0,
            "nil_tier": NILTier.TIER_2_UPPER_P4,
            "qb_tier": QBTier.TIER_3_AVERAGE,
            "qb_continuity": QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
            "strength_of_schedule": 0.0,
        }

    def _seed_db(self, conn) -> None:
        # Teams
        store.upsert_team(
            conn,
            {
                "team_id": "cfbd:ohio_st",
                "cfbd_id": 194,
                "school": "Ohio State",
                "conference": "Big Ten",
            },
        )
        store.upsert_team(
            conn,
            {
                "team_id": "cfbd:michigan",
                "cfbd_id": 130,
                "school": "Michigan",
                "conference": "Big Ten",
            },
        )
        store.upsert_team(
            conn, {"team_id": "cfbd:akron", "cfbd_id": 2006, "school": "Akron", "conference": "MAC"}
        )
        for team_id, conference in (
            ("cfbd:ohio_st", "Big Ten"),
            ("cfbd:michigan", "Big Ten"),
            ("cfbd:akron", "MAC"),
        ):
            store.upsert_team_season(
                conn,
                {
                    "team_id": team_id,
                    "season": 2026,
                    "source": "cfbd",
                    "conference": conference,
                },
            )

        # Talent
        store.insert_team_talent(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "availability_class": "preseason",
                    "talent_composite": 990.0,
                }
            ],
        )

        # Returning production
        store.insert_returning_production(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "availability_class": "preseason",
                    "total_ppa": 0.72,
                    "passing_ppa": 0.85,
                    "receiving_ppa": 0.70,
                    "rushing_ppa": 0.65,
                    "percent_ppa": 0.72,
                    "percent_passing_ppa": 0.85,
                    "percent_receiving_ppa": 0.70,
                    "percent_rushing_ppa": 0.65,
                    "usage": 0.75,
                    "passing_usage": 0.80,
                    "receiving_usage": 0.70,
                    "rushing_usage": 0.70,
                }
            ],
        )

        # Scheduled games
        store.upsert_cfbd_game(
            conn,
            {
                "game_id": "cfbd:game_1",
                "season": 2026,
                "week": 1,
                "season_type": "regular",
                "kickoff_utc": "2026-09-05T16:00:00+00:00",
                "football_date": "2026-09-05",
                "neutral_site": 0,
                "conference_game": 0,
                "home_team_id": "cfbd:ohio_st",
                "away_team_id": "cfbd:akron",
                "venue_name": "Ohio Stadium",
                "completed": 0,
                "source": "cfbd",
            },
        )
        store.upsert_cfbd_game(
            conn,
            {
                "game_id": "cfbd:game_2",
                "season": 2026,
                "week": 13,
                "season_type": "regular",
                "kickoff_utc": "2026-11-28T17:00:00+00:00",
                "football_date": "2026-11-28",
                "neutral_site": 0,
                "conference_game": 1,
                "home_team_id": "cfbd:ohio_st",
                "away_team_id": "cfbd:michigan",
                "venue_name": "Ohio Stadium",
                "completed": 0,
                "source": "cfbd",
            },
        )
        conn.commit()

    def test_load_team_roster_inputs(self, conn) -> None:
        self._seed_db(conn)
        inputs = load_team_roster_inputs(conn, "cfbd:ohio_st", 2026, **self._roster_overrides())

        assert inputs.team_id == "cfbd:ohio_st"
        assert inputs.recruiting_composite == 990.0
        assert isinstance(inputs.returning_production, ReturningProduction)
        assert inputs.returning_production.percent_ppa_offense == 0.72
        assert inputs.returning_production.percent_ppa_defense is None
        assert inputs.conference == "Big Ten"
        assert inputs.nil_tier == NILTier.TIER_2_UPPER_P4

    def test_load_team_schedule(self, conn) -> None:
        self._seed_db(conn)
        ratings = {"cfbd:akron": -15.0, "cfbd:michigan": 18.0}
        schedule = load_team_schedule(conn, "cfbd:ohio_st", 2026, opponent_ratings=ratings)

        assert len(schedule) == 2
        assert schedule[0].opponent_id == "cfbd:akron"
        assert schedule[0].opponent_power_rating == -15.0
        assert schedule[0].is_conference is False
        assert schedule[1].opponent_id == "cfbd:michigan"
        assert schedule[1].opponent_power_rating == 18.0
        assert schedule[1].is_conference is True

    def test_project_team_futures_from_db(self, conn) -> None:
        self._seed_db(conn)
        ratings = {"cfbd:akron": -15.0, "cfbd:michigan": 18.0}
        proj = project_team_futures_from_db(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2027-01-01T00:00:00+00:00",
            opponent_ratings=ratings,
            portal_composite=0.0,
            nil_tier=NILTier.TIER_1_ELITE,
            qb_tier=QBTier.TIER_1_ELITE,
            qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
        )

        assert proj.team_id == "cfbd:ohio_st"
        assert proj.adjusted_rating.adjusted_power_rating > 20.0
        assert len(proj.schedule_projections) == 2
        assert proj.expected_wins > 1.2

    def test_load_team_schedule_as_of_filtering(self, conn) -> None:
        self._seed_db(conn)
        # Update one game's ingested_utc to the future
        conn.execute(
            "UPDATE games SET ingested_utc = '2026-09-01T00:00:00+00:00'"
            " WHERE game_id = 'cfbd:game_2'"
        )
        conn.execute(
            "UPDATE games SET ingested_utc = '2026-08-01T00:00:00+00:00'"
            " WHERE game_id = 'cfbd:game_1'"
        )
        conn.commit()

        # As of August 15, game 2 was not yet ingested
        sched_aug = load_team_schedule(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2026-08-15T00:00:00+00:00",
            opponent_ratings={"cfbd:akron": -15.0, "cfbd:michigan": 18.0},
        )
        assert len(sched_aug) == 1
        assert sched_aug[0].opponent_id == "cfbd:akron"

        # As of September 15, both were ingested
        sched_sep = load_team_schedule(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2026-09-15T00:00:00+00:00",
            opponent_ratings={"cfbd:akron": -15.0, "cfbd:michigan": 18.0},
        )
        assert len(sched_sep) == 2

    def test_empty_schedule_projection(self) -> None:
        inputs = RosterTalentInputs(
            team_id="t-empty",
            recruiting_composite=700.0,
            portal_composite=0.0,
            nil_tier=NILTier.TIER_3_MID_P4,
            returning_production=0.60,
        )
        proj = project_season_futures(inputs, schedule=[])
        assert proj.expected_wins == 0.0
        assert proj.win_variance == 0.0
        assert proj.win_distribution == [1.0]
        assert proj.win_total_evaluations == {}

    def test_single_game_schedule_projection(self) -> None:
        inputs = RosterTalentInputs(
            team_id="t-single",
            recruiting_composite=700.0,
            portal_composite=0.0,
            nil_tier=NILTier.TIER_3_MID_P4,
            returning_production=0.60,
            base_power_rating=10.0,
        )
        game = ScheduledOpponent(
            opponent_id="opp-1", opponent_power_rating=10.0, is_home=True, is_conference=True
        )
        proj = project_season_futures(inputs, schedule=[game], posted_lines=[0.5])
        assert len(proj.schedule_projections) == 1
        assert len(proj.win_distribution) == 2
        assert 0.5 in proj.win_total_evaluations
        assert proj.win_total_evaluations[0.5].prob_push == 0.0
        assert proj.win_total_evaluations[0.5].prob_over > 0.50

    def test_as_of_roster_inputs_sos_leakage_prevented(self, conn) -> None:
        self._seed_db(conn)
        store.insert_team_ratings(
            conn,
            [
                {
                    "snapshot_id": "r-final",
                    "season": 2026,
                    "period": "season_final",
                    "week": None,
                    "season_type": None,
                    "team_id": "cfbd:ohio_st",
                    "source": "sp",
                    "snapshot_scope": "season_final",
                    "provenance_mode": "reconstructed",
                    "as_of_utc": "2026-12-20T00:00:00+00:00",
                    "ingested_utc": "2026-12-20T00:00:00+00:00",
                    "rating": 28.0,
                    "sos": 9.2,
                },
            ],
        )
        conn.commit()

        kwargs = self._roster_overrides()
        kwargs.pop("strength_of_schedule")
        with pytest.raises(SchemaError, match="No admissible strength-of-schedule"):
            load_team_roster_inputs(
                conn,
                "cfbd:ohio_st",
                2026,
                as_of_utc="2026-12-20T00:00:00Z",
                **kwargs,
            )

        inputs_after_final = load_team_roster_inputs(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2026-12-21T00:00:00+00:00",
            **kwargs,
        )
        assert inputs_after_final.strength_of_schedule == 9.2

    def test_opponent_db_rating_and_fcs_fallback(self, conn) -> None:
        self._seed_db(conn)
        store.upsert_team(
            conn,
            {
                "team_id": "cfbd:fcs_opp",
                "cfbd_id": 9999,
                "school": "FCS School",
                "classification": "fbs",
            },
        )
        store.upsert_team_season(
            conn,
            {
                "team_id": "cfbd:fcs_opp",
                "season": 2026,
                "source": "cfbd",
                "classification": "fcs",
            },
        )
        store.upsert_team(
            conn,
            {
                "team_id": "cfbd:rated_opp",
                "cfbd_id": 8888,
                "school": "Rated School",
                "classification": "fbs",
            },
        )
        store.insert_team_ratings(
            conn,
            [
                {
                    "snapshot_id": "r-opp",
                    "season": 2026,
                    "period": "regular:week:01",
                    "week": 1,
                    "season_type": "regular",
                    "team_id": "cfbd:rated_opp",
                    "source": "elo_cfbd",
                    "snapshot_scope": "weekly",
                    "provenance_mode": "reconstructed",
                    "as_of_utc": "2026-08-01T00:00:00+00:00",
                    "ingested_utc": "2026-08-01T00:00:00+00:00",
                    "rating": 1862.5,
                    "sos": 0.0,
                }
            ],
        )
        store.upsert_cfbd_game(
            conn,
            {
                "game_id": "cfbd:game_fcs",
                "season": 2026,
                "week": 2,
                "season_type": "regular",
                "kickoff_utc": "2026-09-12T16:00:00+00:00",
                "football_date": "2026-09-12",
                "neutral_site": 0,
                "conference_game": 0,
                "home_team_id": "cfbd:ohio_st",
                "away_team_id": "cfbd:fcs_opp",
                "venue_name": "Ohio Stadium",
                "completed": 0,
                "source": "cfbd",
            },
        )
        store.upsert_cfbd_game(
            conn,
            {
                "game_id": "cfbd:game_rated",
                "season": 2026,
                "week": 3,
                "season_type": "regular",
                "kickoff_utc": "2026-09-19T16:00:00+00:00",
                "football_date": "2026-09-19",
                "neutral_site": 0,
                "conference_game": 0,
                "home_team_id": "cfbd:ohio_st",
                "away_team_id": "cfbd:rated_opp",
                "venue_name": "Ohio Stadium",
                "completed": 0,
                "source": "cfbd",
            },
        )
        conn.commit()

        schedule = load_team_schedule(
            conn,
            "cfbd:ohio_st",
            2026,
            opponent_ratings={"cfbd:akron": -15.0, "cfbd:michigan": 18.0},
        )
        sched_map = {g.opponent_id: g.opponent_power_rating for g in schedule}
        assert sched_map["cfbd:fcs_opp"] == -25.0
        assert sched_map["cfbd:rated_opp"] == 14.5

    def test_schedule_with_elo_db_rating(self, conn) -> None:
        self._seed_db(conn)
        store.upsert_team(conn, {"team_id": "cfbd:elo_opp", "school": "Elo Team"})
        store.insert_team_ratings(
            conn,
            [
                {
                    "snapshot_id": "r-elo",
                    "season": 2026,
                    "period": "regular:week:01",
                    "week": 1,
                    "season_type": "regular",
                    "team_id": "cfbd:elo_opp",
                    "source": "elo_cfbd",
                    "snapshot_scope": "weekly",
                    "provenance_mode": "reconstructed",
                    "as_of_utc": "2026-08-01T00:00:00+00:00",
                    "ingested_utc": "2026-08-01T00:00:00+00:00",
                    "rating": 1800.0,  # 1800 Elo -> (1800 - 1500) / 25 = +12.0 spread points
                    "sos": 0.0,
                }
            ],
        )
        store.upsert_cfbd_game(
            conn,
            {
                "game_id": "cfbd:game_elo",
                "season": 2026,
                "week": 4,
                "season_type": "regular",
                "kickoff_utc": "2026-09-26T16:00:00+00:00",
                "football_date": "2026-09-26",
                "home_team_id": "cfbd:ohio_st",
                "away_team_id": "cfbd:elo_opp",
            },
        )
        conn.commit()

        schedule = load_team_schedule(
            conn,
            "cfbd:ohio_st",
            2026,
            opponent_ratings={"cfbd:akron": -15.0, "cfbd:michigan": 18.0},
        )
        sched_map = {g.opponent_id: g.opponent_power_rating for g in schedule}
        assert sched_map["cfbd:elo_opp"] == 12.0

    def test_postseason_and_ccg_game_excluded_from_schedule(self, conn) -> None:
        self._seed_db(conn)
        store.upsert_cfbd_game(
            conn,
            {
                "game_id": "cfbd:game_ccg",
                "season": 2026,
                "week": 14,
                "season_type": "championship",
                "kickoff_utc": "2026-12-05T19:00:00+00:00",
                "football_date": "2026-12-05",
                "home_team_id": "cfbd:ohio_st",
                "away_team_id": "cfbd:michigan",
            },
        )
        conn.commit()

        schedule = load_team_schedule(
            conn,
            "cfbd:ohio_st",
            2026,
            opponent_ratings={"cfbd:akron": -15.0, "cfbd:michigan": 18.0},
        )
        # Only the two regular-season games load; the week-14 championship game
        # is excluded. Assert on (opponent, week) pairs rather than a bare count:
        # Michigan is the opponent in BOTH week 13 and the CCG, so len() == 2 also
        # passes if game 1 is wrongly dropped and the CCG wrongly kept. Order is
        # guaranteed by the ORDER BY kickoff_utc ASC in load_team_schedule.
        assert [(g.opponent_id, g.game_week) for g in schedule] == [
            ("cfbd:akron", 1),
            ("cfbd:michigan", 13),
        ]

    def test_completed_games_are_deterministic_in_win_total(self, conn) -> None:
        self._seed_db(conn)
        conn.execute(
            """UPDATE games
               SET completed = 1, status = 'completed', home_points = 35, away_points = 10
               WHERE game_id = 'cfbd:game_1'"""
        )
        conn.execute(
            """UPDATE games
               SET completed = 1, status = 'completed', home_points = 21, away_points = 24
               WHERE game_id = 'cfbd:game_2'"""
        )
        conn.commit()

        schedule = load_team_schedule(
            conn,
            "cfbd:ohio_st",
            2026,
            opponent_ratings={"cfbd:akron": -15.0, "cfbd:michigan": 18.0},
        )
        assert [game.known_result for game in schedule] == [1.0, 0.0]

        inputs = RosterTalentInputs(
            team_id="cfbd:ohio_st",
            recruiting_composite=990.0,
            portal_composite=0.0,
            nil_tier=NILTier.TIER_2_UPPER_P4,
            returning_production=0.72,
            conference="Big Ten",
        )
        projection = project_season_futures(inputs, schedule)
        assert projection.expected_wins == 1.0
        assert projection.win_variance == 0.0
        assert projection.win_distribution == [0.0, 1.0, 0.0]

    def test_missing_opponent_rating_fails_closed(self, conn) -> None:
        self._seed_db(conn)
        with pytest.raises(SchemaError, match="No admissible opponent rating"):
            load_team_schedule(conn, "cfbd:ohio_st", 2026)

    def test_missing_nil_is_not_inferred_from_conference(self, conn) -> None:
        self._seed_db(conn)
        kwargs = self._roster_overrides()
        kwargs.pop("nil_tier")
        with pytest.raises(SchemaError, match="NIL tier or budget is required"):
            load_team_roster_inputs(conn, "cfbd:ohio_st", 2026, **kwargs)

    def test_historical_conference_overrides_mutable_team_dimension(self, conn) -> None:
        self._seed_db(conn)
        store.upsert_team(
            conn,
            {
                "team_id": "cfbd:ohio_st",
                "school": "Ohio State",
                "conference": "SEC",
            },
        )
        conn.commit()

        inputs = load_team_roster_inputs(conn, "cfbd:ohio_st", 2026, **self._roster_overrides())
        assert inputs.conference == "Big Ten"

    def test_store_upsert_minimal_payloads(self, conn) -> None:
        # Verify minimal dictionary inputs don't crash on missing binding parameters
        store.upsert_team(conn, {"team_id": "cfbd:minimal"})
        store.upsert_venue(conn, {"venue_id": "v_minimal", "name": "Minimal Stadium"})
        store.upsert_game(
            conn,
            {
                "game_id": "g_minimal",
                "season": 2026,
                "kickoff_utc": "2026-09-05T16:00:00+00:00",
                "home_team_id": "cfbd:minimal",
                "away_team_id": "cfbd:minimal",
            },
        )
        conn.commit()

        row = conn.execute(
            "SELECT team_id, school FROM teams WHERE team_id = 'cfbd:minimal'"
        ).fetchone()
        assert row["team_id"] == "cfbd:minimal"
        assert row["school"] is None

    def test_sparse_cfbd_update_preserves_completed_state(self, conn) -> None:
        self._seed_db(conn)
        conn.execute(
            """UPDATE games
               SET completed = 1, status = 'completed', home_points = 35, away_points = 10
               WHERE game_id = 'cfbd:game_1'"""
        )
        conn.commit()

        store.upsert_cfbd_game(
            conn,
            {
                "game_id": "cfbd:game_1",
                "season": 2026,
                "kickoff_utc": "2026-09-05T16:00:00+00:00",
                "home_team_id": "cfbd:ohio_st",
                "away_team_id": "cfbd:akron",
            },
        )
        conn.commit()

        row = conn.execute(
            """SELECT week, season_type, neutral_site, conference_game,
                      status, home_points, away_points, completed
               FROM games WHERE game_id = 'cfbd:game_1'"""
        ).fetchone()
        assert dict(row) == {
            "week": 1,
            "season_type": "regular",
            "neutral_site": 0,
            "conference_game": 0,
            "status": "completed",
            "home_points": 35,
            "away_points": 10,
            "completed": 1,
        }

    def test_fbs_independents_conference_naming(self) -> None:
        overall_pmf = calculate_win_total_distribution([0.95] * 12)
        # All variations of independent naming should have 0 CCG prob and 0 auto-bid prob
        for conf_name in ("FBS Independents", "Independent", "independents", "ind", "Notre Dame"):
            ccg_proj = estimate_conference_championship_prob(
                overall_pmf, conf_name, team_rating=15.0
            )
            assert ccg_proj.prob_reach_ccg == 0.0
            assert ccg_proj.prob_win_ccg == 0.0

            # With prob_win_ccg passed as 0.80, independents still have 0 auto-bid
            cfp_p = estimate_cfp_appearance_prob(
                overall_pmf, conf_name, team_rating=15.0, prob_win_ccg=0.80
            )
            # Should not get G5 auto-bid multiplier
            assert cfp_p < 0.90

    def test_g5_rating_sos_smooth_transition(self) -> None:
        overall_pmf = calculate_win_total_distribution([0.98] * 12)
        # Verify monotonically increasing CFP probability without discrete step jumps
        probs = []
        for r in [17.0, 18.0, 19.0, 20.0, 21.0, 22.0]:
            p = estimate_cfp_appearance_prob(
                overall_pmf, "American", team_rating=r, sos=1.0, prob_win_ccg=0.0
            )
            probs.append(p)

        # Non-decreasing
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1]
        # Smooth differences: no jump > 0.10 between 1-point rating increments
        for i in range(len(probs) - 1):
            assert probs[i + 1] - probs[i] < 0.10

    def test_weak_independent_at_large_cutoff(self) -> None:
        # A mediocre 6-6 / 7-5 / 11-1 independent team should not get free CFP tickets
        pmf_11_1 = calculate_win_total_distribution([0.80] * 12)
        p_cfp_weak = estimate_cfp_appearance_prob(
            pmf_11_1, "Independent", team_rating=2.0, sos=-4.0
        )
        assert p_cfp_weak < 0.02
