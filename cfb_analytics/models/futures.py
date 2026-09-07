"""Season Futures, NIL & Roster Valuation Engine (Model 4).

Macro-level predictive framework for college football season futures, incorporating:
- Transfer portal net composite (additions vs. departures)
- Estimated NIL budget and tier classification
- True Talent Composite (recruiting rankings + portal grades + NIL adjustment)
- Returning production (split by offense and defense)
- Strength of Schedule (SOS)
- Quarterback continuity and tier

Core Projections & Calculations:
- Adjusted team power rating and True Talent rating
- Game-by-game schedule projections with win probabilities
- Regular season win total distributions via exact Poisson Binomial closed-form DP
- Fair Over/Under probabilities and devigged edge calculations for posted win totals
- Conference Championship Game appearance and win probabilities
- 12-Team College Football Playoff (CFP) appearance probability estimation
- Monte Carlo season simulation engine with reproducible seeds

Pure Python standard library compliance (no numpy, scipy, pandas, or scikit-learn).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from cfb_analytics.errors import SchemaError
from cfb_analytics.utils import american_to_decimal, decimal_to_american, implied_probability

# ---------------------------------------------------------------------------
# Constants & Model Defaults
# ---------------------------------------------------------------------------

# Point spread standard deviation for margin modeling.
# Matches config/settings.json ("margin_sigma_base": 16.5)
DEFAULT_MARGIN_SIGMA: float = 16.5

# Standard College Football Home Field Advantage in spread points.
# Corresponds to ~60 Elo points (models/elo.py: HFA_ELO_POINTS = 60.0)
DEFAULT_HFA_POINTS: float = 2.5

# Baseline recruiting talent composite for an average FBS program (0.0 spread points)
BASELINE_RECRUITING_COMPOSITE: float = 650.0

# Recruiting points scaling factor: (composite - 650) * 0.055
# A score of 1000 (elite Alabama/Georgia) produces ~+19.25 points above average FBS.
RECRUITING_TALENT_SCALE: float = 0.055

# Weight per net transfer portal composite point
PORTAL_NET_WEIGHT: float = 0.15

# Scaling factor from net transfer portal score to recruiting composite equivalent points
PORTAL_TO_RECRUITING_COMPOSITE_SCALE: float = 2.75

# Baseline returning production percentage across FBS (60%)
BASELINE_RETURNING_PRODUCTION: float = 0.60

# Returning production multipliers (offense and defense)
DEFAULT_RET_PROD_OFFENSE_COEFF: float = 4.0
DEFAULT_RET_PROD_DEFENSE_COEFF: float = 3.0

# Elo scaling factor: 1 spread point is ~25 Elo points
POINTS_TO_ELO_SCALE: float = 25.0
DEFAULT_BASELINE_ELO: float = 1500.0


# ---------------------------------------------------------------------------
# NIL Tiers & Budget Classifications
# ---------------------------------------------------------------------------


class NILTier(StrEnum):
    """NIL budget tier classifications reflecting modern college football compensation."""

    TIER_1_ELITE = "tier_1_elite"  # $18M - $25M+ (Ohio State, Texas, Oregon, Georgia)
    TIER_2_UPPER_P4 = "tier_2_upper_p4"  # $12M - $18M (LSU, Penn State, Michigan, Ole Miss)
    TIER_3_MID_P4 = "tier_3_mid_p4"  # $6M - $12M (Wisconsin, Iowa, Kentucky, Missouri)
    TIER_4_LOWER_P4_HIGH_G5 = (
        "tier_4_lower_p4_high_g5"  # $3M - $6M (Vanderbilt, Boise State, Memphis)
    )
    TIER_5_G5_BASELINE = "tier_5_g5_baseline"  # < $3M (MAC, Sun Belt, C-USA baselines)

    @classmethod
    def from_budget(cls, budget_millions: float) -> NILTier:
        """Classify estimated NIL budget (in millions of dollars) into a tier."""
        budget = _finite_float(budget_millions, name="NIL budget")
        if budget < 0:
            raise SchemaError(f"NIL budget cannot be negative, got {budget_millions!r}")
        if budget >= 18.0:
            return cls.TIER_1_ELITE
        elif budget >= 12.0:
            return cls.TIER_2_UPPER_P4
        elif budget >= 6.0:
            return cls.TIER_3_MID_P4
        elif budget >= 3.0:
            return cls.TIER_4_LOWER_P4_HIGH_G5
        return cls.TIER_5_G5_BASELINE

    @property
    def baseline_budget(self) -> float:
        """Approximate median budget (in millions) for this tier."""
        mapping = {
            self.TIER_1_ELITE: 20.0,
            self.TIER_2_UPPER_P4: 15.0,
            self.TIER_3_MID_P4: 9.0,
            self.TIER_4_LOWER_P4_HIGH_G5: 4.5,
            self.TIER_5_G5_BASELINE: 1.5,
        }
        return mapping[self]

    @property
    def spread_adjustment(self) -> float:
        """Power rating adjustment (in spread points) associated with this tier."""
        mapping = {
            self.TIER_1_ELITE: 3.5,
            self.TIER_2_UPPER_P4: 2.0,
            self.TIER_3_MID_P4: 0.0,
            self.TIER_4_LOWER_P4_HIGH_G5: -2.0,
            self.TIER_5_G5_BASELINE: -4.0,
        }
        return mapping[self]


# ---------------------------------------------------------------------------
# Quarterback Tiers & Continuity
# ---------------------------------------------------------------------------


class QBTier(StrEnum):
    """Quarterback talent tier."""

    UNKNOWN = "unknown"
    TIER_1_ELITE = "tier_1_elite"  # Heisman contender / Top 10 NFL draft pick (+4.5 pts)
    TIER_2_QUALITY_STARTER = (
        "tier_2_quality_starter"  # Proven upper-tier P4 starter / All-Conf (+2.0 pts)
    )
    TIER_3_AVERAGE = "tier_3_average"  # Average FBS starter / Game manager (0.0 pts)
    TIER_4_DEVELOPING_UNPROVEN = (
        "tier_4_developing_unproven"  # Underclassman / low P4 / mid G5 (-2.5 pts)
    )
    TIER_5_LIABILITY = "tier_5_liability"  # Severe weakness / major liability (-5.0 pts)

    @property
    def spread_adjustment(self) -> float:
        mapping = {
            self.UNKNOWN: 0.0,
            self.TIER_1_ELITE: 4.5,
            self.TIER_2_QUALITY_STARTER: 2.0,
            self.TIER_3_AVERAGE: 0.0,
            self.TIER_4_DEVELOPING_UNPROVEN: -2.5,
            self.TIER_5_LIABILITY: -5.0,
        }
        return mapping[self]


class QBContinuity(StrEnum):
    """Quarterback system continuity and experience classification."""

    UNKNOWN = "unknown"
    RETURNING_MULTI_YEAR = "returning_multi_year"  # 2+ years starting in same offense (+1.0 pt)
    RETURNING_STARTER_SAME_SYSTEM = (
        "returning_starter_same_system"  # Returning starter, same offensive coordinator (+0.5 pt)
    )
    RETURNING_STARTER_NEW_OC = (
        "returning_starter_new_oc"  # Returning starter, new offensive coordinator (0.0 pt)
    )
    TRANSFERRED_PROVEN_STARTER = (
        "transferred_proven_starter"  # Transferred in with substantial FBS starts (-0.5 pt)
    )
    TRANSFERRED_UNPROVEN = (
        "transferred_unproven"  # Transferred in with limited live game reps (-1.2 pts)
    )
    FIRST_YEAR_STARTER_IN_SYSTEM = (
        "first_year_starter_in_system"  # Promoted internal backup (-1.0 pt)
    )
    TRUE_FRESHMAN = "true_freshman"  # True freshman starter (-2.0 pts)
    UNRESOLVED_BATTLE = "unresolved_battle"  # Competition unsettled through fall camp (-2.5 pts)

    @property
    def spread_adjustment(self) -> float:
        mapping = {
            self.UNKNOWN: 0.0,
            self.RETURNING_MULTI_YEAR: 1.0,
            self.RETURNING_STARTER_SAME_SYSTEM: 0.5,
            self.RETURNING_STARTER_NEW_OC: 0.0,
            self.TRANSFERRED_PROVEN_STARTER: -0.5,
            self.TRANSFERRED_UNPROVEN: -1.2,
            self.FIRST_YEAR_STARTER_IN_SYSTEM: -1.0,
            self.TRUE_FRESHMAN: -2.0,
            self.UNRESOLVED_BATTLE: -2.5,
        }
        return mapping[self]


# ---------------------------------------------------------------------------
# True Talent Composite & Data Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortalComposite:
    """Transfer portal composite grading (additions vs departures)."""

    additions_score: float
    departures_score: float
    net_composite: float | None = None
    additions_count: int = 0
    departures_count: int = 0

    def __post_init__(self) -> None:
        additions = _finite_float(self.additions_score, name="portal additions score")
        departures = _finite_float(self.departures_score, name="portal departures score")
        object.__setattr__(self, "additions_score", additions)
        object.__setattr__(self, "departures_score", departures)
        if self.additions_count < 0 or self.departures_count < 0:
            raise SchemaError(
                "Portal counts cannot be negative, got "
                f"additions={self.additions_count}, departures={self.departures_count}"
            )
        if self.net_composite is None:
            object.__setattr__(self, "net_composite", round(additions - departures, 2))
        else:
            object.__setattr__(
                self,
                "net_composite",
                _finite_float(self.net_composite, name="portal net composite"),
            )

    @property
    def is_net_positive(self) -> bool:
        return self.net_composite is not None and self.net_composite > 0.0


def calculate_true_talent_composite(
    recruiting_composite: float,
    portal: PortalComposite | float,
    scale: float = PORTAL_TO_RECRUITING_COMPOSITE_SCALE,
) -> float:
    """Calculate the True Talent Composite.

    Combines recruiting rankings with transfer portal grades.
    """
    recruiting = _finite_float(recruiting_composite, name="recruiting composite")
    scale_value = _finite_float(scale, name="portal composite scale")
    if recruiting < 0:
        raise SchemaError(f"Recruiting composite cannot be negative, got {recruiting_composite}")
    if isinstance(portal, (int, float)):
        net_score = float(portal)
    elif isinstance(portal, PortalComposite):
        net_score = (
            portal.net_composite
            if portal.net_composite is not None
            else (portal.additions_score - portal.departures_score)
        )
    else:
        raise SchemaError(f"Expected PortalComposite or float, got {type(portal).__name__}")

    net_score = _finite_float(net_score, name="portal net composite")
    composite = recruiting + net_score * scale_value
    if composite < 0:
        raise SchemaError(f"True talent composite cannot be negative, got {composite}")
    return round(composite, 2)


@dataclass(frozen=True)
class ReturningProduction:
    """Returning production inputs without inventing a defensive split.

    CFBD's returning-production endpoint reports total offensive PPA and
    passing/rushing/receiving components, but no defensive returning-production
    measure. ``percent_ppa_defense`` is therefore optional and is only applied
    when a caller supplies an independently sourced value.
    """

    percent_ppa_offense: float  # 0.0 to 1.0
    percent_ppa_defense: float | None = None  # 0.0 to 1.0 when independently sourced
    percent_overall: float | None = None

    def __post_init__(self) -> None:
        offense = _finite_float(self.percent_ppa_offense, name="returning offense production")
        defense = (
            _finite_float(self.percent_ppa_defense, name="returning defense production")
            if self.percent_ppa_defense is not None
            else None
        )
        object.__setattr__(self, "percent_ppa_offense", offense)
        object.__setattr__(self, "percent_ppa_defense", defense)
        if not (0.0 <= offense <= 1.0):
            raise SchemaError(
                "Returning offense production must be in [0.0, 1.0], "
                f"got {self.percent_ppa_offense}"
            )
        if defense is not None and not (0.0 <= defense <= 1.0):
            raise SchemaError(
                "Returning defense production must be in [0.0, 1.0], "
                f"got {self.percent_ppa_defense}"
            )
        if self.percent_overall is None:
            blended = offense if defense is None else 0.53 * offense + 0.47 * defense
            object.__setattr__(self, "percent_overall", round(blended, 4))
        else:
            overall = _finite_float(self.percent_overall, name="returning overall production")
            if not 0.0 <= overall <= 1.0:
                raise SchemaError(
                    f"Returning overall production must be in [0.0, 1.0], got {overall}"
                )
            object.__setattr__(self, "percent_overall", overall)


@dataclass(frozen=True)
class RosterTalentInputs:
    """Comprehensive input payload for Model 4 team valuation."""

    team_id: str
    recruiting_composite: float
    portal_composite: PortalComposite | float
    nil_tier: NILTier | str
    returning_production: ReturningProduction | float
    qb_tier: QBTier | str = QBTier.UNKNOWN
    qb_continuity: QBContinuity | str = QBContinuity.UNKNOWN
    nil_budget_millions: float | None = None
    strength_of_schedule: float = 0.0
    base_power_rating: float | None = None
    conference: str = "Independent"
    true_talent_composite: float | None = None

    def __post_init__(self) -> None:
        recruiting = _finite_float(self.recruiting_composite, name="recruiting composite")
        object.__setattr__(self, "recruiting_composite", recruiting)
        if recruiting < 0:
            raise SchemaError(
                f"Recruiting composite cannot be negative, got {self.recruiting_composite}"
            )
        if self.true_talent_composite is None:
            composite = calculate_true_talent_composite(
                self.recruiting_composite, self.portal_composite
            )
            object.__setattr__(self, "true_talent_composite", composite)
        else:
            composite = _finite_float(self.true_talent_composite, name="true talent composite")
            if composite < 0:
                raise SchemaError(f"True talent composite cannot be negative, got {composite}")
            object.__setattr__(self, "true_talent_composite", composite)
        _finite_float(self.strength_of_schedule, name="strength of schedule")
        if self.base_power_rating is not None:
            _finite_float(self.base_power_rating, name="base power rating")


@dataclass(frozen=True)
class AdjustedPowerRating:
    """Decomposition of team power rating incorporating roster & NIL adjustments."""

    team_id: str
    base_talent_rating: float
    portal_adjustment: float
    nil_adjustment: float
    returning_production_adjustment: float
    qb_adjustment: float
    adjusted_power_rating: float
    elo_equivalent: float
    true_talent_composite: float | None = None

    @property
    def true_talent_rating(self) -> float:
        """Alias for adjusted_power_rating incorporating portal & NIL factors."""
        return self.adjusted_power_rating


@dataclass(frozen=True)
class ScheduledOpponent:
    """An individual game on a team's schedule."""

    opponent_id: str
    opponent_power_rating: float
    is_home: bool = True
    is_neutral: bool = False
    game_week: int | None = None
    is_conference: bool = False
    known_result: float | None = None

    def __post_init__(self) -> None:
        _finite_float(self.opponent_power_rating, name="opponent power rating")
        if self.known_result is not None:
            result = _finite_float(self.known_result, name="known game result")
            if result not in (0.0, 1.0):
                raise SchemaError(f"Known game result must be 0.0 or 1.0, got {result}")
            object.__setattr__(self, "known_result", result)


@dataclass(frozen=True)
class GameWinProjection:
    """Projected spread and win probability for an individual game."""

    opponent_id: str
    projected_spread: float  # Negative indicates favorite (e.g. -7.5)
    win_probability: float  # 0.001 to 0.999
    is_home: bool
    is_neutral: bool
    is_conference: bool
    game_week: int | None = None


@dataclass(frozen=True)
class WinTotalLineEvaluation:
    """Evaluation of a posted regular season win total line."""

    line: float
    prob_over: float
    prob_under: float
    prob_push: float
    fair_over_american: int | None
    fair_under_american: int | None
    recommended_side: str  # "OVER" | "UNDER" | "PASS"
    edge: float  # Model prob vs market implied prob
    expected_value: float | None = None


@dataclass(frozen=True)
class ConferenceChampionshipProjection:
    """Conference championship appearance and win probabilities."""

    conference: str
    prob_reach_ccg: float
    prob_win_ccg: float
    expected_conference_wins: float


@dataclass(frozen=True)
class SeasonFuturesProjection:
    """Macro-level season futures forecast for a single program."""

    team_id: str
    conference: str
    adjusted_rating: AdjustedPowerRating
    schedule_projections: list[GameWinProjection]
    expected_wins: float
    win_variance: float
    win_stdev: float
    win_distribution: list[float]  # pmf[k] = P(W = k) for k = 0..N
    expected_conference_wins: float
    conf_win_distribution: list[float]
    prob_reach_conference_championship: float
    prob_win_conference_championship: float
    prob_cfp_appearance: float
    win_total_evaluations: dict[float, WinTotalLineEvaluation] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResults:
    """Empirical distributions from Monte Carlo season simulation."""

    n_simulations: int
    mean_wins: float
    median_wins: float
    p10_wins: float
    p25_wins: float
    p75_wins: float
    p90_wins: float
    simulated_win_probabilities: dict[int, float]
    simulated_line_over_probs: dict[float, float]


# ---------------------------------------------------------------------------
# Helper Mathematical Functions
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function (CDF) via stdlib math.erf."""
    value = _finite_float(x, name="normal CDF input")
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _finite_float(value: Any, *, name: str) -> float:
    """Coerce a numeric model input, rejecting NaN, infinity, and non-numbers."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise SchemaError(f"{name} must be a finite number, got {value!r}")
    return number


def _probabilities(values: Sequence[float], *, name: str) -> list[float]:
    probabilities: list[float] = []
    for index, value in enumerate(values):
        probability = _finite_float(value, name=f"{name}[{index}]")
        if not 0.0 <= probability <= 1.0:
            raise SchemaError(f"{name}[{index}] must be between 0.0 and 1.0, got {probability}")
        probabilities.append(probability)
    return probabilities


def _validated_pmf(values: Sequence[float], *, name: str) -> list[float]:
    pmf = _probabilities(values, name=name)
    if not pmf:
        raise SchemaError(f"{name} cannot be empty")
    total = sum(pmf)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise SchemaError(f"{name} must sum to 1.0, got {total}")
    return pmf


def prob_to_american(p: float) -> int | None:
    """Convert a fair probability in (0, 1) to fair American odds."""
    if not (0.0 < p < 1.0):
        return None
    decimal_price = 1.0 / p
    return decimal_to_american(decimal_price)


def clamp(val: float, low: float, high: float) -> float:
    """Clamp a value between low and high bounds."""
    return max(low, min(high, val))


# ---------------------------------------------------------------------------
# Component Adjustment Calculators
# ---------------------------------------------------------------------------


def classify_nil_tier(budget_millions: float) -> NILTier:
    """Classify an estimated NIL budget (in millions) into an NILTier."""
    return NILTier.from_budget(budget_millions)


def nil_rating_adjustment(tier: NILTier | str, budget_millions: float | None = None) -> float:
    """Calculate spread point adjustment from NIL tier and/or exact budget."""
    if isinstance(tier, str):
        try:
            tier_enum = NILTier(tier)
        except ValueError:
            matched = False
            for t in NILTier:
                if t.value.lower() == tier.lower() or t.name.lower() == tier.lower():
                    tier_enum = t
                    matched = True
                    break
            if not matched:
                raise SchemaError(f"Invalid NIL tier {tier!r}") from None
    else:
        tier_enum = tier

    if budget_millions is not None:
        budget = _finite_float(budget_millions, name="NIL budget")
        if budget < 0:
            raise SchemaError(f"NIL budget cannot be negative, got {budget_millions!r}")
        # Continuous linear adjustment anchored at $9.0M (Mid-P4 baseline)
        adj = (budget - 9.0) * 0.35
        return round(clamp(adj, -5.0, 5.0), 2)

    return tier_enum.spread_adjustment


def portal_rating_adjustment(
    portal: PortalComposite | float, weight: float = PORTAL_NET_WEIGHT
) -> float:
    """Calculate power rating adjustment from net transfer portal composite."""
    if isinstance(portal, (int, float)):
        net_score = float(portal)
    elif isinstance(portal, PortalComposite):
        net_score = (
            portal.net_composite
            if portal.net_composite is not None
            else (portal.additions_score - portal.departures_score)
        )
    else:
        raise SchemaError(f"Expected PortalComposite or float, got {type(portal).__name__}")

    net_score = _finite_float(net_score, name="portal net composite")
    resolved_weight = _finite_float(weight, name="portal rating weight")
    adj = net_score * resolved_weight
    return round(clamp(adj, -6.0, 6.0), 2)


def returning_production_adjustment(
    ret_prod: ReturningProduction | float,
    coeff_off: float = DEFAULT_RET_PROD_OFFENSE_COEFF,
    coeff_def: float = DEFAULT_RET_PROD_DEFENSE_COEFF,
) -> float:
    """Calculate rating adjustment from returning production relative to FBS baseline (0.60)."""
    offense_coefficient = _finite_float(coeff_off, name="returning offense coefficient")
    defense_coefficient = _finite_float(coeff_def, name="returning defense coefficient")
    if isinstance(ret_prod, (int, float)):
        val = float(ret_prod)
        if not (0.0 <= val <= 1.0):
            raise SchemaError(
                f"Returning production percentage must be between 0.0 and 1.0, got {val}"
            )
        # A scalar explicitly represents the same value for both phases.
        adj = (offense_coefficient + defense_coefficient) * (
            val - BASELINE_RETURNING_PRODUCTION
        )
    elif isinstance(ret_prod, ReturningProduction):
        for name, pct in (
            ("offense", ret_prod.percent_ppa_offense),
            ("defense", ret_prod.percent_ppa_defense),
        ):
            if pct is None:
                continue
            if not (0.0 <= pct <= 1.0):
                raise SchemaError(f"Returning {name} production must be in [0.0, 1.0], got {pct}")
        adj = offense_coefficient * (
            ret_prod.percent_ppa_offense - BASELINE_RETURNING_PRODUCTION
        )
        if ret_prod.percent_ppa_defense is not None:
            adj += defense_coefficient * (
                ret_prod.percent_ppa_defense - BASELINE_RETURNING_PRODUCTION
            )
    else:
        raise SchemaError(f"Expected ReturningProduction or float, got {type(ret_prod).__name__}")

    return round(clamp(adj, -5.0, 5.0), 2)


def qb_rating_adjustment(tier: QBTier | str, continuity: QBContinuity | str) -> float:
    """Calculate total quarterback spread point adjustment from tier and continuity."""
    if isinstance(tier, str):
        try:
            q_tier = QBTier(tier)
        except ValueError:
            matched = False
            for t in QBTier:
                if t.value.lower() == tier.lower() or t.name.lower() == tier.lower():
                    q_tier = t
                    matched = True
                    break
            if not matched:
                raise SchemaError(f"Invalid QB tier {tier!r}") from None
    else:
        q_tier = tier

    if isinstance(continuity, str):
        try:
            q_cont = QBContinuity(continuity)
        except ValueError:
            matched = False
            for c in QBContinuity:
                if c.value.lower() == continuity.lower() or c.name.lower() == continuity.lower():
                    q_cont = c
                    matched = True
                    break
            if not matched:
                raise SchemaError(f"Invalid QB continuity {continuity!r}") from None
    else:
        q_cont = continuity

    total = q_tier.spread_adjustment + q_cont.spread_adjustment
    return round(clamp(total, -7.5, 7.5), 2)


def calculate_true_talent_rating(inputs: RosterTalentInputs) -> AdjustedPowerRating:
    """Synthesize roster-talent inputs into an adjusted power rating.

    Combines recruiting, portal, NIL, returning production and QB inputs.
    """
    if inputs.base_power_rating is not None:
        base_rating = float(inputs.base_power_rating)
    else:
        # Derive baseline from recruiting talent composite
        recruiting = inputs.recruiting_composite
        base_rating = (recruiting - BASELINE_RECRUITING_COMPOSITE) * RECRUITING_TALENT_SCALE

    portal_adj = portal_rating_adjustment(inputs.portal_composite)
    nil_adj = nil_rating_adjustment(inputs.nil_tier, inputs.nil_budget_millions)
    ret_prod_adj = returning_production_adjustment(inputs.returning_production)
    qb_adj = qb_rating_adjustment(inputs.qb_tier, inputs.qb_continuity)

    adjusted = round(base_rating + portal_adj + nil_adj + ret_prod_adj + qb_adj, 2)
    elo_equiv = round(DEFAULT_BASELINE_ELO + adjusted * POINTS_TO_ELO_SCALE, 1)

    return AdjustedPowerRating(
        team_id=inputs.team_id,
        base_talent_rating=round(base_rating, 2),
        portal_adjustment=portal_adj,
        nil_adjustment=nil_adj,
        returning_production_adjustment=ret_prod_adj,
        qb_adjustment=qb_adj,
        adjusted_power_rating=adjusted,
        elo_equivalent=elo_equiv,
        true_talent_composite=inputs.true_talent_composite,
    )


# ---------------------------------------------------------------------------
# Game-by-Game Schedule Projections
# ---------------------------------------------------------------------------


def project_game_win_probability(
    team_rating: float,
    opp_rating: float,
    *,
    is_home: bool = True,
    is_neutral: bool = False,
    hfa: float = DEFAULT_HFA_POINTS,
    sigma: float = DEFAULT_MARGIN_SIGMA,
) -> tuple[float, float]:
    """Calculate projected point spread and outright win probability for a game.

    Returns:
        (projected_spread, win_probability)
        Note: projected_spread is from the team's perspective;
        negative indicates favorite (e.g. -7.0).
    """
    team = _finite_float(team_rating, name="team rating")
    opponent = _finite_float(opp_rating, name="opponent rating")
    home_field = _finite_float(hfa, name="home-field advantage")
    margin_sigma = _finite_float(sigma, name="margin sigma")
    if margin_sigma <= 0.0:
        raise SchemaError(f"Margin sigma must be greater than 0.0, got {margin_sigma}")

    if is_neutral:
        loc_advantage = 0.0
    elif is_home:
        loc_advantage = home_field
    else:
        loc_advantage = -home_field

    projected_margin = team - opponent + loc_advantage
    spread = -round(projected_margin, 1)

    # Standard normal win probability based on projected margin
    p_win = norm_cdf(projected_margin / margin_sigma)
    clamped_prob = round(clamp(p_win, 0.001, 0.999), 4)

    return spread, clamped_prob


def project_schedule(
    team_rating: float,
    schedule: Sequence[ScheduledOpponent],
    *,
    hfa: float = DEFAULT_HFA_POINTS,
    sigma: float = DEFAULT_MARGIN_SIGMA,
) -> list[GameWinProjection]:
    """Project all games on a team's schedule."""
    projections: list[GameWinProjection] = []
    for game in schedule:
        spread, prob = project_game_win_probability(
            team_rating,
            game.opponent_power_rating,
            is_home=game.is_home,
            is_neutral=game.is_neutral,
            hfa=hfa,
            sigma=sigma,
        )
        if game.known_result is not None:
            prob = game.known_result
        projections.append(
            GameWinProjection(
                opponent_id=game.opponent_id,
                projected_spread=spread,
                win_probability=prob,
                is_home=game.is_home,
                is_neutral=game.is_neutral,
                is_conference=game.is_conference,
                game_week=game.game_week,
            )
        )
    return projections


# ---------------------------------------------------------------------------
# Poisson Binomial Exact Win Distribution (Closed-Form DP)
# ---------------------------------------------------------------------------


def calculate_win_total_distribution(win_probabilities: Sequence[float]) -> list[float]:
    """Compute exact Poisson Binomial PMF for independent Bernoulli game trials.

    pmf[k] is P(W = k) for k = 0, ..., N.
    O(N^2) dynamic programming algorithm, exact to floating-point precision.
    """
    pmf = [1.0]
    for probability in _probabilities(win_probabilities, name="win probabilities"):
        new_pmf = [0.0] * (len(pmf) + 1)
        # Outcome: team loses this game with prob (1 - p)
        for k, val in enumerate(pmf):
            new_pmf[k] += val * (1.0 - probability)
        # Outcome: team wins this game with prob p
        for k, val in enumerate(pmf):
            new_pmf[k + 1] += val * probability
        pmf = new_pmf

    # Normalize to strictly sum to 1.0
    total = sum(pmf)
    if total > 0.0:
        return [p / total for p in pmf]
    return pmf


def expected_wins(win_probabilities: Sequence[float]) -> float:
    """Exact expected wins E[W] = sum(p_i)."""
    return round(sum(_probabilities(win_probabilities, name="win probabilities")), 4)


def win_total_variance(win_probabilities: Sequence[float]) -> float:
    """Exact win total variance Var(W) = sum(p_i * (1 - p_i))."""
    probabilities = _probabilities(win_probabilities, name="win probabilities")
    return round(sum(p * (1.0 - p) for p in probabilities), 4)


# ---------------------------------------------------------------------------
# Posted Win Total Line Evaluation
# ---------------------------------------------------------------------------


def evaluate_win_total_line(
    win_pmf: Sequence[float],
    posted_line: float,
    *,
    market_over_price: int | None = None,
    market_under_price: int | None = None,
    edge_threshold: float = 0.03,
) -> WinTotalLineEvaluation:
    """Evaluate fair Over/Under probabilities and betting edges for a posted win total line.

    Supports both fractional lines (e.g. 7.5, 8.5) where pushes are impossible,
    and integer lines (e.g. 8.0, 9.0) where pushes occur with non-zero probability.
    """
    pmf = _validated_pmf(win_pmf, name="win PMF")
    line = _finite_float(posted_line, name="posted win-total line")
    threshold = _finite_float(edge_threshold, name="edge threshold")
    if threshold < 0.0:
        raise SchemaError(f"Edge threshold cannot be negative, got {threshold}")
    n_games = len(pmf) - 1
    if line < 0 or line > n_games:
        raise SchemaError(f"Posted line {posted_line} is outside possible win range [0, {n_games}]")

    is_integer_line = abs(line - round(line)) < 1e-6
    line_int = int(round(line))

    if is_integer_line:
        prob_push = pmf[line_int] if 0 <= line_int <= n_games else 0.0
        prob_over = sum(pmf[k] for k in range(line_int + 1, n_games + 1))
        prob_under = sum(pmf[k] for k in range(0, line_int))
    else:
        prob_push = 0.0
        ceil_val = math.ceil(line)
        floor_val = math.floor(line)
        prob_over = (
            sum(pmf[k] for k in range(ceil_val, n_games + 1)) if ceil_val <= n_games else 0.0
        )
        prob_under = sum(pmf[k] for k in range(0, floor_val + 1)) if floor_val >= 0 else 0.0

    # 2-way devigged fair probabilities (excluding push)
    decisive_prob = prob_over + prob_under
    if decisive_prob > 0.0:
        fair_over_prob = prob_over / decisive_prob
        fair_under_prob = prob_under / decisive_prob
    else:
        fair_over_prob = 0.5
        fair_under_prob = 0.5

    fair_over_american = prob_to_american(fair_over_prob)
    fair_under_american = prob_to_american(fair_under_prob)

    # Edge analysis against posted market odds
    recommended_side = "PASS"
    edge = 0.0
    ev: float | None = None

    if market_over_price is not None and market_under_price is not None:
        p_market_over = implied_probability(market_over_price)
        p_market_under = implied_probability(market_under_price)

        if p_market_over is not None and p_market_under is not None:
            # Multiplicative devig of market prices
            market_sum = p_market_over + p_market_under
            devigged_mkt_over = p_market_over / market_sum
            devigged_mkt_under = p_market_under / market_sum

            over_edge = fair_over_prob - devigged_mkt_over
            under_edge = fair_under_prob - devigged_mkt_under
            edge = round(max(over_edge, under_edge), 4)

            dec_over = american_to_decimal(market_over_price)
            dec_under = american_to_decimal(market_under_price)
            over_ev = (
                prob_over * dec_over - (prob_over + prob_under) if dec_over is not None else None
            )
            under_ev = (
                prob_under * dec_under - (prob_over + prob_under) if dec_under is not None else None
            )

            if (
                over_edge >= threshold
                and over_edge >= under_edge
                and over_ev is not None
                and over_ev > 0.0
            ):
                recommended_side = "OVER"
                ev = round(over_ev, 4)
            elif under_edge >= threshold and under_ev is not None and under_ev > 0.0:
                recommended_side = "UNDER"
                ev = round(under_ev, 4)

    return WinTotalLineEvaluation(
        line=line,
        prob_over=round(prob_over, 4),
        prob_under=round(prob_under, 4),
        prob_push=round(prob_push, 4),
        fair_over_american=fair_over_american,
        fair_under_american=fair_under_american,
        recommended_side=recommended_side,
        edge=edge,
        expected_value=ev,
    )


# ---------------------------------------------------------------------------
# Conference Championship & 12-Team CFP Estimation
# ---------------------------------------------------------------------------

P4_CONFERENCES = frozenset({"sec", "big ten", "big 12", "acc", "b1g", "big12"})
G5_CONFERENCES = frozenset(
    {"american", "aac", "mountain west", "mwc", "sun belt", "sbc", "mac", "c-usa", "cusa"}
)
INDEPENDENT_CONFERENCES = frozenset(
    {"independent", "ind", "fbs independents", "independents", "notre dame"}
)


def is_p4_conference(conference: str) -> bool:
    """True if conference is one of the Power 4 conferences."""
    return conference.strip().lower() in P4_CONFERENCES


def is_independent_conference(conference: str) -> bool:
    """True if program is an FBS Independent."""
    c = conference.strip().lower()
    return c in INDEPENDENT_CONFERENCES or "notre dame" in c or "independent" in c


def expected_ccg_opponent_rating(conference: str) -> float:
    """Expected power rating of the opposing finalist in the conference championship game."""
    conf = conference.strip().lower()
    if conf in ("sec", "big ten", "b1g"):
        return 20.0
    elif conf in ("big 12", "big12", "acc"):
        return 14.0
    elif conf in ("american", "aac", "mountain west", "mwc"):
        return 7.0
    elif conf in ("sun belt", "sbc"):
        return 4.0
    elif conf in ("mac", "c-usa", "cusa"):
        return 1.0
    return 10.0


def estimate_conference_championship_prob(
    conf_win_pmf: Sequence[float],
    conference: str,
    team_rating: float,
) -> ConferenceChampionshipProjection:
    """Estimate probability of reaching and winning the conference championship game.

    Division-less format: top 2 teams in the conference standings play in the CCG.
    """
    pmf = _validated_pmf(conf_win_pmf, name="conference-win PMF")
    rating = _finite_float(team_rating, name="team rating")
    n_conf_games = len(pmf) - 1
    if n_conf_games <= 0:
        return ConferenceChampionshipProjection(
            conference=conference,
            prob_reach_ccg=0.0,
            prob_win_ccg=0.0,
            expected_conference_wins=0.0,
        )

    if is_independent_conference(conference):
        return ConferenceChampionshipProjection(
            conference=conference,
            prob_reach_ccg=0.0,
            prob_win_ccg=0.0,
            expected_conference_wins=0.0,
        )

    # Conference standings appearance curve conditioned on number of conference wins
    prob_reach = 0.0
    min_wins_threshold = max(0, n_conf_games - 3)
    for w, pw in enumerate(pmf):
        if w < min_wins_threshold:
            continue
        if w >= n_conf_games:
            prob_reach += pw * 0.98
        elif w == n_conf_games - 1:
            prob_reach += pw * 0.75
        elif w == n_conf_games - 2:
            prob_reach += pw * 0.20
        elif w == n_conf_games - 3 and n_conf_games >= 6:
            prob_reach += pw * 0.02

    # Quality adjustment based on team power rating
    rating_bonus = clamp(rating * 0.015, -0.15, 0.15)
    prob_reach = clamp(prob_reach * (1.0 + rating_bonus), 0.0, 0.99)

    # In CCG, opponent is the other top team in conference
    opp_ccg_rating = expected_ccg_opponent_rating(conference)
    p_win_given_reach = norm_cdf((rating - opp_ccg_rating) / DEFAULT_MARGIN_SIGMA)
    prob_win = prob_reach * p_win_given_reach

    exp_conf_wins = sum(w * p for w, p in enumerate(pmf))

    return ConferenceChampionshipProjection(
        conference=conference,
        prob_reach_ccg=round(prob_reach, 4),
        prob_win_ccg=round(prob_win, 4),
        expected_conference_wins=round(exp_conf_wins, 2),
    )


def estimate_cfp_appearance_prob(
    overall_win_pmf: Sequence[float],
    conference: str,
    team_rating: float,
    sos: float = 0.0,
    prob_win_ccg: float = 0.0,
) -> float:
    """Estimate probability of qualifying for the 12-Team College Football Playoff.

    12-Team CFP Structure:
    - 5 highest-ranked conference champions (4 P4 champs + 1 G5 champ)
    - 7 at-large bids selected by the CFP selection committee
    """
    if not overall_win_pmf:
        return 0.0

    pmf = _validated_pmf(overall_win_pmf, name="overall-win PMF")
    if len(pmf) <= 1:
        return 0.0
    rating = _finite_float(team_rating, name="team rating")
    schedule_sos = _finite_float(sos, name="strength of schedule")
    ccg_win_probability = _finite_float(
        prob_win_ccg, name="conference championship win probability"
    )
    if not 0.0 <= ccg_win_probability <= 1.0:
        raise SchemaError(
            "Conference championship win probability must be between 0.0 and 1.0, "
            f"got {ccg_win_probability}"
        )

    conf_clean = conference.strip().lower()
    is_p4 = is_p4_conference(conf_clean)
    is_ind = is_independent_conference(conf_clean)

    # 1. Automatic Qualifier Path (Conference Champion)
    if is_p4:
        # P4 champions are virtually 100% guaranteed to be among top 5 champs
        p_auto_bid = ccg_win_probability * 0.99
    elif not is_ind:
        # G5 champion must be the top-ranked G5 champion among AAC, MWC, SBC, MAC, CUSA
        # Higher team rating and SOS significantly increases chance of being top G5 champ
        g5_rank_mult = clamp(
            0.35 + (rating / 25.0) * 0.45 + (schedule_sos / 15.0) * 0.15,
            0.10,
            0.95,
        )
        p_auto_bid = ccg_win_probability * g5_rank_mult
    else:
        # Independents cannot win a conference championship auto-bid
        p_auto_bid = 0.0

    # 2. At-Large Selection Curve conditioned on loss count
    at_large_prob = 0.0
    n_games = len(pmf) - 1

    for w, pw in enumerate(pmf):
        losses = n_games - w
        if n_games < 6:
            p_select = 0.0
        elif is_p4:
            if losses == 0:
                p_select = 1.00
            elif losses == 1:
                rating_adj = clamp((rating - 16.0) * 0.035, -0.75, 0.05)
                sos_adj = clamp(schedule_sos * 0.02, -0.10, 0.10)
                p_select = clamp(0.92 + rating_adj + sos_adj, 0.05, 0.99)
            elif losses == 2:
                rating_adj = clamp((rating - 18.0) * 0.045, -0.70, 0.20)
                sos_adj = clamp(schedule_sos * 0.03, -0.15, 0.15)
                p_select = clamp(0.70 + rating_adj + sos_adj, 0.00, 0.95)
            elif losses == 3:
                rating_adj = clamp((rating - 20.0) * 0.025, -0.20, 0.20)
                sos_adj = clamp(schedule_sos * 0.04, -0.05, 0.20)
                p_select = clamp(0.12 + rating_adj + sos_adj, 0.00, 0.35)
            else:
                p_select = 0.00
        elif is_ind:
            # Independent programs (e.g. Notre Dame, UConn) have no auto-bid
            # and rely entirely on at-large
            effective_rating = rating + clamp(schedule_sos * 0.5, -3.0, 3.0)
            if losses == 0:
                if effective_rating >= 14.0:
                    p_select = clamp(0.70 + (effective_rating - 14.0) * 0.05, 0.70, 1.00)
                else:
                    p_select = clamp((effective_rating - 6.0) * 0.08, 0.00, 0.70)
            elif losses == 1:
                if effective_rating >= 12.0:
                    p_select = clamp(0.30 + (effective_rating - 12.0) * 0.065, 0.00, 0.95)
                else:
                    p_select = 0.00
            elif losses == 2:
                if effective_rating >= 16.0:
                    p_select = clamp(0.20 + (effective_rating - 16.0) * 0.07, 0.00, 0.85)
                else:
                    p_select = 0.00
            elif losses == 3:
                if effective_rating >= 19.0:
                    p_select = clamp((effective_rating - 19.0) * 0.04, 0.00, 0.25)
                else:
                    p_select = 0.00
            else:
                p_select = 0.00
        else:
            # G5 teams rarely receive at-large bids.
            # Without winning the CCG, a G5 team cannot get the auto-bid and
            # virtually never gets an at-large bid
            # (~0.0 or <= 0.02 unless team_rating is exceptionally high e.g. >= 20.0).
            effective_rating = rating + clamp(schedule_sos * 0.5, -2.0, 4.0)
            if losses == 0 and w >= 10:
                if effective_rating >= 18.0:
                    p_select = clamp((effective_rating - 18.0) * 0.04, 0.00, 0.35)
                else:
                    p_select = 0.00
            elif losses == 1 and w >= 10 and effective_rating >= 22.0:
                p_select = clamp((effective_rating - 22.0) * 0.02, 0.00, 0.10)
            else:
                p_select = 0.00

        at_large_prob += pw * p_select

    # Combine auto-bid path and at-large path (independent events model)
    # P(CFP) = P(Auto) + (1 - P(Auto)) * P(At-Large)
    total_prob = p_auto_bid + (1.0 - p_auto_bid) * at_large_prob
    return round(clamp(total_prob, 0.0, 0.999), 4)


# ---------------------------------------------------------------------------
# High-Level Season Futures Engine
# ---------------------------------------------------------------------------


def project_season_futures(
    inputs: RosterTalentInputs,
    schedule: Sequence[ScheduledOpponent],
    posted_lines: Sequence[float] | None = None,
    *,
    hfa: float = DEFAULT_HFA_POINTS,
    sigma: float = DEFAULT_MARGIN_SIGMA,
) -> SeasonFuturesProjection:
    """End-to-end execution of Model 4 Season Futures & NIL Roster Valuation."""
    # 1. Roster Valuation & Adjusted Power Rating
    adjusted_rating = calculate_true_talent_rating(inputs)

    # 2. Game-by-game schedule projections
    game_projections = project_schedule(
        adjusted_rating.adjusted_power_rating,
        schedule,
        hfa=hfa,
        sigma=sigma,
    )

    all_win_probs = [gp.win_probability for gp in game_projections]
    conf_win_probs = [gp.win_probability for gp in game_projections if gp.is_conference]

    # 3. Exact Poisson Binomial win distributions
    overall_pmf = calculate_win_total_distribution(all_win_probs)
    conf_pmf = calculate_win_total_distribution(conf_win_probs) if conf_win_probs else [1.0]

    exp_total_wins = expected_wins(all_win_probs)
    win_var = win_total_variance(all_win_probs)
    win_sd = round(math.sqrt(win_var), 4)

    # 4. Conference Championship projections
    conf_proj = estimate_conference_championship_prob(
        conf_pmf,
        inputs.conference,
        adjusted_rating.adjusted_power_rating,
    )

    # 5. 12-Team CFP appearance estimation
    cfp_prob = estimate_cfp_appearance_prob(
        overall_pmf,
        inputs.conference,
        adjusted_rating.adjusted_power_rating,
        sos=inputs.strength_of_schedule,
        prob_win_ccg=conf_proj.prob_win_ccg,
    )

    # 6. Evaluate posted regular season win totals
    n_games = len(schedule)
    valid_lines: list[float]
    if n_games == 0:
        valid_lines = []
    elif posted_lines is not None:
        valid_lines = []
        for raw_line in posted_lines:
            line = _finite_float(raw_line, name="posted win-total line")
            if not 0.0 <= line <= n_games:
                raise SchemaError(
                    f"Posted line {line} is outside possible win range [0, {n_games}]"
                )
            valid_lines.append(line)
        valid_lines = sorted(set(valid_lines))
    else:
        min_line = 0.5
        max_line = max(min_line, n_games - 0.5)
        lines_to_eval = [
            exp_total_wins - 1.0,
            exp_total_wins - 0.5,
            round(exp_total_wins * 2) / 2.0,
            exp_total_wins + 0.5,
            exp_total_wins + 1.0,
        ]
        valid_lines = sorted(
            {clamp(round(line * 2) / 2.0, min_line, max_line) for line in lines_to_eval}
        )

    line_evaluations: dict[float, WinTotalLineEvaluation] = {}
    for line in valid_lines:
        line_evaluations[line] = evaluate_win_total_line(overall_pmf, line)

    return SeasonFuturesProjection(
        team_id=inputs.team_id,
        conference=inputs.conference,
        adjusted_rating=adjusted_rating,
        schedule_projections=game_projections,
        expected_wins=exp_total_wins,
        win_variance=win_var,
        win_stdev=win_sd,
        win_distribution=overall_pmf,
        expected_conference_wins=conf_proj.expected_conference_wins,
        conf_win_distribution=conf_pmf,
        prob_reach_conference_championship=conf_proj.prob_reach_ccg,
        prob_win_conference_championship=conf_proj.prob_win_ccg,
        prob_cfp_appearance=cfp_prob,
        win_total_evaluations=line_evaluations,
    )


# ---------------------------------------------------------------------------
# Monte Carlo Simulation Engine (Pure Python Standard Library)
# ---------------------------------------------------------------------------


def simulate_season_monte_carlo(
    schedule_win_probs: Sequence[float],
    posted_lines: Sequence[float] = (),
    *,
    n_simulations: int = 10000,
    seed: int | None = 42,
) -> SimulationResults:
    """Monte Carlo season simulation for empirical percentile calculations and verification.

    Uses stdlib random.Random for deterministic reproducibility when seed is supplied.
    """
    if n_simulations < 1:
        raise SchemaError("n_simulations must be >= 1")

    probabilities = _probabilities(schedule_win_probs, name="schedule win probabilities")

    rng = random.Random(seed)
    win_counts: list[int] = []
    n_games = len(probabilities)

    for _ in range(n_simulations):
        wins = 0
        for p in probabilities:
            if rng.random() < p:
                wins += 1
        win_counts.append(wins)

    win_counts.sort()

    mean_w = round(sum(win_counts) / n_simulations, 4)
    median_w = float(win_counts[n_simulations // 2])
    p10_w = float(win_counts[int(n_simulations * 0.10)])
    p25_w = float(win_counts[int(n_simulations * 0.25)])
    p75_w = float(win_counts[int(n_simulations * 0.75)])
    p90_w = float(win_counts[int(n_simulations * 0.90)])

    # Frequency distribution
    freq: dict[int, int] = {k: 0 for k in range(n_games + 1)}
    for w in win_counts:
        freq[w] = freq.get(w, 0) + 1
    sim_probs = {k: round(count / n_simulations, 4) for k, count in freq.items()}

    # Over line probabilities
    over_probs: dict[float, float] = {}
    for raw_line in posted_lines:
        line = _finite_float(raw_line, name="posted win-total line")
        if not 0.0 <= line <= n_games:
            raise SchemaError(f"Posted line {line} is outside possible win range [0, {n_games}]")
        over_count = sum(1 for w in win_counts if w > line)
        over_probs[line] = round(over_count / n_simulations, 4)

    return SimulationResults(
        n_simulations=n_simulations,
        mean_wins=mean_w,
        median_wins=median_w,
        p10_wins=p10_w,
        p25_wins=p25_w,
        p75_wins=p75_w,
        p90_wins=p90_w,
        simulated_win_probabilities=sim_probs,
        simulated_line_over_probs=over_probs,
    )
