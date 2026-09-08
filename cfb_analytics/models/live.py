"""Live Micro-Markets Modeling Engine.

Provides real-time in-game win probability, next-drive outcome estimation,
and dynamic live team totals projections. Strictly team-level micro-markets;
no player props allowed. Pure standard library design (no numpy, scipy, pandas).

The heuristics are uncalibrated research estimates. Structured outputs are
shadow-only and can never be actionable recommendations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cfb_analytics.errors import SchemaError
from cfb_analytics.features.live import REGULATION_SECONDS, DriveOutcome, GameState
from cfb_analytics.utils import decimal_to_american

# Supported live micro-market tokens
SUPPORTED_LIVE_MARKETS: tuple[str, ...] = (
    "live_win_probability",
    "live_moneyline",
    "next_drive_outcome",
    "live_team_total",
    "live_game_total",
    "live_spread",
    "live_drive_points",
)

MARKET_ALIASES: dict[str, str] = {
    "win_prob": "live_win_probability",
    "win_probability": "live_win_probability",
    "live_wp": "live_win_probability",
    "moneyline": "live_moneyline",
    "ml": "live_moneyline",
    "live_ml": "live_moneyline",
    "drive_outcome": "next_drive_outcome",
    "next_drive": "next_drive_outcome",
    "next_drive_result": "next_drive_outcome",
    "drive_result": "next_drive_outcome",
    "team_total": "live_team_total",
    "team_total_points": "live_team_total",
    "live_tt": "live_team_total",
    "game_total": "live_game_total",
    "total": "live_game_total",
    "live_total": "live_game_total",
    "spread": "live_spread",
    "live_spread": "live_spread",
    "drive_points": "live_drive_points",
}


# Standard deviation of college football game margin (full 60 min regulation)
CFB_GAME_MARGIN_SIGMA: float = 14.5
MODEL_STATUS = "uncalibrated_shadow"
MODEL_VERSION = "live_micro_markets_v1"


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise SchemaError(f"{name} must be numeric, got {value!r}")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(resolved):
        raise SchemaError(f"{name} must be finite, got {value!r}")
    return resolved


def _validated_lines(lines: tuple[float, ...], *, name: str) -> tuple[float, ...]:
    resolved = tuple(_finite_float(line, name=name) for line in lines)
    if any(line < 0.0 for line in resolved):
        raise SchemaError(f"{name} cannot contain negative totals")
    if len(set(resolved)) != len(resolved):
        raise SchemaError(f"{name} cannot contain duplicate lines")
    return resolved


def is_player_prop(market_or_type: str) -> bool:
    """True if market or type token suggests an individual player prop."""
    norm = market_or_type.strip().lower().replace("-", "_").replace(" ", "_")
    player_indicators = (
        "player",
        "passer",
        "rusher",
        "receiver",
        "touchdown",
        "anytime",
        "first_td",
        "last_td",
        "reception",
        "interception",
        "sack",
        "tackle",
        "field_goals_made",
        "longest_rush",
        "longest_pass",
        "longest_rec",
    )
    if any(ind in norm for ind in player_indicators):
        return True

    return (
        norm.startswith("pass_")
        or norm.startswith("rush_")
        or norm.startswith("rec_")
        or norm.startswith("passing_")
        or norm.startswith("rushing_")
        or norm.startswith("receiving_")
        or norm.endswith("_player")
    )


def validate_live_market(market: str, *, market_type: str = "LIVE_MICRO_MARKET") -> str:
    """Validate and normalize a live micro-market name.

    Raises SchemaError if the market is a player prop or unsupported.
    """
    if is_player_prop(market) or is_player_prop(market_type):
        raise SchemaError("Player props are strictly prohibited in Live Micro-Markets")

    norm_type = market_type.strip().upper()
    if norm_type not in ("LIVE_MICRO_MARKET", "TEAM_PROP", "GAMELINE"):
        raise SchemaError(f"Unsupported market_type {market_type!r} for live micro-markets")

    norm_market = market.strip().lower().replace("-", "_").replace(" ", "_")
    canonical = MARKET_ALIASES.get(norm_market, norm_market)

    if canonical not in SUPPORTED_LIVE_MARKETS:
        raise SchemaError(
            f"Unsupported live market {market!r}; supported markets are: {SUPPORTED_LIVE_MARKETS}"
        )
    allowed_types = {
        "live_win_probability": {"LIVE_MICRO_MARKET"},
        "live_moneyline": {"LIVE_MICRO_MARKET", "GAMELINE"},
        "next_drive_outcome": {"LIVE_MICRO_MARKET"},
        "live_team_total": {"LIVE_MICRO_MARKET", "TEAM_PROP"},
        "live_game_total": {"LIVE_MICRO_MARKET", "GAMELINE"},
        "live_spread": {"LIVE_MICRO_MARKET", "GAMELINE"},
        "live_drive_points": {"LIVE_MICRO_MARKET"},
    }
    if norm_type not in allowed_types[canonical]:
        raise SchemaError(f"Market {canonical!r} is incompatible with market_type {norm_type!r}")
    return canonical


def calculate_expected_points(down: int, distance: int, yardline: int) -> float:
    """Analytical scrimmage Expected Points (EP) model for college football.

    Computes expected points added for the offense given the current down,
    distance, and distance to opponent goal line (yardline 0 to 100).

    - Yardline 1 (at opponent 1-yard line): ~+6.1 points
    - Yardline 20 (red zone 1st & 10): ~+4.4 points
    - Yardline 50 (midfield 1st & 10): ~+2.8 points
    - Yardline 75 (touchback at own 25, 1st & 10): ~+1.6 points
    - Yardline 99 (own 1-yard line, backed up): ~ -1.1 points
    """
    if not (1 <= down <= 4):
        raise SchemaError(f"Down must be in [1, 4], got {down}")
    if not (1 <= distance <= 99):
        raise SchemaError(f"Distance must be in [1, 99], got {distance}")
    if not (0 <= yardline <= 100):
        raise SchemaError(f"Yardline must be in [0, 100], got {yardline}")

    # Field position fraction x in [0.01, 0.99]
    # 0.01 at own goal line, 0.99 at opponent goal line
    x = (100 - yardline) / 100.0

    # Continuous polynomial for 1st & 10 baseline
    base_1st = -1.20 + 13.0 * x - 8.5 * (x**2) + 2.9 * (x**3)

    if down == 1:
        # Distance adjustment relative to standard 10 yards
        dist_adj = -0.05 * (distance - 10)
        return round(base_1st + dist_adj, 2)

    elif down == 2:
        # 2nd down baseline drops relative to 1st, plus distance penalty
        dist_adj = -0.08 * (distance - 5)
        return round(base_1st - 0.45 + dist_adj, 2)

    elif down == 3:
        # 3rd down value depends sharply on distance to convert
        dist_adj = -0.15 * (distance - 3)
        return round(base_1st - 1.20 + dist_adj, 2)

    else:
        # 4th down decision model: max among FG attempt, punt, and conversion
        ep_candidates: list[float] = []

        # 1. Field goal option if within 38 yards of goal (FG <= 55 yards)
        if yardline <= 38:
            fg_dist = yardline + 17
            fg_prob = max(0.15, min(0.98, 1.0 - 0.016 * max(0, fg_dist - 20)))
            ep_fg = 3.0 * fg_prob - 1.6 * (1.0 - fg_prob)
            ep_candidates.append(ep_fg)

        # 2. Punt option: net ~38 yards, opponent receives ball
        opp_yardline = 100 - max(10, yardline - 38)
        opp_x = (100 - opp_yardline) / 100.0
        opp_base_1st = -1.20 + 13.0 * opp_x - 8.5 * (opp_x**2) + 2.9 * (opp_x**3)
        ep_punt = -opp_base_1st - 0.35  # Opponent expected points inverted
        ep_candidates.append(ep_punt)

        # 3. Go-for-it option (especially viable for short yardage <= 3 yards)
        if distance <= 3:
            conv_prob = 0.68 - 0.10 * (distance - 1)
            ep_conv = base_1st + 0.20
            fail_opp_yardline = 100 - yardline
            fail_opp_x = (100 - fail_opp_yardline) / 100.0
            fail_opp_base = (
                -1.20 + 13.0 * fail_opp_x - 8.5 * (fail_opp_x**2) + 2.9 * (fail_opp_x**3)
            )
            ep_fail = -fail_opp_base
            ep_go = conv_prob * ep_conv + (1.0 - conv_prob) * ep_fail
            ep_candidates.append(ep_go)

        return round(max(ep_candidates) if ep_candidates else ep_punt, 2)


def calculate_win_probability(
    state: GameState,
    pregame_home_margin: float = 0.0,
    team_id: str | None = None,
) -> float:
    """Analytical, state-based in-game win probability model.

    Combines:
    1. Pre-game expected margin (decaying with remaining game time).
    2. Current score differential.
    3. Possession and scrimmage field position (via Expected Points model).
    4. In-game rolling momentum differential.
    5. Clock management, timeouts, and kneel-down scenarios.

    Returns probability in [0.0, 1.0] that `team_id` (or home_team) wins.
    """
    target_team = team_id or state.home_team_id
    if target_team not in (state.home_team_id, state.away_team_id):
        raise SchemaError(f"Target team {target_team!r} is not in this game")
    pregame_margin = _finite_float(pregame_home_margin, name="pregame_home_margin")

    time_remaining = state.seconds_remaining_in_game
    if state.is_final:
        if state.home_score > state.away_score:
            return 1.0 if target_team == state.home_team_id else 0.0
        if state.home_score < state.away_score:
            return 0.0 if target_team == state.home_team_id else 1.0

    if state.is_overtime:
        raise SchemaError("Unresolved overtime requires possession-series state or is_final=True")
    if time_remaining <= 0:
        raise SchemaError("Expired regulation requires is_final=True")

    # 3. Active regulation play
    time_fraction = time_remaining / float(REGULATION_SECONDS)
    score_diff_home = state.home_score - state.away_score

    # Scrimmage possession value
    scrimmage_ep = calculate_expected_points(state.down, state.distance, state.yardline)
    if state.possession_team_id == state.home_team_id:
        possession_ep_home = scrimmage_ep
    else:
        possession_ep_home = -scrimmage_ep

    # Adjusted score differential including current drive expected points
    effective_lead_home = score_diff_home + possession_ep_home

    # Momentum advantage
    momentum_diff = state.home_momentum.momentum_factor - state.away_momentum.momentum_factor

    # Pre-game expectation decays linearly with remaining time fraction
    # Momentum adds subtle drift damped by sqrt(time_fraction)
    remaining_expected_margin = pregame_margin * time_fraction + 0.75 * momentum_diff * math.sqrt(
        time_fraction
    )

    expected_final_margin = effective_lead_home + remaining_expected_margin

    # Kneel-down / Clock-kill scenarios
    # When a team has the lead and possession, and remaining seconds are within
    # what the offense can bleed via kneeldowns given opponent timeouts.
    if state.possession_team_id == state.home_team_id and score_diff_home > 0:
        plays_to_downs = 4 - state.down
        bleedable_seconds = max(0, plays_to_downs * 40 - state.away_timeouts * 35)
        if time_remaining <= bleedable_seconds:
            home_prob = 0.9999
            return home_prob if target_team == state.home_team_id else round(1.0 - home_prob, 4)

    if state.possession_team_id == state.away_team_id and score_diff_home < 0:
        plays_to_downs = 4 - state.down
        bleedable_seconds = max(0, plays_to_downs * 40 - state.home_timeouts * 35)
        if time_remaining <= bleedable_seconds:
            home_prob = 0.0001
            return home_prob if target_team == state.home_team_id else round(1.0 - home_prob, 4)

    # Standard deviation of remaining score differential
    # Scales proportionally with sqrt(time_fraction)
    sigma_remaining = CFB_GAME_MARGIN_SIGMA * math.sqrt(time_fraction)
    # Impose a small positive floor to avoid division by zero near clock expiration
    sigma_effective = max(0.60, sigma_remaining)

    # Probit / Standard Normal Cumulative Distribution Function
    z = expected_final_margin / sigma_effective
    home_win_prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # Bound probabilities strictly inside (0, 1) while clock is ticking
    home_win_prob = max(0.0001, min(0.9999, home_win_prob))

    if target_team == state.home_team_id:
        return round(home_win_prob, 4)
    return round(1.0 - home_win_prob, 4)


@dataclass(frozen=True)
class DriveOutcomeDistribution:
    """Probability distribution over terminal drive outcomes."""

    touchdown: float
    field_goal: float
    punt: float
    turnover_downs: float
    safety: float
    expected_points: float
    model_version: str = MODEL_VERSION
    model_status: str = MODEL_STATUS
    is_actionable: bool = False

    @property
    def turnover(self) -> float:
        return self.turnover_downs

    @property
    def downs(self) -> float:
        return self.turnover_downs

    @property
    def turnover_or_downs(self) -> float:
        return self.turnover_downs

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            DriveOutcome.TOUCHDOWN.value: self.touchdown,
            DriveOutcome.FIELD_GOAL.value: self.field_goal,
            DriveOutcome.PUNT.value: self.punt,
            DriveOutcome.TURNOVER.value: self.turnover_downs,
            DriveOutcome.SAFETY.value: self.safety,
        }


def estimate_next_drive_outcomes(
    start_yardline: int,
    *,
    off_efficiency: float = 0.0,
    def_efficiency: float = 0.0,
    momentum: float = 0.0,
) -> DriveOutcomeDistribution:
    """Estimate probability distribution across terminal drive outcomes.

    Outcomes:
    - Touchdown (TD)
    - Field Goal (FG)
    - Punt (PUNT)
    - Turnover / Downs (TURNOVER)
    - Safety (SAFETY)

    Conditioned on:
    - Starting field position (distance to opponent goal line 1 to 99).
    - Offensive efficiency vs. defensive efficiency differential.
    - Live rolling drive momentum.
    """
    if not (1 <= start_yardline <= 99):
        raise SchemaError(f"Starting yardline must be in [1, 99], got {start_yardline}")

    offense = _finite_float(off_efficiency, name="off_efficiency")
    defense = _finite_float(def_efficiency, name="def_efficiency")
    momentum_value = _finite_float(momentum, name="momentum")

    # Field progression fraction x in [0.01, 0.99]
    # x = 0.01 (own 1-yard line); x = 0.99 (opponent 1-yard line)
    x = (100 - start_yardline) / 100.0

    net_eff = offense - defense
    beta = 1.4 * net_eff + 0.7 * momentum_value

    # Multinomial logistic logits
    # Touchdown: increases sharply as starting position approaches goal line
    l_td = -1.15 + 3.20 * x + 0.75 * beta

    # Field Goal: peaks in scoring range (opponent 15-35), drops at goal line
    l_fg = -1.75 + 2.85 * x * (1.0 - 0.35 * x) + 0.35 * beta

    # Punt: high when backed up in own territory, approaches zero past midfield
    l_punt = 1.10 - 4.20 * x - 0.75 * beta

    # Turnover / Downs: relatively uniform risk across field, slight increase when backed up
    l_turnover = -0.65 - 0.40 * x - 0.40 * beta

    # Safety: negligible across most of field, elevated when backed up inside own 12
    backed_up_severity = max(0.0, min(1.0, (start_yardline - 88) / 11.0))
    l_safety = -6.50 + 4.30 * backed_up_severity - 0.30 * beta

    logits = [l_td, l_fg, l_punt, l_turnover, l_safety]
    max_l = max(logits)

    # Numerically stable Softmax
    exp_vals = [math.exp(logit - max_l) for logit in logits]
    sum_exp = sum(exp_vals)
    probs = [v / sum_exp for v in exp_vals]

    p_td, p_fg, p_punt, p_to, p_safety = probs

    # Expected Points from drive: TD (~6.95 pts with PAT), FG (3.0 pts), Safety (-2.0 pts)
    exp_pts = 6.95 * p_td + 3.0 * p_fg - 2.0 * p_safety

    return DriveOutcomeDistribution(
        touchdown=round(p_td, 4),
        field_goal=round(p_fg, 4),
        punt=round(p_punt, 4),
        turnover_downs=round(p_to, 4),
        safety=round(p_safety, 4),
        expected_points=round(exp_pts, 3),
    )


@dataclass(frozen=True)
class LiveTeamTotal:
    """Live projected team total points and probability of hitting lines."""

    team_id: str
    current_score: int
    remaining_possessions: float
    remaining_expected_points: float
    projected_total: float
    prob_over: dict[float, float] = field(default_factory=dict)
    prob_under: dict[float, float] = field(default_factory=dict)
    prob_push: dict[float, float] = field(default_factory=dict)
    model_version: str = MODEL_VERSION
    model_status: str = MODEL_STATUS
    is_actionable: bool = False


@dataclass(frozen=True)
class LiveTotalsProjection:
    """Comprehensive live totals projection for both teams and game total."""

    home: LiveTeamTotal
    away: LiveTeamTotal
    projected_game_total: float
    blended_tempo_poss_per_60: float
    remaining_game_seconds: int
    prob_game_over: dict[float, float] = field(default_factory=dict)
    prob_game_under: dict[float, float] = field(default_factory=dict)
    prob_game_push: dict[float, float] = field(default_factory=dict)
    model_version: str = MODEL_VERSION
    model_status: str = MODEL_STATUS
    is_actionable: bool = False


def project_live_team_totals(
    state: GameState,
    *,
    pregame_pace: float = 12.0,
    off_eff_home: float = 0.0,
    off_eff_away: float = 0.0,
    def_eff_home: float = 0.0,
    def_eff_away: float = 0.0,
    observed_possessions_home: int | None = None,
    observed_possessions_away: int | None = None,
    lines_home: tuple[float, ...] = (),
    lines_away: tuple[float, ...] = (),
    lines_game: tuple[float, ...] = (),
) -> LiveTotalsProjection:
    """Project live in-game team totals and full-game total.

    Dynamically projects remaining possessions based on elapsed game time and
    observed tempo vs pregame pace expectations, and computes remaining expected
    points conditioned on efficiency and current scrimmage field position.
    """
    pace = _finite_float(pregame_pace, name="pregame_pace")
    if pace <= 0.0:
        raise SchemaError("pregame_pace must be positive")
    off_home = _finite_float(off_eff_home, name="off_eff_home")
    off_away = _finite_float(off_eff_away, name="off_eff_away")
    def_home = _finite_float(def_eff_home, name="def_eff_home")
    def_away = _finite_float(def_eff_away, name="def_eff_away")
    for observed, name in (
        (observed_possessions_home, "observed_possessions_home"),
        (observed_possessions_away, "observed_possessions_away"),
    ):
        if observed is not None and (
            not isinstance(observed, int) or isinstance(observed, bool) or observed < 0
        ):
            raise SchemaError(f"{name} must be a non-negative integer")
    lines_home = _validated_lines(lines_home, name="lines_home")
    lines_away = _validated_lines(lines_away, name="lines_away")
    lines_game = _validated_lines(lines_game, name="lines_game")
    time_remaining = state.seconds_remaining_in_game
    time_elapsed = state.elapsed_game_seconds

    def _evaluate_lines(
        score: int, proj: float, rem_poss: float, lines: tuple[float, ...]
    ) -> tuple[dict[float, float], dict[float, float], dict[float, float]]:
        p_over: dict[float, float] = {}
        p_under: dict[float, float] = {}
        p_push: dict[float, float] = {}
        if rem_poss <= 0.05:
            for line in lines:
                p_over[line] = 1.0 if score > line else 0.0
                p_under[line] = 1.0 if score < line else 0.0
                p_push[line] = 1.0 if score == line else 0.0
            return p_over, p_under, p_push

        sigma = max(0.75, math.sqrt(rem_poss) * 2.65)
        for line in lines:
            z = (line - proj) / sigma
            prob_u = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            p_under[line] = round(max(0.001, min(0.999, prob_u)), 4)
            p_over[line] = round(1.0 - p_under[line], 4)
            p_push[line] = 0.0
        return p_over, p_under, p_push

    if state.is_final:
        home_tt = LiveTeamTotal(
            team_id=state.home_team_id,
            current_score=state.home_score,
            remaining_possessions=0.0,
            remaining_expected_points=0.0,
            projected_total=float(state.home_score),
            prob_over={line: 1.0 if state.home_score > line else 0.0 for line in lines_home},
            prob_under={line: 1.0 if state.home_score < line else 0.0 for line in lines_home},
            prob_push={line: 1.0 if state.home_score == line else 0.0 for line in lines_home},
        )
        away_tt = LiveTeamTotal(
            team_id=state.away_team_id,
            current_score=state.away_score,
            remaining_possessions=0.0,
            remaining_expected_points=0.0,
            projected_total=float(state.away_score),
            prob_over={line: 1.0 if state.away_score > line else 0.0 for line in lines_away},
            prob_under={line: 1.0 if state.away_score < line else 0.0 for line in lines_away},
            prob_push={line: 1.0 if state.away_score == line else 0.0 for line in lines_away},
        )
        game_score = float(state.home_score + state.away_score)
        return LiveTotalsProjection(
            home=home_tt,
            away=away_tt,
            projected_game_total=game_score,
            blended_tempo_poss_per_60=0.0,
            remaining_game_seconds=0,
            prob_game_over={line: 1.0 if game_score > line else 0.0 for line in lines_game},
            prob_game_under={line: 1.0 if game_score < line else 0.0 for line in lines_game},
            prob_game_push={line: 1.0 if game_score == line else 0.0 for line in lines_game},
        )

    if state.is_overtime:
        raise SchemaError("Unresolved overtime requires possession-series state or is_final=True")
    if time_remaining <= 0:
        raise SchemaError("Expired regulation requires is_final=True")

    # 2. In-game dynamic tempo estimation
    total_obs_poss = (observed_possessions_home or 0) + (observed_possessions_away or 0)
    if total_obs_poss > 0 and time_elapsed >= 300:
        # Possessions per team per 60 minutes
        obs_pace_per_team = (total_obs_poss / 2.0) * (float(REGULATION_SECONDS) / time_elapsed)
        obs_weight = min(0.80, time_elapsed / 2700.0)
        blended_pace = obs_weight * obs_pace_per_team + (1.0 - obs_weight) * pace
    else:
        blended_pace = pace

    # Remaining full possessions per team across remaining game time
    fraction_remaining = time_remaining / float(REGULATION_SECONDS)
    future_poss_per_team = blended_pace * fraction_remaining

    # Current possession credit: team on offense has the current drive in progress
    # plus half of future possessions
    current_scrimmage_ep = max(
        0.0, calculate_expected_points(state.down, state.distance, state.yardline)
    )

    # Expected points per future possession (average FBS ~2.15 pts/poss)
    net_eff_home = off_home - def_away
    net_eff_away = off_away - def_home

    pts_per_poss_home = max(
        0.50, 2.15 + 1.20 * net_eff_home + 0.35 * state.home_momentum.momentum_factor
    )
    pts_per_poss_away = max(
        0.50, 2.15 + 1.20 * net_eff_away + 0.35 * state.away_momentum.momentum_factor
    )

    if state.possession_team_id == state.home_team_id:
        rem_poss_home = 1.0 + future_poss_per_team
        rem_poss_away = future_poss_per_team
        rem_pts_home = current_scrimmage_ep + future_poss_per_team * pts_per_poss_home
        rem_pts_away = future_poss_per_team * pts_per_poss_away
    else:
        rem_poss_home = future_poss_per_team
        rem_poss_away = 1.0 + future_poss_per_team
        rem_pts_home = future_poss_per_team * pts_per_poss_home
        rem_pts_away = current_scrimmage_ep + future_poss_per_team * pts_per_poss_away

    proj_home = round(state.home_score + rem_pts_home, 2)
    proj_away = round(state.away_score + rem_pts_away, 2)
    proj_game = round(proj_home + proj_away, 2)

    p_over_home, p_under_home, p_push_home = _evaluate_lines(
        state.home_score, proj_home, rem_poss_home, lines_home
    )
    p_over_away, p_under_away, p_push_away = _evaluate_lines(
        state.away_score, proj_away, rem_poss_away, lines_away
    )
    p_over_game, p_under_game, p_push_game = _evaluate_lines(
        state.home_score + state.away_score,
        proj_game,
        rem_poss_home + rem_poss_away,
        lines_game,
    )

    home_tt = LiveTeamTotal(
        team_id=state.home_team_id,
        current_score=state.home_score,
        remaining_possessions=round(rem_poss_home, 2),
        remaining_expected_points=round(rem_pts_home, 2),
        projected_total=proj_home,
        prob_over=p_over_home,
        prob_under=p_under_home,
        prob_push=p_push_home,
    )

    away_tt = LiveTeamTotal(
        team_id=state.away_team_id,
        current_score=state.away_score,
        remaining_possessions=round(rem_poss_away, 2),
        remaining_expected_points=round(rem_pts_away, 2),
        projected_total=proj_away,
        prob_over=p_over_away,
        prob_under=p_under_away,
        prob_push=p_push_away,
    )

    return LiveTotalsProjection(
        home=home_tt,
        away=away_tt,
        projected_game_total=proj_game,
        blended_tempo_poss_per_60=round(blended_pace * 2.0, 2),
        remaining_game_seconds=time_remaining,
        prob_game_over=p_over_game,
        prob_game_under=p_under_game,
        prob_game_push=p_push_game,
    )


def win_probability_to_american_odds(win_prob: float) -> int | None:
    """Convert fair win probability to fair American moneyline odds.

    Returns None for invalid probabilities outside (0.0, 1.0).
    """
    if not (
        isinstance(win_prob, (int, float)) and math.isfinite(win_prob) and 0.0 < win_prob < 1.0
    ):
        return None
    dec = 1.0 / win_prob
    return decimal_to_american(dec)


def project_live_spread(
    state: GameState,
    pregame_home_margin: float = 0.0,
    lines: tuple[float, ...] = (),
) -> dict[str, Any]:
    """Project live in-game home spread and probability of covering lines.

    Negative line means home favorite (e.g. -7.5).
    """
    margin = _finite_float(pregame_home_margin, name="pregame_home_margin")
    resolved_lines = tuple(_finite_float(line, name="spread line") for line in lines)
    if len(set(resolved_lines)) != len(resolved_lines):
        raise SchemaError("Spread lines cannot contain duplicates")
    time_remaining = state.seconds_remaining_in_game
    if state.is_overtime and not state.is_final:
        raise SchemaError(
            "Unresolved overtime or expired regulation requires richer state or is_final=True"
        )
    if time_remaining <= 0 and not state.is_final:
        raise SchemaError("Expired regulation requires is_final=True")
    if state.is_final:
        actual_margin = float(state.home_score - state.away_score)
        return {
            "projected_home_margin": actual_margin,
            "projected_home_spread": -actual_margin,
            "prob_cover": {
                line: 1.0 if actual_margin + line > 0 else 0.0 for line in resolved_lines
            },
            "prob_push": {
                line: 1.0 if actual_margin + line == 0 else 0.0 for line in resolved_lines
            },
            "model_version": MODEL_VERSION,
            "model_status": MODEL_STATUS,
            "is_actionable": False,
        }

    time_fraction = time_remaining / float(REGULATION_SECONDS)
    score_diff_home = state.home_score - state.away_score
    scrimmage_ep = calculate_expected_points(state.down, state.distance, state.yardline)
    possession_ep_home = (
        scrimmage_ep if state.possession_team_id == state.home_team_id else -scrimmage_ep
    )
    effective_lead_home = score_diff_home + possession_ep_home

    momentum_diff = state.home_momentum.momentum_factor - state.away_momentum.momentum_factor
    remaining_expected_margin = margin * time_fraction + 0.75 * momentum_diff * math.sqrt(
        time_fraction
    )
    expected_final_margin = effective_lead_home + remaining_expected_margin

    sigma_remaining = max(0.60, CFB_GAME_MARGIN_SIGMA * math.sqrt(time_fraction))

    prob_cover: dict[float, float] = {}
    for line in resolved_lines:
        z = (expected_final_margin + line) / sigma_remaining
        p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        prob_cover[line] = round(max(0.0001, min(0.9999, p)), 4)

    return {
        "projected_home_margin": round(expected_final_margin, 2),
        "projected_home_spread": round(-expected_final_margin, 2),
        "prob_cover": prob_cover,
        "prob_push": {line: 0.0 for line in resolved_lines},
        "model_version": MODEL_VERSION,
        "model_status": MODEL_STATUS,
        "is_actionable": False,
    }


def estimate_drive_points_distribution(
    start_yardline: int,
    *,
    off_efficiency: float = 0.0,
    def_efficiency: float = 0.0,
    momentum: float = 0.0,
) -> dict[int, float]:
    """Probability distribution over discrete points scored on current/next drive.

    Outcomes are 7 (TD + PAT), 8 (TD + 2pt), 6 (missed PAT), 3 (FG),
    0 (no score), and -2 (safety).
    """
    outcomes = estimate_next_drive_outcomes(
        start_yardline,
        off_efficiency=off_efficiency,
        def_efficiency=def_efficiency,
        momentum=momentum,
    )
    p_7 = round(outcomes.touchdown * 0.92, 4)
    p_8 = round(outcomes.touchdown * 0.05, 4)
    p_6 = round(outcomes.touchdown * 0.03, 4)
    p_3 = outcomes.field_goal
    p_neg2 = outcomes.safety
    p_0 = round(max(0.0, 1.0 - (p_7 + p_8 + p_6 + p_3 + p_neg2)), 4)

    return {
        7: p_7,
        3: p_3,
        0: p_0,
        8: p_8,
        6: p_6,
        -2: p_neg2,
    }
