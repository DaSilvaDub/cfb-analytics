"""Model 3 Team Props Scanner (Strictly No Player Props).

Projects and evaluates team points, rushing yards, receiving yards, and offensive yards
against posted sportsbook lines:
- Strictly TEAM_PROP only: rejects player props with PlayerPropProhibitedError (subclass of SchemaError and ValueError).
- Calibrated probability with hard cap at 84% (PROB_CAP = 0.84).
- Blowout sit-QB filter: favorite receiving props dropped at spread <= -20.0 (SIT_QB_SPREAD = -20.0).
- Underdog receiving props dropped at spread >= +28.0 (DOG_REC_BLOWOUT = 28.0).
"""

from __future__ import annotations

import math

from cfb_analytics.errors import SchemaError
from cfb_analytics.models.team_props import (
    MARKET_ALIASES,
    SUPPORTED_TEAM_PROPS,
)
from cfb_analytics.scanner.models import (
    SHADOW_MODE_DISCLAIMER,
    MispricedOpportunity,
    PlayTier,
    QualificationStatus,
)
from cfb_analytics.scanner.scoring import calculate_play_score
from cfb_analytics.utils import american_to_decimal, decimal_to_american

# Thresholds and calibration parameters
PROB_CAP: float = 0.84
SIT_QB_SPREAD: float = -20.0
DOG_REC_BLOWOUT: float = 28.0
BLOWOUT_GAME_YARDS: float = 28.0

# Standard deviations for prop families
STD_TEAM_POINTS: float = 7.0
STD_TEAM_YARDS_POSTED: float = 35.0
STD_GAME_YARDS_POSTED: float = 50.0


class PlayerPropProhibitedError(SchemaError, ValueError):
    """Raised when a player prop is submitted to the Model 3 team props scanner."""

    pass


def norm_cdf(z: float) -> float:
    """Cumulative distribution function for standard normal distribution using stdlib math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def validate_team_prop(market: str, market_type: str = "TEAM_PROP") -> str:
    """Validate that market is strictly a TEAM_PROP and not a player prop.

    Raises:
        PlayerPropProhibitedError: If market_type != 'TEAM_PROP' or player prop tokens are found.
    """
    mt = market_type.strip().upper()
    if mt != "TEAM_PROP":
        raise PlayerPropProhibitedError(
            f"Model 3 strictly requires market_type='TEAM_PROP', got {market_type!r}"
        )

    norm = market.strip().lower().replace("-", "_").replace(" ", "_")

    # Explicit player prop keywords and prefixes
    player_tokens = ("player", "passer", "rusher", "receiver", "touchdown", "anytime", "first_td")
    if (
        any(token in norm for token in player_tokens)
        or norm.endswith("_player")
        or norm.startswith("pass_")
        or norm.startswith("rec_")
    ):
        raise PlayerPropProhibitedError(
            f"Player props are strictly prohibited in Model 3; received {market!r}"
        )

    canonical = MARKET_ALIASES.get(norm, norm)
    if canonical not in SUPPORTED_TEAM_PROPS:
        raise PlayerPropProhibitedError(
            f"Unsupported team prop market {market!r}; supported: {SUPPORTED_TEAM_PROPS}"
        )
    return canonical


def is_sit_qb_candidate(
    market: str,
    spread: float,
    *,
    is_favorite: bool = True,
) -> bool:
    """Determine whether a receiving prop is disqualified by the blowout sit-QB filter.

    Invariants:
    - Favorite receiving props are dropped when spread <= -20.0 (sit-QB risk).
    - Underdog receiving props are dropped when spread >= +28.0 (blowout game state).
    """
    norm = market.strip().lower().replace("-", "_").replace(" ", "_")
    canonical = MARKET_ALIASES.get(norm, norm)
    if canonical != "team_receiving_yards":
        return False

    if is_favorite and spread <= SIT_QB_SPREAD:
        return True
    if not is_favorite and spread >= DOG_REC_BLOWOUT:
        return True
    return False


def calibrated_prop_prob(
    canonical_market: str,
    projected: float,
    line: float,
    side: str = "OVER",
    *,
    posted: bool = True,
) -> float:
    """Calculate calibrated cover probability with a hard cap at 84% (PROB_CAP = 0.84).

    Residual standard deviations:
    - Points: 7.0
    - Yards (posted): 35.0
    - Yards (inferred): max(55.0, 0.22 * line)
    """
    if canonical_market == "team_total_points":
        std = STD_TEAM_POINTS
    elif canonical_market in ("team_rushing_yards", "team_receiving_yards"):
        std = STD_TEAM_YARDS_POSTED if posted else max(55.0, 0.22 * line)
    else:  # team_offensive_yards or game yards
        std = STD_GAME_YARDS_POSTED if posted else max(70.0, 0.16 * line)

    side_clean = side.strip().upper()
    if side_clean == "OVER":
        z = (projected - line) / std
    elif side_clean == "UNDER":
        z = (line - projected) / std
    else:
        raise ValueError(f"side must be 'OVER' or 'UNDER', got {side!r}")

    raw_prob = norm_cdf(z)
    # Apply hard cap at 84.0%
    capped_prob = min(PROB_CAP, max(0.01, raw_prob))
    return round(capped_prob, 4)


def calculate_prop_edge(projected: float, line: float, side: str = "OVER") -> float:
    """Calculate quantitative production edge in points or yards."""
    side_clean = side.strip().upper()
    if side_clean == "OVER":
        return round(projected - line, 2)
    elif side_clean == "UNDER":
        return round(line - projected, 2)
    else:
        raise ValueError(f"side must be 'OVER' or 'UNDER', got {side!r}")


def calculate_expected_value(win_prob: float, decimal_price: float) -> float:
    """Calculate expected value (EV) per unit staked: EV = win_prob * decimal_price - 1.0."""
    return round(win_prob * decimal_price - 1.0, 4)


def evaluate_team_prop_candidate(
    game_id: str,
    market: str,
    side: str,
    line: float,
    projected_value: float,
    posted_price_american: int,
    consensus_fair_prob: float,
    *,
    market_type: str = "TEAM_PROP",
    spread: float | None = None,
    is_favorite: bool = True,
    drop_sit_qb: bool = True,
    posted: bool = True,
    method_spread: float = 0.0,
    n_books: int = 3,
    historical_hit_rate: float | None = None,
    sample_size: int = 20,
    confirming_insights_count: int = 0,
    open_line: float | None = None,
    current_line: float | None = None,
    rlm_flag: bool = False,
    best_book: str | None = None,
    game_label: str = "",
    kickoff_et: str = "",
    flags: tuple[str, ...] = (),
    data_quality_score: float = 100.0,
) -> MispricedOpportunity | None:
    """Evaluate a team prop candidate and return an immutable MispricedOpportunity.

    Rejects player props with PlayerPropProhibitedError.
    Filters out receiving props if sit-QB blowout invariant is triggered.
    """
    canonical_market = validate_team_prop(market, market_type=market_type)
    side_clean = side.strip().upper()
    if side_clean not in ("OVER", "UNDER"):
        raise ValueError(f"side must be 'OVER' or 'UNDER', got {side!r}")

    # Blowout sit-QB filter
    if spread is not None and is_sit_qb_candidate(canonical_market, spread, is_favorite=is_favorite):
        if drop_sit_qb:
            return None
        flags = tuple(list(flags) + ["sit_qb_blowout_risk"])

    model_p = calibrated_prop_prob(
        canonical_market,
        projected_value,
        line,
        side=side_clean,
        posted=posted,
    )
    edge_units = calculate_prop_edge(projected_value, line, side=side_clean)
    prob_edge = round(model_p - consensus_fair_prob, 4)

    decimal_price = american_to_decimal(posted_price_american)
    if decimal_price is None:
        decimal_price = 1.0

    ev = calculate_expected_value(model_p, decimal_price)

    if 0.0 < consensus_fair_prob < 1.0:
        fair_dec = 1.0 / consensus_fair_prob
        fair_american = decimal_to_american(fair_dec) or 0
    else:
        fair_american = 0

    play_score_obj = calculate_play_score(
        market_type=canonical_market,
        edge_metric=edge_units,
        model_prob=model_p,
        posted_price_american=posted_price_american,
        side=side_clean,
        historical_hit_rate=historical_hit_rate,
        sample_size=sample_size,
        confirming_insights_count=confirming_insights_count,
        open_line=open_line,
        current_line=current_line,
        rlm_flag=rlm_flag,
        ev=ev,
        n_books=n_books,
        consensus_fair_prob=consensus_fair_prob,
        data_quality_score=data_quality_score,
        flags=flags,
    )

    tier_str = (
        play_score_obj.tier.value
        if isinstance(play_score_obj.tier, PlayTier)
        else str(play_score_obj.tier)
    )
    status_str = (
        play_score_obj.qualification_status.value
        if isinstance(play_score_obj.qualification_status, QualificationStatus)
        else str(play_score_obj.qualification_status)
    )

    return MispricedOpportunity(
        game_id=game_id,
        market_type="TEAM_PROP",
        market=canonical_market,
        side=side_clean,
        line=line,
        posted_price_american=posted_price_american,
        posted_price_decimal=round(decimal_price, 4),
        consensus_fair_prob=round(consensus_fair_prob, 4),
        consensus_fair_price_american=fair_american,
        model_projected_line=round(projected_value, 2),
        model_prob=model_p,
        edge_pct=prob_edge,
        ev=ev,
        method_spread=round(method_spread, 4),
        n_books=n_books,
        play_score=play_score_obj.total_score,
        play_tier=tier_str,
        qual_status=status_str,
        best_book=best_book,
        game_label=game_label,
        kickoff_et=kickoff_et,
        flags=flags,
        rejection_reasons=play_score_obj.rejection_reasons,
        shadow_mode_disclaimer=SHADOW_MODE_DISCLAIMER,
    )
