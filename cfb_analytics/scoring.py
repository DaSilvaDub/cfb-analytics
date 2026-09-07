"""Candidate Play Scoring and Team Props Evaluation Pipeline.

Connects Model 3 (TeamPropsInputs and project_team_production) with devigged
consensus fair odds from models.devig / features.market to score wagering
opportunities on a 0-to-100 Play Score scale.

Enforces strict filtering of player props (only team props like
team_offensive_yards, team_rushing_yards, team_receiving_yards, and
team_total_points are supported).
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cfb_analytics.errors import DevigError, SchemaError
from cfb_analytics.models.devig import (
    consensus_probabilities,
    devig,
)
from cfb_analytics.models.team_props import (
    MARKET_ALIASES,
    SUPPORTED_TEAM_PROPS,
    TeamPropsInputs,
    TeamPropsProjection,
    default_team_props_inputs,
    is_player_prop,
    is_supported_team_prop,
    normalize_team_prop_market,
    project_team_production,
)
from cfb_analytics.utils import american_to_decimal, implied_probability, to_utc_iso


@dataclass(frozen=True)
class TeamPropCandidate:
    """A team prop betting opportunity to be evaluated."""

    game_id: str
    team_id: str
    market: str
    side: str
    line: float
    market_type: str = "TEAM_PROP"
    as_of_utc: str | None = None
    consensus_price: int | None = None
    opposite_consensus_price: int | None = None
    best_price: int | None = None
    best_book: str | None = None
    n_books: int = 0
    hold: float | None = None
    devigged_fair_prob: float | None = None
    historical_hit_rate: float | None = None
    sample_size: int = 0
    confirming_insights_count: int = 0
    open_line: float | None = None
    current_line: float | None = None
    rlm_flag: bool = False
    book_quotes: Mapping[str, Sequence[float | int]] | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateScore:
    """Evaluated and scored betting candidate."""

    game_id: str
    team_id: str
    market: str
    side: str
    line: float
    as_of_utc: str | None
    projected_value: float
    consensus_price: int | None
    best_price: int | None
    best_book: str | None
    devigged_fair_prob: float | None
    model_prob: float
    n_books: int
    edge: float
    expected_value: float | None
    data_quality_score: float
    hit_rate_score: float
    edge_score: float
    movement_score: float
    price_value_score: float
    sample_size_score: float
    confirming_score: float
    play_score: float
    tier: str
    qualification_status: str
    rejection_reason: str | None = None

    @property
    def is_actionable(self) -> bool:
        """Play Score >= 70 threshold, positive edge, and pricing required for actionable plays."""
        return self.qualification_status == "QUALIFIED"


def resolve_devigged_fair_prob(candidate: TeamPropCandidate, side_upper: str) -> float | None:
    """Resolve devigged consensus fair probability using cfb_analytics.models.devig."""
    if candidate.n_books < 3:
        return None

    if candidate.devigged_fair_prob is not None and 0.0 < candidate.devigged_fair_prob < 1.0:
        return candidate.devigged_fair_prob

    # 1. Multi-book quotes
    if candidate.book_quotes:
        try:
            prob_quotes: dict[str, list[float]] = {}
            for book, quotes in candidate.book_quotes.items():
                converted: list[float] = []
                for q in quotes:
                    if isinstance(q, int) or (isinstance(q, float) and (q > 1.0 or q < 0.0)):
                        p = implied_probability(q)
                        if p is not None:
                            converted.append(p)
                    else:
                        converted.append(float(q))
                if len(converted) >= 2:
                    prob_quotes[book] = converted
            if len(prob_quotes) >= 3:
                res = consensus_probabilities(prob_quotes, method="shin")
                idx = 0 if side_upper == "OVER" else 1
                if idx < len(res):
                    return round(res[idx], 4)
        except DevigError:
            pass

    # 2. Two-way consensus prices
    if candidate.consensus_price is not None and candidate.opposite_consensus_price is not None:
        p_side = implied_probability(candidate.consensus_price)
        p_opp = implied_probability(candidate.opposite_consensus_price)
        if p_side is not None and p_opp is not None:
            try:
                fair = devig([p_side, p_opp], method="shin")
                return round(fair[0], 4)
            except DevigError:
                pass

    return None


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution using stdlib math."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def calculate_model_probability(
    canonical_market: str, side: str, line: float, projected_val: float
) -> float:
    """Derive model win probability from projected team production vs market line."""
    if canonical_market == "team_total_points":
        std_dev = 7.0
    elif canonical_market in ("team_rushing_yards", "team_receiving_yards"):
        std_dev = 35.0
    else:  # team_offensive_yards
        std_dev = 50.0

    z = (projected_val - line) / std_dev
    p_over = norm_cdf(z)
    side_upper = side.strip().upper()
    prob = p_over if side_upper == "OVER" else (1.0 - p_over)
    return round(max(0.01, min(0.99, prob)), 4)


def score_hit_rate(hit_rate: float | None) -> float:
    """Component 1: Outlier Historical Hit Rate (0 to 25)."""
    if hit_rate is None:
        return 0.0
    rate = hit_rate / 100.0 if hit_rate > 1.0 else hit_rate
    if rate >= 0.80:
        return 25.0
    elif rate >= 0.75:
        return 22.0
    elif rate >= 0.70:
        return 19.0
    elif rate >= 0.65:
        return 15.0
    elif rate >= 0.60:
        return 10.0
    elif rate >= 0.55:
        return 5.0
    return 0.0


def score_edge(canonical_market: str, edge: float) -> float:
    """Component 2: Estimated Quantitative Edge (0 to 20)."""
    norm = canonical_market.strip().lower().replace("-", "_").replace(" ", "_")
    canonical = MARKET_ALIASES.get(norm, norm)
    if canonical == "team_total_points":
        if edge >= 7.0:
            return 20.0
        elif edge >= 4.5:
            return 16.0
        elif edge >= 2.5:
            return 12.0
        elif edge >= 1.0:
            return 7.0
        elif edge > 0.0:
            return 4.0
        return 0.0
    else:
        # Yardage markets
        if edge >= 35.0:
            return 20.0
        elif edge >= 20.0:
            return 16.0
        elif edge >= 10.0:
            return 12.0
        elif edge >= 3.0:
            return 7.0
        elif edge > 0.0:
            return 4.0
        return 0.0


def score_line_movement(
    side: str,
    open_line: float | None,
    current_line: float | None,
    rlm_flag: bool = False,
) -> float:
    """Component 3: Line Movement Confirmation (0 to 15)."""
    if rlm_flag:
        return 15.0
    if open_line is None or current_line is None:
        return 4.0  # Neutral baseline
    diff = current_line - open_line
    side_upper = side.strip().upper()
    if side_upper == "OVER":
        if diff < -1.5:
            return 14.0
        elif diff < 0:
            return 10.0
        elif diff == 0:
            return 4.0
        elif diff <= 1.5:
            return 2.0
        return 1.0
    elif side_upper == "UNDER":
        if diff > 1.5:
            return 14.0
        elif diff > 0:
            return 10.0
        elif diff == 0:
            return 4.0
        elif diff >= -1.5:
            return 2.0
        return 1.0
    return 4.0


def score_price_value(
    best_price: int | None,
    consensus_price: int | None,
    model_prob: float | None,
) -> float:
    """Component 4: Price / Odds Value (0 to 15)."""
    actionable_price = best_price if best_price is not None else consensus_price
    if actionable_price is None or model_prob is None or not 0.0 < model_prob < 1.0:
        return 0.0
    best_implied = implied_probability(actionable_price)
    if best_implied is None:
        return 0.0
    price_edge = model_prob - best_implied
    if price_edge >= 0.05:
        return 15.0
    elif price_edge >= 0.03:
        return 12.0
    elif price_edge >= 0.01:
        return 9.0
    elif price_edge >= -0.02:
        return 6.0
    return 2.0


def calculate_expected_value(
    model_prob: float,
    best_price: int | None,
    consensus_price: int | None,
) -> float | None:
    """Return expected profit per unit staked at the actionable price."""
    actionable_price = best_price if best_price is not None else consensus_price
    decimal_price = american_to_decimal(actionable_price)
    if decimal_price is None or not 0.0 < model_prob < 1.0:
        return None
    return round(model_prob * decimal_price - 1.0, 4)


def score_sample_size(sample_size: int) -> float:
    """Component 5: Sample Size (0 to 10)."""
    if sample_size >= 50:
        return 10.0
    elif sample_size >= 30:
        return 8.0
    elif sample_size >= 20:
        return 6.0
    elif sample_size >= 10:
        return 4.0
    elif sample_size >= 5:
        return 2.0
    return 0.0


def score_confirming_insights(confirming_count: int) -> float:
    """Component 6: Multiple Confirming Insights (0 to 15)."""
    if confirming_count >= 4:
        return 15.0
    elif confirming_count == 3:
        return 12.0
    elif confirming_count == 2:
        return 8.0
    elif confirming_count == 1:
        return 4.0
    return 0.0


def assign_play_tier(play_score: float, edge: float | None = None) -> str:
    """Assign Play Tier according to 70+ qualification threshold.

    Per Minimum Qualification Gates, automatically downgrades to PASS when
    no measurable edge exists (edge <= 0).
    """
    if edge is not None and edge <= 0.0:
        return "PASS"
    if play_score >= 90.0:
        return "ELITE"
    elif play_score >= 80.0:
        return "STRONG"
    elif play_score >= 70.0:
        return "QUALIFIED"
    elif play_score >= 60.0:
        return "LEAN"
    return "PASS"


def extract_projected_value(canonical_market: str, projection: TeamPropsProjection) -> float:
    """Map canonical team prop market to its projected metric."""
    projections = {
        "team_offensive_yards": projection.projected_team_offensive_yards,
        "team_rushing_yards": projection.projected_team_rushing_yards,
        "team_receiving_yards": projection.projected_team_receiving_yards,
        "team_total_points": projection.projected_team_total_points,
    }
    try:
        return projections[canonical_market]
    except KeyError as exc:
        raise SchemaError(f"Unsupported market {canonical_market!r}") from exc


def score_candidate(
    candidate: TeamPropCandidate,
    inputs: TeamPropsInputs,
    *,
    strict: bool = True,
) -> CandidateScore:
    """Score an individual team prop wagering candidate against Model 3 inputs."""
    # 1. Strict filtering by the authoritative market family. Names alone are
    # ambiguous: RUSHING_YARDS may describe either a team or a player.
    if candidate.market_type.strip().upper() != "TEAM_PROP":
        if strict:
            raise SchemaError(
                f"Model 3 requires market_type='TEAM_PROP'; received {candidate.market_type!r}"
            )
        return CandidateScore(
            game_id=candidate.game_id,
            team_id=candidate.team_id,
            market=candidate.market,
            side=candidate.side,
            line=candidate.line,
            as_of_utc=candidate.as_of_utc,
            projected_value=0.0,
            consensus_price=candidate.consensus_price,
            best_price=candidate.best_price,
            best_book=candidate.best_book,
            devigged_fair_prob=candidate.devigged_fair_prob,
            model_prob=0.0,
            n_books=candidate.n_books,
            edge=0.0,
            expected_value=None,
            data_quality_score=inputs.data_quality_score,
            hit_rate_score=0.0,
            edge_score=0.0,
            movement_score=0.0,
            price_value_score=0.0,
            sample_size_score=0.0,
            confirming_score=0.0,
            play_score=0.0,
            tier="PASS",
            qualification_status="PASS",
            rejection_reason="player_prop_strictly_prohibited",
        )

    canonical_market = normalize_team_prop_market(
        candidate.market, market_type=candidate.market_type
    )

    # 2. Model 3 Production Projection
    projection = project_team_production("TEAM_PROP", inputs)
    projected_val = extract_projected_value(canonical_market, projection)

    # 3. Quantitative Edge and Win Probability
    side_upper = candidate.side.strip().upper()
    if side_upper == "OVER":
        edge = round(projected_val - candidate.line, 2)
    elif side_upper == "UNDER":
        edge = round(candidate.line - projected_val, 2)
    else:
        raise SchemaError(f"Invalid side {candidate.side!r}; expected OVER or UNDER")

    model_prob = calculate_model_probability(
        canonical_market, side_upper, candidate.line, projected_val
    )

    # 4. Resolve Devigged Consensus Fair Odds via models.devig
    devigged_prob = resolve_devigged_fair_prob(candidate, side_upper)

    # 5. Component Scoring
    s_hit = score_hit_rate(candidate.historical_hit_rate)
    s_edge = score_edge(canonical_market, edge)
    s_move = score_line_movement(
        side_upper, candidate.open_line, candidate.current_line, candidate.rlm_flag
    )
    s_price = score_price_value(candidate.best_price, candidate.consensus_price, model_prob)
    s_sample = score_sample_size(candidate.sample_size)
    s_conf = score_confirming_insights(candidate.confirming_insights_count)

    total_play_score = round(min(100.0, s_hit + s_edge + s_move + s_price + s_sample + s_conf), 1)
    expected_value = calculate_expected_value(
        model_prob, candidate.best_price, candidate.consensus_price
    )

    insufficient: list[str] = []
    if candidate.n_books < 3:
        insufficient.append("thin_market")
    if devigged_prob is None:
        insufficient.append("missing_fair_probability")
    if candidate.best_price is None and candidate.consensus_price is None:
        insufficient.append("price_unavailable")
    if inputs.data_quality_score < 75.0:
        insufficient.append("insufficient_model_data")
    if candidate.sample_size < 5:
        insufficient.append("sample_too_small")

    disqualifying_flags = (
        "arb",
        "invalid",
        "placeholder",
        "single_book",
        "thin_market",
        "incomplete",
    )
    if any(token in flag.lower() for flag in candidate.flags for token in disqualifying_flags):
        insufficient.append("disqualifying_market_flag")

    failed_value: list[str] = []
    if edge <= 0.0:
        failed_value.append("non_positive_line_edge")
    if expected_value is None or expected_value <= 0.0:
        failed_value.append("non_positive_expected_value")
    if total_play_score < 70.0:
        failed_value.append("play_score_below_threshold")

    if insufficient:
        qualification_status = "INSUFFICIENT_DATA"
    elif failed_value:
        qualification_status = "PASS"
    else:
        qualification_status = "QUALIFIED"

    rejection_reasons = [*insufficient, *failed_value]
    tier = (
        assign_play_tier(total_play_score, edge=edge)
        if qualification_status == "QUALIFIED"
        else "PASS"
    )

    return CandidateScore(
        game_id=candidate.game_id,
        team_id=candidate.team_id,
        market=canonical_market,
        side=side_upper,
        line=candidate.line,
        as_of_utc=candidate.as_of_utc,
        projected_value=projected_val,
        consensus_price=candidate.consensus_price,
        best_price=candidate.best_price,
        best_book=candidate.best_book,
        devigged_fair_prob=devigged_prob,
        model_prob=model_prob,
        n_books=candidate.n_books,
        edge=edge,
        expected_value=expected_value,
        data_quality_score=inputs.data_quality_score,
        hit_rate_score=s_hit,
        edge_score=s_edge,
        movement_score=s_move,
        price_value_score=s_price,
        sample_size_score=s_sample,
        confirming_score=s_conf,
        play_score=total_play_score,
        tier=tier,
        qualification_status=qualification_status,
        rejection_reason=",".join(rejection_reasons) or None,
    )


def filter_team_prop_candidates(
    candidates: Sequence[TeamPropCandidate | Mapping[str, Any]],
) -> list[TeamPropCandidate]:
    """Strictly filter out player props, returning only supported team props."""
    filtered: list[TeamPropCandidate] = []
    for item in candidates:
        if isinstance(item, TeamPropCandidate):
            cand = item
        else:
            cand = TeamPropCandidate(
                game_id=str(item["game_id"]),
                team_id=str(item["team_id"]),
                market=str(item["market"]),
                side=str(item["side"]),
                line=float(item["line"]),
                market_type=str(item.get("market_type", "")),
                as_of_utc=item.get("as_of_utc"),
                consensus_price=item.get("consensus_price"),
                opposite_consensus_price=item.get("opposite_consensus_price"),
                best_price=item.get("best_price"),
                best_book=item.get("best_book"),
                n_books=int(item.get("n_books", 0)),
                hold=item.get("hold"),
                devigged_fair_prob=item.get("devigged_fair_prob"),
                historical_hit_rate=item.get("historical_hit_rate"),
                sample_size=int(item.get("sample_size", 0)),
                confirming_insights_count=int(item.get("confirming_insights_count", 0)),
                open_line=item.get("open_line"),
                current_line=item.get("current_line"),
                rlm_flag=bool(item.get("rlm_flag", False)),
                book_quotes=item.get("book_quotes"),
                flags=tuple(item.get("flags", ())),
            )

        if not is_supported_team_prop(cand.market, market_type=cand.market_type):
            continue
        filtered.append(cand)

    return filtered


# Bounds on the opponent-defense multipliers. CFBD defensive success rate
# ALLOWED spans roughly 0.33 (elite) to 0.52 (worst) against a 0.42 FBS mean,
# and explosiveness allowed roughly 1.05 to 1.55 against a 1.25 mean, so the
# real ratio never leaves [0.79, 1.24]. Anything outside this band is a bad
# snapshot (a percentage stored where a fraction belongs, a one-game sample),
# not a real matchup, and must not scale a projection unbounded.
_OPP_ADJ_MIN = 0.80
_OPP_ADJ_MAX = 1.25


def _opponent_multiplier(allowed: Any, fbs_baseline: float) -> float:
    """Clamped ``allowed / fbs_baseline`` matchup multiplier, or a neutral 1.0.

    ``allowed`` is a CFBD ``side = 'def'`` value (what the defense gives up), so a
    higher number means a weaker defense and a multiplier above 1.0. Missing, NULL,
    non-numeric, non-positive or non-finite values return 1.0 -- "no signal, not a
    guess", the convention ``features/advanced_stats.py`` already uses.
    """
    if allowed is None or fbs_baseline <= 0:
        return 1.0
    try:
        value = float(allowed)
    except (ValueError, TypeError):
        return 1.0
    if value <= 0 or not math.isfinite(value):
        return 1.0
    return max(_OPP_ADJ_MIN, min(_OPP_ADJ_MAX, value / fbs_baseline))


def build_team_props_inputs_from_db(
    conn: sqlite3.Connection,
    team_id: str,
    *,
    opponent_team_id: str | None = None,
    season: int | None = None,
    as_of_utc: str | None = None,
) -> TeamPropsInputs:
    """Build team-specific TeamPropsInputs from stored statistics and ratings.

    Reads offensive efficiency metrics from team_season_advanced and player_game_passing
    when available, gracefully falling back to standard FBS averages.

    When ``opponent_team_id`` is given, the opponent's ``side = 'def'`` snapshot supplies
    two matchup multipliers against the FBS baselines: success rate allowed / 0.42 and
    explosiveness allowed / 1.25. Both are "allowed" quantities, so a higher value means
    a weaker defense (see ``features/advanced_stats.py`` for CFBD's asymmetric off/def
    sign convention). Both are clamped to ``[_OPP_ADJ_MIN, _OPP_ADJ_MAX]``, and a missing,
    NULL, non-finite or post-cutoff opponent snapshot yields a neutral 1.0 rather than a guess.

    The multipliers reach the projection ONLY through the production fields --
    ``yards_per_completion`` (explosiveness) and ``yards_before_contact`` (success rate).
    See the comment at the application site for why the rate fields stay opponent-neutral.
    """
    baseline = default_team_props_inputs()
    if not team_id:
        return baseline
    cutoff = to_utc_iso(as_of_utc)
    if cutoff is None:
        raise SchemaError("Team-prop inputs require a valid as_of_utc cutoff")

    query_adv = """
        SELECT season, week, success_rate, explosiveness, line_yards, plays, drives,
               passing_success_rate, rushing_success_rate
        FROM team_season_advanced
        WHERE team_id = ? AND side = 'off'
    """
    params_adv: list[Any] = [team_id]
    if season is not None:
        query_adv += " AND season = ?"
        params_adv.append(season)
    query_adv += " AND as_of_utc <= ?"
    params_adv.append(cutoff)
    query_adv += " ORDER BY season DESC, as_of_utc DESC LIMIT 1"

    row_adv = conn.execute(query_adv, params_adv).fetchone()

    # Resolve season for opponent defense and passing queries to prevent cross-season mixing
    adv_season = int(row_adv["season"]) if (row_adv and row_adv["season"] is not None) else None
    resolved_season = season if season is not None else adv_season

    # Query opponent defense for defensive success rate and explosiveness allowed
    sr_mult = 1.0
    exp_mult = 1.0
    opp_clean_id = opponent_team_id.strip() if opponent_team_id else None
    if opp_clean_id and opp_clean_id != team_id:
        query_opp = """
            SELECT success_rate, explosiveness
            FROM team_season_advanced
            WHERE team_id = ? AND side = 'def'
        """
        params_opp: list[Any] = [opp_clean_id]
        if resolved_season is not None:
            query_opp += " AND season = ?"
            params_opp.append(resolved_season)
        query_opp += " AND as_of_utc <= ?"
        params_opp.append(cutoff)
        query_opp += " ORDER BY season DESC, as_of_utc DESC LIMIT 1"

        row_opp_def = conn.execute(query_opp, params_opp).fetchone()
        if row_opp_def is not None:
            sr_mult = _opponent_multiplier(
                row_opp_def["success_rate"], baseline.offensive_success_rate
            )
            exp_mult = _opponent_multiplier(
                row_opp_def["explosiveness"], baseline.explosiveness
            )

    query_pass = """
        SELECT SUM(p.completions) as comp, SUM(p.attempts) as att,
               SUM(p.yards) as yds, COUNT(DISTINCT p.game_id) as games
        FROM player_game_passing AS p
        JOIN games AS g ON g.game_id = p.game_id
        WHERE p.team_id = ? AND g.kickoff_utc < ?
    """
    params_pass: list[Any] = [team_id, cutoff]
    if resolved_season is not None:
        query_pass += " AND p.season = ?"
        params_pass.append(resolved_season)

    row_pass = conn.execute(query_pass, params_pass).fetchone()

    raw_success_rate = (
        float(row_adv["success_rate"])
        if (row_adv and row_adv["success_rate"] is not None)
        else baseline.offensive_success_rate
    )
    raw_explosiveness = (
        float(row_adv["explosiveness"])
        if (row_adv and row_adv["explosiveness"] is not None)
        else baseline.explosiveness
    )
    raw_line_yards = (
        float(row_adv["line_yards"])
        if (row_adv and row_adv["line_yards"] is not None)
        else baseline.yards_before_contact
    )

    # Opponent strength is applied ONCE, and only through the production channel
    # (yards_per_completion / yards_before_contact below). project_team_points
    # already re-normalises offensive_success_rate by 0.42 and explosiveness by
    # 1.25 -- the very baselines these multipliers are built from -- so scaling
    # the rate fields here too would apply the matchup a second time and make
    # projected points move by m**2: a 1.20x defense produced a 1.39x points
    # swing and a 1.30x defense a 1.61x swing. The rate fields therefore stay
    # opponent-neutral (they describe this offense's own identity); the matchup
    # lives entirely in the projected yardage, which every supported market --
    # yards AND points -- cascades from.
    success_rate = round(max(0.0, min(1.0, raw_success_rate)), 4)
    explosiveness = round(max(0.0, raw_explosiveness), 4)
    line_yards = round(max(0.0, raw_line_yards * sr_mult), 2)

    comp = int(row_pass["comp"]) if (row_pass and row_pass["comp"]) else 0
    att = int(row_pass["att"]) if (row_pass and row_pass["att"]) else 0
    yds = int(row_pass["yds"]) if (row_pass and row_pass["yds"]) else 0
    n_games = int(row_pass["games"]) if (row_pass and row_pass["games"]) else 0

    plays = int(row_adv["plays"]) if (row_adv and row_adv["plays"]) else 0
    drives = int(row_adv["drives"]) if (row_adv and row_adv["drives"]) else 0
    snapshot_week = int(row_adv["week"]) if (row_adv and row_adv["week"]) else 0
    observed_games = n_games or snapshot_week

    if observed_games > 0 and drives > 0 and plays > 0:
        pace = round(max(50.0, min(95.0, plays / observed_games)), 1)
        expected_possessions = round(max(9.0, min(16.0, drives / observed_games)), 1)
    else:
        pace = baseline.pace
        expected_possessions = baseline.expected_possession_count

    if att > 0 and comp > 0:
        comp_prob = round(max(0.40, min(0.85, comp / att)), 3)
        raw_ypc = yds / comp
        ypc = round(max(6.0, min(20.0, raw_ypc * exp_mult)), 2)
        pass_per_game = att / max(1, n_games)
        expected_pass = round(max(18.0, min(55.0, pass_per_game)), 1)
        expected_rush = round(max(15.0, min(60.0, pace - expected_pass)), 1)
    else:
        comp_prob = baseline.completion_probability
        ypc = round(max(6.0, min(20.0, baseline.yards_per_completion * exp_mult)), 2)
        expected_pass = baseline.expected_pass_attempts
        expected_rush = baseline.expected_rushing_attempts

    data_quality_score = 0.0
    if row_adv is not None:
        data_quality_score += 50.0
    if att > 0 and comp > 0 and n_games > 0:
        data_quality_score += 50.0

    return TeamPropsInputs(
        pace=pace,
        expected_possession_count=expected_possessions,
        offensive_success_rate=success_rate,
        explosiveness=explosiveness,
        expected_pass_attempts=expected_pass,
        expected_rushing_attempts=expected_rush,
        completion_probability=comp_prob,
        yards_per_completion=ypc,
        yards_before_contact=line_yards,
        yards_after_contact=baseline.yards_after_contact,
        data_quality_score=data_quality_score,
    )


def load_team_props_inputs_for_slate(
    conn: sqlite3.Connection,
    slate_date: str,
    *,
    season: int | None = None,
    as_of_utc: str | None = None,
) -> dict[str, TeamPropsInputs]:
    """Load or derive TeamPropsInputs for all teams playing on a given slate."""
    games = conn.execute(
        "SELECT home_team_id, away_team_id, season FROM games WHERE football_date = ?",
        (slate_date,),
    ).fetchall()
    inputs_by_team: dict[str, TeamPropsInputs] = {}
    for g in games:
        game_season = season
        if game_season is None and g["season"] is not None:
            game_season = int(g["season"])
        home_id = str(g["home_team_id"]) if g["home_team_id"] else None
        away_id = str(g["away_team_id"]) if g["away_team_id"] else None
        if home_id and home_id not in inputs_by_team:
            inputs_by_team[home_id] = build_team_props_inputs_from_db(
                conn,
                home_id,
                opponent_team_id=away_id,
                season=game_season,
                as_of_utc=as_of_utc,
            )
        if away_id and away_id not in inputs_by_team:
            inputs_by_team[away_id] = build_team_props_inputs_from_db(
                conn,
                away_id,
                opponent_team_id=home_id,
                season=game_season,
                as_of_utc=as_of_utc,
            )
    return inputs_by_team


def score_candidates(
    candidates: Sequence[TeamPropCandidate | Mapping[str, Any]],
    inputs_by_team: Mapping[str, TeamPropsInputs],
    *,
    filter_player_props: bool = True,
    default_inputs: TeamPropsInputs | None = None,
) -> list[CandidateScore]:
    """Score a collection of candidates, strictly filtering out player props."""
    to_score = filter_team_prop_candidates(candidates) if filter_player_props else candidates
    scores: list[CandidateScore] = []
    fallback = default_inputs or default_team_props_inputs()

    for item in to_score:
        if isinstance(item, TeamPropCandidate):
            cand = item
        else:
            converted = filter_team_prop_candidates([item])
            if not converted:
                continue
            cand = converted[0]
        team_inputs = inputs_by_team.get(cand.team_id, fallback)
        try:
            score = score_candidate(cand, team_inputs, strict=False)
            scores.append(score)
        except SchemaError:
            continue

    return sorted(scores, key=lambda s: s.play_score, reverse=True)


def score_slate_team_props(
    conn: sqlite3.Connection,
    slate_date: str,
    *,
    inputs_by_team: Mapping[str, TeamPropsInputs] | None = None,
    season: int | None = None,
    as_of_utc: str | None = None,
) -> list[CandidateScore]:
    """Score all team props available for games on a given slate."""
    cutoff = to_utc_iso(as_of_utc)
    if cutoff is None:
        raise SchemaError("Team-prop scoring requires a valid as_of_utc cutoff")

    games = conn.execute(
        """SELECT game_id, home_team_id, away_team_id FROM games
           WHERE football_date = ?""",
        (slate_date,),
    ).fetchall()
    if not games:
        return []

    game_ids = [row["game_id"] for row in games]
    placeholders = ", ".join("?" for _ in game_ids)

    # Select exactly one point-in-time snapshot per team-prop identity. The
    # kickoff predicate is a second guard against closing/postgame leakage.
    consensus_rows = conn.execute(
        f"""WITH ranked AS (
                   SELECT t.*,
                          ROW_NUMBER() OVER (
                              PARTITION BY t.game_id, t.team_id, t.market,
                                           t.line, t.side
                              ORDER BY t.as_of_utc DESC
                          ) AS snapshot_rank
                   FROM team_prop_consensus AS t
                   JOIN games AS g ON g.game_id = t.game_id
                   WHERE t.game_id IN ({placeholders})
                     AND t.as_of_utc <= ?
                     AND t.as_of_utc < g.kickoff_utc
               )
               SELECT game_id, team_id, market, line, side, as_of_utc,
                      consensus_price, best_price, best_book, hold, n_books,
                      prob_shin, prob_multiplicative, flags
               FROM ranked
               WHERE snapshot_rank = 1""",
        [*game_ids, cutoff],
    ).fetchall()

    # Apply the same cutoff and latest-snapshot selection to movement.
    movements = {
        (row["game_id"], row["team_id"], row["market"], row["side"]): dict(row)
        for row in conn.execute(
            f"""WITH ranked AS (
                       SELECT m.*,
                              ROW_NUMBER() OVER (
                                  PARTITION BY m.game_id, m.team_id, m.market, m.side
                                  ORDER BY m.as_of_utc DESC
                              ) AS snapshot_rank
                       FROM team_prop_movement AS m
                       JOIN games AS g ON g.game_id = m.game_id
                       WHERE m.game_id IN ({placeholders})
                         AND m.as_of_utc <= ?
                         AND m.as_of_utc < g.kickoff_utc
                   )
                   SELECT game_id, team_id, market, side, open_line,
                          current_line, rlm_flag
                   FROM ranked
                   WHERE snapshot_rank = 1""",
            [*game_ids, cutoff],
        ).fetchall()
    }

    candidates: list[TeamPropCandidate] = []
    inputs_map = (
        dict(inputs_by_team)
        if inputs_by_team is not None
        else load_team_props_inputs_for_slate(conn, slate_date, season=season, as_of_utc=cutoff)
    )

    for row in consensus_rows:
        market_name = str(row["market"])
        if not is_supported_team_prop(market_name, market_type="TEAM_PROP"):
            continue

        game_id = str(row["game_id"])
        team_id = str(row["team_id"])
        side = str(row["side"]).upper()
        if side not in {"OVER", "UNDER"}:
            continue

        fair_prob = row["prob_shin"] if row["prob_shin"] is not None else row["prob_multiplicative"]
        move = movements.get((game_id, team_id, market_name, side), {})
        try:
            raw_flags = json.loads(row["flags"] or "[]")
            flags = tuple(str(flag) for flag in raw_flags) if isinstance(raw_flags, list) else ()
        except (TypeError, ValueError, json.JSONDecodeError):
            flags = ("invalid_flags",)

        candidates.append(
            TeamPropCandidate(
                game_id=game_id,
                team_id=team_id,
                market=market_name,
                side=side,
                line=float(row["line"]),
                market_type="TEAM_PROP",
                as_of_utc=str(row["as_of_utc"]),
                consensus_price=row["consensus_price"],
                best_price=row["best_price"],
                best_book=row["best_book"],
                n_books=int(row["n_books"]),
                hold=row["hold"],
                devigged_fair_prob=fair_prob,
                open_line=move.get("open_line"),
                current_line=move.get("current_line"),
                rlm_flag=bool(move.get("rlm_flag", 0)),
                flags=flags,
            )
        )
    return score_candidates(candidates, inputs_map)


def score_daily_slates(
    conn: sqlite3.Connection,
    slates: Sequence[str],
    *,
    inputs_by_team: Mapping[str, TeamPropsInputs] | None = None,
    season: int | None = None,
    as_of_utc: str | None = None,
) -> list[CandidateScore]:
    """Score all team props across active slates."""
    all_scores: list[CandidateScore] = []
    for slate in slates:
        all_scores.extend(
            score_slate_team_props(
                conn, slate, inputs_by_team=inputs_by_team, season=season, as_of_utc=as_of_utc
            )
        )
    return sorted(all_scores, key=lambda s: s.play_score, reverse=True)


__all__ = [
    "CandidateScore",
    "SUPPORTED_TEAM_PROPS",
    "TeamPropCandidate",
    "build_team_props_inputs_from_db",
    "calculate_expected_value",
    "filter_team_prop_candidates",
    "is_player_prop",
    "is_supported_team_prop",
    "load_team_props_inputs_for_slate",
    "normalize_team_prop_market",
    "resolve_devigged_fair_prob",
    "score_candidate",
    "score_candidates",
    "score_daily_slates",
    "score_slate_team_props",
]
