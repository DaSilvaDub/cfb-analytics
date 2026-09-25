"""Honest vs. Junk Tape Analysis and EPA Floor/Ceiling Engine.

College football non-conference slates routinely feature elite Power programs
running up astronomical scores and EPA efficiency against outmatched FCS programs
(e.g., 52-0 vs. Mississippi Valley State, 49-3 vs. Sacramento State). These exhibitions
reflect gross talent disparities rather than sustainable offensive execution.

This module categorizes opponents into competitive tiers, filters cupcake blowout
exhibitions from offensive ceilings and baseline identities, preserves struggle tape
to establish true performance floors, and calculates honest EPA / success rate bounds.
"""

from __future__ import annotations

from collections.abc import Sequence

from cfb_analytics.reasoning.models import (
    OpponentTier,
    TapeCategory,
    TapeGame,
    TapeProfile,
)

# Canonical conference groupings (case-insensitive)
P4_CONFERENCES: frozenset[str] = frozenset({
    "sec",
    "southeastern",
    "southeastern conference",
    "big ten",
    "big ten conference",
    "b1g",
    "big 12",
    "big 12 conference",
    "big12",
    "acc",
    "atlantic coast conference",
    "pac-12",
    "pac 12",
    "pac12",  # Historical Power
})

G5_CONFERENCES: frozenset[str] = frozenset({
    "american",
    "american athletic",
    "american athletic conference",
    "aac",
    "mountain west",
    "mountain west conference",
    "mwc",
    "sun belt",
    "sun belt conference",
    "sbc",
    "mid-american",
    "mid-american conference",
    "mac",
    "conference usa",
    "c-usa",
    "cusa",
})

FCS_CONFERENCES: frozenset[str] = frozenset({
    "big sky",
    "mvfc",
    "missouri valley football",
    "missouri valley",
    "caa",
    "ivy",
    "ivy league",
    "southern",
    "socon",
    "southland",
    "swac",
    "meac",
    "patriot",
    "patriot league",
    "pioneer",
    "ovc",
    "united athletic",
    "nec",
    "fcs",
})

# Thresholds for junk cupcake blowout identification
BLOWOUT_MARGIN_THRESHOLD: int = 24
BLOWOUT_POINTS_SCORED: int = 42
BLOWOUT_POINTS_ALLOWED: int = 10


def classify_opponent(
    *,
    classification: str | None = None,
    conference: str | None = None,
    school: str | None = None,
    elo_rating: float | None = None,
    talent_composite: float | None = None,
) -> OpponentTier:
    """Classify an opponent into Power, Honest FBS, Bottom G5, or Cupcake FCS."""
    c_lower = (classification or "").strip().lower()
    conf_lower = (conference or "").strip().lower()
    school_lower = (school or "").strip().lower()

    # 1. FCS Check
    if c_lower == "fcs" or conf_lower in FCS_CONFERENCES:
        return OpponentTier.CUPCAKE_FCS

    # 2. Notre Dame Special Case (Independent with Elite Power Roster)
    if "notre dame" in school_lower:
        return OpponentTier.POWER_CONFERENCE

    # 3. Power Conference Check
    if conf_lower in P4_CONFERENCES:
        return OpponentTier.POWER_CONFERENCE

    # 4. Bottom-10 G5 Check (Severe talent / rating deficiency)
    if conf_lower in G5_CONFERENCES or c_lower == "fbs":
        if (elo_rating is not None and elo_rating < 1300.0) or (
            talent_composite is not None and talent_composite < 450.0
        ):
            return OpponentTier.BOTTOM_G5
        if conf_lower in {
            "mac",
            "mid-american",
            "mid-american conference",
            "c-usa",
            "cusa",
            "conference usa",
        }:
            if elo_rating is not None and elo_rating < 1350.0:
                return OpponentTier.BOTTOM_G5
        return OpponentTier.HONEST_FBS

    return OpponentTier.UNKNOWN


def classify_tape_game(
    points_scored: int,
    points_allowed: int,
    opponent_tier: OpponentTier,
) -> TapeCategory:
    """Classify a completed game into honest tape or cupcake blowout."""
    margin = points_scored - points_allowed

    if opponent_tier == OpponentTier.POWER_CONFERENCE:
        return TapeCategory.HONEST_POWER

    if opponent_tier == OpponentTier.HONEST_FBS:
        return TapeCategory.HONEST_FBS

    if opponent_tier in (OpponentTier.CUPCAKE_FCS, OpponentTier.BOTTOM_G5):
        # A blowout against cupcake competition is JUNK tape
        if margin >= BLOWOUT_MARGIN_THRESHOLD or (
            points_scored >= BLOWOUT_POINTS_SCORED
            and points_allowed <= BLOWOUT_POINTS_ALLOWED
        ):
            return TapeCategory.CUPCAKE_BLOWOUT
        # A close game or loss against a cupcake is HONEST STRUGGLE tape (establishes floor!)
        return TapeCategory.HONEST_STRUGGLE

    return TapeCategory.UNCLASSIFIED


def create_tape_game(
    *,
    game_id: str,
    season: int,
    week: int,
    kickoff_utc: str,
    opponent_team_id: str,
    opponent_name: str,
    opponent_conference: str,
    opponent_classification: str,
    is_home: bool,
    points_scored: int,
    points_allowed: int,
    offensive_epa: float,
    defensive_epa: float,
    offensive_success_rate: float,
    defensive_success_rate: float,
    rushing_yards: float | None = None,
    net_passing_yards: float | None = None,
    total_yards: float | None = None,
    elo_rating: float | None = None,
    talent_composite: float | None = None,
) -> TapeGame:
    """Helper to instantiate a fully classified TapeGame."""
    margin = points_scored - points_allowed
    tier = classify_opponent(
        classification=opponent_classification,
        conference=opponent_conference,
        school=opponent_name,
        elo_rating=elo_rating,
        talent_composite=talent_composite,
    )
    category = classify_tape_game(points_scored, points_allowed, tier)
    is_cupcake_blowout = category == TapeCategory.CUPCAKE_BLOWOUT

    return TapeGame(
        game_id=game_id,
        season=season,
        week=week,
        kickoff_utc=kickoff_utc,
        opponent_team_id=opponent_team_id,
        opponent_name=opponent_name,
        opponent_conference=opponent_conference,
        opponent_classification=opponent_classification,
        is_home=is_home,
        points_scored=points_scored,
        points_allowed=points_allowed,
        margin=margin,
        offensive_epa=offensive_epa,
        defensive_epa=defensive_epa,
        offensive_success_rate=offensive_success_rate,
        defensive_success_rate=defensive_success_rate,
        rushing_yards=rushing_yards,
        net_passing_yards=net_passing_yards,
        total_yards=total_yards,
        opponent_tier=tier,
        tape_category=category,
        is_cupcake_blowout=is_cupcake_blowout,
    )


def analyze_tape(
    games: Sequence[TapeGame],
    team_id: str = "",
    team_name: str = "",
    *,
    prior_epa: float = 0.0,
    prior_sr: float = 0.42,
    prior_points: float = 27.0,
) -> TapeProfile:
    """Analyze a team's tape, isolating honest games and calculating true efficiency bounds."""
    all_games = tuple(games)
    total_games = len(all_games)

    # Filter out cupcake blowouts
    honest_games = tuple(g for g in all_games if not g.is_cupcake_blowout)
    cupcake_games = tuple(g for g in all_games if g.is_cupcake_blowout)
    cupcake_count = len(cupcake_games)
    honest_count = len(honest_games)

    has_honest = honest_count > 0

    if has_honest:
        # Honest Offensive Floors & Ceilings
        off_floor_epa = min(g.offensive_epa for g in honest_games)
        off_ceil_epa = max(g.offensive_epa for g in honest_games)
        mean_off_epa = sum(g.offensive_epa for g in honest_games) / honest_count

        off_floor_sr = min(g.offensive_success_rate for g in honest_games)
        off_ceil_sr = max(g.offensive_success_rate for g in honest_games)
        mean_off_sr = sum(g.offensive_success_rate for g in honest_games) / honest_count

        off_floor_pts = min(g.points_scored for g in honest_games)
        off_ceil_pts = max(g.points_scored for g in honest_games)
        mean_pts_scored = sum(g.points_scored for g in honest_games) / honest_count

        # Honest Defensive Floors & Ceilings (lower EPA allowed = better defense)
        def_floor_epa = min(g.defensive_epa for g in honest_games)
        def_ceil_epa = max(g.defensive_epa for g in honest_games)
        mean_def_epa = sum(g.defensive_epa for g in honest_games) / honest_count

        def_floor_sr = min(g.defensive_success_rate for g in honest_games)
        def_ceil_sr = max(g.defensive_success_rate for g in honest_games)
        mean_def_sr = sum(g.defensive_success_rate for g in honest_games) / honest_count

        def_floor_pts = min(g.points_allowed for g in honest_games)
        def_ceil_pts = max(g.points_allowed for g in honest_games)
        mean_pts_allowed = sum(g.points_allowed for g in honest_games) / honest_count

        confidence_penalty = 5.0 if honest_count == 1 else 0.0

        # Honest YPP estimate
        ypp_vals = [
            (g.total_yards / 68.0)
            for g in honest_games
            if g.total_yards is not None and g.total_yards > 0
        ]
        honest_ypp = round(sum(ypp_vals) / len(ypp_vals), 2) if ypp_vals else 5.5
    else:
        # Zero Honest Games Fallback: Regress to Preseason Prior
        # NEVER use the cupcake blowout stats as the ceiling!
        off_floor_epa = round(prior_epa - 0.20, 3)
        off_ceil_epa = round(prior_epa + 0.10, 3)
        mean_off_epa = round(prior_epa, 3)

        off_floor_sr = round(prior_sr - 0.10, 3)
        off_ceil_sr = round(prior_sr + 0.06, 3)
        mean_off_sr = round(prior_sr, 3)

        off_floor_pts = max(7, int(prior_points - 14))
        off_ceil_pts = int(prior_points + 7)
        mean_pts_scored = round(prior_points, 1)

        def_floor_epa = round(prior_epa - 0.10, 3)
        def_ceil_epa = round(prior_epa + 0.20, 3)
        mean_def_epa = round(prior_epa, 3)

        def_floor_sr = round(prior_sr - 0.06, 3)
        def_ceil_sr = round(prior_sr + 0.10, 3)
        mean_def_sr = round(prior_sr, 3)

        def_floor_pts = max(7, int(prior_points - 10))
        def_ceil_pts = int(prior_points + 14)
        mean_pts_allowed = round(prior_points, 1)

        confidence_penalty = 25.0  # Massive epistemic confidence penalty
        honest_ypp = 5.5

    # Quantify Cupcake Inflation Gap
    raw_ceil_pts = max((g.points_scored for g in all_games), default=0)
    raw_ceil_epa = max((g.offensive_epa for g in all_games), default=0.0)

    inflation_gap_pts = max(0.0, float(raw_ceil_pts - off_ceil_pts))
    inflation_gap_epa = max(0.0, round(raw_ceil_epa - off_ceil_epa, 3))

    is_inflated = (
        cupcake_count > 0
        and (inflation_gap_pts >= 14.0 or inflation_gap_epa >= 0.18)
    ) or (not has_honest)

    # Generate Descriptive Tape Summary
    if not has_honest:
        summary = (
            f"WARNING: Zero honest tape ({cupcake_count} cupcake blowouts filtered). "
            f"Raw ceiling {raw_ceil_pts} pts (+{raw_ceil_epa:.2f} EPA) unverified against Power competition; "
            f"regressed ceiling capped at {off_ceil_pts} pts (+{off_ceil_epa:.2f} EPA)."
        )
    elif is_inflated:
        summary = (
            f"CUPCAKE INFLATION DETECTED: {cupcake_count} cupcake blowouts removed. "
            f"Raw ceiling {raw_ceil_pts} pts (+{raw_ceil_epa:.2f} EPA) inflated by +{inflation_gap_pts:.0f} pts "
            f"over honest ceiling {off_ceil_pts} pts (+{off_ceil_epa:.2f} EPA across {honest_count} honest games). "
            f"Honest offensive floor is {off_floor_pts} pts ({off_floor_epa:+.2f} EPA)."
        )
    else:
        summary = (
            f"Honest tape baseline: {honest_count} games evaluated. "
            f"Offensive bounds: {off_floor_pts}-{off_ceil_pts} pts (EPA {off_floor_epa:+.2f} to {off_ceil_epa:+.2f}, "
            f"SR {off_floor_sr:.1%}-{off_ceil_sr:.1%}). "
            f"Defensive bounds: {def_floor_pts}-{def_ceil_pts} pts allowed."
        )

    return TapeProfile(
        honest_games=honest_games,
        offensive_floor_epa=round(off_floor_epa, 3),
        offensive_ceiling_epa=round(off_ceil_epa, 3),
        defensive_floor_epa=round(def_floor_epa, 3),
        defensive_ceiling_epa=round(def_ceil_epa, 3),
        cupcake_games_filtered=cupcake_count,
        honest_success_rate=round(mean_off_sr, 3),
        honest_yards_per_play=honest_ypp,
        team_id=team_id,
        team_name=team_name,
        total_games=total_games,
        honest_games_count=honest_count,
        has_honest_tape=has_honest,
        is_cupcake_inflated=is_inflated,
        mean_offensive_epa=round(mean_off_epa, 3),
        offensive_floor_success_rate=round(off_floor_sr, 3),
        offensive_ceiling_success_rate=round(off_ceil_sr, 3),
        mean_offensive_success_rate=round(mean_off_sr, 3),
        mean_defensive_epa=round(mean_def_epa, 3),
        defensive_floor_success_rate=round(def_floor_sr, 3),
        defensive_ceiling_success_rate=round(def_ceil_sr, 3),
        mean_defensive_success_rate=round(mean_def_sr, 3),
        offensive_floor_points=off_floor_pts,
        offensive_ceiling_points=off_ceil_pts,
        mean_points_scored=round(mean_pts_scored, 1),
        defensive_floor_points=def_floor_pts,
        defensive_ceiling_points=def_ceil_pts,
        mean_points_allowed=round(mean_pts_allowed, 1),
        raw_ceiling_points=raw_ceil_pts,
        raw_ceiling_epa=round(raw_ceil_epa, 3),
        inflation_gap_points=inflation_gap_pts,
        inflation_gap_epa=inflation_gap_epa,
        tape_confidence_penalty=confidence_penalty,
        cupcake_games=cupcake_games,
        all_games=all_games,
        tape_summary=summary,
    )
