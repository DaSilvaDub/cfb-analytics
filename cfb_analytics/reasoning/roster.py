"""Roster Talent Composite, Blue-Chip Ratios, and Trench Attrition Engine.

Evaluates program talent composites, high-school recruiting classes, transfer
portal net scores, and trench attrition (starting OL/DL injuries with leverage
weights):
- 247/On3 True Talent Composite: Recruiting Composite + (Net Portal Composite * 2.75).
- Blue-Chip Ratio (BCR >= 0.50) and depth analysis.
- Starting OL/DL trench attrition with snap-share leverage weights; line yards & sack rate distortions.
- Trench matchup index verifying Grok Rule D (thin dogs are traps) exception gating.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from cfb_analytics.reasoning.models import (
    TalentProfile,
    TrenchHealth,
    TrenchMatchupResult,
)


def compute_true_talent_composite(
    recruiting_composite: float,
    net_portal: float,
) -> float:
    """Calculate 247/On3 True Talent Composite.

    TTC = Recruiting Composite + (Net Portal Composite * 2.75).
    """
    if recruiting_composite < 0.0:
        raise ValueError(f"Recruiting composite cannot be negative: {recruiting_composite}")
    return round(recruiting_composite + (net_portal * 2.75), 2)


def compute_blue_chip_ratio(
    four_star_count: int,
    five_star_count: int,
    total_signees: int,
) -> float:
    """Calculate Blue-Chip Ratio over 4-year recruiting cycle.

    BCR = (4-star + 5-star signees) / total signees.
    Championship baseline threshold: BCR >= 0.50.
    """
    if total_signees <= 0:
        return 0.0
    bcr = (four_star_count + five_star_count) / float(total_signees)
    return round(max(0.0, min(1.0, bcr)), 4)


def build_talent_profile(
    *,
    team_id: str,
    recruiting_composite: float,
    net_portal: float = 0.0,
    recruiting_rank: int | None = None,
    four_star_count: int = 0,
    five_star_count: int = 0,
    total_signees: int = 0,
) -> TalentProfile:
    """Build a comprehensive TalentProfile."""
    ttc = compute_true_talent_composite(recruiting_composite, net_portal)
    bcr = compute_blue_chip_ratio(four_star_count, five_star_count, total_signees)
    is_blue_chip = bcr >= 0.50

    if ttc >= 900.0 or bcr >= 0.50:
        tier = "ELITE"
    elif ttc >= 775.0 or bcr >= 0.30:
        tier = "UPPER_P4"
    elif ttc >= 675.0:
        tier = "MID_P4"
    elif ttc >= 550.0:
        tier = "LOWER_P4_HIGH_G5"
    else:
        tier = "G5_BASELINE"

    return TalentProfile(
        team_id=team_id,
        recruiting_composite=recruiting_composite,
        recruiting_rank=recruiting_rank,
        net_portal_composite=net_portal,
        true_talent_composite=ttc,
        blue_chip_ratio=bcr,
        is_blue_chip_program=is_blue_chip,
        talent_tier=tier,
    )


def evaluate_trench_attrition(
    team_id: str,
    injuries: Sequence[dict[str, Any]] | list[dict[str, Any]],
    base_line_yards: float = 3.0,
    base_sack_rate: float = 0.065,
) -> TrenchHealth:
    """Evaluates OL and DL starting attrition from availability reports.

    Leverage weights assign higher penalties to blindside Left Tackles and Centers
    who coordinate line protection and snap cadence.
    """
    ol_weights: dict[str, float] = {
        "LT": 1.25,
        "C": 1.25,
        "LG": 1.0,
        "RG": 1.0,
        "RT": 1.0,
        "OT": 1.1,
        "OG": 1.0,
        "OL": 1.0,
    }
    dl_weights: dict[str, float] = {
        "DT": 1.10,
        "NT": 1.10,
        "DE": 1.15,
        "EDGE": 1.15,
        "DL": 1.12,
    }
    severities: dict[str, float] = {
        "out": 1.00,
        "out for season": 1.00,
        "doubtful": 0.85,
        "questionable": 0.50,
        "probable": 0.15,
    }

    ol_missing: list[str] = []
    dl_missing: list[str] = []
    ol_weighted_loss = 0.0
    dl_weighted_loss = 0.0

    for inj in injuries:
        pos = str(inj.get("position", "")).strip().upper()
        grp = str(inj.get("position_group", "")).strip().upper()
        desig_str = str(inj.get("designation", "")).strip().lower()
        sev = severities.get(desig_str, 0.0)
        player_desc = f"{pos} ({inj.get('designation', 'Unavailable')})"

        if grp == "OL" or pos in ol_weights:
            w = ol_weights.get(pos, 1.0)
            ol_weighted_loss += w * sev
            if sev >= 0.50:
                ol_missing.append(player_desc)
        elif grp == "DL" or pos in dl_weights:
            w = dl_weights.get(pos, 1.12)
            dl_weighted_loss += w * sev
            if sev >= 0.50:
                dl_missing.append(player_desc)

    ol_attr = round(min(1.0, ol_weighted_loss / 5.50), 3)
    dl_attr = round(min(1.0, dl_weighted_loss / 4.50), 3)
    comp_attr = round(0.55 * ol_attr + 0.45 * dl_attr, 3)

    line_yards_m = round(max(0.65, 1.0 - 0.28 * ol_attr), 3)
    rush_sr_m = round(max(0.70, 1.0 - 0.20 * ol_attr), 3)
    sack_allowed_m = round(1.0 + 0.65 * ol_attr, 3)
    stuff_rate_m = round(max(0.60, 1.0 - 0.35 * dl_attr), 3)
    havoc_m = round(max(0.55, 1.0 - 0.40 * dl_attr), 3)
    sack_gen_m = round(max(0.50, 1.0 - 0.45 * dl_attr), 3)

    return TrenchHealth(
        team_id=team_id,
        ol_attrition=ol_attr,
        dl_attrition=dl_attr,
        composite_trench_attrition=comp_attr,
        ol_missing_starters=tuple(ol_missing),
        dl_missing_starters=tuple(dl_missing),
        line_yards_mult=line_yards_m,
        rush_success_rate_mult=rush_sr_m,
        sack_rate_allowed_mult=sack_allowed_m,
        def_stuff_rate_mult=stuff_rate_m,
        def_havoc_mult=havoc_m,
        def_sack_rate_mult=sack_gen_m,
        is_trench_compromised=(comp_attr >= 0.35) or (ol_attr >= 0.35) or (dl_attr >= 0.35),
    )


def evaluate_trench_matchup(
    offense_team_id: str,
    defense_team_id: str,
    ol_health: TrenchHealth,
    dl_health: TrenchHealth,
    base_line_yards: float = 3.0,
    base_sack_rate: float = 0.065,
) -> TrenchMatchupResult:
    """Head-to-head confrontation between offensive line and defensive front."""
    proj_line_yards = round(
        base_line_yards * ol_health.line_yards_mult * (2.0 - dl_health.def_stuff_rate_mult),
        2,
    )
    proj_sack_rate = round(
        base_sack_rate * ol_health.sack_rate_allowed_mult * dl_health.def_sack_rate_mult,
        4,
    )

    net_trench_push = proj_line_yards - base_line_yards
    net_pass_pro = dl_health.def_havoc_mult - ol_health.sack_rate_allowed_mult
    trench_mismatch_score = round(net_trench_push * 1.5 - net_pass_pro * 2.0, 2)
    is_decisive = abs(trench_mismatch_score) >= 1.25

    summary = (
        f"Trench Matchup: {offense_team_id} OL (attrition {ol_health.ol_attrition:.0%}) vs "
        f"{defense_team_id} DL (attrition {dl_health.dl_attrition:.0%}). "
        f"Proj Line Yards: {proj_line_yards:.2f}, Proj Sack Rate: {proj_sack_rate:.1%}. "
        f"Mismatch Score: {trench_mismatch_score:+.2f} ({'DECISIVE' if is_decisive else 'Balanced'})."
    )

    return TrenchMatchupResult(
        offense_team_id=offense_team_id,
        defense_team_id=defense_team_id,
        ol_health=ol_health,
        dl_health=dl_health,
        projected_line_yards=proj_line_yards,
        projected_sack_rate=proj_sack_rate,
        trench_mismatch_score=trench_mismatch_score,
        is_decisive_mismatch=is_decisive,
        summary_text=summary,
    )
