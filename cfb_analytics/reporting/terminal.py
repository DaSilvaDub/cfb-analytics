"""Authoritative Terminal Card Renderers for cfb-analytics.

Renders:
1. Authoritative 7-dimensional reasoning cards (docs/grok_rules.md §5).
2. Mispriced opportunity cards across spreads, totals, and team props.
3. Structured 3-10 leg parlay ticket cards with fragility and marginal EV audits.
4. CLI slate board and mispriced tables.

Strictly stdlib-only.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from typing import Any

from cfb_analytics.governance.models import (
    SHADOW_MODE_DISCLAIMER,
    GovernanceAction,
    GovernanceVerdict,
    ParlayTicket,
)
from cfb_analytics.reasoning.models import ReasoningCard
from cfb_analytics.scanner.models import MispricedOpportunity

DIVIDER_HEAVY = "=" * 80
DIVIDER_LIGHT = "-" * 80


def _wrap(text: str, width: int = 76, indent: str = "   ") -> str:
    """Cleanly wrap text with standard indentation."""
    if not text:
        return f"{indent}None."
    return textwrap.fill(text.strip(), width=width, initial_indent=indent, subsequent_indent=indent)


def render_reasoning_card(
    card: ReasoningCard,
    verdict: GovernanceVerdict | None = None,
    *,
    network: str | None = None,
) -> str:
    """Render the authoritative 7-dimensional terminal card per docs/grok_rules.md §5."""
    lines: list[str] = []

    # 1. Header Line 1: MATCHUP | Kickoff | Network
    matchup = card.matchup_label or card.game_id
    kickoff = card.kickoff_et or "TBD"
    tv_net = network or getattr(card, "network", None) or "TBD"
    lines.append(f"MATCHUP: {matchup} | Kickoff: {kickoff} | Network: {tv_net}")

    # 2. Header Line 2: MARKET | RECOMMENDED PLAY (Confidence: X.X/10 - TIER)
    market = card.market
    rec_play = card.recommended_play

    # Governance adjustments
    conf = card.confidence
    tier = card.tier
    is_vetoed = card.is_favorite_vetoed

    if verdict is not None:
        if round(verdict.adjusted_confidence, 1) != round(card.confidence, 1):
            conf_str = f"{verdict.adjusted_confidence:.1f}/10 [adj from {card.confidence:.1f}/10]"
        else:
            conf_str = f"{conf:.1f}/10"

        if verdict.action == GovernanceAction.VETO:
            tier = "VETOED"
            is_vetoed = True
        elif verdict.action == GovernanceAction.DOWNGRADE:
            tier = f"{card.tier} (DOWNGRADED)"

        if verdict.target_market_override:
            rec_play = f"{rec_play} -> [REDIRECT: {verdict.target_market_override}]"
    else:
        conf_str = f"{conf:.1f}/10"
        if is_vetoed:
            tier = "VETOED"

    lines.append(f"MARKET: {market} | RECOMMENDED PLAY: {rec_play} (Confidence: {conf_str} - {tier})")
    lines.append("")

    # 3. Seven Numbered Sections per docs/grok_rules.md §5
    lines.append("1. Tape Evaluation: Honest vs Junk")
    lines.append(_wrap(card.tape_summary))
    lines.append("")

    lines.append("2. Position & QB Edge Breakdown")
    lines.append(_wrap(card.position_qb_summary))
    lines.append("")

    lines.append("3. Weather, Venue & Travel Factors")
    lines.append(_wrap(card.weather_venue_summary))
    lines.append("")

    lines.append("4. Injuries & Trench Health")
    lines.append(_wrap(card.injuries_trench_summary))
    lines.append("")

    lines.append("5. Program Structure & Situational Spot")
    lines.append(_wrap(card.program_continuity_summary))
    lines.append("")

    lines.append("6. Mathematical Edge vs Market Consensus")
    lines.append(_wrap(card.mathematical_edge_summary))
    lines.append("")

    lines.append("7. What NOT to Bet (Contra-Indications)")
    contra = list(card.contra_indications)
    if verdict and verdict.counter_theses:
        for ct in verdict.counter_theses:
            if ct not in contra:
                contra.append(f"Governance Warning: {ct}")

    if contra:
        for item in contra:
            lines.append(f"   - {item}")
    else:
        lines.append("   - None identified. Affirmative edge confirmed across all dimensions.")

    # 4. Veto / Counter-Thesis Section
    if is_vetoed:
        lines.append("")
        lines.append(DIVIDER_HEAVY)
        lines.append("[VETO / COUNTER-THESIS]")
        lines.append("STATUS: STRICT AVOID / BETTING FORBIDDEN")
        rules: list[str] = []
        if verdict and verdict.triggered_rules:
            rules.extend(verdict.triggered_rules)
        if card.rule_triggers:
            for rt in card.rule_triggers:
                if rt not in rules:
                    rules.append(rt)
        if rules:
            lines.append(f"Triggered Governance Rules: {', '.join(rules)}")
        thesis = card.counter_thesis
        if not thesis and verdict and verdict.counter_theses:
            thesis = " ".join(verdict.counter_theses)
        if thesis:
            lines.append(f"Counter-Thesis: {thesis}")
        lines.append(DIVIDER_HEAVY)
    elif verdict and verdict.action == GovernanceAction.DOWNGRADE:
        lines.append("")
        lines.append(DIVIDER_LIGHT)
        lines.append(f"[GOVERNANCE ACTION: {verdict.action.value}]")
        if verdict.triggered_rules:
            lines.append(f"Triggered Rules: {', '.join(verdict.triggered_rules)}")
        if verdict.notes:
            lines.append(f"Notes: {verdict.notes}")
        lines.append(DIVIDER_LIGHT)

    lines.append("")
    lines.append(card.shadow_mode_disclaimer)
    return "\n".join(lines)


def render_mispriced_card(
    opp: MispricedOpportunity,
    verdict: GovernanceVerdict | None = None,
) -> str:
    """Render structured terminal summary for mispriced spread/total/prop opportunity."""
    lines: list[str] = []
    lines.append(DIVIDER_HEAVY)
    line_fmt = f"{opp.line:+g}" if opp.market_type == "SPREAD" else f"{opp.line:.1f}"
    lines.append(f"MISPRICED OPPORTUNITY: {opp.market} {opp.side} {line_fmt}")
    lines.append(DIVIDER_HEAVY)

    lines.append(f"Matchup:         {opp.game_label or opp.game_id} (Game ID: {opp.game_id})")
    lines.append(f"Kickoff:         {opp.kickoff_et or 'TBD'}")
    lines.append(f"Market Type:     {opp.market_type} ({opp.market})")
    lines.append(f"Play Tier:       {opp.play_tier} (Score: {opp.play_score:.1f}/100) | Status: {opp.qual_status}")
    lines.append(DIVIDER_LIGHT)

    lines.append("PRICING & EDGE QUANTIFICATION:")
    best_book_str = f" [Best: {opp.best_book}]" if opp.best_book else ""
    lines.append(
        f"  Posted Price:          {opp.posted_price_american:+d} "
        f"({opp.posted_price_decimal:.3f}){best_book_str}"
    )
    proj_fmt = f"{opp.model_projected_line:+g}" if opp.market_type == "SPREAD" else f"{opp.model_projected_line:.1f}"
    lines.append(f"  Model Projected Line:  {proj_fmt}")
    lines.append(f"  Model Win/Cover Prob:  {opp.model_prob * 100:.1f}%")
    lines.append(
        f"  Consensus Fair Prob:   {opp.consensus_fair_prob * 100:.1f}% "
        f"(Fair Price: {opp.consensus_fair_price_american:+d})"
    )
    lines.append(f"  Quantified Edge:       {opp.edge_pct * 100:+.1f}% ({opp.edge_pct:+.4f})")
    lines.append(f"  Expected Value (EV):   {opp.ev * 100:+.2f}%")
    lines.append(
        f"  Devig Method Spread:   {opp.method_spread * 100:.2f}pp ({opp.method_spread:.4f}) "
        f"across {opp.n_books} contributing books"
    )
    lines.append(f"  Actionable:            {'YES' if opp.is_actionable else 'NO'}")
    lines.append(DIVIDER_LIGHT)

    lines.append("FLAGS & AUDIT:")
    lines.append(f"  Market Flags:          {', '.join(opp.flags) if opp.flags else 'None'}")
    lines.append(f"  Rejection Reasons:     {', '.join(opp.rejection_reasons) if opp.rejection_reasons else 'None'}")

    if verdict is not None:
        lines.append(DIVIDER_LIGHT)
        lines.append("GOVERNANCE AUDIT:")
        action_val = verdict.action.value if hasattr(verdict.action, "value") else str(verdict.action)
        lines.append(f"  Action:                {action_val}")
        lines.append(f"  Triggered Rules:       {', '.join(verdict.triggered_rules) if verdict.triggered_rules else 'None'}")
        lines.append(f"  Adjusted Confidence:   {verdict.adjusted_confidence:.1f}/10 (Original: {verdict.original_confidence:.1f}/10)")
        lines.append(f"  Parlay Eligible:       {'YES' if verdict.parlay_eligible else 'NO'}")
        lines.append(f"  Target Override:       {verdict.target_market_override or 'None'}")
        lines.append(f"  Counter-Theses:        {'; '.join(verdict.counter_theses) if verdict.counter_theses else 'None'}")
        lines.append(f"  Notes:                 {verdict.notes or 'None'}")

    lines.append(DIVIDER_HEAVY)
    lines.append(opp.shadow_mode_disclaimer)
    lines.append(DIVIDER_HEAVY)
    return "\n".join(lines)


def render_parlay_card(ticket: ParlayTicket) -> str:
    """Render structured summary of a 3-10 leg parlay ticket."""
    lines: list[str] = []
    lines.append(DIVIDER_HEAVY)
    lines.append(f"PARLAY TICKET: {ticket.parlay_id} ({ticket.leg_count} LEGS / {ticket.leg_count}-Leg Optimized Multi-Matchup)")
    lines.append(DIVIDER_HEAVY)

    # 1. Summary Metrics
    fragility_tier = (
        "LOW RISK" if ticket.fragility_index < 0.25
        else "MODERATE RISK" if ticket.fragility_index < 0.45
        else "HIGH RISK"
    )
    lines.append("PORTFOLIO SUMMARY:")
    lines.append(f"  Total Payout:          {ticket.total_odds_american:+d} ({ticket.total_payout_multiplier:.2f}x stake)")
    lines.append(f"  Joint Win Prob:        {ticket.joint_win_prob * 100:.2f}% (Raw Unadjusted: {ticket.raw_win_prob * 100:.2f}%)")
    lines.append(f"  Correlation Penalty:   {ticket.correlation_penalty * 100:.2f}% risk discount")
    lines.append(f"  Expected Value (EV):   {ticket.ev * 100:+.2f}%")
    lines.append(f"  Fragility Index:       {ticket.fragility_index:.3f} / 1.000 [{fragility_tier}]")
    lines.append(f"  Efficiency Ratio:      {ticket.efficiency_ratio:.2f} (EV per unit fragility)")
    lines.append(f"  Actionable:            {'YES' if ticket.is_actionable else 'NO'}")
    lines.append(DIVIDER_LIGHT)

    # 2. Component Legs Table
    lines.append(f"COMPONENT LEGS ({ticket.leg_count} of 3-10):")
    lines.append("  #  Team vs Opponent               Conf      Odds    Fair P   Edge%    Hazards")
    lines.append("  -- ------------------------------ --------- ------- -------- -------- -------")
    for idx, leg in enumerate(ticket.legs, 1):
        matchup = f"{leg.team} vs {leg.opponent}"[:30]
        conf = leg.conference[:9] if leg.conference else "FBS"

        hazards_list: list[str] = []
        if getattr(leg, "weather_hazard", False):
            hazards_list.append("Weather")
        if getattr(leg, "qb_news_risk", False):
            hazards_list.append("QB News")
        if getattr(leg, "is_heavy_favorite", False):
            hazards_list.append("Chalk")
        hazards_str = ", ".join(hazards_list) if hazards_list else "None"

        lines.append(
            f"  {idx:>2}. {matchup:<30} {conf:<9} {leg.odds_american:>+7d} "
            f"{leg.fair_prob * 100:>7.1f}% {leg.edge_pct * 100:>+7.1f}%   {hazards_str}"
        )
    lines.append(DIVIDER_LIGHT)

    # 3. Marginal Leg Audit
    if ticket.marginal_analyses:
        lines.append("MARGINAL LEG AUDIT & FRAGILITY CONTRIBUTION:")
        lines.append("  #  Team            Marginal EV   Delta Fragility   Efficiency   Parasitic?")
        lines.append("  -- --------------- ------------- ----------------- ------------ ----------")
        for m in ticket.marginal_analyses:
            parasitic_str = "YES (PARASITIC)" if m.is_parasitic else "NO"
            lines.append(
                f"  {m.leg_index + 1:>2}. {m.team[:15]:<15} {m.marginal_ev * 100:>+12.2f}% "
                f"{m.delta_fragility:>+16.4f} {m.marginal_efficiency:>11.2f}   {parasitic_str}"
            )
        lines.append(DIVIDER_LIGHT)

    # 4. Vulnerability & Weakest Link
    lines.append("VULNERABILITY & PRUNING AUDIT:")
    if ticket.weakest_leg:
        lines.append(f"  Weakest Link:          {ticket.weakest_leg.team} ({ticket.weakest_leg.fair_prob * 100:.1f}% fair win prob)")
    parasitic_msg = (
        "WARNING: Contains parasitic leg degrading portfolio EV!"
        if ticket.has_parasitic_leg
        else "CLEAN: All legs contribute positive marginal EV"
    )
    lines.append(f"  Parasitic Leg Check:   {parasitic_msg}")

    if ticket.alternate_pruned_ticket:
        alt = ticket.alternate_pruned_ticket
        lines.append(
            f"  Pruned Alternative:    {alt.parlay_id} ({alt.leg_count} legs, "
            f"Odds: {alt.total_odds_american:+d}, EV: {alt.ev * 100:+.2f}%, Fragility: {alt.fragility_index:.3f})"
        )

    lines.append(DIVIDER_HEAVY)
    lines.append(ticket.disclaimer)
    lines.append(DIVIDER_HEAVY)
    return "\n".join(lines)


def render_board_terminal(
    date: str,
    rows: Sequence[Any],
    min_prob: float = 0.0,
    with_reasoning: bool = False,
    cards: dict[str, ReasoningCard] | None = None,
    verdicts: dict[str, GovernanceVerdict] | None = None,
) -> str:
    """Render the full moneyline board and optional reasoning cards to string."""
    cards_map = cards or {}
    verdicts_map = verdicts or {}
    lines: list[str] = []

    lines.append(f"MONEYLINE BOARD - {date}   [{SHADOW_MODE_DISCLAIMER}]")
    lines.append(
        f"\n{'team':<7} {'opp':<7} {'price':>7} {'best':>7} {'book':<11} "
        f"{'fair%':>7} {'spread':>7} {'hold':>6} {'bk':>3}  flags"
    )

    for row in rows:
        prob = row["prob_shin"] or row["prob_multiplicative"]
        if prob is None or prob < min_prob:
            continue
        team = row["home"] if row["side"] == "HOME" else row["away"]
        opp = row["away"] if row["side"] == "HOME" else row["home"]
        flags_raw = row["flags"] if "flags" in row.keys() else "[]"
        flags = ",".join(json.loads(flags_raw or "[]")) if isinstance(flags_raw, str) else ""
        lines.append(
            f"{team or '?':<7} {opp or '?':<7} {row['consensus_price']:>7} "
            f"{row['best_price']:>7} {(row['best_book'] or ''):<11} "
            f"{prob * 100:>6.1f}% {(row['prob_spread'] or 0) * 100:>6.2f}pp "
            f"{(row['hold'] or 0) * 100:>5.1f}% {row['n_books']:>3}  {flags}"
        )

    if with_reasoning and cards_map:
        lines.append("")
        lines.append(DIVIDER_HEAVY)
        lines.append("MULTI-FACTOR REASONING CARDS")
        lines.append(DIVIDER_HEAVY)
        for key, card in cards_map.items():
            verdict = verdicts_map.get(key)
            lines.append("")
            lines.append(render_reasoning_card(card, verdict=verdict))

    lines.append(
        "\nfair% is the vig-free market probability (Shin). spread is the "
        "disagreement\nbetween devig methods - wide means the fair number is "
        "method-dependent."
    )
    lines.append("This is the MARKET's view only. No model probability or edge exists yet.")
    lines.append(f"\n{SHADOW_MODE_DISCLAIMER}")
    return "\n".join(lines)


def render_mispriced_terminal(
    date: str,
    candidates: Sequence[MispricedOpportunity],
    verdicts: dict[str, GovernanceVerdict] | None = None,
    min_edge: float = 0.02,
) -> str:
    """Render the mispriced board table and cards."""
    verdicts_map = verdicts or {}
    lines: list[str] = []

    lines.append(f"MISPRICED BOARD - {date}   [{SHADOW_MODE_DISCLAIMER}]")
    if not candidates:
        lines.append(f"\nNo mispriced opportunities found for {date} (min_edge={min_edge}).")
        lines.append(f"\n{SHADOW_MODE_DISCLAIMER}")
        return "\n".join(lines)

    lines.append(
        f"\n {'#':>2} | {'market':<7} | {'game':<28} | {'side':<5} | {'line':>6} | "
        f"{'fair%':>6} | {'model%':>6} | {'edge%':>7} | {'spread':>7} | {'score':>5} | "
        f"{'tier':<9} | {'qual':<9} | {'action':<8} | best px (book)"
    )
    lines.append("-" * 140)

    vetoed_candidates: list[tuple[int, MispricedOpportunity, GovernanceVerdict | None]] = []

    for idx, opp in enumerate(candidates, 1):
        key = f"{opp.game_id}:{opp.market_type}:{opp.side}"
        verdict = verdicts_map.get(key)
        action_str = verdict.action.value if verdict and hasattr(verdict.action, "value") else (str(verdict.action) if verdict else "APPROVE")
        if verdict and verdict.action == GovernanceAction.VETO:
            vetoed_candidates.append((idx, opp, verdict))

        game_str = (opp.game_label or opp.game_id)[:28]
        line_str = f"{opp.line:+g}" if opp.market_type == "SPREAD" else f"{opp.line:.1f}"
        best_str = f"{opp.posted_price_american:+d} ({opp.best_book or ''})"

        lines.append(
            f" {idx:>2} | {opp.market_type:<7} | {game_str:<28} | {opp.side:<5} | {line_str:>6} | "
            f"{opp.consensus_fair_prob * 100:>5.1f}% | {opp.model_prob * 100:>5.1f}% | "
            f"{opp.edge_pct * 100:>+6.1f}% | {opp.method_spread * 100:>5.2f}pp | {opp.play_score:>5.1f} | "
            f"{str(opp.play_tier):<9} | {str(opp.qual_status):<9} | {action_str:<8} | {best_str}"
        )

    if vetoed_candidates:
        lines.append("")
        lines.append(DIVIDER_HEAVY)
        lines.append("GOVERNANCE VETO CALLOUTS")
        lines.append(DIVIDER_HEAVY)
        for idx, opp, v in vetoed_candidates:
            lines.append(f"\n[GOVERNANCE VETO] Candidate #{idx} ({opp.market} {opp.side} {opp.line}):")
            rules = ", ".join(v.triggered_rules) if v and v.triggered_rules else "Negative Gate"
            lines.append(f"  Triggered Rules: {rules}")
            theses = "; ".join(v.counter_theses) if v and v.counter_theses else "Failed governance qualification."
            lines.append(f"  Counter-Thesis:  {theses}")

    lines.append(f"\nTotal: {len(candidates)} mispriced opportunities detected.")
    lines.append(SHADOW_MODE_DISCLAIMER)
    return "\n".join(lines)
