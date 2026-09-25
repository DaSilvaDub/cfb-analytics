"""Structured JSON Export Utilities for cfb-analytics.

Serializes:
- Reasoning cards & governance verdicts
- Mispriced opportunities & scanner candidates
- Multi-leg parlay tickets & marginal fragility audits
- Board summaries

Strictly enforces shadow mode watermark:
UNPROMOTED - shadow output, not decision-grade
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from cfb_analytics.governance.models import (
    SHADOW_MODE_DISCLAIMER,
    GovernanceVerdict,
    ParlayTicket,
)
from cfb_analytics.reasoning.models import ReasoningCard
from cfb_analytics.scanner.models import MispricedOpportunity


def _resolve_verdict(
    key: str,
    verdicts: GovernanceVerdict
    | Sequence[GovernanceVerdict]
    | Mapping[str, GovernanceVerdict]
    | None,
    index: int = 0,
) -> GovernanceVerdict | None:
    """Helper to match a GovernanceVerdict to a card or opportunity."""
    if verdicts is None:
        return None
    if isinstance(verdicts, GovernanceVerdict):
        return verdicts
    if isinstance(verdicts, Mapping):
        if key in verdicts:
            return verdicts[key]
        # Try fallback matching by candidate_id or game_id
        for v in verdicts.values():
            if v.candidate_id == key or getattr(v, "game_id", None) == key:
                return v
        return None
    if isinstance(verdicts, Sequence):
        for v in verdicts:
            if v.candidate_id == key or getattr(v, "game_id", None) == key:
                return v
        if 0 <= index < len(verdicts):
            return verdicts[index]
    return None


def export_reasoning_json(
    cards: ReasoningCard | Sequence[ReasoningCard],
    verdicts: GovernanceVerdict
    | Sequence[GovernanceVerdict]
    | Mapping[str, GovernanceVerdict]
    | None = None,
    *,
    slate_date: str | None = None,
    indent: int = 2,
) -> str:
    """Serialize ReasoningCard instances and optional verdicts to structured JSON."""
    is_single = isinstance(cards, ReasoningCard)
    card_list: Sequence[ReasoningCard] = [cards] if is_single else cards

    records: list[dict[str, Any]] = []
    for idx, card in enumerate(card_list):
        candidate_key = f"{card.game_id}:{card.market}:{card.side}"
        verdict = _resolve_verdict(candidate_key, verdicts, idx)

        record: dict[str, Any] = {
            "game_id": card.game_id,
            "market": card.market,
            "side": card.side,
            "recommended_play": card.recommended_play,
            "confidence": round(card.confidence, 2),
            "tier": card.tier,
            "matchup_label": card.matchup_label,
            "kickoff_et": card.kickoff_et,
            "line": card.line,
            "price_american": card.price_american,
            "is_favorite_vetoed": card.is_favorite_vetoed,
            "counter_thesis": card.counter_thesis,
            "rule_triggers": list(card.rule_triggers),
            "tape_summary": card.tape_summary,
            "position_qb_summary": card.position_qb_summary,
            "weather_venue_summary": card.weather_venue_summary,
            "injuries_trench_summary": card.injuries_trench_summary,
            "program_continuity_summary": card.program_continuity_summary,
            "mathematical_edge_summary": card.mathematical_edge_summary,
            "contra_indications": list(card.contra_indications),
            "dimensions": {
                "tape_evaluation": card.tape_summary,
                "position_qb": card.position_qb_summary,
                "weather_venue": card.weather_venue_summary,
                "injuries_trench": card.injuries_trench_summary,
                "program_structure": card.program_continuity_summary,
                "mathematical_edge": card.mathematical_edge_summary,
                "contra_indications": list(card.contra_indications),
            },
            "verdict": verdict.to_dict() if verdict else None,
            "governance_verdict": verdict.to_dict() if verdict else None,
            "disclaimer": SHADOW_MODE_DISCLAIMER,
        }
        records.append(record)

    # If single card without slate_date container request, still provide disclaimer
    if is_single and slate_date is None:
        return json.dumps(records[0], indent=indent)

    payload = {
        "date": slate_date or (card_list[0].kickoff_et if card_list else ""),
        "disclaimer": SHADOW_MODE_DISCLAIMER,
        "stamp": SHADOW_MODE_DISCLAIMER,
        "count": len(records),
        "cards": records,
        "records": records,
    }
    return json.dumps(payload, indent=indent)


def export_mispriced_json(
    opportunities: MispricedOpportunity | Sequence[MispricedOpportunity],
    verdicts: GovernanceVerdict
    | Sequence[GovernanceVerdict]
    | Mapping[str, GovernanceVerdict]
    | None = None,
    *,
    slate_date: str | None = None,
    min_edge: float | None = None,
    indent: int = 2,
) -> str:
    """Serialize MispricedOpportunity instances and optional verdicts to structured JSON."""
    is_single = isinstance(opportunities, MispricedOpportunity)
    opp_list: Sequence[MispricedOpportunity] = [opportunities] if is_single else opportunities

    records: list[dict[str, Any]] = []
    for idx, opp in enumerate(opp_list):
        candidate_key = f"{opp.game_id}:{opp.market_type}:{opp.side}"
        verdict = _resolve_verdict(candidate_key, verdicts, idx)

        record: dict[str, Any] = {
            "game_id": opp.game_id,
            "game_label": opp.game_label,
            "kickoff_et": opp.kickoff_et,
            "market_type": opp.market_type,
            "market": opp.market,
            "side": opp.side,
            "line": opp.line,
            "posted_price_american": opp.posted_price_american,
            "posted_price_decimal": round(opp.posted_price_decimal, 4),
            "consensus_fair_prob": round(opp.consensus_fair_prob, 4),
            "consensus_fair_price_american": opp.consensus_fair_price_american,
            "model_projected_line": round(opp.model_projected_line, 2),
            "model_prob": round(opp.model_prob, 4),
            "edge_pct": round(opp.edge_pct, 4),
            "ev": round(opp.ev, 4),
            "method_spread": round(opp.method_spread, 4),
            "n_books": opp.n_books,
            "play_score": round(opp.play_score, 2),
            "play_tier": str(
                opp.play_tier.value if hasattr(opp.play_tier, "value") else opp.play_tier
            ),
            "qual_status": str(
                opp.qual_status.value if hasattr(opp.qual_status, "value") else opp.qual_status
            ),
            "is_actionable": opp.is_actionable,
            "best_book": opp.best_book,
            "flags": list(opp.flags),
            "rejection_reasons": list(opp.rejection_reasons),
            "verdict": verdict.to_dict() if verdict else None,
            "governance_verdict": verdict.to_dict() if verdict else None,
            "disclaimer": SHADOW_MODE_DISCLAIMER,
        }
        records.append(record)

    if is_single and slate_date is None and min_edge is None:
        return json.dumps(records[0], indent=indent)

    payload = {
        "date": slate_date or "",
        "min_edge": min_edge,
        "disclaimer": SHADOW_MODE_DISCLAIMER,
        "stamp": SHADOW_MODE_DISCLAIMER,
        "total_detected": len(records),
        "count": len(records),
        "candidates": records,
        "records": records,
    }
    return json.dumps(payload, indent=indent)


def export_parlay_json(
    tickets: ParlayTicket | Sequence[ParlayTicket],
    *,
    indent: int = 2,
) -> str:
    """Serialize ParlayTicket instances to structured JSON."""
    is_single = isinstance(tickets, ParlayTicket)
    ticket_list: Sequence[ParlayTicket] = [tickets] if is_single else tickets

    serialized_tickets = [t.to_dict() for t in ticket_list]

    if is_single:
        return json.dumps(serialized_tickets[0], indent=indent)

    payload = {
        "disclaimer": SHADOW_MODE_DISCLAIMER,
        "stamp": SHADOW_MODE_DISCLAIMER,
        "count": len(serialized_tickets),
        "tickets": serialized_tickets,
    }
    return json.dumps(payload, indent=indent)


def export_board_json(
    date: str,
    rows: Sequence[Any],
    min_prob: float = 0.0,
    with_reasoning: bool = False,
    cards: dict[str, ReasoningCard] | None = None,
    verdicts: dict[str, GovernanceVerdict] | None = None,
    *,
    indent: int = 2,
) -> str:
    """Serialize moneyline board entries, reasoning cards, and governance verdicts to JSON."""
    cards_map = cards or {}
    verdicts_map = verdicts or {}

    entries: list[dict[str, Any]] = []
    card_records: list[dict[str, Any]] = []

    for r in rows:
        prob = r["prob_shin"] or r["prob_multiplicative"]
        entry_key = f"{r['game_id']}:ML:{r['side']}"
        card = cards_map.get(entry_key)
        verdict = verdicts_map.get(entry_key)

        entry_dict = {
            "game_id": r["game_id"],
            "kickoff_utc": r["kickoff_utc"] if "kickoff_utc" in r.keys() else "",
            "home": r["home"] if "home" in r.keys() else "",
            "away": r["away"] if "away" in r.keys() else "",
            "side": r["side"],
            "consensus_price": r["consensus_price"],
            "best_price": r["best_price"],
            "best_book": r["best_book"],
            "prob_shin": r["prob_shin"],
            "prob_multiplicative": r["prob_multiplicative"],
            "prob_spread": r["prob_spread"],
            "hold": r["hold"],
            "n_books": r["n_books"],
            "flags": json.loads(r["flags"] or "[]")
            if ("flags" in r.keys() and isinstance(r["flags"], str))
            else [],
            "reasoning_card": card.to_dict() if hasattr(card, "to_dict") else None,
            "governance_verdict": verdict.to_dict() if verdict else None,
            "disclaimer": SHADOW_MODE_DISCLAIMER,
        }
        entries.append(entry_dict)

        if card is not None:
            c_dict: dict[str, Any] = {
                "game_id": card.game_id,
                "recommended_play": card.recommended_play,
                "confidence": card.confidence,
                "tier": card.tier,
                "tape_summary": card.tape_summary,
                "position_qb_summary": card.position_qb_summary,
                "weather_venue_summary": card.weather_venue_summary,
                "injuries_trench_summary": card.injuries_trench_summary,
                "program_continuity_summary": card.program_continuity_summary,
                "mathematical_edge_summary": card.mathematical_edge_summary,
                "contra_indications": list(card.contra_indications),
                "is_favorite_vetoed": card.is_favorite_vetoed,
                "counter_thesis": card.counter_thesis,
                "verdict": verdict.to_dict() if verdict else None,
                "disclaimer": SHADOW_MODE_DISCLAIMER,
            }
            card_records.append(c_dict)

    payload = {
        "date": date,
        "disclaimer": SHADOW_MODE_DISCLAIMER,
        "stamp": SHADOW_MODE_DISCLAIMER,
        "min_prob": min_prob,
        "with_reasoning": with_reasoning,
        "entries": entries,
        "cards": card_records,
    }
    return json.dumps(payload, indent=indent)
