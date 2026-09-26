"""Governance Gate Orchestrator (Milestone 3).

Integrates:
- Quantitative opportunities from Mispriced Line Scanner (Milestone 2)
- 6-dimensional situational context and dossiers from Reasoning Engine (Milestone 1)
- Negative Favorite Gate counter-theses and anti-chalk gating
- Grok Heuristic Rules A through F execution and arbitration
- Moneyline Parlay qualification and provisioning for ParlayOptimizer

All components operate strictly in Shadow Mode:
UNPROMOTED - shadow output, not decision-grade.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cfb_analytics.governance.models import (
    SHADOW_MODE_DISCLAIMER,
    GovernanceAction,
    GovernanceVerdict,
    ParlayLeg,
    ParlayTicket,
)
from cfb_analytics.governance.parlay import ParlayOptimizer
from cfb_analytics.governance.rules import (
    GrokRuleEngine,
    _to_candidate_wager,
)
from cfb_analytics.reasoning.models import (
    ReasoningCard,
    SituationalContext,
)
from cfb_analytics.reasoning.negative_gate import NegativeFavoriteGate


class GovernanceGate:
    """Central orchestrator for Grok rule gating, negative favorite enforcement,
    and parlay eligibility provisioning."""

    def __init__(
        self,
        rules_engine: GrokRuleEngine | None = None,
        negative_gate: NegativeFavoriteGate | None = None,
        min_actionable_confidence: float = 6.0,
        parlay_min_confidence: float = 7.0,
    ) -> None:
        self.rules_engine = rules_engine if rules_engine is not None else GrokRuleEngine()
        self.negative_gate = negative_gate if negative_gate is not None else NegativeFavoriteGate()
        self.min_actionable_confidence = min_actionable_confidence
        self.parlay_min_confidence = parlay_min_confidence

    def evaluate_candidate(
        self,
        candidate: Any,
        reasoning_card: ReasoningCard | None = None,
        context: SituationalContext | None = None,
    ) -> GovernanceVerdict:
        """Evaluate a single candidate opportunity through the complete governance lifecycle."""
        cw = _to_candidate_wager(candidate)

        # Baseline original confidence
        if reasoning_card is not None:
            original_confidence = reasoning_card.confidence
        elif hasattr(candidate, "play_score"):
            original_confidence = min(10.0, max(0.0, float(candidate.play_score) / 10.0))
        else:
            original_confidence = cw.confidence

        triggered_rules: list[str] = []
        counter_theses: list[str] = []
        notes_parts: list[str] = []
        confidence_deltas: list[float] = []

        is_neg_vetoed = False
        is_neg_downgraded = False

        # 1. Baseline Sanity Check on MispricedOpportunity
        qual_status = getattr(candidate, "qual_status", "QUALIFIED")
        if qual_status in ("INSUFFICIENT_DATA", "STALE"):
            return GovernanceVerdict(
                candidate_id=cw.candidate_id,
                action=GovernanceAction.PASS,
                triggered_rules=(),
                counter_theses=(),
                original_confidence=original_confidence,
                adjusted_confidence=min(original_confidence, 4.0),
                parlay_eligible=False,
                target_market_override=None,
                notes=f"Qualification rejected: {qual_status}",
                disclaimer=SHADOW_MODE_DISCLAIMER,
            )

        if getattr(candidate, "edge_pct", 0.05) <= 0.0 or getattr(candidate, "ev", 0.05) <= 0.0:
            return GovernanceVerdict(
                candidate_id=cw.candidate_id,
                action=GovernanceAction.PASS,
                triggered_rules=(),
                counter_theses=(),
                original_confidence=original_confidence,
                adjusted_confidence=min(original_confidence, 4.0),
                parlay_eligible=False,
                target_market_override=None,
                notes="Non-positive edge or EV.",
                disclaimer=SHADOW_MODE_DISCLAIMER,
            )

        # 2. Reasoning Card Negative Gate Inspection
        if reasoning_card is not None:
            if reasoning_card.is_favorite_vetoed:
                is_neg_vetoed = True
                triggered_rules.append("Negative Favorite Gate")
                if reasoning_card.counter_thesis:
                    counter_theses.append(reasoning_card.counter_thesis)
                notes_parts.append("Reasoning dossier vetoed favorite.")
                confidence_deltas.append(-3.0)

            for contra in getattr(reasoning_card, "contra_indications", ()):
                if "fatigue" in contra.lower() or "trap" in contra.lower() or "attrition" in contra.lower():
                    notes_parts.append(contra)

        # 3. Dynamic Negative Favorite Gate Evaluation
        if context is not None and not is_neg_vetoed:
            is_fav = (
                (cw.market_type == "SPREAD" and cw.line < 0)
                or (cw.market_type == "ML" and (cw.odds_american < 0 or cw.fair_prob > 0.50))
            )
            if is_fav:
                neg_res = self.negative_gate.evaluate(
                    context,
                    market=cw.market,
                    side=cw.side,
                    line=cw.line,
                    model_prob=cw.fair_prob,
                    consensus_fair_prob=cw.fair_prob,
                    price_american=cw.odds_american,
                )
                if neg_res.is_vetoed:
                    is_neg_vetoed = True
                    triggered_rules.append("Negative Favorite Gate")
                    if neg_res.counter_thesis:
                        counter_theses.append(neg_res.counter_thesis)
                    confidence_deltas.append(-neg_res.confidence_penalty)
                elif neg_res.gate_status == "FLAGGED":
                    is_neg_downgraded = True
                    triggered_rules.append("Negative Gate (Flagged)")
                    confidence_deltas.append(-neg_res.confidence_penalty)

        # 4. Sequential Grok Rules Execution (Rules A through F)
        target_override: str | None = None
        has_grok_veto = False
        has_grok_downgrade = False
        has_grok_pass = False

        if context is not None:
            grok_results = self.rules_engine.evaluate_rules(cw, context, reasoning_card)
            for gr in grok_results:
                triggered_rules.append(gr.rule_id)
                confidence_deltas.append(gr.confidence_delta)
                if gr.counter_thesis:
                    counter_theses.append(gr.counter_thesis)
                if gr.target_market_override and not target_override:
                    target_override = gr.target_market_override
                if gr.notes:
                    notes_parts.append(gr.notes)

                if gr.action == GovernanceAction.VETO:
                    has_grok_veto = True
                elif gr.action == GovernanceAction.DOWNGRADE:
                    has_grok_downgrade = True
                elif gr.action == GovernanceAction.PASS:
                    has_grok_pass = True

        # 5. Multi-Rule Action Precedence Arbitration (Strict: VETO > DOWNGRADE > PASS > APPROVE)
        if is_neg_vetoed or has_grok_veto:
            final_action = GovernanceAction.VETO
        elif is_neg_downgraded or has_grok_downgrade:
            final_action = GovernanceAction.DOWNGRADE
        elif has_grok_pass:
            final_action = GovernanceAction.PASS
        else:
            final_action = GovernanceAction.APPROVE

        # 6. Confidence Adjustment Calculation
        total_delta = sum(confidence_deltas)
        adjusted_confidence = min(10.0, max(0.0, original_confidence + total_delta))
        if final_action == GovernanceAction.VETO:
            adjusted_confidence = min(adjusted_confidence, 4.0)
        elif final_action == GovernanceAction.DOWNGRADE:
            adjusted_confidence = min(adjusted_confidence, 6.5)

        # 7. Parlay Eligibility Gate
        parlay_eligible = (
            cw.market_type == "ML"
            and final_action == GovernanceAction.APPROVE
            and adjusted_confidence >= self.parlay_min_confidence
            and len(counter_theses) == 0
        )

        notes = " | ".join(notes_parts) if notes_parts else "Cleared governance: No contra-indications."

        return GovernanceVerdict(
            candidate_id=cw.candidate_id,
            action=final_action,
            triggered_rules=tuple(dict.fromkeys(triggered_rules)),  # deduplicate preserving order
            counter_theses=tuple(counter_theses),
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
            parlay_eligible=parlay_eligible,
            target_market_override=target_override,
            notes=notes,
            disclaimer=SHADOW_MODE_DISCLAIMER,
        )

    def evaluate_slate(
        self,
        candidates: Sequence[Any],
        reasoning_cards: Mapping[str, ReasoningCard] | None = None,
        contexts: Mapping[str, SituationalContext] | None = None,
    ) -> list[GovernanceVerdict]:
        """Batch evaluate a full slate of candidate opportunities."""
        verdicts: list[GovernanceVerdict] = []
        for cand in candidates:
            gid = getattr(cand, "game_id", "")
            card = reasoning_cards.get(gid) if reasoning_cards else None
            ctx = contexts.get(gid) if contexts else None
            verdict = self.evaluate_candidate(cand, reasoning_card=card, context=ctx)
            verdicts.append(verdict)
        return verdicts

    def get_parlay_qualified_candidates(
        self,
        candidates: Sequence[Any],
        verdicts: Sequence[GovernanceVerdict] | Mapping[str, GovernanceVerdict],
        contexts: Mapping[str, SituationalContext] | None = None,
    ) -> list[ParlayLeg]:
        """Filter approved moneyline candidates into eligible ParlayLeg records."""
        # Index verdicts by candidate_id
        if isinstance(verdicts, Mapping):
            verdict_map = verdicts
        else:
            verdict_map = {v.candidate_id: v for v in verdicts}

        qualified_legs: list[ParlayLeg] = []

        for cand in candidates:
            cid = getattr(cand, "candidate_id", None)
            if not cid:
                cid = f"{cand.game_id}:{cand.market_type}:{cand.side}"

            verdict = verdict_map.get(cid)
            if verdict is None or not verdict.parlay_eligible:
                continue

            # Strictly Moneylines
            m_type = getattr(cand, "market_type", getattr(cand, "market", ""))
            if m_type != "ML":
                continue

            gid = cand.game_id
            ctx = contexts.get(gid) if contexts else None

            # Resolve team and opponent names
            side = cand.side.upper()
            if ctx is not None:
                team = ctx.home_team if side == "HOME" else ctx.away_team
                opponent = ctx.away_team if side == "HOME" else ctx.home_team
                conf = getattr(ctx, "conference", "")
                is_dome = getattr(ctx, "is_dome", getattr(ctx.weather, "is_dome", False))
                wind = getattr(ctx.weather, "wind_kph", 0.0)
                precip = getattr(ctx.weather, "precip_mm", 0.0)
                weather_hazard = (not is_dome) and (wind >= 29.0 or precip >= 2.5)
                qb_risk = not (ctx.qb_home_confirmed if side == "HOME" else ctx.qb_away_confirmed)
            else:
                team = getattr(cand, "team", side)
                opponent = getattr(cand, "opponent", "Opponent")
                conf = getattr(cand, "conference", "")
                weather_hazard = getattr(cand, "weather_hazard", False)
                qb_risk = getattr(cand, "qb_news_risk", False)

            odds_american = getattr(cand, "posted_price_american", -110)
            fair_prob = getattr(cand, "consensus_fair_prob", getattr(cand, "model_prob", 0.50))
            edge_pct = getattr(cand, "edge_pct", 0.05)

            qualified_legs.append(
                ParlayLeg(
                    candidate_id=verdict.candidate_id,
                    game_id=gid,
                    team=team,
                    opponent=opponent,
                    market_type="ML",
                    line=getattr(cand, "line", 0.0),
                    odds_american=odds_american,
                    fair_prob=fair_prob,
                    edge_pct=edge_pct,
                    conference=conf,
                    weather_hazard=weather_hazard,
                    qb_news_risk=qb_risk,
                    is_heavy_favorite=odds_american <= -600,
                )
            )

        return qualified_legs

    def provision_parlays(
        self,
        candidates: Sequence[Any],
        verdicts: Sequence[GovernanceVerdict] | Mapping[str, GovernanceVerdict],
        contexts: Mapping[str, SituationalContext] | None = None,
        optimizer: ParlayOptimizer | None = None,
    ) -> list[ParlayTicket]:
        """Convert eligible candidates and optimize multi-leg parlays."""
        legs = self.get_parlay_qualified_candidates(candidates, verdicts, contexts)
        if len(legs) < 3:
            return []
        opt = optimizer or ParlayOptimizer()
        return opt.optimize_parlays(legs)
