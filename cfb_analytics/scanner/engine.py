"""Mispriced Scanner Engine (Milestone 2).

Orchestrates multi-market scanning across Spreads, Totals, Team Props, and Moneylines:
- Computes vig-free fair prices using both Shin and Multiplicative methods (leveraging cfb_analytics.models.devig).
- Computes method spread = max(prob_shin, prob_mult) - min(prob_shin, prob_mult).
- Computes edge % = model_prob - consensus_fair_prob.
- Interfaces cleanly with ReasoningCard from cfb_analytics.reasoning.
- Ranks and filters actionable mispriced betting opportunities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cfb_analytics.errors import DevigError
from cfb_analytics.models.devig import (
    consensus_probabilities,
    devig,
)
from cfb_analytics.reasoning.models import ReasoningCard
from cfb_analytics.scanner.models import (
    SHADOW_MODE_DISCLAIMER,
    MispricedOpportunity,
    PlayTier,
    QualificationStatus,
)
from cfb_analytics.scanner.props import (
    PlayerPropProhibitedError,
    evaluate_team_prop_candidate,
)
from cfb_analytics.scanner.scoring import calculate_play_score
from cfb_analytics.scanner.spreads import (
    SPREAD_SIGMA_BASE,
    evaluate_spread_candidate,
)
from cfb_analytics.scanner.totals import (
    TOTAL_SIGMA_BASE,
    evaluate_total_candidate,
)
from cfb_analytics.utils import (
    american_to_decimal,
    decimal_to_american,
    implied_probability,
)


class MispricedScanner:
    """Orchestrator for detecting, devigging, and scoring mispriced college football lines."""

    def __init__(self, *, default_margin_sigma: float = SPREAD_SIGMA_BASE, default_total_sigma: float = TOTAL_SIGMA_BASE) -> None:
        self.margin_sigma = default_margin_sigma
        self.total_sigma = default_total_sigma

    def devig_two_way(
        self,
        price_side: int,
        price_opp: int,
    ) -> tuple[float, float, float]:
        """Devig a 2-way market using both Shin and Multiplicative methods.

        Returns:
            (prob_shin, prob_mult, method_spread)
        """
        p_side = implied_probability(price_side)
        p_opp = implied_probability(price_opp)
        if p_side is None or p_opp is None:
            raise DevigError(f"Invalid American odds: {price_side}, {price_opp}")

        quotes = [p_side, p_opp]
        fair_shin = devig(quotes, method="shin")
        fair_mult = devig(quotes, method="multiplicative")

        prob_shin = round(fair_shin[0], 4)
        prob_mult = round(fair_mult[0], 4)
        method_spread = round(abs(prob_shin - prob_mult), 4)
        return prob_shin, prob_mult, method_spread

    def devig_book_quotes(
        self,
        book_quotes: Mapping[str, Sequence[float | int]],
        side_index: int = 0,
    ) -> tuple[float, float, float]:
        """Devig multi-book quotes using both Shin and Multiplicative methods.

        Args:
            book_quotes: Mapping of book names to price lists (American odds or probabilities).
            side_index: Index of the outcome to extract (e.g. 0 for home/over, 1 for away/under).

        Returns:
            (prob_shin, prob_mult, method_spread)
        """
        if not book_quotes:
            raise DevigError("Empty book quotes provided")

        clean_quotes: dict[str, list[float]] = {}
        for book, quotes in book_quotes.items():
            parsed: list[float] = []
            for q in quotes:
                if isinstance(q, int) or (isinstance(q, float) and (q > 1.0 or q < 0.0)):
                    implied = implied_probability(q)
                    if implied is not None:
                        parsed.append(implied)
                else:
                    parsed.append(float(q))
            if len(parsed) >= 2:
                clean_quotes[book] = parsed

        if not clean_quotes:
            raise DevigError("No valid book quotes after parsing")

        res_shin = consensus_probabilities(clean_quotes, method="shin")
        res_mult = consensus_probabilities(clean_quotes, method="multiplicative")

        if side_index >= len(res_shin) or side_index >= len(res_mult):
            raise DevigError(f"side_index {side_index} out of bounds")

        prob_shin = round(res_shin[side_index], 4)
        prob_mult = round(res_mult[side_index], 4)
        method_spread = round(abs(prob_shin - prob_mult), 4)
        return prob_shin, prob_mult, method_spread

    def scan_spread(
        self,
        game_id: str,
        side: str,
        line: float,
        projected_margin: float,
        posted_price_american: int,
        consensus_fair_prob: float,
        *,
        market_spread: float | None = None,
        method_spread: float = 0.0,
        n_books: int = 3,
        reasoning_card: ReasoningCard | None = None,
        **kwargs: Any,
    ) -> MispricedOpportunity:
        """Scan and evaluate a Game Spread opportunity."""
        flags: tuple[str, ...] = kwargs.pop("flags", ())
        is_vetoed = False
        rejection_notes: list[str] = []

        if reasoning_card is not None:
            if reasoning_card.is_favorite_vetoed and side.strip().upper() == reasoning_card.side.strip().upper():
                is_vetoed = True
                if reasoning_card.counter_thesis:
                    rejection_notes.append(f"counter_thesis: {reasoning_card.counter_thesis}")
            if reasoning_card.contra_indications:
                flags = tuple(list(flags) + list(reasoning_card.contra_indications))
            if "confirming_insights_count" not in kwargs:
                # Estimate confirming insights from non-empty card sections
                conf_count = sum(
                    1
                    for s in (
                        reasoning_card.tape_summary,
                        reasoning_card.position_qb_summary,
                        reasoning_card.weather_venue_summary,
                        reasoning_card.injuries_trench_summary,
                        reasoning_card.program_continuity_summary,
                    )
                    if s and "pass" not in s.lower() and "unfavorable" not in s.lower()
                )
                kwargs["confirming_insights_count"] = conf_count

        opp = evaluate_spread_candidate(
            game_id=game_id,
            side=side,
            line=line,
            projected_margin=projected_margin,
            posted_price_american=posted_price_american,
            consensus_fair_prob=consensus_fair_prob,
            market_spread=market_spread,
            sigma=self.margin_sigma,
            method_spread=method_spread,
            n_books=n_books,
            flags=flags,
            **kwargs,
        )

        if is_vetoed:
            return self._apply_reasoning_veto(opp, reasoning_card)
        return opp

    def scan_total(
        self,
        game_id: str,
        side: str,
        line: float,
        projected_total: float,
        posted_price_american: int,
        consensus_fair_prob: float,
        *,
        method_spread: float = 0.0,
        n_books: int = 3,
        reasoning_card: ReasoningCard | None = None,
        **kwargs: Any,
    ) -> MispricedOpportunity:
        """Scan and evaluate a Game Total opportunity."""
        flags: tuple[str, ...] = kwargs.pop("flags", ())
        if reasoning_card is not None:
            if reasoning_card.contra_indications:
                flags = tuple(list(flags) + list(reasoning_card.contra_indications))

        return evaluate_total_candidate(
            game_id=game_id,
            side=side,
            line=line,
            projected_total=projected_total,
            posted_price_american=posted_price_american,
            consensus_fair_prob=consensus_fair_prob,
            sigma=self.total_sigma,
            method_spread=method_spread,
            n_books=n_books,
            flags=flags,
            **kwargs,
        )

    def scan_team_prop(
        self,
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
        method_spread: float = 0.0,
        n_books: int = 3,
        reasoning_card: ReasoningCard | None = None,
        **kwargs: Any,
    ) -> MispricedOpportunity | None:
        """Scan and evaluate a Team Prop opportunity (strictly team props)."""
        flags: tuple[str, ...] = kwargs.pop("flags", ())
        if reasoning_card is not None:
            if reasoning_card.contra_indications:
                flags = tuple(list(flags) + list(reasoning_card.contra_indications))

        return evaluate_team_prop_candidate(
            game_id=game_id,
            market=market,
            side=side,
            line=line,
            projected_value=projected_value,
            posted_price_american=posted_price_american,
            consensus_fair_prob=consensus_fair_prob,
            market_type=market_type,
            spread=spread,
            is_favorite=is_favorite,
            drop_sit_qb=drop_sit_qb,
            method_spread=method_spread,
            n_books=n_books,
            flags=flags,
            **kwargs,
        )

    def scan_moneyline(
        self,
        game_id: str,
        side: str,
        posted_price_american: int,
        model_prob: float,
        consensus_fair_prob: float,
        *,
        method_spread: float = 0.0,
        n_books: int = 3,
        reasoning_card: ReasoningCard | None = None,
        **kwargs: Any,
    ) -> MispricedOpportunity:
        """Scan and evaluate a Moneyline opportunity."""
        side_clean = side.strip().upper()
        if side_clean not in ("HOME", "AWAY"):
            raise ValueError(f"side must be 'HOME' or 'AWAY', got {side!r}")

        prob_edge = round(model_prob - consensus_fair_prob, 4)
        decimal_price = american_to_decimal(posted_price_american) or 1.0
        ev = round(model_prob * decimal_price - 1.0, 4)

        if 0.0 < consensus_fair_prob < 1.0:
            fair_american = decimal_to_american(1.0 / consensus_fair_prob) or 0
        else:
            fair_american = 0

        flags: tuple[str, ...] = kwargs.pop("flags", ())
        is_vetoed = False
        if reasoning_card is not None:
            if reasoning_card.is_favorite_vetoed and side_clean == reasoning_card.side.strip().upper():
                is_vetoed = True
            if reasoning_card.contra_indications:
                flags = tuple(list(flags) + list(reasoning_card.contra_indications))

        play_score_obj = calculate_play_score(
            market_type="ML",
            edge_metric=prob_edge,
            model_prob=model_prob,
            posted_price_american=posted_price_american,
            side=side_clean,
            ev=ev,
            n_books=n_books,
            consensus_fair_prob=consensus_fair_prob,
            flags=flags,
            is_vetoed=is_vetoed,
            **kwargs,
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

        opp = MispricedOpportunity(
            game_id=game_id,
            market_type="ML",
            market="ML",
            side=side_clean,
            line=0.0,
            posted_price_american=posted_price_american,
            posted_price_decimal=round(decimal_price, 4),
            consensus_fair_prob=round(consensus_fair_prob, 4),
            consensus_fair_price_american=fair_american,
            model_projected_line=round(model_prob, 4),
            model_prob=round(model_prob, 4),
            edge_pct=prob_edge,
            ev=ev,
            method_spread=round(method_spread, 4),
            n_books=n_books,
            play_score=play_score_obj.total_score,
            play_tier=tier_str,
            qual_status=status_str,
            best_book=kwargs.get("best_book"),
            game_label=kwargs.get("game_label", ""),
            kickoff_et=kwargs.get("kickoff_et", ""),
            flags=flags,
            rejection_reasons=play_score_obj.rejection_reasons,
            shadow_mode_disclaimer=SHADOW_MODE_DISCLAIMER,
        )

        if is_vetoed:
            return self._apply_reasoning_veto(opp, reasoning_card)
        return opp

    def _apply_reasoning_veto(
        self,
        opp: MispricedOpportunity,
        card: ReasoningCard | None,
    ) -> MispricedOpportunity:
        """Downgrade candidate when reasoning card counter-thesis vetoes the favorite."""
        thesis_note = f"counter_thesis: {card.counter_thesis}" if card and card.counter_thesis else "vetoed_by_reasoning"
        new_rejections = tuple(list(opp.rejection_reasons) + [thesis_note, "favorite_vetoed"])
        new_flags = tuple(list(opp.flags) + ["vetoed_by_negative_gate"])

        return MispricedOpportunity(
            game_id=opp.game_id,
            market_type=opp.market_type,
            market=opp.market,
            side=opp.side,
            line=opp.line,
            posted_price_american=opp.posted_price_american,
            posted_price_decimal=opp.posted_price_decimal,
            consensus_fair_prob=opp.consensus_fair_prob,
            consensus_fair_price_american=opp.consensus_fair_price_american,
            model_projected_line=opp.model_projected_line,
            model_prob=opp.model_prob,
            edge_pct=opp.edge_pct,
            ev=opp.ev,
            method_spread=opp.method_spread,
            n_books=opp.n_books,
            play_score=opp.play_score,
            play_tier=PlayTier.AVOID.value,
            qual_status=QualificationStatus.REVIEW.value,
            best_book=opp.best_book,
            game_label=opp.game_label,
            kickoff_et=opp.kickoff_et,
            flags=new_flags,
            rejection_reasons=new_rejections,
            shadow_mode_disclaimer=SHADOW_MODE_DISCLAIMER,
        )

    def scan_slate(
        self,
        candidates: Sequence[dict[str, Any]],
        *,
        reasoning_cards: Mapping[str, ReasoningCard] | None = None,
        sort_by: str = "edge_pct",
    ) -> list[MispricedOpportunity]:
        """Scan a slate of market candidates, devigging and scoring each."""
        opportunities: list[MispricedOpportunity] = []
        cards_map = reasoning_cards or {}

        for c in candidates:
            market_type = c.get("market_type", "SPREAD").upper()
            game_id = c["game_id"]
            card_key = f"{game_id}:{market_type}:{c.get('side', '')}"
            card = cards_map.get(card_key) or cards_map.get(game_id)

            # Auto-compute Shin and Multiplicative devig if quotes or two-way prices are provided
            consensus_fair_prob = c.get("consensus_fair_prob")
            method_spread = c.get("method_spread", 0.0)

            if consensus_fair_prob is None:
                if "book_quotes" in c and c["book_quotes"]:
                    side_idx = 0 if c.get("side", "").upper() in ("HOME", "OVER") else 1
                    p_shin, p_mult, spread = self.devig_book_quotes(c["book_quotes"], side_index=side_idx)
                    consensus_fair_prob = p_shin
                    method_spread = spread
                elif "price_side" in c and "price_opp" in c:
                    p_shin, p_mult, spread = self.devig_two_way(c["price_side"], c["price_opp"])
                    consensus_fair_prob = p_shin
                    method_spread = spread

            if consensus_fair_prob is None:
                continue

            # Route by market type
            if market_type == "SPREAD":
                opp = self.scan_spread(
                    game_id=game_id,
                    side=c["side"],
                    line=c["line"],
                    projected_margin=c["projected_margin"],
                    posted_price_american=c["posted_price_american"],
                    consensus_fair_prob=consensus_fair_prob,
                    market_spread=c.get("market_spread"),
                    method_spread=method_spread,
                    n_books=c.get("n_books", 3),
                    reasoning_card=card,
                    historical_hit_rate=c.get("historical_hit_rate"),
                    sample_size=c.get("sample_size", 20),
                    confirming_insights_count=c.get("confirming_insights_count", 0),
                    open_line=c.get("open_line"),
                    current_line=c.get("current_line"),
                    rlm_flag=c.get("rlm_flag", False),
                    best_book=c.get("best_book"),
                    game_label=c.get("game_label", ""),
                    kickoff_et=c.get("kickoff_et", ""),
                    flags=tuple(c.get("flags", ())),
                    data_quality_score=c.get("data_quality_score", 100.0),
                )
                opportunities.append(opp)
            elif market_type == "TOTAL":
                opp = self.scan_total(
                    game_id=game_id,
                    side=c["side"],
                    line=c["line"],
                    projected_total=c["projected_total"],
                    posted_price_american=c["posted_price_american"],
                    consensus_fair_prob=consensus_fair_prob,
                    method_spread=method_spread,
                    n_books=c.get("n_books", 3),
                    reasoning_card=card,
                    historical_hit_rate=c.get("historical_hit_rate"),
                    sample_size=c.get("sample_size", 20),
                    confirming_insights_count=c.get("confirming_insights_count", 0),
                    open_line=c.get("open_line"),
                    current_line=c.get("current_line"),
                    rlm_flag=c.get("rlm_flag", False),
                    best_book=c.get("best_book"),
                    game_label=c.get("game_label", ""),
                    kickoff_et=c.get("kickoff_et", ""),
                    flags=tuple(c.get("flags", ())),
                    data_quality_score=c.get("data_quality_score", 100.0),
                )
                opportunities.append(opp)
            elif market_type == "TEAM_PROP":
                try:
                    opp = self.scan_team_prop(
                        game_id=game_id,
                        market=c["market"],
                        side=c["side"],
                        line=c["line"],
                        projected_value=c["projected_value"],
                        posted_price_american=c["posted_price_american"],
                        consensus_fair_prob=consensus_fair_prob,
                        market_type="TEAM_PROP",
                        spread=c.get("spread"),
                        is_favorite=c.get("is_favorite", True),
                        drop_sit_qb=c.get("drop_sit_qb", True),
                        method_spread=method_spread,
                        n_books=c.get("n_books", 3),
                        reasoning_card=card,
                        historical_hit_rate=c.get("historical_hit_rate"),
                        sample_size=c.get("sample_size", 20),
                        confirming_insights_count=c.get("confirming_insights_count", 0),
                        open_line=c.get("open_line"),
                        current_line=c.get("current_line"),
                        rlm_flag=c.get("rlm_flag", False),
                        best_book=c.get("best_book"),
                        game_label=c.get("game_label", ""),
                        kickoff_et=c.get("kickoff_et", ""),
                        flags=tuple(c.get("flags", ())),
                        data_quality_score=c.get("data_quality_score", 100.0),
                    )
                    if opp is not None:
                        opportunities.append(opp)
                except PlayerPropProhibitedError:
                    # Player props are strictly discarded or rejected
                    continue
            elif market_type == "ML":
                opp = self.scan_moneyline(
                    game_id=game_id,
                    side=c["side"],
                    posted_price_american=c["posted_price_american"],
                    model_prob=c["model_prob"],
                    consensus_fair_prob=consensus_fair_prob,
                    method_spread=method_spread,
                    n_books=c.get("n_books", 3),
                    reasoning_card=card,
                    historical_hit_rate=c.get("historical_hit_rate"),
                    sample_size=c.get("sample_size", 20),
                    confirming_insights_count=c.get("confirming_insights_count", 0),
                    open_line=c.get("open_line"),
                    current_line=c.get("current_line"),
                    rlm_flag=c.get("rlm_flag", False),
                    best_book=c.get("best_book"),
                    game_label=c.get("game_label", ""),
                    kickoff_et=c.get("kickoff_et", ""),
                    flags=tuple(c.get("flags", ())),
                    data_quality_score=c.get("data_quality_score", 100.0),
                )
                opportunities.append(opp)

        # Sort results
        if sort_by == "edge_pct":
            opportunities.sort(key=lambda o: -o.edge_pct)
        elif sort_by == "play_score":
            opportunities.sort(key=lambda o: -o.play_score)
        elif sort_by == "ev":
            opportunities.sort(key=lambda o: -o.ev)

        return opportunities

    def filter_actionable(
        self,
        opportunities: Sequence[MispricedOpportunity],
    ) -> list[MispricedOpportunity]:
        """Filter opportunities down to only actionable plays."""
        return [o for o in opportunities if o.is_actionable]
