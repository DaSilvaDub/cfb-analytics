"""Moneyline Parlay Optimization Engine (Milestone 3).

Constructs, scores, and optimizes 3-to-10 leg moneyline parlays with:
- Non-linear same-conference, weather volatility, QB news risk, and heavy public favorite penalties.
- Bounded Fragility Index (Phi in [0.0, 1.0]) and Efficiency Ratio (rho = EV / Phi).
- Marginal leg value analysis and parasitic leg pruning.
- Deterministic multi-objective slate search (SAFEST, HIGHEST_EV, BEST_RISK_REWARD, and per-size optimal boards).

All algorithms strictly adhere to:
- Pure Python standard library primitives (math, dataclasses, itertools, typing).
- Absolute immutability via frozen dataclasses and tuples.
- Mandatory Shadow Mode disclaimer watermark.
"""

from __future__ import annotations

from enum import Enum
import itertools
from typing import Any, Sequence

from cfb_analytics.governance.models import (
    SHADOW_MODE_DISCLAIMER,
    MarginalLegAnalysis,
    ParlayLeg,
    ParlaySlateSummary,
    ParlayTicket,
    compute_correlation_penalty,
    compute_efficiency_ratio,
    compute_fragility_index,
    compute_parlay_payout,
)


class OptimizerMode(str, Enum):
    """Target objective for multi-leg parlay recommendation."""

    SAFEST = "SAFEST"                    # Maximizes joint win probability with positive EV
    HIGHEST_EV = "HIGHEST_EV"            # Maximizes expected return subject to fragility <= 0.60
    BEST_RISK_REWARD = "BEST_RISK_REWARD" # Maximizes efficiency ratio (EV / Fragility)


def _to_parlay_leg(leg: Any) -> ParlayLeg:
    """Adapt any leg-like object into a canonical immutable ParlayLeg."""
    if isinstance(leg, ParlayLeg):
        return leg
    candidate_id = getattr(leg, "candidate_id", f"{getattr(leg, 'game_id', 'g')}:ML:{getattr(leg, 'team_name', getattr(leg, 'team', 'T'))}")
    game_id = getattr(leg, "game_id", "cfbd:0000")
    team = getattr(leg, "team_name", getattr(leg, "team", "Team"))
    opponent = getattr(leg, "opponent_name", getattr(leg, "opponent", "Opponent"))
    market_type = getattr(leg, "market_type", "ML")
    line = getattr(leg, "line", 0.0)
    odds_american = getattr(leg, "price_american", getattr(leg, "odds_american", -110))
    fair_prob = getattr(leg, "model_prob", getattr(leg, "fair_prob", getattr(leg, "consensus_fair_prob", 0.50)))
    edge_pct = getattr(leg, "edge_pct", 0.05)
    conference = getattr(leg, "conference", "")
    weather_hazard = getattr(leg, "weather_hazard", False)
    if not weather_hazard:
        wind = getattr(leg, "effective_wind_kph", 0.0)
        precip = getattr(leg, "precip_mm", 0.0)
        is_dome = getattr(leg, "is_dome", False)
        if not is_dome and (wind >= 29.0 or precip >= 2.5):
            weather_hazard = True
    qb_news_risk = getattr(leg, "qb_news_risk", not getattr(leg, "qb_confirmed", True))
    is_heavy_fav = getattr(leg, "is_heavy_favorite", odds_american <= -600)

    return ParlayLeg(
        candidate_id=candidate_id,
        game_id=game_id,
        team=team,
        opponent=opponent,
        market_type=market_type,
        line=line,
        odds_american=odds_american,
        fair_prob=fair_prob,
        edge_pct=edge_pct,
        conference=conference,
        weather_hazard=weather_hazard,
        qb_news_risk=qb_news_risk,
        is_heavy_favorite=is_heavy_fav,
    )


class ParlayOptimizer:
    """Combinatorial optimization engine for multi-leg moneyline tickets."""

    def __init__(
        self,
        min_legs: int = 3,
        max_legs: int = 10,
        max_pool_size: int = 16,
    ) -> None:
        self.min_legs = min_legs
        self.max_legs = max_legs
        self.max_pool_size = max_pool_size

    def evaluate_ticket(
        self,
        legs: Sequence[ParlayLeg],
        parlay_id: str = "parlay:eval",
        compute_marginal: bool = True,
    ) -> ParlayTicket:
        """Evaluate a specific combination of legs and return a complete ParlayTicket."""
        leg_tuple = tuple(_to_parlay_leg(l) for l in legs)
        k = len(leg_tuple)

        # 1. Payout Calculation
        odds_american, mult = compute_parlay_payout(leg_tuple)

        # 2. Raw & Adjusted Joint Probability
        raw_prob = 1.0
        for leg in leg_tuple:
            raw_prob *= leg.fair_prob
        raw_prob = round(raw_prob, 6)

        penalty = compute_correlation_penalty(leg_tuple)
        adj_prob = max(0.0, raw_prob * max(0.0, 1.0 - penalty))
        adj_prob = round(adj_prob, 6)

        # 3. Expected Value & Fragility
        ev = round((adj_prob * mult) - 1.0, 4)
        fragility = compute_fragility_index(leg_tuple, adj_prob)
        efficiency = compute_efficiency_ratio(ev, fragility)

        # 4. Marginal Leg Analysis & Parasitic Pruning
        marginal_analyses: list[MarginalLegAnalysis] = []
        alternate_ticket: ParlayTicket | None = None

        if compute_marginal and k >= 3:
            for idx, leg in enumerate(leg_tuple):
                # Sub-ticket without leg k
                sub_legs = leg_tuple[:idx] + leg_tuple[idx + 1 :]
                sub_ticket = self.evaluate_ticket(sub_legs, parlay_id=f"{parlay_id}:sub_{idx}", compute_marginal=False)

                m_ev = round(ev - sub_ticket.ev, 4)
                d_frag = round(fragility - sub_ticket.fragility_index, 4)
                # A leg is parasitic if adding it reduces or fails to improve net EV
                is_parasitic = m_ev <= 0.0

                marginal_analyses.append(
                    MarginalLegAnalysis(
                        leg_index=idx,
                        candidate_id=leg.candidate_id,
                        team=leg.team,
                        prob_before=sub_ticket.joint_win_prob,
                        prob_after=adj_prob,
                        payout_before=sub_ticket.total_payout_multiplier,
                        payout_after=mult,
                        marginal_ev=m_ev,
                        delta_fragility=d_frag,
                        is_parasitic=is_parasitic,
                    )
                )

            # Check if any leg is parasitic or if dropping a leg strictly improves efficiency
            if k >= 4:
                parasitic_indices = [m.leg_index for m in marginal_analyses if m.is_parasitic]
                if parasitic_indices:
                    # Drop the worst parasitic leg (lowest marginal EV)
                    worst_idx = min(parasitic_indices, key=lambda i: marginal_analyses[i].marginal_ev)
                    best_sub_legs = leg_tuple[:worst_idx] + leg_tuple[worst_idx + 1 :]
                    alternate_ticket = self.evaluate_ticket(best_sub_legs, parlay_id=f"{parlay_id}:pruned", compute_marginal=True)

        return ParlayTicket(
            parlay_id=parlay_id,
            legs=leg_tuple,
            leg_count=k,
            total_odds_american=odds_american,
            total_payout_multiplier=mult,
            joint_win_prob=adj_prob,
            raw_win_prob=raw_prob,
            correlation_penalty=penalty,
            ev=ev,
            fragility_index=fragility,
            efficiency_ratio=efficiency,
            marginal_analyses=tuple(marginal_analyses),
            alternate_pruned_ticket=alternate_ticket,
            disclaimer=SHADOW_MODE_DISCLAIMER,
        )

    def optimize_parlays(
        self,
        legs: Sequence[Any],
        target_count: int | None = None,
    ) -> list[ParlayTicket]:
        """Construct and optimize parlays from input legs.

        Enforces strict 3 to 10 leg sizing.
        Raises ValueError if input legs < 3.
        """
        parsed_legs = [_to_parlay_leg(l) for l in legs]
        if len(parsed_legs) < self.min_legs:
            raise ValueError(f"Parlay must contain between {self.min_legs} and {self.max_legs} legs, got {len(parsed_legs)}")

        # Deduplicate games: at most one side per game_id
        game_map: dict[str, ParlayLeg] = {}
        for leg in parsed_legs:
            gid = leg.game_id
            if gid not in game_map or leg.fair_prob > game_map[gid].fair_prob:
                game_map[gid] = leg
        unique_legs = list(game_map.values())

        if len(unique_legs) < self.min_legs:
            raise ValueError(f"Parlay must contain between {self.min_legs} and {self.max_legs} unique game legs, got {len(unique_legs)}")

        # Sort by individual quality score (fair_prob + edge_pct)
        unique_legs.sort(key=lambda l: (l.fair_prob + l.edge_pct), reverse=True)
        pool = unique_legs[: self.max_pool_size]

        target_sizes = [target_count] if target_count is not None else list(range(self.min_legs, min(len(pool), self.max_legs) + 1))

        results: list[ParlayTicket] = []
        ticket_counter = 1

        for k in target_sizes:
            if not (self.min_legs <= k <= self.max_legs):
                continue
            for combo in itertools.combinations(pool, k):
                # Fast Pruning Check 1: Max 3 teams from same conference
                conf_counts: dict[str, int] = {}
                prune = False
                for leg in combo:
                    c = leg.conference.strip().lower()
                    if c and c not in ("independent", "fcs", "unknown", ""):
                        conf_counts[c] = conf_counts.get(c, 0) + 1
                        if conf_counts[c] >= 4:
                            prune = True
                            break
                if prune:
                    continue

                # Fast Pruning Check 2: Max 1 unconfirmed QB
                unconfirmed_qb_count = sum(1 for leg in combo if leg.qb_news_risk)
                if unconfirmed_qb_count >= 2:
                    continue

                ticket = self.evaluate_ticket(combo, parlay_id=f"parlay:{k}leg:{ticket_counter:03d}")
                results.append(ticket)
                ticket_counter += 1

        # Sort by efficiency ratio descending
        results.sort(key=lambda t: t.efficiency_ratio, reverse=True)
        return results

    def find_optimal_tickets(
        self,
        candidates: Sequence[Any],
        slate_date: str = "2026-09-26",
    ) -> ParlaySlateSummary:
        """Execute complete multi-objective combinatorial search across slate candidates."""
        parsed = [_to_parlay_leg(c) for c in candidates]
        n_eval = len(parsed)

        # Deduplicate games
        game_map: dict[str, ParlayLeg] = {}
        for leg in parsed:
            gid = leg.game_id
            if gid not in game_map or leg.fair_prob > game_map[gid].fair_prob:
                game_map[gid] = leg
        qualified_legs = list(game_map.values())
        n_qual = len(qualified_legs)

        if n_qual < self.min_legs:
            return ParlaySlateSummary(
                slate_date=slate_date,
                candidates_evaluated=n_eval,
                qualified_legs_count=n_qual,
                total_combinations_screened=0,
                safest_parlay=None,
                highest_ev_parlay=None,
                best_risk_reward_parlay=None,
                optimal_by_size=(),
                unjustified_sizes=tuple(range(self.min_legs, self.max_legs + 1)),
                status="INSUFFICIENT_QUALIFIED_LEGS",
                disclaimer=SHADOW_MODE_DISCLAIMER,
            )

        try:
            tickets = self.optimize_parlays(qualified_legs)
        except ValueError:
            return ParlaySlateSummary(
                slate_date=slate_date,
                candidates_evaluated=n_eval,
                qualified_legs_count=n_qual,
                total_combinations_screened=0,
                safest_parlay=None,
                highest_ev_parlay=None,
                best_risk_reward_parlay=None,
                optimal_by_size=(),
                unjustified_sizes=tuple(range(self.min_legs, self.max_legs + 1)),
                status="INSUFFICIENT_QUALIFIED_LEGS",
                disclaimer=SHADOW_MODE_DISCLAIMER,
            )

        if not tickets:
            return ParlaySlateSummary(
                slate_date=slate_date,
                candidates_evaluated=n_eval,
                qualified_legs_count=n_qual,
                total_combinations_screened=0,
                safest_parlay=None,
                highest_ev_parlay=None,
                best_risk_reward_parlay=None,
                optimal_by_size=(),
                status="NO_POSITIVE_EV",
                disclaimer=SHADOW_MODE_DISCLAIMER,
            )

        # 1. Safest Parlay: Highest joint win probability with EV > 0 (or highest joint prob)
        pos_ev_tickets = [t for t in tickets if t.ev > 0.0]
        pool_for_safest = pos_ev_tickets if pos_ev_tickets else tickets
        safest = max(pool_for_safest, key=lambda t: t.joint_win_prob)

        # 2. Highest EV: Max EV subject to Fragility <= 0.60 and prob >= 0.10
        controlled_fragility = [t for t in tickets if t.fragility_index <= 0.60 and t.joint_win_prob >= 0.10]
        if controlled_fragility:
            highest_ev = max(controlled_fragility, key=lambda t: t.ev)
        else:
            highest_ev = max(tickets, key=lambda t: t.ev)

        # 3. Best Risk/Reward: Max Efficiency Ratio (EV / Fragility)
        best_risk_reward = max(tickets, key=lambda t: t.efficiency_ratio)

        # 4. Per-Size Optimal Board
        by_size: dict[int, ParlayTicket] = {}
        for t in tickets:
            if t.leg_count not in by_size or t.efficiency_ratio > by_size[t.leg_count].efficiency_ratio:
                by_size[t.leg_count] = t
        optimal_sizes = tuple(by_size[k] for k in sorted(by_size.keys()))

        unjustified = tuple(k for k in range(self.min_legs, self.max_legs + 1) if k not in by_size)

        return ParlaySlateSummary(
            slate_date=slate_date,
            candidates_evaluated=n_eval,
            qualified_legs_count=n_qual,
            total_combinations_screened=len(tickets),
            safest_parlay=safest,
            highest_ev_parlay=highest_ev,
            best_risk_reward_parlay=best_risk_reward,
            optimal_by_size=optimal_sizes,
            unjustified_sizes=unjustified,
            status="SUCCESS",
            disclaimer=SHADOW_MODE_DISCLAIMER,
        )
