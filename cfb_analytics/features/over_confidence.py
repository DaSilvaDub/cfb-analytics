"""Ranked OVER confidence for team/game rushing and receiving.

This is the Saturday board produced from the ranked slate: Top 25 games plus
each conference's top two. Script (blowout sit-QB vs trailing air raid) and
small-sample haircuts are applied to Model 3 projections. Player props are
never emitted.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from cfb_analytics import config
from cfb_analytics.errors import SchemaError
from cfb_analytics.features.ranked_slate import RankedSlateGame, select_ranked_slate
from cfb_analytics.models.team_props import TeamPropsInputs, project_team_production
from cfb_analytics.scoring import load_team_props_inputs_for_slate, norm_cdf
from cfb_analytics.utils import FOOTBALL_TZ, to_utc_iso, utc_now_iso

OVER_MARKETS = (
    "team_rushing_yards",
    "team_receiving_yards",
    "game_rushing_yards",
    "game_receiving_yards",
)


@dataclass(frozen=True)
class OverConfidencePick:
    """One OVER-side rushing or receiving candidate, ranked by model probability."""

    rank: int
    game_id: str
    game_label: str
    kickoff_et: str
    market: str
    pick: str
    team_id: str | None
    side: str
    line: float
    projected: float
    over_prob: float
    confidence: float
    tier: str
    inclusion_reasons: tuple[str, ...]
    flags: tuple[str, ...]
    family: str
    kickoff_utc: str


def format_kickoff_et(kickoff_utc: str) -> str:
    """US Eastern kickoff as ``4:00 p.m.``."""
    iso = to_utc_iso(kickoff_utc)
    if iso is None:
        raise SchemaError(f"Unparseable kickoff {kickoff_utc!r}")
    moment = datetime.fromisoformat(iso).astimezone(FOOTBALL_TZ)
    hour = moment.hour % 12 or 12
    suffix = "a.m." if moment.hour < 12 else "p.m."
    return f"{hour}:{moment.minute:02d} {suffix}"


_LEAGUE_YARDS = {
    "team_receiving_yards": 230.0,
    "team_rushing_yards": 155.0,
    "game_receiving_yards": 460.0,
    "game_rushing_yards": 310.0,
}
_LINE_OFFSET = {
    "team_receiving_yards": 12.0,
    "team_rushing_yards": 8.0,
    "game_receiving_yards": 20.0,
    "game_rushing_yards": 15.0,
}
_LINE_CAPS = {
    "team_receiving_yards": (145.5, 355.5),
    "team_rushing_yards": (85.5, 245.5),
    "game_receiving_yards": (320.5, 680.5),
    "game_rushing_yards": (200.5, 480.5),
}


SIT_QB_SPREAD = -20.0
BLOWOUT_GAME_YARDS = 28.0  # combined rush/rec overs die when one side is emptied
DOG_REC_BLOWOUT = 28.0
PROB_CAP = 0.84
HIGH_TIER = 0.70
MEDIUM_TIER = 0.60


def inferred_line(
    baseline: float,
    market: str = "team_receiving_yards",
    opp_allowed: float | None = None,
) -> float:
    """Book-like ``x.5`` mark from identity yards, not from the model projection.

    Sportsbooks regress early-season tape toward this opponent's yards allowed
    when known, else league average. Snapping the mark to the projection itself
    made every OVER a coin flip and buried SMU receiving in the 52% pile.
    """
    league = _LEAGUE_YARDS.get(market, 230.0)
    offset = _LINE_OFFSET.get(market, 12.0)
    defense = league if opp_allowed is None else opp_allowed
    raw = 0.46 * baseline + 0.40 * defense + offset
    snapped = max(0.5, math.floor((raw - 0.5) / 5.0) * 5 + 0.5)
    lo, hi = _LINE_CAPS.get(market, (0.5, 999.5))
    return min(hi, max(lo, snapped))


def matchup_project(scripted: float, opp_allowed: float | None) -> float:
    """Blend script-adjusted identity with opponent yards allowed."""
    if opp_allowed is None:
        return scripted
    return round(0.55 * scripted + 0.45 * opp_allowed, 1)


def calibrated_over_prob(market: str, projected: float, line: float, *, posted: bool) -> float:
    """OVER probability with a wider residual on inferred marks, capped at 84%.

    A 35-yard std on a 150-yard cupcake gap prints 99%. Early-season inferred
    marks are not that sharp.
    """
    if market in ("team_rushing_yards", "team_receiving_yards"):
        std = 35.0 if posted else max(55.0, 0.22 * line)
    else:
        std = 50.0 if posted else max(70.0, 0.16 * line)
    z = (projected - line) / std
    return round(max(0.01, min(PROB_CAP, float(norm_cdf(z)))), 4)


def assign_tier(over_prob: float) -> str:
    if over_prob >= HIGH_TIER:
        return "HIGH"
    if over_prob >= MEDIUM_TIER:
        return "MEDIUM"
    return "LEAN"


def over_confidence_from_prob(over_prob: float) -> float:
    """Map OVER probability onto the 1-10 session confidence scale."""
    return round(max(0.1, min(9.9, over_prob * 10.9)), 1)


def apply_script_adjustment(
    inputs: TeamPropsInputs,
    spread: float | None,
    *,
    opp_pass_allowed: float | None = None,
) -> TeamPropsInputs:
    """Shift rush/pass volume from the team's spread (negative = favorite).

    Close games do NOT automatically throw. SMU rec missed (222 vs 300.5) while
    combined rush cashed (522 vs 290.5). Pass volume only inflates when the
    offense is already pass-heavy AND the opponent leaks pass yards.
    """
    if spread is None:
        return inputs
    if spread <= -35:
        rush_m, pass_m = 1.10, 0.82
    elif spread <= -20:
        rush_m, pass_m = 1.06, 0.90
    elif spread <= -10:
        rush_m, pass_m = 1.03, 0.97
    elif spread >= 35:
        rush_m, pass_m = 0.78, 1.08
    elif spread >= 20:
        rush_m, pass_m = 0.88, 1.04
    elif abs(spread) <= 4:
        pass_share = inputs.expected_pass_attempts / max(
            1.0, inputs.expected_pass_attempts + inputs.expected_rushing_attempts
        )
        pass_heavy = pass_share >= 0.55 or inputs.expected_pass_attempts >= 36.0
        leaky = opp_pass_allowed is not None and opp_pass_allowed >= 250.0
        if pass_heavy and leaky:
            rush_m, pass_m = 0.96, 1.08
        else:
            rush_m, pass_m = 1.04, 1.00
    else:
        return inputs
    rush = round(max(12.0, min(65.0, inputs.expected_rushing_attempts * rush_m)), 1)
    passing = round(max(10.0, min(55.0, inputs.expected_pass_attempts * pass_m)), 1)
    return replace(
        inputs,
        expected_rushing_attempts=rush,
        expected_pass_attempts=passing,
        pace=round(rush + passing, 1),
    )


def apply_sample_haircut(inputs: TeamPropsInputs) -> TeamPropsInputs:
    """Discount cupcake-inflated early-season production (two-game samples)."""
    if inputs.data_quality_score >= 100.0:
        return inputs
    return replace(
        inputs,
        yards_per_completion=round(inputs.yards_per_completion * 0.94, 2),
        yards_before_contact=round(inputs.yards_before_contact * 0.94, 2),
        yards_after_contact=round(inputs.yards_after_contact * 0.94, 2),
    )


def _home_spread(conn: sqlite3.Connection, game_id: str) -> float | None:
    row = conn.execute(
        """SELECT line FROM market_consensus
           WHERE game_id = ? AND market = 'SPREAD' AND side = 'HOME' AND n_books >= 3
           ORDER BY n_books DESC, as_of_utc DESC LIMIT 1""",
        (game_id,),
    ).fetchone()
    if row is None or row["line"] is None:
        return None
    return float(row["line"])


def _posted_line(
    conn: sqlite3.Connection,
    game_id: str,
    team_id: str | None,
    market: str,
    *,
    cutoff_utc: str,
) -> tuple[float | None, int]:
    if team_id is None:
        return None, 0
    row = conn.execute(
        """SELECT line, n_books FROM team_prop_consensus
           WHERE game_id = ? AND team_id = ? AND market = ? AND side = 'OVER'
             AND as_of_utc <= ?
           ORDER BY n_books DESC, as_of_utc DESC LIMIT 1""",
        (game_id, team_id, market, cutoff_utc),
    ).fetchone()
    if row is not None and row["line"] is not None:
        return float(row["line"]), int(row["n_books"] or 0)
    snap = conn.execute(
        """WITH latest AS (
               SELECT MAX(captured_utc) AS captured_utc
               FROM team_prop_odds_snapshots
               WHERE game_id = ? AND team_id = ? AND market = ? AND side = 'OVER'
                 AND captured_utc <= ?
           )
           SELECT p.line, COUNT(DISTINCT p.book) AS n_books
           FROM team_prop_odds_snapshots AS p
           JOIN latest ON p.captured_utc = latest.captured_utc
           WHERE p.game_id = ? AND p.team_id = ? AND p.market = ? AND p.side = 'OVER'
           GROUP BY p.line
           HAVING n_books >= 2
           ORDER BY n_books DESC, p.line
           LIMIT 1""",
        (game_id, team_id, market, cutoff_utc, game_id, team_id, market),
    ).fetchone()
    if snap is None or snap["line"] is None:
        return None, 0
    return float(snap["line"]), int(snap["n_books"] or 0)


def classify_family(market: str, team_spread: float | None) -> str:
    if market in ("game_rushing_yards", "game_receiving_yards"):
        return "game_yards"
    if market == "team_receiving_yards":
        return "team_rec"
    if market == "team_rushing_yards" and team_spread is not None and team_spread < 0:
        return "favorite_team_rush"
    return "other_team_rush"


def _pass_yards_allowed(
    conn: sqlite3.Connection,
    opponent_id: str,
    *,
    cutoff_utc: str,
    season: int | None,
) -> float | None:
    query = """
        SELECT AVG(game_yds) AS allowed FROM (
            SELECT SUM(p.yards) AS game_yds
            FROM player_game_passing AS p
            JOIN games AS g ON g.game_id = p.game_id
            WHERE g.kickoff_utc < ?
              AND (g.home_team_id = ? OR g.away_team_id = ?)
              AND p.team_id != ?
    """
    params: list[object] = [cutoff_utc, opponent_id, opponent_id, opponent_id]
    if season is not None:
        query += " AND p.season = ?"
        params.append(season)
    query += " GROUP BY p.game_id)"
    row = conn.execute(query, params).fetchone()
    if row is None or row["allowed"] is None:
        return None
    return float(row["allowed"])


def _rush_yards_allowed_proxy(
    conn: sqlite3.Connection,
    opponent_id: str,
    *,
    cutoff_utc: str,
    season: int | None,
) -> float | None:
    query = """
        SELECT rushing_success_rate FROM team_season_advanced
        WHERE team_id = ? AND side = 'def' AND as_of_utc <= ?
    """
    params: list[object] = [opponent_id, cutoff_utc]
    if season is not None:
        query += " AND season = ?"
        params.append(season)
    query += " ORDER BY as_of_utc DESC LIMIT 1"
    row = conn.execute(query, params).fetchone()
    if row is None or row["rushing_success_rate"] is None:
        return None
    rate = float(row["rushing_success_rate"])
    if rate <= 0:
        return None
    return max(80.0, min(250.0, 155.0 * (rate / 0.42)))


def _sit_qb_receiving(market: str, team_spread: float | None, home_spread: float | None) -> bool:
    """Drop blowout receiving and huge-spread combined yardage overs.

    2026-09-19 settlement: favorite team rush cashed; dog rec and blowout
    *game* rush/rec missed when the trailer was emptied (UTEP 57 rush, Kent 63 rec).
    """
    if market == "team_receiving_yards":
        if team_spread is not None and team_spread <= SIT_QB_SPREAD:
            return True
        if team_spread is not None and team_spread >= DOG_REC_BLOWOUT:
            return True
    if market == "game_receiving_yards":
        return home_spread is not None and abs(home_spread) >= abs(SIT_QB_SPREAD)
    if market == "game_rushing_yards":
        return home_spread is not None and abs(home_spread) >= BLOWOUT_GAME_YARDS
    return False


def _project(inputs: TeamPropsInputs) -> tuple[float, float]:
    projection = project_team_production("TEAM_PROP", inputs)
    return (
        projection.projected_team_rushing_yards,
        projection.projected_team_receiving_yards,
    )


def build_over_confidence_board(
    conn: sqlite3.Connection,
    slate_date: str,
    *,
    as_of_utc: str | None = None,
    inputs_by_team: dict[str, TeamPropsInputs] | None = None,
    min_prob: float = 0.50,
    season: int | None = None,
) -> list[OverConfidencePick]:
    """OVER-only rushing/receiving board, greatest model probability first."""
    slate = select_ranked_slate(conn, slate_date, season=season)
    if not slate:
        return []
    as_of = as_of_utc or utc_now_iso()
    loaded = inputs_by_team or load_team_props_inputs_for_slate(
        conn, slate_date, season=season, as_of_utc=as_of, with_weather=True
    )
    raw: list[OverConfidencePick] = []
    for game in slate:
        cutoff = game.kickoff_utc
        home_spread = _home_spread(conn, game.game_id)
        away_spread = None if home_spread is None else -home_spread
        home_id = apply_sample_haircut(_inputs_for(loaded, game, game.home_team_id))
        away_id_in = apply_sample_haircut(_inputs_for(loaded, game, game.away_team_id))
        home_pass_allowed = _pass_yards_allowed(
            conn, game.home_team_id, cutoff_utc=cutoff, season=game.season
        )
        away_pass_allowed = _pass_yards_allowed(
            conn, game.away_team_id, cutoff_utc=cutoff, season=game.season
        )
        # Close-game pass bump needs the *opponent's* pass yards allowed.
        home_in = apply_script_adjustment(home_id, home_spread, opp_pass_allowed=away_pass_allowed)
        away_in = apply_script_adjustment(
            away_id_in, away_spread, opp_pass_allowed=home_pass_allowed
        )
        home_rush_id, home_rec_id = _project(home_id)
        away_rush_id, away_rec_id = _project(away_id_in)
        home_rush_s, home_rec_s = _project(home_in)
        away_rush_s, away_rec_s = _project(away_in)
        home_rush_allowed = _rush_yards_allowed_proxy(
            conn, game.home_team_id, cutoff_utc=cutoff, season=game.season
        )
        away_rush_allowed = _rush_yards_allowed_proxy(
            conn, game.away_team_id, cutoff_utc=cutoff, season=game.season
        )
        away_rush = matchup_project(away_rush_s, home_rush_allowed)
        home_rush = matchup_project(home_rush_s, away_rush_allowed)
        away_rec = matchup_project(away_rec_s, home_pass_allowed)
        home_rec = matchup_project(home_rec_s, away_pass_allowed)
        kickoff = format_kickoff_et(game.kickoff_utc)
        label = f"{game.away_school} at {game.home_school}"
        candidates = (
            (
                "team_rushing_yards",
                f"{game.away_alias} rush",
                game.away_team_id,
                away_rush,
                away_rush_id,
                home_rush_allowed,
                away_spread,
            ),
            (
                "team_rushing_yards",
                f"{game.home_alias} rush",
                game.home_team_id,
                home_rush,
                home_rush_id,
                away_rush_allowed,
                home_spread,
            ),
            (
                "team_receiving_yards",
                f"{game.away_alias} rec",
                game.away_team_id,
                away_rec,
                away_rec_id,
                home_pass_allowed,
                away_spread,
            ),
            (
                "team_receiving_yards",
                f"{game.home_alias} rec",
                game.home_team_id,
                home_rec,
                home_rec_id,
                away_pass_allowed,
                home_spread,
            ),
            (
                "game_rushing_yards",
                f"{game.away_alias}/{game.home_alias} game rush",
                None,
                away_rush + home_rush,
                away_rush_id + home_rush_id,
                None
                if home_rush_allowed is None or away_rush_allowed is None
                else home_rush_allowed + away_rush_allowed,
                home_spread,
            ),
            (
                "game_receiving_yards",
                f"{game.away_alias}/{game.home_alias} game rec",
                None,
                away_rec + home_rec,
                away_rec_id + home_rec_id,
                None
                if home_pass_allowed is None or away_pass_allowed is None
                else home_pass_allowed + away_pass_allowed,
                home_spread,
            ),
        )
        for market, pick, team_id, projected, identity, opp_allowed, team_spread in candidates:
            if _sit_qb_receiving(market, team_spread, home_spread):
                continue
            line_cutoff = cutoff if as_of >= cutoff else as_of
            posted, n_books = _posted_line(
                conn, game.game_id, team_id, market, cutoff_utc=line_cutoff
            )
            line = (
                posted
                if posted is not None
                else inferred_line(identity, market, opp_allowed=opp_allowed)
            )
            over_prob = calibrated_over_prob(market, projected, line, posted=posted is not None)
            if over_prob < min_prob:
                continue
            flags: list[str] = list(game.inclusion_reasons)
            if posted is None:
                flags.append("inferred_mark")
            elif n_books < 8:
                flags.append("thin_books")
            if home_spread is not None and abs(home_spread) >= abs(SIT_QB_SPREAD):
                flags.append("blowout_script")
            inputs = loaded.get(team_id) if team_id else None
            if inputs is not None and (
                inputs.data_quality_score < 100.0 or inputs.yards_per_completion >= 14.0
            ):
                flags.append("cupcake_tape")
            raw.append(
                OverConfidencePick(
                    rank=0,
                    game_id=game.game_id,
                    game_label=label,
                    kickoff_et=kickoff,
                    market=market,
                    pick=pick,
                    team_id=team_id,
                    side="OVER",
                    line=line,
                    projected=round(projected, 1),
                    over_prob=over_prob,
                    confidence=over_confidence_from_prob(over_prob),
                    tier=assign_tier(over_prob),
                    inclusion_reasons=game.inclusion_reasons,
                    flags=tuple(flags),
                    family=classify_family(market, team_spread),
                    kickoff_utc=game.kickoff_utc,
                )
            )
    raw.sort(key=lambda row: (-row.over_prob, row.game_label, row.market, row.pick))
    return [replace(row, rank=index) for index, row in enumerate(raw, start=1)]


def _inputs_for(
    loaded: dict[str, TeamPropsInputs], game: RankedSlateGame, team_id: str
) -> TeamPropsInputs:
    try:
        return loaded[team_id]
    except KeyError as exc:
        raise SchemaError(
            f"No team-prop inputs for {team_id} in ranked game {game.game_id}"
        ) from exc


def render_over_confidence_board(
    slate_date: str, rows: list[OverConfidencePick], *, min_prob: float
) -> str:
    stamp = f"   [{config.SHADOW_STAMP}]" if config.is_shadow_mode() else ""
    lines = [
        f"OVER CONFIDENCE BOARD - {slate_date}{stamp}",
        "Team rushing, game rushing, team receiving, game receiving. OVER only.",
        f"Slate: AP Top 25 games + each conference's top two. min P(Over)={min_prob:.0%}.",
        "",
        f"{'rk':>3} {'tier':<6} {'P(Over)':>8} {'conf':>5} {'market':<22} {'pick':<34} "
        f"{'mark':>6} {'proj':>6} {'kickoff':<11} game  flags",
    ]
    if not rows:
        lines.append("(no ranked-slate OVER candidates)")
    for row in rows:
        market = row.market.replace("_", " ")
        flags = ",".join(row.flags)
        lines.append(
            f"{row.rank:>3} {row.tier:<6} {row.over_prob * 100:>7.1f}% {row.confidence:>5.1f} "
            f"{market:<22} {row.pick:<34} {row.line:>6.1f} {row.projected:>6.1f} "
            f"{row.kickoff_et:<11} {row.game_label}  {flags}"
        )
    lines.append(
        "\nP(Over) is capped at 84% (early-season inferred marks are not 99% locks). "
        "HIGH >= 70%, MEDIUM >= 60%, LEAN >= 50%. "
        "Favorite receiving is dropped when the spread is -20 or more (sit-QB). "
        "Marks are posted team-prop consensus (or a 2-book snapshot) when present, "
        "else opponent-blended inferred x.5 lines. Publishing freezes this board "
        "into over_board_snapshots; settlement grades that snapshot, never a rebuild."
    )
    if config.is_shadow_mode():
        lines.append(config.SHADOW_STAMP)
    return "\n".join(lines)


@dataclass
class OverBoardSettlement:
    """Cash/miss of the *published* OVER board vs team box rush/pass yards."""

    slate_date: str
    evaluated: int = 0
    cash: int = 0
    miss: int = 0
    push: int = 0
    skipped: int = 0
    inferred_cash: int = 0
    inferred_n: int = 0
    posted_cash: int = 0
    posted_n: int = 0
    by_tier: dict[str, tuple[int, int]] | None = None
    by_family: dict[str, tuple[int, int]] | None = None
    source: str = "missing"

    def as_text(self) -> str:
        stamp = f"   [{config.SHADOW_STAMP}]" if config.is_shadow_mode() else ""
        lines = [
            f"OVER BOARD SETTLEMENT - {self.slate_date}{stamp}",
            f"  source : {self.source}",
            f"  graded : {self.evaluated}  cash {self.cash}  miss {self.miss}  push {self.push}  "
            f"no-box {self.skipped}",
        ]
        if self.source == "missing":
            lines.append(
                "  no published snapshot; run over-board before settle (never live-rebuild)"
            )
        if self.inferred_n:
            lines.append(
                f"  inferred marks : {self.inferred_cash}/{self.inferred_n} "
                f"({self.inferred_cash / self.inferred_n * 100:.0f}%)"
            )
        if self.posted_n:
            lines.append(
                f"  posted marks   : {self.posted_cash}/{self.posted_n} "
                f"({self.posted_cash / self.posted_n * 100:.0f}%)"
            )
        if self.by_tier:
            for tier in ("HIGH", "MEDIUM", "LEAN"):
                hits, n = self.by_tier.get(tier, (0, 0))
                if n:
                    lines.append(f"  {tier:<6} {hits}/{n} ({hits / n * 100:.0f}%)")
        if self.by_family:
            for family in (
                "favorite_team_rush",
                "other_team_rush",
                "team_rec",
                "game_yards",
            ):
                hits, n = self.by_family.get(family, (0, 0))
                if n:
                    lines.append(f"  {family:<20} {hits}/{n} ({hits / n * 100:.0f}%)")
        return "\n".join(lines)


def persist_over_board(
    conn: sqlite3.Connection,
    slate_date: str,
    rows: Sequence[OverConfidencePick | dict[str, Any]],
    *,
    as_of_utc: str | None = None,
) -> int:
    """Freeze the published board. Settlement grades this snapshot only.

    Games that have already kicked off are never replaced. An empty rebuild
    does not delete a previously published slate.
    """
    published_utc = as_of_utc or utc_now_iso()
    frozen = {
        str(row["game_id"])
        for row in conn.execute(
            """SELECT DISTINCT game_id FROM over_board_snapshots
               WHERE slate_date = ? AND kickoff_utc <= ?""",
            (slate_date, published_utc),
        )
    }
    payload = [
        _snapshot_row(slate_date, row, published_utc)
        for row in rows
        if (row.game_id if isinstance(row, OverConfidencePick) else str(row["game_id"]))
        not in frozen
    ]
    if not payload:
        return 0
    unfrozen = {
        str(row["game_id"])
        for row in conn.execute(
            """SELECT DISTINCT game_id FROM over_board_snapshots
               WHERE slate_date = ? AND kickoff_utc > ?""",
            (slate_date, published_utc),
        )
    }
    replace_ids = unfrozen | {str(row["game_id"]) for row in payload}
    for game_id in replace_ids:
        conn.execute(
            "DELETE FROM over_board_snapshots WHERE slate_date = ? AND game_id = ?",
            (slate_date, game_id),
        )
    cursor = conn.executemany(
        """INSERT INTO over_board_snapshots
           (slate_date, rank, game_id, market, pick, team_id, side, line, projected,
            over_prob, confidence, tier, family, kickoff_utc, kickoff_et, game_label,
            flags, inclusion_reasons, published_utc)
           VALUES (:slate_date, :rank, :game_id, :market, :pick, :team_id, :side, :line,
                   :projected, :over_prob, :confidence, :tier, :family, :kickoff_utc,
                   :kickoff_et, :game_label, :flags, :inclusion_reasons, :published_utc)""",
        payload,
    )
    return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0


def _snapshot_row(
    slate_date: str, row: OverConfidencePick | dict[str, Any], published_utc: str
) -> dict[str, Any]:
    if isinstance(row, OverConfidencePick):
        return {
            "slate_date": slate_date,
            "rank": row.rank,
            "game_id": row.game_id,
            "market": row.market,
            "pick": row.pick,
            "team_id": row.team_id,
            "side": row.side,
            "line": row.line,
            "projected": row.projected,
            "over_prob": row.over_prob,
            "confidence": row.confidence,
            "tier": row.tier,
            "family": row.family,
            "kickoff_utc": row.kickoff_utc,
            "kickoff_et": row.kickoff_et,
            "game_label": row.game_label,
            "flags": json.dumps(list(row.flags)),
            "inclusion_reasons": json.dumps(list(row.inclusion_reasons)),
            "published_utc": published_utc,
        }
    flags = row.get("flags") or ()
    reasons = row.get("inclusion_reasons") or ()
    return {
        "slate_date": slate_date,
        "rank": int(row.get("rank") or 0),
        "game_id": str(row["game_id"]),
        "market": str(row["market"]),
        "pick": str(row["pick"]),
        "team_id": row.get("team_id"),
        "side": str(row.get("side") or "OVER"),
        "line": float(row["line"]),
        "projected": float(row["projected"]),
        "over_prob": float(row["over_prob"]),
        "confidence": float(row["confidence"]),
        "tier": str(row["tier"]),
        "family": str(row.get("family") or classify_family(str(row["market"]), None)),
        "kickoff_utc": str(row["kickoff_utc"]),
        "kickoff_et": str(row.get("kickoff_et") or ""),
        "game_label": str(row.get("game_label") or ""),
        "flags": json.dumps(list(flags)),
        "inclusion_reasons": json.dumps(list(reasons)),
        "published_utc": published_utc,
    }


def load_published_board(conn: sqlite3.Connection, slate_date: str) -> list[OverConfidencePick]:
    rows = conn.execute(
        """SELECT rank, game_id, game_label, kickoff_et, market, pick, team_id, side,
                  line, projected, over_prob, confidence, tier, inclusion_reasons, flags,
                  family, kickoff_utc
           FROM over_board_snapshots
           WHERE slate_date = ?
           ORDER BY rank""",
        (slate_date,),
    ).fetchall()
    return [
        OverConfidencePick(
            rank=int(row["rank"]),
            game_id=str(row["game_id"]),
            game_label=str(row["game_label"]),
            kickoff_et=str(row["kickoff_et"]),
            market=str(row["market"]),
            pick=str(row["pick"]),
            team_id=None if row["team_id"] is None else str(row["team_id"]),
            side=str(row["side"]),
            line=float(row["line"]),
            projected=float(row["projected"]),
            over_prob=float(row["over_prob"]),
            confidence=float(row["confidence"]),
            tier=str(row["tier"]),
            inclusion_reasons=tuple(json.loads(row["inclusion_reasons"])),
            flags=tuple(json.loads(row["flags"])),
            family=str(row["family"]),
            kickoff_utc=str(row["kickoff_utc"]),
        )
        for row in rows
    ]


def _box_yards(
    conn: sqlite3.Connection, game_id: str, team_id: str
) -> tuple[float | None, float | None]:
    row = conn.execute(
        """SELECT rushing_yards, net_passing_yards FROM team_game_box
           WHERE game_id = ? AND team_id = ?""",
        (game_id, team_id),
    ).fetchone()
    if row is None:
        return None, None
    rush = None if row["rushing_yards"] is None else float(row["rushing_yards"])
    rec = None if row["net_passing_yards"] is None else float(row["net_passing_yards"])
    return rush, rec


def _actual_for_pick(conn: sqlite3.Connection, pick: OverConfidencePick) -> float | None:
    if pick.market == "team_rushing_yards" and pick.team_id:
        rush, _ = _box_yards(conn, pick.game_id, pick.team_id)
        return rush
    if pick.market == "team_receiving_yards" and pick.team_id:
        _, rec = _box_yards(conn, pick.game_id, pick.team_id)
        return rec
    game = conn.execute(
        "SELECT home_team_id, away_team_id FROM games WHERE game_id = ?",
        (pick.game_id,),
    ).fetchone()
    if game is None:
        return None
    home_rush, home_rec = _box_yards(conn, pick.game_id, str(game["home_team_id"]))
    away_rush, away_rec = _box_yards(conn, pick.game_id, str(game["away_team_id"]))
    if pick.market == "game_rushing_yards":
        if home_rush is None or away_rush is None:
            return None
        return home_rush + away_rush
    if pick.market == "game_receiving_yards":
        if home_rec is None or away_rec is None:
            return None
        return home_rec + away_rec
    return None


def settle_over_board(conn: sqlite3.Connection, slate_date: str) -> OverBoardSettlement:
    """Grade the frozen snapshot against team_game_box. Never rebuild the board."""
    board = load_published_board(conn, slate_date)
    result = OverBoardSettlement(
        slate_date=slate_date,
        by_tier={},
        by_family={},
        source="snapshot" if board else "missing",
    )
    if not board:
        return result
    tiers: dict[str, list[int]] = {"HIGH": [0, 0], "MEDIUM": [0, 0], "LEAN": [0, 0]}
    families: dict[str, list[int]] = {}
    for pick in board:
        actual = _actual_for_pick(conn, pick)
        if actual is None:
            result.skipped += 1
            continue
        result.evaluated += 1
        inferred = "inferred_mark" in pick.flags
        if actual > pick.line:
            result.cash += 1
            hit = 1
        elif actual < pick.line:
            result.miss += 1
            hit = 0
        else:
            result.push += 1
            continue
        if inferred:
            result.inferred_n += 1
            result.inferred_cash += hit
        else:
            result.posted_n += 1
            result.posted_cash += hit
        tiers.setdefault(pick.tier, [0, 0])
        tiers[pick.tier][1] += 1
        tiers[pick.tier][0] += hit
        family = pick.family or classify_family(pick.market, None)
        families.setdefault(family, [0, 0])
        families[family][1] += 1
        families[family][0] += hit
    result.by_tier = {k: (v[0], v[1]) for k, v in tiers.items()}
    result.by_family = {k: (v[0], v[1]) for k, v in families.items()}
    return result
