"""Post-game settlement and calibration metrics for market consensus and models.

Evaluates pre-kickoff vig-free market consensus and research model predictions
against actual final game scores. Computes favorite win rates, upset logs,
Over/Under splits, and formal calibration metrics: Brier score, Expected
Calibration Error (ECE), and Log-loss.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from cfb_analytics import config
from cfb_analytics.utils import utc_now_iso


@dataclass(frozen=True)
class MLUpset:
    favorite: str
    underdog: str
    fav_prob: float
    fav_price: int
    fav_score: int
    dog_score: int


@dataclass(frozen=True)
class CalibrationBucket:
    tier: str
    total: int
    correct: int
    win_pct: float
    mean_confidence: float


@dataclass(frozen=True)
class LineMoveSummary:
    spread_moves_tracked: int
    spread_avg_move_pts: float
    totals_moves_tracked: int
    totals_avg_move_pts: float
    rlm_signals_count: int


@dataclass
class SlateSettlement:
    slate_date: str
    evaluated_at_utc: str
    games_completed: int
    ml_games: int
    ml_fav_correct: int
    ml_fav_total: int
    ml_fav_win_pct: float
    brier_score: float | None
    ece: float | None
    log_loss: float | None
    buckets: list[CalibrationBucket] = field(default_factory=list)
    upsets: list[MLUpset] = field(default_factory=list)
    totals_overs: int = 0
    totals_unders: int = 0
    totals_pushes: int = 0
    team_props_fav_hits: int = 0
    team_props_evaluated: int = 0
    clv_summary: LineMoveSummary | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_actionable"] = False
        data["mode"] = "shadow" if config.is_shadow_mode() else "promoted"
        return data

    def as_text(self) -> str:
        lines = [
            f"SLATE SETTLEMENT REPORT - {self.slate_date}",
            f"  evaluated at UTC : {self.evaluated_at_utc}",
            f"  completed games  : {self.games_completed}",
        ]
        if config.is_shadow_mode():
            lines.append(f"  [{config.SHADOW_STAMP}]")

        lines.append("")
        if self.ml_fav_total > 0:
            lines += [
                "--- Moneyline Consensus Favorite Accuracy ---",
                f"Overall Favorite Record: {self.ml_fav_correct}/{self.ml_fav_total} "
                f"({self.ml_fav_win_pct:.1f}%)",
                "",
                f"{'Tier':<10} {'Record':<10} {'Win %':>7} {'Confidence':>12}",
                "-" * 43,
            ]
            for b in self.buckets:
                if b.total > 0:
                    rec_str = f"{b.correct}/{b.total}"
                    lines.append(
                        f"{b.tier:<10} {rec_str:<10} {b.win_pct:>6.1f}% "
                        f"{b.mean_confidence * 100:>11.1f}%"
                    )

            if self.brier_score is not None:
                lines += [
                    "",
                    "--- Calibration Metrics (Shin Consensus vs Winner) ---",
                    f"  Brier Score : {self.brier_score:.4f}  (0.0 = perfect, 0.25 = random)",
                ]
                if self.ece is not None:
                    lines.append(
                        f"  ECE         : {self.ece * 100:.2f}pp  (Expected Calibration Error)"
                    )
                if self.log_loss is not None:
                    lines.append(f"  Log-Loss    : {self.log_loss:.4f}")

            if self.upsets:
                lines += [
                    "",
                    f"--- Slate Upsets ({len(self.upsets)}) ---",
                ]
                for u in self.upsets:
                    lines.append(
                        f"  {u.underdog:<7} def. {u.favorite:<7} "
                        f"({u.dog_score}-{u.fav_score}) | "
                        f"{u.favorite} was {u.fav_prob * 100:.1f}% ({u.fav_price:+d})"
                    )

        total_games = self.totals_overs + self.totals_unders + self.totals_pushes
        if total_games > 0:
            over_pct = (
                self.totals_overs / max(self.totals_overs + self.totals_unders, 1) * 100
            )
            under_pct = (
                self.totals_unders / max(self.totals_overs + self.totals_unders, 1) * 100
            )
            lines += [
                "",
                f"--- Game Totals Consensus Split ({total_games} games) ---",
                f"  Overs : {self.totals_overs:>3} ({over_pct:.1f}%)",
                f"  Unders: {self.totals_unders:>3} ({under_pct:.1f}%)",
                f"  Pushes: {self.totals_pushes:>3}",
            ]

        if self.team_props_evaluated > 0:
            tp_pct = self.team_props_fav_hits / self.team_props_evaluated * 100
            lines += [
                "",
                "--- Team Total Props Favored Side ---",
                f"  Hit Rate: {self.team_props_fav_hits}/{self.team_props_evaluated} "
                f"({tp_pct:.1f}%)",
            ]

        if self.clv_summary is not None and (
            self.clv_summary.spread_moves_tracked > 0 or self.clv_summary.totals_moves_tracked > 0
        ):
            lines += [
                "",
                "--- Closing Line Value & Movement Audit ---",
                f"  Spread Moves Tracked : {self.clv_summary.spread_moves_tracked}",
                f"  Avg Spread Movement  : {self.clv_summary.spread_avg_move_pts:.2f} pts",
                f"  Total Moves Tracked  : {self.clv_summary.totals_moves_tracked}",
                f"  Avg Total Movement   : {self.clv_summary.totals_avg_move_pts:.2f} pts",
                f"  Reverse Line Moves   : {self.clv_summary.rlm_signals_count}",
            ]

        return "\n".join(lines)


def _compute_calibration_metrics(
    observations: list[tuple[float, int]],
) -> tuple[float | None, float | None, float | None]:
    """Compute Brier score, ECE, and Log-loss.

    observations: list of (home_probability, home_won) where home_won is 1 or 0.
    """
    if not observations:
        return None, None, None

    n = len(observations)
    brier = sum((p - y) ** 2 for p, y in observations) / n

    eps = 1e-7
    log_loss = -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1.0 - p, eps))
        for p, y in observations
    ) / n

    # Expected Calibration Error (ECE) across probability bins [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, y in observations:
        bin_idx = min(int(p * 5), 4)  # 5 bins of width 0.2
        bins[bin_idx].append((p, y))

    ece = 0.0
    for bin_items in bins.values():
        if bin_items:
            bin_size = len(bin_items)
            acc = sum(y for _, y in bin_items) / bin_size
            conf = sum(p for p, _ in bin_items) / bin_size
            ece += (bin_size / n) * abs(acc - conf)

    return brier, ece, log_loss


def settle_slate(
    conn: sqlite3.Connection,
    slate_date: str,
    *,
    min_books: int = 1,
) -> SlateSettlement:
    """Evaluate market consensus against actual game outcomes for a given slate date."""
    completed_games = conn.execute(
        "SELECT COUNT(*) AS n FROM games WHERE football_date = ? AND completed = 1",
        (slate_date,),
    ).fetchone()["n"]

    # 1. Moneyline Evaluation
    ml_query = """
        SELECT g.game_id, ht.alias as home, at.alias as away,
               g.home_points, g.away_points,
               c.side, c.consensus_price, c.prob_shin, c.prob_multiplicative,
               c.n_books
        FROM market_consensus c
        JOIN games g ON g.game_id = c.game_id
        JOIN teams ht ON ht.team_id = g.home_team_id
        JOIN teams at ON at.team_id = g.away_team_id
        WHERE g.football_date = ? AND c.market = 'ML'
        ORDER BY g.game_id, c.n_books DESC
    """
    games_ml: dict[str, dict[str, Any]] = {}
    for row in conn.execute(ml_query, (slate_date,)).fetchall():
        gid = row["game_id"]
        if gid not in games_ml:
            games_ml[gid] = {
                "home": row["home"],
                "away": row["away"],
                "home_score": row["home_points"],
                "away_score": row["away_points"],
                "home_ml": None,
                "away_ml": None,
            }
        side = row["side"]
        existing = games_ml[gid]["home_ml"] if side == "HOME" else games_ml[gid]["away_ml"]
        if existing is None or (row["n_books"] or 0) > (existing["n_books"] or 0):
            if side == "HOME":
                games_ml[gid]["home_ml"] = row
            else:
                games_ml[gid]["away_ml"] = row

    fav_correct = 0
    fav_total = 0
    upsets: list[MLUpset] = []
    bucket_counts: dict[str, dict[str, Any]] = {
        "80%+": {"correct": 0, "total": 0, "conf_sum": 0.0},
        "70-80%": {"correct": 0, "total": 0, "conf_sum": 0.0},
        "60-70%": {"correct": 0, "total": 0, "conf_sum": 0.0},
        "50-60%": {"correct": 0, "total": 0, "conf_sum": 0.0},
    }
    home_prob_obs: list[tuple[float, int]] = []

    for _gid, g in sorted(games_ml.items(), key=lambda x: x[1]["home"] or ""):
        h_ml = g["home_ml"]
        a_ml = g["away_ml"]
        if not h_ml or not a_ml:
            continue
        if (h_ml["n_books"] or 0) < min_books or (a_ml["n_books"] or 0) < min_books:
            continue

        h_score = g["home_score"]
        a_score = g["away_score"]
        if h_score is None or a_score is None:
            continue

        h_prob = h_ml["prob_shin"] or h_ml["prob_multiplicative"]
        a_prob = a_ml["prob_shin"] or a_ml["prob_multiplicative"]
        if h_prob is None or a_prob is None:
            continue

        h_win = h_score > a_score
        a_win = a_score > h_score
        home_prob_obs.append((h_prob, 1 if h_win else 0))

        if h_prob >= a_prob:
            fav_team = g["home"]
            dog_team = g["away"]
            fav_prob = h_prob
            fav_price = h_ml["consensus_price"]
            fav_won = h_win
            fav_score = h_score
            dog_score = a_score
        else:
            fav_team = g["away"]
            dog_team = g["home"]
            fav_prob = a_prob
            fav_price = a_ml["consensus_price"]
            fav_won = a_win
            fav_score = a_score
            dog_score = h_score

        fav_total += 1
        if fav_won:
            fav_correct += 1
        else:
            upsets.append(
                MLUpset(
                    favorite=fav_team,
                    underdog=dog_team,
                    fav_prob=fav_prob,
                    fav_price=fav_price,
                    fav_score=fav_score,
                    dog_score=dog_score,
                )
            )

        p_pct = fav_prob * 100
        if p_pct >= 80:
            b_key = "80%+"
        elif p_pct >= 70:
            b_key = "70-80%"
        elif p_pct >= 60:
            b_key = "60-70%"
        else:
            b_key = "50-60%"

        bucket_counts[b_key]["total"] += 1
        bucket_counts[b_key]["conf_sum"] += fav_prob
        if fav_won:
            bucket_counts[b_key]["correct"] += 1

    upsets.sort(key=lambda x: x.fav_prob, reverse=True)

    calibration_buckets = []
    for tier, b_data in bucket_counts.items():
        tot = b_data["total"]
        cor = b_data["correct"]
        win_pct = (cor / tot * 100) if tot > 0 else 0.0
        mean_conf = (b_data["conf_sum"] / tot) if tot > 0 else 0.0
        calibration_buckets.append(
            CalibrationBucket(
                tier=tier,
                total=tot,
                correct=cor,
                win_pct=win_pct,
                mean_confidence=mean_conf,
            )
        )

    brier, ece, log_loss = _compute_calibration_metrics(home_prob_obs)
    fav_win_pct = (fav_correct / fav_total * 100) if fav_total > 0 else 0.0

    # 2. Game Totals Evaluation
    total_query = """
        SELECT g.game_id, g.home_points, g.away_points,
               c.side, c.line, c.n_books
        FROM market_consensus c
        JOIN games g ON g.game_id = c.game_id
        WHERE g.football_date = ? AND c.market = 'TOTAL'
    """
    totals_by_game: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"lines": defaultdict(dict), "home_score": None, "away_score": None}
    )
    for row in conn.execute(total_query, (slate_date,)):
        gid = row["game_id"]
        entry = totals_by_game[gid]
        entry["home_score"] = row["home_points"]
        entry["away_score"] = row["away_points"]
        entry["lines"][row["line"]][row["side"]] = row

    overs = unders = pushes = 0
    for _gid, data in totals_by_game.items():
        if data["home_score"] is None or data["away_score"] is None:
            continue
        best_line = None
        max_books = -1
        for line, sides in data["lines"].items():
            if "OVER" in sides and "UNDER" in sides:
                b_count = (sides["OVER"]["n_books"] or 0) + (sides["UNDER"]["n_books"] or 0)
                if b_count > max_books:
                    max_books = b_count
                    best_line = line
        if best_line is not None:
            actual = data["home_score"] + data["away_score"]
            if actual > best_line:
                overs += 1
            elif actual < best_line:
                unders += 1
            else:
                pushes += 1

    # 3. Team Props Evaluation
    tp_query = """
        SELECT tp.game_id, tp.team_id, tp.line, tp.side,
               tp.prob_shin, tp.prob_multiplicative,
               g.home_team_id, g.home_points, g.away_points
        FROM team_prop_consensus tp
        JOIN games g ON g.game_id = tp.game_id
        WHERE g.football_date = ? AND tp.market = 'team_total_points'
    """
    paired_tp: dict[tuple[str, str, float], dict[str, Any]] = {}
    for r in conn.execute(tp_query, (slate_date,)):
        key = (r["game_id"], r["team_id"], r["line"])
        if key not in paired_tp:
            paired_tp[key] = {
                "team_id": r["team_id"],
                "home_team_id": r["home_team_id"],
                "home_points": r["home_points"],
                "away_points": r["away_points"],
                "line": r["line"],
                "OVER": None,
                "UNDER": None,
            }
        paired_tp[key][r["side"]] = r

    tp_fav_hits = 0
    tp_total = 0
    for _key, p in paired_tp.items():
        if not p["OVER"] or not p["UNDER"]:
            continue
        actual = p["home_points"] if p["team_id"] == p["home_team_id"] else p["away_points"]
        if actual is None or actual == p["line"]:
            continue

        o_prob = p["OVER"]["prob_shin"] or p["OVER"]["prob_multiplicative"] or 0.5
        u_prob = p["UNDER"]["prob_shin"] or p["UNDER"]["prob_multiplicative"] or 0.5
        fav_side = "OVER" if o_prob >= u_prob else "UNDER"

        over_hit = actual > p["line"]
        if (fav_side == "OVER" and over_hit) or (fav_side == "UNDER" and not over_hit):
            tp_fav_hits += 1
        tp_total += 1

    # 4. Closing Line Value & Movement Audit
    clv_query = """
        SELECT lm.market, lm.move_magnitude, lm.move_direction, lm.rlm_flag
        FROM line_movement lm
        JOIN games g ON g.game_id = lm.game_id
        WHERE g.football_date = ?
    """
    spread_mags: list[float] = []
    total_mags: list[float] = []
    rlm_count = 0
    for row in conn.execute(clv_query, (slate_date,)).fetchall():
        m = row["market"]
        mag = row["move_magnitude"]
        if m == "SPREAD" and mag is not None:
            spread_mags.append(abs(float(mag)))
        elif m == "TOTAL" and mag is not None:
            total_mags.append(abs(float(mag)))
        if row["rlm_flag"]:
            rlm_count += 1

    clv_summary = LineMoveSummary(
        spread_moves_tracked=len(spread_mags),
        spread_avg_move_pts=sum(spread_mags) / len(spread_mags) if spread_mags else 0.0,
        totals_moves_tracked=len(total_mags),
        totals_avg_move_pts=sum(total_mags) / len(total_mags) if total_mags else 0.0,
        rlm_signals_count=rlm_count,
    )

    return SlateSettlement(
        slate_date=slate_date,
        evaluated_at_utc=utc_now_iso(),
        games_completed=completed_games,
        ml_games=len(games_ml),
        ml_fav_correct=fav_correct,
        ml_fav_total=fav_total,
        ml_fav_win_pct=fav_win_pct,
        brier_score=brier,
        ece=ece,
        log_loss=log_loss,
        buckets=calibration_buckets,
        upsets=upsets,
        totals_overs=overs,
        totals_unders=unders,
        totals_pushes=pushes,
        team_props_fav_hits=tp_fav_hits,
        team_props_evaluated=tp_total,
        clv_summary=clv_summary,
    )
