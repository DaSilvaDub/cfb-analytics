"""One slate report covering every market family the pipeline prices.

``board`` (moneylines), ``mispriced`` (spreads, totals) and ``over-board``
(rushing/receiving OVERs) each answer one question and each print on their
own. Read in isolation, the OVER board looks like the whole pipeline. This
report puts all four families side by side, one primary line per game, from
the current capture only - so a missing family reads as "no current market"
rather than vanishing.

Every family section is always printed, even when empty. Nothing here is a
recommendation: model columns are uncalibrated shadow output.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cfb_analytics import config
from cfb_analytics.features.current_market import current_primary_market, fair_prob
from cfb_analytics.features.over_confidence import OverConfidencePick, format_kickoff_et

FAMILIES: tuple[str, ...] = ("ML", "SPREAD", "TOTAL", "YARDAGE")
FAMILY_TITLES = {
    "ML": "MONEYLINES",
    "SPREAD": "SPREADS",
    "TOTAL": "GAME TOTALS",
    "YARDAGE": "RUSHING / RECEIVING OVERS",
}


@dataclass(frozen=True)
class GameLineRow:
    """One game's primary line in one market, with the model's view beside it."""

    game_id: str
    game_label: str
    kickoff_et: str
    kickoff_utc: str
    market: str
    line: float | None
    # Prices / fair probabilities keyed by side (HOME/AWAY or OVER/UNDER).
    prices: dict[str, int | None]
    fair: dict[str, float | None]
    lines: dict[str, float | None]
    n_books: int
    hold: float | None
    as_of_utc: str
    model_projected: float | None = None
    model_prob: dict[str, float] = field(default_factory=dict)

    def edge(self, side: str) -> float | None:
        model = self.model_prob.get(side)
        market = self.fair.get(side)
        if model is None or market is None:
            return None
        return round(model - market, 4)

    def best_side(self) -> tuple[str, float] | None:
        """The side the model prefers, with its edge over the fair price."""
        edges = [(side, e) for side in self.fair if (e := self.edge(side)) is not None]
        if not edges:
            return None
        return max(edges, key=lambda item: item[1])


@dataclass
class SlateReport:
    slate_date: str
    games: int
    rows: dict[str, list[GameLineRow]]
    yardage: list[OverConfidencePick]
    yardage_source: str
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        counts = {family: len(self.rows.get(family, [])) for family in ("ML", "SPREAD", "TOTAL")}
        counts["YARDAGE"] = len(self.yardage)
        return counts


def _model_prob(projected: float, line: float, sigma: float) -> float:
    from cfb_analytics.features.mispriced import _model_prob_from_edge

    return _model_prob_from_edge(projected, line, sigma)


def _game_rows(
    primary: Sequence[Mapping[str, Any]],
    *,
    margins: Mapping[str, float],
    totals: Mapping[str, float],
) -> dict[str, list[GameLineRow]]:
    from cfb_analytics.features.mispriced import SPREAD_SIGMA, TOTAL_SIGMA

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in primary:
        grouped.setdefault((str(row["game_id"]), str(row["market"])), []).append(row)

    out: dict[str, list[GameLineRow]] = {"ML": [], "SPREAD": [], "TOTAL": []}
    for (game_id, market_code), sides in grouped.items():
        if market_code not in out:
            continue
        first = sides[0]
        by_side = {str(r["side"]).upper(): r for r in sides}
        model_projected: float | None = None
        model_prob: dict[str, float] = {}

        if market_code == "ML" and game_id in margins:
            margin = margins[game_id]
            model_projected = margin
            p_home = _model_prob(margin, 0.0, SPREAD_SIGMA)
            model_prob = {"HOME": p_home, "AWAY": round(1.0 - p_home, 4)}
        elif market_code == "SPREAD" and game_id in margins and "HOME" in by_side:
            margin = margins[game_id]
            model_projected = margin
            home_line = by_side["HOME"]["line"]
            if home_line is not None:
                # HOME -7 covers when the home margin beats 7.
                p_home = _model_prob(margin, -float(home_line), SPREAD_SIGMA)
                model_prob = {"HOME": p_home, "AWAY": round(1.0 - p_home, 4)}
        elif market_code == "TOTAL" and game_id in totals and first["line"] is not None:
            projected = totals[game_id]
            model_projected = projected
            p_over = _model_prob(projected, float(first["line"]), TOTAL_SIGMA)
            model_prob = {"OVER": p_over, "UNDER": round(1.0 - p_over, 4)}

        display_line: float | None
        if market_code == "SPREAD":
            home = by_side.get("HOME")
            display_line = home["line"] if home is not None else first["line"]
        elif market_code == "ML":
            display_line = None
        else:
            display_line = first["line"]

        out[market_code].append(
            GameLineRow(
                game_id=game_id,
                game_label=f"{first['away'] or '?'} at {first['home'] or '?'}",
                kickoff_et=format_kickoff_et(str(first["kickoff_utc"] or "")),
                kickoff_utc=str(first["kickoff_utc"] or ""),
                market=market_code,
                line=display_line,
                prices={side: r["consensus_price"] for side, r in by_side.items()},
                fair={side: fair_prob(r) for side, r in by_side.items()},
                lines={side: r["line"] for side, r in by_side.items()},
                n_books=min(int(r["n_books"] or 0) for r in sides),
                hold=first["hold"],
                as_of_utc=str(first["as_of_utc"]),
                model_projected=model_projected,
                model_prob=model_prob,
            )
        )
    for rows in out.values():
        rows.sort(key=lambda r: (r.kickoff_utc, r.game_label))
    return out


def _yardage(
    conn: sqlite3.Connection, slate_date: str, *, min_prob: float
) -> tuple[list[OverConfidencePick], str]:
    """The published OVER snapshot when one exists, else an unpublished preview.

    Never publishes: freezing ``over_board_snapshots`` is ``over-board``'s job,
    and settlement grades that frozen snapshot.
    """
    from cfb_analytics.features.over_confidence import (
        build_over_confidence_board,
        load_published_board,
    )

    published = load_published_board(conn, slate_date)
    if published:
        return [p for p in published if p.over_prob >= min_prob], "published snapshot"
    return (
        build_over_confidence_board(conn, slate_date, min_prob=min_prob),
        "unpublished preview",
    )


def build_slate_report(
    conn: sqlite3.Connection,
    slate_date: str,
    *,
    min_over_prob: float = 0.50,
) -> SlateReport:
    from cfb_analytics.features.mispriced import _load_model_margins, _load_model_totals

    notes: list[str] = []
    games = conn.execute(
        "SELECT COUNT(*) AS n FROM games WHERE football_date = ?", (slate_date,)
    ).fetchone()["n"]

    primary = current_primary_market(conn, slate_date)
    margins = _load_model_margins(conn, slate_date)
    totals = _load_model_totals(conn, slate_date)
    if not margins:
        notes.append("no Elo/Ridge ratings stored: spread and moneyline model columns are blank")
    if not totals:
        notes.append("no team-production inputs: game-total model column is blank")
    rows = _game_rows(primary, margins=margins, totals=totals)

    try:
        yardage, source = _yardage(conn, slate_date, min_prob=min_over_prob)
    except Exception as exc:  # the OVER board has its own inputs; it must not sink the report
        yardage, source = [], "unavailable"
        notes.append(f"OVER board unavailable: {str(exc)[:160]}")

    return SlateReport(
        slate_date=slate_date,
        games=int(games or 0),
        rows=rows,
        yardage=yardage,
        yardage_source=source,
        notes=notes,
    )


def _fmt_price(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:+d}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _fmt_num(value: float | None, spec: str = ".1f") -> str:
    return "-" if value is None else format(value, spec)


def _fmt_lean(row: GameLineRow, labels: Mapping[str, str]) -> str:
    best = row.best_side()
    if best is None:
        return "-"
    side, edge = best
    return f"{labels.get(side, side)} {edge * 100:+.1f}pp"


def _team_labels(row: GameLineRow) -> dict[str, str]:
    away, _, home = row.game_label.partition(" at ")
    return {"HOME": home, "AWAY": away, "OVER": "OVER", "UNDER": "UNDER"}


def render_slate_report(report: SlateReport) -> str:
    stamp = f"   [{config.SHADOW_STAMP}]" if config.is_shadow_mode() else ""
    counts = report.counts()
    lines = [
        f"SLATE REPORT - {report.slate_date}{stamp}",
        f"games on slate: {report.games}   "
        + "   ".join(f"{FAMILY_TITLES[f].lower()}: {counts[f]}" for f in FAMILIES),
        "Current capture only, one primary line per game (most books, then most balanced).",
    ]

    ml = report.rows.get("ML", [])
    lines += ["", f"== {FAMILY_TITLES['ML']} ==",
              f"{'kickoff':<11} {'game':<30} {'away':>6} {'home':>6} "
              f"{'fair away':>9} {'fair home':>9} {'model home':>10} {'bk':>3}  lean"]
    if not ml:
        lines.append("(no current moneyline market)")
    for row in ml:
        lines.append(
            f"{row.kickoff_et:<11} {row.game_label:<30} "
            f"{_fmt_price(row.prices.get('AWAY')):>6} {_fmt_price(row.prices.get('HOME')):>6} "
            f"{_fmt_pct(row.fair.get('AWAY')):>9} {_fmt_pct(row.fair.get('HOME')):>9} "
            f"{_fmt_pct(row.model_prob.get('HOME')):>10} {row.n_books:>3}  "
            f"{_fmt_lean(row, _team_labels(row))}"
        )

    spreads = report.rows.get("SPREAD", [])
    lines += ["", f"== {FAMILY_TITLES['SPREAD']} ==",
              f"{'kickoff':<11} {'game':<30} {'home ln':>7} {'price':>6} "
              f"{'fair home':>9} {'model mgn':>9} {'model home':>10} {'bk':>3}  lean"]
    if not spreads:
        lines.append("(no current spread market)")
    for row in spreads:
        lines.append(
            f"{row.kickoff_et:<11} {row.game_label:<30} {_fmt_num(row.line, '+.1f'):>7} "
            f"{_fmt_price(row.prices.get('HOME')):>6} {_fmt_pct(row.fair.get('HOME')):>9} "
            f"{_fmt_num(row.model_projected, '+.1f'):>9} "
            f"{_fmt_pct(row.model_prob.get('HOME')):>10} {row.n_books:>3}  "
            f"{_fmt_lean(row, _team_labels(row))}"
        )

    totals = report.rows.get("TOTAL", [])
    lines += ["", f"== {FAMILY_TITLES['TOTAL']} ==",
              f"{'kickoff':<11} {'game':<30} {'total':>6} {'over':>6} {'under':>6} "
              f"{'fair over':>9} {'model tot':>9} {'model over':>10} {'bk':>3}  lean"]
    if not totals:
        lines.append("(no current game-total market)")
    for row in totals:
        lines.append(
            f"{row.kickoff_et:<11} {row.game_label:<30} {_fmt_num(row.line):>6} "
            f"{_fmt_price(row.prices.get('OVER')):>6} {_fmt_price(row.prices.get('UNDER')):>6} "
            f"{_fmt_pct(row.fair.get('OVER')):>9} {_fmt_num(row.model_projected):>9} "
            f"{_fmt_pct(row.model_prob.get('OVER')):>10} {row.n_books:>3}  "
            f"{_fmt_lean(row, _team_labels(row))}"
        )

    lines += ["", f"== {FAMILY_TITLES['YARDAGE']} ({report.yardage_source}) ==",
              f"{'rk':>3} {'tier':<6} {'P(Over)':>8} {'market':<22} {'pick':<34} "
              f"{'mark':>6} {'proj':>6} {'kickoff':<11} game"]
    if not report.yardage:
        lines.append("(no ranked-slate OVER candidates)")
    for pick in report.yardage:
        lines.append(
            f"{pick.rank:>3} {pick.tier:<6} {pick.over_prob * 100:>7.1f}% "
            f"{pick.market.replace('_', ' '):<22} {pick.pick:<34} {pick.line:>6.1f} "
            f"{pick.projected:>6.1f} {pick.kickoff_et:<11} {pick.game_label}"
        )

    if report.notes:
        lines += ["", "notes:"] + [f"  - {note}" for note in report.notes]
    lines += [
        "",
        "fair = vig-free market probability (Shin). model = uncalibrated Elo/Ridge "
        "(spread, moneyline) or team-production (total) projection. lean = the side "
        "the model prefers and its gap to fair, in percentage points. A lean is not a "
        "bet: governance verdicts live in `mispriced` and `board --with-reasoning`.",
    ]
    if config.is_shadow_mode():
        lines.append(config.SHADOW_STAMP)
    return "\n".join(lines)


def slate_report_json(report: SlateReport) -> dict[str, Any]:
    def game_row(row: GameLineRow) -> dict[str, Any]:
        best = row.best_side()
        return {
            "game_id": row.game_id,
            "game": row.game_label,
            "kickoff_et": row.kickoff_et,
            "kickoff_utc": row.kickoff_utc,
            "market": row.market,
            "line": row.line,
            "lines": row.lines,
            "prices": row.prices,
            "fair_prob": row.fair,
            "model_projected": row.model_projected,
            "model_prob": row.model_prob,
            "lean": {"side": best[0], "edge": best[1]} if best else None,
            "n_books": row.n_books,
            "hold": row.hold,
            "as_of_utc": row.as_of_utc,
        }

    return {
        "date": report.slate_date,
        "stamp": config.SHADOW_STAMP if config.is_shadow_mode() else None,
        "games": report.games,
        "counts": report.counts(),
        "moneylines": [game_row(r) for r in report.rows.get("ML", [])],
        "spreads": [game_row(r) for r in report.rows.get("SPREAD", [])],
        "game_totals": [game_row(r) for r in report.rows.get("TOTAL", [])],
        "yardage_overs": {
            "source": report.yardage_source,
            "picks": [
                {
                    "rank": p.rank,
                    "tier": p.tier,
                    "over_prob": p.over_prob,
                    "market": p.market,
                    "pick": p.pick,
                    "line": p.line,
                    "projected": p.projected,
                    "kickoff_et": p.kickoff_et,
                    "game": p.game_label,
                    "game_id": p.game_id,
                }
                for p in report.yardage
            ],
        },
        "notes": report.notes,
    }


__all__ = [
    "FAMILIES",
    "GameLineRow",
    "SlateReport",
    "build_slate_report",
    "render_slate_report",
    "slate_report_json",
]
