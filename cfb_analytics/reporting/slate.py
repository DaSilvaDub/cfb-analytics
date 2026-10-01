"""One research report for every slate market, including missing coverage.

Read-only: yardage rows come from the frozen published board. Market probability
is devigged consensus; model probability is a separate uncalibrated estimate.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict
from typing import Any

from cfb_analytics.config import SHADOW_STAMP
from cfb_analytics.features.current_market import load_current_market_rows
from cfb_analytics.features.mispriced import (
    SPREAD_SIGMA,
    TOTAL_SIGMA,
    _load_model_margins,
    _model_prob_from_edge,
)
from cfb_analytics.features.over_confidence import format_kickoff_et, load_published_board
from cfb_analytics.features.weather import project_slate_game_totals
from cfb_analytics.utils import utc_now_iso

_SECTIONS = {"ML": "moneylines", "SPREAD": "spreads", "TOTAL": "game_totals"}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flags(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
    return [str(flag) for flag in value] if isinstance(value, (list, tuple)) else []


def build_slate_report(conn: sqlite3.Connection, slate_date: str) -> dict[str, Any]:
    """Combine latest paired game lines and published yardage, without filtering edge."""
    generated = utc_now_iso()
    games = [
        dict(row)
        for row in conn.execute(
            """SELECT g.game_id, g.kickoff_utc, g.season,
                      COALESCE(h.school, h.alias, g.home_team_id) AS home,
                      COALESCE(a.school, a.alias, g.away_team_id) AS away
               FROM games g LEFT JOIN teams h ON h.team_id = g.home_team_id
               LEFT JOIN teams a ON a.team_id = g.away_team_id
               WHERE g.football_date = ? ORDER BY g.kickoff_utc, g.game_id""",
            (slate_date,),
        ).fetchall()
    ]
    report: dict[str, Any] = {
        "slate_date": slate_date,
        "generated_utc": generated,
        "stamp": SHADOW_STAMP,
        "actionable": False,
        "model_status": "uncalibrated_shadow",
        "games": games,
        "moneylines": [],
        "spreads": [],
        "game_totals": [],
        "yardage_props": [],
        "missing_coverage": [],
        "diagnostics": [],
    }
    current = load_current_market_rows(conn, slate_date)
    margins = _load_model_margins(conn, slate_date)
    try:
        # Let the projection derive season per game, including season-boundary slates.
        totals = project_slate_game_totals(
            conn, slate_date, as_of_utc=generated, with_weather=True
        )
    except (sqlite3.Error, ValueError, KeyError) as exc:
        totals = {}
        report["diagnostics"].append(f"Game total projection unavailable: {exc}")
    indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in current:
        indexed.setdefault((str(row["game_id"]), row["market"]), []).append(row)
    for game in games:
        game_id = str(game["game_id"])
        base = {
            "game_id": game_id,
            "game_label": f"{game['away']} at {game['home']}",
            "kickoff_utc": game["kickoff_utc"],
            "kickoff_et": format_kickoff_et(game["kickoff_utc"]),
            "actionable": False,
            "model_status": "uncalibrated_shadow",
        }
        margin = _number(margins.get(game_id))
        projection = totals.get(game_id)
        projected_total = (
            _number(projection.projected_game_total) if projection is not None else None
        )
        for market, section in _SECTIONS.items():
            rows = indexed.get((game_id, market), [])
            if not rows:
                report["missing_coverage"].append({"game_id": game_id, "market": market})
                rows = [{"market": market, "side": None, "flags": ["missing_quote"]}]
            for row in rows:
                entry = {**base, **row, "actionable": False}
                entry["flags"] = _flags(row.get("flags"))
                entry["quote_status"] = "available" if row.get("as_of_utc") else "missing"
                fair = next(
                    (
                        p
                        for name in ("prob_shin", "prob_multiplicative", "prob_power")
                        if (p := _number(row.get(name))) is not None and 0 < p < 1
                    ),
                    None,
                )
                entry["market_fair_probability"] = fair
                entry["model_probability"] = None
                entry["model_margin"] = margin if market != "TOTAL" else None
                entry["projected_game_total"] = projected_total if market == "TOTAL" else None
                entry["weather_impact"] = (
                    asdict(projection.weather_impact)
                    if market == "TOTAL"
                    and projection is not None
                    and projection.weather_impact is not None
                    else None
                )
                side = row.get("side")
                line = _number(row.get("line"))
                if margin is not None and market in ("ML", "SPREAD"):
                    signed_margin = margin if side == "HOME" else -margin
                    if side in ("HOME", "AWAY") and (market == "ML" or line is not None):
                        entry["model_probability"] = _model_prob_from_edge(
                            signed_margin, 0.0 if market == "ML" else -float(line), SPREAD_SIGMA
                        )
                elif projected_total is not None and market == "TOTAL" and line is not None:
                    over = _model_prob_from_edge(projected_total, line, TOTAL_SIGMA)
                    if side in ("OVER", "UNDER"):
                        entry["model_probability"] = over if side == "OVER" else round(1 - over, 4)
                if entry["model_probability"] is None:
                    entry["flags"].append("model_probability_unavailable")
                report[section].append(entry)

    publication = {
        (str(row["game_id"]), str(row["market"]), str(row["pick"])): row["published_utc"]
        for row in conn.execute(
            """SELECT game_id, market, pick, published_utc FROM over_board_snapshots
               WHERE slate_date = ?""",
            (slate_date,),
        ).fetchall()
    }
    for pick in load_published_board(conn, slate_date):
        entry = asdict(pick)
        inferred = "inferred_mark" in pick.flags
        entry.update(
            actionable=False,
            model_status="uncalibrated_shadow",
            model_probability=pick.over_prob,
            market_fair_probability=None,
            line_source="inferred" if inferred else "posted_at_publication",
            inferred_line=inferred,
            published_utc=publication.get((pick.game_id, pick.market, pick.pick)),
        )
        report["yardage_props"].append(entry)
    prop_games = {row["game_id"] for row in report["yardage_props"]}
    for game in games:
        if game["game_id"] not in prop_games:
            report["missing_coverage"].append(
                {"game_id": game["game_id"], "market": "YARDAGE", "reason": "no_published_props"}
            )
    return report


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _percent(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.1%}"


def render_slate_report(report: dict[str, Any]) -> str:
    """Render all four market families with explicit probability and source labels."""
    lines = [
        f"# Slate report — {report['slate_date']}",
        "",
        report["stamp"],
        "",
        f"Generated: {report['generated_utc']}. Kickoffs are US Eastern.",
        "",
        "Market fair probability is devigged book consensus. Model probability is an "
        "uncalibrated research estimate. Every row is non-actionable. Game-line tables "
        "show one current paired line per game and market, regardless of edge.",
    ]
    for title, section in (
        ("Moneylines", "moneylines"),
        ("Spreads", "spreads"),
        ("Game totals", "game_totals"),
    ):
        lines.extend(["", f"## {title}", ""])
        rows = report[section]
        if not rows:
            lines.append("No scheduled games on this Eastern slate date.")
            continue
        lines.extend([
            "| Game / ET | Side | Line | Best odds / book | Books | Market fair P | "
            "Model P | Model projection | Capture UTC | Flags |",
            "|---|---|---:|---|---:|---:|---:|---:|---|---|",
        ])
        for row in rows:
            projection = (
                row.get("projected_game_total") if section == "game_totals"
                else row.get("model_margin")
            )
            flags = list(row["flags"])
            if row.get("weather_impact") is not None:
                flags.append("stored weather included")
            cells = [
                f"{row['game_label']} / {row['kickoff_et']}", row.get("side"), row.get("line"),
                f"{_cell(row.get('best_price'))} / {_cell(row.get('best_book'))}",
                row.get("n_books"), _percent(row["market_fair_probability"]),
                _percent(row["model_probability"]), projection, row.get("as_of_utc"),
                ", ".join(flags),
            ]
            lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
        if section != "game_totals":
            lines.extend(["", "Model projection is home margin in points (positive favors home)."])
    lines.extend(["", "## Yardage props", ""])
    if report["yardage_props"]:
        lines.extend([
            "Published snapshot; inferred lines are model marks, not sportsbook offerings.",
            "",
            "| Game / ET | Prop | OVER line | Line source | Projected yards | Model P | "
            "Published UTC | Flags |",
            "|---|---|---:|---|---:|---:|---|---|",
        ])
        for row in report["yardage_props"]:
            cells = [
                f"{row['game_label']} / {row['kickoff_et']}", row["pick"], row["line"],
                row["line_source"], row["projected"], _percent(row["model_probability"]),
                row["published_utc"], ", ".join(row["flags"]),
            ]
            lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    else:
        lines.append("No published yardage board for this date. This report does not rebuild it.")
    if report["missing_coverage"]:
        lines.extend(["", "## Missing coverage", ""])
        labels = {g["game_id"]: f"{g['away']} at {g['home']}" for g in report["games"]}
        for item in report["missing_coverage"]:
            lines.append(f"- {labels[item['game_id']]}: {item['market']} unavailable.")
    if report["diagnostics"]:
        lines.extend(["", "## Diagnostics", "", *report["diagnostics"]])
    return "\n".join(lines) + "\n"
