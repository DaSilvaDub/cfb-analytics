"""Current paired primary gamelines for slate reports and scanners."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Any

from cfb_analytics import config
from cfb_analytics.features import market
from cfb_analytics.utils import to_utc_iso, utc_now_iso


def load_current_market_rows(
    conn: sqlite3.Connection, slate_date: str, *, as_of_utc: str | None = None
) -> list[dict[str, Any]]:
    """One paired line per game/market, using each source's latest capture.

    Prefer explicit primary quotes, otherwise the most-booked paired line.
    Historical or post-kickoff observations never supply a newer capture.
    """
    cutoff = to_utc_iso(as_of_utc or utc_now_iso())
    if cutoff is None:
        raise ValueError("as_of_utc must be a valid timestamp")
    games = conn.execute(
        """SELECT g.game_id, g.kickoff_utc, g.home_team_id, g.away_team_id,
                  COALESCE(h.alias, h.school) home, COALESCE(a.alias, a.school) away
           FROM games g JOIN teams h ON h.team_id=g.home_team_id
           JOIN teams a ON a.team_id=g.away_team_id
           WHERE g.football_date=? ORDER BY g.kickoff_utc, g.game_id""",
        (slate_date,),
    ).fetchall()
    output: list[dict[str, Any]] = []
    settings = config.settings()["market"]
    sharp = tuple(config.sources()["outlier"].get("sharp_books", ()))
    for game in games:
        kickoff = to_utc_iso(game["kickoff_utc"])
        if kickoff is None:
            continue
        raw = conn.execute("SELECT * FROM odds_snapshots WHERE game_id=?", (game["game_id"],))
        eligible = []
        latest: dict[tuple[str, str], str] = {}
        for record in raw:
            row = dict(record)
            if row["market"] not in ("ML", "SPREAD", "TOTAL"):
                continue
            try:
                captured = to_utc_iso(row["captured_utc"])
            except (ValueError, TypeError):
                continue
            if captured is None:
                continue
            if captured > cutoff or captured >= kickoff:
                continue
            row["captured_utc"] = captured
            key = (row["source"], row["market"])
            latest[key] = max(latest.get(key, captured), captured)
            eligible.append(row)
        current = [r for r in eligible if r["captured_utc"] == latest[r["source"], r["market"]]]
        groups: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
        for row in current:
            groups[market.group_key(row)].append(row)
        candidates: dict[str, list[tuple[tuple, list[dict[str, Any]]]]] = defaultdict(list)
        for (_, code, _), quotes in groups.items():
            # A book carried by both sources gets its latest observation once.
            unique: dict[tuple[str, str], dict[str, Any]] = {}
            for row in quotes:
                key = (row["book"], row["side"])
                if key not in unique or row["captured_utc"] > unique[key]["captured_utc"]:
                    unique[key] = row
            quotes = list(unique.values())
            captured = max(r["captured_utc"] for r in quotes)
            consensus = market.build_consensus(
                game["game_id"],
                code,
                quotes,
                as_of_utc=captured,
                sharp_books=sharp,
                min_books_for_consensus=int(settings["min_books_for_consensus"]),
            )
            expected = {"OVER", "UNDER"} if code == "TOTAL" else {"HOME", "AWAY"}
            if consensus is None or {q.side for q in consensus.sides} != expected:
                continue
            primary_books = {r["book"] for r in quotes if r["is_primary"]}
            flags = list(consensus.flags)
            stale_after = float(settings.get("odds_stale_after_minutes", 360))
            if any(
                (
                    datetime.fromisoformat(cutoff) - datetime.fromisoformat(r["captured_utc"])
                ).total_seconds()
                > stale_after * 60
                for r in quotes
            ):
                flags.append("stale_odds")
            if not primary_books:
                flags.append("representative_line_inferred")
            entries = []
            for q in consensus.sides:
                entries.append(
                    {
                        **dict(game),
                        "market": code,
                        "side": q.side,
                        "line": q.line,
                        "as_of_utc": captured,
                        "n_books": q.n_books,
                        "consensus_price": q.consensus_price,
                        "best_price": q.best_price,
                        "best_book": q.best_book,
                        "hold": consensus.hold,
                        "anchor": consensus.anchor,
                        "prob_shin": q.probs.get("shin"),
                        "prob_multiplicative": q.probs.get("multiplicative"),
                        "prob_power": q.probs.get("power"),
                        "prob_spread": market.summarise_probability_spread(consensus, q.side),
                        "flags": json.dumps(flags),
                    }
                )
            # Select a whole pair; never choose each side from different ladders.
            score = (
                bool(primary_books),
                len(primary_books),
                min(q.n_books for q in consensus.sides),
                -abs(next(iter(consensus.sides)).line or 0.0),
            )
            candidates[code].append((score, entries))
        if not eligible:
            # A consensus-only store is valid (e.g. imported research artifacts).
            # Retain its real timestamp and explicitly flag missing raw quotes.
            stored = conn.execute(
                "SELECT * FROM market_consensus WHERE game_id=?", (game["game_id"],)
            ).fetchall()
            by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in stored:
                row = dict(record)
                captured = to_utc_iso(row["as_of_utc"])
                if captured is None:
                    continue
                if (
                    row["market"] in ("ML", "SPREAD", "TOTAL")
                    and captured <= cutoff
                    and captured < kickoff
                ):
                    by_market[row["market"]].append(row)
            for code, records in by_market.items():
                newest = max(r["as_of_utc"] for r in records)
                pairs: dict[float, list[dict[str, Any]]] = defaultdict(list)
                for row in records:
                    if row["as_of_utc"] == newest:
                        pairs[abs(row["line"]) if code == "SPREAD" else row["line"]].append(row)
                for entries in pairs.values():
                    expected = {"OVER", "UNDER"} if code == "TOTAL" else {"HOME", "AWAY"}
                    if len(entries) != 2 or {r["side"] for r in entries} != expected:
                        continue
                    for row in entries:
                        row.update(dict(game))
                        row["flags"] = json.dumps(
                            [*json.loads(row["flags"] or "[]"), "raw_quotes_unavailable"]
                        )
                    candidates[code].append(
                        (
                            (
                                False,
                                0,
                                min(r["n_books"] for r in entries),
                                -abs(entries[0]["line"]),
                            ),
                            entries,
                        )
                    )
        for code in ("ML", "SPREAD", "TOTAL"):
            if candidates[code]:
                output.extend(max(candidates[code], key=lambda item: item[0])[1])
    return output
