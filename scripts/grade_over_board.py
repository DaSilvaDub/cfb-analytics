"""Grade the published 2026-09-19 over-board against CFBD box scores."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from cfb_analytics.sources.cfbd import CFBDClient

BOARD = Path(r"C:\Users\dasil\OneDrive\Desktop\today\cfb_over_board_2026-09-19.md")
OUT = Path(r"C:\Users\dasil\OneDrive\Desktop\today\cfb_over_board_settlement_2026-09-19.md")

# pick token -> CFBD school name
ALIAS = {
    "MISS": "Ole Miss",
    "TENN": "Tennessee",
    "LSU": "LSU",
    "SMU": "SMU",
    "LOU": "Louisville",
    "UTEP": "UTEP",
    "MICH": "Michigan",
    "WVU": "West Virginia",
    "UVA": "Virginia",
    "IU": "Indiana",
    "FSU": "Florida State",
    "ALA": "Alabama",
    "UNM": "New Mexico",
    "OU": "Oklahoma",
    "PSU": "Penn State",
    "BYU": "BYU",
    "CSU": "Colorado State",
    "UTAH": "Utah",
    "UK": "Kentucky",
    "TA&M": "Texas A&M",
    "MSU": "Michigan State",
    "ND": "Notre Dame",
    "WKU": "Western Kentucky",
    "UGA": "Georgia",
    "RUTG": "Rutgers",
    "TROY": "Troy",
    "MIZ": "Missouri",
    "OSU": "Ohio State",
    "ARK": "Arkansas",
    "UTSA": "UTSA",
    "KENN": "Kennesaw State",
    "USC": "USC",
    "USU": "Utah State",
    "BUF": "Buffalo",
    "TEX": "Texas",
    "KENT": "Kent State",
    "IOWA": "Iowa",
    "Northern Iowa": "Northern Iowa",
}


def parse_board(text: str) -> list[dict]:
    rows = []
    pat = re.compile(
        r"^\s*(\d+)\s+(HIGH|MEDIUM|LEAN)\s+(\d+\.\d+)%\s+(\d+\.\d+)\s+"
        r"(team rushing yards|team receiving yards|game rushing yards|game receiving yards)\s+"
        r"(.+?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+:\d+ [ap]\.m\.)\s{2,}(.+?)\s{2,}(\S.*)$"
    )
    for line in text.splitlines():
        m = pat.match(line)
        if not m:
            continue
        rows.append(
            {
                "rk": int(m.group(1)),
                "tier": m.group(2),
                "prob": float(m.group(3)) / 100.0,
                "conf": float(m.group(4)),
                "market": m.group(5),
                "pick": m.group(6).strip(),
                "mark": float(m.group(7)),
                "proj": float(m.group(8)),
                "kickoff": m.group(9),
                "game": m.group(10).strip(),
                "flags": m.group(11).strip(),
            }
        )
    return rows


def box_maps(client: CFBDClient) -> tuple[dict, dict]:
    """Return (rush_by_school, rec_by_school) for week 3, keyed by (school, opponent)."""
    payload = client._get_rows("/games/teams", year=2026, week=3, seasonType="regular")
    rush: dict[tuple[str, str], float] = {}
    rec: dict[tuple[str, str], float] = {}
    completed: set[str] = set()
    for game in payload:
        teams = game.get("teams") or []
        names = []
        stats_by = {}
        for side in teams:
            school = str(side.get("team") or side.get("school") or "").strip()
            names.append(school)
            cats = {str(s.get("category")): s.get("stat") for s in (side.get("stats") or [])}
            stats_by[school] = cats
            if side.get("points") is not None:
                completed.add(school)
        if len(names) != 2:
            continue
        a, b = names
        for school, opp in ((a, b), (b, a)):
            cats = stats_by.get(school) or {}
            try:
                rush[(school, opp)] = float(
                    cats.get("rushingYards") or cats.get("rushing yards") or "nan"
                )
            except (TypeError, ValueError):
                pass
            try:
                rec[(school, opp)] = float(
                    cats.get("netPassingYards")
                    or cats.get("passingYards")
                    or cats.get("netPassingYards")
                    or "nan"
                )
            except (TypeError, ValueError):
                pass
    return rush, rec


def opponents(game_label: str) -> tuple[str, str]:
    # "Kennesaw State at Tennessee" -> away, home
    if " at " not in game_label:
        raise ValueError(game_label)
    away, home = game_label.split(" at ", 1)
    return away.strip(), home.strip()


def resolve_actual(row: dict, rush: dict, rec: dict) -> float | None:
    away, home = opponents(row["game"])
    market = row["market"]
    pick = row["pick"]
    if market.startswith("game"):
        if market.endswith("rushing yards"):
            a = rush.get((away, home))
            b = rush.get((home, away))
        else:
            a = rec.get((away, home))
            b = rec.get((home, away))
        if a is None or b is None:
            return None
        return a + b
    token = pick.replace(" rush", "").replace(" rec", "").strip()
    school = ALIAS.get(token, token)
    opp = home if school == away else away if school == home else None
    if opp is None:
        return None
    table = rush if market.endswith("rushing yards") else rec
    return table.get((school, opp))


def grade(actual: float, mark: float) -> str:
    if actual > mark:
        return "CASH"
    if actual < mark:
        return "MISS"
    return "PUSH"


def main() -> None:
    text = BOARD.read_text(encoding="utf-8")
    rows = parse_board(text)
    client = CFBDClient()
    rush, rec = box_maps(client)
    results = []
    for row in rows:
        actual = resolve_actual(row, rush, rec)
        if actual is None:
            row["actual"] = None
            row["result"] = "NO_BOX"
        else:
            row["actual"] = actual
            row["result"] = grade(actual, row["mark"])
        results.append(row)

    def rate(subset: list[dict]) -> str:
        graded = [r for r in subset if r["result"] in ("CASH", "MISS")]
        if not graded:
            return "n/a"
        cash = sum(1 for r in graded if r["result"] == "CASH")
        return f"{cash}/{len(graded)} ({cash / len(graded) * 100:.0f}%)"

    by_tier = defaultdict(list)
    by_market = defaultdict(list)
    by_flag = defaultdict(list)
    for r in results:
        by_tier[r["tier"]].append(r)
        by_market[r["market"]].append(r)
        for flag in r["flags"].split(","):
            by_flag[flag.strip()].append(r)

    lines = [
        "OVER BOARD SETTLEMENT - 2026-09-19",
        "UNPROMOTED - shadow output, not decision-grade",
        "",
        f"Published picks: {len(results)}",
        f"Overall (box available): {rate(results)}",
        "",
        "By tier",
        f"  HIGH   {rate(by_tier['HIGH'])}",
        f"  MEDIUM {rate(by_tier['MEDIUM'])}",
        f"  LEAN   {rate(by_tier['LEAN'])}",
        "",
        "By market",
    ]
    for m in (
        "team rushing yards",
        "game rushing yards",
        "team receiving yards",
        "game receiving yards",
    ):
        lines.append(f"  {m}: {rate(by_market[m])}")
    lines += [
        "",
        "By flag (overlapping)",
        f"  blowout_script {rate(by_flag['blowout_script'])}",
        f"  cupcake_tape   {rate(by_flag['cupcake_tape'])}",
        f"  inferred_mark  {rate(by_flag['inferred_mark'])}",
        "",
        f"{'rk':>3} {'res':<7} {'tier':<6} {'pick':<28} {'mark':>6} {'act':>6} {'kickoff':<11} game",
    ]
    for r in results:
        act = "" if r["actual"] is None else f"{r['actual']:.0f}"
        lines.append(
            f"{r['rk']:>3} {r['result']:<7} {r['tier']:<6} {r['pick']:<28} "
            f"{r['mark']:>6.1f} {act:>6} {r['kickoff']:<11} {r['game']}"
        )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:80]))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
