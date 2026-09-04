"""The scheduled daily job.

Designed for an unattended cloud runner, which drives three rules:

1. **Degrade, don't abort.** A source without a credential, or one that has
   expired, is reported and skipped. One dead source must not cost the run the
   data the other sources would have produced.
2. **Say what did not happen.** A silent partial run is worse than a failed
   one, so every skip is named in the summary with the reason and the fix.
3. **Only ingest slates inside the operating window.** Measured 2026-09-01:
   within ~5 days of kickoff every game carries 11-12 moneyline books; at 11
   days out the median game carries one and 36 of 42 fall below the consensus
   floor. Fetching further out spends API calls on markets that cannot be
   devigged anyway.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from cfb_analytics import config
from cfb_analytics.errors import CfbAnalyticsError
from cfb_analytics.utils import FOOTBALL_TZ, football_date, utc_now_iso


@dataclass
class SourceOutcome:
    name: str
    status: str  # "ok" | "skipped" | "failed"
    detail: str = ""
    rows: int = 0

    @property
    def symbol(self) -> str:
        return {"ok": "OK", "skipped": "SKIP", "failed": "FAIL"}.get(self.status, "?")


@dataclass
class DailyReport:
    started_utc: str
    slates: list[str] = field(default_factory=list)
    outcomes: list[SourceOutcome] = field(default_factory=list)
    market_rows: int = 0
    movement_rows: int = 0
    games: int = 0
    bootstrapped: bool = False

    @property
    def ok(self) -> bool:
        """False only when something actively broke and nothing succeeded.

        A *skipped* source is an expected state -- a credential that was never
        configured -- so an all-skipped run is green. Turning a known-missing
        key into a red build every morning trains people to ignore the light,
        which costs more than it catches. A *failed* source with no successes
        is red, because something broke that was expected to work.
        """
        statuses = {outcome.status for outcome in self.outcomes}
        return "ok" in statuses or "failed" not in statuses

    def as_text(self) -> str:
        lines = [
            f"cfb-analytics daily run  {self.started_utc}",
            f"  slates in window : {', '.join(self.slates) if self.slates else '(none)'}",
            f"  games            : {self.games}",
            "",
            "  sources:",
        ]
        for outcome in self.outcomes:
            lines.append(f"    {outcome.symbol:<5} {outcome.name:<10} {outcome.detail}")
        lines += [
            "",
            f"  consensus rows   : {self.market_rows}",
            f"  movement rows    : {self.movement_rows}",
        ]
        if config.is_shadow_mode():
            lines.append(f"\n  {config.SHADOW_STAMP}")
        return "\n".join(lines)


def slates_in_window(conn: sqlite3.Connection, *, now: datetime | None = None) -> list[str]:
    """Stored slate dates from today up to the operating-window horizon."""
    horizon_days = int(config.settings()["market"].get("max_days_to_kickoff_for_report", 5))
    moment = now or datetime.now(FOOTBALL_TZ)
    today = football_date(moment.astimezone(FOOTBALL_TZ).isoformat())
    latest = (moment + timedelta(days=horizon_days)).astimezone(FOOTBALL_TZ).date().isoformat()
    rows = conn.execute(
        """SELECT DISTINCT football_date FROM games
           WHERE football_date >= ? AND football_date <= ?
           ORDER BY football_date""",
        (today, latest),
    ).fetchall()
    return [str(row["football_date"]) for row in rows]


def games_for_season(conn: sqlite3.Connection, season: int) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM games WHERE season = ?", (season,)
        ).fetchone()["n"]
    )


def _bootstrap_if_empty(conn: sqlite3.Connection, report: DailyReport, season: int) -> None:
    """Populate this season's schedule when the store comes up empty.

    A fresh cloud runner restores from the ``data`` branch; on the very first
    run that branch does not exist, so the store has no games and CFBD lines
    have nothing to attach to. Rather than depending on someone remembering to
    seed it by hand, the job notices and fixes itself.

    Deliberately **this season only**. The 2014-2025 historical load is for the
    model and the backtest, not for pricing tonight's slate; running twelve
    seasons of fetches inside a daily cron would spend minutes and a lot of API
    calls every morning for data that never changes. Trigger that separately
    with `backfill-cfbd`.
    """
    if games_for_season(conn, season) > 0:
        return
    if not config.has_cfbd_key():
        report.outcomes.append(SourceOutcome(
            "bootstrap", "skipped",
            f"store has no {season} games and no {config.CFBD_ENV_VAR} to fetch them"))
        return
    try:
        from cfb_analytics.ingest.cfbd_ingest import backfill_years
        from cfb_analytics.sources.cfbd import CFBDClient

        summary = backfill_years(conn, CFBDClient(), start_year=season, end_year=season)
        report.bootstrapped = True
        report.outcomes.append(SourceOutcome(
            "bootstrap", "ok",
            f"store was empty for {season}; loaded {summary.games} games, "
            f"{summary.teams} teams, {summary.venues} venues",
            rows=summary.games))
    except CfbAnalyticsError as exc:
        report.outcomes.append(SourceOutcome("bootstrap", "failed", str(exc)[:200]))


def _run_cfbd_lines(conn: sqlite3.Connection, report: DailyReport, season: int) -> None:
    if not config.has_cfbd_key():
        report.outcomes.append(SourceOutcome(
            "cfbd", "skipped",
            f"no {config.CFBD_ENV_VAR} configured. {config.CFBD_HOW}"))
        return
    try:
        from cfb_analytics.ingest.cfbd_lines import ingest_lines
        from cfb_analytics.sources.cfbd import CFBDClient

        summary = ingest_lines(conn, CFBDClient(), season)
        report.movement_rows += summary.movement_rows
        report.outcomes.append(SourceOutcome(
            "cfbd", "ok",
            f"{summary.odds_rows} odds rows, {summary.movement_rows} movement rows, "
            f"{summary.games_matched} stored games priced "
            f"(feed carries {summary.games_seen} season-wide)",
            rows=summary.odds_rows))
    except CfbAnalyticsError as exc:
        report.outcomes.append(SourceOutcome("cfbd", "failed", str(exc)[:200]))


def _run_outlier(conn: sqlite3.Connection, report: DailyReport, slates: list[str]) -> None:
    """Outlier is best-effort in a scheduled context.

    Its access token lives 24 hours and is refreshed by an interactive
    Playwright + email-OTP login, so an unattended runner will find it expired
    on the second day. The leg stays because a locally-run daily job (or a
    freshly uploaded session) still benefits, but a 403 here is expected
    rather than alarming and must not fail the run.
    """
    from cfb_analytics.errors import AuthRequiredError

    try:
        from cfb_analytics.ingest.outlier_ingest import ingest_slate
        from cfb_analytics.sources.outlier import OutlierClient

        client = OutlierClient()
        total = 0
        for slate in slates:
            total += ingest_slate(conn, client, slate).odds_rows
        report.outcomes.append(SourceOutcome(
            "outlier", "ok", f"{total} odds rows across {len(slates)} slate(s)", rows=total))
    except AuthRequiredError:
        report.outcomes.append(SourceOutcome(
            "outlier", "skipped",
            "session expired (access token lives 24h). Refresh it locally in the "
            "outlier project; unattended refresh needs email OTP."))
    except CfbAnalyticsError as exc:
        report.outcomes.append(SourceOutcome("outlier", "failed", str(exc)[:200]))


def run_daily(
    conn: sqlite3.Connection,
    *,
    season: int | None = None,
    with_outlier: bool = True,
    bootstrap: bool = True,
    now: datetime | None = None,
) -> DailyReport:
    report = DailyReport(started_utc=utc_now_iso())
    moment = now or datetime.now(FOOTBALL_TZ)
    year = season or moment.year

    if bootstrap:
        # Must precede the lines ingest: lines attach to stored games, so
        # an empty store would silently price nothing.
        _bootstrap_if_empty(conn, report, year)

    _run_cfbd_lines(conn, report, year)

    slates = slates_in_window(conn, now=moment)
    if with_outlier and slates:
        _run_outlier(conn, report, slates)
    elif with_outlier:
        report.outcomes.append(SourceOutcome(
            "outlier", "skipped", "no stored slates inside the operating window"))

    report.slates = slates_in_window(conn, now=moment)
    if report.slates:
        from cfb_analytics.features.build_market import build_market_for_slate

        for slate in report.slates:
            summary = build_market_for_slate(conn, slate)
            report.market_rows += summary.consensus_rows
            report.movement_rows += summary.movement_rows
            report.games += summary.games

    return report
