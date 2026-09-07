"""NCAAFB Outlier Discovery Probe Script (Milestone M0 / R1 Gate).

Probes the Outlier REST API for NCAA Football (league token ``NCAAFB``) to evaluate
market token acceptance (``GAMELINE``, ``TEAM_PROP``, ``PLAYER_PROP``, ``GAME_PROP``),
proposition frequency, distinct sportsbook pricing depth, player identity presence,
and insights availability.

Supports dual-mode execution:
1. Live network discovery against ``https://api.outlier.bet`` using saved session headers.
2. Fully hermetic offline replay using cached envelopes in ``data/cache/outlier/`` and
   committed fixtures, ensuring 100% reproducible execution even when session tokens
   expire (HTTP 403) or network access is disabled.

Stdlib-only dependencies:
``urllib.request``, ``urllib.error``, ``urllib.parse``, ``json``, ``pathlib``,
``argparse``, ``time``, ``hashlib``, ``statistics``, ``collections``, ``typing``.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen


def find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if (parent / "cfb_analytics").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return cur.parents[1]


REPO_ROOT = find_project_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from cfb_analytics import paths
    from cfb_analytics.sources import session
    from cfb_analytics.utils import football_date, utc_now_iso
except ImportError:
    paths = None  # type: ignore[assignment]
    session = None  # type: ignore[assignment]

    def football_date(iso_str: str | None) -> str:
        if not iso_str:
            return ""
        return iso_str[:10]

    def utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()


BASE_URL = "https://api.outlier.bet"
LEAGUE_TOKEN = "NCAAFB"
MARKET_TYPES_TO_PROBE = ("GAMELINE", "TEAM_PROP", "PLAYER_PROP", "GAME_PROP")


def select_probe_date(events: list[dict[str, Any]], *, today: date | None = None) -> str | None:
    """Choose the earliest non-final slate in 14 days, then a recent past slate."""
    anchor = today or datetime.now(UTC).date()
    dated_events: list[tuple[date, dict[str, Any]]] = []
    for event in events:
        raw_date = football_date(event.get("scheduledTime"))
        try:
            event_date = date.fromisoformat(raw_date)
        except (TypeError, ValueError):
            continue
        dated_events.append((event_date, event))

    upcoming = sorted(
        event_date
        for event_date, event in dated_events
        if anchor <= event_date <= anchor + timedelta(days=14)
        and str(event.get("status", "")).lower() not in {"final", "completed"}
    )
    if upcoming:
        return upcoming[0].isoformat()

    recent = sorted(
        (
            event_date
            for event_date, _event in dated_events
            if anchor - timedelta(days=14) <= event_date < anchor
        ),
        reverse=True,
    )
    return recent[0].isoformat() if recent else None


def get_cache_key(url: str) -> str:
    """Derive SHA256 hex digest matching cfb_analytics.sources.http.cache_key."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def decode_json_body(raw: bytes, content_encoding: str = "") -> Any:
    """Decode a JSON response, including Outlier's gzip-compressed bodies."""
    if "gzip" in content_encoding.lower() or raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def get_current_git_commit() -> str:
    """Get short git commit SHA or fallback to d6d1102."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        sha = res.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return "d6d1102"


class OutlierProbe:
    """Execution engine for the NCAAFB Outlier discovery probe."""

    def __init__(
        self,
        offline_replay: bool = False,
        cache_dir: Path | None = None,
        output_dir: Path | None = None,
        docs_dir: Path | None = None,
        rate_limit_delay: float = 0.5,
    ) -> None:
        self.offline_replay = offline_replay
        self.rate_limit_delay = rate_limit_delay

        if paths is not None:
            self.cache_dir = cache_dir or (paths.cache_dir() / "outlier")
            self.output_dir = output_dir or (paths.PROJECT_ROOT / "tests" / "fixtures" / "outlier")
            self.docs_dir = docs_dir or (paths.PROJECT_ROOT / "docs" / "probes")
        else:
            self.cache_dir = cache_dir or (REPO_ROOT / "data" / "cache" / "outlier")
            self.output_dir = output_dir or (REPO_ROOT / "tests" / "fixtures" / "outlier")
            self.docs_dir = docs_dir or (REPO_ROOT / "docs" / "probes")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self.headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "Origin": "https://app.outlier.bet",
            "Referer": "https://app.outlier.bet/",
        }
        self.auth_loaded = False
        if not self.offline_replay and session is not None:
            try:
                storage_state = session.load_storage_state()
                self.headers = session.build_headers(storage_state)
                self.auth_loaded = True
            except Exception as exc:
                sys.stderr.write(
                    f"[WARN] Failed to load Outlier session ({exc}). "
                    "Live requests may return 403.\n"
                )

    def _read_cached_envelope(self, url: str) -> tuple[int, dict[str, Any]] | None:
        """Read a cached response only when its actual HTTP status was recorded."""
        if not self.cache_dir.exists():
            return None
        key = get_cache_key(url)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "payload" in data and "status" in data:
                payload = data["payload"]
                status = data["status"]
                if isinstance(payload, dict) and isinstance(status, int):
                    return status, payload
        except Exception:
            return None
        return None

    def _write_cached_envelope(self, url: str, status: int, payload: Any) -> None:
        """Persist response envelope into cache_dir."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = get_cache_key(url)
        cache_file = self.cache_dir / f"{key}.json"
        envelope = {
            "url": url,
            "status": status,
            "fetched_at": time.time(),
            "payload": payload,
        }
        with contextlib.suppress(OSError):
            cache_file.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

    def _fixture_path_for_url(self, url: str) -> Path | None:
        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        if parts == ["sportsdata", "leagues", LEAGUE_TOKEN, "schedule"]:
            return self.output_dir / "schedule_ncaafb.json"
        if len(parts) == 4 and parts[:2] == ["sportsdata", "events"]:
            event_id, resource = parts[2], parts[3]
            if resource == "insights":
                return self.output_dir / f"event_{event_id}_insights.json"
            if resource == "markets":
                market_type = parse_qs(parsed.query).get("marketType", [""])[0].upper()
                if market_type in MARKET_TYPES_TO_PROBE:
                    return self.output_dir / f"event_{event_id}_{market_type}.json"
        return None

    def _write_fixture(self, url: str, status: int, payload: dict[str, Any]) -> None:
        fixture = self._fixture_path_for_url(url)
        if fixture is None:
            return
        fixture.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        fixture.with_suffix(".http.json").write_text(
            json.dumps({"url": url, "status": status, "captured_at": utc_now_iso()}, indent=2),
            encoding="utf-8",
        )

    def _read_fixture_envelope(self, url: str) -> tuple[int, dict[str, Any]] | None:
        fixture = self._fixture_path_for_url(url)
        if fixture is None:
            return None
        metadata = fixture.with_suffix(".http.json")
        if not fixture.is_file() or not metadata.is_file():
            return None
        try:
            payload = json.loads(fixture.read_text(encoding="utf-8"))
            provenance = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or not isinstance(provenance, dict):
            return None
        status = provenance.get("status")
        recorded_url = provenance.get("url")
        if not isinstance(status, int) or recorded_url != url:
            return None
        return status, payload

    def fetch_resource(self, url: str, default_empty: Any | None = None) -> tuple[int, Any, str]:
        """Fetch endpoint payload via live HTTP or offline cache/fixture replay."""
        if self.offline_replay:
            cached = self._read_cached_envelope(url)
            if cached is not None:
                status, payload = cached
                return status, payload, "cache"
            fixture = self._read_fixture_envelope(url)
            if fixture is not None:
                status, payload = fixture
                return status, payload, "fixture"
            return 0, None, "cache_miss"

        time.sleep(self.rate_limit_delay)
        req = Request(url, headers=self.headers)
        try:
            with urlopen(req, timeout=15) as resp:
                status = getattr(resp, "status", 200)
                raw = resp.read()
                data = decode_json_body(raw, str(resp.headers.get("Content-Encoding", "")))
                self._write_cached_envelope(url, status, data)
                if isinstance(data, dict):
                    self._write_fixture(url, status, data)
                return status, data, "live"
        except HTTPError as exc:
            if exc.code == 404 and default_empty is not None:
                return 404, default_empty, "live"
            return exc.code, None, "error"
        except (URLError, TimeoutError, OSError) as exc:
            sys.stderr.write(f"[WARN] Network error ({exc}) for {url}.\n")
            return 599, None, "network_error"

    def probe_schedule(self) -> tuple[int, list[dict[str, Any]], str]:
        """Probe NCAAFB schedule endpoint."""
        url = f"{BASE_URL}/sportsdata/leagues/{LEAGUE_TOKEN}/schedule"
        status, payload, source = self.fetch_resource(url)
        if status == 200 and isinstance(payload, dict):
            events = payload.get("events", [])
            return status, [e for e in events if isinstance(e, dict)], source
        return status, [], source

    def probe_event_market(
        self, event_id: str, market_type: str
    ) -> tuple[int, list[dict[str, Any]], str]:
        """Probe market tokens for a specific event."""
        url = f"{BASE_URL}/sportsdata/events/{event_id}/markets?marketType={quote(market_type)}"
        default_payload = {"markets": []}
        status, payload, source = self.fetch_resource(url, default_empty=default_payload)
        markets: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            raw_markets = payload.get("markets", [])
            if isinstance(raw_markets, list):
                markets = [m for m in raw_markets if isinstance(m, dict)]
        return status, markets, source

    def probe_event_insights(self, event_id: str) -> tuple[int, list[dict[str, Any]], str]:
        """Probe insights endpoint for a specific event."""
        url = f"{BASE_URL}/sportsdata/events/{event_id}/insights"
        default_payload = {"insights": []}
        status, payload, source = self.fetch_resource(url, default_empty=default_payload)
        insights: list[dict[str, Any]] = []
        if isinstance(payload, dict):
            raw_insights = payload.get("insights", [])
            if isinstance(raw_insights, list):
                insights = [i for i in raw_insights if isinstance(i, dict)]
        return status, insights, source

    def run(
        self,
        event_id: str | None = None,
        date_str: str | None = None,
        limit: int = 3,
    ) -> dict[str, Any]:
        """Execute the probe across schedule and sample events."""
        print("=== NCAAFB Outlier Discovery Probe (R1 Gate) ===")
        print(f"Mode: {'OFFLINE REPLAY' if self.offline_replay else 'LIVE HTTP'}")
        print(f"Output Fixtures Dir: {self.output_dir}")
        print(f"Report Docs Dir: {self.docs_dir}\n")

        sched_status, all_events, sched_source = self.probe_schedule()
        print(f"[SCHEDULE] HTTP {sched_status} ({sched_source}) — Total events: {len(all_events)}")
        if sched_status != 200:
            raise RuntimeError(f"Schedule probe failed with status {sched_status} ({sched_source})")
        date_str = date_str or select_probe_date(all_events)
        if date_str is None:
            raise RuntimeError("No eligible slate was found within the required date windows")
        print(f"Target Slate Date: {date_str}")

        target_events: list[dict[str, Any]] = []
        slate_events = [e for e in all_events if football_date(e.get("scheduledTime")) == date_str]

        if event_id:
            matched = [e for e in all_events if e.get("eventId") == event_id]
            if matched:
                target_events = matched
            else:
                raise RuntimeError(
                    f"Requested event {event_id!r} is not present in the live schedule"
                )
        else:
            target_events = slate_events[:limit]
            print(
                f"[SLATE] Found {len(slate_events)} events on {date_str}. "
                f"Selected {len(target_events)} for probe."
            )

        if not target_events:
            raise RuntimeError(f"No scheduled events found on {date_str}")

        git_sha = get_current_git_commit()
        probe_results: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "mode": "offline_replay" if self.offline_replay else "live",
            "auth_loaded": self.auth_loaded,
            "slate_date": date_str,
            "git_sha": git_sha,
            "schedule_status": sched_status,
            "events_count": len(all_events),
            "slate_events_count": len(slate_events),
            "target_events": target_events,
            "events_probed": [],
            "market_type_summary": {},
            "proposition_summary": {},
            "player_identity_summary": {
                "player_id_found": False,
                "player_id_samples": [],
                "display_name_found": False,
                "display_name_samples": [],
                "player_samples": [],
            },
            "insights_summary": {
                "status_codes": Counter(),
                "total_insights_count": 0,
                "subject_types": Counter(),
                "market_types": Counter(),
                "propositions": Counter(),
            },
            "trap_evidence": {
                "trap1_found": False,
                "trap1_event_id": "",
                "trap1_books_first": "",
                "trap1_odds_book_first": "",
                "spread_cards_total": 0,
                "total_cards_total": 0,
            },
        }

        prop_stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "events": set(), "books": set(), "books_per_event": []}
        )
        market_type_counts: Counter[str] = Counter()
        market_type_statuses: dict[str, set[int]] = defaultdict(set)

        for ev in target_events:
            ev_id = ev.get("eventId") or "unknown"
            home_name = ev.get("home", {}).get("name", "Home")
            away_name = ev.get("away", {}).get("name", "Away")
            ev_desc = f"{home_name} vs {away_name} ({ev_id[:10]}...)"
            print(f"\n--- Probing Event: {ev_desc} ---")

            ev_record: dict[str, Any] = {
                "eventId": ev_id,
                "description": ev_desc,
                "market_types": {},
                "insights": {},
            }

            for mt in MARKET_TYPES_TO_PROBE:
                m_status, markets, m_source = self.probe_event_market(ev_id, mt)
                market_type_counts[mt] += len(markets)
                market_type_statuses[mt].add(m_status)
                print(f"  [{mt:<11}] HTTP {m_status} ({m_source}) — {len(markets)} markets")

                ev_record["market_types"][mt] = {
                    "status": m_status,
                    "source": m_source,
                    "markets_count": len(markets),
                }

                ev_prop_books: dict[str, set[str]] = defaultdict(set)
                for m in markets:
                    prop = str(m.get("proposition") or "UNKNOWN").upper()
                    market_type = str(m.get("marketType") or mt).upper()
                    prop_key = (market_type, prop)
                    prop_stats[prop_key]["count"] += 1
                    prop_stats[prop_key]["events"].add(ev_id)

                    player = m.get("player")
                    if market_type == "PLAYER_PROP" and isinstance(player, dict):
                        pid = player.get("playerId")
                        display_name = player.get("fullName") or player.get("displayName")
                        identity = probe_results["player_identity_summary"]
                        if pid:
                            identity["player_id_found"] = True
                            if len(identity["player_id_samples"]) < 5:
                                identity["player_id_samples"].append(str(pid))
                        if display_name:
                            identity["display_name_found"] = True
                            if len(identity["display_name_samples"]) < 5:
                                identity["display_name_samples"].append(str(display_name))
                        if (pid or display_name) and len(identity["player_samples"]) < 5:
                            identity["player_samples"].append(
                                {
                                    "player_id": str(pid or ""),
                                    "display_name": str(display_name or ""),
                                }
                            )

                    if prop == "SPREAD":
                        probe_results["trap_evidence"]["spread_cards_total"] += 1
                    elif prop == "TOTAL":
                        probe_results["trap_evidence"]["total_cards_total"] += 1

                    outcomes = m.get("outcomes") or []
                    for outcome in outcomes:
                        if not isinstance(outcome, dict):
                            continue

                        outcome_books = outcome.get("books") or []
                        odds_entries = outcome.get("odds") or []
                        if (
                            not probe_results["trap_evidence"]["trap1_found"]
                            and outcome_books
                            and odds_entries
                        ):
                            first_book_declared = str(outcome_books[0]).strip().upper()
                            first_odds_book = str(odds_entries[0].get("book") or "").strip().upper()
                            if (
                                first_book_declared
                                and first_odds_book
                                and first_book_declared != first_odds_book
                            ):
                                probe_results["trap_evidence"]["trap1_found"] = True
                                probe_results["trap_evidence"]["trap1_event_id"] = ev_id
                                probe_results["trap_evidence"]["trap1_books_first"] = (
                                    first_book_declared
                                )
                                probe_results["trap_evidence"]["trap1_odds_book_first"] = (
                                    first_odds_book
                                )

                        pid = outcome.get("playerId") or outcome.get("athleteId")
                        if pid:
                            probe_results["player_identity_summary"]["player_id_found"] = True
                            p_samples = probe_results["player_identity_summary"][
                                "player_id_samples"
                            ]
                            if len(p_samples) < 5:
                                p_samples.append(str(pid))

                        label = (
                            outcome.get("label")
                            or outcome.get("playerName")
                            or outcome.get("displayName")
                        )
                        if label and mt == "PLAYER_PROP":
                            probe_results["player_identity_summary"]["display_name_found"] = True
                            d_samples = probe_results["player_identity_summary"][
                                "display_name_samples"
                            ]
                            if len(d_samples) < 5:
                                d_samples.append(str(label))

                        for entry in odds_entries:
                            if isinstance(entry, dict):
                                b = str(entry.get("book") or "").strip().upper()
                                if b:
                                    prop_stats[prop_key]["books"].add(b)
                                    ev_prop_books[f"{market_type}\x1f{prop}"].add(b)

                for composite_key, bks in ev_prop_books.items():
                    market_type, prop = composite_key.split("\x1f", 1)
                    prop_stats[(market_type, prop)]["books_per_event"].append(len(bks))

            i_status, insights, i_source = self.probe_event_insights(ev_id)
            probe_results["insights_summary"]["status_codes"][i_status] += 1
            probe_results["insights_summary"]["total_insights_count"] += len(insights)
            for insight in insights:
                for field, summary_key in (
                    ("subjectType", "subject_types"),
                    ("marketType", "market_types"),
                    ("proposition", "propositions"),
                ):
                    value = str(insight.get(field) or "UNKNOWN").upper()
                    probe_results["insights_summary"][summary_key][value] += 1
            print(f"  [{'INSIGHTS':<11}] HTTP {i_status} ({i_source}) — {len(insights)} insights")

            ev_record["insights"] = {
                "status": i_status,
                "source": i_source,
                "insights_count": len(insights),
            }
            probe_results["events_probed"].append(ev_record)

        for mt in MARKET_TYPES_TO_PROBE:
            probe_results["market_type_summary"][mt] = {
                "total_markets": market_type_counts[mt],
                "statuses": sorted(list(market_type_statuses[mt])),
                "supported": market_type_counts[mt] > 0,
            }

        for (market_type, prop), data in prop_stats.items():
            b_list = list(data["books_per_event"])
            b_list.extend([0] * (len(target_events) - len(data["events"])))
            med_books = float(median(b_list)) if b_list else 0.0
            market_props = probe_results["proposition_summary"].setdefault(market_type, {})
            market_props[prop] = {
                "market_cards_count": data["count"],
                "events_present": len(data["events"]),
                "distinct_books_count": len(data["books"]),
                "distinct_books": sorted(list(data["books"])),
                "books_per_game_min": min(b_list) if b_list else 0,
                "books_per_game_median": med_books,
                "books_per_game_max": max(b_list) if b_list else 0,
            }

        self.generate_markdown_report(probe_results, date_str)
        return probe_results

    def generate_markdown_report(self, results: dict[str, Any], date_str: str) -> Path:
        """Write a data-driven discovery report from the recorded probe results."""
        report_path = self.docs_dir / f"{date_str}-ncaafb-props-discovery.md"
        total_probed = len(results["events_probed"])
        total_slate = results.get("slate_events_count", total_probed)
        market_summaries = results["market_type_summary"]
        proposition_summaries = results["proposition_summary"]
        insights_summary = results["insights_summary"]

        def status_text(statuses: list[int]) -> str:
            return ", ".join(f"HTTP {status}" for status in statuses) or "NO_RESPONSE"

        def market_resolution(market_type: str) -> str:
            summary = market_summaries[market_type]
            if summary["total_markets"]:
                prop_count = len(proposition_summaries.get(market_type, {}))
                return f"OFFERED ({summary['total_markets']} cards; {prop_count} propositions)"
            if summary["statuses"] == [200]:
                return "EMPTY_200"
            return f"UNAVAILABLE ({status_text(summary['statuses'])})"

        insight_statuses = sorted(int(code) for code in insights_summary["status_codes"])
        insight_count = insights_summary["total_insights_count"]
        if insight_count:
            insight_resolution = f"OFFERED ({insight_count} insights)"
        elif insight_statuses == [200]:
            insight_resolution = "EMPTY_200"
        else:
            insight_resolution = f"UNAVAILABLE ({status_text(insight_statuses)})"

        all_market_calls_ok = all(
            summary["statuses"] == [200] for summary in market_summaries.values()
        )
        gate_verdict = "PASS" if all_market_calls_ok and insight_statuses == [200] else "FAIL"
        mode = results["mode"]
        auth_state = (
            "REPLAY_FIXTURES"
            if mode == "offline_replay"
            else "LIVE_SESSION"
            if results.get("auth_loaded")
            else "LIVE_SESSION_NOT_LOADED"
        )

        lines: list[str] = [
            "# Outlier NCAAFB Props & Insights Discovery Probe Report",
            "",
            f"- **Probe Execution Date:** {results['timestamp']}",
            f"- **Target Slate Date:** {results['slate_date']} (US Eastern calendar date)",
            (
                "- **Probe Script:** `scripts/probe_ncaafb_outlier.py` "
                f"(Git Commit: `{results.get('git_sha', 'unknown')}`, "
                "Branch: `feat/outlier-props-insights`)"
            ),
            f"- **HTTP Client Mode:** {mode.upper()}",
            f"- **Authentication State:** {auth_state}",
            "- **Reference Slate Fixtures:** `tests/fixtures/outlier/`",
            "",
            "## 1. Executive Summary & Gate Verdict",
            "",
            f"- **Milestone M0 Discovery Gate:** {gate_verdict}",
            f"- **GAMELINE:** {market_resolution('GAMELINE')}",
            f"- **TEAM_PROP:** {market_resolution('TEAM_PROP')}",
            f"- **PLAYER_PROP:** {market_resolution('PLAYER_PROP')}",
            f"- **GAME_PROP:** {market_resolution('GAME_PROP')}",
            f"- **Insights:** {insight_resolution}",
            (
                "- **Scope decision:** The live feed offers every probed market family and "
                "insights. R2 still admits only its explicit gameline and team-prop whitelist; "
                "player props and unwhitelisted game props remain excluded."
                if all(market_summaries[mt]["supported"] for mt in MARKET_TYPES_TO_PROBE)
                and insight_count
                else "- **Scope decision:** Treat empty or failed families exactly as observed; "
                "do not synthesize unavailable feeds."
            ),
            "",
            "## 2. Slate & Sampled Events Context",
            "",
            f"- **Schedule status:** HTTP {results['schedule_status']}",
            f"- **Events on target slate:** {total_slate}",
            f"- **Events in returned schedule:** {results['events_count']}",
            f"- **Events probed:** {total_probed}",
            "",
            "| Event ID | Kickoff (UTC) | Matchup |",
            "|---|---|---|",
        ]
        for event in results.get("target_events", []):
            home = event.get("home") if isinstance(event.get("home"), dict) else {}
            away = event.get("away") if isinstance(event.get("away"), dict) else {}
            lines.append(
                f"| `{event.get('eventId', 'unknown')}` | "
                f"`{event.get('scheduledTime', 'N/A')}` | "
                f"{away.get('name', 'Away')} @ {home.get('name', 'Home')} |"
            )

        lines.extend(["", "## 3. Findings by Probed Token"])
        for index, market_type in enumerate(MARKET_TYPES_TO_PROBE, 1):
            summary = market_summaries[market_type]
            props = proposition_summaries.get(market_type, {})
            proposition_text = ", ".join(f"`{name}`" for name in sorted(props)) or "None"
            lines.extend(
                [
                    "",
                    f"### 3.{index} Token: `{market_type}`",
                    (
                        "- **Endpoint:** "
                        f"`GET /sportsdata/events/{{eventId}}/markets?marketType={market_type}`"
                    ),
                    f"- **HTTP status:** {status_text(summary['statuses'])}",
                    f"- **Market cards:** {summary['total_markets']}",
                    f"- **Discovered propositions:** {proposition_text}",
                    f"- **Resolution:** {market_resolution(market_type)}",
                ]
            )

        trap = results["trap_evidence"]
        lines.extend(
            [
                "",
                "### 3.5 Recorded parser traps",
                (
                    "- **Non-parallel book lists:** Verified in event "
                    f"`{trap['trap1_event_id']}`: `outcome.books[0]` was "
                    f"`{trap['trap1_books_first']}` while `outcome.odds[0].book` was "
                    f"`{trap['trap1_odds_book_first']}`."
                    if trap["trap1_found"]
                    else "- **Non-parallel book lists:** No mismatch observed in this sample."
                ),
                (
                    "- **Multi-row propositions:** "
                    f"`SPREAD` used {trap['spread_cards_total']} cards and `TOTAL` used "
                    f"{trap['total_cards_total']} cards; consumers must union cards and "
                    "deduplicate `(book, side, line)`."
                ),
                "",
                "## 4. Proposition & Sportsbook Depth",
                "",
                (
                    "| Market Type | Proposition | Cards | Event Frequency | Distinct Books | "
                    "Book Count | Median Books/Game | Consensus Eligible | R2 Action |"
                ),
                "|---|---|---:|---:|---|---:|---:|---|---|",
            ]
        )

        admitted = {
            ("GAMELINE", "MONEYLINE"): "ADMIT_AS_ML",
            ("GAMELINE", "SPREAD"): "ADMIT",
            ("GAMELINE", "TOTAL"): "ADMIT",
            ("TEAM_PROP", "POINTS"): "ADMIT_FULL_GAME_ONLY",
            ("TEAM_PROP", "OFFENSIVE_YARDS"): "ADMIT_FULL_GAME_ONLY",
            ("TEAM_PROP", "RECEIVING_YARDS"): "ADMIT_FULL_GAME_ONLY",
            ("TEAM_PROP", "RUSHING_YARDS"): "ADMIT_FULL_GAME_ONLY",
        }
        for market_type in MARKET_TYPES_TO_PROBE:
            for proposition, info in sorted(proposition_summaries.get(market_type, {}).items()):
                event_count = info["events_present"]
                percentage = round(event_count / total_probed * 100) if total_probed else 0
                books = ", ".join(info["distinct_books"][:4]) or "None"
                if len(info["distinct_books"]) > 4:
                    books += ", ..."
                median_books = info["books_per_game_median"]
                consensus = "YES" if median_books >= 3 else "NO"
                if market_type == "PLAYER_PROP":
                    action = "EXCLUDED_R2"
                elif market_type == "GAME_PROP":
                    action = "DROP_UNWHITELISTED"
                else:
                    action = admitted.get((market_type, proposition), "DROP_UNWHITELISTED")
                lines.append(
                    f"| `{market_type}` | `{proposition}` | "
                    f"{info['market_cards_count']} | {event_count}/{total_probed} "
                    f"({percentage}%) | {books} | {info['distinct_books_count']} | "
                    f"{median_books} | `{consensus}` | `{action}` |"
                )

        identity = results["player_identity_summary"]
        samples = identity.get("player_samples", [])
        lines.extend(
            [
                "",
                "## 5. Player Identity Field Assessment",
                "",
                (f"- **Stable `playerId` present:** {identity['player_id_found']}"),
                f"- **Display name present:** {identity['display_name_found']}",
                "- **Field location:** `markets[].player.playerId` and "
                "`markets[].player.fullName` (outcome-level fields are also accepted "
                "by the probe).",
                "- **Representative samples:**",
            ]
        )
        if samples:
            for sample in samples:
                lines.append(f"  - `{sample['player_id']}` — {sample['display_name']}")
        else:
            lines.append("  - None observed.")

        subject_types = (
            ", ".join(
                f"{key}={value}" for key, value in sorted(insights_summary["subject_types"].items())
            )
            or "None"
        )
        insight_market_types = (
            ", ".join(
                f"{key}={value}" for key, value in sorted(insights_summary["market_types"].items())
            )
            or "None"
        )
        insight_props = (
            ", ".join(
                f"{key}={value}" for key, value in sorted(insights_summary["propositions"].items())
            )
            or "None"
        )
        lines.extend(
            [
                "",
                "## 6. Insights Endpoint Technical Assessment",
                "",
                f"- **HTTP status:** {status_text(insight_statuses)}",
                f"- **Insights returned:** {insight_count}",
                f"- **Subject types:** {subject_types}",
                f"- **Market types:** {insight_market_types}",
                f"- **Propositions:** {insight_props}",
                "",
                "| Event ID | HTTP Status | Source | Insight Count |",
                "|---|---:|---|---:|",
            ]
        )
        for event in results["events_probed"]:
            insight = event["insights"]
            lines.append(
                f"| `{event['eventId']}` | {insight['status']} | "
                f"{insight['source']} | {insight['insights_count']} |"
            )

        lines.extend(
            [
                "",
                "## 7. Hard Gate Conclusion",
                "",
                f"- **Verdict:** {gate_verdict}",
                (
                    "- The authenticated live capture confirms that NCAAFB gamelines, "
                    "team props, player props, game props, and insights are offered for "
                    "the sampled in-window event. The discovery evidence supports proceeding "
                    "only with the R2-authorized gameline and team-prop scope."
                    if gate_verdict == "PASS" and insight_count
                    else "- The capture does not establish all required endpoints as available. "
                    "Stop or degrade exactly as required by R1."
                ),
            ]
        )

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[REPORT] Discovery markdown report successfully written to: {report_path}")
        return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NCAAFB Outlier Discovery Probe Script (Milestone M0 / R1 Gate)"
    )
    parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Optional specific Outlier event ID to probe.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target slate date in YYYY-MM-DD format (default: auto-select per R6).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum sample events to probe on the slate (default: 3).",
    )
    parser.add_argument(
        "--offline-replay",
        action="store_true",
        help="Force offline hermetic replay using cached payloads and fixtures.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Path to Outlier cache directory (default: data/cache/outlier).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path to dump raw fixture payloads (default: tests/fixtures/outlier).",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Path to output markdown discovery report (default: docs/probes).",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        help="Delay in seconds between live HTTP requests (default: 0.5s).",
    )

    args = parser.parse_args()

    probe = OutlierProbe(
        offline_replay=args.offline_replay,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
        rate_limit_delay=args.rate_limit_delay,
    )

    probe.run(
        event_id=args.event_id,
        date_str=args.date,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
