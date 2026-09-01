"""Small shared helpers: time, identity hashing, and odds conversion."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cfb_analytics.errors import ConfigError

try:
    # College football's day boundary is the US Eastern calendar date, not UTC.
    # Windows ships no IANA database, hence the `tzdata` dependency.
    FOOTBALL_TZ = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError as exc:  # pragma: no cover - environment defect
    raise ConfigError(
        "No IANA time-zone database found. Install the 'tzdata' package: "
        "slate dates are US Eastern calendar dates and cannot be computed without it."
    ) from exc


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def to_utc_iso(value: Any) -> str | None:
    """Normalise a timestamp to a UTC ISO-8601 string.

    Accepts the offset forms the Outlier feed emits (``2026-09-05T19:30:00-0700``
    and ``...-07:00``) as well as trailing ``Z``. Returns None when the value is
    absent or unparseable — the caller decides whether that is fatal, because a
    silently defaulted timestamp would corrupt every as-of join downstream.
    """
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # "-0700" -> "-07:00"; fromisoformat only accepts the colon form before 3.11.
    if len(text) >= 5 and text[-5] in "+-" and text[-3] != ":":
        text = f"{text[:-2]}:{text[-2:]}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="seconds")


def football_date(kickoff_utc: str) -> str:
    """The slate a kickoff belongs to: its **US Eastern** calendar date.

    Not the UTC date. Verified against the live feed on 2026-08-31: grouping the
    2026-09-05 Saturday slate by UTC date is wrong in both directions -- it pulls
    in four Friday-night games kicking 00:00-01:00Z on the 5th, and drops four
    Saturday-night West Coast games (UCLA@CAL, WKU@NEV, UNLV@HAW, CMU@UNM) that
    kick 02:00-02:30Z on the 6th. That is 8 of a 34-game slate misassigned.

    Eastern reproduces the feed's own ``dayOfWeek`` code exactly, which
    ``weekday_matches_feed`` uses as a cross-check.
    """
    parsed = datetime.fromisoformat(kickoff_utc)
    return parsed.astimezone(FOOTBALL_TZ).date().isoformat()


def weekday_matches_feed(kickoff_utc: str, day_of_week: Any) -> bool | None:
    """Does the Eastern weekday agree with the feed's ``dayOfWeek`` code?

    The feed's codes match ``date.weekday()`` (Mon=0 ... Sun=6). Returns None
    when the feed omits the code, so a missing value is not read as a mismatch.
    """
    if day_of_week in (None, ""):
        return None
    try:
        expected = int(day_of_week)
    except (TypeError, ValueError):
        return None
    parsed = datetime.fromisoformat(kickoff_utc).astimezone(FOOTBALL_TZ)
    return parsed.weekday() == expected


def stable_id(*parts: Any) -> str:
    """Deterministic id from its parts.

    Used for ``odds_snapshots.snapshot_id`` so that re-ingesting an unchanged
    observation collides with the existing row instead of duplicating it. The
    full digest is kept: truncating and then claiming uniqueness by construction
    is a bug waiting for a birthday collision.
    """
    payload = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def american_to_decimal(price: Any) -> float | None:
    """American odds to decimal. Returns None for absent or nonsensical input."""
    if price in (None, ""):
        return None
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    if value == 0 or value != value:
        return None
    if value > 0:
        return 1.0 + value / 100.0
    return 1.0 + 100.0 / abs(value)


def decimal_to_american(decimal_price: Any) -> int | None:
    try:
        value = float(decimal_price)
    except (TypeError, ValueError):
        return None
    if value <= 1.0:
        return None
    if value >= 2.0:
        return round((value - 1.0) * 100.0)
    return round(-100.0 / (value - 1.0))


def implied_probability(price_american: Any) -> float | None:
    """Raw (vig-inclusive) implied probability from an American price."""
    decimal_price = american_to_decimal(price_american)
    if decimal_price is None:
        return None
    return 1.0 / decimal_price
