"""Cross-source identity: one real game must be one row.

The store was carrying every current-slate game twice -- once under Outlier's
``eventId`` and once under CFBD's ``cfbd:<id>`` -- because the two ingests mint
primary keys in disjoint namespaces and nothing mapped between them. The board
then listed every side twice and consensus was computed over a split book set.

These tests pin the resolution rule (and its refusals) so the namespaces cannot
drift apart again.
"""

from __future__ import annotations

import pytest
from conftest import seed_canonical_game, seed_canonical_team

from cfb_analytics.db import _merge_outlier_identities
from cfb_analytics.ingest import store
from cfb_analytics.ingest.identity import CanonicalResolver
from cfb_analytics.sources.outlier import OddsRow

SLATE = "2026-09-05"
CANONICAL_KICKOFF = "2026-09-05T23:30:00+00:00"


@pytest.fixture
def canonical(conn):
    """A canonical store shaped like the real one: CFBD teams and one game."""
    washington = seed_canonical_team(conn, 264, "Washington", "WASH",
                                     aliases=("Washington Huskies", "U-Dub", "Huskies"))
    washington_state = seed_canonical_team(conn, 265, "Washington State", "WSU")
    # Two more schools whose NICKNAME collides with Washington's, which is what
    # makes a nickname-only join unsafe: three FBS teams are the Huskies.
    seed_canonical_team(conn, 41, "UConn", "CONN", aliases=("Huskies",))
    seed_canonical_team(conn, 2459, "Northern Illinois", "NIU", aliases=("Huskies",))
    game = seed_canonical_game(conn, "cfbd:401", washington, washington_state)
    conn.commit()
    return {"home": washington, "away": washington_state, "game": game}


# Outlier names a team by its NICKNAME, not its school: `name` is "Huskies"
# where CFBD's `school` is "Washington". `market` carries the school.
OUTLIER_HOME = {"team_id": "0b32c48f", "school": "Huskies",
                "alias": "WASH", "market": "Washington"}
OUTLIER_AWAY = {"team_id": "68d2f1ec", "school": "Cougars",
                "alias": "WSU", "market": "Washington State"}


class TestTeamResolution:
    def test_resolves_on_market_matching_canonical_school(self, conn, canonical):
        resolver = CanonicalResolver(conn)
        assert resolver.team_id(OUTLIER_HOME) == canonical["home"]

    def test_resolves_on_alias_when_the_market_name_differs(self, conn, canonical):
        resolver = CanonicalResolver(conn)
        # "Wash. State" is not the canonical school string; the abbreviation is.
        team = {**OUTLIER_AWAY, "market": "Wash. State"}
        assert resolver.team_id(team) == canonical["away"]

    def test_resolves_through_a_cfbd_alternate_name(self, conn, canonical):
        resolver = CanonicalResolver(conn)
        team = {"team_id": "x", "school": "Huskies", "alias": "UW", "market": "U-Dub"}
        assert resolver.team_id(team) == canonical["home"]

    def test_refuses_a_nickname_shared_by_three_schools(self, conn, canonical):
        """'Huskies' is UConn, Northern Illinois AND Washington.

        Resolving it to whichever row sorts first is the venue-name bug in a
        new place: an ambiguous name must resolve to nothing.
        """
        resolver = CanonicalResolver(conn)
        team = {"team_id": "x", "school": "Huskies", "alias": None, "market": None}
        assert resolver.team_id(team) is None

    def test_returns_none_for_a_team_not_in_the_canonical_store(self, conn, canonical):
        resolver = CanonicalResolver(conn)
        team = {"team_id": "x", "school": "Jackrabbits",
                "alias": "SDST", "market": "South Dakota State"}
        assert resolver.team_id(team) is None


class TestGameResolution:
    def test_resolves_the_canonical_game(self, conn, canonical):
        resolver = CanonicalResolver(conn)
        match = resolver.game(home_team_id=canonical["home"],
                              away_team_id=canonical["away"], football_date=SLATE)
        assert match is not None
        assert match.game_id == canonical["game"]
        assert match.orientation_agrees is True

    def test_tolerates_a_one_day_slate_disagreement(self, conn, canonical):
        """The two feeds can straddle Eastern midnight on a late kickoff.

        A 02:30Z Sunday kick is Saturday's slate; if one source rounds the other
        way the dates differ by exactly one day for the same game.
        """
        resolver = CanonicalResolver(conn)
        match = resolver.game(home_team_id=canonical["home"],
                              away_team_id=canonical["away"],
                              football_date="2026-09-06")
        assert match is not None and match.game_id == canonical["game"]

    def test_reports_a_home_away_disagreement_instead_of_re_keying(self, conn, canonical):
        """Sides are stored as HOME/AWAY, so orientation is not cosmetic.

        Adopting CFBD's orientation silently would relabel every spread and
        moneyline price onto the wrong team.
        """
        resolver = CanonicalResolver(conn)
        match = resolver.game(home_team_id=canonical["away"],
                              away_team_id=canonical["home"], football_date=SLATE)
        assert match is not None
        assert match.orientation_agrees is False

    def test_returns_none_when_no_canonical_game_exists(self, conn, canonical):
        resolver = CanonicalResolver(conn)
        assert resolver.game(home_team_id=canonical["home"],
                             away_team_id=canonical["away"],
                             football_date="2026-10-31") is None


CAPTURED = "2026-09-04T18:00:00+00:00"


def _odds(game_id, book, price):
    return OddsRow(
        game_id=game_id, market_id=None, book=book, market="ML", side="HOME",
        line=0.0, price_american=price, price_decimal=1.5, is_primary=True,
        captured_utc=CAPTURED, source="outlier",
    )


def _consensus(conn, game_id):
    conn.execute(
        """INSERT INTO market_consensus
           (game_id, market, line, side, as_of_utc, n_books, consensus_price, anchor)
           VALUES (?, 'ML', 0.0, 'HOME', ?, 2, -150, 'all_books')""",
        (game_id, CAPTURED),
    )


@pytest.fixture
def split_game(conn, canonical):
    """The store as the bug left it: one game living under two keys."""
    store.upsert_team(conn, {"team_id": "out-wash", "school": "Huskies",
                             "alias": "WASH", "market": "Washington"})
    store.upsert_team(conn, {"team_id": "out-wsu", "school": "Cougars",
                             "alias": "WSU", "market": "Washington State"})
    store.upsert_game(conn, {
        "game_id": "evt-9", "season": 2026, "kickoff_utc": CANONICAL_KICKOFF,
        "football_date": SLATE, "day_of_week": 5,
        "home_team_id": "out-wash", "away_team_id": "out-wsu",
        "venue_name": "Husky Stadium", "network": "FOX", "status": "pregame",
    })
    # Half the books landed on each key -- which is why neither side's
    # consensus was ever computed over the whole market.
    store.insert_odds(conn, [_odds("evt-9", "FANATICS", -150),
                             _odds("evt-9", "NOVIG", -155)])
    store.insert_odds(conn, [_odds(canonical["game"], "DRAFTKINGS", -148)])
    _consensus(conn, "evt-9")
    _consensus(conn, canonical["game"])
    conn.commit()
    return canonical


class TestIdentityMigration:
    """Migration 011 folds the Outlier-keyed rows onto the canonical game."""

    def test_the_duplicate_game_row_is_gone(self, conn, split_game):
        _merge_outlier_identities(conn)
        rows = conn.execute("SELECT game_id, source FROM games").fetchall()
        assert [(r["game_id"], r["source"]) for r in rows] == [(split_game["game"], "cfbd")]

    def test_every_captured_price_survives_on_the_canonical_game(self, conn, split_game):
        """Raw observations are re-keyed, never dropped: a past capture time
        cannot be re-fetched, so losing one loses it permanently."""
        _merge_outlier_identities(conn)
        rows = conn.execute("SELECT game_id, book FROM odds_snapshots").fetchall()
        assert {r["book"] for r in rows} == {"FANATICS", "NOVIG", "DRAFTKINGS"}
        assert {r["game_id"] for r in rows} == {split_game["game"]}

    def test_snapshot_id_is_rehashed_so_re_ingest_stays_idempotent(self, conn, split_game):
        """``snapshot_id`` is a hash OF ``game_id``.

        Carrying the old id across the re-key would leave a row whose id no
        longer matches its content, so the next capture of that same unchanged
        price would hash differently and insert a second copy -- silently
        double-counting a book.
        """
        _merge_outlier_identities(conn)
        expected = _odds(split_game["game"], "FANATICS", -150).snapshot_id
        stored = conn.execute(
            "SELECT snapshot_id FROM odds_snapshots WHERE book = 'FANATICS'"
        ).fetchone()["snapshot_id"]
        assert stored == expected

        before = conn.execute("SELECT COUNT(*) AS n FROM odds_snapshots").fetchone()["n"]
        store.insert_odds(conn, [_odds(split_game["game"], "FANATICS", -150)])
        after = conn.execute("SELECT COUNT(*) AS n FROM odds_snapshots").fetchone()["n"]
        assert after == before, "re-ingesting an unchanged price must not duplicate it"

    def test_consensus_is_cleared_on_both_sides_of_the_merge(self, conn, split_game):
        """A consensus computed over half the books is wrong on both keys."""
        _merge_outlier_identities(conn)
        remaining = conn.execute("SELECT COUNT(*) AS n FROM market_consensus").fetchone()["n"]
        assert remaining == 0, "stale split-book consensus must be rebuilt, not carried"

    def test_outlier_naming_is_kept_as_an_alias(self, conn, split_game):
        _merge_outlier_identities(conn)
        aliases = {
            (r["alias"], r["alias_type"]) for r in conn.execute(
                "SELECT alias, alias_type FROM team_aliases WHERE source = 'outlier'")
        }
        assert ("Washington", "market") in aliases
        assert ("WASH", "abbreviation") in aliases

    def test_orphan_outlier_team_rows_are_removed(self, conn, split_game):
        _merge_outlier_identities(conn)
        left = conn.execute(
            "SELECT COUNT(*) AS n FROM teams WHERE cfbd_id IS NULL").fetchone()["n"]
        assert left == 0

    def test_is_idempotent(self, conn, split_game):
        _merge_outlier_identities(conn)
        first = conn.execute("SELECT COUNT(*) AS n FROM odds_snapshots").fetchone()["n"]
        _merge_outlier_identities(conn)
        second = conn.execute("SELECT COUNT(*) AS n FROM odds_snapshots").fetchone()["n"]
        assert second == first

    def test_an_unresolvable_game_is_left_alone(self, conn, canonical):
        """Not force-matched and not deleted -- it stays visible."""
        store.upsert_team(conn, {"team_id": "out-x", "school": "Jackrabbits",
                                 "alias": "SDST", "market": "South Dakota State"})
        store.upsert_team(conn, {"team_id": "out-y", "school": "Bison",
                                 "alias": "NDSU", "market": "North Dakota State"})
        store.upsert_game(conn, {
            "game_id": "evt-fcs", "season": 2026, "kickoff_utc": CANONICAL_KICKOFF,
            "football_date": SLATE, "day_of_week": 5,
            "home_team_id": "out-x", "away_team_id": "out-y",
            "venue_name": "Dakota Dome", "network": None, "status": "pregame",
        })
        conn.commit()
        _merge_outlier_identities(conn)
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM games WHERE game_id = 'evt-fcs'"
        ).fetchone()["n"] == 1
