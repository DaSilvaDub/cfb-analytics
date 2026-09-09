"""Unit and integration tests for recruiting rankings and transfer portal ingestion pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from cfb_analytics.errors import SchemaError
from cfb_analytics.features.futures import load_team_roster_inputs
from cfb_analytics.ingest import store
from cfb_analytics.ingest.cfbd_fundamentals import (
    backfill_portal,
    backfill_recruiting,
    backfill_recruiting_and_portal,
)
from cfb_analytics.models.futures import (
    NILTier,
    PortalComposite,
    QBContinuity,
    QBTier,
    build_team_portal_composites,
    calculate_player_portal_score,
)
from cfb_analytics.sources.cfbd import parse_recruiting_team, parse_transfer_player

# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------


def _team_row(cfbd_id: int, school: str) -> dict[str, object]:
    return {
        "id": cfbd_id,
        "school": school,
        "mascot": "Mascots",
        "abbreviation": school[:3].upper(),
        "alternateNames": [school],
        "conference": "Big Ten",
        "division": None,
        "classification": "fbs",
        "location": None,
    }


class FakeCfbdRecruitingPortalClient:
    def __init__(
        self,
        *,
        teams: list[dict[str, object]],
        recruiting_by_year: dict[int, list[dict[str, object]]] | None = None,
        portal_by_year: dict[int, list[dict[str, object]]] | None = None,
    ) -> None:
        self.teams = teams
        self.recruiting_by_year = recruiting_by_year or {}
        self.portal_by_year = portal_by_year or {}

    def fetch_fbs_teams(self, year: int) -> list[dict[str, object]]:
        return self.teams

    def fetch_recruiting_teams(
        self, year: int, *, team: str | None = None
    ) -> list[dict[str, object]]:
        rows = self.recruiting_by_year.get(year, [])
        if team:
            return [r for r in rows if r.get("team") == team]
        return rows

    def fetch_transfer_portal(self, year: int) -> list[dict[str, object]]:
        return self.portal_by_year.get(year, [])


# ---------------------------------------------------------------------------
# 1. Parsing Tests
# ---------------------------------------------------------------------------


class TestParseRecruitingTeam:
    def test_parses_valid_row(self) -> None:
        row = {
            "year": 2026,
            "rank": 1,
            "team": "Georgia",
            "points": 315.5,
        }
        parsed = parse_recruiting_team(row)
        assert parsed["season"] == 2026
        assert parsed["team_name"] == "Georgia"
        assert parsed["rank"] == 1
        assert parsed["points"] == 315.5
        assert parsed["recruiting_composite"] == 315.5
        assert parsed["availability_class"] == "preseason"
        assert parsed["as_of_utc"] == "2026-08-01T00:00:00+00:00"

    def test_preserves_explicit_as_of_utc(self) -> None:
        row = {"year": 2026, "rank": 2, "team": "Ohio State", "points": 305.2}
        parsed = parse_recruiting_team(row, as_of_utc="2026-08-15T00:00:00+00:00")
        assert parsed["as_of_utc"] == "2026-08-15T00:00:00+00:00"

    def test_missing_team_raises_schema_error(self) -> None:
        with pytest.raises(SchemaError):
            parse_recruiting_team({"year": 2026, "rank": 1, "points": 300.0})

    def test_invalid_year_raises_schema_error(self) -> None:
        with pytest.raises(SchemaError):
            parse_recruiting_team({"year": "not_an_int", "team": "Alabama", "points": 300.0})

    def test_optional_rank_and_points_handled(self) -> None:
        row = {"year": 2026, "team": "Akron", "rank": None, "points": None}
        parsed = parse_recruiting_team(row)
        assert parsed["rank"] is None
        assert parsed["points"] is None
        assert parsed["recruiting_composite"] is None


class TestParseTransferPlayer:
    def test_parses_valid_portal_row(self) -> None:
        row = {
            "season": 2026,
            "firstName": "Caleb",
            "lastName": "Downs",
            "position": "S",
            "origin": "Alabama",
            "destination": "Ohio State",
            "transferDate": "2026-01-20T19:30:00.000Z",
            "rating": 0.9850,
            "stars": 5,
            "eligibility": "Immediate",
        }
        parsed = parse_transfer_player(row)
        assert parsed["season"] == 2026
        assert parsed["first_name"] == "Caleb"
        assert parsed["last_name"] == "Downs"
        assert parsed["position"] == "S"
        assert parsed["origin_name"] == "Alabama"
        assert parsed["destination_name"] == "Ohio State"
        assert parsed["transfer_date"] == "2026-01-20T19:30:00+00:00"
        assert parsed["rating"] == 0.9850
        assert parsed["stars"] == 5
        assert parsed["eligibility"] == "Immediate"
        assert parsed["as_of_utc"] == "2026-01-20T19:30:00+00:00"

    def test_handles_string_rating_and_missing_optional_fields(self) -> None:
        row = {
            "season": 2026,
            "firstName": "John",
            "lastName": "Doe",
            "position": "QB",
            "origin": "MAC Team",
            "destination": None,
            "transferDate": None,
            "rating": "0.8500",
            "stars": None,
            "eligibility": None,
        }
        parsed = parse_transfer_player(row, as_of_utc="2026-08-01T00:00:00+00:00")
        assert parsed["rating"] == 0.8500
        assert parsed["destination_name"] is None
        assert parsed["transfer_date"] is None
        assert parsed["stars"] is None
        assert parsed["as_of_utc"] == "2026-08-01T00:00:00+00:00"

    def test_unparseable_transfer_date_falls_back_to_season_preseason_cutoff(self) -> None:
        row = {
            "season": 2026,
            "firstName": "Unknown",
            "lastName": "Player",
            "origin": "Alabama",
            "destination": "Ohio State",
            "transferDate": "unparseable_string_not_iso",
        }
        parsed = parse_transfer_player(row)
        assert parsed["transfer_date"] == "unparseable_string_not_iso"
        assert parsed["as_of_utc"] == "2026-08-01T00:00:00+00:00"

    def test_invalid_season_raises_schema_error(self) -> None:
        with pytest.raises(SchemaError):
            parse_transfer_player({"season": "bad_season"})


# ---------------------------------------------------------------------------
# 2. Portal Scoring & Composite Builder Tests
# ---------------------------------------------------------------------------


class TestPortalScoringAndAggregation:
    def test_calculate_player_portal_score_variants(self) -> None:
        # Decimal 0-1 scale -> scaled to 0-10
        assert calculate_player_portal_score({"rating": 0.89}) == 8.9
        assert calculate_player_portal_score({"rating": 0.985}) == 9.85
        # 0-10 scale -> unchanged
        assert calculate_player_portal_score({"rating": 8.5}) == 8.5
        # 0-100 scale -> divided by 10
        assert calculate_player_portal_score({"rating": 89.0}) == 8.9
        # Over-scale clamping
        assert calculate_player_portal_score({"rating": 120.0}) == 10.0
        # Stars fallback
        assert calculate_player_portal_score({"stars": 5}) == 10.0
        assert calculate_player_portal_score({"stars": 4}) == 8.0
        assert calculate_player_portal_score({"stars": 3}) == 6.0
        assert calculate_player_portal_score({"stars": 2}) == 4.0
        assert calculate_player_portal_score({"stars": 1}) == 2.0
        assert calculate_player_portal_score({"stars": 7}) == 10.0
        # Default baseline
        assert calculate_player_portal_score({}) == 5.0
        assert calculate_player_portal_score({"rating": None, "stars": None}) == 5.0

    def test_uncommitted_player_portal_aggregation(self) -> None:
        players = [
            {
                "origin_team_id": "cfbd:ohio_st",
                "destination_team_id": None,
                "rating": 0.90,
            }
        ]
        composites = build_team_portal_composites(
            players, season=2026, as_of_utc="2026-08-01T00:00:00+00:00"
        )
        assert len(composites) == 1
        osu = composites[0]
        assert osu["team_id"] == "cfbd:ohio_st"
        assert osu["departures_count"] == 1
        assert osu["departures_score"] == 9.0
        assert osu["additions_count"] == 0
        assert osu["additions_score"] == 0.0
        assert osu["net_composite"] == -9.0

    def test_build_team_portal_composites(self) -> None:
        players = [
            # Ohio State addition from Alabama
            {
                "origin_team_id": "cfbd:alabama",
                "destination_team_id": "cfbd:ohio_st",
                "rating": 0.95,
            },
            # Ohio State addition from Kansas State
            {
                "origin_team_id": "cfbd:kansas_st",
                "destination_team_id": "cfbd:ohio_st",
                "rating": 0.92,
            },
            # Ohio State departure to Syracuse
            {
                "origin_team_id": "cfbd:ohio_st",
                "destination_team_id": "cfbd:syracuse",
                "rating": 0.88,
            },
        ]
        composites = build_team_portal_composites(
            players,
            season=2026,
            as_of_utc="2026-08-01T00:00:00+00:00",
            fbs_team_ids=["cfbd:ohio_st", "cfbd:alabama", "cfbd:michigan"],
        )
        comp_by_team = {c["team_id"]: c for c in composites}

        # Ohio State: 2 additions (9.5 + 9.2 = 18.7), 1 departure (8.8), net = 9.9
        osu = comp_by_team["cfbd:ohio_st"]
        assert osu["additions_count"] == 2
        assert osu["departures_count"] == 1
        assert osu["additions_score"] == 18.7
        assert osu["departures_score"] == 8.8
        assert osu["net_composite"] == 9.9
        assert osu["availability_class"] == "preseason"
        assert osu["as_of_utc"] == "2026-08-01T00:00:00+00:00"

        # Alabama: 0 additions, 1 departure (9.5), net = -9.5
        bama = comp_by_team["cfbd:alabama"]
        assert bama["additions_count"] == 0
        assert bama["departures_count"] == 1
        assert bama["net_composite"] == -9.5

        # Michigan: 0 additions, 0 departures, net = 0.0
        mich = comp_by_team["cfbd:michigan"]
        assert mich["additions_count"] == 0
        assert mich["departures_count"] == 0
        assert mich["net_composite"] == 0.0


# ---------------------------------------------------------------------------
# 3. Database Store Idempotency & Persistence Tests
# ---------------------------------------------------------------------------


class TestDatabaseStorage:
    def test_insert_team_recruiting_is_idempotent(self, conn) -> None:
        store.upsert_team(
            conn, {"team_id": "cfbd:10", "school": "Georgia", "alias": "UGA", "market": "Athens"}
        )
        rows = [
            {
                "season": 2026,
                "team_id": "cfbd:10",
                "rank": 1,
                "points": 315.5,
                "recruiting_composite": 315.5,
                "availability_class": "preseason",
                "as_of_utc": "2026-08-01T00:00:00+00:00",
            }
        ]
        first = store.insert_team_recruiting(conn, rows)
        second = store.insert_team_recruiting(conn, rows)
        assert first == 1
        assert second == 0

        stored = conn.execute(
            "SELECT * FROM team_recruiting WHERE team_id = 'cfbd:10' AND season = 2026"
        ).fetchone()
        assert stored is not None
        assert stored["rank"] == 1
        assert stored["points"] == 315.5
        assert stored["recruiting_composite"] == 315.5
        assert stored["as_of_utc"] == "2026-08-01T00:00:00+00:00"

    def test_insert_transfer_portal_players_is_idempotent(self, conn) -> None:
        store.upsert_team(
            conn,
            {"team_id": "cfbd:1", "school": "Ohio State", "alias": "OSU", "market": "Columbus"},
        )
        store.upsert_team(
            conn,
            {"team_id": "cfbd:2", "school": "Alabama", "alias": "BAMA", "market": "Tuscaloosa"},
        )
        players = [
            {
                "season": 2026,
                "first_name": "Julian",
                "last_name": "Sayin",
                "position": "QB",
                "origin_team_id": "cfbd:2",
                "destination_team_id": "cfbd:1",
                "origin_name": "Alabama",
                "destination_name": "Ohio State",
                "transfer_date": "2026-01-21T18:00:00+00:00",
                "rating": 0.98,
                "stars": 5,
                "eligibility": "Immediate",
                "as_of_utc": "2026-01-21T18:00:00+00:00",
            }
        ]
        first = store.insert_transfer_portal_players(conn, players)
        second = store.insert_transfer_portal_players(conn, players)
        assert first == 1
        assert second == 0

        stored = conn.execute(
            "SELECT * FROM transfer_portal_players WHERE season = 2026"
        ).fetchall()
        assert len(stored) == 1
        assert stored[0]["first_name"] == "Julian"
        assert stored[0]["rating"] == 0.98
        assert stored[0]["destination_team_id"] == "cfbd:1"

    def test_insert_team_portal_composites_is_idempotent(self, conn) -> None:
        store.upsert_team(
            conn,
            {"team_id": "cfbd:1", "school": "Ohio State", "alias": "OSU", "market": "Columbus"},
        )
        rows = [
            {
                "season": 2026,
                "team_id": "cfbd:1",
                "additions_score": 25.0,
                "departures_score": 5.0,
                "net_composite": 20.0,
                "additions_count": 3,
                "departures_count": 1,
                "availability_class": "preseason",
                "as_of_utc": "2026-08-01T00:00:00+00:00",
            }
        ]
        first = store.insert_team_portal_composites(conn, rows)
        second = store.insert_team_portal_composites(conn, rows)
        assert first == 1
        assert second == 0

        stored = conn.execute(
            "SELECT * FROM team_portal_composites WHERE team_id = 'cfbd:1' AND season = 2026"
        ).fetchone()
        assert stored is not None
        assert stored["net_composite"] == 20.0
        assert stored["additions_count"] == 3


# ---------------------------------------------------------------------------
# 4. Backfill Functions Tests
# ---------------------------------------------------------------------------


class TestBackfillRecruitingAndPortal:
    def test_backfill_recruiting_pipeline(self, conn) -> None:
        teams = [_team_row(101, "Ohio State"), _team_row(102, "Michigan")]
        recruiting_data = {
            2026: [
                {"year": 2026, "team": "Ohio State", "rank": 1, "points": 310.0},
                {"year": 2026, "team": "Michigan", "rank": 5, "points": 275.0},
                {"year": 2026, "team": "Non-FBS College", "rank": 150, "points": 50.0},
            ]
        }
        client: Any = FakeCfbdRecruitingPortalClient(
            teams=teams, recruiting_by_year=recruiting_data
        )

        summary = backfill_recruiting(conn, client, start_year=2026, end_year=2026)
        assert summary.seasons == 1
        assert summary.recruiting == 2
        assert summary.filtered == 1

        stored = conn.execute("SELECT COUNT(*) AS n FROM team_recruiting").fetchone()["n"]
        assert stored == 2

    def test_backfill_portal_pipeline(self, conn) -> None:
        teams = [_team_row(101, "Ohio State"), _team_row(102, "Michigan")]
        portal_data = {
            2026: [
                {
                    "season": 2026,
                    "firstName": "Will",
                    "lastName": "Howard",
                    "position": "QB",
                    "origin": "Kansas State",
                    "destination": "Ohio State",
                    "transferDate": "2026-01-05T12:00:00Z",
                    "rating": 0.91,
                    "stars": 4,
                },
                {
                    "season": 2026,
                    "firstName": "Player",
                    "lastName": "Two",
                    "position": "WR",
                    "origin": "Ohio State",
                    "destination": "Michigan",
                    "transferDate": "2026-01-10T12:00:00Z",
                    "rating": 0.89,
                    "stars": 4,
                },
            ]
        }
        client: Any = FakeCfbdRecruitingPortalClient(teams=teams, portal_by_year=portal_data)

        summary = backfill_portal(conn, client, start_year=2026, end_year=2026)
        assert summary.seasons == 1
        assert summary.players == 2
        assert summary.composites >= 2

        players_stored = conn.execute(
            "SELECT COUNT(*) AS n FROM transfer_portal_players"
        ).fetchone()["n"]
        assert players_stored == 2

        comp_stored = conn.execute(
            "SELECT COUNT(*) AS n FROM team_portal_composites"
        ).fetchone()["n"]
        assert comp_stored >= 2

    def test_backfill_recruiting_and_portal_combined(self, conn) -> None:
        teams = [_team_row(101, "Ohio State")]
        rec_data = {2026: [{"year": 2026, "team": "Ohio State", "rank": 1, "points": 300.0}]}
        portal_data = {
            2026: [
                {
                    "season": 2026,
                    "firstName": "A",
                    "lastName": "B",
                    "position": "WR",
                    "origin": None,
                    "destination": "Ohio State",
                    "rating": 0.90,
                }
            ]
        }
        client: Any = FakeCfbdRecruitingPortalClient(
            teams=teams, recruiting_by_year=rec_data, portal_by_year=portal_data
        )

        rec_summary, portal_summary = backfill_recruiting_and_portal(
            conn, client, start_year=2026, end_year=2026
        )
        assert rec_summary.recruiting == 1
        assert portal_summary.players == 1

    def test_backfill_invalid_years_raises(self, conn) -> None:
        client: Any = FakeCfbdRecruitingPortalClient(teams=[])
        with pytest.raises(SchemaError):
            backfill_recruiting(conn, client, start_year=2026, end_year=2025)
        with pytest.raises(SchemaError):
            backfill_portal(conn, client, start_year=2026, end_year=2025)


# ---------------------------------------------------------------------------
# 5. Model 4 Features Wiring Tests
# ---------------------------------------------------------------------------


class TestFuturesFeatureLayerWiring:
    @staticmethod
    def _seed_base_data(conn) -> None:
        store.upsert_team(
            conn,
            {
                "team_id": "cfbd:ohio_st",
                "cfbd_id": 194,
                "school": "Ohio State",
                "conference": "Big Ten",
            },
        )
        store.upsert_team_season(
            conn,
            {
                "team_id": "cfbd:ohio_st",
                "season": 2026,
                "source": "cfbd",
                "conference": "Big Ten",
            },
        )
        store.insert_returning_production(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "availability_class": "preseason",
                    "percent_ppa": 0.75,
                }
            ],
        )

    def test_auto_loads_recruiting_and_portal_composites_from_db(self, conn) -> None:
        self._seed_base_data(conn)
        # Seed recruiting rankings
        store.insert_team_recruiting(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "rank": 2,
                    "points": 980.0,
                    "recruiting_composite": 980.0,
                    "availability_class": "preseason",
                    "as_of_utc": "2026-08-01T00:00:00+00:00",
                }
            ],
        )
        # Seed team portal composite
        store.insert_team_portal_composites(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "additions_score": 30.0,
                    "departures_score": 10.0,
                    "net_composite": 20.0,
                    "additions_count": 3,
                    "departures_count": 1,
                    "availability_class": "preseason",
                    "as_of_utc": "2026-08-01T00:00:00+00:00",
                }
            ],
        )

        manifest: dict[str, object] = {}
        inputs = load_team_roster_inputs(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2026-08-15T00:00:00+00:00",
            nil_tier=NILTier.TIER_1_ELITE,
            qb_tier=QBTier.TIER_1_ELITE,
            qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
            strength_of_schedule=5.0,
            input_manifest=manifest,
        )

        # Loaded automatically without passing portal_composite or recruiting_composite
        assert inputs.recruiting_composite == 980.0
        assert isinstance(inputs.portal_composite, PortalComposite)
        assert inputs.portal_composite.net_composite == 20.0
        assert inputs.portal_composite.additions_score == 30.0
        assert inputs.portal_composite.departures_score == 10.0
        assert inputs.portal_composite.additions_count == 3
        assert inputs.portal_composite.departures_count == 1
        assert "team_recruiting" in manifest
        assert "team_portal" in manifest

    def test_aggregates_portal_composite_from_players_table(self, conn) -> None:
        self._seed_base_data(conn)
        store.upsert_team(
            conn,
            {"team_id": "cfbd:kansas_st", "cfbd_id": 2306, "school": "Kansas State"},
        )
        store.upsert_team(
            conn,
            {"team_id": "cfbd:syracuse", "cfbd_id": 183, "school": "Syracuse"},
        )
        store.insert_team_talent(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "availability_class": "preseason",
                    "talent_composite": 950.0,
                }
            ],
        )
        # Store individual player movements but NO team_portal_composites row
        store.insert_transfer_portal_players(
            conn,
            [
                {
                    "season": 2026,
                    "first_name": "Player",
                    "last_name": "In",
                    "origin_team_id": "cfbd:kansas_st",
                    "destination_team_id": "cfbd:ohio_st",
                    "rating": 0.90,  # score 9.0
                    "as_of_utc": "2026-05-01T00:00:00+00:00",
                },
                {
                    "season": 2026,
                    "first_name": "Player",
                    "last_name": "Out",
                    "origin_team_id": "cfbd:ohio_st",
                    "destination_team_id": "cfbd:syracuse",
                    "rating": 0.80,  # score 8.0
                    "as_of_utc": "2026-05-02T00:00:00+00:00",
                },
            ],
        )

        inputs = load_team_roster_inputs(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2026-08-15T00:00:00+00:00",
            nil_tier=NILTier.TIER_2_UPPER_P4,
            qb_tier=QBTier.TIER_2_QUALITY_STARTER,
            qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
            strength_of_schedule=3.0,
        )

        # Recruiting fell back to team_talent (950.0)
        assert inputs.recruiting_composite == 950.0
        # Portal was aggregated from transfer_portal_players: 9.0 - 8.0 = 1.0
        assert isinstance(inputs.portal_composite, PortalComposite)
        assert inputs.portal_composite.additions_score == 9.0
        assert inputs.portal_composite.departures_score == 8.0
        assert inputs.portal_composite.net_composite == 1.0
        assert inputs.portal_composite.additions_count == 1
        assert inputs.portal_composite.departures_count == 1

    def test_caller_overrides_take_precedence(self, conn) -> None:
        self._seed_base_data(conn)
        store.insert_team_recruiting(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "points": 700.0,
                    "availability_class": "preseason",
                    "as_of_utc": "2026-08-01T00:00:00+00:00",
                }
            ],
        )
        store.insert_team_portal_composites(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "additions_score": 10.0,
                    "departures_score": 10.0,
                    "net_composite": 0.0,
                    "availability_class": "preseason",
                    "as_of_utc": "2026-08-01T00:00:00+00:00",
                }
            ],
        )

        override_portal = PortalComposite(additions_score=40.0, departures_score=10.0)
        inputs = load_team_roster_inputs(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2026-08-15T00:00:00+00:00",
            portal_composite=override_portal,
            recruiting_composite=999.0,
            nil_tier=NILTier.TIER_1_ELITE,
            qb_tier=QBTier.TIER_1_ELITE,
            qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
            strength_of_schedule=5.0,
        )

        assert inputs.recruiting_composite == 999.0
        assert inputs.portal_composite == override_portal
        assert inputs.portal_composite.net_composite == 30.0

    def test_as_of_point_in_time_filtering_prevents_leakage(self, conn) -> None:
        self._seed_base_data(conn)
        # Future recruiting record dated August 20
        store.insert_team_recruiting(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "points": 990.0,
                    "availability_class": "preseason",
                    "as_of_utc": "2026-08-20T00:00:00+00:00",
                }
            ],
        )
        # Future portal record dated August 20
        store.insert_team_portal_composites(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "additions_score": 50.0,
                    "departures_score": 5.0,
                    "net_composite": 45.0,
                    "availability_class": "preseason",
                    "as_of_utc": "2026-08-20T00:00:00+00:00",
                }
            ],
        )

        # Query as of August 10 -> future rows must be excluded
        with pytest.raises(SchemaError):
            load_team_roster_inputs(
                conn,
                "cfbd:ohio_st",
                2026,
                as_of_utc="2026-08-10T00:00:00+00:00",
                nil_tier=NILTier.TIER_1_ELITE,
                qb_tier=QBTier.TIER_1_ELITE,
                qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
                strength_of_schedule=5.0,
            )

    def test_missing_portal_data_raises_schema_error(self, conn) -> None:
        self._seed_base_data(conn)
        store.insert_team_talent(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "availability_class": "preseason",
                    "talent_composite": 900.0,
                }
            ],
        )
        with pytest.raises(SchemaError, match="Portal composite is required"):
            load_team_roster_inputs(
                conn,
                "cfbd:ohio_st",
                2026,
                nil_tier=NILTier.TIER_1_ELITE,
                qb_tier=QBTier.TIER_1_ELITE,
                qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
                strength_of_schedule=5.0,
            )

    def test_december_transfer_is_inadmissible_for_september_projection(self, conn) -> None:
        """P1 regression: December transfers must not be admissible for September projections."""
        self._seed_base_data(conn)
        store.insert_team_talent(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "availability_class": "preseason",
                    "talent_composite": 980.0,
                }
            ],
        )
        # Transfer occurs in December 2026
        store.insert_transfer_portal_players(
            conn,
            [
                {
                    "season": 2026,
                    "first_name": "Winter",
                    "last_name": "Transfer",
                    "position": "QB",
                    "destination_team_id": "cfbd:ohio_st",
                    "origin_name": "Other",
                    "destination_name": "Ohio State",
                    "transfer_date": "2026-12-15T12:00:00+00:00",
                    "rating": 0.95,
                    "stars": 4,
                    "as_of_utc": "2026-12-15T12:00:00+00:00",
                }
            ],
        )
        # September projection: December transfer is inadmissible; no other portal rows -> error
        with pytest.raises(SchemaError, match="Portal composite is required"):
            load_team_roster_inputs(
                conn,
                "cfbd:ohio_st",
                2026,
                as_of_utc="2026-09-01T00:00:00+00:00",
                nil_tier=NILTier.TIER_1_ELITE,
                qb_tier=QBTier.TIER_1_ELITE,
                qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
                strength_of_schedule=5.0,
            )

        # December 20 projection: transfer is now admissible
        inputs = load_team_roster_inputs(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2026-12-20T00:00:00+00:00",
            nil_tier=NILTier.TIER_1_ELITE,
            qb_tier=QBTier.TIER_1_ELITE,
            qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
            strength_of_schedule=5.0,
        )
        assert isinstance(inputs.portal_composite, PortalComposite)
        assert inputs.portal_composite.additions_count == 1

    def test_roster_talent_composite_prioritized_over_recruiting_class_points(
        self, conn
    ) -> None:
        """P1 regression: team_talent (980 -> +18.15) must be prioritized over
        recruiting points (315.5 -> -18.40).
        """
        self._seed_base_data(conn)
        # Seed recruiting class points (annual class points, e.g. 315.5)
        store.insert_team_recruiting(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "rank": 1,
                    "points": 315.5,
                    "recruiting_composite": None,
                    "availability_class": "preseason",
                    "as_of_utc": "2026-08-01T00:00:00+00:00",
                }
            ],
        )
        # Seed roster talent composite (650 baseline, 980 for top team)
        store.insert_team_talent(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "availability_class": "preseason",
                    "talent_composite": 980.0,
                }
            ],
        )
        store.insert_team_portal_composites(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "additions_score": 10.0,
                    "departures_score": 5.0,
                    "net_composite": 5.0,
                    "additions_count": 1,
                    "departures_count": 1,
                    "availability_class": "preseason",
                    "as_of_utc": "2026-08-01T00:00:00+00:00",
                }
            ],
        )

        manifest: dict[str, Any] = {}
        inputs = load_team_roster_inputs(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2026-08-15T00:00:00+00:00",
            nil_tier=NILTier.TIER_1_ELITE,
            qb_tier=QBTier.TIER_1_ELITE,
            qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
            strength_of_schedule=5.0,
            input_manifest=manifest,
        )

        assert inputs.recruiting_composite == 980.0
        # Model 4 baseline is 650.0, scale factor 0.055: (980 - 650) * 0.055 = +18.15
        expected_spread_adj = round((inputs.recruiting_composite - 650.0) * 0.055, 2)
        assert expected_spread_adj == 18.15
        assert "team_talent" in manifest
        assert "team_recruiting" in manifest

    def test_corrected_player_ratings_are_not_double_counted(self, conn) -> None:
        """P1 regression: updated/corrected transfer portal player records
        must not double-count scores.
        """
        self._seed_base_data(conn)
        store.insert_team_talent(
            conn,
            [
                {
                    "season": 2026,
                    "team_id": "cfbd:ohio_st",
                    "availability_class": "preseason",
                    "talent_composite": 980.0,
                }
            ],
        )
        # Initial transfer record (rating 0.85)
        store.insert_transfer_portal_players(
            conn,
            [
                {
                    "season": 2026,
                    "first_name": "Will",
                    "last_name": "Howard",
                    "position": "QB",
                    "destination_team_id": "cfbd:ohio_st",
                    "origin_name": "Kansas State",
                    "destination_name": "Ohio State",
                    "transfer_date": "2026-01-05T12:00:00+00:00",
                    "rating": 0.85,
                    "stars": 4,
                    "as_of_utc": "2026-01-05T12:00:00+00:00",
                }
            ],
        )
        # Corrected transfer record (rating updated to 0.95 with later as_of)
        store.insert_transfer_portal_players(
            conn,
            [
                {
                    "season": 2026,
                    "first_name": "Will",
                    "last_name": "Howard",
                    "position": "QB",
                    "destination_team_id": "cfbd:ohio_st",
                    "origin_name": "Kansas State",
                    "destination_name": "Ohio State",
                    "transfer_date": "2026-01-05T12:00:00+00:00",
                    "rating": 0.95,
                    "stars": 4,
                    "as_of_utc": "2026-01-10T12:00:00+00:00",
                }
            ],
        )

        inputs = load_team_roster_inputs(
            conn,
            "cfbd:ohio_st",
            2026,
            as_of_utc="2026-08-15T00:00:00+00:00",
            nil_tier=NILTier.TIER_1_ELITE,
            qb_tier=QBTier.TIER_1_ELITE,
            qb_continuity=QBContinuity.RETURNING_STARTER_SAME_SYSTEM,
            strength_of_schedule=5.0,
        )
        assert isinstance(inputs.portal_composite, PortalComposite)
        assert inputs.portal_composite.additions_count == 1
        expected_score = calculate_player_portal_score({"rating": 0.95, "stars": 4})
        assert inputs.portal_composite.additions_score == expected_score
