"""Open-Meteo source and the weather ingest."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cfb_analytics.errors import SchemaError, SourceError
from cfb_analytics.ingest import store
from cfb_analytics.ingest.weather_ingest import ingest_weather
from cfb_analytics.sources.weather import (
    WeatherClient,
    choose_endpoint,
    observation_at,
)

KICKOFF = "2026-09-05T23:30:00+00:00"
NOW = datetime(2026, 9, 4, 12, tzinfo=UTC)


def hourly(start_hour=20, hours=6, **overrides):
    # Real Open-Meteo hours roll into the next day; naive "start+i" arithmetic
    # produces T24:00, which is not a valid timestamp.
    base = datetime(2026, 9, 5, start_hour, tzinfo=UTC)
    times = [
        (base + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)
    ]
    series = {
        "time": times,
        "temperature_2m": [18.0 + i for i in range(hours)],
        "wind_speed_10m": [10.0 + i for i in range(hours)],
        "wind_gusts_10m": [20.0 + i for i in range(hours)],
        "wind_direction_10m": [180.0] * hours,
        "precipitation": [0.0] * hours,
        "precipitation_probability": [5.0] * hours,
        "relative_humidity_2m": [60.0] * hours,
    }
    series.update(overrides)
    return series


class TestChooseEndpoint:
    def test_upcoming_game_uses_the_forecast(self):
        assert choose_endpoint(KICKOFF, now=NOW) == "forecast"

    def test_game_beyond_the_horizon_has_no_forecast(self):
        assert choose_endpoint("2026-11-01T23:30:00+00:00", now=NOW) is None

    def test_old_game_uses_the_reanalysis_archive(self):
        assert choose_endpoint("2026-08-01T23:30:00+00:00", now=NOW) == "archive"

    def test_recent_past_game_falls_in_the_reanalysis_gap(self):
        """Yesterday's game: forecast is spent, ERA5 has not published yet.
        That gap is a real answer, not something to interpolate over."""
        assert choose_endpoint("2026-09-03T23:30:00+00:00", now=NOW) is None

    def test_naive_timestamp_is_treated_as_utc(self):
        assert choose_endpoint("2026-09-05T23:30:00", now=NOW) == "forecast"


class TestObservationAt:
    def test_picks_the_hour_nearest_kickoff(self):
        obs = observation_at(hourly(), KICKOFF)
        # 23:30 kickoff sits between the 23:00 and 00:00 samples; 23:00 is nearest.
        assert obs.observed_hour_utc.startswith("2026-09-05T23:00")
        assert obs.temp_c == pytest.approx(21.0)
        assert obs.wind_kph == pytest.approx(13.0)

    def test_returns_none_when_the_series_does_not_reach_kickoff(self):
        """A day of data that stops at 18:00 cannot describe a 23:30 kickoff."""
        assert observation_at(hourly(start_hour=8, hours=6), KICKOFF) is None

    def test_empty_series_is_none(self):
        assert observation_at({"time": []}, KICKOFF) is None

    def test_missing_field_becomes_none_not_zero(self):
        series = hourly()
        del series["precipitation"]
        obs = observation_at(series, KICKOFF)
        assert obs.precip_mm is None, "a missing field must not read as 0mm of rain"

    def test_unparseable_timestamp_is_refused(self):
        assert observation_at(hourly(time=["not-a-time"]), KICKOFF) is None


class StubHttp:
    def __init__(self, payload):
        self.payload = payload
        self.urls: list[str] = []

    def get_payload(self, url):
        self.urls.append(url)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class TestWeatherClient:
    def test_pins_units_explicitly(self):
        http = StubHttp({"hourly": hourly()})
        WeatherClient(http=http).fetch_hourly(35.0, -97.0, "2026-09-05", archive=False)
        url = http.urls[0]
        assert "temperature_unit=celsius" in url
        assert "wind_speed_unit=kmh" in url
        assert "precipitation_unit=mm" in url
        assert "timezone=UTC" in url

    def test_archive_flag_switches_the_endpoint(self):
        http = StubHttp({"hourly": hourly()})
        client = WeatherClient(http=http)
        client.fetch_hourly(35.0, -97.0, "2026-08-01", archive=True)
        client.fetch_hourly(35.0, -97.0, "2026-09-05", archive=False)
        assert "archive-api" in http.urls[0]
        assert "archive-api" not in http.urls[1]

    def test_missing_hourly_block_raises(self):
        with pytest.raises(SchemaError, match="hourly"):
            WeatherClient(http=StubHttp({"latitude": 35.0})).fetch_hourly(
                35.0, -97.0, "2026-09-05", archive=False)


@pytest.fixture
def seeded(conn):
    store.upsert_venue(conn, {
        "venue_id": "v-out", "name": "Open Air Field", "city": "Norman", "state": "OK",
        "latitude": 35.205, "longitude": -97.442, "elevation_m": 350.0,
        "surface": "grass", "dome": 0, "capacity": 80000, "timezone": "America/Chicago"})
    store.upsert_venue(conn, {
        "venue_id": "v-dome", "name": "Indoor Dome", "city": "Syracuse", "state": "NY",
        "latitude": 43.036, "longitude": -76.136, "elevation_m": 120.0,
        "surface": "turf", "dome": 1, "capacity": 49000, "timezone": "America/New_York"})
    store.upsert_venue(conn, {
        "venue_id": "v-nogeo", "name": "Unmapped Field", "city": None, "state": None,
        "latitude": None, "longitude": None, "elevation_m": None,
        "surface": None, "dome": 0, "capacity": None, "timezone": None})
    for tid, name in (("h", "Home U"), ("a", "Away U")):
        store.upsert_team(conn, {"team_id": tid, "school": name, "alias": name, "market": name})
    return conn


def add_game(conn, game_id, venue_id, kickoff=KICKOFF, slate="2026-09-05"):
    conn.execute(
        """INSERT INTO games (game_id, season, kickoff_utc, football_date, home_team_id,
                              away_team_id, venue_id, source, ingested_utc)
           VALUES (?, 2026, ?, ?, 'h', 'a', ?, 'cfbd', 'x')""",
        (game_id, kickoff, slate, venue_id))


class TestIngestWeather:
    def test_writes_an_observation_for_an_outdoor_game(self, seeded):
        add_game(seeded, "g-out", "v-out")
        summary = ingest_weather(
            seeded, ["2026-09-05"],
            client=WeatherClient(http=StubHttp({"hourly": hourly()})), now=NOW)
        assert summary.written == 1
        row = seeded.execute("SELECT * FROM weather WHERE game_id='g-out'").fetchone()
        assert row["wind_kph"] == pytest.approx(13.0)
        assert row["is_indoor"] == 0
        assert row["is_forecast"] == 1

    def test_dome_is_recorded_as_indoor_rather_than_skipped(self, seeded):
        """'Roof' and 'we could not fetch' are different facts. A dome is a
        KNOWN absence of wind; a missing row is an unknown."""
        add_game(seeded, "g-dome", "v-dome")
        summary = ingest_weather(
            seeded, ["2026-09-05"],
            client=WeatherClient(http=StubHttp({"hourly": hourly()})), now=NOW)
        assert summary.indoor == 1
        assert summary.written == 0
        row = seeded.execute("SELECT * FROM weather WHERE game_id='g-dome'").fetchone()
        assert row["is_indoor"] == 1
        assert row["wind_kph"] is None

    def test_venue_without_coordinates_is_counted_not_guessed(self, seeded):
        add_game(seeded, "g-nogeo", "v-nogeo")
        summary = ingest_weather(
            seeded, ["2026-09-05"],
            client=WeatherClient(http=StubHttp({"hourly": hourly()})), now=NOW)
        assert summary.no_coordinates == 1
        assert summary.written == 0

    def test_game_with_no_venue_link_is_counted(self, seeded):
        add_game(seeded, "g-novenue", None)
        summary = ingest_weather(
            seeded, ["2026-09-05"],
            client=WeatherClient(http=StubHttp({"hourly": hourly()})), now=NOW)
        assert summary.no_venue == 1

    def test_game_beyond_the_forecast_horizon_is_reported(self, seeded):
        add_game(seeded, "g-far", "v-out",
                 kickoff="2026-11-01T23:30:00+00:00", slate="2026-11-01")
        summary = ingest_weather(
            seeded, ["2026-11-01"],
            client=WeatherClient(http=StubHttp({"hourly": hourly()})), now=NOW)
        assert summary.outside_window == 1
        assert summary.written == 0

    def test_fetch_failure_is_counted_and_does_not_abort_the_slate(self, seeded):
        add_game(seeded, "g-a", "v-out")
        add_game(seeded, "g-b", "v-out")
        client = WeatherClient(http=StubHttp(SourceError("open-meteo down")))
        summary = ingest_weather(seeded, ["2026-09-05"], client=client, now=NOW)
        assert summary.failed == 2
        assert summary.written == 0

    def test_rerun_at_the_same_stamp_does_not_duplicate(self, seeded):
        add_game(seeded, "g-out", "v-out")
        client = WeatherClient(http=StubHttp({"hourly": hourly()}))
        for _ in range(2):
            ingest_weather(seeded, ["2026-09-05"], client=client, now=NOW,
                           as_of_utc="2026-09-04T12:00:00+00:00")
        n = seeded.execute("SELECT COUNT(*) n FROM weather").fetchone()["n"]
        assert n == 1

    def test_a_later_capture_is_a_new_row(self, seeded):
        """Forecasts move; the store keeps each capture so the backtest can ask
        what was known at any point before kickoff."""
        add_game(seeded, "g-out", "v-out")
        client = WeatherClient(http=StubHttp({"hourly": hourly()}))
        ingest_weather(seeded, ["2026-09-05"], client=client, now=NOW,
                       as_of_utc="2026-09-04T12:00:00+00:00")
        ingest_weather(seeded, ["2026-09-05"], client=client, now=NOW,
                       as_of_utc="2026-09-05T12:00:00+00:00")
        n = seeded.execute("SELECT COUNT(*) n FROM weather").fetchone()["n"]
        assert n == 2

    def test_records_hours_to_kickoff(self, seeded):
        add_game(seeded, "g-out", "v-out")
        ingest_weather(seeded, ["2026-09-05"],
                       client=WeatherClient(http=StubHttp({"hourly": hourly()})), now=NOW)
        row = seeded.execute("SELECT hours_to_kick FROM weather").fetchone()
        assert row["hours_to_kick"] == pytest.approx(35.5, abs=0.1)

    def test_empty_slate_list_does_nothing(self, seeded):
        assert ingest_weather(seeded, [], now=NOW).games_considered == 0


class TestVenueResolution:
    """Outlier supplies a venue name only; the link must never be guessed."""

    def test_unique_name_resolves(self, seeded):
        assert store.resolve_venue_id_by_name(seeded, "Open Air Field") == "v-out"

    def test_ambiguous_name_resolves_to_none(self, seeded):
        store.upsert_venue(seeded, {
            "venue_id": "v-dup1", "name": "Memorial Stadium", "city": "A", "state": "A",
            "latitude": 1.0, "longitude": 1.0, "elevation_m": None, "surface": None,
            "dome": 0, "capacity": None, "timezone": None})
        store.upsert_venue(seeded, {
            "venue_id": "v-dup2", "name": "Memorial Stadium", "city": "B", "state": "B",
            "latitude": 50.0, "longitude": 50.0, "elevation_m": None, "surface": None,
            "dome": 0, "capacity": None, "timezone": None})
        assert store.resolve_venue_id_by_name(seeded, "Memorial Stadium") is None

    def test_unknown_name_is_none(self, seeded):
        assert store.resolve_venue_id_by_name(seeded, "Nowhere Field") is None

    def test_missing_name_is_none(self, seeded):
        assert store.resolve_venue_id_by_name(seeded, None) is None

    def test_outlier_game_links_itself_on_insert(self, seeded):
        store.upsert_game(seeded, {
            "game_id": "g-link", "season": 2026, "kickoff_utc": KICKOFF,
            "football_date": "2026-09-05", "day_of_week": 5,
            "home_team_id": "h", "away_team_id": "a",
            "venue_name": "Open Air Field", "network": None, "status": "pregame"})
        row = seeded.execute("SELECT venue_id FROM games WHERE game_id='g-link'").fetchone()
        assert row["venue_id"] == "v-out"
