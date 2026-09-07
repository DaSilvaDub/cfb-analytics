from __future__ import annotations

import gzip
import json
from datetime import date

from scripts.probe_ncaafb_outlier import (
    OutlierProbe,
    decode_json_body,
    get_cache_key,
    select_probe_date,
)


def test_replay_cache_miss_does_not_fabricate_http_200(tmp_path):
    probe = OutlierProbe(
        offline_replay=True,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "fixtures",
        docs_dir=tmp_path / "docs",
    )

    status, payload, source = probe.fetch_resource(
        "https://example.test/missing", default_empty={"markets": []}
    )

    assert (status, payload, source) == (0, None, "cache_miss")


def test_replay_requires_recorded_http_status(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    url = "https://example.test/legacy"
    cache_file = cache_dir / f"{get_cache_key(url)}.json"
    cache_file.write_text(json.dumps({"payload": {"markets": []}}), encoding="utf-8")
    probe = OutlierProbe(
        offline_replay=True,
        cache_dir=cache_dir,
        output_dir=tmp_path / "fixtures",
        docs_dir=tmp_path / "docs",
    )

    assert probe.fetch_resource(url) == (0, None, "cache_miss")


def test_replay_preserves_recorded_http_status(tmp_path):
    probe = OutlierProbe(
        offline_replay=True,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "fixtures",
        docs_dir=tmp_path / "docs",
    )
    url = "https://example.test/not-found"
    probe._write_cached_envelope(url, 404, {"markets": []})

    assert probe.fetch_resource(url) == (404, {"markets": []}, "cache")


def test_replay_uses_fixture_only_with_matching_status_provenance(tmp_path):
    output_dir = tmp_path / "fixtures"
    probe = OutlierProbe(
        offline_replay=True,
        cache_dir=tmp_path / "empty-cache",
        output_dir=output_dir,
        docs_dir=tmp_path / "docs",
    )
    url = "https://api.outlier.bet/sportsdata/leagues/NCAAFB/schedule"
    fixture = output_dir / "schedule_ncaafb.json"
    fixture.write_text(json.dumps({"events": []}), encoding="utf-8")
    fixture.with_suffix(".http.json").write_text(
        json.dumps({"url": url, "status": 200}), encoding="utf-8"
    )

    assert probe.fetch_resource(url) == (200, {"events": []}, "fixture")


def test_replay_rejects_fixture_without_status_provenance(tmp_path):
    output_dir = tmp_path / "fixtures"
    probe = OutlierProbe(
        offline_replay=True,
        cache_dir=tmp_path / "empty-cache",
        output_dir=output_dir,
        docs_dir=tmp_path / "docs",
    )
    url = "https://api.outlier.bet/sportsdata/leagues/NCAAFB/schedule"
    (output_dir / "schedule_ncaafb.json").write_text(json.dumps({"events": []}), encoding="utf-8")

    assert probe.fetch_resource(url) == (0, None, "cache_miss")


def test_select_probe_date_prefers_earliest_upcoming_non_final_slate():
    events = [
        {"scheduledTime": "2026-09-10T20:00:00+00:00", "status": "pregame"},
        {"scheduledTime": "2026-09-08T20:00:00+00:00", "status": "pregame"},
        {"scheduledTime": "2026-09-07T20:00:00+00:00", "status": "final"},
    ]

    assert select_probe_date(events, today=date(2026, 9, 7)) == "2026-09-08"


def test_select_probe_date_falls_back_to_most_recent_past_slate():
    events = [
        {"scheduledTime": "2026-09-03T20:00:00+00:00", "status": "final"},
        {"scheduledTime": "2026-09-05T20:00:00+00:00", "status": "final"},
    ]

    assert select_probe_date(events, today=date(2026, 9, 7)) == "2026-09-05"


def test_decode_json_body_handles_gzip_payloads():
    payload = gzip.compress(json.dumps({"markets": []}).encode("utf-8"))

    assert decode_json_body(payload, "gzip") == {"markets": []}
