"""Offline subprocess regressions; no database, network or pipeline imports."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PS = shutil.which("powershell.exe") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(PS is None, reason="PowerShell required")


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def run(code):
    guard = ROOT / "scripts/publish_guard.ps1"
    return subprocess.run(
        [PS, "-NoProfile", "-Command", f". {quote(guard)}; {code}"],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "message,exit_code",
    [
        ("HTTP 403", 3),
        ("ENOSPC", 0),
        ("database is locked", 0),
        ("0 matches", 0),
        ("slate-date mismatch", 0),
    ],
)
def test_failed_native_stage_exits_nonzero_before_publish(tmp_path, message, exit_code):
    stage = tmp_path / "stage.py"
    stage.write_text(f"print({message!r})\nraise SystemExit({exit_code})\n")
    published = tmp_path / "published.txt"
    result = run(
        f"Invoke-CheckedNative {quote(sys.executable)} @({quote(stage)}); "
        f"'must not publish' | Set-Content -LiteralPath {quote(published)}"
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert not published.exists()


def test_stderr_logging_alone_is_not_failure(tmp_path):
    stage = tmp_path / "stage.py"
    stage.write_text("import sys\nprint('normal log', file=sys.stderr)\nprint('ok')\n")
    result = run(f"Invoke-CheckedNative {quote(sys.executable)} @({quote(stage)})")
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize(
    "base_date,games",
    [
        ("2026-11-04", [{"football_date": "2026-11-05"}]),
        ("2026-11-05", []),
        ("2026-11-05", [{"football_date": "2026-11-06"}]),
    ],
)
def test_simulation_date_and_zero_matches_withhold(tmp_path, base_date, games):
    sim = tmp_path / "sim.json"
    sim.write_text(
        json.dumps(
            {"base_date": base_date, "generated_utc": "2026-11-05T12:00:00Z", "games": games}
        )
    )
    result = run(
        f"Assert-CfbSimulation {quote(sim)} '2026-11-05' ([datetime]'2026-11-05T11:00:00Z')"
    )
    assert result.returncode != 0


def test_csv_date_mismatch_is_withheld(tmp_path):
    csv = tmp_path / "pack.csv"
    csv.write_text("game_id,kickoff_et\ncfbd:1,2026-11-06T12:00:00-05:00\n")
    result = run(f"Assert-CfbCsv {quote(csv)} '2026-11-05' ([datetime]'2020-01-01')")
    assert result.returncode != 0


@pytest.mark.parametrize(
    "script",
    [
        "run_daily_pipeline.ps1",
        "run_next_day_pipeline.ps1",
        "refresh_cfb_pack.ps1",
        "run_nightly_audit.ps1",
    ],
)
def test_active_hold_stops_before_data_or_publish_directories(tmp_path, script):
    hold_dir = tmp_path / "docs" / "remediation"
    hold_dir.mkdir(parents=True)
    (hold_dir / "publication_hold.json").write_text(
        json.dumps({"active": True, "reason": "audit quarantine"})
    )
    result = subprocess.run(
        [
            PS,
            "-NoProfile",
            "-File",
            str(ROOT / "scripts" / script),
            "-RepoRoot",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Publication withheld" in result.stderr
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "artifacts").exists()
