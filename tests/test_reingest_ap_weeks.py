"""The AP week re-ingest printer lists commands and does not ingest."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_dry_run_lists_weeks_5_through_9_and_does_not_ingest(capsys, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "reingest_ap_weeks.py"
    spec = importlib.util.spec_from_file_location("reingest_ap_weeks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.commands(2026) == [
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 5",
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 6",
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 7",
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 8",
        "python -m cfb_analytics.cli backfill-rankings --year 2026 --week 9",
    ]
    assert module.main([]) == 0
    captured = capsys.readouterr()
    assert "DRY RUN — not executed" in captured.out
    assert "week 5" in captured.out and "week 9" in captured.out
    assert not (tmp_path / "data").exists() or not list((tmp_path / "data").rglob("*"))
