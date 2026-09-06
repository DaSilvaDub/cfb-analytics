"""tests/test_fixtures_cleanliness.py: Audit fixtures for sensitive tokens and credentials."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Regex matching AWS Cognito / JWT token signatures (header.payload.signature)
JWT_PATTERN = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")

# Banned key names that would indicate authorization state or session credential leakage
BANNED_KEYS = {
    "accesstoken",
    "idtoken",
    "refreshtoken",
    "authorization",
    "bearer",
    "cookie",
    "storage_state",
}


def _check_banned_keys(obj: Any, file_path: Path) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            lower_k = str(k).lower()
            assert lower_k not in BANNED_KEYS, f"Banned auth key '{k}' found in fixture {file_path}"
            _check_banned_keys(v, file_path)
    elif isinstance(obj, list):
        for item in obj:
            _check_banned_keys(item, file_path)


def test_no_auth_tokens_or_secrets_in_fixtures() -> None:
    json_files = list(FIXTURES_DIR.glob("**/*.json"))
    assert len(json_files) > 0, f"No fixtures found to audit under {FIXTURES_DIR}"

    for file_path in json_files:
        content = file_path.read_text(encoding="utf-8")

        # 1. Assert zero JWT token patterns in file content
        jwt_matches = JWT_PATTERN.findall(content)
        assert not jwt_matches, f"Found JWT token pattern in {file_path}: {jwt_matches[:1]}"

        # 2. Assert zero raw cookie headers or bearer prefixes
        assert "bearer ey" not in content.lower(), f"Found Bearer token string in {file_path}"
        assert "connect.sid=" not in content.lower(), f"Found session cookie in {file_path}"

        # 3. Assert zero banned auth keys in JSON hierarchy
        data = json.loads(content)
        _check_banned_keys(data, file_path)
