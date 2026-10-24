"""Official and restricted data-source registry.

The registry records sources without silently downloading or redistributing
third-party data. Each downloaded payload must be checksum-pinned locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    with registry_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported data registry schema")
    return payload


def validate_registry(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for entry in payload.get("sources", []):
        for field in ("id", "purpose", "url", "access", "license_status"):
            if not entry.get(field):
                errors.append(f"{entry.get('id', '<unknown>')}: missing {field}")
        if entry.get("access") == "restricted" and entry.get("download_allowed") is True:
            errors.append(f"{entry['id']}: restricted source cannot be download_allowed")
    return errors
