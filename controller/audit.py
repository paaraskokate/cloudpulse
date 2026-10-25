"""Append-only JSONL audit log shared by dashboard decisions and AWS scaling.

Every row records timestamp, mode (dry_run / live_scaling), data source,
observed & predicted CPU, requested/final capacity, result and reason.
No credentials or secrets are ever written here.
"""
from __future__ import annotations

import json
from pathlib import Path

# .../cloudpulse/controller/audit.py -> parents[1] = project root
DEFAULT_AUDIT_PATH = Path(__file__).resolve().parents[1] / "data" / "audit_log.jsonl"


def _resolve(path: Path) -> Path:
    """Allow callers to pass either a file or a directory for the audit log."""
    if path.exists() and path.is_dir():
        return path / "audit_log.jsonl"
    if not path.suffix and not path.exists():
        # looks like a directory to be created -> store the log inside it
        return path / "audit_log.jsonl"
    return path


def append_row(row: dict, path: str | Path | None = None) -> Path:
    """Append one JSON line; creates the file/folder if needed."""
    path = _resolve(Path(path) if path else DEFAULT_AUDIT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return path


def read_rows(path: str | Path | None = None, limit: int = 200) -> list[dict]:
    """Read rows newest-first. Tolerates a truncated last line."""
    path = _resolve(Path(path) if path else DEFAULT_AUDIT_PATH)
    if not path.exists() or not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip corrupted line instead of failing the dashboard
    return list(reversed(rows))[:limit]
