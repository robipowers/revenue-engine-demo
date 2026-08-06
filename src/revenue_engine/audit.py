"""Append-only JSONL audit trail: the governance & auditability layer.

Every pipeline decision lands here as one line with an ISO-8601 UTC
timestamp and a SHA-256 hash of the decision inputs. The file is only ever
opened in append mode — nothing in this codebase truncates or rewrites it
(routing.reset_state deliberately leaves it alone).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from revenue_engine import config

AUDIT_FILENAME = "audit.jsonl"


def audit_path(runs_dir: Path | str | None = None) -> Path:
    return Path(runs_dir) / AUDIT_FILENAME if runs_dir else config.RUNS_DIR / AUDIT_FILENAME


def _hash_inputs(inputs: dict) -> str:
    canonical = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def append(
    stage: str,
    lead_id: str,
    inputs: dict,
    decision: dict,
    rule_fired: str,
    fallback_used: bool = False,
    runs_dir: Path | str | None = None,
) -> dict:
    """Append one audit record and return it."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "lead_id": lead_id,
        "inputs": inputs,
        "inputs_sha256": _hash_inputs(inputs),
        "decision": decision,
        "rule_fired": rule_fired,
        "fallback_used": fallback_used,
    }
    path = audit_path(runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:  # append-only, by design
        f.write(json.dumps(record, default=str) + "\n")
    return record


def read_tail(n: int = 10, runs_dir: Path | str | None = None) -> list[dict]:
    """Last n audit records (oldest of the tail first), for the demo CLI."""
    path = audit_path(runs_dir)
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-n:]]
