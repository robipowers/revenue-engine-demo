"""Deterministic routing: territory map -> availability-aware round-robin -> SLA tier.

Rotation state persists as JSON in runs/ (no in-memory cache), so assignments
survive process restarts. Unavailable reps ("on vacation") are skipped and the
skip is audited. reset_state clears the rotation only — never the audit log.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from revenue_engine import audit, config

STATE_FILENAME = "routing_state.json"

# SLA tiers by propensity score.
P1_THRESHOLD = 0.7
SLA_MINUTES = {"P1": 15, "P2": 240}


@dataclass(frozen=True)
class RoutingDecision:
    territory: str
    rep: str
    priority: str  # P1 | P2
    sla_minutes: int
    rule_fired: str  # territory_map | fallback_territory
    fallback_used: bool
    skipped_reps: list[str] = field(default_factory=list)


def _state_path(runs_dir: Path | str | None = None) -> Path:
    base = Path(runs_dir) if runs_dir else config.RUNS_DIR
    return base / STATE_FILENAME


def _valid_state(raw) -> bool:
    """Rotation state must be a dict of {territory: non-negative int}."""
    if not isinstance(raw, dict):
        return False
    return all(
        isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 0
        for k, v in raw.items()
    )


def _load_state(runs_dir) -> dict:
    """Load rotation state; a corrupt or invalid file degrades to empty state.

    Rotation is best-effort by design: a truncated write or a hand-edited
    file must never kill a run. The discard is audited, never silent.
    """
    path = _state_path(runs_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        raw = None
    if not _valid_state(raw):
        audit.append(
            stage="routing_state_reset",
            lead_id="system",
            inputs={"path": str(path)},
            decision={"reason": "corrupt_state_discarded"},
            rule_fired="corrupt_state_discarded",
            runs_dir=runs_dir,
        )
        return {}
    return raw


def _save_state(state: dict, runs_dir) -> None:
    """Atomic write: temp file + os.replace, so a killed process can never
    leave a truncated routing_state.json behind."""
    path = _state_path(runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def reset_state(runs_dir: Path | str | None = None) -> None:
    """Clear round-robin rotation state. NEVER touches the audit log."""
    path = _state_path(runs_dir)
    if path.exists():
        path.unlink()


def route_lead(
    lead_id: str,
    country: str,
    state: str | None,
    propensity_score: float,
    runs_dir: Path | str | None = None,
    territories: dict[str, list[dict]] | None = None,
    territory: str | None = None,
) -> RoutingDecision:
    """Route one lead: resolve territory, pick the next available rep, set SLA.

    `territories`/`territory` overrides exist for tests; production callers
    pass only the lead fields and the config map applies.
    """
    rosters = territories if territories is not None else config.TERRITORIES

    if territory is None:
        territory = config.resolve_territory(country, state)
        known = country == "United States" or country in config.COUNTRY_TO_TERRITORY
        fallback_used = not known
    else:
        fallback_used = False
    rule_fired = "fallback_territory" if fallback_used else "territory_map"

    roster = rosters.get(territory)
    if not roster:
        raise ValueError(f"territory {territory!r} has no rep roster")

    rotation = _load_state(runs_dir)
    idx = max(0, int(rotation.get(territory, 0))) % len(roster)

    skipped: list[str] = []
    assigned = None
    for _ in range(len(roster)):
        rep = roster[idx]
        idx = (idx + 1) % len(roster)
        if rep["available"]:
            assigned = rep["name"]
            break
        skipped.append(rep["name"])
        audit.append(
            stage="routing_skip",
            lead_id=lead_id,
            inputs={"territory": territory, "rep": rep["name"]},
            decision={"skipped_rep": rep["name"], "reason": "rep_unavailable"},
            rule_fired="skip_unavailable_rep",
            runs_dir=runs_dir,
        )
    if assigned is None:
        raise ValueError(f"no available reps in territory {territory!r}")

    rotation[territory] = idx
    _save_state(rotation, runs_dir)

    priority = "P1" if propensity_score >= P1_THRESHOLD else "P2"
    decision = RoutingDecision(
        territory=territory,
        rep=assigned,
        priority=priority,
        sla_minutes=SLA_MINUTES[priority],
        rule_fired=rule_fired,
        fallback_used=fallback_used,
        skipped_reps=skipped,
    )
    audit.append(
        stage="routing",
        lead_id=lead_id,
        inputs={"country": country, "state": state, "propensity_score": propensity_score},
        decision={
            "territory": territory,
            "rep": assigned,
            "priority": priority,
            "sla_minutes": decision.sla_minutes,
        },
        rule_fired=rule_fired,
        fallback_used=fallback_used,
        runs_dir=runs_dir,
    )
    return decision
