"""Tests for deterministic routing (round-robin, availability, SLA) and audit."""

import importlib
import json

import pytest

from revenue_engine import audit, config, routing

# A territory with 3 AVAILABLE reps and one on vacation, to exercise
# distribution and the skip-with-audit path in one place.
TEST_TERRITORIES = {
    "TEST": [
        {"name": "Rep A", "available": True},
        {"name": "Rep Vacation", "available": False},
        {"name": "Rep B", "available": True},
        {"name": "Rep C", "available": True},
    ],
}


def _audit_lines(runs_dir):
    path = runs_dir / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Round-robin + availability
# ---------------------------------------------------------------------------

def test_round_robin_distributes_evenly_and_audits_skips(tmp_path):
    reps = []
    for i in range(6):
        decision = routing.route_lead(
            lead_id=f"lead-{i}",
            country="United States",
            state="CA",
            propensity_score=0.5,
            runs_dir=tmp_path,
            territories=TEST_TERRITORIES,
            territory="TEST",
        )
        reps.append(decision.rep)

    counts = {rep: reps.count(rep) for rep in set(reps)}
    assert counts == {"Rep A": 2, "Rep B": 2, "Rep C": 2}
    assert "Rep Vacation" not in counts

    lines = _audit_lines(tmp_path)
    skips = [l for l in lines if l["stage"] == "routing_skip"]
    assert skips, "skipping an unavailable rep must be audited"
    assert all(l["decision"]["skipped_rep"] == "Rep Vacation" for l in skips)


def test_round_robin_state_survives_restart(tmp_path):
    first = routing.route_lead(
        lead_id="lead-a", country="United States", state="CA", propensity_score=0.5,
        runs_dir=tmp_path, territories=TEST_TERRITORIES, territory="TEST",
    )
    # Simulate a process restart: reload the module, wiping any in-memory state.
    importlib.reload(routing)
    second = routing.route_lead(
        lead_id="lead-b", country="United States", state="CA", propensity_score=0.5,
        runs_dir=tmp_path, territories=TEST_TERRITORIES, territory="TEST",
    )
    assert second.rep != first.rep, "rotation must continue from persisted state"


def test_real_config_territory_skips_unavailable_rep(tmp_path):
    # EMEA in config has Priya Raman on vacation; Tomas Lindqvist covers.
    decision = routing.route_lead(
        lead_id="lead-emea", country="Germany", state=None,
        propensity_score=0.5, runs_dir=tmp_path,
    )
    assert decision.territory == "EMEA"
    assert decision.rep == "Tomas Lindqvist"


# ---------------------------------------------------------------------------
# Territory resolution + SLA tiers
# ---------------------------------------------------------------------------

def test_unknown_country_falls_back_to_default_territory(tmp_path):
    decision = routing.route_lead(
        lead_id="lead-x", country="Atlantis", state=None,
        propensity_score=0.5, runs_dir=tmp_path,
    )
    assert decision.territory == config.DEFAULT_TERRITORY
    assert decision.rule_fired == "fallback_territory"
    lines = _audit_lines(tmp_path)
    routed = [l for l in lines if l["stage"] == "routing"]
    assert routed[-1]["rule_fired"] == "fallback_territory"
    assert routed[-1]["fallback_used"] is True


def test_sla_priority_tiers(tmp_path):
    hot = routing.route_lead(
        lead_id="hot", country="United States", state="NY",
        propensity_score=0.82, runs_dir=tmp_path,
    )
    assert (hot.priority, hot.sla_minutes) == ("P1", 15)
    cold = routing.route_lead(
        lead_id="cold", country="United States", state="NY",
        propensity_score=0.31, runs_dir=tmp_path,
    )
    assert (cold.priority, cold.sla_minutes) == ("P2", 240)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

def test_every_routing_decision_produces_exactly_one_audit_line(tmp_path):
    for i in range(4):
        routing.route_lead(
            lead_id=f"lead-{i}", country="United States", state="TX",
            propensity_score=0.5, runs_dir=tmp_path,
        )
    lines = _audit_lines(tmp_path)
    routed = [l for l in lines if l["stage"] == "routing"]
    assert len(routed) == 4


def test_audit_appends_never_truncates(tmp_path):
    audit.append("match", "lead-1", {"name": "x"}, {"status": "matched"},
                 "domain_exact", runs_dir=tmp_path)
    first = _audit_lines(tmp_path)
    audit.append("match", "lead-2", {"name": "y"}, {"status": "no_match"},
                 "fuzzy_name", runs_dir=tmp_path)
    second = _audit_lines(tmp_path)
    assert len(second) == len(first) + 1
    assert second[0] == first[0], "earlier records must be untouched"


def test_audit_record_shape(tmp_path):
    record = audit.append(
        "routing", "lead-9", {"country": "Japan"}, {"rep": "Kenji Watanabe"},
        "territory_map", fallback_used=False, runs_dir=tmp_path,
    )
    line = _audit_lines(tmp_path)[-1]
    assert line == record
    assert line["stage"] == "routing"
    assert line["lead_id"] == "lead-9"
    assert line["rule_fired"] == "territory_map"
    assert line["fallback_used"] is False
    assert len(line["inputs_sha256"]) == 64
    assert line["ts"].endswith("+00:00") or line["ts"].endswith("Z")


def test_read_tail(tmp_path):
    for i in range(5):
        audit.append("match", f"lead-{i}", {"i": i}, {"ok": True}, "r", runs_dir=tmp_path)
    tail = audit.read_tail(3, runs_dir=tmp_path)
    assert [t["lead_id"] for t in tail] == ["lead-2", "lead-3", "lead-4"]


def test_reset_state_clears_rotation_but_never_audit(tmp_path):
    first = routing.route_lead(
        lead_id="l1", country="United States", state="CA", propensity_score=0.5,
        runs_dir=tmp_path, territories=TEST_TERRITORIES, territory="TEST",
    )
    audit_before = _audit_lines(tmp_path)
    routing.reset_state(runs_dir=tmp_path)
    assert _audit_lines(tmp_path) == audit_before, "reset_state must NEVER touch the audit log"
    again = routing.route_lead(
        lead_id="l2", country="United States", state="CA", propensity_score=0.5,
        runs_dir=tmp_path, territories=TEST_TERRITORIES, territory="TEST",
    )
    assert again.rep == first.rep, "rotation restarts from the beginning after reset"


# ---------------------------------------------------------------------------
# Rotation-state robustness: corrupt/invalid state degrades, never crashes
# ---------------------------------------------------------------------------

def _write_state(runs_dir, content: str):
    path = runs_dir / "routing_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_corrupt_state_file_degrades_and_audits(tmp_path):
    _write_state(tmp_path, "not json{")
    decision = routing.route_lead(
        lead_id="lead-corrupt", country="United States", state="CA",
        propensity_score=0.5, runs_dir=tmp_path,
        territories=TEST_TERRITORIES, territory="TEST",
    )
    assert decision.rep == "Rep A", "corrupt state must reset rotation to the start"
    resets = [l for l in _audit_lines(tmp_path) if l["stage"] == "routing_state_reset"]
    assert resets, "discarding corrupt rotation state must be audited"
    assert resets[-1]["rule_fired"] == "corrupt_state_discarded"


def test_negative_index_state_discarded(tmp_path):
    _write_state(tmp_path, json.dumps({"TEST": -1}))
    decision = routing.route_lead(
        lead_id="lead-neg", country="United States", state="CA",
        propensity_score=0.5, runs_dir=tmp_path,
        territories=TEST_TERRITORIES, territory="TEST",
    )
    assert decision.rep == "Rep A"
    stages = [l["stage"] for l in _audit_lines(tmp_path)]
    assert "routing_state_reset" in stages


def test_non_int_state_value_discarded(tmp_path):
    _write_state(tmp_path, json.dumps({"TEST": "evil"}))
    decision = routing.route_lead(
        lead_id="lead-str", country="United States", state="CA",
        propensity_score=0.5, runs_dir=tmp_path,
        territories=TEST_TERRITORIES, territory="TEST",
    )
    assert decision.rep == "Rep A"
    stages = [l["stage"] for l in _audit_lines(tmp_path)]
    assert "routing_state_reset" in stages


def test_state_save_is_atomic_no_temp_left_behind(tmp_path):
    routing.route_lead(
        lead_id="lead-atomic", country="United States", state="CA",
        propensity_score=0.5, runs_dir=tmp_path,
        territories=TEST_TERRITORIES, territory="TEST",
    )
    assert (tmp_path / "routing_state.json").exists()
    assert not list(tmp_path.glob("*.tmp")), "atomic save must not leave temp files"
    state = json.loads((tmp_path / "routing_state.json").read_text())
    assert all(isinstance(v, int) and v >= 0 for v in state.values())
