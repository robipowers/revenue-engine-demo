"""In-process tests for the Relay dashboard server (revenue_engine.ui).

The server runs on an ephemeral port in a background thread with runs_dir
pointed at a pytest tmp dir, so nothing pollutes the repo's runs/.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from revenue_engine import config, ui


@pytest.fixture()
def server(tmp_path):
    srv = ui.make_server(port=0, runs_dir=tmp_path)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _get(base, path):
    with urllib.request.urlopen(base + path) as res:
        return res.status, json.loads(res.read())


def _post(base, path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_index_served(server):
    with urllib.request.urlopen(server + "/") as res:
        body = res.read().decode()
    assert res.status == 200
    assert "Relay" in body and "REVENUE ENGINE" in body


def test_leads_shape(server):
    status, data = _get(server, "/api/leads")
    assert status == 200
    leads = data["leads"]
    assert len(leads) >= 3
    for lead in leads:
        for key in ("file", "lead_id", "company", "contact_name", "title",
                    "country", "state", "notes"):
            assert key in lead
    assert {l["file"] for l in leads} >= {"lead_001", "lead_002", "lead_003"}


def test_territories_shape(server):
    status, data = _get(server, "/api/territories")
    assert status == 200
    assert data["territories"] == config.TERRITORIES
    assert data["p1_threshold"] == 0.7
    assert data["sla_minutes"] == {"P1": 15, "P2": 240}
    assert data["country_map"] == config.COUNTRY_TO_TERRITORY
    assert set(data["us_east_states"]) == config.US_EAST_STATES
    assert data["match_auto"] == 90.0 and data["match_ambiguous"] == 70.0


def test_stats_endpoint(server):
    status, data = _get(server, "/api/stats")
    assert status == 200
    w = data["warehouse"]
    assert w["accounts"] == 40 and w["contacts"] == 120
    assert w["opportunities"] == 80
    assert w["opps_won"] + w["opps_lost"] + w["opps_open"] == 80
    assert w["activities"] > 0
    m = data["model"]
    assert 0.5 < m["holdout_auc"] < 1.0
    assert m["n_features"] == 6
    assert set(m["coefficients"]) == {
        "activity_count_30d", "distinct_contacts_engaged",
        "days_since_last_activity", "prior_opportunity_count",
        "employee_band", "industry_fit",
    }
    assert data["inbound_leads"] >= 3
    assert isinstance(data["audit_records"], int)
    assert isinstance(data["audit_tail"], list)
    assert data["territories"] == 4 and data["reps"] == 8


def test_desk_reflects_audit_history(server):
    _post(server, "/api/run", {"lead": "lead_001", "fresh": True})
    _post(server, "/api/approve", {"lead": "lead_001", "decision": "approve"})
    status, data = _get(server, "/api/desk")
    assert status == 200
    # lead_001 (Germany) routes to EMEA: Priya is away, Tomas gets the lead.
    tomas = data["reps"]["Tomas Lindqvist"]
    assert tomas["routed"] == 1 and tomas["applied"] == 1
    assert tomas["last_ts"]
    assert data["reps"]["Priya Raman"]["skipped"] == 1
    assert data["rotation"].get("EMEA") == 0  # wrapped past the 2-rep roster


def test_run_fresh_returns_pending_and_records(server):
    status, data = _post(server, "/api/run", {"lead": "lead_001", "fresh": True})
    assert status == 200
    assert data["status"] == "pending_approval"
    assert isinstance(data["records"], list) and data["records"]
    stages = [r["stage"] for r in data["records"]]
    assert "match" in stages and "routing" in stages and "pending_approval" in stages
    for rec in data["records"]:
        for key in ("ts", "stage", "lead_id", "inputs_sha256", "decision",
                    "rule_fired", "fallback_used"):
            assert key in rec
    proposal = data["proposal"]
    assert proposal["lead_id"] == "L-1001"
    assert proposal["territory"] == "EMEA"
    assert proposal["rep"]  # a real rep was assigned
    # EMEA has one rep on vacation: the skip must be audited in this run.
    assert "routing_skip" in stages
    # Real per-feature contributions ride along for the score card readout.
    explain = data["score_explain"]
    assert isinstance(explain, list) and len(explain) == 6
    for row in explain:
        assert set(row) == {"feature", "value", "raw", "contribution"}
    # Sorted by |contribution|: the first entry is the dominant factor.
    mags = [abs(r["contribution"]) for r in explain]
    assert mags == sorted(mags, reverse=True)
    # The enrich card's account context rides along: real named contacts,
    # a real activity feed, and TRUE totals (not display-capped counts).
    acct = data["account"]
    assert acct["account"]["name"] == "Meterhouse"
    assert len(acct["contacts"]) == 3
    for contact in acct["contacts"]:
        assert contact["name"] and contact["title"] and contact["email"]
    assert acct["recent_activities"]
    for act in acct["recent_activities"]:
        assert act["type"] and act["ts"] and act["contact"]
    totals = acct["totals"]
    assert totals["activities_total"] >= len(acct["recent_activities"])
    assert totals["opps_total"] == totals["opps_won"] + totals["opps_lost"] + len(
        acct["open_opportunities"]
    )


def test_stats_last_score_after_run(server):
    _post(server, "/api/run", {"lead": "lead_001", "fresh": True})
    status, data = _get(server, "/api/stats")
    assert status == 200
    last = data["last_score"]
    assert last["lead_id"] == "L-1001"
    assert 0.0 <= float(last["score"]) <= 1.0
    assert isinstance(last["explain"], list) and last["explain"]


def test_approve_resolves_pending(server):
    _post(server, "/api/run", {"lead": "lead_001", "fresh": True})
    status, data = _post(server, "/api/approve",
                         {"lead": "lead_001", "decision": "approve"})
    assert status == 200
    assert data["status"] == "assignment_applied"
    assert [r["stage"] for r in data["records"]] == ["assignment_applied"]
    # Repeat approve replays the recorded outcome idempotently.
    status, data = _post(server, "/api/approve",
                         {"lead": "lead_001", "decision": "approve"})
    assert status == 200
    assert data["status"] == "assignment_applied"
    assert data.get("idempotent_replay") is True
    assert data["records"] == []


def test_corrupt_pending_maps_to_409(server, tmp_path):
    _post(server, "/api/run", {"lead": "lead_002", "fresh": True})
    pending = tmp_path / "pending" / "L-1002.json"
    assert pending.exists()
    pending.write_text(json.dumps({"lead_id": "L-1002", "proposal": "not-a-dict"}))
    status, data = _post(server, "/api/approve",
                         {"lead": "lead_002", "decision": "approve"})
    assert status == 409
    assert "corrupt" in data["error"]
    # The fail-closed discard is audited and surfaced in the delta.
    assert any(r["stage"] == "pending_discarded" for r in data.get("records", []))


def test_unknown_lead_404_and_bad_decision_400(server):
    status, data = _post(server, "/api/run", {"lead": "lead_999", "fresh": True})
    assert status == 404 and "error" in data
    status, data = _post(server, "/api/approve",
                         {"lead": "lead_001", "decision": "maybe"})
    assert status == 400 and "error" in data
