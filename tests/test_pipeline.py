"""Tests for the U4 agentic pipeline: stages, judgment fallbacks, HITL gate,
guardrails, and the offline hard guarantee."""

import json
import socket
from datetime import datetime
from pathlib import Path

import pytest

from revenue_engine import pipeline, routing, warehouse

REPO_ROOT = Path(__file__).resolve().parent.parent
INBOUND = REPO_ROOT / "data" / "inbound"
LEAD_CLEAN = INBOUND / "lead_001.json"
LEAD_MESSY = INBOUND / "lead_002.json"
LEAD_NOMATCH = INBOUND / "lead_003.json"


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    """Machines with a real key must still exercise the fallback path."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _audit_records(runs_dir):
    path = Path(runs_dir) / "audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stages(runs_dir):
    return [r["stage"] for r in _audit_records(runs_dir)]


# ---------------------------------------------------------------------------
# Lead fixtures sanity
# ---------------------------------------------------------------------------

def test_lead_fixtures_exist_and_shape():
    for path in (LEAD_CLEAN, LEAD_MESSY, LEAD_NOMATCH):
        lead = json.loads(path.read_text())
        for key in ("lead_id", "company", "contact_name", "email", "title", "country", "notes"):
            assert key in lead, f"{path.name} missing {key}"
    messy = json.loads(LEAD_MESSY.read_text())
    assert "domain" not in messy or not messy.get("domain"), "lead_002 must force the fuzzy path"


# ---------------------------------------------------------------------------
# Clean lead end-to-end (offline)
# ---------------------------------------------------------------------------

def test_clean_lead_end_to_end_offline(tmp_path):
    result = pipeline.run(LEAD_CLEAN, runs_dir=tmp_path)
    assert result["status"] == "pending_approval"
    assert result["match"]["method"] == "domain_exact"
    assert result["match"]["account_name"] == "Meterhouse"
    # Intent summary produced via keyword fallback when there is no key.
    intent_stage = next(s for s in result["stages"] if s["stage"] == "intent_summary")
    assert intent_stage["fallback_used"] is True
    assert result["intent"]["intent"]
    assert len(result["intent"]["summary"].split()) <= 20
    assert 0.0 <= result["score"] <= 1.0
    assert result["proposal"]["rep"]
    assert _stages(tmp_path)[-1] == "pending_approval"


def test_full_run_stages_in_order(tmp_path):
    pipeline.run(LEAD_MESSY, approve=True, runs_dir=tmp_path)
    stages = [s for s in _stages(tmp_path) if s != "routing_skip"]
    assert stages == [
        "match",
        "tie_break",
        "enrich",
        "intent_summary",
        "score",
        "routing",
        "pending_approval",
        "assignment_applied",
    ]


# ---------------------------------------------------------------------------
# HITL gate
# ---------------------------------------------------------------------------

def test_no_assignment_without_approval(tmp_path):
    result = pipeline.run(LEAD_CLEAN, runs_dir=tmp_path)
    assert result["status"] == "pending_approval"
    stages = _stages(tmp_path)
    assert "assignment_applied" not in stages
    assert "pending_approval" in stages


def test_approve_applies_and_audits(tmp_path):
    result = pipeline.run(LEAD_CLEAN, approve=True, runs_dir=tmp_path)
    assert result["status"] == "assignment_applied"
    records = _audit_records(tmp_path)
    assert records[-1]["stage"] == "assignment_applied"
    assert records[-1]["decision"]["rep"] == result["proposal"]["rep"]
    # No pending leak: pending_approval exists but is superseded by the apply.
    assert _stages(tmp_path).index("pending_approval") < len(records) - 1


def test_reject_routes_to_review(tmp_path):
    result = pipeline.run(LEAD_CLEAN, approve=False, runs_dir=tmp_path)
    assert result["status"] == "routed_to_review"
    assert _stages(tmp_path)[-1] == "routed_to_review"
    assert "assignment_applied" not in _stages(tmp_path)


# ---------------------------------------------------------------------------
# Replayability: two fresh runs -> identical assignments
# ---------------------------------------------------------------------------

def test_two_fresh_runs_identical(tmp_path):
    routing.reset_state(tmp_path)
    first = pipeline.run(LEAD_MESSY, approve=True, runs_dir=tmp_path)
    routing.reset_state(tmp_path)
    second = pipeline.run(LEAD_MESSY, approve=True, runs_dir=tmp_path)
    assert first["proposal"] == second["proposal"]
    assert first["score"] == second["score"]


# ---------------------------------------------------------------------------
# Demo clock anchor: frozen dataset => frozen clock
# ---------------------------------------------------------------------------

def test_demo_as_of_anchored_to_dataset_not_wall_clock(tmp_path, monkeypatch):
    """Scores must be identical regardless of the real wall clock."""
    baseline = pipeline.run(LEAD_CLEAN, runs_dir=tmp_path / "a")["score"]

    class FarFuture(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2031, 1, 1)

    # Even if some code path reached for "now", the score must not move.
    monkeypatch.setattr(pipeline, "datetime", FarFuture, raising=False)
    shifted = pipeline.run(LEAD_CLEAN, runs_dir=tmp_path / "b")["score"]
    assert shifted == baseline

    anchor = warehouse.demo_as_of()
    assert anchor.year == 2026  # pinned inside the frozen dataset's timeline


def test_priority_tiers_p1_and_p2(tmp_path):
    """The demo must show both SLA tiers: lead_001 hot (P1), lead_002 cool (P2)."""
    hot = pipeline.run(LEAD_CLEAN, runs_dir=tmp_path / "hot")
    assert hot["score"] >= 0.7
    assert hot["proposal"]["priority"] == "P1"
    assert hot["proposal"]["sla_minutes"] == 15

    cool = pipeline.run(LEAD_MESSY, runs_dir=tmp_path / "cool")
    assert cool["score"] < 0.7
    assert cool["proposal"]["priority"] == "P2"
    assert cool["proposal"]["sla_minutes"] == 240


# ---------------------------------------------------------------------------
# Judgment steps: tie-break + intent fallbacks
# ---------------------------------------------------------------------------

def test_messy_lead_triggers_tie_break_fallback(tmp_path):
    result = pipeline.run(LEAD_MESSY, runs_dir=tmp_path)
    tie = next(s for s in result["stages"] if s["stage"] == "tie_break")
    assert tie["fallback_used"] is True
    assert result["match"]["status"] == "ambiguous"
    assert len(result["match"]["candidates"]) >= 2
    # Deterministic fallback: highest score, account_id tiebreak -> Hivemetric (id 3).
    assert result["resolved_account"]["name"] == "Hivemetric"


def test_intent_summary_keyword_fallback():
    intent, fallback_used = pipeline.llm_intent_summary(
        "need pricing and integration details before end of quarter", llm=False
    )
    assert fallback_used is True
    assert intent["intent"] == "pricing_inquiry"
    assert len(intent["summary"].split()) <= 20


def test_llm_guardrail_out_of_candidate_answer_rejected(monkeypatch):
    """LLM can never expand its own action space: an answer outside the
    candidate list is rejected and the deterministic fallback fires."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(pipeline, "_call_llm", lambda *a, **k: "Totally Fake Corp")
    candidates = [
        {"account_id": 3, "name": "Hivemetric", "score": 85.7},
        {"account_id": 12, "name": "Pinemetric", "score": 85.7},
    ]
    choice, fallback_used = pipeline.llm_tie_break(
        candidates, {"company": "Hine Metric Co"}, llm=True
    )
    assert fallback_used is True
    assert choice["account_id"] == 3  # highest score, lowest account_id


def test_llm_error_falls_back(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")

    def _boom(*a, **k):
        raise RuntimeError("simulated API failure")

    monkeypatch.setattr(pipeline, "_call_llm", _boom)
    intent, fallback_used = pipeline.llm_intent_summary("book a demo please", llm=True)
    assert fallback_used is True
    assert intent["intent"] == "demo_request"


# ---------------------------------------------------------------------------
# No-match -> new-account path
# ---------------------------------------------------------------------------

def test_no_match_exits_via_new_account_path(tmp_path):
    result = pipeline.run(LEAD_NOMATCH, runs_dir=tmp_path)
    assert result["status"] == "new_account"
    assert result["proposal"]["action"] == "create_account"
    assert result["proposal"]["company"]
    stages = _stages(tmp_path)
    assert stages[-1] == "new_account"
    assert "routing" not in stages and "score" not in stages


# ---------------------------------------------------------------------------
# THE HARD GUARANTEE: fully offline, zero network
# ---------------------------------------------------------------------------

def test_offline_guarantee_full_pipeline_with_sockets_blocked(tmp_path, monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during offline run")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    result = pipeline.run(LEAD_MESSY, approve=True, runs_dir=tmp_path)
    assert result["status"] == "assignment_applied"
    for stage in result["stages"]:
        if stage["stage"] in ("tie_break", "intent_summary"):
            assert stage["fallback_used"] is True


def test_no_top_level_anthropic_import():
    src = REPO_ROOT / "src"
    offenders = []
    for path in src.rglob("*.py"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import anthropic", "from anthropic")):
                # allowed only when indented (inside a guarded function body)
                if line == stripped:
                    offenders.append(str(path))
    assert not offenders, f"top-level anthropic import in: {offenders}"
