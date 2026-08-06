"""Tests for the U5 MCP context layer: three read tools over the warehouse.

The server is exercised two ways: direct function calls (the decorated
functions stay plain callables) and in-process MCP dispatch via
server.call_tool, which validates arguments and result serialization the
same way a connected client would.
"""

import asyncio
import json
from pathlib import Path

from revenue_engine import mcp_server
from revenue_engine.config import TERRITORIES

REPO_ROOT = Path(__file__).resolve().parent.parent


def _call(name: str, arguments: dict):
    """Invoke a tool through the MCP dispatch path and decode the result."""
    result = asyncio.run(mcp_server.mcp.call_tool(name, arguments))
    assert result.is_error is False
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def _tool_names() -> list[str]:
    tools = asyncio.run(mcp_server.mcp.list_tools())
    return sorted(t.name for t in tools)


# ---------------------------------------------------------------------------
# Server registration
# ---------------------------------------------------------------------------

def test_module_imports_without_starting_server():
    # Importing at module top already succeeded; the server object exists
    # and nothing is listening (run() is guarded under __main__).
    assert mcp_server.mcp.name == "revenue-engine-context"


def test_exactly_three_tools_registered():
    assert _tool_names() == ["account_context", "lookup_account", "territory_rules"]


# ---------------------------------------------------------------------------
# lookup_account
# ---------------------------------------------------------------------------

def test_lookup_account_domain_exact():
    out = mcp_server.lookup_account("meterhouse.io")
    assert out["match_type"] == "domain_exact"
    assert out["account"]["name"] == "Meterhouse"
    assert out["account"]["territory"] == "EMEA"


def test_lookup_account_fuzzy_returns_ranked_candidates():
    out = _call("lookup_account", {"query": "Hine Metric Co"})
    assert out["match_type"] == "fuzzy_name"
    candidates = out["candidates"]
    assert len(candidates) >= 2
    for cand in candidates:
        assert set(cand) == {"account_id", "name", "score"}
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True), "candidates must be ranked"
    assert candidates[0]["name"] == "Hivemetric"  # same scoring as matching.py
    assert out["policy"]["auto_match_threshold"] == 90.0
    assert out["policy"]["ambiguous_threshold"] == 70.0


def test_lookup_account_no_plausible_match_still_structured():
    out = mcp_server.lookup_account("Zephyr Robotics")
    assert out["match_type"] == "fuzzy_name"
    assert all(c["score"] < 90.0 for c in out["candidates"])


# ---------------------------------------------------------------------------
# account_context
# ---------------------------------------------------------------------------

def test_account_context_documented_shape():
    lookup = mcp_server.lookup_account("meterhouse.io")
    account_id = lookup["account"]["id"]
    out = _call("account_context", {"account_id": account_id})
    assert set(out) >= {
        "account", "contacts", "recent_activities", "open_opportunities",
        "propensity_score", "as_of",
    }
    assert out["account"]["id"] == account_id
    assert len(out["contacts"]) == 3
    assert 0 < len(out["recent_activities"]) <= 10
    assert 0.0 <= out["propensity_score"] <= 1.0
    for opp in out["open_opportunities"]:
        assert opp["outcome"] is None


def test_account_context_invalid_id_structured_error():
    out = mcp_server.account_context(999_999)
    assert "error" in out
    assert out["error"]["code"] == "unknown_account"
    assert "999999" in out["error"]["message"]
    # And through the MCP dispatch path too: an error dict, not a traceback.
    dispatched = _call("account_context", {"account_id": 999_999})
    assert dispatched["error"]["code"] == "unknown_account"


# ---------------------------------------------------------------------------
# territory_rules
# ---------------------------------------------------------------------------

def test_territory_rules_shape():
    out = _call("territory_rules", {})
    assert set(out["territories"]) == set(TERRITORIES)
    emea = out["territories"]["EMEA"]
    reps = {r["name"]: r["available"] for r in emea["reps"]}
    assert reps["Priya Raman"] is False  # the vacation-skip rep is visible
    assert out["sla_tiers"]["P1"]["sla_minutes"] == 15
    assert out["sla_tiers"]["P2"]["sla_minutes"] == 240
    assert out["sla_tiers"]["P1"]["min_propensity_score"] == 0.7
    assert out["default_territory"] == "AMER-West"
