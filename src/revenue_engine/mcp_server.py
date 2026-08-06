"""MCP context layer: the warehouse exposed as three read-only tools.

This is the "unified data layer" claim made concrete: any MCP client (an
agent, an IDE, `mcp dev`) sees the SAME account data through the SAME
warehouse.py helpers the pipeline uses — one source of truth, two consumers.

Tools (read-only, no writes, no network beyond stdio):
    lookup_account(query)      domain-exact hit or ranked fuzzy candidates,
                               scored identically to matching.py
    account_context(account_id) contacts + last-10 activities + open opps +
                               current propensity score (as of the frozen
                               demo clock)
    territory_rules()          territory map, rep rosters w/ availability,
                               SLA tiers

The module is importable without starting anything; `python -m
revenue_engine.mcp_server` runs the stdio server. Note: mcp 2.0 renamed
the FastMCP class to MCPServer — same decorator API.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from rapidfuzz import fuzz

from revenue_engine import config, model, routing, warehouse
from revenue_engine.features import features_as_of
from revenue_engine.matching import (
    AMBIGUOUS_THRESHOLD,
    AUTO_MATCH_THRESHOLD,
    normalize_name,
)

MAX_CANDIDATES = 5
MAX_ACTIVITIES = 10

mcp = MCPServer(
    "revenue-engine-context",
    instructions=(
        "Read-only context layer over the revenue warehouse. Resolve a "
        "company with lookup_account, pull full account context (including "
        "the current propensity score) with account_context, and inspect "
        "routing policy with territory_rules."
    ),
)

_MODEL_CACHE: dict[str, Any] = {}


def _model_payload() -> dict:
    if "payload" not in _MODEL_CACHE:
        _MODEL_CACHE["payload"] = model.load()
    return _MODEL_CACHE["payload"]


@mcp.tool()
def lookup_account(query: str) -> dict[str, Any]:
    """Resolve a company domain or name to a warehouse account.

    A domain match returns the full account record; otherwise returns
    fuzzy-name candidates ranked by the same rapidfuzz scoring the
    pipeline's matcher uses, with the matcher's thresholds attached.
    """
    con = warehouse.connect()
    try:
        account = warehouse.get_account_by_domain(con, query)
        if account:
            return {"match_type": "domain_exact", "query": query, "account": account}

        normalized = normalize_name(query)
        scored = [
            {
                "account_id": acct["id"],
                "name": acct["name"],
                "score": round(fuzz.token_set_ratio(normalized, normalize_name(acct["name"])), 1),
            }
            for acct in warehouse.search_accounts_by_name(con)
        ]
        scored.sort(key=lambda c: (-c["score"], c["account_id"]))
        return {
            "match_type": "fuzzy_name",
            "query": query,
            "candidates": scored[:MAX_CANDIDATES],
            "policy": {
                "auto_match_threshold": AUTO_MATCH_THRESHOLD,
                "ambiguous_threshold": AMBIGUOUS_THRESHOLD,
            },
        }
    finally:
        con.close()


@mcp.tool()
def account_context(account_id: int) -> dict[str, Any]:
    """Full context for one account: contacts, last 10 activities, open
    opportunities, and the current propensity score (computed as of the
    frozen demo clock). Unknown ids return a structured error dict."""
    con = warehouse.connect()
    try:
        context = warehouse.get_account_context(con, account_id)
        if context is None:
            return {
                "error": {
                    "code": "unknown_account",
                    "message": f"no account with id {account_id}; "
                               "use lookup_account to resolve one",
                }
            }
        as_of = warehouse.demo_as_of()
        propensity = model.score(features_as_of(con, account_id, as_of), _model_payload())
        return {
            "account": context["account"],
            "contacts": context["contacts"],
            "recent_activities": context["recent_activities"][:MAX_ACTIVITIES],
            "open_opportunities": context["open_opportunities"],
            "propensity_score": round(propensity, 4),
            "as_of": as_of.isoformat(),
        }
    finally:
        con.close()


@mcp.tool()
def territory_rules() -> dict[str, Any]:
    """Routing policy: territory rosters with rep availability, SLA tiers
    by propensity score, and the geography-to-territory map."""
    return {
        "territories": {
            name: {"reps": [dict(rep) for rep in roster]}
            for name, roster in config.TERRITORIES.items()
        },
        "sla_tiers": {
            "P1": {
                "min_propensity_score": routing.P1_THRESHOLD,
                "sla_minutes": routing.SLA_MINUTES["P1"],
            },
            "P2": {
                "min_propensity_score": 0.0,
                "sla_minutes": routing.SLA_MINUTES["P2"],
            },
        },
        "default_territory": config.DEFAULT_TERRITORY,
        "country_to_territory": dict(config.COUNTRY_TO_TERRITORY),
        "us_east_states": sorted(config.US_EAST_STATES),
    }


if __name__ == "__main__":
    mcp.run()  # stdio transport; never executed on import
