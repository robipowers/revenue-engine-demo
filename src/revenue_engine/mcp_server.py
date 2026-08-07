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

import sys
from typing import Any

import duckdb
from mcp.server.mcpserver import MCPServer

from revenue_engine import config, model, routing, warehouse
from revenue_engine.features import features_as_of
from revenue_engine.matching import (
    AMBIGUOUS_THRESHOLD,
    AUTO_MATCH_THRESHOLD,
    match_account,
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

# Missing/unreadable data artifacts (seed DB, trained model): the failure
# classes the DB-backed tools convert into a structured error dict instead
# of leaking a raw traceback to the MCP client.
_ARTIFACT_ERRORS = (duckdb.Error, FileNotFoundError, OSError, ValueError)

_REBUILD_HINT = (
    "rebuild with `python -m revenue_engine.warehouse --rebuild` and "
    "`python -m revenue_engine.model --train`, or check REVENUE_ENGINE_ROOT"
)


def _artifacts_error(exc: Exception) -> dict[str, Any]:
    return {
        "error": {
            "code": "artifacts_not_found",
            "message": f"data artifacts unavailable ({exc}); {_REBUILD_HINT}",
        }
    }


def _model_payload() -> dict:
    if "payload" not in _MODEL_CACHE:
        _MODEL_CACHE["payload"] = model.load()
    return _MODEL_CACHE["payload"]


@mcp.tool()
def lookup_account(query: str) -> dict[str, Any]:
    """Resolve a company domain or name to a warehouse account.

    Calls the SAME matcher the pipeline uses (matching.match_account), so
    the status, score and candidate policy here are exactly what a lead
    with this company/domain would get: matched | ambiguous | no_match,
    with candidates only inside the ambiguous band.
    Missing data artifacts return a structured error dict, never a traceback.
    """
    try:
        con = warehouse.connect()
    except _ARTIFACT_ERRORS as exc:
        return _artifacts_error(exc)
    try:
        match = match_account(con, name=query, domain=query)
        payload: dict[str, Any] = {
            "query": query,
            "status": match.status,
            "method": match.method,
            "score": match.score,
            "candidates": match.candidates[:MAX_CANDIDATES],
            "policy": {
                "auto_match_threshold": AUTO_MATCH_THRESHOLD,
                "ambiguous_threshold": AMBIGUOUS_THRESHOLD,
            },
        }
        if match.status == "matched":
            payload["account_id"] = match.account_id
            payload["account_name"] = match.account_name
        return payload
    except _ARTIFACT_ERRORS as exc:
        return _artifacts_error(exc)
    finally:
        con.close()


@mcp.tool()
def account_context(account_id: int) -> dict[str, Any]:
    """Full context for one account: contacts, last 10 activities, open
    opportunities, and the current propensity score (computed as of the
    frozen demo clock). Unknown ids and missing data artifacts return a
    structured error dict."""
    try:
        con = warehouse.connect()
    except _ARTIFACT_ERRORS as exc:
        return _artifacts_error(exc)
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
        # Activities are cut at the same as-of clock the score was computed
        # at — the context shown beside a score never contains events the
        # score could not have seen.
        recent = [
            {"id": r[0], "contact_id": r[1], "type": r[2], "ts": r[3]}
            for r in con.execute(
                "SELECT id, contact_id, type, ts FROM activities "
                "WHERE account_id = ? AND ts <= CAST(? AS TIMESTAMP) "
                "ORDER BY ts DESC LIMIT ?",
                [account_id, as_of, MAX_ACTIVITIES],
            ).fetchall()
        ]
        return {
            "account": context["account"],
            "contacts": context["contacts"],
            "recent_activities": recent,
            "open_opportunities": context["open_opportunities"],
            "propensity_score": round(propensity, 4),
            "as_of": as_of.isoformat(),
        }
    except _ARTIFACT_ERRORS as exc:
        return _artifacts_error(exc)
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


def _missing_artifacts() -> list[str]:
    """Startup preflight: name every required on-disk artifact that is absent."""
    missing = []
    if not config.SEED_DB.exists():
        missing.append(f"seed warehouse: {config.SEED_DB}")
    if not model.MODEL_PATH.exists():
        missing.append(f"trained model: {model.MODEL_PATH}")
    return missing


if __name__ == "__main__":
    problems = _missing_artifacts()
    if problems:
        for problem in problems:
            print(f"missing artifact: {problem}", file=sys.stderr)
        print(_REBUILD_HINT, file=sys.stderr)
        raise SystemExit(1)
    mcp.run()  # stdio transport; never executed on import
