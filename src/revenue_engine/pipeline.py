"""Agentic lead pipeline: match -> judgment -> enrich -> score -> route -> HITL.

THE HARD GUARANTEE: with no ANTHROPIC_API_KEY and no llm flag, every stage
runs identically via deterministic fallbacks — zero network. The two
judgment steps (tie_break, intent_summary) are the ONLY places an LLM can
participate, and it can never expand its own action space: a tie-break
answer outside the candidate list, an over-long intent summary, an error,
or an empty response all collapse to the deterministic fallback. The
`anthropic` package is lazy-imported inside the guarded call path only.

Every stage appends one audit record; the HITL gate never applies an
assignment without an explicit approve.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from revenue_engine import audit, config, model, routing, warehouse
from revenue_engine.features import features_as_of
from revenue_engine.matching import match_account

# Deterministic intent taxonomy shared by the LLM prompt and the fallback.
INTENT_LABELS = ["pricing_inquiry", "demo_request", "evaluation", "support", "general_interest"]

_INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("pricing_inquiry", ("pricing", "price", "cost", "quote", "budget")),
    ("demo_request", ("demo", "trial", "walkthrough", "meeting")),
    ("evaluation", ("evaluat", "compar", "vendor", "migration", "integrat", "poc")),
    ("support", ("help", "issue", "problem", "support", "broken")),
]

MAX_SUMMARY_WORDS = 20
LLM_MODEL = "claude-haiku-4-5"
LLM_TIMEOUT_SECONDS = 10.0


# ---------------------------------------------------------------------------
# LLM plumbing (the ONLY network-capable path, always guarded)
# ---------------------------------------------------------------------------

def _llm_available(llm: bool) -> bool:
    return bool(llm and os.environ.get("ANTHROPIC_API_KEY"))


def _call_llm(prompt: str, max_tokens: int = 100) -> str:
    """Single constrained LLM call. Lazy import — never at module top level."""
    import anthropic  # noqa: PLC0415 — lazy by design (offline guarantee)

    client = anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Judgment step 1: tie-break among ambiguous match candidates
# ---------------------------------------------------------------------------

def _deterministic_tie_break(candidates: list[dict]) -> dict:
    """Highest rapidfuzz score wins; account_id breaks exact ties."""
    return sorted(candidates, key=lambda c: (-c["score"], c["account_id"]))[0]


def llm_tie_break(candidates: list[dict], lead: dict, llm: bool = False) -> tuple[dict, bool]:
    """Pick one candidate. Returns (candidate, fallback_used).

    Guardrail: the LLM MUST answer with a name from the candidate list;
    anything else (out-of-list, error, empty) -> deterministic fallback.
    """
    if _llm_available(llm):
        names = [c["name"] for c in candidates]
        prompt = (
            "A lead entered our CRM with the company name "
            f"{lead.get('company')!r} (contact: {lead.get('contact_name')}, "
            f"email: {lead.get('email')}). Which existing account is it most "
            f"likely referring to? You MUST answer with exactly one name from "
            f"this list and nothing else: {names}"
        )
        try:
            answer = _call_llm(prompt, max_tokens=50)
            for candidate in candidates:
                if answer.strip().lower() == candidate["name"].lower():
                    return candidate, False
            # Out-of-candidate-list answer: rejected, fall through to fallback.
        except Exception:
            pass  # error/timeout: fall through to fallback
    return _deterministic_tie_break(candidates), True


# ---------------------------------------------------------------------------
# Judgment step 2: intent summary from free-text notes
# ---------------------------------------------------------------------------

def _deterministic_intent(notes: str) -> dict:
    lowered = notes.lower()
    label = "general_interest"
    for intent, keywords in _INTENT_KEYWORDS:
        if any(k in lowered for k in keywords):
            label = intent
            break
    words = notes.split()
    summary = " ".join(words[:MAX_SUMMARY_WORDS])
    return {"intent": label, "summary": summary}


def llm_intent_summary(notes: str, llm: bool = False) -> tuple[dict, bool]:
    """Summarize lead notes into (intent label, <=20-word summary).

    Guardrail: label must come from INTENT_LABELS and the summary must be
    <=20 words; anything else -> deterministic keyword fallback.
    """
    if _llm_available(llm):
        prompt = (
            "Classify these CRM lead notes. Reply with exactly two lines:\n"
            f"line 1: one label from {INTENT_LABELS}\n"
            f"line 2: a summary of at most {MAX_SUMMARY_WORDS} words.\n"
            f"Notes: {notes!r}"
        )
        try:
            answer = _call_llm(prompt, max_tokens=100)
            lines = [line.strip() for line in answer.splitlines() if line.strip()]
            if len(lines) >= 2:
                label, summary = lines[0].lower(), lines[1]
                if label in INTENT_LABELS and 0 < len(summary.split()) <= MAX_SUMMARY_WORDS:
                    return {"intent": label, "summary": summary}, False
            # Malformed/over-long/empty: fall through to fallback.
        except Exception:
            pass
    return _deterministic_intent(notes), True


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _record(stages: list[dict], record: dict) -> None:
    stages.append(
        {
            "stage": record["stage"],
            "decision": record["decision"],
            "rule_fired": record["rule_fired"],
            "fallback_used": record["fallback_used"],
        }
    )


def run(
    lead_path: Path | str,
    llm: bool = False,
    approve: bool | None = None,
    runs_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
    as_of: datetime | None = None,
) -> dict:
    """Run the full pipeline for one inbound lead JSON file.

    approve: None -> stop at the HITL gate (pending_approval);
             True -> apply the assignment; False -> route to human review.
    as_of: scoring clock; defaults to warehouse.demo_as_of() — the frozen
           dataset gets a frozen clock, so scores stay meaningful on any
           demo day instead of decaying with the real calendar.
    """
    lead = json.loads(Path(lead_path).read_text())
    lead_id = lead["lead_id"]
    stages: list[dict] = []
    result: dict = {"lead_id": lead_id, "stages": stages, "llm": llm}

    con = warehouse.connect(db_path or config.SEED_DB)
    try:
        # ---- Stage: match -------------------------------------------------
        match = match_account(con, name=lead.get("company"), domain=lead.get("domain"))
        _record(
            stages,
            audit.append(
                stage="match",
                lead_id=lead_id,
                inputs={"company": lead.get("company"), "domain": lead.get("domain")},
                decision={
                    "status": match.status,
                    "account_id": match.account_id,
                    "account_name": match.account_name,
                    "score": match.score,
                    "candidates": match.candidates,
                },
                rule_fired=f"match_{match.method}",
                runs_dir=runs_dir,
            ),
        )
        result["match"] = {
            "status": match.status,
            "account_id": match.account_id,
            "account_name": match.account_name,
            "score": match.score,
            "method": match.method,
            "candidates": match.candidates,
        }

        # ---- Stage: tie_break (judgment, only when ambiguous) -------------
        if match.status == "ambiguous":
            choice, fallback_used = llm_tie_break(match.candidates, lead, llm=llm)
            account_id, account_name = choice["account_id"], choice["name"]
            _record(
                stages,
                audit.append(
                    stage="tie_break",
                    lead_id=lead_id,
                    inputs={"candidates": match.candidates, "company": lead.get("company")},
                    decision={"account_id": account_id, "account_name": account_name},
                    rule_fired="llm_tie_break" if not fallback_used else "fallback_highest_score",
                    fallback_used=fallback_used,
                    runs_dir=runs_dir,
                ),
            )
        elif match.status == "matched":
            account_id, account_name = match.account_id, match.account_name

        # ---- Stage: new_account (structured exit on no-match) -------------
        else:
            proposal = {
                "action": "create_account",
                "company": lead.get("company"),
                "domain": lead.get("domain"),
                "country": lead.get("country"),
                "state": lead.get("state"),
                "best_fuzzy_score": match.score,
            }
            _record(
                stages,
                audit.append(
                    stage="new_account",
                    lead_id=lead_id,
                    inputs={"company": lead.get("company"), "domain": lead.get("domain")},
                    decision=proposal,
                    rule_fired="no_match_below_threshold",
                    runs_dir=runs_dir,
                ),
            )
            result.update(status="new_account", proposal=proposal)
            return result

        result["resolved_account"] = {"account_id": account_id, "name": account_name}

        # ---- Stage: enrich ------------------------------------------------
        context = warehouse.get_account_context(con, account_id)
        account = context["account"]
        _record(
            stages,
            audit.append(
                stage="enrich",
                lead_id=lead_id,
                inputs={"account_id": account_id},
                decision={
                    "industry": account["industry"],
                    "employee_band": account["employee_band"],
                    "territory": account["territory"],
                    "n_contacts": len(context["contacts"]),
                    "n_recent_activities": len(context["recent_activities"]),
                    "n_open_opportunities": len(context["open_opportunities"]),
                },
                rule_fired="warehouse_account_context",
                runs_dir=runs_dir,
            ),
        )

        # ---- Stage: intent_summary (judgment, when notes exist) -----------
        notes = (lead.get("notes") or "").strip()
        if notes:
            intent, fallback_used = llm_intent_summary(notes, llm=llm)
            result["intent"] = intent
            _record(
                stages,
                audit.append(
                    stage="intent_summary",
                    lead_id=lead_id,
                    inputs={"notes": notes},
                    decision=intent,
                    rule_fired="llm_intent_summary" if not fallback_used else "fallback_keyword_rules",
                    fallback_used=fallback_used,
                    runs_dir=runs_dir,
                ),
            )

        # ---- Stage: score -------------------------------------------------
        payload = model.load(model_path or model.MODEL_PATH)
        scoring_clock = as_of or warehouse.demo_as_of(db_path or config.SEED_DB)
        features = features_as_of(con, account_id, scoring_clock)
        propensity = model.score(features, payload)
        result["score"] = propensity
        _record(
            stages,
            audit.append(
                stage="score",
                lead_id=lead_id,
                inputs={"account_id": account_id, "as_of": scoring_clock, "features": features},
                decision={"propensity_score": round(propensity, 4)},
                rule_fired="propensity_model_v1",
                runs_dir=runs_dir,
            ),
        )
    finally:
        con.close()

    # ---- Stage: route (routing module writes its own audit record) --------
    decision = routing.route_lead(
        lead_id=lead_id,
        country=lead.get("country"),
        state=lead.get("state"),
        propensity_score=propensity,
        runs_dir=runs_dir,
    )
    stages.append(
        {
            "stage": "routing",
            "decision": {
                "territory": decision.territory,
                "rep": decision.rep,
                "priority": decision.priority,
                "sla_minutes": decision.sla_minutes,
            },
            "rule_fired": decision.rule_fired,
            "fallback_used": decision.fallback_used,
        }
    )

    # ---- HITL gate ---------------------------------------------------------
    proposal = {
        "lead_id": lead_id,
        "account_id": account_id,
        "account_name": account_name,
        "rep": decision.rep,
        "territory": decision.territory,
        "priority": decision.priority,
        "sla_minutes": decision.sla_minutes,
        "propensity_score": round(propensity, 4),
        "intent": result.get("intent"),
    }
    result["proposal"] = proposal
    _record(
        stages,
        audit.append(
            stage="pending_approval",
            lead_id=lead_id,
            inputs={"proposal": proposal},
            decision={"awaiting": "human_approval", "proposal": proposal},
            rule_fired="hitl_gate",
            runs_dir=runs_dir,
        ),
    )

    if approve is True:
        _record(
            stages,
            audit.append(
                stage="assignment_applied",
                lead_id=lead_id,
                inputs={"proposal": proposal},
                decision={"rep": decision.rep, "territory": decision.territory,
                          "priority": decision.priority},
                rule_fired="human_approved",
                runs_dir=runs_dir,
            ),
        )
        result["status"] = "assignment_applied"
    elif approve is False:
        _record(
            stages,
            audit.append(
                stage="routed_to_review",
                lead_id=lead_id,
                inputs={"proposal": proposal},
                decision={"disposition": "manual_review_queue"},
                rule_fired="human_rejected",
                runs_dir=runs_dir,
            ),
        )
        result["status"] = "routed_to_review"
    else:
        result["status"] = "pending_approval"
    return result
