"""Agentic lead pipeline: match -> judgment -> enrich -> score -> route -> HITL.

THE HARD GUARANTEE: with no ANTHROPIC_API_KEY and no llm flag, every stage
runs identically via deterministic fallbacks — zero network. The two
judgment steps (tie_break, intent_summary) are the ONLY places an LLM can
participate, and it can never expand its own action space: a tie-break
answer outside the candidate list, an over-long intent summary, an error,
or an empty response all collapse to the deterministic fallback. The
`anthropic` package is lazy-imported inside the guarded call path only.

Every stage appends one audit record; the HITL gate never applies an
assignment without an explicit approve. A pending proposal persists to
runs/pending/<lead_id>.json, and an approve/reject applies EXACTLY that
stored proposal — the human always acts on the proposal they saw.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
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
MAX_SUMMARY_CHARS = 200
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
LLM_MODEL = "claude-haiku-4-5"
LLM_TIMEOUT_SECONDS = 10.0

PENDING_DIRNAME = "pending"
RESOLVED_DIRNAME = "resolved"
OUTBOX_DIRNAME = "outbox"

# Lead fields that must be str or null when present (lead_id is required).
_LEAD_STR_FIELDS = (
    "company", "domain", "notes", "country", "state", "contact_name", "email", "title",
)

# lead_id doubles as a filename under runs/{pending,resolved,outbox}/, so it
# must be a bounded identifier — no separators, no traversal, no hidden files.
_LEAD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# A routing proposal a human can approve must carry every field the human was
# shown. Anything less is corrupt and fails closed at the approval gate.
_PROPOSAL_REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "proposal_id": str,
    "lead_id": str,
    "account_id": int,
    "account_name": str,
    "rep": str,
    "territory": str,
    "priority": str,
    "sla_minutes": int,
    "propensity_score": (int, float),
}


def _safe_lead_id(lead_id: str) -> bool:
    return isinstance(lead_id, str) and bool(_LEAD_ID_RE.fullmatch(lead_id))


# ---------------------------------------------------------------------------
# LLM plumbing (the ONLY network-capable path, always guarded)
# ---------------------------------------------------------------------------

_LLM_UNAVAILABLE_MSG = "llm_unavailable: anthropic not installed (install extra: llm)"
_llm_failure_reason: str | None = None


def _llm_available(llm: bool) -> bool:
    return bool(llm and os.environ.get("ANTHROPIC_API_KEY"))


def _note_llm_unavailable() -> None:
    """Missing package is NOT the same failure as an API error: say so."""
    global _llm_failure_reason
    _llm_failure_reason = _LLM_UNAVAILABLE_MSG
    print(_LLM_UNAVAILABLE_MSG, file=sys.stderr)


def _consume_llm_failure_reason() -> str | None:
    global _llm_failure_reason
    reason, _llm_failure_reason = _llm_failure_reason, None
    return reason


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
        except ImportError:
            _note_llm_unavailable()
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

    Guardrail: label must come from INTENT_LABELS, the summary must be
    <=20 words and <=200 characters (control characters stripped);
    anything else -> deterministic keyword fallback. The summary is still
    model-authored free text: renderers must present it as untrusted.
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
                label = lines[0].lower()
                summary = _CONTROL_CHARS.sub(" ", lines[1]).strip()
                if (
                    label in INTENT_LABELS
                    and 0 < len(summary.split()) <= MAX_SUMMARY_WORDS
                    and len(summary) <= MAX_SUMMARY_CHARS
                ):
                    return {"intent": label, "summary": summary}, False
            # Malformed/over-long/empty: fall through to fallback.
        except ImportError:
            _note_llm_unavailable()
        except Exception:
            pass
    return _deterministic_intent(notes), True


# ---------------------------------------------------------------------------
# Lead validation
# ---------------------------------------------------------------------------

def validate_lead(lead) -> None:
    """Reject malformed lead payloads with ONE clear error naming each field."""
    if not isinstance(lead, dict):
        raise ValueError("invalid lead: expected a JSON object")
    problems = []
    lead_id = lead.get("lead_id")
    if not isinstance(lead_id, str) or not lead_id.strip():
        problems.append("lead_id (required, non-empty string)")
    elif not _safe_lead_id(lead_id):
        # lead_id becomes a filename under runs/ — a path-shaped id like
        # "../escaped" must never reach the filesystem layer.
        problems.append(
            "lead_id (must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}; "
            "no slashes or other separators)"
        )
    for field_name in _LEAD_STR_FIELDS:
        value = lead.get(field_name)
        if value is not None and not isinstance(value, str):
            problems.append(
                f"{field_name} (expected string or null, got {type(value).__name__})"
            )
    if problems:
        raise ValueError("invalid lead: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# HITL pending-proposal persistence
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, payload: dict) -> None:
    """Temp sibling + os.replace, mirroring routing._save_state: a killed
    process can never leave a truncated JSON file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def _state_path(lead_id: str, dirname: str, runs_dir: Path | str | None) -> Path:
    """Resolve runs/<dirname>/<lead_id>.json, refusing any lead_id that could
    escape the directory. Both the identifier grammar and the resolved path
    are checked — defense in depth, not one or the other."""
    if not _safe_lead_id(lead_id):
        raise ValueError(f"unsafe lead_id {lead_id!r}: not a valid identifier")
    base = Path(runs_dir) if runs_dir else config.RUNS_DIR
    directory = base / dirname
    path = directory / f"{lead_id}.json"
    if path.resolve().parent != directory.resolve():
        raise ValueError(f"unsafe lead_id {lead_id!r}: escapes {dirname}/")
    return path


def _pending_path(lead_id: str, runs_dir: Path | str | None) -> Path:
    return _state_path(lead_id, PENDING_DIRNAME, runs_dir)


def _resolved_path(lead_id: str, runs_dir: Path | str | None) -> Path:
    return _state_path(lead_id, RESOLVED_DIRNAME, runs_dir)


def _outbox_path(lead_id: str, runs_dir: Path | str | None) -> Path:
    return _state_path(lead_id, OUTBOX_DIRNAME, runs_dir)


def _valid_proposal(proposal, lead_id: str) -> bool:
    """A pending proposal is only approvable if it still carries every field
    the human reviewed, with the right types, for the right lead."""
    if not isinstance(proposal, dict):
        return False
    for field_name, expected in _PROPOSAL_REQUIRED_FIELDS.items():
        value = proposal.get(field_name)
        if not isinstance(value, expected) or isinstance(value, bool):
            return False
        if isinstance(value, str) and not value.strip():
            return False
    return proposal["lead_id"] == lead_id


def _store_pending(lead_id: str, proposal: dict, runs_dir,
                   lead_path: Path | str | None = None) -> None:
    payload: dict = {"lead_id": lead_id, "proposal": proposal}
    if lead_path is not None:
        # Remembering the source path lets a later approve find this pending
        # proposal even after the inbound lead file is deleted or edited.
        payload["lead_path"] = str(Path(lead_path).resolve())
    _atomic_write_json(_pending_path(lead_id, runs_dir), payload)


def _load_pending(lead_id: str, runs_dir) -> dict | None:
    path = _pending_path(lead_id, runs_dir)
    if not path.exists():
        return None
    try:
        pending = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(pending, dict) or not _valid_proposal(pending.get("proposal"), lead_id):
        return None
    return pending


def _clear_pending(lead_id: str, runs_dir) -> None:
    path = _pending_path(lead_id, runs_dir)
    if path.exists():
        path.unlink()


def _store_resolved(lead_id: str, disposition: str, proposal: dict, runs_dir) -> None:
    """Record the terminal disposition so a repeated approve/reject replays
    the recorded outcome instead of re-running anything."""
    _atomic_write_json(
        _resolved_path(lead_id, runs_dir),
        {"lead_id": lead_id, "disposition": disposition, "proposal": proposal},
    )


def _load_resolved(lead_id: str, runs_dir) -> dict | None:
    path = _resolved_path(lead_id, runs_dir)
    if not path.exists():
        return None
    try:
        resolved = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if (
        not isinstance(resolved, dict)
        or not isinstance(resolved.get("disposition"), str)
        or not isinstance(resolved.get("proposal"), dict)
    ):
        return None
    return resolved


def _write_outbox(lead_id: str, proposal: dict, runs_dir) -> str:
    """The assignment side effect the HITL gate protects: an idempotent
    outbox record a CRM adapter would consume. Written once per lead —
    a replayed approve never rewrites it."""
    path = _outbox_path(lead_id, runs_dir)
    if not path.exists():
        _atomic_write_json(path, {
            "action": "apply_assignment",
            "lead_id": lead_id,
            "proposal_id": proposal.get("proposal_id"),
            "rep": proposal.get("rep"),
            "territory": proposal.get("territory"),
            "priority": proposal.get("priority"),
            "sla_minutes": proposal.get("sla_minutes"),
            "applied_at": datetime.now(timezone.utc).isoformat(),
        })
    return f"runs/{OUTBOX_DIRNAME}/{lead_id}.json"


def _check_proposal_id(lead_id: str, proposal: dict, proposal_id: str | None,
                       runs_dir) -> None:
    """Bind the decision to the exact proposal the human reviewed. A caller
    that names a proposal_id must name the stored one; a mismatch means the
    reviewed proposal was superseded — audited and rejected, never applied."""
    if proposal_id is None:
        return
    stored = proposal.get("proposal_id")
    if proposal_id != stored:
        audit.append(
            stage="stale_approval_rejected",
            lead_id=lead_id,
            inputs={"requested_proposal_id": proposal_id,
                    "stored_proposal_id": stored},
            decision={"reason": "proposal_id_mismatch"},
            rule_fired="stale_approval",
            runs_dir=runs_dir,
        )
        raise ValueError(
            f"approval names proposal {proposal_id!r} but the stored proposal "
            f"for lead {lead_id!r} is {stored!r} — the proposal you reviewed "
            "no longer exists; re-open the lead, review the current proposal, "
            "then decide (audited as stale_approval_rejected)"
        )


def _resolve_pending(lead_id: str, pending: dict, approve: bool,
                     llm: bool, runs_dir, proposal_id: str | None = None) -> dict:
    """Apply or reject the STORED pending proposal — never re-run the
    pipeline, never advance rotation state a second time."""
    proposal = pending["proposal"]
    _check_proposal_id(lead_id, proposal, proposal_id, runs_dir)
    stages: list[dict] = []
    result: dict = {"lead_id": lead_id, "stages": stages, "llm": llm,
                    "proposal": proposal}
    if approve:
        outbox = _write_outbox(lead_id, proposal, runs_dir)
        _record(
            stages,
            audit.append(
                stage="assignment_applied",
                lead_id=lead_id,
                inputs={"proposal": proposal},
                decision={"rep": proposal.get("rep"),
                          "territory": proposal.get("territory"),
                          "priority": proposal.get("priority"),
                          "proposal_id": proposal.get("proposal_id"),
                          "outbox": outbox},
                rule_fired="human_approved",
                runs_dir=runs_dir,
            ),
        )
        result["status"] = "assignment_applied"
    else:
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
    _store_resolved(lead_id, result["status"], proposal, runs_dir)
    _clear_pending(lead_id, runs_dir)
    return result


def _extract_lead_id(lead_path: Path | str) -> str | None:
    """Cheap lead_id extraction for the approve path: read the JSON and
    require only a non-empty string lead_id — full validation happens later,
    and only when the pipeline actually has to run."""
    try:
        lead = json.loads(Path(lead_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(lead, dict):
        return None
    lead_id = lead.get("lead_id")
    if isinstance(lead_id, str) and _safe_lead_id(lead_id):
        return lead_id
    return None


def _find_pending_lead_id_by_path(lead_path: Path | str, runs_dir) -> str | None:
    """When the source lead file is gone or unreadable, recover the lead_id
    from a stored pending proposal that remembers this lead_path."""
    base = Path(runs_dir) if runs_dir else config.RUNS_DIR
    pending_dir = base / PENDING_DIRNAME
    if not pending_dir.is_dir():
        return None
    target = str(Path(lead_path).resolve())
    for path in sorted(pending_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("lead_path") == target:
            return path.stem
    return None


def _resolve_from_state(lead_id: str, approve: bool, llm: bool, runs_dir,
                        proposal_id: str | None = None) -> dict | None:
    """Approve/reject state machine, consulted BEFORE any pipeline re-run.

    Returns a result dict when the stored state fully answers the request,
    None when a fresh run is required (no pending, no resolved record).
    Raises ValueError (fail closed) on a corrupt pending proposal: the
    operator must never have an unseen, re-computed proposal auto-applied.
    """
    pending_file = _pending_path(lead_id, runs_dir)
    if pending_file.exists():
        pending = _load_pending(lead_id, runs_dir)
        if pending is not None:
            return _resolve_pending(lead_id, pending, approve, llm, runs_dir,
                                    proposal_id=proposal_id)
        # Corrupt/invalid pending: audited discard + hard stop, never a
        # silent re-run that would apply a proposal no human ever saw.
        audit.append(
            stage="pending_discarded",
            lead_id=lead_id,
            inputs={"pending_path": str(pending_file)},
            decision={"reason": "corrupt_pending_proposal"},
            rule_fired="corrupt_pending",
            runs_dir=runs_dir,
        )
        pending_file.unlink()
        raise ValueError(
            f"pending proposal for lead {lead_id!r} was corrupt and has been "
            "discarded (audited as pending_discarded); re-run WITHOUT "
            "--approve/--reject to generate a fresh proposal, review it, "
            "then approve"
        )
    resolved = _load_resolved(lead_id, runs_dir)
    if resolved is not None:
        # Idempotent replay: the disposition already happened. Return the
        # recorded outcome unchanged — no new audit decision, no pipeline
        # re-run, no rotation advance. A named proposal_id must still match
        # the proposal that was actually resolved.
        _check_proposal_id(lead_id, resolved["proposal"], proposal_id, runs_dir)
        return {
            "lead_id": lead_id,
            "stages": [],
            "llm": llm,
            "proposal": resolved["proposal"],
            "status": resolved["disposition"],
            "idempotent_replay": True,
        }
    return None


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


def _record_llm_unavailable(stages, lead_id, judgment_stage, runs_dir) -> None:
    reason = _consume_llm_failure_reason()
    if reason:
        _record(
            stages,
            audit.append(
                stage="llm_unavailable",
                lead_id=lead_id,
                inputs={"judgment_stage": judgment_stage},
                decision={"reason": reason},
                rule_fired="llm_unavailable",
                fallback_used=True,
                runs_dir=runs_dir,
            ),
        )


def run(
    lead_path: Path | str,
    llm: bool = False,
    approve: bool | None = None,
    runs_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    model_path: Path | str | None = None,
    as_of: datetime | None = None,
    proposal_id: str | None = None,
    renew: bool = False,
) -> dict:
    """Run the full pipeline for one inbound lead JSON file.

    approve: None -> stop at the HITL gate (pending_approval) and persist
             the proposal to runs/pending/<lead_id>.json; when a valid
             pending proposal already exists it is RETURNED, not re-computed
             — a preview can never silently overwrite what a human is
             reviewing. Pass renew=True to explicitly supersede it (audited).
             True -> apply the assignment (a stored pending proposal, when
             one exists, is applied as-is without re-running the pipeline);
             False -> route to human review (likewise consumes any pending).
    proposal_id: when set on an approve/reject, the stored proposal must
             carry exactly this id — a mismatch fails closed (audited as
             stale_approval_rejected) instead of applying a proposal the
             human never reviewed.
    as_of: scoring clock; defaults to warehouse.demo_as_of() — the frozen
           dataset gets a frozen clock, so scores stay meaningful on any
           demo day instead of decaying with the real calendar.
    """
    # HITL integrity: an approve/reject consults stored state FIRST — it
    # consumes the STORED proposal (the one the human actually saw) or
    # replays the recorded disposition, and only falls through to a fresh
    # run when neither exists. Rotation state can therefore never advance
    # twice for one lead across pending -> approve -> repeat-approve.
    if approve is not None:
        lead_id = _extract_lead_id(lead_path)
        if lead_id is None:
            # Source lead file deleted/unreadable: the stored pending
            # proposal (which remembers its lead_path) is still approvable.
            lead_id = _find_pending_lead_id_by_path(lead_path, runs_dir)
        if lead_id is not None:
            resolution = _resolve_from_state(lead_id, approve, llm, runs_dir,
                                             proposal_id=proposal_id)
            if resolution is not None:
                return resolution

    # Reading, parsing and validating the lead sit INSIDE the audited
    # failure boundary: a malformed payload leaves a pipeline_error record,
    # never a silent exit. Until a lead_id is parsed, the file stem is the
    # best identifier available for the record.
    lead_id_hint = Path(lead_path).stem
    try:
        lead = json.loads(Path(lead_path).read_text())
        validate_lead(lead)
    except Exception as exc:
        audit.append(
            stage="pipeline_error",
            lead_id=lead_id_hint,
            inputs={"lead_path": str(lead_path)},
            decision={"error": type(exc).__name__, "message": str(exc),
                      "last_stage": None},
            rule_fired="pipeline_exception",
            runs_dir=runs_dir,
        )
        raise
    lead_id = lead["lead_id"]

    # Immutable review contract: while a valid pending proposal exists, a
    # plain preview run returns THAT proposal instead of computing a new
    # one. Only an explicit renew supersedes it, and the supersede itself
    # is audited before the old proposal is discarded.
    if approve is None:
        existing = _load_pending(lead_id, runs_dir)
        if existing is not None:
            if not renew:
                return {
                    "lead_id": lead_id,
                    "stages": [],
                    "llm": llm,
                    "proposal": existing["proposal"],
                    "status": "pending_approval",
                    "pending_replayed": True,
                }
            audit.append(
                stage="pending_superseded",
                lead_id=lead_id,
                inputs={"superseded_proposal_id":
                        existing["proposal"].get("proposal_id")},
                decision={"reason": "explicit_renew"},
                rule_fired="pending_superseded",
                runs_dir=runs_dir,
            )
            _clear_pending(lead_id, runs_dir)

    stages: list[dict] = []
    result: dict = {"lead_id": lead_id, "stages": stages, "llm": llm}
    try:
        return _execute(
            lead, lead_id, stages, result,
            llm=llm, approve=approve, runs_dir=runs_dir,
            db_path=db_path, model_path=model_path, as_of=as_of,
            lead_path=lead_path,
        )
    except Exception as exc:
        # Every run terminates with a recorded outcome: a crash leaves a
        # terminal pipeline_error record, never a silent partial trail.
        audit.append(
            stage="pipeline_error",
            lead_id=lead_id,
            inputs={"lead_path": str(lead_path)},
            decision={
                "error": type(exc).__name__,
                "message": str(exc),
                "last_stage": stages[-1]["stage"] if stages else None,
            },
            rule_fired="pipeline_exception",
            runs_dir=runs_dir,
        )
        raise


def _execute(
    lead: dict,
    lead_id: str,
    stages: list[dict],
    result: dict,
    llm: bool,
    approve: bool | None,
    runs_dir,
    db_path,
    model_path,
    as_of,
    lead_path=None,
) -> dict:
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
            _record_llm_unavailable(stages, lead_id, "tie_break", runs_dir)
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
            _record_llm_unavailable(stages, lead_id, "intent_summary", runs_dir)

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
        "proposal_id": uuid.uuid4().hex[:12],
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
        outbox = _write_outbox(lead_id, proposal, runs_dir)
        _record(
            stages,
            audit.append(
                stage="assignment_applied",
                lead_id=lead_id,
                inputs={"proposal": proposal},
                decision={"rep": decision.rep, "territory": decision.territory,
                          "priority": decision.priority,
                          "proposal_id": proposal["proposal_id"],
                          "outbox": outbox},
                rule_fired="human_approved",
                runs_dir=runs_dir,
            ),
        )
        result["status"] = "assignment_applied"
        _store_resolved(lead_id, result["status"], proposal, runs_dir)
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
        _store_resolved(lead_id, result["status"], proposal, runs_dir)
    else:
        # Persist the pending proposal so a later approve/reject acts on
        # EXACTLY this proposal instead of re-running the pipeline.
        _store_pending(lead_id, proposal, runs_dir, lead_path=lead_path)
        result["status"] = "pending_approval"
    return result
