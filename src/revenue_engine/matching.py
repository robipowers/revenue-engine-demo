"""Account matching: exact domain shortcut, then fuzzy name with a 3-band policy.

Policy: score >= 90 auto-match; 70-89 ambiguous (candidates go to the U4
judgment step); < 70 no-match (new-account path). Deterministic — no LLM here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from revenue_engine.warehouse import get_account_by_domain, search_accounts_by_name

AUTO_MATCH_THRESHOLD = 90.0
AMBIGUOUS_THRESHOLD = 70.0

# Legal suffixes only — brand words like "Labs" or "Systems" stay.
_LEGAL_SUFFIXES = {"inc", "llc", "ltd", "corp", "corporation", "incorporated", "limited", "co"}


@dataclass(frozen=True)
class MatchResult:
    status: str  # matched | ambiguous | no_match
    account_id: int | None = None
    account_name: str | None = None
    score: float | None = None
    method: str = "none"  # domain_exact | fuzzy_name | none
    candidates: list[dict] = field(default_factory=list)


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, drop trailing legal suffixes."""
    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    tokens = cleaned.split()
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def match_account(con, name: str | None = None, domain: str | None = None) -> MatchResult:
    """Match a lead to a warehouse account by domain (exact) or name (fuzzy)."""
    if domain:
        account = get_account_by_domain(con, domain)
        if account:
            return MatchResult(
                status="matched",
                account_id=account["id"],
                account_name=account["name"],
                score=100.0,
                method="domain_exact",
            )

    if not name:
        return MatchResult(status="no_match")

    query = normalize_name(name)
    scored = [
        {
            "account_id": acct["id"],
            "name": acct["name"],
            "score": round(fuzz.token_set_ratio(query, normalize_name(acct["name"])), 1),
        }
        for acct in search_accounts_by_name(con)
    ]
    if not scored:
        # Empty accounts table: nothing to match against — new-account path.
        return MatchResult(status="no_match")
    scored.sort(key=lambda c: (-c["score"], c["account_id"]))
    best = scored[0]

    if best["score"] >= AUTO_MATCH_THRESHOLD:
        return MatchResult(
            status="matched",
            account_id=best["account_id"],
            account_name=best["name"],
            score=best["score"],
            method="fuzzy_name",
        )
    candidates = [c for c in scored if c["score"] >= AMBIGUOUS_THRESHOLD]
    if candidates:
        return MatchResult(
            status="ambiguous",
            score=best["score"],
            method="fuzzy_name",
            candidates=candidates,
        )
    return MatchResult(status="no_match", score=best["score"], method="fuzzy_name")
