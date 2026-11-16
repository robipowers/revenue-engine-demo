"""Salesforce adapter: route real Leads in an org with the engine's own rules.

Reads open Leads, Accounts, users and queues through the Salesforce CLI
(`sf ... --json`), then for each Lead:

  match   lead-to-account, same 3-band policy as the warehouse matcher
  route   matched  -> the Account's owner (if that user is active)
          otherwise -> the territory queue for the Lead's country/state
  write   OwnerId, Matched_Account__c, Routing_Rule__c on the Lead

Default is a dry run that only prints proposals. `--apply` writes. Every
match, route and write is one record in the append-only audit log.

What this adapter does NOT do (yet): propensity scoring (the model's
features live in the demo warehouse, not in the org), round robin inside a
queue, SLA timers, lead conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

from revenue_engine import audit, config, matching

DEFAULT_ORG = os.environ.get("RELAY_SF_ORG", "robi-dev")
QUEUE_PREFIX = "Relay_"

# A lead writing from a personal mailbox tells us nothing about its company.
FREE_MAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "live.com", "icloud.com", "me.com", "aol.com", "proton.me", "protonmail.com",
    "gmx.com", "gmx.de", "web.de", "mail.com",
}


class SalesforceError(RuntimeError):
    """The Salesforce CLI failed or returned something unreadable."""


def _run_sf(args: list[str]) -> dict:
    """Run one `sf` command with --json and return its `result`."""
    try:
        proc = subprocess.run(
            ["sf", *args, "--json"], capture_output=True, text=True, timeout=120
        )
    except FileNotFoundError as exc:
        raise SalesforceError("Salesforce CLI (`sf`) is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise SalesforceError(f"sf {args[0:3]} timed out") from exc
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SalesforceError(f"sf returned non-JSON output: {proc.stdout[:200]!r}") from exc
    if payload.get("status") != 0:
        raise SalesforceError(payload.get("message") or f"sf failed: {payload}")
    return payload.get("result") or {}


def _quote(value) -> str:
    """One value for `sf data update record --values`."""
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


class SalesforceClient:
    def __init__(self, org: str = DEFAULT_ORG, runner=_run_sf):
        self.org = org
        self._run = runner

    def query(self, soql: str) -> list[dict]:
        result = self._run(["data", "query", "--target-org", self.org, "--query", soql])
        return result.get("records", [])

    def update(self, sobject: str, record_id: str, values: dict) -> None:
        pairs = " ".join(f"{k}={_quote(v)}" for k, v in values.items())
        self._run([
            "data", "update", "record", "--target-org", self.org,
            "--sobject", sobject, "--record-id", record_id, "--values", pairs,
        ])


# --- pure helpers -----------------------------------------------------------

def domain_of(value: str | None) -> str | None:
    """Bare host from a website or an email: 'https://www.X.com/a' -> 'x.com'."""
    if not value:
        return None
    text = value.strip().lower()
    if "@" in text and "/" not in text:
        text = text.rsplit("@", 1)[1]
    else:
        text = urlparse(text if "//" in text else f"//{text}").hostname or ""
    if text.startswith("www."):
        text = text[4:]
    return text or None


def lead_domain(lead: dict) -> str | None:
    """Company domain for a Lead: work-email domain first, then Website."""
    email_domain = domain_of(lead.get("Email"))
    if email_domain and email_domain not in FREE_MAIL_DOMAINS:
        return email_domain
    return domain_of(lead.get("Website"))


def queue_name(territory: str) -> str:
    return QUEUE_PREFIX + territory.replace("-", "_")


@dataclass(frozen=True)
class Proposal:
    lead_id: str
    lead_name: str
    match_status: str          # matched | ambiguous | no_match
    match_method: str          # domain_exact | fuzzy_name | none
    match_score: float | None
    account_id: str | None
    account_name: str | None
    territory: str
    rule_fired: str            # account_owner | territory_queue | ...
    fallback_used: bool
    new_owner_id: str | None
    new_owner_name: str | None
    changes: dict = field(default_factory=dict)   # fields that would change
    candidates: list[dict] = field(default_factory=list)


def propose(lead: dict, accounts: list[dict], active_user_ids: set[str],
            queues: dict[str, dict]) -> Proposal:
    """Decide match + owner for one Lead. Pure: no I/O, no writes."""
    records = [
        {"id": a["Id"], "name": a["Name"], "domain": domain_of(a.get("Website"))}
        for a in accounts
    ]
    by_id = {a["Id"]: a for a in accounts}
    match = matching.match_records(records, name=lead.get("Company"), domain=lead_domain(lead))

    country = lead.get("Country") or ""
    territory = config.resolve_territory(country, lead.get("StateCode") or lead.get("State"))
    territory_known = country == "United States" or country in config.COUNTRY_TO_TERRITORY
    queue = queues.get(queue_name(territory))

    account_id = account_name = None
    owner_id = owner_name = None
    fallback = False

    if match.status == "matched":
        account_id, account_name = match.account_id, match.account_name
        account = by_id[account_id]
        if account.get("OwnerId") in active_user_ids:
            owner_id = account["OwnerId"]
            owner_name = (account.get("Owner") or {}).get("Name")
            rule = "account_owner"
        else:
            rule, fallback = "account_owner_inactive_to_queue", True
    elif match.status == "ambiguous":
        # Never guess between accounts on a live record: a person decides.
        rule = "ambiguous_match_to_queue"
    else:
        rule = "territory_queue"

    if owner_id is None:
        if queue:
            owner_id, owner_name = queue["Id"], queue["Name"]
            if not territory_known:
                rule, fallback = "fallback_territory_queue", True
        else:
            rule, fallback = "no_queue_configured", True

    wanted = {"Routing_Rule__c": rule}
    if owner_id:
        wanted["OwnerId"] = owner_id
    if account_id:
        wanted["Matched_Account__c"] = account_id
    changes = {k: v for k, v in wanted.items() if lead.get(k) != v}

    return Proposal(
        lead_id=lead["Id"],
        lead_name=lead.get("Name") or "",
        match_status=match.status,
        match_method=match.method,
        match_score=match.score,
        account_id=account_id,
        account_name=account_name,
        territory=territory,
        rule_fired=rule,
        fallback_used=fallback,
        new_owner_id=owner_id,
        new_owner_name=owner_name,
        changes=changes,
        candidates=match.candidates,
    )


# --- org I/O ----------------------------------------------------------------

LEAD_FIELDS = (
    "Id, Name, Company, Email, Website, Country, State, StateCode, "
    "OwnerId, Owner.Name, Matched_Account__c, Routing_Rule__c"
)


def load_org(client: SalesforceClient, lead_id: str | None = None) -> dict:
    where = "IsConverted = false"
    if lead_id:
        if not lead_id.isalnum():
            raise SalesforceError(f"not a Salesforce record id: {lead_id!r}")
        where += f" AND Id = '{lead_id}'"
    leads = client.query(f"SELECT {LEAD_FIELDS} FROM Lead WHERE {where} ORDER BY CreatedDate")
    accounts = client.query("SELECT Id, Name, Website, OwnerId, Owner.Name FROM Account")
    users = client.query("SELECT Id FROM User WHERE IsActive = true AND UserType = 'Standard'")
    queues = client.query(
        "SELECT Id, Name, DeveloperName FROM Group "
        f"WHERE Type = 'Queue' AND DeveloperName LIKE '{QUEUE_PREFIX}%'"
    )
    return {
        "leads": leads,
        "accounts": accounts,
        "active_user_ids": {u["Id"] for u in users},
        "queues": {q["DeveloperName"]: q for q in queues},
    }


def route_org(client: SalesforceClient, apply: bool = False, lead_id: str | None = None,
              runs_dir=None) -> list[Proposal]:
    """Propose (and with apply=True, write) routing for open Leads in the org."""
    org = load_org(client, lead_id)
    proposals: list[Proposal] = []
    for lead in org["leads"]:
        p = propose(lead, org["accounts"], org["active_user_ids"], org["queues"])
        proposals.append(p)
        audit.append(
            stage="sf_match", lead_id=p.lead_id,
            inputs={"company": lead.get("Company"), "domain": lead_domain(lead),
                    "accounts_considered": len(org["accounts"])},
            decision={"status": p.match_status, "method": p.match_method,
                      "score": p.match_score, "account_id": p.account_id,
                      "candidates": p.candidates},
            rule_fired=p.match_method, runs_dir=runs_dir,
        )
        audit.append(
            stage="sf_route", lead_id=p.lead_id,
            inputs={"country": lead.get("Country"), "state": lead.get("StateCode"),
                    "match_status": p.match_status, "current_owner": lead.get("OwnerId")},
            decision={"territory": p.territory, "owner_id": p.new_owner_id,
                      "owner_name": p.new_owner_name, "changes": p.changes},
            rule_fired=p.rule_fired, fallback_used=p.fallback_used, runs_dir=runs_dir,
        )
        if not apply:
            continue
        if not p.changes:
            audit.append(
                stage="sf_write", lead_id=p.lead_id, inputs={"changes": {}},
                decision={"written": False}, rule_fired="no_change", runs_dir=runs_dir,
            )
            continue
        client.update("Lead", p.lead_id, p.changes)
        audit.append(
            stage="sf_write", lead_id=p.lead_id, inputs={"changes": p.changes},
            decision={"written": True, "org": client.org},
            rule_fired="lead_updated", runs_dir=runs_dir,
        )
    return proposals


def _print(proposals: list[Proposal], applied: bool) -> None:
    if not proposals:
        print("No open leads found.")
        return
    for p in proposals:
        score = "" if p.match_score is None else f" {p.match_score:g}"
        match = f"{p.match_status} ({p.match_method}{score})"
        if p.account_name:
            match += f" -> {p.account_name}"
        print(f"{p.lead_name}  [{p.lead_id}]")
        print(f"  match   {match}")
        if p.candidates:
            names = ", ".join(f"{c['name']} {c['score']:g}" for c in p.candidates)
            print(f"          candidates: {names}")
        print(f"  route   {p.rule_fired}  territory {p.territory}  owner {p.new_owner_name or '(unchanged)'}")
        verb = "wrote" if applied else "would write"
        print(f"  {verb:<7} {p.changes or 'nothing (already routed)'}")
    if not applied:
        print("\nDry run. Nothing was written. Re-run with --apply to write to the org.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--org", default=DEFAULT_ORG, help="sf alias or username")
    parser.add_argument("--lead", help="route one Lead by record id")
    parser.add_argument("--apply", action="store_true", help="write the proposals to the org")
    parser.add_argument("--json", action="store_true", help="print proposals as JSON")
    args = parser.parse_args()
    try:
        proposals = route_org(SalesforceClient(args.org), apply=args.apply, lead_id=args.lead)
    except SalesforceError as exc:
        raise SystemExit(f"salesforce: {exc}")
    if args.json:
        print(json.dumps([asdict(p) for p in proposals], indent=2))
    else:
        _print(proposals, applied=args.apply)


if __name__ == "__main__":
    main()
