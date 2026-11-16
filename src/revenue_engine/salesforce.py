"""Salesforce adapter: route real Leads in an org with the engine's own rules.

Reads Leads, Accounts, Contacts, users and queues through the Salesforce CLI
(`sf ... --json`). For each open Lead, in this order:

  consent   a lead from a consent region with no consent goes to a hold queue
  dedupe    a lead whose email already belongs to a routed lead gets that owner
  match     existing Contact email -> exact domain -> fuzzy name (3 bands);
            duplicate accounts are resolved by ordered tiebreakers;
            two different companies tied on name are never guessed
  owner     matched -> the account's owner, else an active owner up the
            parent chain; otherwise the territory queue
  assign    inside a queue: least-loaded member who is active, in working
            hours and under the open-lead cap; else the lead waits in the queue
  sla       a due time, from how hot the lead is

Default is a dry run. `--apply` writes OwnerId and the Routing_* fields.
`--sla` lists leads past their due time (and with --apply, requeues them).
Rules live in config/routing.json. Every decision is one audit record.

Not here: propensity scoring (the model's features live in the demo
warehouse, not the org), lead conversion, bulk-API writes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from rapidfuzz import fuzz, process

from revenue_engine import audit, config
from revenue_engine.matching import normalize_name

DEFAULT_ORG = os.environ.get("RELAY_SF_ORG", "robi-dev")
QUEUE_PREFIX = "Relay_"
RULES_PATH = config.ROOT / "config" / "routing.json"

# A lead writing from a personal mailbox tells us nothing about its company.
FREE_MAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "live.com", "icloud.com", "me.com", "aol.com", "proton.me", "protonmail.com",
    "gmx.com", "gmx.de", "web.de", "mail.com",
}

DEFAULT_RULES = {
    "match": {"auto_threshold": 90, "ambiguous_threshold": 70},
    "tiebreakers": ["open_opportunity", "customer", "latest_activity", "oldest_account"],
    "customer_types": ["Customer - Direct", "Customer - Channel"],
    "hierarchy": {"mode": "matched_then_parent"},   # or "ultimate_parent"
    "consent": {"regions": [], "hold_queue": "Relay_Consent_Hold"},
    "round_robin": {"cap_open_leads": 25},
    "working_hours": {},
    "sla_minutes": {"hot": 15, "matched": 60, "default": 240},
    "sla_unworked_statuses": ["Open - Not Contacted"],
}


def load_rules(path: Path | str | None = None) -> dict:
    """Rules from config/routing.json over the defaults. A missing file is fine."""
    rules = json.loads(json.dumps(DEFAULT_RULES))
    p = Path(path) if path else RULES_PATH
    if p.exists():
        for key, value in json.loads(p.read_text()).items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict) and isinstance(rules.get(key), dict):
                rules[key].update(value)
            else:
                rules[key] = value
    return rules


# --- CLI client -------------------------------------------------------------

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
    if isinstance(value, bool):
        return "true" if value else "false"
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


# --- small pure helpers -----------------------------------------------------

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


def parse_sf_datetime(value: str | None) -> datetime | None:
    """'2026-11-16T17:40:00.000+0000' (or ...Z) -> aware datetime."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace("+0000", "+00:00"))


def format_sf_datetime(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def is_user(record_id: str | None) -> bool:
    return bool(record_id) and record_id.startswith("005")


def in_working_hours(territory: str, rules: dict, now: datetime) -> bool:
    hours = rules["working_hours"].get(territory)
    if not hours:
        return True
    local = now.astimezone(ZoneInfo(hours["tz"]))
    return local.weekday() < 5 and hours["start"] <= local.hour < hours["end"]


# --- org snapshot -----------------------------------------------------------

@dataclass
class Org:
    leads: list[dict] = field(default_factory=list)
    accounts: list[dict] = field(default_factory=list)
    users: dict[str, str] = field(default_factory=dict)            # active user id -> name
    queues: dict[str, dict] = field(default_factory=dict)          # DeveloperName -> record
    queue_members: dict[str, list[str]] = field(default_factory=dict)   # queue id -> user ids
    open_opps: dict[str, int] = field(default_factory=dict)        # account id -> open opps
    contacts_by_email: dict[str, str] = field(default_factory=dict)     # email -> account id
    lead_load: dict[str, int] = field(default_factory=dict)        # owner id -> open leads


class AccountIndex:
    """Accounts indexed once, so matching a lead does not rescan everything."""

    def __init__(self, org: Org, rules: dict):
        self.accounts = org.accounts
        self.by_id = {a["Id"]: a for a in org.accounts}
        self.open_opps = org.open_opps
        self.customer_types = set(rules["customer_types"])
        self.names = [normalize_name(a.get("Name") or "") for a in org.accounts]
        self.by_domain: dict[str, list[dict]] = {}
        for acct in org.accounts:
            dom = domain_of(acct.get("Website"))
            if dom:
                self.by_domain.setdefault(dom, []).append(acct)

    def is_customer(self, acct: dict) -> bool:
        return acct.get("Type") in self.customer_types

    def has_open_opp(self, acct: dict) -> bool:
        return self.open_opps.get(acct["Id"], 0) > 0


def tiebreak(pool: list[dict], index: AccountIndex, rules: dict) -> tuple[dict, str]:
    """Pick one of several records for the SAME company, by ordered business rules."""
    keys = {
        "open_opportunity": (lambda a: index.has_open_opp(a), max),
        "customer": (lambda a: index.is_customer(a), max),
        "latest_activity": (lambda a: a.get("LastActivityDate") or "", max),
        "oldest_account": (lambda a: a.get("CreatedDate") or "9999", min),
    }
    for name in rules["tiebreakers"]:
        fn, pick = keys[name]
        best = pick(fn(a) for a in pool)
        narrowed = [a for a in pool if fn(a) == best]
        if len(narrowed) < len(pool):
            pool = narrowed
            if len(pool) == 1:
                return pool[0], name
    return min(pool, key=lambda a: a["Id"]), "lowest_id"


@dataclass(frozen=True)
class Match:
    status: str                      # matched | ambiguous | no_match
    method: str = "none"             # contact_email | domain_exact | fuzzy_name | none
    score: float | None = None
    account: dict | None = None
    candidates: list[dict] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)   # ids of duplicate accounts
    tiebreak: str | None = None


def match_lead(lead: dict, org: Org, index: AccountIndex, rules: dict) -> Match:
    """Lead-to-account match. Strongest evidence first."""
    email = (lead.get("Email") or "").strip().lower()
    contact_account = org.contacts_by_email.get(email) if email else None
    if contact_account in index.by_id:
        return Match("matched", "contact_email", 100.0, index.by_id[contact_account])

    same_domain = index.by_domain.get(lead_domain(lead) or "", [])
    if len(same_domain) == 1:
        return Match("matched", "domain_exact", 100.0, same_domain[0])
    if len(same_domain) > 1:
        winner, rule = tiebreak(same_domain, index, rules)
        return Match("matched", "domain_exact", 100.0, winner,
                     duplicates=[a["Id"] for a in same_domain], tiebreak=rule)

    query = normalize_name(lead.get("Company") or "")
    if not query or not index.accounts:
        return Match("no_match")
    auto = rules["match"]["auto_threshold"]
    hits = process.extract(
        query, index.names, scorer=fuzz.token_set_ratio,
        score_cutoff=rules["match"]["ambiguous_threshold"], limit=None,
    )
    if not hits:
        return Match("no_match", "fuzzy_name")
    scored = sorted(
        ((round(score, 1), index.accounts[i]) for _, score, i in hits),
        key=lambda pair: (-pair[0], pair[1]["Id"]),
    )
    candidates = [{"account_id": a["Id"], "name": a["Name"], "score": s} for s, a in scored]
    top = scored[0][0]
    if top < auto:
        return Match("ambiguous", "fuzzy_name", top, candidates=candidates)
    best_name = normalize_name(scored[0][1]["Name"])
    if len({normalize_name(a["Name"]) for s, a in scored if s == top}) > 1:
        # Two different companies fit equally well. A person decides.
        return Match("ambiguous", "fuzzy_name", top, candidates=candidates)
    same_company = [a for s, a in scored if s >= auto and normalize_name(a["Name"]) == best_name]
    if len(same_company) > 1:
        winner, rule = tiebreak(same_company, index, rules)
        return Match("matched", "fuzzy_name", top, winner,
                     duplicates=[a["Id"] for a in same_company], tiebreak=rule)
    return Match("matched", "fuzzy_name", top, scored[0][1])


def resolve_account_owner(account: dict, org: Org, index: AccountIndex,
                          rules: dict) -> tuple[str | None, str]:
    """(owner user id, rule) for a matched account, walking parents if needed."""
    chain, seen, node = [], set(), account
    while node and node["Id"] not in seen:
        seen.add(node["Id"])
        chain.append(node)
        node = index.by_id.get(node.get("ParentId"))
    if rules["hierarchy"]["mode"] == "ultimate_parent" and len(chain) > 1:
        top = chain[-1]
        if top.get("OwnerId") in org.users:
            return top["OwnerId"], "ultimate_parent_owner"
    for depth, node in enumerate(chain):
        if node.get("OwnerId") in org.users:
            if depth:
                return node["OwnerId"], "parent_account_owner"
            rule = "customer_account_owner" if index.is_customer(node) else "account_owner"
            return node["OwnerId"], rule
    return None, "account_owner_inactive"


@dataclass
class RunState:
    """What changes while a batch runs: rep load and who owns which email."""
    load: dict[str, int] = field(default_factory=dict)
    email_owner: dict[str, tuple[str, str]] = field(default_factory=dict)   # email -> (owner, lead id)


def pick_from_queue(territory: str, org: Org, rules: dict, now: datetime,
                    state: RunState) -> tuple[str | None, str, str]:
    """(owner id, rule, reason). A rep if one is eligible, else the queue itself."""
    queue = org.queues.get(queue_name(territory))
    if not queue:
        return None, "no_queue_configured", "no queue for territory"
    members = [m for m in org.queue_members.get(queue["Id"], []) if m in org.users]
    if not members:
        return queue["Id"], "territory_queue", "queue has no active members"
    if not in_working_hours(territory, rules, now):
        return queue["Id"], "territory_queue", "outside working hours"
    cap = rules["round_robin"]["cap_open_leads"]
    eligible = [m for m in members if state.load.get(m, 0) < cap]
    if not eligible:
        return queue["Id"], "territory_queue", "all reps at cap"
    rep = min(eligible, key=lambda m: (state.load.get(m, 0), m))
    return rep, "round_robin_least_loaded", f"load {state.load.get(rep, 0)} of cap {cap}"


@dataclass(frozen=True)
class Proposal:
    lead_id: str
    lead_name: str
    match_status: str
    match_method: str
    match_score: float | None
    account_id: str | None
    account_name: str | None
    territory: str
    rule_fired: str
    fallback_used: bool
    new_owner_id: str | None
    new_owner_name: str | None
    note: str
    sla_due: str | None
    changes: dict = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    duplicate_accounts: list[str] = field(default_factory=list)


FALLBACK_RULES = {"no_queue_configured", "account_owner_inactive_to_queue",
                  "fallback_territory_queue"}


def propose(lead: dict, org: Org, index: AccountIndex, rules: dict, now: datetime,
            state: RunState, reroute: bool = False) -> Proposal:
    """Decide match, owner and SLA for one Lead. No I/O, no writes."""
    country = lead.get("Country") or ""
    territory = config.resolve_territory(country, lead.get("StateCode") or lead.get("State"))
    territory_known = country == "United States" or country in config.COUNTRY_TO_TERRITORY
    match = match_lead(lead, org, index, rules)
    account = match.account
    email = (lead.get("Email") or "").strip().lower()

    notes = [match.status if match.status != "matched"
             else f"{match.method} {match.score:g}"]
    if match.duplicates:
        notes.append(f"{len(match.duplicates)} duplicate accounts, tiebreak {match.tiebreak}")
    if match.status == "ambiguous":
        notes.append("candidates: " + ", ".join(
            f"{c['name']} {c['score']:g}" for c in match.candidates[:4]))

    def base(rule, owner_id, changes, sla_due=None):
        name = org.users.get(owner_id) or next(
            (q["Name"] for q in org.queues.values() if q["Id"] == owner_id), None)
        return Proposal(
            lead_id=lead["Id"], lead_name=lead.get("Name") or "",
            match_status=match.status, match_method=match.method, match_score=match.score,
            account_id=account["Id"] if account else None,
            account_name=account["Name"] if account else None,
            territory=territory, rule_fired=rule,
            fallback_used=rule in FALLBACK_RULES,
            new_owner_id=owner_id, new_owner_name=name,
            note="; ".join(notes)[:255], sla_due=sla_due, changes=changes,
            candidates=match.candidates, duplicate_accounts=match.duplicates,
        )

    # A lead a rep already owns is settled. Re-running must not shuffle owners.
    if lead.get("Routing_Rule__c") and is_user(lead.get("OwnerId")) and not reroute:
        state.email_owner.setdefault(email, (lead["OwnerId"], lead["Id"]))
        notes[:] = ["settled: a rep already owns this lead"]
        return base("already_routed", lead["OwnerId"], {})

    owner_id, rule = None, None

    # 1. Consent gate: nobody may work this lead yet.
    consent = rules["consent"]
    if territory in consent["regions"] and not lead.get("Consent_Given__c"):
        hold = org.queues.get(consent["hold_queue"])
        notes.append(f"no consent on record for {territory}")
        if hold:
            owner_id, rule = hold["Id"], "consent_hold"
        else:
            rule = "no_queue_configured"

    # 2. Same person already routed: keep them with the same owner.
    if rule is None and email in state.email_owner and state.email_owner[email][1] != lead["Id"]:
        owner_id, rule = state.email_owner[email][0], "duplicate_lead_same_owner"
        notes.append(f"same email as lead {state.email_owner[email][1]}")

    # 3. Matched account: its owner, or an active owner up the parent chain.
    if rule is None and account:
        owner_id, rule = resolve_account_owner(account, org, index, rules)
        if owner_id is None:
            rule = None
            notes.append("account owner inactive")

    # 4. Everything else: the territory queue, then a rep inside it.
    if rule is None:
        owner_id, rule, reason = pick_from_queue(territory, org, rules, now, state)
        notes.append(reason)
        if rule == "territory_queue":
            if match.status == "ambiguous":
                rule = "ambiguous_match_to_queue"
            elif account:
                rule = "account_owner_inactive_to_queue"
            elif not territory_known:
                rule = "fallback_territory_queue"

    if is_user(owner_id):
        state.load[owner_id] = state.load.get(owner_id, 0) + 1
        if email and rule != "consent_hold":
            state.email_owner.setdefault(email, (owner_id, lead["Id"]))

    # 5. SLA due time, from how hot the lead is.
    sla = rules["sla_minutes"]
    hot = account and (index.has_open_opp(account) or index.is_customer(account))
    minutes = sla["hot"] if hot else sla["matched"] if account else sla["default"]
    created = parse_sf_datetime(lead.get("CreatedDate")) or now
    due = created + timedelta(minutes=minutes)
    sla_due = format_sf_datetime(due)

    note = "; ".join(notes)[:255]
    changes = {}
    if lead.get("Routing_Rule__c") != rule:
        changes["Routing_Rule__c"] = rule
    if (lead.get("Routing_Note__c") or "") != note:
        changes["Routing_Note__c"] = note
    if owner_id and lead.get("OwnerId") != owner_id:
        changes["OwnerId"] = owner_id
    if account and lead.get("Matched_Account__c") != account["Id"]:
        changes["Matched_Account__c"] = account["Id"]
    if parse_sf_datetime(lead.get("Routing_SLA_Due__c")) != due.replace(microsecond=0):
        changes["Routing_SLA_Due__c"] = sla_due
    return base(rule, owner_id, changes, sla_due)


def route_leads(org: Org, rules: dict, now: datetime, reroute: bool = False) -> list[Proposal]:
    """Propose routing for every lead in the snapshot, oldest first."""
    index = AccountIndex(org, rules)
    state = RunState(load=dict(org.lead_load))
    leads = sorted(org.leads, key=lambda l: l.get("CreatedDate") or "")
    return [propose(lead, org, index, rules, now, state, reroute) for lead in leads]


@dataclass(frozen=True)
class Breach:
    lead_id: str
    lead_name: str
    owner_id: str
    minutes_late: int
    action: str                 # requeue | in_queue
    changes: dict = field(default_factory=dict)


def sla_breaches(org: Org, rules: dict, now: datetime) -> list[Breach]:
    """Unworked leads past their due time. A rep's breach goes back to the queue."""
    out = []
    for lead in org.leads:
        due = parse_sf_datetime(lead.get("Routing_SLA_Due__c"))
        if not due or due >= now or lead.get("Status") not in rules["sla_unworked_statuses"]:
            continue
        late = int((now - due).total_seconds() // 60)
        territory = config.resolve_territory(
            lead.get("Country") or "", lead.get("StateCode") or lead.get("State"))
        queue = org.queues.get(queue_name(territory))
        if is_user(lead.get("OwnerId")) and queue:
            changes = {"OwnerId": queue["Id"], "Routing_Rule__c": "sla_breach_requeued",
                       "Routing_Note__c": f"unworked {late} min past SLA; requeued"[:255]}
            out.append(Breach(lead["Id"], lead.get("Name") or "", lead["OwnerId"], late,
                              "requeue", changes))
        else:
            out.append(Breach(lead["Id"], lead.get("Name") or "", lead.get("OwnerId") or "",
                              late, "in_queue"))
    return out


# --- org I/O ----------------------------------------------------------------

LEAD_FIELDS = (
    "Id, Name, Company, Email, Website, Country, State, StateCode, Status, CreatedDate, "
    "OwnerId, Owner.Name, Matched_Account__c, Routing_Rule__c, Routing_Note__c, "
    "Routing_SLA_Due__c, Consent_Given__c"
)


def load_org(client: SalesforceClient, lead_id: str | None = None) -> Org:
    where = "IsConverted = false"
    if lead_id:
        if not lead_id.isalnum():
            raise SalesforceError(f"not a Salesforce record id: {lead_id!r}")
        where += f" AND Id = '{lead_id}'"
    q = client.query
    queues = q("SELECT Id, Name, DeveloperName FROM Group "
               f"WHERE Type = 'Queue' AND DeveloperName LIKE '{QUEUE_PREFIX}%'")
    members: dict[str, list[str]] = {}
    for row in q("SELECT GroupId, UserOrGroupId FROM GroupMember WHERE Group.Type = 'Queue'"):
        members.setdefault(row["GroupId"], []).append(row["UserOrGroupId"])
    return Org(
        leads=q(f"SELECT {LEAD_FIELDS} FROM Lead WHERE {where} ORDER BY CreatedDate"),
        accounts=q("SELECT Id, Name, Website, OwnerId, Owner.Name, ParentId, Type, "
                   "LastActivityDate, CreatedDate FROM Account"),
        users={u["Id"]: u["Name"] for u in
               q("SELECT Id, Name FROM User WHERE IsActive = true AND UserType = 'Standard'")},
        queues={g["DeveloperName"]: g for g in queues},
        queue_members=members,
        open_opps={r["AccountId"]: r["n"] for r in
                   q("SELECT AccountId, COUNT(Id) n FROM Opportunity "
                     "WHERE IsClosed = false AND AccountId != null GROUP BY AccountId")},
        contacts_by_email={c["Email"].lower(): c["AccountId"] for c in
                           q("SELECT Email, AccountId FROM Contact "
                             "WHERE Email != null AND AccountId != null")},
        lead_load={r["OwnerId"]: r["n"] for r in
                   q("SELECT OwnerId, COUNT(Id) n FROM Lead "
                     "WHERE IsConverted = false GROUP BY OwnerId")},
    )


def route_org(client: SalesforceClient, apply: bool = False, lead_id: str | None = None,
              runs_dir=None, now: datetime | None = None, rules: dict | None = None,
              reroute: bool = False) -> list[Proposal]:
    """Propose (and with apply=True, write) routing for open Leads in the org."""
    rules = rules or load_rules()
    now = now or datetime.now(timezone.utc)
    org = load_org(client, lead_id)
    proposals = route_leads(org, rules, now, reroute)
    for p in proposals:
        audit.append(
            stage="sf_match", lead_id=p.lead_id,
            inputs={"lead": p.lead_name, "accounts_considered": len(org.accounts)},
            decision={"status": p.match_status, "method": p.match_method,
                      "score": p.match_score, "account_id": p.account_id,
                      "candidates": p.candidates, "duplicate_accounts": p.duplicate_accounts},
            rule_fired=p.match_method, runs_dir=runs_dir,
        )
        audit.append(
            stage="sf_route", lead_id=p.lead_id,
            inputs={"territory": p.territory, "match_status": p.match_status, "note": p.note},
            decision={"owner_id": p.new_owner_id, "owner_name": p.new_owner_name,
                      "sla_due": p.sla_due, "changes": p.changes},
            rule_fired=p.rule_fired, fallback_used=p.fallback_used, runs_dir=runs_dir,
        )
        if not apply:
            continue
        if p.changes:
            client.update("Lead", p.lead_id, p.changes)
        audit.append(
            stage="sf_write", lead_id=p.lead_id, inputs={"changes": p.changes},
            decision={"written": bool(p.changes), "org": client.org},
            rule_fired="lead_updated" if p.changes else "no_change", runs_dir=runs_dir,
        )
    return proposals


def check_sla(client: SalesforceClient, apply: bool = False, runs_dir=None,
              now: datetime | None = None, rules: dict | None = None) -> list[Breach]:
    rules = rules or load_rules()
    now = now or datetime.now(timezone.utc)
    breaches = sla_breaches(load_org(client), rules, now)
    for b in breaches:
        written = False
        if apply and b.changes:
            client.update("Lead", b.lead_id, b.changes)
            written = True
        audit.append(
            stage="sf_sla", lead_id=b.lead_id,
            inputs={"owner_id": b.owner_id, "minutes_late": b.minutes_late},
            decision={"action": b.action, "written": written, "changes": b.changes},
            rule_fired="sla_breach", runs_dir=runs_dir,
        )
    return breaches


# --- benchmark (no org needed) ----------------------------------------------

def bench(n_leads: int, n_accounts: int = 2000) -> dict:
    """Time pure matching + routing on generated data. Nothing touches the org."""
    import random
    rng = random.Random(config.RANDOM_SEED)
    words = ["north", "blue", "iron", "delta", "apex", "lumen", "harbor", "cedar", "vertex",
             "pioneer", "summit", "quartz", "atlas", "nova", "ember", "tidal", "orchard"]
    tails = ["systems", "labs", "logistics", "health", "capital", "foods", "media", "energy"]
    accounts = []
    for i in range(n_accounts):
        name = f"{rng.choice(words)}{rng.choice(words)} {rng.choice(tails)} {i}"
        accounts.append({"Id": f"001{i:012d}", "Name": name.title(),
                         "Website": f"{name.replace(' ', '')}.example", "OwnerId": "005A",
                         "ParentId": None, "Type": None, "CreatedDate": "2026-01-01"})
    countries = ["United States", "Germany", "Japan", "France", "Australia"]
    leads = []
    for i in range(n_leads):
        acct = rng.choice(accounts)
        kind = rng.random()
        company = acct["Name"] if kind < 0.6 else f"Unknown Co {i}"
        email = f"p{i}@{domain_of(acct['Website'])}" if kind < 0.3 else f"p{i}@gmail.com"
        leads.append({"Id": f"00Q{i:012d}", "Name": f"Lead {i}", "Company": company,
                      "Email": email, "Country": rng.choice(countries), "StateCode": "NY",
                      "OwnerId": "005A", "CreatedDate": "2026-11-16T12:00:00.000+0000",
                      "Consent_Given__c": True})
    org = Org(leads=leads, accounts=accounts, users={"005A": "Rep A"},
              queues={queue_name(t): {"Id": f"00G{t}", "Name": t, "DeveloperName": queue_name(t)}
                      for t in config.TERRITORIES})
    start = time.perf_counter()
    proposals = route_leads(org, load_rules(), datetime.now(timezone.utc))
    seconds = time.perf_counter() - start
    return {"leads": n_leads, "accounts": n_accounts, "seconds": round(seconds, 2),
            "leads_per_second": round(n_leads / seconds),
            "matched": sum(p.match_status == "matched" for p in proposals)}


# --- CLI --------------------------------------------------------------------

def _print(proposals: list[Proposal], applied: bool) -> None:
    if not proposals:
        print("No open leads found.")
        return
    for p in proposals:
        print(f"{p.lead_name}  [{p.lead_id}]")
        acct = f" -> {p.account_name}" if p.account_name else ""
        print(f"  match   {p.match_status}{acct}")
        print(f"  route   {p.rule_fired}  territory {p.territory}  "
              f"owner {p.new_owner_name or '(unchanged)'}")
        print(f"  note    {p.note}")
        if p.sla_due:
            print(f"  sla     due {p.sla_due}")
        verb = "wrote" if applied else "would write"
        print(f"  {verb:<7} {p.changes or 'nothing'}")
    if not applied:
        print("\nDry run. Nothing was written. Re-run with --apply to write to the org.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--org", default=DEFAULT_ORG, help="sf alias or username")
    parser.add_argument("--lead", help="route one Lead by record id")
    parser.add_argument("--apply", action="store_true", help="write to the org")
    parser.add_argument("--reroute", action="store_true",
                        help="re-decide leads a rep already owns")
    parser.add_argument("--sla", action="store_true", help="list leads past their SLA due time")
    parser.add_argument("--bench", type=int, metavar="N",
                        help="time routing of N generated leads (no org)")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args()
    if args.bench:
        print(json.dumps(bench(args.bench), indent=2))
        return
    client = SalesforceClient(args.org)
    try:
        if args.sla:
            breaches = check_sla(client, apply=args.apply)
            if args.json:
                print(json.dumps([asdict(b) for b in breaches], indent=2))
            elif not breaches:
                print("No leads past their SLA.")
            for b in breaches if not args.json else []:
                print(f"{b.lead_name}  [{b.lead_id}]  {b.minutes_late} min late  {b.action}")
            return
        proposals = route_org(client, apply=args.apply, lead_id=args.lead, reroute=args.reroute)
    except SalesforceError as exc:
        raise SystemExit(f"salesforce: {exc}")
    if args.json:
        print(json.dumps([asdict(p) for p in proposals], indent=2))
    else:
        _print(proposals, applied=args.apply)


if __name__ == "__main__":
    main()
