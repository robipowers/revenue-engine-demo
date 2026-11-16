"""Tests for the Salesforce adapter. No network: the sf CLI is faked."""

from datetime import datetime, timezone

import pytest

from revenue_engine import audit, matching, salesforce as sf

ROBI, ANA, BEN, GONE = "005ROBI", "005ANA", "005BEN", "005GONE"
# Monday 2026-11-16 15:00 UTC: inside working hours for AMER and EMEA, outside for APAC.
NOW = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)
CREATED = "2026-11-16T14:00:00.000+0000"

RULES = sf.load_rules("/nonexistent")
RULES["consent"]["regions"] = ["EMEA"]
RULES["working_hours"] = {
    "EMEA": {"tz": "Europe/London", "start": 8, "end": 18},
    "APAC": {"tz": "Asia/Singapore", "start": 8, "end": 18},
}


def acct(id, name, website=None, owner=ROBI, **over):
    base = {"Id": id, "Name": name, "Website": website, "OwnerId": owner,
            "ParentId": None, "Type": None, "LastActivityDate": None,
            "CreatedDate": "2026-01-01T00:00:00.000+0000"}
    base.update(over)
    return base


def lead(**over):
    base = {"Id": "00QLEAD", "Name": "Lena Vogel", "Company": "Hive Metric Co",
            "Email": "lena.vogel@hivemetric.com", "Website": None, "Country": "Germany",
            "State": None, "StateCode": None, "Status": "Open - Not Contacted",
            "CreatedDate": CREATED, "OwnerId": ROBI, "Matched_Account__c": None,
            "Routing_Rule__c": None, "Routing_Note__c": None, "Routing_SLA_Due__c": None,
            "Consent_Given__c": True}
    base.update(over)
    return base


def queues():
    names = ["Relay_AMER_East", "Relay_AMER_West", "Relay_EMEA", "Relay_APAC",
             "Relay_Consent_Hold"]
    return {n: {"Id": f"00G{n}", "Name": n.replace("_", " "), "DeveloperName": n} for n in names}


def org(leads=(), accounts=None, **over):
    accounts = accounts if accounts is not None else [
        acct("001HIVE", "Hivemetric", "hivemetric.com"),
        acct("001PINE", "Pinemetric", "https://www.pinemetric.io/about"),
    ]
    base = dict(leads=list(leads), accounts=accounts,
                users={ROBI: "Robi Powers", ANA: "Ana Ruiz", BEN: "Ben Cole"}, queues=queues())
    base.update(over)
    return sf.Org(**base)


def one(l, o=None, rules=RULES, now=NOW, **kw):
    o = o or org()
    o.leads = [l]
    return sf.route_leads(o, rules, now, **kw)[0]


# --- helpers ---------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("hivemetric.com", "hivemetric.com"),
    ("https://www.Pinemetric.io/about?x=1", "pinemetric.io"),
    ("lena@hivemetric.com", "hivemetric.com"),
    ("", None),
    (None, None),
])
def test_domain_of(raw, expected):
    assert sf.domain_of(raw) == expected


def test_free_mail_domain_is_ignored_and_website_used():
    assert sf.lead_domain(lead(Email="l@gmail.com", Website="www.hivemetric.com")) == "hivemetric.com"
    assert sf.lead_domain(lead(Email="l@gmail.com")) is None


def test_match_records_agrees_with_policy_bands():
    records = [{"id": "a", "name": "Hivemetric", "domain": "hivemetric.com"},
               {"id": "b", "name": "Pinemetric", "domain": "pinemetric.io"}]
    assert matching.match_records(records, name="x", domain="HIVEMETRIC.com").method == "domain_exact"
    assert matching.match_records(records, name="Hine Metric Co").status == "ambiguous"
    assert matching.match_records([], name="Anything").status == "no_match"


def test_rules_file_overrides_defaults(tmp_path):
    path = tmp_path / "routing.json"
    path.write_text('{"_readme": "x", "round_robin": {"cap_open_leads": 3}}')
    rules = sf.load_rules(path)
    assert rules["round_robin"]["cap_open_leads"] == 3
    assert rules["sla_minutes"]["hot"] == 15          # untouched default
    assert "_readme" not in rules


def test_shipped_rules_file_loads():
    rules = sf.load_rules()
    assert rules["consent"]["regions"] == ["EMEA"]
    assert set(rules["tiebreakers"]) <= {"open_opportunity", "customer", "latest_activity",
                                         "oldest_account"}


# --- matching --------------------------------------------------------------

def test_domain_match_routes_to_account_owner():
    p = one(lead())
    assert (p.match_status, p.match_method, p.match_score) == ("matched", "domain_exact", 100.0)
    assert p.account_id == "001HIVE" and p.rule_fired == "account_owner"
    assert p.new_owner_id == ROBI and "OwnerId" not in p.changes
    assert p.changes["Matched_Account__c"] == "001HIVE"


def test_existing_contact_email_beats_everything():
    o = org(contacts_by_email={"lena@gmail.com": "001PINE"})
    p = one(lead(Email="Lena@Gmail.com", Company="Totally Different"), o)
    assert p.match_method == "contact_email" and p.account_id == "001PINE"


def test_two_different_companies_tied_are_never_guessed():
    p = one(lead(Company="Hine Metric Co", Email="k@gmail.com", Country="Japan"))
    assert p.match_status == "ambiguous" and p.account_id is None
    assert p.rule_fired == "ambiguous_match_to_queue" and p.new_owner_id == "00GRelay_APAC"
    assert "Hivemetric" in p.note and "Pinemetric" in p.note


def test_duplicate_accounts_on_one_domain_use_open_opportunity_tiebreak():
    accounts = [acct("001A", "Hivemetric", "hivemetric.com", owner=ANA),
                acct("001B", "Hivemetric Inc", "www.hivemetric.com", owner=BEN)]
    p = one(lead(), org(accounts=accounts, open_opps={"001B": 2}))
    assert p.account_id == "001B" and p.new_owner_id == BEN
    assert p.duplicate_accounts == ["001A", "001B"]
    assert "2 duplicate accounts, tiebreak open_opportunity" in p.note


def test_duplicate_tiebreak_falls_through_to_customer_then_activity_then_age():
    a = acct("001A", "Hivemetric", "hivemetric.com")
    b = acct("001B", "Hivemetric", "hivemetric.com", Type="Customer - Direct")
    idx = sf.AccountIndex(org(accounts=[a, b]), RULES)
    assert sf.tiebreak([a, b], idx, RULES) == (b, "customer")
    c = acct("001C", "Hivemetric", "hivemetric.com", LastActivityDate="2026-11-01")
    assert sf.tiebreak([a, c], idx, RULES) == (c, "latest_activity")
    d = acct("001D", "Hivemetric", "hivemetric.com", CreatedDate="2020-01-01T00:00:00.000+0000")
    assert sf.tiebreak([a, d], idx, RULES) == (d, "oldest_account")
    assert sf.tiebreak([a, dict(a, Id="001Z")], idx, RULES)[1] == "lowest_id"


def test_same_name_duplicates_without_domain_are_tiebroken_not_ambiguous():
    accounts = [acct("001A", "Northwind Traders"), acct("001B", "Northwind Traders, Inc.")]
    p = one(lead(Company="Northwind Traders", Email="x@gmail.com"),
            org(accounts=accounts, open_opps={"001A": 1}))
    assert p.match_status == "matched" and p.account_id == "001A"
    assert len(p.duplicate_accounts) == 2


def test_no_match_goes_to_territory_queue():
    p = one(lead(Company="Zzyzx Aerospace", Email="a@zzyzx.aero"))
    assert p.match_status == "no_match" and p.rule_fired == "territory_queue"
    assert p.new_owner_id == "00GRelay_EMEA" and "Matched_Account__c" not in p.changes


def test_us_state_picks_east_or_west_queue():
    kw = dict(Company="Zzyzx", Email=None, Country="United States")
    assert one(lead(StateCode="NY", **kw)).new_owner_id == "00GRelay_AMER_East"
    assert one(lead(StateCode="CA", **kw)).new_owner_id == "00GRelay_AMER_West"


def test_unknown_country_uses_fallback_territory_and_says_so():
    p = one(lead(Company="Zzyzx", Email=None, Country="Atlantis"))
    assert p.rule_fired == "fallback_territory_queue" and p.fallback_used


# --- owner resolution ------------------------------------------------------

def test_inactive_owner_climbs_to_parent_account_owner():
    accounts = [acct("001PARENT", "Globex Holdings", "globex.com", owner=ANA),
                acct("001CHILD", "Globex Germany", "globex.de", owner=GONE, ParentId="001PARENT")]
    p = one(lead(Company="Globex Germany", Email="h@globex.de"), org(accounts=accounts))
    assert p.account_id == "001CHILD"                 # matched record stays the subsidiary
    assert p.rule_fired == "parent_account_owner" and p.new_owner_id == ANA


def test_ultimate_parent_mode_routes_to_top_of_hierarchy():
    rules = sf.load_rules("/nonexistent")
    rules["hierarchy"]["mode"] = "ultimate_parent"
    accounts = [acct("001TOP", "Globex Holdings", "globex.com", owner=ANA),
                acct("001MID", "Globex Europe", "globex.eu", owner=BEN, ParentId="001TOP"),
                acct("001LOW", "Globex Germany", "globex.de", owner=ROBI, ParentId="001MID")]
    p = one(lead(Email="h@globex.de"), org(accounts=accounts), rules=rules)
    assert p.rule_fired == "ultimate_parent_owner" and p.new_owner_id == ANA


def test_parent_cycle_does_not_loop_forever():
    accounts = [acct("001A", "Loop A", "loop.com", owner=GONE, ParentId="001B"),
                acct("001B", "Loop B", "loopb.com", owner=GONE, ParentId="001A")]
    p = one(lead(Email="x@loop.com"), org(accounts=accounts))
    assert p.rule_fired == "account_owner_inactive_to_queue" and p.fallback_used


def test_customer_account_is_labelled_and_gets_hot_sla():
    accounts = [acct("001HIVE", "Hivemetric", "hivemetric.com", Type="Customer - Direct")]
    p = one(lead(), org(accounts=accounts))
    assert p.rule_fired == "customer_account_owner"
    assert p.sla_due == "2026-11-16T14:15:00.000Z"    # created 14:00 + 15 min


# --- consent, duplicate leads ----------------------------------------------

def test_emea_lead_without_consent_is_held():
    p = one(lead(Consent_Given__c=False))
    assert p.rule_fired == "consent_hold" and p.new_owner_id == "00GRelay_Consent_Hold"
    assert p.account_id == "001HIVE"                  # still matched, just not workable
    assert "no consent on record for EMEA" in p.note


def test_consent_rule_only_applies_to_configured_regions():
    p = one(lead(Consent_Given__c=False, Country="United States", StateCode="NY"))
    assert p.rule_fired == "account_owner"


def test_duplicate_lead_follows_the_first_leads_owner():
    o = org(queue_members={"00GRelay_EMEA": [ANA, BEN]})
    first = lead(Id="00Q1", Company="Zzyzx", Email="sam@zzyzx.aero", CreatedDate=CREATED)
    again = lead(Id="00Q2", Company="Zzyzx GmbH", Email="SAM@zzyzx.aero",
                 CreatedDate="2026-11-16T14:30:00.000+0000")
    o.leads = [again, first]                          # order in must not matter
    by_id = {p.lead_id: p for p in sf.route_leads(o, RULES, NOW)}
    assert by_id["00Q1"].rule_fired == "round_robin_least_loaded"
    assert by_id["00Q2"].rule_fired == "duplicate_lead_same_owner"
    assert by_id["00Q2"].new_owner_id == by_id["00Q1"].new_owner_id


# --- round robin -----------------------------------------------------------

def unmatched(i, **over):
    return lead(Id=f"00Q{i}", Name=f"Lead {i}", Company=f"Unknown {i}", Email=f"p{i}@u{i}.example",
                **over)


def test_round_robin_spreads_a_batch_by_load():
    o = org([unmatched(i) for i in range(4)], queue_members={"00GRelay_EMEA": [ANA, BEN]},
            lead_load={ANA: 2})
    owners = [p.new_owner_id for p in sf.route_leads(o, RULES, NOW)]
    assert owners == [BEN, BEN, ANA, BEN]             # Ben catches up to Ana first
    assert all(p.rule_fired == "round_robin_least_loaded" for p in sf.route_leads(o, RULES, NOW))


def test_round_robin_skips_inactive_members_and_respects_cap():
    rules = sf.load_rules("/nonexistent")
    rules["round_robin"]["cap_open_leads"] = 1
    o = org([unmatched(i) for i in range(3)], queue_members={"00GRelay_EMEA": [GONE, ANA]})
    props = sf.route_leads(o, rules, NOW)
    assert props[0].new_owner_id == ANA
    assert props[1].new_owner_id == "00GRelay_EMEA" and "all reps at cap" in props[1].note


def test_outside_working_hours_lead_waits_in_queue():
    o = org(queue_members={"00GRelay_APAC": [ANA]})
    p = one(unmatched(1, Country="Japan"), o)         # 15:00 UTC is 23:00 in Singapore
    assert p.new_owner_id == "00GRelay_APAC" and "outside working hours" in p.note


def test_queue_without_members_keeps_the_lead():
    p = one(unmatched(1))
    assert p.new_owner_id == "00GRelay_EMEA" and "no active members" in p.note


def test_missing_queue_changes_no_owner():
    p = one(unmatched(1), org(queues={}))
    assert p.rule_fired == "no_queue_configured" and p.fallback_used
    assert p.new_owner_id is None and "OwnerId" not in p.changes


# --- idempotency -----------------------------------------------------------

def test_lead_a_rep_already_owns_is_not_reshuffled():
    settled = lead(OwnerId=ANA, Routing_Rule__c="round_robin_least_loaded")
    p = one(settled, org(queue_members={"00GRelay_EMEA": [ANA, BEN]}))
    assert p.rule_fired == "already_routed" and p.changes == {}
    assert one(settled, reroute=True).rule_fired == "account_owner"


def test_second_pass_proposes_no_changes():
    l = unmatched(1)
    first = one(l)
    routed = dict(l, **first.changes)
    assert one(routed).changes == {}


# --- SLA -------------------------------------------------------------------

def test_sla_due_depends_on_how_hot_the_lead_is():
    assert one(unmatched(1)).sla_due == "2026-11-16T18:00:00.000Z"       # default 240
    assert one(lead()).sla_due == "2026-11-16T15:00:00.000Z"             # matched 60
    assert one(lead(), org(open_opps={"001HIVE": 1})).sla_due == "2026-11-16T14:15:00.000Z"


def test_sla_breach_requeues_a_reps_lead_and_reports_queue_leads():
    late = "2026-11-16T14:00:00.000+0000"
    o = org([
        lead(Id="00Q1", OwnerId=ANA, Routing_SLA_Due__c=late),
        lead(Id="00Q2", OwnerId="00GRelay_EMEA", Routing_SLA_Due__c=late),
        lead(Id="00Q3", OwnerId=ANA, Routing_SLA_Due__c=late, Status="Working - Contacted"),
        lead(Id="00Q4", OwnerId=ANA, Routing_SLA_Due__c="2026-11-16T16:00:00.000+0000"),
    ])
    breaches = {b.lead_id: b for b in sf.sla_breaches(o, RULES, NOW)}
    assert set(breaches) == {"00Q1", "00Q2"}
    assert breaches["00Q1"].action == "requeue" and breaches["00Q1"].minutes_late == 60
    assert breaches["00Q1"].changes["OwnerId"] == "00GRelay_EMEA"
    assert breaches["00Q2"].action == "in_queue" and breaches["00Q2"].changes == {}


# --- org I/O ---------------------------------------------------------------

class FakeClient:
    org = "fake-org"

    def __init__(self, leads):
        self.leads, self.updates = leads, []

    def query(self, soql):
        if "COUNT(Id)" in soql:
            return []
        if "FROM Lead" in soql:
            return self.leads
        if "FROM Account" in soql:
            return [acct("001HIVE", "Hivemetric", "hivemetric.com")]
        if "FROM User" in soql:
            return [{"Id": ROBI, "Name": "Robi Powers"}]
        if "FROM GroupMember" in soql or "FROM Contact" in soql:
            return []
        if "FROM Group" in soql:
            return list(queues().values())
        raise AssertionError(soql)

    def update(self, sobject, record_id, values):
        self.updates.append((sobject, record_id, values))


def test_dry_run_writes_nothing_but_audits(tmp_path):
    client = FakeClient([lead()])
    sf.route_org(client, apply=False, runs_dir=tmp_path, now=NOW, rules=RULES)
    assert client.updates == []
    assert [r["stage"] for r in audit.read_tail(10, runs_dir=tmp_path)] == ["sf_match", "sf_route"]


def test_apply_writes_changes_then_is_idempotent(tmp_path):
    l = lead()
    client = FakeClient([l])
    sf.route_org(client, apply=True, runs_dir=tmp_path, now=NOW, rules=RULES)
    (sobject, record_id, values), = client.updates
    assert (sobject, record_id) == ("Lead", "00QLEAD")
    assert values["Routing_Rule__c"] == "account_owner" and values["Matched_Account__c"] == "001HIVE"
    assert audit.read_tail(1, runs_dir=tmp_path)[0]["rule_fired"] == "lead_updated"
    client.leads = [dict(l, **values)]
    client.updates = []
    sf.route_org(client, apply=True, runs_dir=tmp_path, now=NOW, rules=RULES)
    assert client.updates == []
    assert audit.read_tail(1, runs_dir=tmp_path)[0]["rule_fired"] == "no_change"


def test_check_sla_applies_requeue_and_audits(tmp_path):
    client = FakeClient([lead(OwnerId=ROBI, Routing_SLA_Due__c="2026-11-16T14:00:00.000+0000")])
    breaches = sf.check_sla(client, apply=True, runs_dir=tmp_path, now=NOW, rules=RULES)
    assert len(breaches) == 1 and client.updates[0][2]["Routing_Rule__c"] == "sla_breach_requeued"
    assert audit.read_tail(1, runs_dir=tmp_path)[0]["stage"] == "sf_sla"


def test_client_builds_update_command_with_quoted_values():
    calls = []
    client = sf.SalesforceClient("my-org", runner=lambda args: calls.append(args) or {})
    client.update("Lead", "00Q1", {"OwnerId": "00G1", "Routing_Note__c": "o'brien; x",
                                   "Consent_Given__c": True})
    args = calls[0]
    assert args[:3] == ["data", "update", "record"]
    assert args[args.index("--values") + 1] == (
        "OwnerId='00G1' Routing_Note__c='o\\'brien; x' Consent_Given__c=true")


def test_load_org_rejects_a_bad_lead_id():
    with pytest.raises(sf.SalesforceError):
        sf.load_org(FakeClient([]), lead_id="x' OR Name != '")


def test_missing_cli_raises_clear_error(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(sf.subprocess, "run", boom)
    with pytest.raises(sf.SalesforceError, match="not installed"):
        sf._run_sf(["data", "query"])


def test_bench_routes_a_generated_batch_without_an_org():
    result = sf.bench(300, n_accounts=200)
    assert result["leads"] == 300 and result["matched"] > 100
