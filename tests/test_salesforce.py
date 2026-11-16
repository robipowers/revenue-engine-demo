"""Tests for the Salesforce adapter. No network: the sf CLI is faked."""

import pytest

from revenue_engine import audit, matching, salesforce as sf

ROBI = "005ROBI"
GONE = "005GONE"

ACCOUNTS = [
    {"Id": "001HIVE", "Name": "Hivemetric", "Website": "hivemetric.com",
     "OwnerId": ROBI, "Owner": {"Name": "Robi Powers"}},
    {"Id": "001PINE", "Name": "Pinemetric", "Website": "https://www.pinemetric.io/about",
     "OwnerId": ROBI, "Owner": {"Name": "Robi Powers"}},
    {"Id": "001OLD", "Name": "Oldcastle Freight", "Website": "oldcastlefreight.com",
     "OwnerId": GONE, "Owner": {"Name": "Former Rep"}},
]
QUEUES = {
    name: {"Id": f"00G{name}", "Name": name.replace("_", " "), "DeveloperName": name}
    for name in ("Relay_AMER_East", "Relay_AMER_West", "Relay_EMEA", "Relay_APAC")
}
ACTIVE = {ROBI}


def lead(**over):
    base = {"Id": "00QLEAD", "Name": "Lena Vogel", "Company": "Hive Metric Co",
            "Email": "lena.vogel@hivemetric.com", "Website": None, "Country": "Germany",
            "State": None, "StateCode": None, "OwnerId": ROBI,
            "Matched_Account__c": None, "Routing_Rule__c": None}
    base.update(over)
    return base


class FakeClient:
    org = "fake-org"

    def __init__(self, leads):
        self.leads = leads
        self.updates = []

    def query(self, soql):
        if "FROM Lead" in soql:
            return self.leads
        if "FROM Account" in soql:
            return ACCOUNTS
        if "FROM User" in soql:
            return [{"Id": ROBI}]
        if "FROM Group" in soql:
            return list(QUEUES.values())
        raise AssertionError(soql)

    def update(self, sobject, record_id, values):
        self.updates.append((sobject, record_id, values))


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
    assert sf.lead_domain(lead(Email="lena@gmail.com", Website="www.hivemetric.com")) == "hivemetric.com"
    assert sf.lead_domain(lead(Email="lena@gmail.com")) is None


def test_match_records_agrees_with_policy_bands():
    records = [{"id": "a", "name": "Hivemetric", "domain": "hivemetric.com"},
               {"id": "b", "name": "Pinemetric", "domain": "pinemetric.io"}]
    assert matching.match_records(records, name="x", domain="HIVEMETRIC.com").method == "domain_exact"
    assert matching.match_records(records, name="Hivemetric Inc.").status == "matched"
    amb = matching.match_records(records, name="Hine Metric Co")
    assert amb.status == "ambiguous" and len(amb.candidates) == 2
    assert matching.match_records(records, name="Zzyzx Aerospace").status == "no_match"
    assert matching.match_records([], name="Anything").status == "no_match"


# --- propose ---------------------------------------------------------------

def test_domain_match_routes_to_account_owner():
    p = sf.propose(lead(), ACCOUNTS, ACTIVE, QUEUES)
    assert (p.match_status, p.match_method, p.match_score) == ("matched", "domain_exact", 100.0)
    assert p.account_id == "001HIVE"
    assert p.rule_fired == "account_owner" and p.new_owner_id == ROBI
    # Owner is already Robi, so only the two routing fields change.
    assert p.changes == {"Routing_Rule__c": "account_owner", "Matched_Account__c": "001HIVE"}


def test_no_match_goes_to_territory_queue():
    p = sf.propose(lead(Company="Zzyzx Aerospace", Email="a@zzyzx.aero"), ACCOUNTS, ACTIVE, QUEUES)
    assert p.match_status == "no_match" and p.territory == "EMEA"
    assert p.rule_fired == "territory_queue" and p.new_owner_id == "00GRelay_EMEA"
    assert "Matched_Account__c" not in p.changes


def test_us_state_picks_east_or_west_queue():
    east = sf.propose(lead(Company="Zzyzx", Email=None, Country="United States", StateCode="NY"),
                      ACCOUNTS, ACTIVE, QUEUES)
    west = sf.propose(lead(Company="Zzyzx", Email=None, Country="United States", StateCode="CA"),
                      ACCOUNTS, ACTIVE, QUEUES)
    assert east.new_owner_id == "00GRelay_AMER_East"
    assert west.new_owner_id == "00GRelay_AMER_West"


def test_ambiguous_match_is_never_guessed():
    p = sf.propose(lead(Company="Hine Metric Co", Email="lena@gmail.com"), ACCOUNTS, ACTIVE, QUEUES)
    assert p.match_status == "ambiguous" and p.account_id is None
    assert p.rule_fired == "ambiguous_match_to_queue"
    assert p.new_owner_id == "00GRelay_EMEA"
    assert {c["name"] for c in p.candidates} >= {"Hivemetric", "Pinemetric"}


def test_inactive_account_owner_falls_back_to_queue():
    p = sf.propose(lead(Company="Oldcastle Freight", Email="x@oldcastlefreight.com"),
                   ACCOUNTS, ACTIVE, QUEUES)
    assert p.account_id == "001OLD"
    assert p.rule_fired == "account_owner_inactive_to_queue" and p.fallback_used
    assert p.new_owner_id == "00GRelay_EMEA"


def test_unknown_country_uses_fallback_territory_and_says_so():
    p = sf.propose(lead(Company="Zzyzx", Email=None, Country="Atlantis"), ACCOUNTS, ACTIVE, QUEUES)
    assert p.rule_fired == "fallback_territory_queue" and p.fallback_used


def test_missing_queue_changes_no_owner():
    p = sf.propose(lead(Company="Zzyzx", Email=None), ACCOUNTS, ACTIVE, queues={})
    assert p.rule_fired == "no_queue_configured" and p.fallback_used
    assert p.new_owner_id is None and "OwnerId" not in p.changes


def test_already_routed_lead_has_no_changes():
    routed = lead(Matched_Account__c="001HIVE", Routing_Rule__c="account_owner")
    assert sf.propose(routed, ACCOUNTS, ACTIVE, QUEUES).changes == {}


# --- route_org -------------------------------------------------------------

def test_dry_run_writes_nothing_but_audits(tmp_path):
    client = FakeClient([lead()])
    proposals = sf.route_org(client, apply=False, runs_dir=tmp_path)
    assert len(proposals) == 1 and client.updates == []
    stages = [r["stage"] for r in audit.read_tail(10, runs_dir=tmp_path)]
    assert stages == ["sf_match", "sf_route"]


def test_apply_writes_changes_and_audits(tmp_path):
    client = FakeClient([lead(Company="Zzyzx", Email=None)])
    sf.route_org(client, apply=True, runs_dir=tmp_path)
    assert client.updates == [("Lead", "00QLEAD", {
        "Routing_Rule__c": "territory_queue", "OwnerId": "00GRelay_EMEA"})]
    last = audit.read_tail(1, runs_dir=tmp_path)[0]
    assert last["stage"] == "sf_write" and last["decision"]["written"] is True


def test_apply_is_idempotent(tmp_path):
    routed = lead(Matched_Account__c="001HIVE", Routing_Rule__c="account_owner")
    client = FakeClient([routed])
    sf.route_org(client, apply=True, runs_dir=tmp_path)
    assert client.updates == []
    assert audit.read_tail(1, runs_dir=tmp_path)[0]["rule_fired"] == "no_change"


# --- client ----------------------------------------------------------------

def test_client_builds_update_command_with_quoted_values():
    calls = []
    client = sf.SalesforceClient("my-org", runner=lambda args: calls.append(args) or {})
    client.update("Lead", "00Q1", {"OwnerId": "00G1", "Routing_Rule__c": "o'brien rule"})
    args = calls[0]
    assert args[:3] == ["data", "update", "record"]
    assert args[args.index("--target-org") + 1] == "my-org"
    assert args[args.index("--values") + 1] == "OwnerId='00G1' Routing_Rule__c='o\\'brien rule'"


def test_client_query_returns_records():
    client = sf.SalesforceClient("o", runner=lambda args: {"records": [{"Id": "1"}]})
    assert client.query("SELECT Id FROM Lead") == [{"Id": "1"}]


def test_load_org_rejects_a_bad_lead_id():
    with pytest.raises(sf.SalesforceError):
        sf.load_org(FakeClient([]), lead_id="x' OR Name != '")


def test_missing_cli_raises_clear_error(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(sf.subprocess, "run", boom)
    with pytest.raises(sf.SalesforceError, match="not installed"):
        sf._run_sf(["data", "query"])
