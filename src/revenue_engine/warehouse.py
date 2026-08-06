"""Mini data warehouse: DuckDB star schema + deterministic synthetic seed.

The generator injects real signal: accounts that go on to WIN an opportunity
show meaningfully more activity (and more distinct engaged contacts) in the
30 days before the opportunity is created than accounts that lose. A later
unit trains a logistic regression on features computed AS OF each
opportunity's created_at, so the signal lives in the pre-creation window.

Rebuild the seed database:

    python -m revenue_engine.warehouse --rebuild
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from revenue_engine.config import (
    EMPLOYEE_BANDS,
    INDUSTRY_FIT,
    RANDOM_SEED,
    SEED_DB,
    resolve_territory,
)

# ---------------------------------------------------------------------------
# Curated fictional universe (no lorem ipsum, no Test Corp)
# ---------------------------------------------------------------------------

# (name, domain, industry, country, state-or-None)
COMPANIES: list[tuple[str, str, str, str, str | None]] = [
    ("Ledgerline", "ledgerline.com", "Fintech", "United States", "NY"),
    ("Parcelbase", "parcelbase.com", "Logistics", "United States", "IL"),
    ("Hivemetric", "hivemetric.io", "SaaS", "United States", "CA"),
    ("Northbeam Labs", "northbeamlabs.com", "SaaS", "United States", "WA"),
    ("Fielddesk", "fielddesk.com", "SaaS", "United States", "TX"),
    ("Quotient Systems", "quotientsystems.com", "Fintech", "United States", "MA"),
    ("Draftwell", "draftwell.io", "SaaS", "United States", "CO"),
    ("Copperstack", "copperstack.com", "Cybersecurity", "United States", "VA"),
    ("Brightquay", "brightquay.com", "Fintech", "United Kingdom", None),
    ("Tallyforge", "tallyforge.com", "Fintech", "United States", "GA"),
    ("Wharfline", "wharfline.co.uk", "Logistics", "United Kingdom", None),
    ("Pinemetric", "pinemetric.com", "SaaS", "United States", "OR"),
    ("Gatewell Health", "gatewellhealth.com", "Healthcare Technology", "United States", "FL"),
    ("Framepost", "framepost.io", "Media", "United States", "CA"),
    ("Lumenride", "lumenride.com", "Transportation", "United States", "AZ"),
    ("Harborlytics", "harborlytics.com", "SaaS", "United States", "NJ"),
    ("Quillstack", "quillstack.com", "SaaS", "Canada", None),
    ("Vantagemill", "vantagemill.com", "Manufacturing", "United States", "OH"),
    ("Ridgeplane", "ridgeplane.io", "Cybersecurity", "United States", "UT"),
    ("Fernbase", "fernbase.com", "Healthcare Technology", "United States", "MN"),
    ("Cobbleworks", "cobbleworks.com", "Retail Technology", "United States", "PA"),
    ("Slateport", "slateport.io", "Logistics", "Netherlands", None),
    ("Marrowtech", "marrowtech.com", "Healthcare Technology", "United States", "NC"),
    ("Ovalgrid", "ovalgrid.com", "Energy", "United States", "TX"),
    ("Signalfoundry", "signalfoundry.com", "SaaS", "United States", "NY"),
    ("Meterhouse", "meterhouse.io", "Energy", "Germany", None),
    ("Crestbound", "crestbound.com", "Fintech", "Australia", None),
    ("Palisade Data", "palisadedata.com", "Cybersecurity", "United States", "MD"),
    ("Torchvale", "torchvale.com", "Manufacturing", "United States", "MI"),
    ("Bramblewick", "bramblewick.co.uk", "Retail Technology", "United Kingdom", None),
    ("Nimbuslane", "nimbuslane.com", "SaaS", "United States", "CA"),
    ("Osprey Freight", "ospreyfreight.com", "Logistics", "United States", "TN"),
    ("Kelpline Marine", "kelplinemarine.com", "Maritime", "Singapore", None),
    ("Sableworks", "sableworks.io", "SaaS", "Japan", None),
    ("Thornfield Analytics", "thornfieldanalytics.com", "SaaS", "Ireland", None),
    ("Juniper Ledger", "juniperledger.com", "Fintech", "United States", "CT"),
    ("Beaconhold", "beaconhold.com", "Insurance Technology", "United States", "WI"),
    ("Graniteway", "graniteway.com", "Construction Technology", "United States", "NH"),
    ("Petalstream", "petalstream.com", "E-commerce", "France", None),
    ("Windrose Systems", "windrosesystems.com", "SaaS", "India", None),
]

FIRST_NAMES = [
    "Ava", "Liam", "Noah", "Emma", "Olivia", "Ethan", "Sofia", "Mason",
    "Isabella", "Lucas", "Mia", "Daniel", "Amelia", "Henry", "Chloe",
    "Marcus", "Nina", "Victor", "Elena", "Owen", "Priya", "Diego",
    "Hannah", "Felix", "Grace", "Rafael", "Ines", "Theo", "Maren", "Kai",
]

LAST_NAMES = [
    "Calloway", "Trent", "Vasquez", "Okonkwo", "Marsh", "Ellery", "Novak",
    "Whitfield", "Iyer", "Donnelly", "Ashcroft", "Beaumont", "Castellanos",
    "Drummond", "Eastman", "Fairbanks", "Grantham", "Holloway", "Ibarra",
    "Jansen", "Kessler", "Lindholm", "Moreau", "Nakamura", "Ostrander",
    "Pemberton", "Quintero", "Rowley", "Sandoval", "Thackeray",
]

TITLES = [
    "VP of Revenue Operations", "Director of Marketing", "Head of Sales",
    "Chief Revenue Officer", "Marketing Operations Manager",
    "Sales Operations Lead", "Director of Demand Generation",
    "VP of Growth", "Revenue Systems Manager", "Head of Business Development",
]

ACTIVITY_TYPES = [
    "email_open", "email_reply", "website_visit", "webinar_attended",
    "demo_request", "content_download", "meeting_held", "pricing_page_view",
]

BASE_DATE = datetime(2026, 1, 15)

DDL = """
CREATE TABLE accounts (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    domain VARCHAR NOT NULL UNIQUE,
    industry VARCHAR NOT NULL,
    employee_band VARCHAR NOT NULL,
    country VARCHAR NOT NULL,
    state VARCHAR,
    territory VARCHAR NOT NULL
);
CREATE TABLE contacts (
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    name VARCHAR NOT NULL,
    email VARCHAR NOT NULL,
    title VARCHAR NOT NULL
);
CREATE TABLE activities (
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    contact_id INTEGER NOT NULL REFERENCES contacts(id),
    type VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL
);
CREATE TABLE opportunities (
    id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    stage VARCHAR NOT NULL,
    amount DOUBLE NOT NULL,
    outcome VARCHAR,
    created_at TIMESTAMP NOT NULL,
    closed_at TIMESTAMP
);
"""


# ---------------------------------------------------------------------------
# Deterministic generator
# ---------------------------------------------------------------------------

def _generate(rng: random.Random):
    accounts = []
    for i, (name, domain, industry, country, state) in enumerate(COMPANIES, start=1):
        accounts.append(
            {
                "id": i,
                "name": name,
                "domain": domain,
                "industry": industry,
                "employee_band": rng.choice(EMPLOYEE_BANDS),
                "country": country,
                "state": state,
                "territory": resolve_territory(country, state),
            }
        )

    # 3 contacts per account -> 120 contacts, unique names via pool pairing.
    name_pairs = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(name_pairs)
    contacts = []
    contact_id = 0
    contacts_by_account: dict[int, list[int]] = {}
    for acct in accounts:
        contacts_by_account[acct["id"]] = []
        for _ in range(3):
            first, last = name_pairs[contact_id]
            contact_id += 1
            contacts.append(
                {
                    "id": contact_id,
                    "account_id": acct["id"],
                    "name": f"{first} {last}",
                    "email": f"{first.lower()}.{last.lower()}@{acct['domain']}",
                    "title": rng.choice(TITLES),
                }
            )
            contacts_by_account[acct["id"]].append(contact_id)

    # 80 opportunities: 30 won, 30 lost, 20 open, two per account.
    outcomes = ["won"] * 30 + ["lost"] * 30 + ["open"] * 20
    rng.shuffle(outcomes)
    opportunities = []
    activities = []
    activity_id = 0

    def add_activity(account_id: int, contact_id: int, ts: datetime):
        nonlocal activity_id
        activity_id += 1
        activities.append(
            {
                "id": activity_id,
                "account_id": account_id,
                "contact_id": contact_id,
                "type": rng.choice(ACTIVITY_TYPES),
                "ts": ts,
            }
        )

    opp_id = 0
    for slot in range(2):
        for acct in accounts:
            opp_id += 1
            outcome = outcomes[opp_id - 1]
            created_at = BASE_DATE + timedelta(
                days=rng.randint(20 + slot * 60, 75 + slot * 60),
                hours=rng.randint(8, 18),
                minutes=rng.randint(0, 59),
            )
            if outcome == "open":
                stage = rng.choice(["Discovery", "Evaluation", "Negotiation"])
                closed_at = None
                outcome_val = None
            else:
                stage = "Closed Won" if outcome == "won" else "Closed Lost"
                closed_at = created_at + timedelta(days=rng.randint(21, 65))
                outcome_val = outcome
            amount = round(rng.uniform(18_000, 240_000) / 500) * 500
            opportunities.append(
                {
                    "id": opp_id,
                    "account_id": acct["id"],
                    "stage": stage,
                    "amount": float(amount),
                    "outcome": outcome_val,
                    "created_at": created_at,
                    "closed_at": closed_at,
                }
            )

            # Inject the pre-creation signal. Distributions deliberately
            # OVERLAP (won and lost share a band of plausible activity) so a
            # trained model lands at a credible holdout AUC rather than the
            # fake-looking perfect separation of a toy dataset.
            roster = contacts_by_account[acct["id"]]
            if outcome == "won":
                n_acts = rng.randint(4, 13)
                engaged = rng.sample(roster, rng.randint(2, 3))
            elif outcome == "open":
                n_acts = rng.randint(3, 9)
                engaged = rng.sample(roster, rng.randint(1, 3))
            else:  # lost: thinner, more single-threaded engagement
                n_acts = rng.randint(1, 8)
                engaged = rng.sample(roster, rng.randint(1, 2))
            for _ in range(n_acts):
                ts = created_at - timedelta(
                    days=rng.uniform(0.5, 29.5), hours=rng.uniform(0, 6)
                )
                add_activity(acct["id"], rng.choice(engaged), ts)

    # Background noise spread across all accounts and the full timeline
    # (also blurs the pre-creation windows, which keeps AUC honest).
    for _ in range(200):
        acct = rng.choice(accounts)
        ts = BASE_DATE + timedelta(days=rng.uniform(-10, 170), hours=rng.uniform(0, 23))
        add_activity(acct["id"], rng.choice(contacts_by_account[acct["id"]]), ts)

    return accounts, contacts, activities, opportunities


def rebuild(db_path: Path | str = SEED_DB) -> Path:
    """Regenerate the seed warehouse from scratch. Fully deterministic."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(RANDOM_SEED)
    accounts, contacts, activities, opportunities = _generate(rng)

    con = duckdb.connect(str(db_path))
    try:
        con.execute(DDL)
        con.executemany(
            "INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (a["id"], a["name"], a["domain"], a["industry"], a["employee_band"],
                 a["country"], a["state"], a["territory"])
                for a in accounts
            ],
        )
        con.executemany(
            "INSERT INTO contacts VALUES (?, ?, ?, ?, ?)",
            [(c["id"], c["account_id"], c["name"], c["email"], c["title"]) for c in contacts],
        )
        con.executemany(
            "INSERT INTO activities VALUES (?, ?, ?, ?, ?)",
            [(x["id"], x["account_id"], x["contact_id"], x["type"], x["ts"]) for x in activities],
        )
        con.executemany(
            "INSERT INTO opportunities VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (o["id"], o["account_id"], o["stage"], o["amount"], o["outcome"],
                 o["created_at"], o["closed_at"])
                for o in opportunities
            ],
        )
    finally:
        con.close()
    return db_path


# ---------------------------------------------------------------------------
# Query helpers (used by the pipeline AND the MCP server)
# ---------------------------------------------------------------------------

def connect(db_path: Path | str = SEED_DB, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=read_only)


def _rows_as_dicts(cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_account_by_domain(con, domain: str) -> dict | None:
    cur = con.execute("SELECT * FROM accounts WHERE lower(domain) = lower(?)", [domain])
    rows = _rows_as_dicts(cur)
    return rows[0] if rows else None


def search_accounts_by_name(con) -> list[dict]:
    """All account ids + names, for fuzzy matching upstream."""
    cur = con.execute("SELECT id, name, domain FROM accounts ORDER BY id")
    return _rows_as_dicts(cur)


def get_account_context(con, account_id: int) -> dict | None:
    """Everything an agent needs to reason about one account."""
    cur = con.execute("SELECT * FROM accounts WHERE id = ?", [account_id])
    accounts = _rows_as_dicts(cur)
    if not accounts:
        return None
    contacts = _rows_as_dicts(
        con.execute("SELECT * FROM contacts WHERE account_id = ? ORDER BY id", [account_id])
    )
    recent_activities = _rows_as_dicts(
        con.execute(
            "SELECT * FROM activities WHERE account_id = ? ORDER BY ts DESC LIMIT 20",
            [account_id],
        )
    )
    open_opportunities = _rows_as_dicts(
        con.execute(
            "SELECT * FROM opportunities WHERE account_id = ? AND outcome IS NULL "
            "ORDER BY created_at DESC",
            [account_id],
        )
    )
    return {
        "account": accounts[0],
        "contacts": contacts,
        "recent_activities": recent_activities,
        "open_opportunities": open_opportunities,
    }


def activity_count_as_of(con, account_id: int, as_of, window_days: int = 30) -> int:
    """Activities for the account in the window ending at as_of (exclusive)."""
    return con.execute(
        "SELECT count(*) FROM activities WHERE account_id = ? "
        "AND ts >= CAST(? AS TIMESTAMP) - to_days(?) AND ts < CAST(? AS TIMESTAMP)",
        [account_id, as_of, window_days, as_of],
    ).fetchone()[0]


def distinct_contacts_as_of(con, account_id: int, as_of, window_days: int = 30) -> int:
    """Distinct engaged contacts for the account in the window ending at as_of."""
    return con.execute(
        "SELECT count(DISTINCT contact_id) FROM activities WHERE account_id = ? "
        "AND ts >= CAST(? AS TIMESTAMP) - to_days(?) AND ts < CAST(? AS TIMESTAMP)",
        [account_id, as_of, window_days, as_of],
    ).fetchone()[0]


def prior_opp_count_as_of(con, account_id: int, as_of) -> int:
    """Opportunities created strictly before as_of for the account."""
    return con.execute(
        "SELECT count(*) FROM opportunities WHERE account_id = ? "
        "AND created_at < CAST(? AS TIMESTAMP)",
        [account_id, as_of],
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Seed warehouse management")
    parser.add_argument("--rebuild", action="store_true", help="regenerate the seed DuckDB")
    args = parser.parse_args()
    if args.rebuild:
        path = rebuild()
        con = connect(path)
        try:
            counts = {
                t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("accounts", "contacts", "activities", "opportunities")
            }
        finally:
            con.close()
        print(f"rebuilt {path}")
        for table, n in counts.items():
            print(f"  {table}: {n} rows")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
