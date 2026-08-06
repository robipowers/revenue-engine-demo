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

# Deliberately wide pools. A small pool crossed with a small pool is the
# classic generated-data tell: with 30x30 and 120 draws, every first name and
# every surname shows up ~4 times and a scanner spots it instantly.
FIRST_NAMES = [
    "Adaeze", "Alejandro", "Amara", "Amelia", "Anders", "Anika", "Arjun", "Ava",
    "Beatriz", "Bianca", "Callum", "Camille", "Cassian", "Chloe", "Clara", "Daniel",
    "Daniela", "Declan", "Diego", "Dmitri", "Elena", "Elias", "Emma", "Esme",
    "Ethan", "Fatima", "Felix", "Freya", "Gabriel", "Giulia", "Grace", "Hana",
    "Hannah", "Henry", "Hendrik", "Imani", "Ines", "Ingrid", "Isabel", "Isabella",
    "Jonas", "Junko", "Kai", "Kaveh", "Keiko", "Lars", "Leila", "Liam",
    "Lucas", "Lucia", "Maren", "Malik", "Marcus", "Marta", "Mason", "Mateo",
    "Mia", "Miriam", "Nadia", "Niamh", "Nina", "Noah", "Noor", "Olamide",
    "Olivia", "Oscar", "Owen", "Paloma", "Priya", "Priyanka", "Rafael", "Rafaela",
    "Rohan", "Rosa", "Sana", "Sebastian", "Sigrid", "Sofia", "Soren", "Tariq",
    "Thandeka", "Theo", "Tobias", "Valentina", "Victor", "Viktor", "Wren",
    "Yasmin", "Yuki", "Zara",
]

LAST_NAMES = [
    "Adeyemi", "Ashcroft", "Ashworth", "Beaumont", "Bergqvist", "Bhatt", "Calderon",
    "Calloway", "Castellanos", "Chandra", "Delacroix", "Donnelly", "Drummond",
    "Eastman", "Ellery", "Eriksen", "Fairbanks", "Fontaine", "Gallagher", "Ghosh",
    "Grantham", "Halvorsen", "Hartmann", "Holloway", "Ibarra", "Ishikawa", "Iyer",
    "Jansen", "Kaur", "Kessler", "Kowalski", "Lindgren", "Lindholm", "Loughlin",
    "Maleki", "Marchetti", "Marsh", "Mbeki", "Moreau", "Nakamura", "Nakashima",
    "Novak", "Okonkwo", "Ostrander", "Oyelaran", "Pemberton", "Pereira", "Petrov",
    "Quintero", "Rasmussen", "Ravelo", "Rowley", "Salinas", "Sandoval", "Sorensen",
    "Suzuki", "Tanaka", "Thackeray", "Thorne", "Trent", "Vasquez", "Villanueva",
    "Walsh", "Weatherby", "Whitfield", "Yamamoto", "Zielinski",
]

TITLES = [
    "VP of Revenue Operations", "Director of Revenue Operations", "Head of RevOps",
    "Revenue Systems Manager", "Sales Operations Lead", "Sales Operations Manager",
    "Director of Sales Operations", "Marketing Operations Manager",
    "Marketing Operations Lead", "Director of Demand Generation",
    "Head of Demand Generation", "Director of Growth Marketing", "VP of Growth",
    "VP of Marketing", "Lifecycle Marketing Manager", "Chief Revenue Officer",
    "VP of Sales", "Regional Sales Director", "Head of Sales",
    "Head of Business Development", "Director of Partnerships",
    "GTM Systems Lead", "Salesforce Administrator", "CRM Platform Owner",
    "Director of Sales Enablement", "Sales Enablement Manager",
    "Director of Customer Success", "Customer Success Manager",
    "Growth Operations Manager", "Business Systems Analyst",
]

ACTIVITY_TYPES = [
    "email_open", "email_reply", "website_visit", "webinar_attended",
    "demo_request", "content_download", "meeting_held", "pricing_page_view",
]

BASE_DATE = datetime(2026, 1, 15)

# The account data/inbound/lead_001.json matches on domain. The demo must show
# a P1 and a P2 side by side, so this one is given live recent engagement.
HOT_BEAT_ACCOUNT = "Meterhouse"

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

    # 3 contacts per account -> 120 contacts. Pairs are chosen so no first
    # name and no surname is reused more than twice across the whole book;
    # a plain shuffled cartesian product repeats each part ~4 times, which is
    # the thing that makes a contact list read as generated.
    name_pairs = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(name_pairs)
    needed = len(COMPANIES) * 3
    first_used: dict[str, int] = {}
    last_used: dict[str, int] = {}
    picked: list[tuple[str, str]] = []
    taken: set[tuple[str, str]] = set()
    for cap in (1, 2, 3):  # widen the cap only if the book cannot be filled
        for pair in name_pairs:
            if len(picked) >= needed:
                break
            if pair in taken:
                continue
            first, last = pair
            if first_used.get(first, 0) < cap and last_used.get(last, 0) < cap:
                picked.append(pair)
                taken.add(pair)
                first_used[first] = first_used.get(first, 0) + 1
                last_used[last] = last_used.get(last, 0) + 1
        if len(picked) >= needed:
            break
    assert len(picked) == needed and len(set(picked)) == needed
    name_pairs = picked
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

    # 80 opportunities: 30 won, 30 lost, 20 open. Spread 1-4 per account
    # rather than exactly 2 -- a constant count makes prior_opportunity_count
    # a dead model feature and every account's win history identical.
    outcomes = ["won"] * 30 + ["lost"] * 30 + ["open"] * 20
    rng.shuffle(outcomes)
    per_account = [1] * 14 + [2] * 16 + [3] * 6 + [4] * 4   # sums to 80
    rng.shuffle(per_account)
    slots = [(acct, slot) for acct, n in zip(accounts, per_account)
             for slot in range(n)]
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
    if True:
        for acct, slot in slots:
            opp_id += 1
            outcome = outcomes[opp_id - 1]
            # Space an account's opportunities far enough apart that their
            # 30-day feature windows do not overlap. Stacking them blurs each
            # opportunity's own pre-creation signal into its neighbours and
            # collapses the model's AUC.
            created_at = BASE_DATE + timedelta(
                days=rng.randint(20 + slot * 36, 56 + slot * 36),
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
            # Priced the way a quote actually is: seats x an annual per-seat
            # rate, with an occasional negotiated discount. A uniform band of
            # round $500 multiples is the loudest "generated data" tell there
            # is, and it makes the Amount column useless for sorting.
            seats = max(5, int(rng.lognormvariate(3.5, 0.85)))
            per_seat = rng.choice([540, 660, 780, 900, 1_080, 1_200, 1_450])
            amount = seats * per_seat
            if rng.random() < 0.35:                      # negotiated down
                amount = int(amount * rng.uniform(0.82, 0.95))
            amount = min(max(amount, 6_000), 940_000)
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

    # Narrative shaping, stated plainly: the demo has to show BOTH SLA tiers,
    # so the account the hot beat matches carries a live surge of engagement in
    # the days before the scoring clock, and the account the messy beat resolves
    # to stays quiet. These accounts are scored by the same model on the same
    # features as every other account -- only their activity history is arranged,
    # exactly as the rest of this fixture is.
    clock = max(o["created_at"] for o in opportunities) + timedelta(hours=1)
    by_name = {a["name"]: a for a in accounts}
    hot = by_name.get(HOT_BEAT_ACCOUNT)
    if hot is not None:
        roster = contacts_by_account[hot["id"]]
        engaged = roster[:2] if len(roster) >= 2 else roster
        for i in range(14):
            ts = clock - timedelta(days=rng.uniform(0.5, 26.0), hours=rng.uniform(0, 20))
            add_activity(hot["id"], engaged[i % len(engaged)], ts)

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


# Frozen dataset => frozen clock. The seed data never moves, so live scoring
# against the real wall clock would decay toward zero as the calendar drifts
# past the last recorded activity. Anchoring "now" to the dataset keeps
# scores meaningful on any demo day. The anchor is the creation moment of
# the newest opportunity (+1h) — the point where the freshest pre-creation
# activity cluster is fully inside the 30-day feature window, so accounts
# with genuinely strong recent signal score high. (max(activities.ts) was
# considered and rejected empirically: trailing background noise pushes that
# anchor past every activity cluster and no account can score above ~0.18.)
_DEMO_AS_OF_CACHE: dict[str, datetime] = {}


def demo_as_of(db_path: Path | str = SEED_DB) -> datetime:
    """The pipeline's deterministic 'now', computed once per database."""
    key = str(Path(db_path).resolve())
    if key not in _DEMO_AS_OF_CACHE:
        con = connect(db_path)
        try:
            latest = con.execute("SELECT max(created_at) FROM opportunities").fetchone()[0]
        finally:
            con.close()
        if latest is None:
            raise ValueError(
                f"warehouse {db_path} has no opportunities, so no demo clock "
                "can be anchored; rebuild with "
                "`python -m revenue_engine.warehouse --rebuild`"
            )
        _DEMO_AS_OF_CACHE[key] = latest + timedelta(hours=1)
    return _DEMO_AS_OF_CACHE[key]


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
