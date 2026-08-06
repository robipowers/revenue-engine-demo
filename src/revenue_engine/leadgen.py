"""Deterministic generator for a book of inbound demo leads.

The three hand-authored leads (L-1001..L-1003) are the interview beats: one
clean domain match that routes around a rep on vacation, one messy name that
goes ambiguous and falls through to the tie-break, one that matches nothing and
escalates. They are teaching examples and this module never touches them.

What this adds is volume, so the routing surfaces have something real to show:
eight reps instead of two with any load, four territories that all light up, and
an audit log whose rows differ from each other.

Honesty contract: this fabricates INPUTS only — the same thing
`warehouse.rebuild` already does for accounts, contacts and activities. Every
decision, score, assignment and audit record still comes from running the real
pipeline over these leads. Nothing downstream of the engine is written by hand.

Each candidate is checked against the REAL matcher before being kept, so the
advertised path mix (exact domain / fuzzy auto / ambiguous / no match) is a
measured property of the output rather than an intention.

    python -m revenue_engine.leadgen --generate 40
    python -m revenue_engine.leadgen --generate 40 --run
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

from revenue_engine import config, matching, warehouse

INBOUND_DIR = Path(__file__).resolve().parents[2] / "data" / "inbound"

# The hand-authored beats. Never regenerated, never deleted.
RESERVED_STEMS = {"lead_001", "lead_002", "lead_003"}
GENERATED_PREFIX = "lead_g"

# Target share of each matching path. Verified against the real matcher, so a
# band that cannot be filled shrinks rather than silently mislabelling a lead.
PATH_MIX = [
    ("domain_exact", 0.40),   # lead carries the account's real domain
    ("fuzzy_auto", 0.25),     # name mangled but still >= 90
    ("ambiguous", 0.15),      # 70-89, goes to the tie-break step
    ("no_match", 0.20),       # < 70, creates an account and escalates
]

FIRST_NAMES = [
    "Adaeze", "Alejandro", "Amara", "Anders", "Anika", "Beatriz", "Bianca", "Callum",
    "Camille", "Cassian", "Daniela", "Declan", "Dmitri", "Elena", "Elias", "Esme",
    "Fatima", "Felix", "Freya", "Gabriel", "Giulia", "Hana", "Hendrik", "Imani",
    "Ingrid", "Isabel", "Jonas", "Junko", "Kaveh", "Keiko", "Lars", "Leila",
    "Lucia", "Malik", "Marta", "Mateo", "Miriam", "Nadia", "Niamh", "Noor",
    "Olamide", "Oscar", "Paloma", "Priyanka", "Rafaela", "Rohan", "Rosa", "Sana",
    "Sebastian", "Sigrid", "Soren", "Tariq", "Thandeka", "Tobias", "Valentina",
    "Viktor", "Wren", "Yasmin", "Yuki", "Zara",
]
LAST_NAMES = [
    "Adeyemi", "Ashworth", "Bergqvist", "Bhatt", "Calderon", "Castellanos", "Chandra",
    "Delacroix", "Eriksen", "Fontaine", "Gallagher", "Ghosh", "Halvorsen", "Hartmann",
    "Ibarra", "Ishikawa", "Kaur", "Kowalski", "Lindgren", "Loughlin", "Maleki",
    "Marchetti", "Mbeki", "Moreau", "Nakashima", "Novak", "Okonkwo", "Oyelaran",
    "Pereira", "Petrov", "Quintero", "Rasmussen", "Ravelo", "Salinas", "Sandoval",
    "Sorensen", "Suzuki", "Tanaka", "Thorne", "Vasquez", "Villanueva", "Walsh",
    "Weatherby", "Yamamoto", "Zielinski",
]
TITLES = [
    "VP of Revenue Operations", "Director of Revenue Operations", "Head of RevOps",
    "Sales Operations Manager", "Director of Sales Operations", "Marketing Operations Lead",
    "Head of Demand Generation", "Director of Growth Marketing", "Chief Revenue Officer",
    "VP of Sales", "Regional Sales Director", "Head of Business Development",
    "GTM Systems Lead", "Salesforce Administrator", "Director of Sales Enablement",
    "Head of Partnerships", "Growth Operations Manager", "Director of Customer Success",
    "VP of Marketing", "Lifecycle Marketing Manager",
]

# Notes are written to land on a specific keyword-rule intent, so the intent
# column varies instead of reading general_interest for everything.
NOTE_TEMPLATES: dict[str, list[str]] = {
    "pricing_inquiry": [
        "Asked for pricing for a team of {n}. Budget is approved for this quarter.",
        "Wants a quote before the board meeting. Needs cost per seat at {n} users.",
        "hey - what does this cost for {n} seats? budget already signed off",
        "Requested pricing tiers and whether the {n}-seat band has annual discounting.",
    ],
    "demo_request": [
        "Attended the webinar and wants a demo for the ops team of {n}.",
        "Asked for a walkthrough with the RevOps team. Timeline is this quarter.",
        "wants a demo asap -- team of {n}, looking to move before end of quarter",
        "Booked time for a technical demo. Wants routing and matching covered.",
    ],
    "evaluation": [
        "Re-evaluating vendors this quarter. Asked about migration from their current tool.",
        "Running a POC against two other vendors. Needs integration detail for Salesforce.",
        "comparing us against incumbents,, needs integration + migration answers b4 EOQ",
        "Formal evaluation underway. Asked who handles migration and how long it takes.",
    ],
    "support": [
        "Existing routing rules are broken after their Salesforce release. Needs help.",
        "Reported an issue with duplicate leads. Asked for support contact.",
        "having a problem with lead assignment -- some reps not getting anything",
    ],
    "general_interest": [
        "Inbound from the pricing page. Early-stage, wants a technical overview call.",
        "Downloaded the routing guide. No timeline given yet.",
        "Signed up for the newsletter and asked to be contacted next quarter.",
    ],
}

# Company names for the no-match path: plausible, and unlike anything seeded.
NO_MATCH_COMPANIES = [
    "Aldergrove Freight", "Basalt Bioworks", "Cindermill Foods", "Drakemoor Legal",
    "Everline Transit", "Foxglove Textiles", "Grantham Dairy", "Halloway Ceramics",
    "Ironvale Mining", "Juniper Rail", "Kestrelbank Trust", "Larkfield Timber",
    "Marlowe Shipping", "Northgate Foundry", "Oakhurst Vineyards", "Pellworth Paper",
    "Quarrybrook Stone", "Redhaven Fisheries", "Stonewell Brewing", "Thistledown Wool",
]

_MANGLE_SUFFIXES = ["Inc", "LLC", "Ltd", "Co", "Corp", "Group", "Holdings"]


def _mangle_light(name: str, rng: random.Random) -> str:
    """Small, realistic sloppiness that should still clear the 90 auto-match bar."""
    choice = rng.randrange(4)
    if choice == 0:
        return f"{name} {rng.choice(_MANGLE_SUFFIXES)}"
    if choice == 1:
        return name.upper()
    if choice == 2:
        return name.lower()
    return f"{name}."


def _mangle_heavy(name: str, rng: random.Random) -> str:
    """Enough damage to fall out of auto-match and into the ambiguous band."""
    choice = rng.randrange(4)
    if choice == 0:  # split a compound name, as a human typing it would
        mid = max(3, len(name) // 2)
        return f"{name[:mid]} {name[mid:]} {rng.choice(['Co', 'Inc', 'Group'])}"
    if choice == 1:  # drop the tail
        return name[: max(4, len(name) - 3)]
    if choice == 2:  # transpose two interior characters
        i = rng.randrange(1, max(2, len(name) - 2))
        return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    return f"{name[: max(4, len(name) - 2)]} {rng.choice(['Systems', 'Partners', 'Labs'])}"


def _email(first: str, last: str, domain: str, rng: random.Random) -> str:
    pattern = rng.randrange(3)
    local = {
        0: f"{first}.{last}",
        1: f"{first[0]}{last}",
        2: first,
    }[pattern].lower()
    return f"{local}@{domain}"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())[:18] or "example"


def _classify(con, company: str, domain: str | None) -> str:
    """Ask the REAL matcher which path this lead would take."""
    result = matching.match_account(con, name=company, domain=domain)
    if result.status == "matched":
        return "domain_exact" if result.method == "domain_exact" else "fuzzy_auto"
    if result.status == "ambiguous":
        return "ambiguous"
    return "no_match"


def generate(count: int = 40, seed: int | None = None) -> list[dict]:
    """Build `count` leads whose matching paths are verified, not assumed."""
    rng = random.Random(config.RANDOM_SEED if seed is None else seed)
    con = warehouse.connect()
    try:
        accounts = [
            {"id": r[0], "name": r[1], "domain": r[2], "country": r[3], "state": r[4]}
            for r in con.execute(
                "SELECT id, name, domain, country, state FROM accounts ORDER BY id"
            ).fetchall()
        ]
        quota = {path: max(1, round(count * share)) for path, share in PATH_MIX}
        made: dict[str, int] = {path: 0 for path in quota}
        used_names: set[str] = set()
        used_companies: set[str] = set()
        leads: list[dict] = []
        countries = sorted(config.COUNTRY_TO_TERRITORY)

        # Deterministic candidate order: every account is offered to every path
        # before repeating, so territories spread instead of clustering.
        attempts = 0
        while sum(made.values()) < count and attempts < count * 60:
            attempts += 1
            wanted = next(
                (p for p in quota if made[p] < quota[p] and sum(made.values()) < count),
                None,
            )
            if wanted is None:
                break

            if wanted == "no_match":
                company = NO_MATCH_COMPANIES[made[wanted] % len(NO_MATCH_COMPANIES)]
                domain = None
                country = countries[attempts % len(countries)]
                state = None
            else:
                account = accounts[(attempts * 7) % len(accounts)]
                country, state = account["country"], account["state"]
                if wanted == "domain_exact":
                    company, domain = account["name"], account["domain"]
                elif wanted == "fuzzy_auto":
                    company, domain = _mangle_light(account["name"], rng), None
                else:
                    company, domain = _mangle_heavy(account["name"], rng), None

            if company.lower() in used_companies:
                continue
            if _classify(con, company, domain) != wanted:
                continue  # the real matcher disagreed — discard, do not relabel

            first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
            full = f"{first} {last}"
            if full in used_names:
                continue
            intent = rng.choice(list(NOTE_TEMPLATES))
            notes = rng.choice(NOTE_TEMPLATES[intent]).format(n=rng.choice([4, 6, 8, 12, 20, 35]))
            mail_domain = domain or f"{_slug(company)}.com"

            used_names.add(full)
            used_companies.add(company.lower())
            made[wanted] += 1
            lead = {
                "lead_id": f"L-2{len(leads) + 1:03d}",
                "company": company,
                "contact_name": full,
                "email": _email(first, last, mail_domain, rng),
                "title": rng.choice(TITLES),
                "country": country,
                "notes": notes,
                "_path": wanted,
            }
            if domain:
                lead["domain"] = domain
            if state:
                lead["state"] = state
            leads.append(lead)
    finally:
        con.close()
    return leads


def write(leads: list[dict], inbound_dir: Path | None = None) -> list[Path]:
    """Write generated leads, replacing any previous generated batch.

    The hand-authored beats in RESERVED_STEMS are never touched.
    """
    target = Path(inbound_dir) if inbound_dir else INBOUND_DIR
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob(f"{GENERATED_PREFIX}*.json"):
        stale.unlink()
    written = []
    for i, lead in enumerate(leads, start=1):
        payload = {k: v for k, v in lead.items() if not k.startswith("_")}
        path = target / f"{GENERATED_PREFIX}{i:03d}.json"
        if path.stem in RESERVED_STEMS:  # defensive; prefix makes this impossible
            continue
        path.write_text(json.dumps(payload, indent=2) + "\n")
        written.append(path)
    return written


def run_all(inbound_dir: Path | None = None, runs_dir=None) -> dict:
    """Route every generated lead through the REAL pipeline and resolve most.

    Terminal states are spread deterministically (approve / reject / leave
    pending) so the audit log shows all three, but each one is produced by the
    engine's own HITL path — never written directly.
    """
    from revenue_engine import pipeline, routing

    target = Path(inbound_dir) if inbound_dir else INBOUND_DIR
    routing.reset_state(runs_dir)
    tally = {"applied": 0, "rejected": 0, "pending": 0, "review": 0, "failed": 0}
    for i, path in enumerate(sorted(target.glob(f"{GENERATED_PREFIX}*.json"))):
        try:
            result = pipeline.run(path, approve=None, runs_dir=runs_dir)
        except Exception:
            tally["failed"] += 1
            continue
        if result.get("status") != "pending_approval":
            tally["review"] += 1
            continue
        decision = i % 5           # 0,1,2 -> approve | 3 -> reject | 4 -> leave pending
        if decision == 4:
            tally["pending"] += 1
            continue
        try:
            pipeline.run(path, approve=(decision != 3), runs_dir=runs_dir)
            tally["applied" if decision != 3 else "rejected"] += 1
        except Exception:
            tally["failed"] += 1
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", type=int, metavar="N",
                        help="generate N inbound leads into data/inbound/")
    parser.add_argument("--run", action="store_true",
                        help="route every generated lead through the real pipeline")
    parser.add_argument("--clean", action="store_true",
                        help="delete generated leads, keeping the hand-authored beats")
    args = parser.parse_args()

    if args.clean:
        removed = list(INBOUND_DIR.glob(f"{GENERATED_PREFIX}*.json"))
        for path in removed:
            path.unlink()
        print(f"removed {len(removed)} generated leads; {len(RESERVED_STEMS)} beats kept")
        return

    if args.generate:
        leads = generate(args.generate)
        written = write(leads)
        mix: dict[str, int] = {}
        for lead in leads:
            mix[lead["_path"]] = mix.get(lead["_path"], 0) + 1
        print(f"wrote {len(written)} leads to {INBOUND_DIR}")
        print("verified path mix:", dict(sorted(mix.items())))
        if len(written) < args.generate:
            print(f"note: asked for {args.generate}, the matcher only confirmed {len(written)}",
                  file=sys.stderr)

    if args.run:
        tally = run_all()
        print("routed:", tally)


if __name__ == "__main__":
    main()
