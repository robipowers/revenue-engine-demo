"""Central configuration: seed, paths, territories, rep rosters, industry fit."""

from __future__ import annotations

import os
from pathlib import Path

# Single seed that makes every layer of the demo deterministic.
RANDOM_SEED = 42

# Paths resolve from the repo root, derived from this package's location
# (src/revenue_engine/config.py -> two parents up). Override with
# REVENUE_ENGINE_ROOT when running against a relocated data/model layout.
ROOT = Path(
    os.environ.get("REVENUE_ENGINE_ROOT") or Path(__file__).resolve().parents[2]
).resolve()
DATA_DIR = ROOT / "data"
SEED_DB = DATA_DIR / "seed" / "warehouse.duckdb"
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"

DEFAULT_TERRITORY = "AMER-West"

# Territory -> rep roster. Exactly one rep is unavailable (on vacation),
# so routing has to demonstrate a fallback path.
TERRITORIES: dict[str, list[dict]] = {
    "AMER-East": [
        {"name": "Maya Delgado", "available": True},
        {"name": "Chris Okafor", "available": True},
    ],
    "AMER-West": [
        {"name": "Jordan Feldman", "available": True},
        {"name": "Alicia Zhang", "available": True},
    ],
    "EMEA": [
        {"name": "Priya Raman", "available": False},  # on vacation
        {"name": "Tomas Lindqvist", "available": True},
    ],
    "APAC": [
        {"name": "Kenji Watanabe", "available": True},
        {"name": "Sarah Nguyen", "available": True},
    ],
}

# US states east of the Mississippi (roughly) route to AMER-East;
# everything else in the US routes to AMER-West.
US_EAST_STATES = {
    "NY", "MA", "NJ", "PA", "CT", "VA", "NC", "GA", "FL", "OH", "MI", "IL",
    "MD", "DC", "TN", "SC", "IN", "KY", "AL", "MS", "WI", "VT", "NH", "ME",
    "RI", "DE", "WV",
}

COUNTRY_TO_TERRITORY: dict[str, str] = {
    "United Kingdom": "EMEA",
    "Germany": "EMEA",
    "France": "EMEA",
    "Netherlands": "EMEA",
    "Ireland": "EMEA",
    "Spain": "EMEA",
    "Sweden": "EMEA",
    "Australia": "APAC",
    "Japan": "APAC",
    "Singapore": "APAC",
    "India": "APAC",
    "New Zealand": "APAC",
    "Canada": "AMER-East",
    "Mexico": "AMER-West",
    "Brazil": "AMER-East",
}

# Industries the (fictional) product sells best into; used for fit scoring.
INDUSTRY_FIT = [
    "SaaS",
    "Fintech",
    "Healthcare Technology",
    "Logistics",
    "Cybersecurity",
]

EMPLOYEE_BANDS = ["1-50", "51-200", "201-1000", "1001-5000", "5000+"]


def resolve_territory(country: str, state: str | None) -> str:
    """Map a country (and US state) to exactly one territory."""
    if country == "United States":
        if state and state.upper() in US_EAST_STATES:
            return "AMER-East"
        return "AMER-West"
    return COUNTRY_TO_TERRITORY.get(country, DEFAULT_TERRITORY)
