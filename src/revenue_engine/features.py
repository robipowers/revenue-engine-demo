"""Feature engineering for the propensity model, with strict as-of semantics.

Leakage guard: every feature is computed *as of* a reference timestamp.
Training rows use the opportunity's created_at (only activity in the 30 days
BEFORE creation counts); live scoring uses "now". The same code path serves
both, so train/serve skew is structurally impossible.

Defaults for structurally-absent values (so any matched account scores
without error): counts default to 0 and days_since_last_activity defaults to
the documented cap of 365. Malformed input dicts raise ValueError instead.
"""

from __future__ import annotations

from datetime import datetime

from revenue_engine.config import EMPLOYEE_BANDS, INDUSTRY_FIT
from revenue_engine.warehouse import (
    activity_count_as_of,
    distinct_contacts_as_of,
    prior_opp_count_as_of,
)

FEATURE_NAMES = [
    "activity_count_30d",
    "distinct_contacts_engaged",
    "days_since_last_activity",
    "prior_opportunity_count",
    "employee_band",
    "industry_fit",
]

# Documented cap: an account with no recorded activity (or activity older than
# the cap) is treated as "365 days since last touch".
DAYS_SINCE_LAST_ACTIVITY_CAP = 365

_COUNT_FEATURES = {
    "activity_count_30d",
    "distinct_contacts_engaged",
    "prior_opportunity_count",
}


def employee_band_ordinal(band: str | int) -> int:
    """Map an employee band label to its ordinal (0 = smallest)."""
    if isinstance(band, int):
        if 0 <= band < len(EMPLOYEE_BANDS):
            return band
        raise ValueError(
            f"employee_band ordinal {band} out of range 0..{len(EMPLOYEE_BANDS) - 1}"
        )
    if band in EMPLOYEE_BANDS:
        return EMPLOYEE_BANDS.index(band)
    raise ValueError(
        f"employee_band {band!r} is not one of {EMPLOYEE_BANDS} or an ordinal"
    )


def industry_fit(industry: str) -> int:
    """1 if the industry is in the product's sweet spot, else 0."""
    return 1 if industry in INDUSTRY_FIT else 0


def days_since_last_activity_as_of(con, account_id: int, as_of: datetime) -> int:
    """Whole days since the last activity strictly before as_of, capped."""
    last_ts = con.execute(
        "SELECT max(ts) FROM activities WHERE account_id = ? "
        "AND ts < CAST(? AS TIMESTAMP)",
        [account_id, as_of],
    ).fetchone()[0]
    if last_ts is None:
        return DAYS_SINCE_LAST_ACTIVITY_CAP
    days = int((as_of - last_ts).total_seconds() // 86_400)
    return max(0, min(days, DAYS_SINCE_LAST_ACTIVITY_CAP))


def features_as_of(con, account_id: int, as_of: datetime) -> dict:
    """Compute the full feature dict for an account as of a timestamp.

    Used verbatim for training (as_of = opportunity created_at) and live
    scoring (as_of = now).
    """
    account = con.execute(
        "SELECT employee_band, industry FROM accounts WHERE id = ?", [account_id]
    ).fetchone()
    if account is None:
        raise ValueError(f"unknown account_id {account_id}")
    band, industry = account
    return {
        "activity_count_30d": activity_count_as_of(con, account_id, as_of, window_days=30),
        "distinct_contacts_engaged": distinct_contacts_as_of(con, account_id, as_of, window_days=30),
        "days_since_last_activity": days_since_last_activity_as_of(con, account_id, as_of),
        "prior_opportunity_count": prior_opp_count_as_of(con, account_id, as_of),
        "employee_band": band,
        "industry_fit": industry_fit(industry),
    }


def vectorize(features: dict) -> list[float]:
    """Validate a feature dict and return the model input vector.

    Applies the documented defaults for structurally-absent values; raises
    ValueError with a clear message for anything malformed.
    """
    if not isinstance(features, dict):
        raise ValueError(
            f"features must be a dict of feature_name -> value, got {type(features).__name__}"
        )
    unknown = set(features) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature keys: {sorted(unknown)}")
    if "employee_band" not in features:
        raise ValueError("missing required feature 'employee_band'")

    vector: list[float] = []
    for name in FEATURE_NAMES:
        if name == "employee_band":
            vector.append(float(employee_band_ordinal(features[name])))
            continue
        if name in _COUNT_FEATURES:
            value = features.get(name, 0)
        elif name == "days_since_last_activity":
            value = features.get(name, DAYS_SINCE_LAST_ACTIVITY_CAP)
        else:  # industry_fit
            value = features.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"feature {name!r} must be numeric, got {value!r}")
        vector.append(float(value))
    return vector
