# Model Card — Opportunity Win Propensity (`propensity.joblib`)

Logistic regression (scikit-learn `Pipeline`: `StandardScaler` → `LogisticRegression`)
that scores an account's likelihood of winning an opportunity, in [0, 1].
Trained deterministically with `RANDOM_SEED = 42`; retrain any time with
`python -m revenue_engine.model --train`.

## Features

| Feature | Type | Source | Default when absent |
|---|---|---|---|
| `activity_count_30d` | count | activities in the 30 days before the as-of timestamp | 0 |
| `distinct_contacts_engaged` | count | distinct contacts with activity in that 30-day window | 0 |
| `days_since_last_activity` | int (capped) | days from last activity strictly before as-of; capped at 365 | 365 (documented cap) |
| `prior_opportunity_count` | count | opportunities created strictly before as-of | 0 |
| `employee_band` | ordinal 0–4 | account employee band (`1-50` → 0 … `5000+` → 4) | required |
| `industry_fit` | binary | 1 if account industry is in `INDUSTRY_FIT` (config) | 0 |

## As-of semantics (leakage guard)

Every feature is computed **as of a reference timestamp** through a single
code path (`features.features_as_of`):

- **Training** uses each opportunity's `created_at` — only activity that
  existed *before* the opportunity was created can influence its label's
  features. A test proves this by injecting post-creation activity and
  asserting the training features do not move.
- **Live scoring** uses "now", with identical queries, so there is no
  train/serve skew.

Structurally-absent values fall back to the documented defaults above, so any
matched account scores without error; malformed input dicts raise a clear
`ValueError` instead of silently defaulting.

## Training data

- Source: the deterministic synthetic warehouse (`data/seed/warehouse.duckdb`)
- 60 closed opportunities: **30 won / 30 lost** (perfectly balanced classes)
- Stratified 70/30 split: **42 training rows, 18 holdout rows**

## Evaluation

- **Holdout AUC: 0.877** (stratified holdout, seed 42)
- The synthetic generator deliberately overlaps the won/lost activity
  distributions so the model lands in a *credible* band rather than the
  perfect separation a naive toy dataset produces. An AUC of 1.0 would be a
  red flag for leakage; a test enforces 0.65 < AUC < 0.99.

## Limitations (read before trusting a score)

- **Synthetic data.** Every account, contact, activity, and outcome is
  generated. The model demonstrates the *pipeline* (as-of features, leakage
  guards, deterministic training, honest evaluation), not real-world lift.
- **Tiny n.** 60 labeled rows and an 18-row holdout: the AUC has wide error
  bars and would not survive contact with production data volumes.
- **The signal is injected**, then recovered — circular by construction. On
  real CRM data, feature engineering and validation would need far more care
  (temporal cross-validation, calibration curves, drift monitoring).
- **Illustrative only.** Do not use for real routing, forecasting, or any
  revenue decision.
