"""Tests for the propensity model: determinism, calibrated AUC, leakage guard,
defaults, purity, input validation, model-card honesty, and shipped-artifact
drift guards."""

import hashlib
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pytest

from revenue_engine import features, model, warehouse
from revenue_engine.config import RANDOM_SEED, ROOT


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("model_wh") / "warehouse.duckdb"
    warehouse.rebuild(path)
    return path


@pytest.fixture(scope="module")
def trained(db_path, tmp_path_factory):
    model_path = tmp_path_factory.mktemp("models") / "propensity.joblib"
    return model.train(db_path=db_path, model_path=model_path)


def test_training_is_deterministic(db_path, tmp_path):
    run_a = model.train(db_path=db_path, model_path=tmp_path / "a.joblib")
    run_b = model.train(db_path=db_path, model_path=tmp_path / "b.joblib")
    clf_a = run_a["model"].named_steps["logreg"]
    clf_b = run_b["model"].named_steps["logreg"]
    assert np.array_equal(clf_a.coef_, clf_b.coef_), "coefficients differ across runs"
    assert np.array_equal(clf_a.intercept_, clf_b.intercept_)
    assert run_a["holdout_auc"] == run_b["holdout_auc"]


def test_holdout_auc_in_credible_band(trained):
    auc = trained["holdout_auc"]
    assert auc > 0.65, f"model has no useful signal (AUC={auc:.3f})"
    assert auc < 0.99, (
        f"AUC={auc:.3f} looks like perfect separation / leakage; "
        "the demo deliberately calibrates signal into a credible band"
    )


def test_no_training_feature_uses_post_creation_activity(db_path, tmp_path):
    """Constructed proof: injecting activity AFTER an opportunity's created_at
    must not change any training feature for that opportunity."""
    writable = tmp_path / "leak.duckdb"
    writable.write_bytes(Path(db_path).read_bytes())
    con = duckdb.connect(str(writable))
    try:
        opp = con.execute(
            "SELECT id, account_id, created_at FROM opportunities "
            "WHERE outcome IS NOT NULL ORDER BY id LIMIT 1"
        ).fetchone()
        opp_id, account_id, created_at = opp
        before = features.features_as_of(con, account_id, created_at)

        contact_id = con.execute(
            "SELECT id FROM contacts WHERE account_id = ? LIMIT 1", [account_id]
        ).fetchone()[0]
        next_id = con.execute("SELECT max(id) + 1 FROM activities").fetchone()[0]
        for i in range(25):  # a loud burst of FUTURE activity
            con.execute(
                "INSERT INTO activities VALUES (?, ?, ?, 'email_open', ?)",
                [next_id + i, account_id, contact_id, created_at + timedelta(days=1 + i % 5)],
            )
        after = features.features_as_of(con, account_id, created_at)
    finally:
        con.close()
    assert before == after, "training features leaked post-creation activity"


def test_zero_activity_account_scores_with_defaults(db_path, trained):
    """A matched account with no recorded activity must score without error."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        account_id = con.execute("SELECT min(id) FROM accounts").fetchone()[0]
        long_ago = con.execute("SELECT min(ts) FROM activities").fetchone()[0] - timedelta(days=400)
        feats = features.features_as_of(con, account_id, long_ago)
    finally:
        con.close()
    assert feats["activity_count_30d"] == 0
    assert feats["distinct_contacts_engaged"] == 0
    assert feats["days_since_last_activity"] == features.DAYS_SINCE_LAST_ACTIVITY_CAP
    p = model.score(feats, trained["model"])
    assert 0.0 <= p <= 1.0


def test_partial_dict_defaults_missing_counts(trained):
    feats = {"employee_band": "51-200", "industry_fit": 1}
    p = model.score(feats, trained["model"])
    assert 0.0 <= p <= 1.0


def test_score_is_pure(trained):
    feats = {
        "activity_count_30d": 7,
        "distinct_contacts_engaged": 2,
        "days_since_last_activity": 3,
        "prior_opportunity_count": 1,
        "employee_band": "201-1000",
        "industry_fit": 1,
    }
    scores = {model.score(feats, trained["model"]) for _ in range(5)}
    assert len(scores) == 1, "same input produced different scores"


def test_malformed_input_raises_clear_error(trained):
    with pytest.raises(ValueError, match="dict"):
        model.score(["not", "a", "dict"], trained["model"])
    with pytest.raises(ValueError, match="unknown feature"):
        model.score({"totally_bogus_feature": 1}, trained["model"])
    with pytest.raises(ValueError, match="employee_band"):
        model.score({"activity_count_30d": 3}, trained["model"])
    with pytest.raises(ValueError, match="employee_band"):
        model.score({"employee_band": "not-a-band", "industry_fit": 0}, trained["model"])
    with pytest.raises(ValueError, match="numeric"):
        model.score(
            {"activity_count_30d": "lots", "employee_band": "1-50", "industry_fit": 1},
            trained["model"],
        )


def test_model_card_lists_every_feature():
    card = (ROOT / "models" / "model_card.md").read_text()
    for name in features.FEATURE_NAMES:
        assert name in card, f"model card missing feature {name}"


def test_shipped_artifacts_exist():
    assert (ROOT / "models" / "propensity.joblib").exists()
    assert (ROOT / "data" / "seed" / "warehouse.duckdb").exists()


def _table_hash(con, table: str) -> str:
    rows = con.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def test_shipped_warehouse_matches_fresh_rebuild(tmp_path):
    """Drift guard: the committed seed DB must equal what the generator
    produces today, table by table."""
    fresh = tmp_path / "fresh.duckdb"
    warehouse.rebuild(fresh)
    con_fresh = duckdb.connect(str(fresh), read_only=True)
    con_shipped = duckdb.connect(
        str(ROOT / "data" / "seed" / "warehouse.duckdb"), read_only=True
    )
    try:
        for table in ("accounts", "contacts", "activities", "opportunities"):
            assert _table_hash(con_fresh, table) == _table_hash(con_shipped, table), (
                f"shipped seed DB table {table} drifted from the generator; "
                "regenerate with: uv run python -m revenue_engine.warehouse --rebuild"
            )
    finally:
        con_fresh.close()
        con_shipped.close()


def test_shipped_model_payload_matches_code():
    """Drift guard: the committed model must carry the current feature set
    and the canonical seed."""
    payload = model.load(model.MODEL_PATH)
    assert payload["feature_names"] == list(features.FEATURE_NAMES), (
        "shipped model feature_names drifted from features.FEATURE_NAMES; "
        "retrain with: uv run python -m revenue_engine.model --train"
    )
    assert payload["random_seed"] == RANDOM_SEED
