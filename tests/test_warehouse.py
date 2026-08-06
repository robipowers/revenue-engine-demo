"""Tests for the synthetic warehouse: determinism, integrity, and injected signal."""

import hashlib
from pathlib import Path

import duckdb
import pytest

from revenue_engine import config, warehouse


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("wh") / "warehouse.duckdb"
    warehouse.rebuild(path)
    return path


@pytest.fixture(scope="module")
def con(db_path):
    connection = duckdb.connect(str(db_path), read_only=True)
    yield connection
    connection.close()


def _table_hash(con, table: str) -> str:
    rows = con.execute(f"SELECT * FROM {table} ORDER BY id").fetchall()
    digest = hashlib.sha256(repr(rows).encode()).hexdigest()
    return digest


def test_rebuild_is_deterministic(tmp_path):
    path_a = tmp_path / "a.duckdb"
    path_b = tmp_path / "b.duckdb"
    warehouse.rebuild(path_a)
    warehouse.rebuild(path_b)
    con_a = duckdb.connect(str(path_a), read_only=True)
    con_b = duckdb.connect(str(path_b), read_only=True)
    try:
        for table in ("accounts", "contacts", "activities", "opportunities"):
            assert _table_hash(con_a, table) == _table_hash(con_b, table), (
                f"table {table} differs between two seeded rebuilds"
            )
    finally:
        con_a.close()
        con_b.close()


def test_row_counts_in_expected_bands(con):
    counts = {
        table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in ("accounts", "contacts", "activities", "opportunities")
    }
    assert 30 <= counts["accounts"] <= 60
    assert 80 <= counts["contacts"] <= 200
    assert 400 <= counts["activities"] <= 1000
    assert 50 <= counts["opportunities"] <= 120


def test_foreign_key_integrity(con):
    orphan_contacts = con.execute(
        "SELECT count(*) FROM contacts c LEFT JOIN accounts a ON c.account_id = a.id "
        "WHERE a.id IS NULL"
    ).fetchone()[0]
    orphan_activity_accounts = con.execute(
        "SELECT count(*) FROM activities x LEFT JOIN accounts a ON x.account_id = a.id "
        "WHERE a.id IS NULL"
    ).fetchone()[0]
    orphan_activity_contacts = con.execute(
        "SELECT count(*) FROM activities x LEFT JOIN contacts c ON x.contact_id = c.id "
        "WHERE c.id IS NULL"
    ).fetchone()[0]
    orphan_opps = con.execute(
        "SELECT count(*) FROM opportunities o LEFT JOIN accounts a ON o.account_id = a.id "
        "WHERE a.id IS NULL"
    ).fetchone()[0]
    assert orphan_contacts == 0
    assert orphan_activity_accounts == 0
    assert orphan_activity_contacts == 0
    assert orphan_opps == 0

    cross_account = con.execute(
        "SELECT count(*) FROM activities x JOIN contacts c ON x.contact_id = c.id "
        "WHERE x.account_id <> c.account_id"
    ).fetchone()[0]
    assert cross_account == 0, "activity contact belongs to a different account"


def test_signal_won_opps_have_more_precreation_activity(con):
    """Won opps must show materially more activity in the 30 days before creation."""
    rows = con.execute(
        """
        SELECT o.outcome,
               avg(cnt) AS mean_activity
        FROM (
            SELECT o.id, o.outcome,
                   (SELECT count(*) FROM activities x
                    WHERE x.account_id = o.account_id
                      AND x.ts >= o.created_at - INTERVAL 30 DAY
                      AND x.ts < o.created_at) AS cnt
            FROM opportunities o
            WHERE o.outcome IN ('won', 'lost')
        ) o
        GROUP BY o.outcome
        """
    ).fetchall()
    means = dict(rows)
    assert set(means) == {"won", "lost"}
    assert means["won"] > means["lost"], f"no pre-creation signal: {means}"
    assert means["won"] >= 1.5 * means["lost"], f"signal too weak for AUC gate: {means}"


def test_signal_won_opps_have_more_distinct_engaged_contacts(con):
    rows = con.execute(
        """
        SELECT o.outcome, avg(cnt)
        FROM (
            SELECT o.id, o.outcome,
                   (SELECT count(DISTINCT x.contact_id) FROM activities x
                    WHERE x.account_id = o.account_id
                      AND x.ts >= o.created_at - INTERVAL 30 DAY
                      AND x.ts < o.created_at) AS cnt
            FROM opportunities o
            WHERE o.outcome IN ('won', 'lost')
        ) o
        GROUP BY o.outcome
        """
    ).fetchall()
    means = dict(rows)
    assert means["won"] > means["lost"]


def test_every_account_maps_to_exactly_one_territory(con):
    rows = con.execute(
        "SELECT id, country, state, territory FROM accounts ORDER BY id"
    ).fetchall()
    valid = set(config.TERRITORIES)
    for account_id, country, state, territory in rows:
        assert territory in valid, f"account {account_id} has unknown territory {territory}"
        assert territory == config.resolve_territory(country, state), (
            f"account {account_id} territory column disagrees with mapping"
        )


def test_config_has_exactly_one_unavailable_rep():
    unavailable = [
        rep["name"]
        for reps in config.TERRITORIES.values()
        for rep in reps
        if not rep["available"]
    ]
    assert len(unavailable) == 1


def test_query_helpers(con):
    accounts = warehouse.search_accounts_by_name(con)
    assert len(accounts) >= 30
    sample_id, sample_name = accounts[0]["id"], accounts[0]["name"]

    domain = con.execute(
        "SELECT domain FROM accounts WHERE id = ?", [sample_id]
    ).fetchone()[0]
    by_domain = warehouse.get_account_by_domain(con, domain)
    assert by_domain is not None
    assert by_domain["id"] == sample_id
    assert warehouse.get_account_by_domain(con, "no-such-domain.example") is None

    context = warehouse.get_account_context(con, sample_id)
    assert context["account"]["name"] == sample_name
    assert isinstance(context["contacts"], list)
    assert isinstance(context["recent_activities"], list)
    assert isinstance(context["open_opportunities"], list)

    as_of = con.execute("SELECT max(ts) FROM activities").fetchone()[0]
    n_all = warehouse.activity_count_as_of(con, sample_id, as_of, window_days=3650)
    n_recent = warehouse.activity_count_as_of(con, sample_id, as_of, window_days=30)
    assert n_all >= n_recent >= 0
    n_contacts = warehouse.distinct_contacts_as_of(con, sample_id, as_of, window_days=3650)
    assert 0 <= n_contacts <= 10
    n_opps = warehouse.prior_opp_count_as_of(con, sample_id, as_of)
    assert n_opps >= 0


def test_demo_as_of_empty_opportunities_raises_clear_error(tmp_path):
    """No opportunities means no demo clock anchor: fail with a rebuild hint,
    never a bare TypeError."""
    path = tmp_path / "empty.duckdb"
    setup = duckdb.connect(str(path))
    setup.execute(warehouse.DDL)
    setup.close()
    with pytest.raises(ValueError, match="rebuild"):
        warehouse.demo_as_of(path)
