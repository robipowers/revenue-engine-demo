"""Tests for fuzzy account matching: domain shortcut, thresholds, candidates."""

from pathlib import Path

import duckdb
import pytest

from revenue_engine import matching, warehouse


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    path = tmp_path_factory.mktemp("wh") / "warehouse.duckdb"
    warehouse.rebuild(path)
    connection = duckdb.connect(str(path), read_only=True)
    yield connection
    connection.close()


def test_normalize_strips_legal_suffixes_and_punctuation():
    assert matching.normalize_name("Ledgerlyne Inc.") == "ledgerlyne"
    assert matching.normalize_name("Quotient Systems, LLC") == "quotient systems"
    assert matching.normalize_name("Wharfline Ltd") == "wharfline"
    # "Labs" is part of the brand, never stripped.
    assert matching.normalize_name("Northbeam Labs") == "northbeam labs"


def test_exact_domain_match_shortcut(con):
    result = matching.match_account(con, name="Totally Wrong Name", domain="ledgerline.com")
    assert result.status == "matched"
    assert result.method == "domain_exact"
    assert result.account_name == "Ledgerline"
    assert result.account_id is not None


def test_domain_match_is_case_insensitive(con):
    result = matching.match_account(con, name=None, domain="LEDGERLINE.COM")
    assert result.status == "matched"
    assert result.method == "domain_exact"


def test_fuzzy_match_above_90_auto_matches(con):
    result = matching.match_account(con, name="Ledgerlyne Inc.")
    assert result.status == "matched"
    assert result.method == "fuzzy_name"
    assert result.account_name == "Ledgerline"
    assert result.score >= 90


def test_fuzzy_70_to_89_returns_ambiguous_with_candidates(con):
    result = matching.match_account(con, name="Quotient Sys")
    assert result.status == "ambiguous"
    assert result.account_id is None
    assert result.candidates, "ambiguous result must carry candidates for the judgment step"
    top = result.candidates[0]
    assert top["name"] == "Quotient Systems"
    assert 70 <= top["score"] < 90
    # Candidates sorted best-first.
    scores = [c["score"] for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_below_70_returns_no_match(con):
    result = matching.match_account(con, name="Zephyr Dynamics")
    assert result.status == "no_match"
    assert result.account_id is None
    assert not result.candidates


def test_unknown_domain_falls_through_to_fuzzy(con):
    result = matching.match_account(con, name="Ledgerlyne Inc.", domain="ledgerlyne.example")
    assert result.status == "matched"
    assert result.method == "fuzzy_name"
