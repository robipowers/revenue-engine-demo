"""Tests for the inbound lead generator.

The generator fabricates INPUTS only. The guarantee that matters is that it
never mislabels one: every lead it claims takes a given matching path is
re-checked here against the real matcher, so a lead that quietly stopped
being ambiguous (or stopped matching at all) fails the suite instead of
silently weakening the demo.
"""

from __future__ import annotations

import json

import pytest

from revenue_engine import leadgen, matching, warehouse


@pytest.fixture(scope="module")
def leads():
    return leadgen.generate(24)


def test_generation_is_deterministic():
    assert leadgen.generate(16) == leadgen.generate(16)


def test_every_lead_takes_the_path_it_claims(leads):
    """The honesty check: the real matcher must agree with the label."""
    con = warehouse.connect()
    try:
        for lead in leads:
            result = matching.match_account(
                con, name=lead["company"], domain=lead.get("domain")
            )
            claimed = lead["_path"]
            if claimed == "domain_exact":
                assert result.status == "matched" and result.method == "domain_exact"
            elif claimed == "fuzzy_auto":
                assert result.status == "matched" and result.method == "fuzzy_name"
                assert result.score >= matching.AUTO_MATCH_THRESHOLD
            elif claimed == "ambiguous":
                # Ambiguous is a SCORE BAND, not a candidate count: one
                # candidate at 80 is ambiguous exactly as two at 85.7 are.
                assert result.status == "ambiguous"
                assert (matching.AMBIGUOUS_THRESHOLD
                        <= result.score < matching.AUTO_MATCH_THRESHOLD)
                assert result.candidates
            else:
                assert result.status == "no_match"
    finally:
        con.close()


def test_covers_every_matching_path(leads):
    paths = {lead["_path"] for lead in leads}
    assert paths == {"domain_exact", "fuzzy_auto", "ambiguous", "no_match"}


def test_leads_are_well_formed_and_distinct(leads):
    for lead in leads:
        for field in ("lead_id", "company", "contact_name", "email", "title",
                      "country", "notes"):
            assert lead.get(field), f"{field} missing from {lead['lead_id']}"
        assert "@" in lead["email"]
        assert lead["country"] in config_countries()
        if lead["_path"] == "no_match":
            assert "domain" not in lead, "a no-match lead must not carry a domain"
    # Repeated people or company names are the visible tell of a generator.
    assert len({l["lead_id"] for l in leads}) == len(leads)
    assert len({l["contact_name"] for l in leads}) == len(leads)
    assert len({l["company"].lower() for l in leads}) == len(leads)


def config_countries():
    from revenue_engine import config
    return set(config.COUNTRY_TO_TERRITORY) | {"United States"}


def test_write_replaces_generated_batch_and_never_touches_the_beats(tmp_path):
    beat = tmp_path / "lead_001.json"
    beat.write_text(json.dumps({"lead_id": "L-1001", "company": "Meterhouse"}))
    stale = tmp_path / f"{leadgen.GENERATED_PREFIX}999.json"
    stale.write_text("{}")

    written = leadgen.write(leadgen.generate(8), inbound_dir=tmp_path)

    assert len(written) == 8
    assert not stale.exists(), "a previous generated batch must be cleared"
    assert beat.exists(), "hand-authored beats must survive regeneration"
    assert json.loads(beat.read_text())["lead_id"] == "L-1001"
    # The internal path label is a build-time artifact, not part of the payload.
    for path in written:
        assert "_path" not in json.loads(path.read_text())


def test_generated_leads_are_routable_by_the_real_pipeline(tmp_path):
    """A generated lead must survive the actual engine, not just the matcher."""
    from revenue_engine import pipeline, routing

    written = leadgen.write(leadgen.generate(6), inbound_dir=tmp_path)
    routing.reset_state(tmp_path)
    seen = set()
    for path in written:
        result = pipeline.run(path, approve=None, runs_dir=tmp_path)
        assert result["lead_id"].startswith("L-2")
        # A matched lead stops at the HITL gate; an unmatched one opens a new
        # account and escalates. Both are terminal states the engine defines.
        assert result.get("status") in ("pending_approval", "new_account",
                                        "routed_to_review")
        seen.add(result["status"])
        assert result.get("stages"), "every run must record at least one stage"
    assert "pending_approval" in seen, "the batch must exercise the HITL gate"
