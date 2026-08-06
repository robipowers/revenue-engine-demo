# UI Brief — Live pipeline dashboard (one mind, before fan-out)

## What this is
A local web dashboard over the REAL pipeline for the LeanData interview demo. One page.
No frameworks, no build step, no external requests: one Python stdlib server + one
self-contained index.html (inline CSS/JS, system fonts, pure HTML/CSS/SVG visuals).

## Quality bar (gauntlet)
Blind-competitive against LeanData's own site (refs/leandata-home.png) restyled as a
PRODUCT screen. Robi's bar (final judge): product proof and density over whitespace
restraint; bold, alive, believable. It must read as "the autonomous revenue engine,
running" in 3 seconds.

## Brand
Dark theme, LeanData-family green accent (#2ecc71 range) + white text, red for SEV/skip
accents. Title: "Relay — Revenue Engine" (our product), subtitle nods to "intelligent
GTM orchestration" vocabulary. Monospace for data/audit, sans for chrome. NOT a clone
of their marketing site: it's the product console their graphics imply.

## Layout (single screen, 1440 target, must not break at 1024)
- Left rail: INBOUND — the three lead cards (from /api/leads) with name/company/notes
  preview; click to run. A "fresh" toggle (default on).
- Center: THE PIPELINE — five stage nodes in a connected flow (MATCH → ENRICH → INTENT
  → SCORE → ROUTE) that light up IN SEQUENCE as the run's audit records replay
  (~450ms stagger), each node filling with its real decision data (match score +
  candidates, account context, intent chip, score dial 0..1 with P1/P2 threshold mark,
  territory + chosen rep). Ambiguous match shows the two tied candidates and the
  tie-break resolution. Fallback steps get an amber "deterministic fallback" chip.
- Right rail top: ROUTING DESK — territory rosters from /api/territories; the chosen
  rep highlights green; an unavailable rep renders greyed with "on vacation" badge and
  a red skip marker when the run skipped them.
- Right rail bottom / footer band: GOVERNANCE — live audit stream (monospace rows:
  time, stage, rule, fallback flag) appending in sync with the replay; input-hash
  shown truncated.
- HITL: when a run ends pending_approval, a fixed action bar appears: proposal summary
  (rep, territory, priority/SLA, score) + APPROVE / REJECT buttons → calls
  /api/approve → stage flips to assignment_applied / routed_to_review and the audit
  stream appends the record. Idempotent replay renders a "already resolved" state.
- Errors (corrupt state ValueError etc.) render as a red banner with the message.

## Server (src/revenue_engine/ui.py, stdlib http.server only)
- GET /            → ui/index.html
- GET /api/leads   → [{lead_id, file, company, contact_name, title, country, state, notes}]
- GET /api/territories → config.TERRITORIES (+ which reps available)
- POST /api/run    → {lead: <lead_id or file>, fresh: bool}: snapshot audit.jsonl line
  count, call pipeline.run(path, approve=None) (reset_state first when fresh), return
  {status, proposal?, records: <new audit lines parsed>, error?}
- POST /api/approve→ {lead: ..., decision: "approve"|"reject"}: pipeline.run with
  approve=True/False; same return shape.
- Serve on 127.0.0.1:5210. `python -m revenue_engine.ui` starts it and prints the URL.
- All JSON; ValueError → HTTP 409 {"error": msg}; никогда tracebacks to the client
  (structured error only). No writes beyond what the pipeline itself does.

## Honesty constraints
Every number on screen comes from the pipeline's real return + real audit records of
THIS run. No invented metrics, no fake charts. The replay animation is presentation
timing only. Believable density everywhere; no lorem ipsum.

## Verification
- `uv run pytest` stays green (server has a light test: endpoints return documented
  shapes in-process).
- Screenshot gate: tools/shot.mjs against the running server, desktop 1440; critic
  compares vs refs/leandata-home.png with Robi's bar; Robi is the final judge.
