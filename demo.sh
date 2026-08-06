#!/usr/bin/env bash
# One-command interview demo runner.
# Press Enter between beats so you control the pacing while you talk.
# Usage:  ./demo.sh
set -euo pipefail

cd "$(dirname "$0")"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; OFF=$'\033[0m'

beat () {
  printf '\n%s\n' "${DIM}--------------------------------------------------------------------${OFF}"
  printf '%s\n' "${BOLD}${GREEN}$1${OFF}"
  printf '%s\n' "${DIM}$2${OFF}"
  printf '%s' "${DIM}[Enter to run]${OFF}"
  read -r _
  printf '\n'
}

clear 2>/dev/null || true
printf '%s\n' "${BOLD}Relay — autonomous revenue engine${OFF}"
printf '%s\n' "${DIM}Runs fully offline. No API keys, no network calls.${OFF}"

beat "BEAT 1 — clean lead, hot score, routed around a rep on vacation" \
     "Exact domain match. Scores 0.94. EMEA rep is unavailable, so it skips and audits that."
uv run python -m revenue_engine.demo data/inbound/lead_001.json --fresh

beat "BEAT 2 — human approves; the system applies exactly what was shown" \
     "The assignment only fires after a human says yes."
uv run python -m revenue_engine.demo data/inbound/lead_001.json --approve

beat "BEAT 3 — messy company name, no domain: AI judgment with a deterministic fallback" \
     "Two candidates tie at 85.7. The AI breaks the tie; with no AI available a rule does it instead."
uv run python -m revenue_engine.demo data/inbound/lead_002.json --fresh

beat "BEAT 4 — governance: every decision is on the record" \
     "Append-only audit log, one line per decision, inputs hashed."
tail -8 runs/audit.jsonl

printf '\n%s\n\n' "${BOLD}${GREEN}Demo complete.${OFF}"
