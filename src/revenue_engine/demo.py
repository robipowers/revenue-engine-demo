"""Demo CLI: run the full agentic pipeline on one inbound lead.

    python -m revenue_engine.demo data/inbound/lead_002.json [--llm] [--approve] [--reject] [--fresh]

Offline-safe by default: without --llm (or without ANTHROPIC_API_KEY) every
judgment step uses its deterministic fallback and nothing touches the
network. --fresh resets the round-robin rotation state only — the audit
log is append-only and never cleared.
"""

from __future__ import annotations

import argparse
import json

from revenue_engine import audit, pipeline, routing

WIDTH = 72


def _print_stages(result: dict) -> None:
    print("=" * WIDTH)
    print(f"REVENUE ENGINE — lead {result['lead_id']}  (llm={'on' if result['llm'] else 'off'})")
    print("=" * WIDTH)
    for stage in result["stages"]:
        flag = "  [fallback]" if stage["fallback_used"] else ""
        print(f"\n▸ {stage['stage'].upper()}  —  rule: {stage['rule_fired']}{flag}")
        for key, value in stage["decision"].items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value, default=str)
            print(f"    {key}: {value}")


def _print_proposal(result: dict) -> None:
    print("\n" + "-" * WIDTH)
    if result["status"] == "new_account":
        print("PROPOSAL: create a new account (no existing match)")
    else:
        print("PROPOSAL: assignment awaiting approval" if result["status"] == "pending_approval"
              else f"PROPOSAL ({result['status']})")
    print("-" * WIDTH)
    print(json.dumps(result["proposal"], indent=2, default=str))
    print(f"\nFINAL STATUS: {result['status']}")
    if result["status"] == "pending_approval":
        print("To approve this assignment, re-run with --approve "
              "(or reject it with --reject).")


def _print_audit_tail(runs_dir=None, n: int = 10) -> None:
    print("\n" + "-" * WIDTH)
    print(f"AUDIT TAIL (last {n} records, append-only)")
    print("-" * WIDTH)
    for record in audit.read_tail(n, runs_dir=runs_dir):
        fb = " fallback" if record["fallback_used"] else ""
        print(f"  {record['ts']}  {record['stage']:<20} {record['rule_fired']}{fb}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic revenue pipeline on one lead")
    parser.add_argument("lead", help="path to an inbound lead JSON file")
    parser.add_argument("--llm", action="store_true",
                        help="enable LLM judgment steps (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--approve", action="store_true", help="approve the proposed assignment")
    parser.add_argument("--reject", action="store_true", help="reject the proposal to human review")
    parser.add_argument("--fresh", action="store_true",
                        help="reset round-robin rotation state (audit log is never touched)")
    args = parser.parse_args()

    if args.approve and args.reject:
        parser.error("--approve and --reject are mutually exclusive")

    if args.fresh:
        routing.reset_state()
        print("(--fresh) routing rotation state reset; audit log untouched.\n")

    approve = True if args.approve else False if args.reject else None
    result = pipeline.run(args.lead, llm=args.llm, approve=approve)

    _print_stages(result)
    _print_proposal(result)
    _print_audit_tail()


if __name__ == "__main__":
    main()
