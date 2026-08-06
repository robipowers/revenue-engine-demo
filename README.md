# Revenue Engine Demo

A working slice of AI GTM orchestration: signals in, orchestrated actions out. A DuckDB star schema acts as the unified data layer, an MCP server exposes it as a context layer any agent can query, a propensity model scores accounts on pre-opportunity engagement signals, and an agentic pipeline blends AI judgment with deterministic guardrails to match, enrich, score, and route inbound leads. Every decision lands in an append-only audit trail (governance and auditability), and no assignment applies without an explicit human approval (the HITL gate). The whole thing runs offline, deterministically, in seconds.

## Quickstart

```
uv sync
uv run pytest
```

(Or `pip install -e . && pytest` if you prefer pip.)

Both build artifacts ship committed for zero-setup: the seed warehouse (`data/seed/warehouse.duckdb`) and the trained model (`models/propensity.joblib`). Rebuilding either is optional and fully deterministic (seed 42):

```
uv run python -m revenue_engine.warehouse --rebuild
uv run python -m revenue_engine.model --train
```

Run the pipeline on any inbound lead:

```
uv run python -m revenue_engine.demo data/inbound/lead_001.json
```

Flags: `--approve` applies the proposed assignment, `--reject` sends it to human review, `--fresh` resets round-robin rotation state (never the audit log), `--llm` enables the two LLM judgment steps. The LLM path needs both `ANTHROPIC_API_KEY` and the `anthropic` package, installed via the optional extra:

```
uv run --extra llm python -m revenue_engine.demo data/inbound/lead_001.json --llm
```

Without the key or the package every step uses its deterministic fallback and nothing touches the network. A missing package is reported distinctly (`llm_unavailable: anthropic not installed (install extra: llm)`) on stderr and in the audit trail, so it is never mistaken for an API failure.

A pending run persists its proposal to `runs/pending/<lead_id>.json`. A later `--approve` or `--reject` consumes that stored proposal exactly as it was shown, without re-running the pipeline or advancing the round-robin rotation a second time.

## The 5-minute demo script

1. **The hot lead.** Run:

   ```
   uv run python -m revenue_engine.demo data/inbound/lead_001.json --fresh
   ```

   Say: "A German VP of RevOps asks for a demo. Domain match is exact (score 100), enrichment pulls her account context from the warehouse, the model scores the account 0.9351 on pre-opportunity engagement signals, so this is P1 with a 15-minute SLA. Watch the routing: EMEA's first rep is on vacation, and the audit tail shows a `routing_skip` record before the assignment lands on Tomas Lindqvist. The skip itself is audited, not silent."

2. **The messy lead.** Run:

   ```
   uv run python -m revenue_engine.demo data/inbound/lead_002.json
   ```

   Say: "This one is ugly: no domain, company typed as 'Hine Metric Co', sloppy notes. Fuzzy matching finds two candidates tied at 85.7 (Hivemetric and Pinemetric), which is exactly where an LLM tie-break earns its keep. Offline it collapses to a deterministic fallback, and the audit record says so (`fallback_used`). The account scores 0.0893, so P2, 4-hour SLA. The contrast with lead 1 is the model doing its job."

3. **Close the loop.** Run:

   ```
   uv run python -m revenue_engine.demo data/inbound/lead_002.json --approve
   ```

   Say: "Nothing was assigned yet. The pipeline always stops at `pending_approval`, persists the proposal to `runs/pending/`, and only an explicit `--approve` writes `assignment_applied`. The approve run consumes the stored proposal: the exact rep you just saw is the rep applied, the pipeline does not re-run, and the round-robin rotation does not advance a second time. `--fresh` resets rotation state without ever touching the audit log."

4. **Governance.** Run:

   ```
   tail -5 runs/audit.jsonl
   ```

   Say: "Every stage of every run is one append-only JSONL record: timestamp, stage, inputs, a SHA-256 hash of those inputs, decision, and which rule fired. Nothing in the codebase can truncate this file. You can replay any decision from here."

5. **The context layer (optional).** The same warehouse helpers are exposed as an MCP server with three read tools, so any MCP client sees the same data the pipeline uses:

   ```
   uv run python -c "from revenue_engine.mcp_server import lookup_account; print(lookup_account('Hine Metric Co'))"
   ```

   Say: "Same ranked candidates as the pipeline's matcher, because it is the same scoring code. `account_context` adds the current propensity score, `territory_rules` exposes rosters and SLA tiers." For a live inspector, `uv run mcp dev src/revenue_engine/mcp_server.py` works but needs `mcp[cli]`, npx, and network, so treat it as an optional manual smoke only.

   Paths anchor to the repo root derived from the installed package location, so the MCP server (and the demo CLI) can be launched from any working directory. Set `REVENUE_ENGINE_ROOT` to point at a different root when the data and model artifacts live elsewhere, for example in an MCP client config that runs the server outside the checkout.

If asked about the no-match path: `lead_003.json` (an unknown company) exits with a structured `create_account` proposal instead of guessing.

## Architecture

```
data/inbound/*.json                    inbound signal
        |
match (rapidfuzz, 3-band policy) --> tie_break (LLM or fallback)
        |
enrich (DuckDB star schema: accounts, contacts, activities, opportunities)
        |
intent_summary (LLM or keyword fallback)
        |
score (logistic regression, features as of the demo clock)
        |
route (territory map -> availability-aware round-robin -> SLA tier)
        |
HITL gate --> approve / reject             every stage: runs/audit.jsonl
```

**The frozen demo clock.** The seed data never moves, so scoring against the real wall clock would decay toward zero as the calendar drifts past the last recorded activity. Instead, "now" is anchored to the dataset itself (the creation moment of the newest opportunity, plus one hour), so scores stay meaningful on any demo day and every run is replayable. The same as-of feature code path serves training and live scoring, which makes train/serve skew structurally impossible.

**The model.** Logistic regression on six as-of features, holdout AUC 0.877 on the synthetic data. Full details, including the leakage guard and known limitations, are in [`models/model_card.md`](models/model_card.md).

**LLM guardrails.** The two judgment steps are the only places an LLM can participate, and it can never expand its own action space: a tie-break answer outside the candidate list, a malformed intent reply, an error, or a timeout all collapse to the deterministic fallback, and the audit record marks which path ran.

## Limitations (honest ones)

The data is synthetic and small (40 accounts, 80 opportunities), so the 0.877 AUC measures the injected signal, not real-world predictive power. The matcher handles domains and fuzzy names but not subsidiaries, mergers, or free-mail domains. Routing covers four territories with two reps each; real territory management has capacity weighting, holdouts, and account ownership rules. The HITL gate is a CLI flag standing in for a real approval UI. The audit log is append-only by construction but not tamper-evident beyond per-record input hashes (no chained hashes or signatures). None of these gaps are hidden; each one is where the production version of this system would start.
