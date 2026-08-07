# Relay — what every screen is, and what it proves

Read this once with the console open. It maps what you see to what it
demonstrates, and gives you the sentence to say for each.

Start it:

```
cd ~/Downloads/leandata-revenue-engine-demo
uv run python -m revenue_engine.leadgen --reset-demo   # clean history first
uv run python -m revenue_engine.ui                      # http://127.0.0.1:5210
```

---

## The one-sentence version

**A lead arrives. The system matches it to an account, enriches it from a
warehouse, judges intent, scores it with a trained model, routes it to a rep,
and stops for a human — writing an append-only record of every step.**

Everything on screen is that sentence, shown from a different angle.

---

## The mental model

Think of it as a **mail room with a paper trail**.

A letter arrives with no address. A clerk figures out which company it belongs
to (**match**), pulls that company's file (**enrich**), reads the letter to see
what they want (**intent**), checks how hot the account is (**score**), decides
who should handle it (**route**), and then — this is the important part — puts
it in an out-tray for a supervisor instead of delivering it (**the HITL gate**).
Every one of those steps gets stamped into a logbook that can only be added to,
never edited.

The demo's whole argument is that last part: **you can audit every decision.**

---

## Screen by screen

### 1. Routing Overview — the landing screen

Six numbers across the top, a live stream on the left, the queue and rep desk
on the right.

- **The stream is the logbook**, newest first. Every row is one decision, with
  the rule that fired and a hash of its inputs.
- **FALLBACKS %** is the headline governance number: how often an AI step
  degraded to a deterministic rule and the system kept working anyway.
- **Rep Desk** shows load per rep, derived from the log — including Priya Raman
  at 0 routed / 7 skipped, because she is on vacation.

> Say: "This is the engine's own record of itself. Nothing here is a dashboard
> metric someone chose — it is a count of what actually happened."

### 2. Routing Queue — the work

43 inbound leads, each with its last outcome, owner, priority and propensity.
The three `L-100x` leads are the hand-authored beats; the `L-2xxx` ones give
the routing surfaces a real book.

Click any row to run it through the engine.

### 3. The lead record — the differentiator

Open `?demo=lead_001`. Four regions:

- **Left rail** — the lead, the matched account, the routing decision, the raw
  inbound note, the known contacts, and what this run wrote.
- **Highlights** — six real numbers, each with its provenance underneath.
- **Decision trace** — the flow. **Solid line = the path this lead took,
  labelled with the condition that routed it. Dashed grey = a branch the flow
  defines but this lead did not take. Dotted = a result.**
- **Stage waterfall** — the same run as a table, with a `Source` chip per stage
  saying which subsystem produced that answer.

> Say: "The diagram is not a picture of the design. It is drawn from this run's
> audit records. A different lead draws a different diagram."

### 4. The two beats, and why they differ

| | `?demo=lead_001` | `?demo=lead_002` |
|---|---|---|
| match | exact domain, score 100 | ambiguous, two candidates tied at 85.7 |
| tie-break | not needed | fires — resolved by lowest account id |
| score | **0.9727** | **0.0017** |
| priority | P1, 15-minute SLA | P2, 4-hour SLA |
| twist | EMEA's first rep is on vacation — the skip is **audited, not silent** | a fuzzy match a human would also have to think about |

That contrast is the demo. One clean lead, one messy one, both handled, both
fully explained.

### 5. Propensity v1 — the model, opened up

Trained coefficients as diverging bars, then the exact contribution breakdown
for the most recent score, ending in `sigmoid(intercept + Σ contributions)`.
Below that, all 40 accounts scored by the same model.

> Say: "Six features, holdout AUC 0.83. Not 1.0 — a perfect score on 60 rows
> would mean leakage, not skill."

### 6. Audit Log — the governance artifact

Every decision, newest first, with `rule_fired` and an input hash. This is the
file the whole demo rests on.

### 7. Territory Management / Round Robin

The literal maps the router reads at decision time, and the coverage maths:
accounts per available rep, pipeline per available rep. Priya Raman's absence
shows up as Tomas carrying all 7 EMEA accounts alone.

---

## The three questions you will get

**"Is any of this real?"**
The sample database is generated, like any demo's. What is real is everything
downstream: every score, routing decision and audit record comes from actually
running the pipeline. Nothing is typed in.

**"What happens when the LLM is down?"**
It already is — there is no API key set. Every AI step has a deterministic
fallback, the fallback is recorded in the audit log with `fallback_used: true`,
and the run completes. That is what the FALLBACKS number counts.

**"What is the scoring clock?"**
A leakage guard. Features are computed as of a fixed timestamp, and the
warehouse deliberately contains later facts the model is not allowed to see —
those rows are chipped `post-clock` in Opportunities. If nothing existed after
the clock, the guard would be meaningless.

---

## Two things to know before you are asked

- **The account lead_001 matches has a deliberate engagement surge** in the
  sample data, so the demo can show a P1 and a P2 side by side. It is scored by
  the same model on the same features as every other account.
- **The audit log is append-only, so it grows while you rehearse.** Run
  `--reset-demo` before you present.
