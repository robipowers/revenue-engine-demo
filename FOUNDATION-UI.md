# Relay console — design foundation

Single source of truth for the rebuild. Written before any building so the console
is one product, not six agents' opinions. Builders and fixers read this first and
do not renegotiate it; only the orchestrator changes it.

Bar: `refs/leandata-flowbuilder.png` (IA + flow grammar), `refs/attio-table.png`
(table craft), `refs/attio-record.png` (record layout), `refs/clay-waterfall.png`
(sequential per-row status).

Thesis: **LeanData's product, if Attio's design team had built it.** Take LeanData's
information architecture and flow grammar so a LeanData interviewer reads it instantly.
Take Attio's craft: light, dense, chip-typed, tight rhythm. Drop the Salesforce
Lightning chrome that surrounds the real product — the dark gray rail, the tab bar,
the 2013 gradients. Relay is its own product that speaks their language.

---

## 1. Non-negotiables carried into every decision

1. **Every number is a real query or a real audit record.** No placeholder charts,
   no invented metrics. If a panel looks thin, fill it with more real data — the
   warehouse has 40 accounts, 120 named contacts, 708 timestamped activities,
   80 opportunities totalling $11,077,000, and the audit log has 365 records.
2. **Fully offline.** One self-contained `ui/index.html`, inline CSS + JS. No CDN,
   no web fonts, no network beyond same-origin `/api/*`. It must never fail on stage.
3. **Light mode only.** Every product in this category is light. Any suggestion
   moving toward dark is rejected.
4. `uv run pytest` stays green (102). `test_index_served` asserts the strings
   `Relay` and `REVENUE ENGINE` remain in the HTML — keep both.
5. Must not break at 1024px.
6. No LeanData logo, wordmark, or brand color as *identity*. Green survives only as
   a semantic state color (matched / live / won), which is shared design vocabulary,
   not their mark.

---

## 2. Information architecture

Nav mirrors LeanData's real left rail, with each item backed by a real endpoint.
Items that would be empty do not exist.

| Nav item | Rows | Source |
|---|---|---|
| **Routing** — Overview | live | `/api/stats`, `/api/audit`, `/api/leads`, `/api/desk` |
| Routing Queue | 3 | `/api/leads` + `/api/run` |
| Routing Insights | 8 reps | `/api/desk` |
| Audit Log | 365 | `/api/audit` |
| Round Robin | 4 territories / 8 reps | `/api/territories` + `/api/desk` |
| Territory Management | 4 | `/api/territories` + `/api/accounts` rollup |
| **Matching** — Accounts | 40 | `/api/accounts` |
| Contacts | 120 | `/api/contacts` |
| Opportunities | 80 | `/api/opportunities` |
| **Model** — Propensity v1 | 6 features | `/api/stats.model` |

**Landing (idle) = Routing Overview.** It must read as "the autonomous revenue engine,
running" in three seconds, so it is a cockpit, not a menu:

- Band 1 — engine strip: six real metrics (accounts, contacts, activities,
  open pipeline $, audit records, holdout AUC) plus a live clock chip showing the
  frozen demo as-of.
- Band 2 — split: left 2/3 is the **Live Decision Stream**, the dense Attio-craft
  table over all 365 audit records; right 1/3 is the **Routing Queue** (3 inbound
  leads, each actionable) above **Rep Desk** (8 reps with real routed/applied/skipped
  load bars).

This puts a genuinely dense table, a live stream, and a pending-work queue on the
landing screen at once. An idle console that shows 365 real decisions and 3 leads
awaiting a human is not a machine that is switched off.

**Record view (`?demo=lead_001` / `lead_002`)** follows `attio-record.png`:
attribute rail left, content right. The five-stage decision trace lives in the
right column as the differentiator, drawn as a real flow with labeled
**Match / No Match** edges in LeanData's grammar.

---

## 3. Design tokens

System fonts only (offline). `ui-sans-serif, -apple-system, "Segoe UI", Roboto,
"Helvetica Neue", Arial, sans-serif`; numerals `font-variant-numeric: tabular-nums`
everywhere a column of numbers appears. Mono: `ui-monospace, SFMono-Regular, Menlo,
Consolas, monospace` for sha256, IDs, timestamps.

### Neutrals
```
--bg          #FFFFFF   page + table surface
--bg-sunken   #F7F7F8   nav rail, toolbars, table header, footer
--bg-hover    #F4F4F5   row hover
--bg-active   #EEEEF0   selected row / active nav
--line        #E8E8EB   1px borders, column dividers
--line-soft   #F0F0F2   inner row separators
--ink         #17171A   primary text
--ink-2       #5C5C66   secondary text, column headers
--ink-3       #8E8E99   tertiary, timestamps, placeholders
```

### Accent — Relay indigo (not LeanData green)
```
--brand       #4F46E5
--brand-ink   #3730A3   text on tint
--brand-tint  #EEF0FF
```

### Semantic states (chips + flow nodes)
```
matched / live / won     --ok    #047857  tint #E7F6EF
pending / fallback / P2  --warn  #B45309  tint #FEF6E7
skip / lost / no-match   --bad   #BE123C  tint #FEECEF
P1 / hot                 --hot   #C2410C  tint #FFF0E8
neutral / open           --cool  #3F5BA9  tint #ECF1FB
```

Chip: `11px/1` semibold, `2px 7px`, radius `4px`, tint bg, `-ink` from its pair.
Never a saturated fill — Attio chips are tinted, not solid.

### Type scale
```
11px  micro: chips, column footers, edge labels        weight 600, tracking .01em
12px  meta: timestamps, sha256, secondary cell text    weight 450
13px  BODY — table cells, nav, attribute values        weight 450   ← density unit
14px  emphasis: attribute values, card labels          weight 500
16px  section headings                                 weight 600
20px  record title, view title                         weight 600, tracking -.01em
28px  card metric                                      weight 600, tracking -.02em
38px  engine-strip hero metric                         weight 650, tracking -.025em
```

### Density (the loss function is density, not whitespace)
```
table row height        36px       header row 38px    footer row 36px
cell padding            0 12px
nav item                28px, 6px gutter
card padding            14px 16px
section gap             20px
grid gutter             12px
radius                  6px cards, 4px chips, 8px panels
shadow                  0 1px 2px rgba(16,17,26,.05) — cards only, never on rows
```

Hard rule against dead air: **no card may have more than 24px of empty body below
its last row.** If a card would, it gets more real data or it gets removed.

---

## 4. Component contracts

**Table** (mirrors `attio-table.png`)
- Header cell = type icon + label, 12px `--ink-2`, `--bg-sunken`, sticky.
  Type icons are inline SVG, 12px: `Aa` text, `#` number, `$` currency, calendar
  date, tag enum, person, link.
- Sortable: click header toggles asc/desc with a caret; sorted column header goes
  `--ink`.
- Row 36px, `--line-soft` bottom border, `--bg-hover` on hover, pointer cursor
  when the row opens a record.
- First column carries a 16px identity square (monogram, deterministic tint from
  the name) + name in `--ink` 13px/500.
- Enum values render as chips. Money right-aligned tabular. Dates 12px `--ink-3`.
- **Footer row is required**: left cell `N count`, other cells show a real
  aggregate (`Σ $11,077,000`, `avg 17.7`) — never a decorative "+ Add calculation"
  with nothing behind it. Sticky to the bottom of the table viewport.

**Flow canvas** (mirrors `leandata-flowbuilder.png`)
- Light canvas `--bg-sunken` with a 24px dot grid.
- Node shapes carry meaning, as in the reference: **pentagon** = match,
  **diamond** = decision, **rounded rect** = action, **pill** = terminal.
- Node fill = the state tint, 1.5px border = the state ink. Small uppercase tag
  chip above each node (`MATCH`, `SCORE`, `ROUTE`) like LeanData's `DupeL` / `L2A`.
- Edges are 1.5px `--ink-3` paths with arrowheads and a **labeled** midpoint chip:
  `Match`, `No Match`, `Skip`, `Next`. Labels sit on a white pill so they stay
  legible over the grid.
- Header strip: `LIVE` badge, flow title, and real actions (Validate / Export
  audit) — never a fake Save & Deploy that does nothing.

**Record** (mirrors `attio-record.png`)
- Left rail 320px: identity block, then attribute rows `icon + label + value` at
  32px each. Labels 13px `--ink-2`, values 13px `--ink`.
- Right: tab strip with real counts, then **Highlights** — a 3-column card grid of
  real numbers (propensity, priority + SLA, open pipeline, activities 30d,
  contacts engaged, match confidence) — then the decision-trace flow, then the
  score contribution bars, then the reverse-chronological activity feed.

**Waterfall status** (from `clay-waterfall.png`)
- The five stages render as a sequential strip with per-stage status glyphs:
  ✓ resolved, ⚠ fallback used, ⤳ skipped. Each carries the real `rule_fired`
  string. This is where the governance story lives.

---

## 5. Data honesty rules for builders

- Bind every rendered value to a field from `/api/*`. If a design idea has no field
  behind it, cut the idea — do not invent the field.
- `fallback_used: true` must be visible, not hidden. It is the point of the demo:
  the LLM stage degraded to deterministic keyword rules and the system kept its
  guarantees.
- Money: `$11,077,000` full precision in footers, `$110.5K` compact in cells.
- Timestamps: render the frozen demo clock as-is; never `Date.now()`.
- The propensity score is `0.9351` — show 3 decimals, never round to "94%" in a
  place where the audit record says otherwise.

---

## 6. Build order

1. Shell: nav + topbar + view router (hash routes, no framework).
2. Table engine: one function that renders any book with column specs, sorting,
   chips, footer aggregates. Every book reuses it.
3. Routing Overview (landing).
4. Record view + decision-trace flow.
5. Remaining books (Accounts, Contacts, Opportunities, Audit, Territories,
   Round Robin, Model).
6. 1024px pass.

Serial only — `ui/index.html` is a single file and parallel builders would clobber
each other. Critics may run in parallel.
