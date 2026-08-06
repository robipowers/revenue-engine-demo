"""Relay — local dashboard server over the REAL pipeline. Stdlib only.

    python -m revenue_engine.ui        ->  http://127.0.0.1:5210

One page (ui/index.html) + a tiny JSON API. Every number the UI shows comes
from pipeline.run()'s real return and the audit records appended by THIS
run (captured as a line-count delta on runs/audit.jsonl). ValueError from
the pipeline (e.g. corrupt pending proposal, fail-closed) maps to HTTP 409
with a structured {"error": ...} body — tracebacks never reach the client.
The server writes nothing beyond what the pipeline itself writes.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from revenue_engine import audit, config, matching, model, pipeline, routing, warehouse
from revenue_engine.features import vectorize

HOST = "127.0.0.1"
PORT = 5210

# The HTML and the inbound lead files ship with the repo, next to src/ —
# they are code, not relocatable data, so they resolve from __file__ and
# deliberately ignore REVENUE_ENGINE_ROOT (which relocates data/model/runs).
REPO_ROOT = Path(__file__).resolve().parents[2]
UI_INDEX = REPO_ROOT / "ui" / "index.html"
INBOUND_DIR = REPO_ROOT / "data" / "inbound"

_LEAD_FIELDS = ("lead_id", "company", "contact_name", "title", "country", "state", "notes")


# ---------------------------------------------------------------------------
# Lead listing / resolution
# ---------------------------------------------------------------------------

def list_leads() -> list[dict]:
    leads = []
    for path in sorted(INBOUND_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        entry: dict = {"file": path.stem}
        for field in _LEAD_FIELDS:
            entry[field] = data.get(field)
        leads.append(entry)
    return leads


def resolve_lead_path(token: str) -> Path | None:
    """Map a lead token (file stem, filename, or lead_id) to an inbound file.

    Resolution is confined to data/inbound/*.json — arbitrary paths are
    never honored, so the API can't be steered at files outside the demo.
    """
    if not isinstance(token, str) or not token.strip():
        return None
    token = token.strip()
    stem = token[:-5] if token.endswith(".json") else token
    stem = Path(stem).name  # strip any directory components
    candidate = INBOUND_DIR / f"{stem}.json"
    if candidate.is_file():
        return candidate
    for path in sorted(INBOUND_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("lead_id") == token:
            return path
    return None


# ---------------------------------------------------------------------------
# Warehouse stats + per-account context (every number is a real query)
# ---------------------------------------------------------------------------

_MODEL_META: dict | None = None


def _model_meta() -> dict:
    """Model card metadata (name, holdout AUC, split sizes), loaded once."""
    global _MODEL_META
    if _MODEL_META is None:
        payload = model.load(model.MODEL_PATH)
        meta = {
            "name": "propensity_v1",
            "holdout_auc": payload["holdout_auc"],
            "n_train": payload["n_train"],
            "n_test": payload["n_test"],
            "n_features": len(payload["feature_names"]),
        }
        try:  # real trained logreg weights (std-scaled), for the score card
            logreg = payload["model"].named_steps["logreg"]
            meta["coefficients"] = dict(
                zip(payload["feature_names"], logreg.coef_[0].tolist())
            )
            meta["intercept"] = float(logreg.intercept_[0])
        except (AttributeError, KeyError, IndexError):
            pass
        _MODEL_META = meta
    return _MODEL_META


def warehouse_stats(runs_dir=None) -> dict:
    """Live warehouse totals for the idle-state header. All real counts."""
    con = warehouse.connect()
    try:
        counts = {
            t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("accounts", "contacts", "activities", "opportunities")
        }
        opp_mix = dict(
            con.execute(
                "SELECT coalesce(outcome, 'open'), count(*) FROM opportunities "
                "GROUP BY 1"
            ).fetchall()
        )
        territory_accounts = dict(
            con.execute(
                "SELECT territory, count(*) FROM accounts GROUP BY territory"
            ).fetchall()
        )
    finally:
        con.close()
    return {
        "warehouse": {
            **counts,
            "opps_won": opp_mix.get("won", 0),
            "opps_lost": opp_mix.get("lost", 0),
            "opps_open": opp_mix.get("open", 0),
        },
        "as_of": warehouse.demo_as_of(),
        "model": _model_meta(),
        "inbound_leads": len(list_leads()),
        "audit_records": len(_audit_lines(runs_dir)),
        "audit_tail": [
            {k: rec.get(k) for k in ("ts", "stage", "lead_id", "rule_fired",
                                     "fallback_used", "inputs_sha256")}
            for rec in _parse_records(_audit_lines(runs_dir)[-30:])
        ],
        "territory_accounts": territory_accounts,
        "territories": len(config.TERRITORIES),
        "reps": sum(len(r) for r in config.TERRITORIES.values()),
        "last_score": _last_score(runs_dir),
    }


def _last_score(runs_dir) -> dict | None:
    """Most recent real score record from this runs_dir's audit log, with
    its per-feature contributions — powers the idle score card."""
    for rec in reversed(_parse_records(_audit_lines(runs_dir))):
        if rec.get("stage") != "score":
            continue
        features = (rec.get("inputs") or {}).get("features")
        decision = rec.get("decision") or {}
        if not isinstance(features, dict) or "propensity_score" not in decision:
            continue
        return {
            "lead_id": rec.get("lead_id"),
            "ts": rec.get("ts"),
            "score": decision["propensity_score"],
            "features": features,
            "explain": score_explain(features),
        }
    return None


def account_context_payload(account_id: int) -> dict | None:
    """Full per-account context for the enrich card: named contacts, a real
    recent-activity feed (joined to contact names), open opps, and TRUE
    totals (not the display-capped counts inside get_account_context)."""
    con = warehouse.connect()
    try:
        ctx = warehouse.get_account_context(con, account_id)
        if ctx is None:
            return None
        as_of = warehouse.demo_as_of()
        # Feed is cut at the scoring clock so recency reads consistently with
        # the pipeline's as-of semantics (no "future" rows from trailing noise).
        activities = [
            {"type": r[0], "ts": r[1], "contact": r[2]}
            for r in con.execute(
                "SELECT a.type, a.ts, c.name FROM activities a "
                "JOIN contacts c ON c.id = a.contact_id "
                "WHERE a.account_id = ? AND a.ts <= CAST(? AS TIMESTAMP) "
                "ORDER BY a.ts DESC LIMIT 8",
                [account_id, as_of],
            ).fetchall()
        ]
        totals = {
            "activities_total": con.execute(
                "SELECT count(*) FROM activities WHERE account_id = ?", [account_id]
            ).fetchone()[0],
            "activities_30d": warehouse.activity_count_as_of(con, account_id, as_of),
            "contacts_engaged_30d": warehouse.distinct_contacts_as_of(con, account_id, as_of),
            "opps_total": con.execute(
                "SELECT count(*) FROM opportunities WHERE account_id = ?", [account_id]
            ).fetchone()[0],
            "opps_won": con.execute(
                "SELECT count(*) FROM opportunities WHERE account_id = ? AND outcome = 'won'",
                [account_id],
            ).fetchone()[0],
            "opps_lost": con.execute(
                "SELECT count(*) FROM opportunities WHERE account_id = ? AND outcome = 'lost'",
                [account_id],
            ).fetchone()[0],
        }
    finally:
        con.close()
    return {
        "account": ctx["account"],
        "contacts": [
            {"name": c["name"], "title": c["title"], "email": c["email"]}
            for c in ctx["contacts"]
        ],
        "recent_activities": activities,
        "open_opportunities": [
            {"stage": o["stage"], "amount": o["amount"], "created_at": o["created_at"]}
            for o in ctx["open_opportunities"]
        ],
        "totals": totals,
        "as_of": as_of,
    }


# ---------------------------------------------------------------------------
# Object books — the console's dense tables. Every column is a real column or
# a real aggregate; nothing here is synthesized to fill a row.
# ---------------------------------------------------------------------------

def accounts_book() -> dict:
    """All accounts with real rollups: engaged contacts, activity counts as of
    the demo clock, opportunity mix, and open pipeline dollars."""
    con = warehouse.connect()
    try:
        as_of = warehouse.demo_as_of()
        rows = con.execute(
            """
            SELECT a.id, a.name, a.domain, a.industry, a.employee_band,
                   a.country, a.state, a.territory,
                   (SELECT count(*) FROM contacts c WHERE c.account_id = a.id),
                   (SELECT count(*) FROM activities t WHERE t.account_id = a.id
                      AND t.ts <= CAST(? AS TIMESTAMP)
                      AND t.ts > CAST(? AS TIMESTAMP) - INTERVAL 30 DAY),
                   (SELECT count(*) FROM activities t WHERE t.account_id = a.id),
                   (SELECT max(t.ts) FROM activities t WHERE t.account_id = a.id
                      AND t.ts <= CAST(? AS TIMESTAMP)),
                   (SELECT count(*) FROM opportunities o WHERE o.account_id = a.id
                      AND o.outcome IS NULL),
                   (SELECT count(*) FROM opportunities o WHERE o.account_id = a.id
                      AND o.outcome = 'won'),
                   (SELECT count(*) FROM opportunities o WHERE o.account_id = a.id
                      AND o.outcome = 'lost'),
                   (SELECT coalesce(sum(o.amount), 0) FROM opportunities o
                      WHERE o.account_id = a.id AND o.outcome IS NULL),
                   (SELECT coalesce(sum(o.amount), 0) FROM opportunities o
                      WHERE o.account_id = a.id AND o.outcome = 'won')
            FROM accounts a ORDER BY a.name
            """,
            [as_of, as_of, as_of],
        ).fetchall()
    finally:
        con.close()
    cols = ("id", "name", "domain", "industry", "employee_band", "country",
            "state", "territory", "contacts", "activities_30d",
            "activities_total", "last_activity", "opps_open", "opps_won",
            "opps_lost", "open_amount", "won_amount")
    return {"accounts": [dict(zip(cols, r)) for r in rows], "as_of": as_of}


def contacts_book() -> dict:
    """Every contact, joined to its account, with real engagement counts."""
    con = warehouse.connect()
    try:
        as_of = warehouse.demo_as_of()
        rows = con.execute(
            """
            SELECT c.id, c.name, c.title, c.email, a.name, a.territory, a.industry,
                   (SELECT count(*) FROM activities t WHERE t.contact_id = c.id
                      AND t.ts <= CAST(? AS TIMESTAMP)),
                   (SELECT max(t.ts) FROM activities t WHERE t.contact_id = c.id
                      AND t.ts <= CAST(? AS TIMESTAMP))
            FROM contacts c JOIN accounts a ON a.id = c.account_id
            ORDER BY a.name, c.name
            """,
            [as_of, as_of],
        ).fetchall()
    finally:
        con.close()
    cols = ("id", "name", "title", "email", "account", "territory", "industry",
            "activities", "last_activity")
    return {"contacts": [dict(zip(cols, r)) for r in rows], "as_of": as_of}


def opportunities_book() -> dict:
    """Every opportunity with its account, stage, amount and real age in days."""
    con = warehouse.connect()
    try:
        as_of = warehouse.demo_as_of()
        rows = con.execute(
            """
            SELECT o.id, a.name, a.territory, a.industry, o.stage, o.amount,
                   o.outcome, o.created_at, o.closed_at,
                   date_diff('day', o.created_at,
                             coalesce(o.closed_at, CAST(? AS TIMESTAMP)))
            FROM opportunities o JOIN accounts a ON a.id = o.account_id
            ORDER BY o.amount DESC
            """,
            [as_of],
        ).fetchall()
    finally:
        con.close()
    cols = ("id", "account", "territory", "industry", "stage", "amount",
            "outcome", "created_at", "closed_at", "age_days")
    return {"opportunities": [dict(zip(cols, r)) for r in rows], "as_of": as_of}


def audit_book(runs_dir=None) -> dict:
    """The full append-only decision log for this runs_dir, newest first.

    Returned verbatim from disk (no re-derivation) so the table on screen is
    literally the governance artifact an auditor would read.
    """
    records = _parse_records(_audit_lines(runs_dir))
    stages: dict[str, int] = {}
    rules: dict[str, int] = {}
    fallbacks = 0
    for rec in records:
        stages[rec.get("stage") or "?"] = stages.get(rec.get("stage") or "?", 0) + 1
        rules[rec.get("rule_fired") or "?"] = rules.get(rec.get("rule_fired") or "?", 0) + 1
        if rec.get("fallback_used"):
            fallbacks += 1
    return {
        "records": list(reversed(records)),
        "total": len(records),
        "stages": stages,
        "rules": rules,
        "fallbacks": fallbacks,
    }


def _read_rotation(runs_dir) -> dict:
    """Rotation positions straight off runs/routing_state.json (read-only —
    unlike routing._load_state this never appends an audit record)."""
    path = (Path(runs_dir) if runs_dir else config.RUNS_DIR) / routing.STATE_FILENAME
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, int)}


def desk_state(runs_dir=None) -> dict:
    """Per-rep load derived from this runs_dir's REAL audit history:
    routed proposals, applied assignments, skips, and last-touch times."""
    reps: dict[str, dict] = {}

    def bucket(name: str) -> dict:
        return reps.setdefault(
            name, {"routed": 0, "applied": 0, "skipped": 0, "last_ts": None}
        )

    for rec in _parse_records(_audit_lines(runs_dir)):
        stage, decision = rec.get("stage"), rec.get("decision") or {}
        if stage == "routing" and decision.get("rep"):
            b = bucket(decision["rep"])
            b["routed"] += 1
            b["last_ts"] = rec.get("ts")
        elif stage == "assignment_applied" and decision.get("rep"):
            b = bucket(decision["rep"])
            b["applied"] += 1
            b["last_ts"] = rec.get("ts")
        elif stage == "routing_skip" and decision.get("skipped_rep"):
            bucket(decision["skipped_rep"])["skipped"] += 1
    return {"rotation": _read_rotation(runs_dir), "reps": reps}


# ---------------------------------------------------------------------------
# Audit delta capture
# ---------------------------------------------------------------------------

def _audit_lines(runs_dir) -> list[str]:
    path = audit.audit_path(runs_dir)
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line.strip()]


def _parse_records(lines: list[str]) -> list[dict]:
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


# ---------------------------------------------------------------------------
# Score explainability (real coefficients x real features, nothing invented)
# ---------------------------------------------------------------------------

_MODEL_PAYLOAD: dict | None = None


def _model_payload() -> dict:
    global _MODEL_PAYLOAD
    if _MODEL_PAYLOAD is None:
        _MODEL_PAYLOAD = model.load()
    return _MODEL_PAYLOAD


def score_explain(features: dict) -> list[dict] | None:
    """Per-feature contribution to the logit for one real feature dict.

    contribution_i = coef_i * (x_i - mean_i) / scale_i — exactly the terms
    the trained StandardScaler+LogisticRegression pipeline sums before the
    sigmoid. Sorted by |contribution| so the dominant factor is first.
    Returns None on any failure rather than guessing.
    """
    try:
        payload = _model_payload()
        pipe = payload["model"]
        scaler = pipe.named_steps["scaler"]
        logreg = pipe.named_steps["logreg"]
        names = payload["feature_names"]
        x = vectorize(features)
        rows = []
        for i, name in enumerate(names):
            z = (x[i] - float(scaler.mean_[i])) / float(scaler.scale_[i])
            rows.append({
                "feature": name,
                "value": x[i],
                "raw": features.get(name),
                "contribution": round(float(logreg.coef_[0][i]) * z, 4),
            })
        rows.sort(key=lambda r: -abs(r["contribution"]))
        return rows
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class RelayHandler(BaseHTTPRequestHandler):
    server_version = "RelayUI/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing -----------------------------------------------------------

    @property
    def runs_dir(self):
        return getattr(self.server, "runs_dir", None)

    def log_message(self, fmt, *args):  # keep the terminal clean
        pass

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > 1_000_000:
            return None
        try:
            data = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    # -- routes -------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_index()
        elif path == "/api/leads":
            self._send_json(200, {"leads": list_leads()})
        elif path == "/api/territories":
            self._send_json(200, {
                "territories": config.TERRITORIES,
                "default_territory": config.DEFAULT_TERRITORY,
                "p1_threshold": routing.P1_THRESHOLD,
                "sla_minutes": routing.SLA_MINUTES,
                "country_map": config.COUNTRY_TO_TERRITORY,
                "us_east_states": sorted(config.US_EAST_STATES),
                "match_auto": matching.AUTO_MATCH_THRESHOLD,
                "match_ambiguous": matching.AMBIGUOUS_THRESHOLD,
            })
        elif path == "/api/stats":
            self._send_json(200, warehouse_stats(self.runs_dir))
        elif path == "/api/desk":
            self._send_json(200, desk_state(self.runs_dir))
        elif path == "/api/accounts":
            self._send_json(200, accounts_book())
        elif path == "/api/contacts":
            self._send_json(200, contacts_book())
        elif path == "/api/opportunities":
            self._send_json(200, opportunities_book())
        elif path == "/api/audit":
            self._send_json(200, audit_book(self.runs_dir))
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._send_json(404, {"error": f"not found: {path}"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        body = self._read_body()
        if body is None:
            self._send_json(400, {"error": "expected a JSON object body"})
            return
        if path == "/api/run":
            self._handle_run(body, approve=None, fresh=bool(body.get("fresh")))
        elif path == "/api/approve":
            decision = body.get("decision")
            if decision not in ("approve", "reject"):
                self._send_json(400, {"error": 'decision must be "approve" or "reject"'})
                return
            self._handle_run(body, approve=(decision == "approve"), fresh=False)
        else:
            self._send_json(404, {"error": f"not found: {path}"})

    def _serve_index(self):
        try:
            html = UI_INDEX.read_bytes()
        except OSError:
            self._send_json(500, {"error": "ui/index.html missing"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(html)

    # -- pipeline calls -----------------------------------------------------

    def _handle_run(self, body: dict, approve: bool | None, fresh: bool) -> None:
        lead_path = resolve_lead_path(body.get("lead"))
        if lead_path is None:
            self._send_json(404, {"error": f"unknown lead: {body.get('lead')!r}"})
            return
        if fresh:
            routing.reset_state(self.runs_dir)
        before = len(_audit_lines(self.runs_dir))
        try:
            result = pipeline.run(lead_path, approve=approve, runs_dir=self.runs_dir)
        except ValueError as exc:
            records = _parse_records(_audit_lines(self.runs_dir)[before:])
            self._send_json(409, {"error": str(exc), "records": records})
            return
        except Exception as exc:  # structured error, never a traceback
            self._send_json(500, {"error": f"internal error: {type(exc).__name__}"})
            return
        records = _parse_records(_audit_lines(self.runs_dir)[before:])
        payload: dict = {
            "lead_id": result["lead_id"],
            "status": result.get("status"),
            "stages": result.get("stages", []),
            "records": records,
            "fresh": fresh,
        }
        if "proposal" in result:
            payload["proposal"] = result["proposal"]
        if result.get("idempotent_replay"):
            payload["idempotent_replay"] = True
        for rec in records:  # real contributions for the score card readout
            if rec.get("stage") == "score":
                features = (rec.get("inputs") or {}).get("features")
                if isinstance(features, dict):
                    payload["score_explain"] = score_explain(features)
                break
        resolved = result.get("resolved_account")
        if resolved and resolved.get("account_id") is not None:
            payload["account"] = account_context_payload(resolved["account_id"])
        self._send_json(200, payload)


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

def make_server(port: int = PORT, runs_dir=None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((HOST, port), RelayHandler)
    server.runs_dir = runs_dir
    return server


def main() -> None:
    server = make_server()
    print(f"Relay — Revenue Engine console: http://{HOST}:{server.server_address[1]}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
