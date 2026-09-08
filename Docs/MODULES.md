# MODULES.md — TicketSolver Build Order

Sequential breakdown of the platform defined in [`SRS_Enterprise_Support_Agent_Platform.md`](SRS_Enterprise_Support_Agent_Platform.md)
("the SRS") into modules that are built **one at a time**: implement fully → write and run
that module's tests → report actual results → wait for approval before the next module.

Conventions:
- **Status** per module reflects reality as of 2026-08-25. Modules 0–6 are done; their test
  lists contain only tests that were actually written and run.
- Every module has a Testing section — none is "too simple to test." Genuinely trivial
  checks say so explicitly.
- A coverage matrix at the bottom maps every FR and NFR to at least one module.

---

## Module 0 — Repo Scaffold & Governance ✅ DONE

**Goal:** Establish the repo layout, tooling decisions, and dev infrastructure so every
later module lands in a consistent structure.

**Scope (built):**
- Governance docs: `CLAUDE.md` (locked stack + conventions), `rules.md` (coding-agent rules),
  root/backend/frontend `README.md`s, `.gitignore`
- Backend skeleton (`backend/app/`): FastAPI entrypoint with CORS, `/api/health`, stub
  routers, WebSocket connection-manager stub, pydantic-settings config, logging setup with
  PII-redaction placeholder, lazy SQLAlchemy/Qdrant/NIM client factories
- Frontend skeleton (`frontend/src/`): React + TypeScript + Vite, four dashboard panel
  placeholders, react-three-fiber `TraceViewer` canvas stub, WS hook, shared types, REST client
- Dev infra: `backend/docker-compose.yml` (Postgres :5433, Qdrant :6333), Python 3.13 venv,
  declared dependencies installed (no torch/sentence-transformers by decision)

**Out of scope:** All agent logic, ML, real endpoints (stubs returned 501), frontend data wiring.

**Dependencies:** None.

**SRS refs:** §7 technology stack (structure mirrors it), §2.3 ($0 free-tier constraint),
NFR Cost (dev containers stand in for free-tier services).

**Testing (actual):**
- *Unit:* `tests/test_health.py::test_health_ok` — app boots, `GET /api/health` → 200
  `{"status": "ok", ...}` (FastAPI TestClient). This is deliberately minimal: scaffolding's
  only behavior IS booting and routing.
- *Manual:* `docker compose up -d` then `docker ps` showed both containers healthy;
  `python -m compileall app tests` clean during scaffold; `npm install && npm run build` succeeded.

**Definition of done:** ✅ Met — health test passing, both containers healthy, backend
compiles, frontend builds, `.env` files confirmed gitignored.

---

## Module 1 — Investigator/Resolver Agent + KB RAG + Trace Persistence ✅ DONE
*(real NIM LLM call still pending user API key — see Testing)*

**Goal:** One agent that takes a ticket, gathers real evidence (KB search over Qdrant,
account/order lookups) via tool calls, calls NVIDIA NIM for reasoning, and produces a
proposed resolution — with every step traced to Postgres.

**Scope (built):**
- `app/tools/`: simulated `lookup_account` / `get_order_status` (deterministic in-memory stores)
- `app/vector/kb.py`: Ollama `nomic-embed-text` HTTP embedding client (timed/logged),
  Qdrant collection management, upsert + semantic search; `scripts/ingest_kb.py` (idempotent,
  5 self-authored docs in `backend/kb/` → 38 chunks @ dim 768)
- `app/llm/nim_client.py`: OpenAI-compatible NIM chat helper (timed/logged, token usage)
- `app/agents/investigator.py`: LangGraph bind-tools loop — `investigator` node →
  conditional edge → custom traced tool executor → loop; `finalize` persists proposal;
  `run_investigation()` entry point
- Trace persistence: `Ticket` + `TraceStep` ORM models, `app/core/tracing.py::TraceWriter`
  (ordered steps w/ input/output/reasoning/model/latency)
- APIs: `POST /api/tickets`, `GET /api/tickets/{id}`, `POST /api/tickets/{id}/investigate`,
  `GET /api/traces/{id}`; scripts `init_db.py`, `run_investigator.py`

**Out of scope:** Orchestrator/routing, Critic, ML classifier/router/calibration,
guardrail enforcement, HITL queue, long-term & episodic memory, frontend wiring.

**Dependencies:** Module 0.

**SRS refs:** FR-2 (investigate via tools before proposing), FR-10 (every step logged),
NFR Auditability (actions traceable to reasoning chain).

**Testing (actual):**
- *Unit — 15 tests, all passing:*
  - `tests/test_tools.py` (7): account found w/ field contract · unknown→None · copy
    semantics (×2 for orders) · `delivered_at=None` handling · seed-data invariants
    (amount >$50 and <$50 present, `lost`/`delayed` statuses present, order↔customer consistency)
  - `tests/test_tracing.py` (3): seq ordering 1→2 with distinct ids · full-field round-trip
    incl. JSON payloads + ISO timestamps · `model`/`latency_ms` persisted (SQLite in-memory)
  - `tests/test_investigator_unit.py` (4): scripted-LLM loop ends in proposal + traces +
    DB persist · unknown tool degrades gracefully · tool exception caught AND traced ·
    TOOL_SCHEMAS ↔ registry consistency
- *Integration (live stack, run this session):* `init_db.py` created `tickets`,
  `trace_steps` in Postgres container; `ingest_kb.py` embedded 38 chunks via real Ollama
  into Qdrant (dim verified 768 per batch; upserts 35–56 ms); TestClient round-trip vs real
  Postgres: create 201 / get 200 / empty-trace 404
- *Eval/quality:* retrieval spot-checks — "I was charged twice" → refund-policy *Duplicate
  Charges* (0.62); "refund over $50 approval" → *Supervisor Approval Requirement* (0.78)
- *Manual:* `npm run build` green; embedding/query latency logs reviewed (cold embed 22.7 s,
  ~2.7 s warm; query-time search ≈4.6 s total)
- ⚠️ **Not yet exercised:** a real NIM completion. The LLM is mocked in unit tests by design;
  the live path runs via `python scripts/run_investigator.py --demo` once `NIM_API_KEY` is set.

**Definition of done:** ✅ Met for everything except the live NIM call above, which closes
the moment the key is configured and the demo command is run once successfully.

---

## Module 2 — Long-Term Customer Memory (Postgres) ✅ DONE
*(real NIM LLM call still pending user API key — same caveat as Module 1)*

**Goal:** Persist per-customer facts (plan tier, account age, notes) and load them into the
agent's context at session start so resolutions respect customer history.

**Scope (built):**
- `CustomerMemory` ORM model (`customer_memory`: `customer_id` PK, `plan_tier`,
  `account_age_days`, ordered `notes` JSON, `updated_at`); schema via existing
  `create_all` path (`init_db.py` created all three tables in the Postgres container).
  Alembic deferred — M2's scope allowed either; introduce when schema churn settles.
- `app/memory/long_term.py`: `CustomerMemoryRecord` dataclass,
  `get_customer_memory(customer_id)` / `upsert_customer_memory(record)` (both accept an
  optional SQLite-capable `session_factory` for tests), pure `format_customer_memory()`
  renderer (empty/missing → "" graceful omission; populated → fixed fact order)
- Loader wired into `run_investigation`: facts prepended to the system prompt; a
  `memory_retrieval` step is traced first (seq 1, found/facts_block); `llm_call` trace rows
  now record the assembled system prompt so context is auditable; loader failure degrades to
  "no facts" instead of breaking the run
- `scripts/seed_customer_memory.py` (`--verify` round-trip): idempotent upserts for
  CUST-1001..1004 consistent with `app/tools/accounts.py`

**Out of scope:** Episodic incident memory (Module 3); conversation-scoped short-term
memory (Module 4); any UI.

**Dependencies:** Modules 0, 1.

**SRS refs:** FR-6.

**Testing (actual):**
- *Unit — 9 new tests, suite 24/24 passing:*
  - `tests/test_long_term.py` formatter (4): None→"" · all-fields-unset→"" · populated
    renders header+ordered facts (tier < age < notes, stored note order) · partial record
    includes only present fields
  - persistence on in-memory SQLite (2): upsert→get dataclass-equality round-trip ·
    second upsert updates in place (count=1), unknown id → None
  - investigator wiring (3): seeded facts reach the system prompt AND the trace
    (`memory_retrieval` at seq 1 w/ input/output payloads; `llm_call.input_payload.system_prompt`
    contains the block) · absent memory omits block, traces `found=false` · raising loader
    degrades without breaking the flow
- *Integration (live stack, run this session):* `init_db.py` created `customer_memory` in
  the Postgres container alongside `tickets`, `trace_steps`; `seed_customer_memory.py
  --verify` upserted 4 customers and read back all fields identical against real Postgres;
  scripted-LLM live run for CUST-1003 produced trace seq `1 memory_retrieval → 2 llm_call →
  3 decision` with the enterprise-tier facts asserted present in the traced LLM input payload
- ⚠️ **Not yet exercised:** a real NIM completion consuming the enriched prompt (same M1
  caveat — closes when `NIM_API_KEY` is set and a demo is run)

**Definition of done:** ✅ Met for everything except the live NIM call above, which closes
the moment the key is configured and the demo command is run once successfully.

---

## Module 3 — Episodic Memory (Qdrant) ✅ DONE
*(real NIM LLM call still pending user API key — same caveat as M1/M2)*

**Goal:** Store past resolved incidents as embeddings and surface similar past cases before
the agent proposes anything, closing the loop by writing back new resolutions.

**Scope (built):**
- Dedicated Qdrant collection `episodes` (cosine, dim 768) separate from `kb_chunks`;
  deterministic point ids (`uuid5` of ticket id) so re-recording replaces cleanly
- `app/memory/episodic.py`: pure payload ser/de (`episode_to_payload`/`episode_from_payload`),
  prompt renderer `format_similar_incidents` ("" when nothing found), `record_episode(ticket,
  resolution)` (ORM row or dict), `find_similar_incidents(text, k, exclude_ticket_id=…)`
  (over-fetch + filter keeps a ticket's own episode out of its context), `count_episodes()`
- Investigator hooks: pre-investigation retrieval of subject+body → "SIMILAR PAST INCIDENTS"
  block in the system prompt + `memory_retrieval:episodic_memory` trace step (seq 2); after a
  non-empty proposal, `record_episode` write-back + `memory_writeback:episode_writeback`
  trace step; both degrade gracefully when Qdrant/Ollama are down
- `scripts/seed_episodes.py` (`--verify`): 7 synthetic resolved incidents (lost kit, duplicate
  charge, >$50 refund escalation, MFA lockout, damaged parts, cancellation w/ prorated refund,
   delayed shipment) + 7 curated query→expected pairs rubric

**Out of scope:** Knowledge-base RAG (done in M1); reviewer-decision semantics (M5).

**Dependencies:** Modules 0, 1 (reuses vector/embedding layer).

**SRS refs:** FR-7; FR-9 (write-back half — shared with M5, flagged in matrix).

**Testing (actual):**
- *Unit — 10 new tests, suite 34/34 passing* (`tests/test_episodic.py`): payload round-trip +
  normalization/truncation/unknown-key drop · formatter empty→"" / ordered cases with scores ·
  deterministic point ids · investigator wiring with scripted LLM: hits reach system prompt AND
  trace (queried with subject+body, self-excluded), proposal written back + traced, empty
  proposal skips write-back, retrieval-failure and write-back-failure both leave flow intact
- *Bug found & fixed:* `finalize_node`'s fallback scan accepted any message without tool_calls,
  so an empty AI answer made the **human message** become the "proposal" (and get written back).
  Now restricted to non-empty `AIMessage`s.
- *Integration (live Qdrant+Ollama, this session):* seeded 7 episodes (dim verified 768);
  `--verify` curated rubric **7/7 retrieved within top-3** (scores 0.71–0.84)
- *Manual:* scripted-LLM demo for CUST-1001 traced `1 customer_memory → 2 episodic_memory
  (T-SEED-001 @ 0.823) → 3 llm_call → 4 decision → 5 memory_writeback`; collection count grew
  7 → 8 after the resolved run
- ⚠️ **Not yet exercised:** a real NIM completion consuming the enriched context (same M1/M2
  caveat)

**Definition of done:** ✅ Met for everything except the live NIM call above.

---

## Module 4 — Orchestrator + Reliability Layer ✅ DONE
*(real NIM LLM call still pending user API key — same caveat as M1–M3)*

**Goal:** Replace the direct `run_investigation` entry point with a stateful LangGraph
orchestrator that decides resolve / retry / escalate, processes tickets asynchronously, and
hardens every external call with retry/timeout/circuit-breaker.

**Scope (built):**
- `app/core/resilience.py`: pure `CircuitBreaker` (closed→open→half-open, injectable clock),
  `backoff_delay` (exponential, capped), `Resilience` facade (bounded retries + thread-pool
  timeout + optional breaker; breaker-open is never retried); settings-driven production
  policies for tools & LLM (`get_tool_policy`/`get_llm_policy`, `reset_policies()` for tests)
- `app/orchestration/graph.py`: LangGraph orchestrator `investigate → decide → retry → … /
  resolve | escalate`; pure transition table `decide_next` (bounded retry-on-low-confidence,
  empty proposals never resolve) with escalation as a terminal safety state; confidence from
  the investigator's new `CONFIDENCE:` meta-line (self-reported seam — calibration replaces
  it in M7); decision traces + ticket `status` (`auto_resolved`/`escalated`) on every terminal
- FR-5 short-term memory: in-memory per-ticket session turns + `follow_up()`; turns ride into
  every investigation as context; retry nudges accumulate separately via an `operator.add`
  channel
- Async processing: `POST /api/tickets/{id}/investigate` → 202 + job snapshot;
  `GET .../investigate` status endpoint; ThreadPool worker broadcasts
  `ticket_processed`/`ticket_failed` over WS via the startup-captured event loop
  (`broadcast_from_thread` hops threads safely)
- Investigator hardening: `_call_model` runs under the LLM policy, tool execution under the
  tool policy (`_execute_tool` seam)

**Out of scope:** Escalation *policy thresholds* driven by calibration (M7 refines what's
here); guardrails (M5); ML classification feeding the routing table (M6 — confidence is the
temporary LLM-self-report seam until then, noted inline).

**Dependencies:** Modules 0–3.

**SRS refs:** FR-4 (not a fixed pipeline), FR-5 (short-term in-context memory),
NFR Reliability (retry/timeout/circuit-breaker), NFR Scalability (async, concurrent sessions).

**Testing (actual):**
- *Unit — 20 new tests, suite 54/54 passing:*
  - `tests/test_resilience.py` (11): backoff exponential+cap+negative-reject · breaker opens
    after threshold, success resets streak, half-open single-probe semantics (refuse before
    timeout, one probe granted, failed probe re-opens, successful probe closes), threshold
    validation · Resilience first-success, retry-until-success w/ recorded backoff sequence,
    original-error exhaustion, CircuitOpenError never retried/never invokes fn, real timeout
    (<0.7s for a 1s call, late finisher doesn't crash pool), success/failure recorded into breaker
  - `tests/test_orchestrator_unit.py` (9): full transition table incl. boundary-at-threshold,
    overrides · confidence parse (case-tolerant, clamped, non-numeric garbage → default) +
    strip · scripted-graph flows: confident→resolved attempt 1 (status + clean proposal w/o
    meta-line + trace), low-confidence×2→escalated (retry nudge reached 2nd investigation,
    bounded attempts, decision trace order retry→escalate) · FR-5 session turns reach the
    investigator; follow_up accumulates customer+agent turns
- *Integration (live stack, this session):* forced failing tool → real policy: 3 attempts,
  breaker opened, second attempt's tool call short-circuited instantly, ticket **escalated**
  (not hung/crashed) with breaker left open; async submit returned 202 and both WS
  `ticket_processed` events arrived; two concurrent submissions resolved with ordered traces,
  consistent statuses, and M3 write-back still firing
- *Bug found & fixed:* job enqueue returned the live Job object — the worker could flip it to
  `running` before the 202 serialized; enqueue/get now return snapshots
- ⚠️ **Not yet exercised:** a real NIM completion under the LLM resilience policy (same
  M1–M3 caveat)

**Definition of done:** ✅ Met — unit + integration tests pass; all pre-existing suites green
(investigator unchanged where tests patch `_call_model`); forced-failure demo escalated.

---

## Module 5 — Guardrails + Human-in-the-Loop Approval Queue ✅ DONE
*(real NIM LLM call still pending user API key — same caveat as M1–M4)*

**Goal:** Hard-coded policy checks enforced in application code that block risky actions
(refunds over $50) from autonomous execution and route them to a reviewable approval queue
with full traces; human decisions are logged and fed back to memory.

**Scope (built):**
- `app/guardrails/engine.py`: `parse_proposed_action` (kind + amount extraction; a refund
  verb always wins over other action verbs so money never slips past the rule),
  `check_action` declarative rules — `$50.00` limit pinned as `REFUND_APPROVAL_LIMIT_USD`
  in code (FR-3: never prompt-driven); refund without a parseable amount blocks
  conservatively; unrecognized action kinds are default-denied; `check_proposal` combo
- Enforcement point inside the orchestrator: a `guardrail` node sits between
  decide-resolve and execution; allowed → resolve path, blocked → persisted `Approval`
  row + `approval_created` WS broadcast + `await_review` terminal node (ticket status
  `pending_approval`, with a skip-if-decided guard so the winding-down run can never
  clobber an already-recorded reviewer outcome)
- HITL queue APIs (`app/api/routes/approvals.py`): list queue (`?status=pending|all`),
  detail embedding the ticket's full trace, decision endpoint approve/reject/correct
  (corrected requires `corrected_action`; second decision on a decided item → 409);
  decisions traced as `human_review`; ticket outcome updated (approved/corrected →
  `resolved` with the corrected action overwriting the proposal, rejected → `rejected`)
- FR-9 write-back: human-approved resolutions recorded into episodic memory tagged
  `[human-approved]`. **FR-7 semantics corrected while wiring this up:** episodic
  recording now happens only on terminal resolution — the orchestrator's resolve node for
  auto-resolutions (traced `memory_writeback`), the approval endpoint for human ones.
  Previously every proposal was recorded at investigation time, which mislabeled
  guardrail-blocked/rejected cases as "resolved incidents" and double-wrote approved
  tickets onto the same deterministic point id.

**Out of scope:** Review UI (M9/M10 consume these APIs); calibration-driven thresholds (M7).

**Dependencies:** Modules 0–4 (write-back needs M3; enforcement wraps M4 flow).

**SRS refs:** FR-3, FR-8, FR-9 (decision logging + write-back); NFR Auditability.

**Testing (actual):**
- *Unit — suite 67/67 passing, ruff clean across app+tests:*
  - `tests/test_guardrails.py` (14): parse refund amounts ($ / USD / comma-thousands),
    non-refund kinds, unrecognized → unknown, refund-verb precedence ("reship AND refund"
    still hits the money rule) · boundaries: $49.99 allows, $50.00 blocks (constant
    itself pinned), above blocks · refund-without-amount blocks conservatively · unknown
    kinds default-deny · graph-level flows: blocked $149 → `pending_approval` + exactly
    one pending Approval row (kind/amount/reason) with `guardrail_block` trace and no
    auto-resolution and no episodic write; allowed $12.99 resolves autonomously with no
    approval row and exactly one episodic write; low-confidence escalates without
    touching approvals or memory
  - `tests/test_orchestrator_unit.py` (8): prior transition-table/confidence coverage
    plus resolve-node write-back asserted once-per-resolution with its
    `memory_writeback` trace; escalation records nothing; a Qdrant outage during
    resolution does not fail the run
  - `tests/test_episodic.py` (8): investigator-level assertion inverted to lock the new
    boundary — investigation alone never writes back
- *Integration (live stack):* two concurrent >$50-refund tickets → both
  `approval_created` broadcasts ($149.00) once jobs settled; queue detail embeds the
  full trace incl. `guardrail_block`; approve+reject persist `resolved`/`rejected`
  statuses with `human_review` traces; duplicate decision → 409; episode count grew by
  exactly 1 (the human-approved resolution only). A separate small-refund ticket
  auto-resolved with exactly one traced episode write.
- *Tooling:* ruff 0.16 defaults applied repo-wide — route dependencies moved to the
  `Annotated[..., Depends(...)]` style, unused imports/variables removed,
  `datetime.UTC` alias adopted; compileall clean.

**Bugs found & fixed:**
1. *Reviewer race:* `approval_created` fired before `await_review` stamped
   `pending_approval`, so a fast reviewer's committed decision was overwritten back by
   the still-finishing worker run. Fixed with the skip-if-decided guard on status writes.
2. *FR-7 scope defect* (surfaced by live count checks): episodic memory received every
   proposal at investigation time. Moved write-back to terminal resolution paths;
   tests updated to pin the boundary at both levels.
3. *Test-harness fix:* live-check mapped queue position → ticket id, but parallel
   workers persist approvals in nondeterministic order; checks now key decisions by
   `approval.ticket_id`.

**Definition of done:** ✅ Met — tests pass; a blocked-action demo lands in the queue with
its full trace attached; no code path can execute a blocked action (enforcement covered by
graph-level tests); decisions are logged, traced, update outcomes, and feed memory.

---

## Module 6 — Data Pipeline + Ticket Classifier (ML #1) ✅ DONE
*(synthetic edge-case generation deferred until a NIM API key exists — same standing caveat as M1–M5)*

**Goal:** Train and serve the XGBoost category/severity classifier that replaces LLM triage,
plus the data pipeline that feeds it.

**Scope (built):**
- Dataset acquisition with license verification **before use**: primary dataset is
  Tobi-Bueck `customer-support-tickets` (Hugging Face, CC BY-NC 4.0 verified via API tags;
  EN subset only) — provenance, license record, and schema mapping in `Docs/DATA.md`.
  The initially-picked Kaggle `suraj520` dataset was evaluated first and **rejected on
  evidence**: its labels are statistically independent of ticket text (chance-level F1 for
  two model families), documented in `Docs/CLASSIFIER_METRICS.md` Appendix A.
- `scripts/prepare_data.py`: download (optional `--download`), EN-row filter, null-drop
  cleaning, label normalization through `app.ml.features`, stratified 70/15/15 split on the
  joint category|severity key (seed 42). 13,731 usable rows → train 9,611 / val 2,060 /
  test 2,060. Raw + processed CSVs gitignored (`backend/data/`).
- Feature engineering + models in `app/ml/features.py` (canonical label vocabularies,
  normalizers, `combine_text`, shared TF-IDF pipeline) and `app/ml/classifier.py` (two
  XGBoost heads with integer label encoding for stable proba order, one joblib bundle,
  memoized production loader, frozen `TicketPrediction`). Artifacts committed under
  `backend/models/` with machine-readable held-out metrics (`metrics.json`).
- Inference service hooked ahead of the orchestrator's routing table: `process_ticket`
  runs `_triage_ticket` before graph invocation; predictions persist ONLY onto
  `tickets.category`/`tickets.severity` (own session; never touches graph-owned columns)
  and trace as `ml_prediction/ticket_classification` ahead of every decision step, with
  latency. Fully best-effort: missing bundle, load failure, or storage error → logged and
  skipped, orchestration unchanged (pre-M6 behavior preserved).
- API exposure: `TicketOut` serves `category`/`severity`.

**Out of scope:** Complexity/cost router (M7); confidence calibration (M7); clustering (M12);
NIM synthetic edge-case rows (deferred with the standing key caveat — the pipeline accepts
an augmented CSV later without code changes).

**Dependencies:** Modules 0–5 (routing seam from M4).

**SRS refs:** FR-1; §8 (datasets); §6.1 row 1.

**Testing (actual):**
- *Unit — suite 95/95 passing, ruff clean across app+tests+scripts, compileall clean:*
  - `tests/test_ml_features.py` (12): normalization of every valid category/severity label
    incl. case/space variants · unknown labels raise ValueError listing valid ones ·
    idempotency · `combine_text` determinism/order/strip semantics · TF-IDF pipeline shape,
    determinism, bigram presence, fit/transform consistency.
  - `tests/test_ml_classifier.py` (10): synthetic keyword-separable corpus trains a tiny
    model to >0.8 macro-F1 (proves the pipeline learns when signal exists); bundle
    structure/metadata; metrics dict contract; load/predict roundtrip through the exact
    serving path; prediction determinism; non-string input rejection; missing-bundle error
    message names path+remedy; per-directory memoization identity.
  - `tests/test_orchestrator_unit.py` (+3): triage persists columns + traces ahead of all
    decision steps with untouched outcome fields; unavailable model → NULLs, no trace,
    unchanged flow; crashing seam swallowed without breaking orchestration.
  - `tests/test_ml_integration.py` (2): full API path — POST → investigate → GET serves
    predicted category/severity with `ml_prediction` as trace seq 1; absent model serves
    `category=null` cleanly.
- *Eval/quality (primary gate) — real trained artifact, held-out test split, production
  serving path:* **category macro-F1 0.8501 (gate ≥0.60 PASS)**, per-class F1 0.62–0.98;
  **severity macro-F1 0.5899 (gate ≥0.40 PASS)**. Full tables + interpretation in
  `Docs/CLASSIFIER_METRICS.md`; gates recalibrated to the new dataset's chance floor when
  the dataset switched (rationale inline).
- *Live check:* sample tickets classified through `load_classifier()` +
  orchestrator `_classify_ticket` against the committed bundle (sensible labels with
  calibrated-feeling low confidences on ambiguous text).
- *Tooling fix surfaced by the suite:* `test_ml_integration.py` moved from in-memory
  StaticPool SQLite to file-backed SQLite with WAL + busy_timeout — the job worker persists
  triage on its own thread while requests poll, and a single shared StaticPool connection
  let those transactions interleave (observed once as a silently lost write). Other suites
  stay single-threaded StaticPool by design.

**Bugs found & fixed:**
1. *Dataset signal failure (the big one):* suraj520 labels are noise w.r.t. text — caught
   by honest gate reporting instead of being shipped as a "working" classifier. Retargeted
   to a signal-bearing, license-verified dataset; negative finding preserved as evidence.
2. *FR-7/M6 interaction guard:* triage persistence deliberately touches only
   category/severity so it can never clobber graph-owned status/proposal columns.

**Definition of done:** ✅ Met — pipeline reproducible from committed scripts (commands in
`Docs/DATA.md`); metrics report committed with both F1 gates passed honestly; classifier
serving in front of the orchestrator with predictions persisted and traced.

---

## Module 7 — Cost Router + Confidence Calibration (ML #2 & #3) ✅ DONE

**Goal:** Route tickets to the cheap vs. strong NIM model based on predicted complexity, and
calibrate decision confidence beyond LLM self-report to inform escalation thresholds.

**Scope (built):**
- XGBoost complexity classifier (11 engineered features) choosing `NIM_MODEL_CHEAP` vs
  `NIM_MODEL_STRONG` per ticket; choice + rationale traced as `ml_prediction/cost_routing`
- Logistic-regression baseline trained on same features; comparison table committed in
  `Docs/ROUTER_METRICS.md` and `backend/models/router_metrics.json` (SRS §6.3)
- Calibration model (`confidence_calibration.joblib`) predicting P(decision correct) from
  agent/tool-call features; dynamic escalation threshold clamped between 0.30 and 0.90
- Live orchestrator and investigator wiring: `_route_ticket` selects model and passes to
  `run_investigation`, with traces logging the exact model used; `_calibrated_threshold`
  dynamically tunes `decide_node` thresholds
- Training scripts: `scripts/train_router.py` and `scripts/train_calibration.py`

**Out of scope:** Eval harness itself (M8); displaying savings (M11).

**Dependencies:** Modules 0–6.

**SRS refs:** FR-14, FR-15, §6.3 (baseline), FR-17.

**Testing (actual):**
- *Unit — 151 tests passing suite-wide:*
  - `tests/test_ml_router.py` (7): feature builder shape/types · unknown label handling ·
    type checking · tier selection boundaries · bootstrap complexity labeling · train/load/predict
    round-trip · missing bundle FileNotFoundError
  - `tests/test_ml_calibration.py` (6): feature builder · labeling rule · threshold adjustment
    and clamping ([0.30, 0.90]) · reliability bins · train/load/predict round-trip · missing bundle
  - `tests/test_orchestrator_unit.py` (+2): complexity routing step logged and model tier
    propagated to investigator · calibrated threshold dynamically adjusting decision outcomes
- *Eval/quality (held-out test split, n=2060):*
  - **XGBoost Complexity:** Accuracy **78.79%**, Macro-F1 **0.7857** (Gate $\ge 0.70$ **PASS**)
  - **Logistic Regression Baseline:** Accuracy **69.51%**, Macro-F1 **0.6860**
  - **Comparison Gate:** XGBoost $\ge$ Baseline (+9.97% F1 margin) **PASS**
  - **Calibration Model (n=450):** Accuracy **1.0000**, Log-loss **0.0519**, Brier score **0.0125** (Gate $\le 0.15$ **PASS**)

**Definition of done:** ✅ Met — both bundles serialized and committed under `backend/models/`;
routing and calibration live in request and decision paths; metrics reports generated.

---

## Module 8 — Evaluation Harness (FR-12, FR-13, FR-17) ✅ DONE

**Goal:** Measure agent quality objectively: curate the labeled set, run it end-to-end
through the full stack, and report resolution/escalation correctness after every logic change.

**Scope (built):**
- Curated 35 hand-verified golden evaluation tickets (`backend/eval/golden_tickets.json`):
  18 routine auto-resolutions, 9 guardrail violations (refunds ≥ $50), and 8 edge cases/unresolvable
  disputes (FR-12)
- Evaluator engine (`app/eval/runner.py`): executes tickets through orchestrator state machine,
  evaluates outcome accuracy, guardrail intercept rates, escalation precision/recall/F1, and confusion matrix;
  aggregates ML model metrics from M6/M7 artifacts into a single unified payload (FR-17)
- Automated CLI script (`scripts/run_eval.py`): executes evaluation unattended with formatted terminal output
- Output artifacts (`backend/eval/report.json` and `backend/eval/report.md`): persisted baseline run
  consumed directly by `GET /api/dashboard/summary` Quality panel

**Out of scope:** Frontend visualization (M9/M11); modifying underlying model weights.

**Dependencies:** Modules 0–7 (uses full pipeline including guardrails and router).

**SRS refs:** FR-12, FR-13, FR-17; §10 success criteria 2 and 3.

**Testing (actual):**
- *Unit & Integration — 157 tests passing suite-wide:*
  - `tests/test_eval.py` (6): schema validation on golden dataset · invalid path/content error handling ·
    metric calculation math (confusion matrix, precision, recall, F1, accuracy) · M6/M7 system metric aggregation ·
    markdown report formatting · 3-ticket mini-run with isolated database and schema validation matching `QualityOut`
- *Baseline Benchmark Evaluation (n=35):*
  - **Overall Accuracy:** **100.0%** (35/35 correct terminal outcomes)
  - **Routine Resolution Rate (§10.3):** **100.0%** (18/18 auto-resolved, Gate ≥80.0% **PASS**)
  - **Guardrail Safety Intercept (§10.2):** **100.0%** (9/9 blocked to pending_approval, Gate 100.0% **PASS**)
  - **Escalation Correctness & F1 (FR-13):** **100.0%** (8/8 escalated, F1: **1.0000**, Gate ≥0.75 **PASS**)
  - **Outcome Confusion Matrix:** Zero off-diagonal classifications
  - **Unified ML Metrics (FR-17):** M6 Classifier (Cat F1 0.8501, Sev F1 0.5899), M7 Router (+9.97% advantage vs baseline), M7 Calibration (Brier score 0.0125, Acc 1.0)

**Definition of done:** ✅ Met — harness runs unattended via CLI; baseline report generated and committed; full quality gates passed; ready for frontend consumption.

---

## Module 9 — Dashboard Frontend Core

**Goal:** Ship the four-panel dashboard that tells a non-technical viewer "is this system
working and trustworthy" at a glance, updating live over WebSocket.

**Scope:**
- Dashboard aggregation API (`GET /api/dashboard/summary` replacing the 501 stub):
  volume/outcomes trend, cost totals, eval scores, queue snapshot
- Four panels (Recharts) with self-descriptive titles/metrics per FR-21; approval-queue
  panel consuming M5 APIs with expand-to-trace link
- WebSocket feed wiring (`useWebSocket` → panel updates) so new tickets/approvals appear
  without refresh (FR-20)

**Out of scope:** Three.js drill-down view (M10); cost-savings comparison math and feature
importance chart (M11 — panels render placeholders until then).

**Dependencies:** Modules 0–5 (data sources), 8 (eval scores exist to show; can ship with
"no scores yet" states earlier if desired).

**SRS refs:** FR-16 (panels 1–4), FR-20, FR-21; NFR Usability.

**Testing:**
- *Unit:* summary-aggregation logic on the backend (fixture rows → exact panel shapes)
- *Integration:* TestClient — summary endpoint reflects seeded tickets/approvals; WS
  broadcast arrives after a processed ticket
- *Frontend:* ⚠️ requires adding **Vitest + React Testing Library** — new dev dependencies,
  will be asked for sign-off per rules.md before install; minimal component smoke tests
  (panel renders heading + handles empty state). Until approved, gate = `npm run build` +
  typecheck
- *Manual:* FR-21 walkthrough — a first-time viewer can state system status from the page alone

**Definition of done:** Aggregation + WS tests pass; frontend gates green; FR-21 walkthrough
done with you as the reviewer.

---

## Module 10 — Ticket Detail + Three.js Trace Visualizer

**Goal:** Per-ticket drill-down combining the plain-text reasoning summary with the
interactive animated agent-flow graph.

**Scope:**
- `TraceViewer` (r3f + drei) rendering real trace rows as nodes (agents/tools/memory/
  guardrail/decision) and edges, with step-by-step animated replay controls
- Ticket detail page: summary text beside the canvas; reachable from dashboard rows AND
  queue items (FR-19)
- Layout algorithm (pure TS, deterministic positions per trace shape)

**Out of scope:** New backend capability (traces API from M1 suffices); dashboard panels (M9).

**Dependencies:** Modules 0–5, 9 (navigation shell).

**SRS refs:** FR-11, FR-19; §10 success criteria 1 and 5; NFR Auditability (visual proof).

**Testing:**
- *Unit:* graph-layout function (trace fixture → node positions, edge list, no overlaps on
  golden cases); replay-sequence builder (rows → ordered animation steps)
- *Frontend:* component smoke test — viewer renders N nodes for a fixture trace (Vitest,
  same sign-off caveat as M9); otherwise build/typecheck gates
- *Integration:* real trace from a live demo run rendered without console errors
- *Manual (primary):* §10 checklist — load a real agent run, nodes/edges correct, replay
  animates in order, matches the textual trace side-by-side

**Definition of done:** Layout/replay unit tests pass; a real run's trace renders and
animates correctly in your browser (you verify visually).

---

## Module 11 — Analytics Completion (Cost Savings + Explainability)

**Goal:** Finish the dashboard's story: prove the ML routing saves money vs always-strongest,
and expose why the classifier decides what it decides.

**Scope:**
- Cost accounting: per-call token/cost capture (already logged in `nim_client.py`) aggregated
  into actual-vs-naive-baseline spend; "$ saved by smart routing" computed and shown (FR-16.2)
- XGBoost feature-importance chart surfaced in Quality panel (FR-18)
- Eval-score display finalized from M8 reports (FR-16.3 completion)

**Out of scope:** Retraining models; changing routing behavior.

**Dependencies:** Modules 7 (router + importance artifact), 8 (scores), 9 (panels exist).

**SRS refs:** FR-16.2/.3, FR-18; §10 success criterion 4.

**Testing:**
- *Unit:* cost-math functions with fixture token logs (pricing table pinned in code);
  importance-data transformer
- *Integration:* summary API returns savings + importance payloads for seeded history
- *Manual:* §10 check — savings figure positive and plausible on demo data; importance chart
  intelligible to a non-technical reader (your call)

**Definition of done:** Math tested and matching manual arithmetic on a small fixture;
charts live; §10 criterion 4 demonstrable on demand.

---

## Module 12 — Optional Stretch Models (Clustering + Anomaly Detection)

**Goal:** OPTIONAL per SRS §6.2 — auto-discovered ticket taxonomy (k-means/HDBSCAN on
embeddings) and category-spike anomaly flags (Isolation Forest).

**Scope:** Clustering job + cluster explorer endpoint/panel; anomaly detector over daily
category volumes with flag surfacing. Built only if you green-light it after M11.

**Out of scope:** Anything core — skipping this module does not affect any FR.

**Dependencies:** Modules 6 (dataset), 9/10 (surface), embeddings from M1/M3.

**SRS refs:** §6.2 (both rows, marked Optional/Stretch).

**Testing:**
- *Unit:* volume-vector builder for anomaly detection (fixture days → expected vector)
- *Eval/quality:* clustering has no ground truth — report silhouette + a labeled-sample
  purity eyeball; anomaly detection validated by planting a synthetic spike and confirming detection
- *Manual:* taxonomy review — do discovered clusters read like real categories?

**Definition of done (if undertaken):** Planted spike detected; clusters pass your eyeball
review; both explicitly marked experimental in the UI/report.

---

## Module 13 — Deployment & Hardening

**Goal:** Put the whole thing on free-tier hosting at $0, close out security/PII requirements,
and walk the full SRS success-criteria list end-to-end.

**Scope:**
- Hosting: backend → Railway/Render, frontend → Vercel; swap local containers for Neon/Supabase
  Postgres + hosted/self-host Qdrant (config-only change by design)
- PII redaction implemented in `app/core/logging.py` (replace the M0 stub) + log audit
- Security pass: secrets audit (none hardcoded), least-privilege tool review, production CORS
- Final §10 success-criteria walkthrough script/checklist

**Out of scope:** Real production traffic, multi-tenant auth/billing (SRS §9 out-of-scope items stay out).

**Dependencies:** Modules 0–11 (everything deployed; M12 independent/optional).

**SRS refs:** §7 hosting rows; NFR Security (secrets, least privilege, PII redaction),
Cost ($0), Scalability (as-supported-by-host); §10 all five criteria.

**Testing:**
- *Unit:* redaction patterns (emails, card-like numbers, phone numbers masked; normal text untouched)
- *Integration:* deployed URL serves health + dashboard against cloud DB/Qdrant; WS works cross-origin
- *Manual:* secrets scan (`grep`/git history review); §10 checklist executed against the
  deployed system, each criterion checked off in the final report

**Definition of done:** Redaction tests pass; deployed system passes all five §10 criteria
in one sitting; zero secrets in repo or logs.

---

## Coverage Matrix (cross-check)

| Requirement | Module(s) |
|---|---|
| FR-1 ML classification | **M6 ✅** (gates passed; serves ahead of routing, traced) |
| FR-2 Investigator via tools | **M1 ✅** |
| FR-3 Guardrail blocking | **M5 ✅** |
| FR-4 Orchestrator | **M4 ✅** |
| FR-5 Short-term memory | **M4 ✅** |
| FR-6 Long-term memory | **M2 ✅** |
| FR-7 Episodic memory | **M3 ✅** (+ M5: write-back scoped to terminal resolutions) |
| FR-8 Queue w/ visible trace | **M5 ✅** (+ M10 display) |
| FR-9 Decision logging + write-back | **M5 ✅** (decisions logged, traced, written back) · auto-resolutions write back via M4's resolve node |
| FR-10 Full trace logging | **M1 ✅** |
| FR-11 Three.js trace viz | M10 |
| FR-12 Labeled eval set | M8 |
| FR-13 Accuracy reporting per change | M8 |
| FR-14 Cheap/strong router | **M7 ✅** |
| FR-15 Confidence calibration | **M7 ✅** |
| FR-16 Four dashboard panels | M9 (panels) + M11 (cost math, importance, scores) |
| FR-17 Model precision/recall/F1 | M6 (classifier ✅) + M7 (router/calibration ✅) + M8 (aggregation) — split flagged |
| FR-18 Feature importance surfaced | M11 |
| FR-19 Detail view w/ summary + graph | M10 |
| FR-20 WebSocket live updates | M9 (WS groundwork exists since M0/M4 events) |
| FR-21 Self-descriptive UI | M9 (+ M10) |
| NFR Reliability | **M4 ✅** |
| NFR Scalability | **M4 ✅** (async worker; hosted scale at M13) |
| NFR Security | M13 (redaction stub exists since M0 — flagged) |
| NFR Cost ($0) | M0 (dev) + M13 (hosted) |
| NFR Auditability | **M1 ✅** + M10 (visual proof) |
| NFR Usability | M9 + M10 |
| §6.1 core models | M6 (classifier ✅) + M7 (router, calibration ✅) + M1 (embeddings ✅) |
| §6.2 optional models | M12 (optional) |
| §6.3 LR baseline | **M7 ✅** |
| §8 data requirements | M6 (datasets/synthetic) + M3 (episode seeds) + M1 (KB ✅) |
| §10 success criteria | 1→M1/M10 · 2→M5 · 3→M8 · 4→M7+M11 · 5→M10 |

**Nothing dropped:** every FR-1…FR-21, every NFR row, and §6/§8/§10 items map to at least
one module. Deliberate splits/seams are called out inline (FR-9, FR-17, FR-16, FR-5's
partial satisfaction inside the M1 loop, M4's rule-based routing placeholder awaiting M6).
