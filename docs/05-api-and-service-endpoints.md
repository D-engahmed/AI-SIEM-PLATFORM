# 05 — API and service endpoints

FastAPI app (`src/api.py`), same process/event loop as the Kafka consumer
(see `docs/04-deployment.md`), on `CML_API_PORT` (default 9100).

## `GET /healthz`

Static `{"status": "ok"}`. Liveness only — does not check Kafka connectivity
or model health; that's what fail-fast startup and `max_consecutive_errors`
are for (see `docs/04-deployment.md`).

## `GET /metrics`

Prometheus text exposition format. Exposes:

- `cml_degraded_messages_total` — incremented once per **real Kafka message**
  processed with `degraded_mode=true`, regardless of escalation outcome
  (upstream data-quality visibility, not an incident-volume metric).
- `cml_feature_psi{feature=...}` — per-feature PSI vs. the training reference
  (see `docs/06-monitoring.md`).
- `cml_monitoring_buffer_fill_ratio` — how full the drift ring buffer is.

## `POST /score`

**Added on direct request** ("model API available to connect"). Runs the
exact same validation → feature-engineering → model → decision path the
Kafka consumer uses (`MLScoringHandler.handle_payload`), and returns what
*would* happen — `{"decision": "escalated"|"dropped"|"rejected", ...}` — but
**never publishes anywhere and never touches Kafka**.

**Deliberate design decision, not asked for but necessary to implement
anything**: this is side-effect-free by design. An endpoint that silently
published real incidents from ad-hoc test/integration traffic would be a much
easier way to pollute the `incidents` topic than "give me an API to connect
to" was likely picturing. If synchronous *publishing* is actually wanted,
that's a separate, larger decision — flagged here, not made unilaterally.

**A real bug this endpoint's own docstring had, fixed rather than left**: the
docstring originally claimed calling `/score` doesn't affect
`cml_degraded_messages_total`. That was false as first written — the
increment lived inside the shared `handle_payload()`, which `/score` calls
directly. Fixed by moving the increment to `CorrelationMLConsumer`'s
Kafka-specific layer instead, so the pure handler (and anything else that
calls it, like this endpoint) never touches production-traffic metrics.
Pinned by `tests/test_api.py::TestScoreEndpoint::test_score_call_does_not_increment_degraded_counter`
and `tests/test_ml_consumer.py::TestDegradedModeFlagPlacement`.

**A real performance bug this endpoint had, found by actually load-testing
it**: the handler was originally called synchronously, unawaited, inside the
`async def score(...)` coroutine. Model inference is a blocking, CPU-bound C
call that doesn't yield to the event loop — under concurrent `/score` load
that serialized every request onto the single event loop thread, and (worse)
was capable of starving the Kafka consumer loop sharing that same event loop.
Fixed with `asyncio.to_thread()`. See `docs/07-stress-testing.md` for the
load-test numbers and an important caveat about interpreting them.

## `GET /monitoring/drift`

JSON drift report — the same PSI numbers `/metrics` exposes as Prometheus
gauges, with WARN/ALERT/INSUFFICIENT_DATA status computed, for direct
debugging rather than through a dashboard. Returns 503 if no reference
distribution was loaded at startup (see `docs/06-monitoring.md`).

## What's deliberately NOT here

- No auth on any endpoint — this service has no user-facing auth story in the
  spec it was given; assumed to sit behind whatever network/gateway
  authentication the platform provides. Not verified.
- No request-body size limit set explicitly on `/score` beyond what
  FastAPI/Starlette default to — the stress tests confirm the *model* handles
  extreme inputs (huge `signal_ids` lists, 10k-char strings) without crashing,
  but that's a different guarantee from "the HTTP layer can't be used to
  exhaust memory with a giant request body."
