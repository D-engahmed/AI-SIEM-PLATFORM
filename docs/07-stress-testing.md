# 07 — Stress testing

Two genuinely different kinds of "stress", tested two different ways.

## 1. Model robustness (`tests/test_model_stress.py`)

Part of the normal fast test suite (in-process, no live server needed). Does
the scoring path survive extreme, malformed, and adversarial-ish input without
crashing, producing NaN/inf, or violating its own `0 <= risk_score <= 100`
contract? These tests don't assert the model is *right* about weird inputs —
only that it doesn't fall over. Covers: fan-in counts up to 1,000,000, ages
from 0 to 1e15, all-fields-missing, adversarial `target_value` strings (SQL
injection attempts, null bytes, unicode direction overrides, 10k-char
strings), statelessness under repetition (200 identical calls give identical
output; extreme inputs don't leak state into subsequent normal ones), and a
`@pytest.mark.slow` 5,000-iteration randomized sweep.

**A wrong assumption caught while writing these, not after**: two tests
originally assumed negative `fan_in_count`/`epoch_age_seconds` were accepted
and passed straight through unvalidated. Checked the actual `GraphFeatures`
source before trusting that assumption — both already have dedicated
`field_validator`s rejecting negative values. Tests corrected to assert the
real (and better) behavior instead of the wrong guess.

## 2. Throughput/latency (`stress/run_load_test.py`)

Standalone script, not a pytest module — needs a **live running process**,
over the network, so it doesn't belong in the fast suite. Fires concurrent
`POST /score` requests for a fixed duration and reports p50/p95/p99 latency
and throughput; exits 1 if p95 exceeds `--latency-budget-ms` (defaults to 50,
matching `CML_API_LATENCY_WARN_MS`).

```
python stress/run_load_test.py --url http://localhost:9100 \
    --concurrency 50 --duration 30 --latency-budget-ms 50
```

### What running it actually found

At 20 concurrent workers, the very first run showed p50=46ms (fine) but
p95=253ms, p99=399ms — a FAIL against the 50ms budget. That gap (p50 fine,
tail much worse) is the specific signature of requests queueing behind
blocking work, not uniformly slow individual requests.

Investigated rather than just reported: `POST /score` was calling
`handler.handle_payload()` directly, unawaited, inside an `async def`
coroutine. Model inference is a blocking, CPU-bound C call that doesn't yield
to the event loop — under concurrent load that serialized every request onto
the single event loop thread, and because this process **also** runs the
Kafka consumer loop on that same event loop (`docs/04-deployment.md`),
concurrent `/score` traffic was capable of starving real Kafka message
processing, not just slowing itself down. Fixed with `asyncio.to_thread()`.

### An honest caveat about the numbers above, not a claim of victory

Re-running the exact same load test after the fix showed **no visible
improvement** (p95 still ~256ms). Dug into why instead of declaring the fix
a bust: this sandbox has exactly **one CPU core** (`nproc` → 1). Confirmed by
hitting the completely trivial `GET /healthz` endpoint — which does no
scoring, no Pydantic body validation, nothing — under the identical 20-way
concurrent load: p50=39ms, p95=183ms, p99=273ms. Nearly identical numbers to
`/score`, on an endpoint with zero model-related work at all. That rules out
the scoring path as the cause of the absolute latency in *this* environment —
it's the load-generator and the server process contending for one shared
core, not an application bug.

Separately timed a single in-process `scorer.score()` call directly (no HTTP,
no asyncio): **0.407ms/call average** over 200 warmed-up calls. The actual
model inference cost is nowhere near the bottleneck.

**What this means, stated plainly**: the `asyncio.to_thread()` fix is still
architecturally correct and worth keeping — it stops `/score` traffic from
being able to starve the Kafka consumer sharing its event loop, which matters
regardless of core count. But this sandbox's absolute latency numbers
(p95≈250ms) are **not** a meaningful estimate of production performance on a
properly provisioned multi-core host, and shouldn't be quoted as one. On real
hardware with more than one core, the practical bottleneck is very unlikely to
be model inference (0.4ms/call) and much more likely to be ordinary HTTP/ASGI
overhead — worth re-running `stress/run_load_test.py` against the actual
target deployment environment before trusting any specific number from it.
