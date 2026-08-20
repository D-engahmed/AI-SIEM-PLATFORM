"""
FastAPI app for correlation-ml-service: /healthz, /metrics, and /score.

Runs on the same event loop / same OS process as the Kafka consumer
(see main.py and metrics.py's module docstring for why).

2026-08-18: added POST /score -- "model API available to connect" (asked
for directly). Design decision made here, not asked for but necessary to
implement anything: /score is a SIDE-EFFECT-FREE dry run. It runs the
exact same MLScoringHandler.handle_payload() the Kafka path uses -- same
validation, same feature engineering, same model, same escalation
threshold -- and returns what WOULD be published, but does not touch
Kafka (no incident produced, no DLQ produced) and does not increment
DEGRADED_MESSAGES_TOTAL (that counter means "a degraded message was
actually processed off the real topic"; a test call through this
endpoint is not that). Rationale: an endpoint that silently published
real incidents from ad-hoc test traffic would be a much easier way to
pollute the `incidents` topic than anyone asking for "an API to connect
to" was likely picturing. If synchronous *publishing* (not just scoring)
is actually wanted, that's a different, larger decision -- flagged here,
not made unilaterally.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError

from metrics import REGISTRY

app = FastAPI(title="correlation-ml-service")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/drift")
async def drift(request: Request) -> dict:
    """
    Human-readable PSI drift report -- the same numbers /metrics exposes
    as Prometheus gauges (cml_feature_psi), but as JSON with the WARN/
    ALERT/INSUFFICIENT_DATA status computed for you, for someone
    debugging directly rather than through a dashboard.
    """
    monitor = getattr(request.app.state, "monitor", None)
    if monitor is None:
        raise HTTPException(
            status_code=503,
            detail="drift monitoring not available -- no feature reference distribution was "
                   "loaded at startup (see training/train_model.py)",
        )
    return {"features": monitor.compute_all()}


@app.post("/score")
async def score(payload: dict[str, Any], request: Request) -> dict:
    """
    Score one ml-scoring-tasks-shaped payload synchronously. Same
    validation/feature/model path as the Kafka consumer; does not publish
    anywhere. See module docstring for why this doesn't just call
    handle_payload's Kafka-producing sibling instead.
    """
    handler = getattr(request.app.state, "handler", None)
    if handler is None:
        # Only reachable if something calls this before main.py finishes
        # startup wiring (see _run() in main.py) -- a real ordering bug,
        # not a normal 4xx, hence 503 not 400/422.
        raise HTTPException(status_code=503, detail="model not loaded yet")

    # BUG FOUND AND FIXED (2026-08-20, via stress/run_load_test.py): this
    # used to call handler.handle_payload(payload) directly, unawaited, in
    # this coroutine. handle_payload() -> scorer.score() -> XGBoost's
    # booster.predict() is a synchronous, CPU-bound C call that does NOT
    # yield to the event loop. Under concurrent /score load that serialized
    # every request onto the single event loop thread -- 20 concurrent
    # workers measured p50=46ms (fine) but p95=253ms, p99=399ms (queueing
    # delay compounding, not per-request slowness). Worse: this process
    # ALSO runs the Kafka consumer loop on the SAME event loop (see
    # metrics.py's module docstring) -- concurrent /score traffic was
    # capable of starving real Kafka message processing, not just slowing
    # itself down. asyncio.to_thread() runs the blocking call in the
    # default thread pool executor instead, freeing the event loop.
    result = await asyncio.to_thread(handler.handle_payload, payload)

    if result.incident is not None:
        return {"decision": "escalated", "incident": _json_safe(result.incident)}
    if result.dlq is not None:
        return {
            "decision": "rejected",
            "stage": result.dlq.stage,
            "error_type": type(result.dlq.error).__name__,
            "error": str(result.dlq.error),
        }
    return {"decision": "dropped", "reason": "below escalation_threshold, or a benign no-op"}


def _json_safe(incident: dict[str, Any]) -> dict[str, Any]:
    # incident dict is model_dump(mode="python") -- has real datetime/UUID
    # objects (see ml_consumer.py's comment on why), which the Kafka
    # producer's orjson layer handles but FastAPI's default JSON encoder
    # needs a hand for. Reuses the exact same serializer so /score's JSON
    # timestamps match the wire format ("...Z"), not FastAPI's default.
    import orjson

    from shared.kafka.base_producer import _serialize

    return orjson.loads(_serialize(incident))
