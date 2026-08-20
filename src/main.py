"""
Entrypoint for correlation-ml-service.

Order matters here and is deliberate:
  1. load config (fail fast on bad env)
  2. load model ONCE (fail fast if artifact missing/corrupt/mismatched)
  3. only then start consuming -- we never want to accept traffic we
     can't score.

2026-08-17: now async, and runs the Kafka consumer alongside the
/metrics FastAPI app (see metrics.py's module docstring for why they
share one process instead of being genuinely separate).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

import uvicorn

import api
from config import get_settings
from ml_consumer import CorrelationMLConsumer, MLScoringHandler
from ml_scorer import ModelScorer
from metrics import REGISTRY
from monitoring import FeatureDriftMonitor

logger = logging.getLogger("correlation_ml.main")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


async def _run(consumer: CorrelationMLConsumer, api_host: str, api_port: int, log_level: str) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)

    await consumer.start()

    uvicorn_config = uvicorn.Config(
        "api:app", host=api_host, port=api_port, log_level=log_level.lower(), loop="asyncio",
    )
    server = uvicorn.Server(uvicorn_config)

    consumer_task = asyncio.create_task(consumer.run_forever(), name="kafka-consumer")
    api_task = asyncio.create_task(server.serve(), name="metrics-api")
    stop_task = asyncio.create_task(stop_event.wait(), name="stop-signal")

    try:
        done, pending = await asyncio.wait(
            {consumer_task, api_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        # Either shutdown was requested, or one of the two real tasks died
        # unexpectedly -- either way, stop the other one cleanly and
        # propagate any real exception.
        server.should_exit = True
        for task in (consumer_task, api_task, stop_task):
            if task not in done:
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if consumer_task in done:
            consumer_task.result()  # re-raise if the consumer loop crashed
        if api_task in done:
            api_task.result()  # re-raise if the API server crashed
    finally:
        await consumer.stop()


def main() -> int:
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger.info("starting correlation-ml-service consumer_group=%s", settings.consumer_group_id)

    try:
        if settings.model_source == "mlflow":
            scorer = ModelScorer.load_from_mlflow(
                tracking_uri=settings.mlflow_tracking_uri,
                model_uri=settings.model_mlflow_uri,
                threshold=settings.escalation_threshold,
                degraded_field_threshold=settings.degraded_mode_imputed_field_threshold,
            )
        else:
            scorer = ModelScorer.load(
                artifact_path=settings.model_artifact_path,
                threshold=settings.escalation_threshold,
                degraded_field_threshold=settings.degraded_mode_imputed_field_threshold,
            )
    except (FileNotFoundError, ValueError) as e:
        # Fail fast and loud -- an orchestrator should crash-loop this and
        # page someone, not start a consumer that can't score anything.
        logger.critical("failed to load model artifact, refusing to start: %s", e)
        return 1

    handler = MLScoringHandler(scorer=scorer)

    # See FeatureDriftMonitor.load()'s docstring -- returns None (not an
    # exception) if the reference artifact isn't there yet. Monitoring
    # degraded, not fatal: scoring still works with monitor=None.
    monitor = FeatureDriftMonitor.load(
        reference_path=settings.feature_reference_path,
        ring_buffer_size=settings.monitoring_ring_buffer_size,
        psi_warn_threshold=settings.psi_warn_threshold,
        psi_alert_threshold=settings.psi_alert_threshold,
        registry=REGISTRY,
    )
    consumer = CorrelationMLConsumer(settings, handler, monitor=monitor)
    # Wires the already-loaded model into POST /score (see api.py's module
    # docstring) -- set before uvicorn starts serving in _run() below, so
    # there's no window where /score is reachable but handler is None.
    api.app.state.handler = handler
    api.app.state.monitor = monitor

    try:
        asyncio.run(_run(consumer, settings.api_host, settings.api_port, settings.log_level))
    except Exception:
        logger.exception("service exited with an unhandled error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
