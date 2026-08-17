"""
Kafka wiring for correlation-ml-service.

HONEST GAP (do not skip this comment): the spec says ml_consumer.py
"inherits from shared.kafka", but no interface for that base class (method
names, callback shape, sync vs async, commit semantics) was provided or
discoverable in this conversation. Guessing a subclass signature for a
library I've never seen would produce code that *looks* wired up but
silently doesn't match the real base class -- worse than not writing it,
because it hides the gap instead of surfacing it.

So this file is split in two, deliberately:

  1. MLScoringHandler -- the actual business logic (parse, validate,
     score, decide incidents-vs-dlq-vs-drop). Pure, synchronous,
     dependency-injected, fully unit tested. This is correct regardless
     of what shared.kafka turns out to look like, and should NOT need to
     change when you wire in the real base class.

  2. StandaloneRunner -- a working confluent_kafka-based outer loop, used
     when shared.kafka isn't importable (e.g. running this repo
     standalone, or in tests/CI). This is the part that's a guess. When
     you have the real shared.kafka.BaseConsumer, replace StandaloneRunner
     with a subclass of it that calls MLScoringHandler.handle_raw() from
     whatever its message callback is named -- that should be close to a
     ~20 line adapter, not a rewrite.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from config import Settings
from ml_scorer import ModelScorer
from schemas import (
    ESCALATED_STATUS,
    DLQEnvelope,
    IncidentEvent,
    MLScoringTask,
    _iso_z,
)
from datetime import datetime, timezone

logger = logging.getLogger("correlation_ml.consumer")

SUPPORTED_TARGET_KINDS = {"user"}  # spec: device tracking disabled upstream, don't build for it


@dataclass
class HandleResult:
    incident: Optional[dict] = None
    dlq: Optional[dict] = None
    # neither set => silently dropped (valid message, model said don't escalate)


class MLScoringHandler:
    """
    Pure per-message logic. No Kafka objects touch this class -- it takes
    raw bytes in and returns a decision. That's what makes it swap-proof
    against whatever shared.kafka's real API looks like, and what makes
    it trivial to unit test (see tests/test_ml_consumer.py).
    """

    def __init__(self, scorer: ModelScorer, consumer_group: str):
        self._scorer = scorer
        self._consumer_group = consumer_group

    def handle_raw(self, raw_bytes: bytes) -> HandleResult:
        # --- stage: deserialize ---
        try:
            raw_text = raw_bytes.decode("utf-8")
            payload_dict = json.loads(raw_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return HandleResult(dlq=self._dlq(
                original_payload={"_raw_undecodable": True, "byte_len": len(raw_bytes)},
                error=e, stage="deserialize",
            ))

        # --- stage: validate ---
        try:
            task = MLScoringTask.model_validate(payload_dict)
        except ValidationError as e:
            return HandleResult(dlq=self._dlq(original_payload=payload_dict, error=e, stage="validate"))

        if task.target_kind not in SUPPORTED_TARGET_KINDS:
            # Fail closed rather than silently mis-scoring a target type we
            # were told never to build logic for.
            err = ValueError(f"unsupported target_kind={task.target_kind!r}")
            return HandleResult(dlq=self._dlq(original_payload=payload_dict, error=err, stage="validate"))

        # --- stage: inference ---
        try:
            result = self._scorer.score(task)
        except Exception as e:  # noqa: BLE001 - model/feature-pipeline errors are genuinely unpredictable
            return HandleResult(dlq=self._dlq(original_payload=payload_dict, error=e, stage="inference"))

        if result.status != ESCALATED_STATUS:
            # Spec: "Drop the message silently otherwise." This is a normal
            # business outcome, not a fault -- no DLQ, no error log spam.
            logger.debug(
                "correlation_id=%s below threshold prob=%.4f model_version=%s",
                task.correlation_id, result.probability, result.model_version,
            )
            return HandleResult()

        # --- stage: publish (build + validate outbound contract) ---
        source_ip = task.signal_context.triggering_source_ip or "unknown"
        linked_keys = {"ip": source_ip, "user": task.target_value}
        now = datetime.now(timezone.utc)
        window_start = task.signal_context.window_start or now
        window_end = task.signal_context.window_end or now

        try:
            incident = IncidentEvent(
                correlation_id=task.correlation_id,
                linked_keys=linked_keys,
                signal_ids=task.signal_context.signal_ids or [],
                risk_score=result.risk_score,
                degraded_mode=result.degraded_mode,
                window_start=window_start,
                window_end=window_end,
                created_at=now,
                updated_at=None,
                model_version=result.model_version,
            )
        except ValidationError as e:
            # e.g. a linked_keys value blew past the 255-char contract limit.
            # This is a real failure, not a silent drop -- we scored it as an
            # incident but can't legally publish it as one.
            return HandleResult(dlq=self._dlq(original_payload=payload_dict, error=e, stage="publish"))

        logger.info(
            "ESCALATED correlation_id=%s risk_score=%.1f degraded_mode=%s model_version=%s imputed=%s",
            task.correlation_id, result.risk_score, result.degraded_mode,
            result.model_version, result.imputed_fields,
        )
        return HandleResult(incident=incident.to_wire_dict())

    def _dlq(self, *, original_payload: dict, error: Exception, stage: str) -> dict:
        logger.warning("DLQ stage=%s error=%s: %s", stage, type(error).__name__, error)
        env = DLQEnvelope(
            original_payload=original_payload,
            error_type=type(error).__name__,
            error_message=str(error)[:2000],  # bound it -- error string is partly attacker-influenced
            stage=stage,
            consumer_group=self._consumer_group,
            failed_at=datetime.now(timezone.utc),
        )
        return env.to_wire_dict()


# --------------------------------------------------------------------------
# Outer loop -- GUESSED framework glue, see module docstring.
# --------------------------------------------------------------------------

try:
    from shared.kafka import BaseKafkaConsumer  # type: ignore  # real thing, if it exists in this repo
    _HAVE_SHARED_KAFKA = True
except ImportError:
    _HAVE_SHARED_KAFKA = False


class StandaloneRunner:
    """
    Working confluent_kafka outer loop, used only when shared.kafka isn't
    available. Fail-closed error handling, non-blocking produce, offsets
    committed only after both produces (incident-or-dlq) are confirmed
    delivered -- so a crash mid-produce causes a redelivery (at-least-once,
    possible duplicate incident) rather than a silent drop.
    """

    def __init__(self, settings: Settings, handler: MLScoringHandler):
        try:
            from confluent_kafka import Consumer, Producer
        except ImportError as e:
            raise RuntimeError(
                "confluent-kafka is not installed and shared.kafka was not importable. "
                "Install confluent-kafka for standalone use, or provide shared.kafka."
            ) from e

        self._settings = settings
        self._handler = handler
        self._consumer = Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.consumer_group_id,
            "enable.auto.commit": False,  # we commit explicitly after successful produce
            "auto.offset.reset": "earliest",
        })
        self._producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
        self._running = False
        self._consecutive_errors = 0

    def _produce_sync(self, topic: str, key: str, value: dict, timeout: float = 10.0) -> bool:
        """Produce one message and block only until delivery is confirmed or times out."""
        delivered = {"ok": False, "err": None}

        def _cb(err, _msg):
            delivered["ok"] = err is None
            delivered["err"] = err

        try:
            self._producer.produce(
                topic, key=key.encode("utf-8"), value=json.dumps(value).encode("utf-8"), callback=_cb,
            )
        except BufferError as e:
            logger.error("producer queue full, backing off: %s", e)
            return False

        remaining = timeout
        step = 0.05
        while remaining > 0 and delivered["err"] is None and not delivered["ok"]:
            self._producer.poll(step)
            remaining -= step
        return delivered["ok"]

    def _handle_signal(self, signum, _frame):
        logger.info("received signal %s, shutting down after current message", signum)
        self._running = False

    def run(self):
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self._consumer.subscribe([self._settings.consume_topic])
        self._running = True
        logger.info(
            "consumer started topic=%s group=%s", self._settings.consume_topic, self._settings.consumer_group_id,
        )

        try:
            while self._running:
                msg = self._consumer.poll(self._settings.consumer_poll_timeout_seconds)
                if msg is None:
                    continue
                if msg.error():
                    self._consecutive_errors += 1
                    logger.error("kafka poll error: %s", msg.error())
                    if self._consecutive_errors >= self._settings.max_consecutive_errors:
                        raise RuntimeError("too many consecutive Kafka errors, failing closed")
                    continue

                result = self._handler.handle_raw(msg.value())

                ok = True
                if result.incident is not None:
                    ok = self._produce_sync(
                        self._settings.produce_topic_incidents,
                        key=result.incident["correlation_id"],
                        value=result.incident,
                    )
                elif result.dlq is not None:
                    ok = self._produce_sync(
                        self._settings.produce_topic_dlq,
                        key=result.dlq.get("original_payload", {}).get("correlation_id", "unknown"),
                        value=result.dlq,
                    )
                # else: silent drop, nothing to produce, offset still advances

                if ok:
                    self._consumer.commit(msg, asynchronous=False)
                    self._consecutive_errors = 0
                else:
                    self._consecutive_errors += 1
                    logger.error("produce failed, NOT committing offset, will redeliver")
                    if self._consecutive_errors >= self._settings.max_consecutive_errors:
                        raise RuntimeError("too many consecutive produce failures, failing closed")
        finally:
            self._producer.flush(10.0)
            self._consumer.close()
            logger.info("consumer stopped cleanly")


def build_runner(settings: Settings, handler: MLScoringHandler):
    if _HAVE_SHARED_KAFKA:
        raise NotImplementedError(
            "shared.kafka was importable but this repo doesn't know its interface yet. "
            "Wire MLScoringHandler.handle_raw() into your BaseKafkaConsumer subclass here."
        )
    return StandaloneRunner(settings, handler)
