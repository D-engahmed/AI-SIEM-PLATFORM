"""
Kafka wiring for correlation-ml-service.

2026-08-17: rewritten against shared/kafka/ (our local stand-in -- see
that package's docstring for why it's local and not the real platform
package). Kept the original two-layer split on purpose:

  1. MLScoringHandler -- pure business logic (validate, score, decide
     incident-vs-dlq-vs-drop). No Kafka objects touch this class. Takes
     an already-deserialized dict now instead of raw bytes, since
     BaseConsumer owns orjson deserialization.

  2. CorrelationMLConsumer(BaseConsumer) -- the thin adapter the original
     module docstring predicted would exist once a real base class showed
     up ("that should be close to a ~20 line adapter, not a rewrite").
     It's wired to our local stand-in instead of the real package, but
     the shape is exactly what was described.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from pydantic import ValidationError

from config import Settings
from metrics import DEGRADED_MESSAGES_TOTAL
from ml_scorer import ModelScorer
from monitoring import FeatureDriftMonitor
from schemas import (
    ESCALATED_STATUS,
    STRATEGY_NAME,
    IncidentEvent,
    MLScoringTask,
    severity_for_score,
)
from shared.kafka.base_consumer import BaseConsumer

logger = logging.getLogger("correlation_ml.consumer")

SUPPORTED_TARGET_KINDS = {"user"}  # spec: device tracking disabled upstream, don't build for it
MAX_SIGNAL_IDS = 500


@dataclass
class DlqInfo:
    payload: dict[str, Any]
    error: Exception
    stage: str


@dataclass
class HandleResult:
    incident: Optional[dict[str, Any]] = None
    dlq: Optional[DlqInfo] = None
    # neither set => silently dropped (valid message, model said don't escalate)
    # Set whenever scoring actually ran (i.e. not on a validate/inference DLQ),
    # regardless of escalation outcome. Kept on HandleResult rather than
    # incremented inside handle_payload() itself (2026-08-18 fix) -- see
    # CorrelationMLConsumer.process_message() for why.
    degraded_mode: bool = False
    # Same reasoning as degraded_mode: the drift monitor's ring buffer must
    # only see REAL topic traffic, not ad-hoc POST /score test calls, so
    # recording it is CorrelationMLConsumer's job, not handle_payload()'s.
    feature_vector: Optional[dict[str, float]] = None


class MLScoringHandler:
    """
    Pure per-message logic. No Kafka objects touch this class -- that's
    what makes it swap-proof against whatever the real shared.kafka turns
    out to look like, and what makes it trivial to unit test (see
    tests/test_ml_consumer.py).
    """

    def __init__(self, scorer: ModelScorer):
        self._scorer = scorer

    def handle_payload(self, payload: dict[str, Any]) -> HandleResult:
        # --- stage: validate ---
        try:
            task = MLScoringTask.model_validate(payload)
        except ValidationError as e:
            return HandleResult(dlq=DlqInfo(payload=payload, error=e, stage="validate"))

        if task.target_kind not in SUPPORTED_TARGET_KINDS:
            # Fail closed rather than silently mis-scoring a target type we
            # were told never to build logic for.
            err = ValueError(f"unsupported target_kind={task.target_kind!r}")
            return HandleResult(dlq=DlqInfo(payload=payload, error=err, stage="validate"))

        # --- stage: inference ---
        try:
            result = self._scorer.score(task)
        except Exception as e:  # noqa: BLE001 - model/feature-pipeline errors are genuinely unpredictable
            return HandleResult(dlq=DlqInfo(payload=payload, error=e, stage="inference"))

        if result.status != ESCALATED_STATUS:
            # Spec: "Drop the message silently otherwise." This is a normal
            # business outcome, not a fault -- no DLQ, no error log spam.
            logger.debug(
                "correlation_id=%s below threshold prob=%.4f model_version=%s",
                task.correlation_id, result.probability, result.model_version,
            )
            return HandleResult(degraded_mode=result.degraded_mode, feature_vector=result.feature_vector)

        # --- stage: publish (build + validate outbound contract) ---
        source_ip = task.signal_context.triggering_source_ip or "unknown"
        now = datetime.now(timezone.utc)
        window_start = task.signal_context.window_start or now
        window_end = task.signal_context.window_end or now

        signal_ids_raw = task.signal_context.signal_ids or []
        truncated = len(signal_ids_raw) > MAX_SIGNAL_IDS
        signal_ids_kept = signal_ids_raw[:MAX_SIGNAL_IDS]

        try:
            correlation_uuid = UUID(task.correlation_id)
            signal_uuids = [UUID(s) for s in signal_ids_kept]
        except ValueError as e:
            return HandleResult(dlq=DlqInfo(payload=payload, error=e, stage="publish"))

        metadata: dict[str, Any] = {
            "strategy_name": STRATEGY_NAME,
            "linked_keys": {"ip": source_ip, "user": task.target_value},
            "degraded_mode": result.degraded_mode,
            "window_start": window_start,
            "window_end": window_end,
            "model_version": result.model_version,
        }
        if truncated:
            metadata["signal_ids_truncated"] = True
            metadata["signal_count"] = len(signal_ids_raw)

        try:
            incident = IncidentEvent(
                title=f"Graph ML Anomaly for user: {task.target_value}",
                source_ip=source_ip,
                username=task.target_value,
                protocol=None,
                severity=severity_for_score(result.risk_score),
                risk_score=result.risk_score,
                correlation_id=correlation_uuid,
                signal_ids=signal_uuids,
                tags=[],
                metadata=metadata,
                created_at=now,
                updated_at=None,
            )
        except ValidationError as e:
            # e.g. a linked_keys value blew past the 255-char contract limit.
            # This is a real failure, not a silent drop -- we scored it as an
            # incident but can't legally publish it as one.
            return HandleResult(dlq=DlqInfo(payload=payload, error=e, stage="publish"))

        logger.info(
            "ESCALATED correlation_id=%s risk_score=%d severity=%s degraded_mode=%s "
            "model_version=%s imputed=%s",
            task.correlation_id, result.risk_score, incident.severity, result.degraded_mode,
            result.model_version, result.imputed_fields,
        )
        # mode="python" (not "json") on purpose -- keeps datetime/UUID as
        # native objects so the producer's orjson serializer (OPT_UTC_Z)
        # controls the final wire format, matching the observed "...Z"
        # timestamp suffix. See shared/kafka/base_producer.py.
        return HandleResult(
            incident=incident.model_dump(mode="python"),
            degraded_mode=result.degraded_mode,
            feature_vector=result.feature_vector,
        )


class CorrelationMLConsumer(BaseConsumer):
    """
    Thin adapter between our local shared.kafka stand-in and the pure
    MLScoringHandler -- the ~20-line layer the original module docstring
    predicted.
    """

    def __init__(self, settings: Settings, handler: MLScoringHandler, monitor: Optional[FeatureDriftMonitor] = None):
        super().__init__(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            topic=settings.consume_topic,
            group_id=settings.consumer_group_id,
            dlq_topic=settings.produce_topic_dlq,
            max_consecutive_errors=settings.max_consecutive_errors,
        )
        self._handler = handler
        self._incidents_topic = settings.produce_topic_incidents
        # Optional: None when training/train_model.py hasn't been run yet
        # to produce a reference distribution -- see FeatureDriftMonitor.load().
        self._monitor = monitor

    async def process_message(self, payload: dict[str, Any]) -> None:
        result = self._handler.handle_payload(payload)

        if self._monitor is not None and result.feature_vector is not None:
            self._monitor.record(result.feature_vector)

        # 2026-08-18: moved here from MLScoringHandler.handle_payload() --
        # this counter is upstream-data-quality visibility for REAL topic
        # traffic (doc3 point 3), regardless of escalation outcome. It
        # belongs on the Kafka-specific adapter, not the pure business
        # logic, precisely because api.py's POST /score also calls
        # handle_payload() for ad-hoc test/integration requests that must
        # NOT be counted as production data-quality signal.
        if result.degraded_mode:
            DEGRADED_MESSAGES_TOTAL.inc()

        if result.incident is not None:
            await self._producer.produce(
                self._incidents_topic,
                key=str(result.incident["correlation_id"]),
                value=result.incident,
            )
        elif result.dlq is not None:
            await self.route_to_dlq(result.dlq.payload, result.dlq.error, stage=result.dlq.stage)
        # else: silent drop -- valid message, model said don't escalate
