"""
Prometheus metrics for correlation-ml-service.

ASSUMPTION FLAG: this registry lives in-process, and main.py runs the
Kafka consumer and the /metrics HTTP endpoint on the SAME asyncio event
loop specifically so DEGRADED_MESSAGES_TOTAL is the exact same Python
object in both places. config.py's own pre-existing comment calls the
API service "a separate process from the Kafka consumer" -- that's not
what's implemented here, because a genuinely separate OS process can't
increment an in-memory Counter living in a different process. If this
service is ever scaled to multiple consumer replicas behind one metrics
endpoint, this needs prometheus_client's multiprocess mode
(PROMETHEUS_MULTIPROC_DIR) instead -- flagged, not implemented, since
that's an infra decision, not a default worth guessing at.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter

REGISTRY = CollectorRegistry()

DEGRADED_MESSAGES_TOTAL = Counter(
    "cml_degraded_messages_total",
    "Messages scored with degraded_mode=true (imputed fields met/exceeded the configured threshold).",
    registry=REGISTRY,
)
