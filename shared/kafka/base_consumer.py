"""
Local stand-in for shared.kafka.base_consumer.BaseConsumer.

See shared/kafka/__init__.py for why this exists instead of the real
package.

Contract subclasses get (matches what the reviewer described for the
real class):
  - override `async def process_message(self, payload: dict) -> None`.
    Raise any exception to have it routed to DLQ automatically.
  - call `await self.route_to_dlq(payload, error)` directly if you want
    to DLQ something without raising (e.g. a business-logic decision,
    not an unexpected failure). This local version also accepts an
    optional `stage` kwarg for observability -- that's an addition
    beyond the 2-arg signature we were given, not a verified part of the
    real interface, so double check it survives the eventual swap.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import orjson
from aiokafka import AIOKafkaConsumer

from shared.kafka.base_producer import BaseProducer

logger = logging.getLogger("shared.kafka.base_consumer")


def _deserialize(raw: Optional[bytes]) -> Any:
    if not raw:
        return None
    return orjson.loads(raw)


class BaseConsumer(ABC):
    """
    Owns the poll loop, orjson deserialization, offset commits, and
    fail-closed error counting. Subclasses only implement
    `process_message`.
    """

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        dlq_topic: str,
        max_consecutive_errors: int = 20,
    ) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._dlq_topic = dlq_topic
        self._max_consecutive_errors = max_consecutive_errors

        self._consumer: Optional[AIOKafkaConsumer] = None
        self._producer = BaseProducer(bootstrap_servers=bootstrap_servers)
        self._consecutive_errors = 0
        self._running = False

    @abstractmethod
    async def process_message(self, payload: dict[str, Any]) -> None:
        """Subclasses implement business logic here."""
        raise NotImplementedError

    async def route_to_dlq(
        self, payload: dict[str, Any], error: Exception, *, stage: str = "unknown"
    ) -> None:
        envelope = {
            "original_payload": payload,
            "error_type": type(error).__name__,
            # bounded -- error string is partly attacker-influenced
            "error_message": str(error)[:2000],
            "stage": stage,
            "consumer_group": self._group_id,
            "failed_at": datetime.now(timezone.utc),
        }
        try:
            await self._producer.produce(self._dlq_topic, key=None, value=envelope)
        except Exception as dlq_err:  # noqa: BLE001
            # Explicit design decision (raised as an open question, then
            # confirmed as the right call): DLQ being unreachable must not
            # crash the main consumer loop. Log loudly and move on -- do
            # NOT count this toward _consecutive_errors, since that
            # counter exists to fail-closed on *processing* problems, not
            # DLQ infrastructure problems. The original message that
            # triggered this DLQ attempt is lost either way at this point
            # (we already committed past it or are about to); that loss is
            # visible in these error logs, which is the tradeoff being
            # made here.
            logger.error(
                "DLQ produce failed (not counted toward fail-closed threshold): %s", dlq_err
            )

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=_deserialize,
        )
        await self._consumer.start()
        await self._producer.start()
        self._running = True
        logger.info("consumer started topic=%s group=%s", self._topic, self._group_id)

    async def stop(self) -> None:
        self._running = False
        if self._consumer is not None:
            await self._consumer.stop()
        await self._producer.stop()
        logger.info("consumer stopped cleanly")

    async def run_forever(self) -> None:
        assert self._consumer is not None, "call start() before run_forever()"
        async for msg in self._consumer:
            if not self._running:
                break

            payload = msg.value
            if payload is None:
                # Deserialize failure or empty message -- can't tell which
                # stage failed inside process_message since we never got
                # that far, so this one really is "unknown".
                err = ValueError("empty or unparseable message value")
                await self.route_to_dlq({"_raw_undecodable": True}, err, stage="deserialize")
                await self._consumer.commit()
                continue

            try:
                await self.process_message(payload)
                self._consecutive_errors = 0
            except Exception as e:  # noqa: BLE001 -- business logic errors are genuinely unpredictable
                logger.error("process_message failed: %s", e)
                self._consecutive_errors += 1
                await self.route_to_dlq(payload, e, stage="process_message")
                if self._consecutive_errors >= self._max_consecutive_errors:
                    raise RuntimeError(
                        "too many consecutive processing errors, failing closed"
                    ) from e

            await self._consumer.commit()
