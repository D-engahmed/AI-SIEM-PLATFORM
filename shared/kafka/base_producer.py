"""
Local stand-in for shared.kafka.base_producer.BaseProducer.

See shared/kafka/__init__.py for why this exists instead of the real
package.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import orjson
from aiokafka import AIOKafkaProducer

logger = logging.getLogger("shared.kafka.base_producer")


def _default(obj: Any) -> str:
    # BUG FIXED (2026-08-17, found by tests/test_wire_serialization.py):
    # orjson's OPT_UTC_Z only Z-suffixes datetimes that are ALREADY UTC --
    # it does NOT convert other offsets to UTC first. A non-UTC aware
    # datetime would previously serialize as "...+03:00" instead of being
    # converted and Z-suffixed, silently breaking the "always UTC Z" wire
    # format every consumer of this topic expects (verified against
    # docs/all_attack_incidents.json). Fixed by handling datetime ourselves
    # via OPT_PASSTHROUGH_DATETIME below, converting to UTC explicitly.
    if isinstance(obj, datetime):
        aware = obj if obj.tzinfo is not None else obj.replace(tzinfo=timezone.utc)
        return aware.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    raise TypeError(f"not JSON serializable: {type(obj)}")


def _serialize(value: dict[str, Any]) -> bytes:
    # OPT_PASSTHROUGH_DATETIME hands every datetime to `_default` above
    # instead of letting orjson's native (UTC-only-aware) datetime handling
    # run. Callers must pass model_dump(mode="python") dicts, not mode="json"
    # ones (mode="json" would have already stringified the datetimes before
    # we get a chance to fix their offset).
    return orjson.dumps(value, default=_default, option=orjson.OPT_PASSTHROUGH_DATETIME)


class BaseProducer:
    """Thin async wrapper around aiokafka.AIOKafkaProducer with orjson
    serialization. One instance per service process; start() once, stop()
    once, produce() many times."""

    def __init__(self, *, bootstrap_servers: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._bootstrap_servers,
            value_serializer=_serialize,
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
        )
        await self._producer.start()
        logger.info("producer started bootstrap_servers=%s", self._bootstrap_servers)

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            logger.info("producer stopped cleanly")

    async def produce(self, topic: str, *, key: Optional[str], value: dict[str, Any]) -> None:
        """Produce one message and wait for broker acknowledgment. Raises on
        failure -- callers decide what "failure" means for their own
        fail-open/fail-closed policy (see base_consumer.py's DLQ handling
        for why DLQ-produce failures specifically are NOT treated the same
        as incident-produce failures)."""
        assert self._producer is not None, "call start() before produce()"
        await self._producer.send_and_wait(topic, value=value, key=key)
