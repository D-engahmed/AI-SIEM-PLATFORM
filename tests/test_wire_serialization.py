"""
Tests the producer's actual wire serialization (orjson + OPT_UTC_Z) --
fills the gap left when the old to_wire_dict()/_iso_z() unit tests were
removed from test_schemas.py. Those functions no longer exist; the "Z"
suffix requirement observed in docs/all_attack_incidents.json is now
enforced by shared/kafka/base_producer.py's _serialize(), so that's what
this file checks, directly, without needing a live Kafka broker.
"""
from __future__ import annotations

import orjson
import pytest

from shared.kafka.base_producer import _serialize


class TestSerializeTimestamps:
    def test_utc_datetime_gets_z_suffix(self):
        from datetime import datetime, timezone

        value = {"created_at": datetime(2026, 8, 12, 10, 25, 3, 123456, tzinfo=timezone.utc)}
        wire = orjson.loads(_serialize(value))
        assert wire["created_at"] == "2026-08-12T10:25:03.123456Z"
        assert "+00:00" not in wire["created_at"]

    def test_non_utc_datetime_converted_then_z_suffixed(self):
        from datetime import datetime, timedelta, timezone

        tz = timezone(timedelta(hours=3))  # e.g. EEST
        value = {"created_at": datetime(2026, 8, 12, 13, 25, 3, tzinfo=tz)}
        wire = orjson.loads(_serialize(value))
        # 13:25 at UTC+3 == 10:25 UTC
        assert wire["created_at"] == "2026-08-12T10:25:03.000000Z"

    def test_uuid_serializes_as_plain_string(self):
        from uuid import UUID

        value = {"correlation_id": UUID("11111111-1111-1111-1111-111111111111")}
        wire = orjson.loads(_serialize(value))
        assert wire["correlation_id"] == "11111111-1111-1111-1111-111111111111"

    def test_null_updated_at_stays_null(self):
        value = {"updated_at": None}
        wire = orjson.loads(_serialize(value))
        assert wire["updated_at"] is None
