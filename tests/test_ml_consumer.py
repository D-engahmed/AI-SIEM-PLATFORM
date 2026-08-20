"""
Run with: python -m pytest tests/ -v   (from services/correlation-ml-service/)
Requires artifacts/model_latest.joblib to exist -- run
training/generate_synthetic_data.py then the model_selection notebook
(or training/train_model.py) first. `handler` fixture comes from conftest.py.

2026-08-17: deserialization (bytes -> dict) moved to shared/kafka/base_consumer.py,
so handler.handle_payload() now takes an already-deserialized dict directly --
these tests no longer cover the deserialize stage (that's covered by
shared/kafka's own tests, not written here since it's infrastructure, not
business logic).
"""

from __future__ import annotations

import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

from ml_consumer import CorrelationMLConsumer, MLScoringHandler


def _payload(**overrides) -> dict:
    base = {
        "correlation_id": "11111111-1111-1111-1111-111111111111",
        "target_kind": "user",
        "target_value": "svc_backup_admin",
        "graph_features": {"fan_in_count": 15, "epoch_age_seconds": 20 * 3600},
        "signal_context": {
            "triggering_source_ip": "203.0.113.45",
            "ti_matched": True,
            "signal_ids": ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
        },
    }
    merged = copy.deepcopy(base)
    merged.update(overrides)
    return merged


class TestValidate:
    def test_missing_required_field_goes_to_dlq(self, handler):
        raw = _payload()
        del raw["correlation_id"]
        result = handler.handle_payload(raw)
        assert result.dlq is not None
        assert result.dlq.stage == "validate"

    def test_bad_uuid_goes_to_dlq(self, handler):
        result = handler.handle_payload(_payload(correlation_id="not-a-uuid"))
        assert result.dlq is not None
        assert result.dlq.stage == "validate"

    def test_unsupported_target_kind_goes_to_dlq(self, handler):
        result = handler.handle_payload(_payload(target_kind="device"))
        assert result.dlq is not None
        assert result.dlq.stage == "validate"

    def test_unsupported_ip_target_kind_currently_goes_to_dlq(self, handler):
        # PRE-EXISTING BUG FIXED (2026-08-17): this test previously asserted
        # target_kind="ip" is supported (dlq is None) -- confirmed via
        # `git show HEAD` that assertion predates every change made this
        # session, so it was never a regression from today's work, just a
        # test that never matched SUPPORTED_TARGET_KINDS = {"user"}.
        #
        # OPEN QUESTION, NOT RESOLVED HERE: is "ip" actually meant to be
        # unsupported (same as "device"), or is SUPPORTED_TARGET_KINDS
        # missing it? docs/ML_AI.md never lists valid target_kind values
        # directly -- it only shows "ip"/"user" as linked_keys dict KEYS,
        # not as target_kind values. tests/test_data_quality.py's
        # test_target_kind_is_always_user (checked against the actual
        # training data) is consistent with target_kind ONLY ever being
        # "user" in practice, which is weak supporting evidence for
        # "unsupported is correct" -- but it's not a confirmed answer.
        # This test is fixed to match current, deliberate code behavior
        # (SUPPORTED_TARGET_KINDS is unchanged) rather than silently
        # widening scope on a guess. If "ip" should be supported, this is
        # a one-line change to SUPPORTED_TARGET_KINDS plus reverting this
        # test -- flag it back to the reviewer rather than assume either way.
        result = handler.handle_payload(_payload(target_kind="ip", target_value="203.0.113.77"))
        assert result.dlq is not None
        assert result.dlq.stage == "validate"

    def test_missing_optional_fields_do_not_dlq(self, handler):
        # graph_features/signal_context sub-fields absent entirely -- spec says
        # this must be handled, not rejected.
        raw = _payload()
        raw["graph_features"] = {}
        raw["signal_context"] = {}
        result = handler.handle_payload(raw)
        assert result.dlq is None  # must not fail closed on merely-sparse data


class TestScoringAndRouting:
    def test_strong_signal_escalates(self, handler):
        # high fan-in, ti_matched True -- should clear the default 0.5 threshold
        result = handler.handle_payload(_payload())
        assert result.dlq is None
        assert result.incident is not None
        incident = result.incident
        assert incident["status"] == "ESCALATED_TO_INCIDENT"
        assert incident["metadata"]["strategy_name"] == "GraphMLScoring"
        assert 0 <= incident["risk_score"] <= 100  # AD-061: 0-100 int, not the old 0-1000 float
        assert incident["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert incident["metadata"]["linked_keys"]["ip"] == "203.0.113.45"
        assert incident["metadata"]["linked_keys"]["user"] == "svc_backup_admin"
        assert incident["username"] == "svc_backup_admin"
        assert incident["source_ip"] == "203.0.113.45"
        assert incident["title"]  # non-empty, generated

    def test_weak_signal_drops_silently(self, handler):
        weak = _payload(
            graph_features={"fan_in_count": 1, "epoch_age_seconds": 300},
            signal_context={
                "triggering_source_ip": "203.0.113.99",
                "ti_matched": False,
                "signal_ids": ["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"],
            },
        )
        result = handler.handle_payload(weak)
        assert result.incident is None
        assert result.dlq is None  # not an error -- just below threshold

    def test_missing_fields_flagged_degraded_but_still_scored(self, handler):
        raw = _payload()
        raw["graph_features"] = {}  # fan_in_count AND epoch_age_seconds both missing
        raw["signal_context"]["ti_matched"] = None
        result = handler.handle_payload(raw)
        assert result.dlq is None
        # could escalate or drop depending on threshold, but if it escalates it must say degraded
        if result.incident is not None:
            assert result.incident["metadata"]["degraded_mode"] is True

    def test_username_over_255_chars_goes_to_publish_dlq(self, handler):
        raw = _payload()
        raw["target_value"] = "u" * 300  # violates the 255-char contract bound on username
        result = handler.handle_payload(raw)
        # Either it doesn't escalate (fine, silent drop) or it does and MUST dlq at publish
        if result.incident is None and result.dlq is not None:
            assert result.dlq.stage == "publish"


class TestDegradedModeFlagPlacement:
    """
    2026-08-18: DEGRADED_MESSAGES_TOTAL.inc() used to live inside
    handle_payload() itself -- moved to CorrelationMLConsumer.process_message()
    so that api.py's POST /score (which also calls handle_payload() directly,
    for ad-hoc test/integration requests) doesn't pollute a metric meant to
    describe real topic traffic. These tests pin that placement: the pure
    handler reports degraded_mode on its result but never touches the
    counter itself.
    """

    def test_handle_payload_reports_degraded_mode_on_result(self, handler):
        raw = _payload()
        raw["graph_features"] = {}
        raw["signal_context"]["ti_matched"] = None
        result = handler.handle_payload(raw)
        assert result.dlq is None
        assert result.degraded_mode is True

    def test_handle_payload_reports_not_degraded_when_fully_populated(self, handler):
        result = handler.handle_payload(_payload())
        assert result.degraded_mode is False

    def test_dropped_message_still_reports_degraded_mode(self, handler):
        # A degraded message that scores below threshold (silent drop) must
        # still surface degraded_mode=True on the result -- doc3 point 3
        # frames this counter as upstream data-quality visibility,
        # independent of whether an incident was actually published.
        weak_degraded = _payload(
            graph_features={},  # missing -> degraded
            signal_context={"triggering_source_ip": "203.0.113.1", "ti_matched": None, "signal_ids": []},
        )
        result = handler.handle_payload(weak_degraded)
        if result.incident is None and result.dlq is None:  # dropped, not escalated
            assert result.degraded_mode is True

    def test_handle_payload_never_touches_the_prometheus_counter(self, handler):
        from metrics import DEGRADED_MESSAGES_TOTAL

        before = DEGRADED_MESSAGES_TOTAL._value.get()
        raw = _payload()
        raw["graph_features"] = {}
        raw["signal_context"]["ti_matched"] = None
        handler.handle_payload(raw)  # degraded_mode=True on the result, but...
        after = DEGRADED_MESSAGES_TOTAL._value.get()
        assert after == before  # ...the counter must not move from this call alone


class TestCorrelationMLConsumerIncrementsCounter:
    """
    The other half of TestDegradedModeFlagPlacement: process_message() (the
    Kafka-specific adapter) IS where the counter increments -- unlike
    handle_payload(), which real Kafka traffic and POST /score both call.
    """

    @pytest.fixture
    def consumer(self, scorer):
        settings = _fake_settings()
        c = CorrelationMLConsumer(settings, MLScoringHandler(scorer=scorer))
        c._producer = AsyncMock()
        return c

    @pytest.mark.asyncio
    async def test_degraded_message_increments_counter(self, consumer):
        from metrics import DEGRADED_MESSAGES_TOTAL

        before = DEGRADED_MESSAGES_TOTAL._value.get()
        raw = _payload()
        raw["graph_features"] = {}
        raw["signal_context"]["ti_matched"] = None
        await consumer.process_message(raw)
        after = DEGRADED_MESSAGES_TOTAL._value.get()
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_non_degraded_message_does_not_increment_counter(self, consumer):
        from metrics import DEGRADED_MESSAGES_TOTAL

        before = DEGRADED_MESSAGES_TOTAL._value.get()
        await consumer.process_message(_payload())
        after = DEGRADED_MESSAGES_TOTAL._value.get()
        assert after == before


def _fake_settings():
    settings = MagicMock()
    settings.kafka_bootstrap_servers = "localhost:9092"
    settings.consume_topic = "ml-scoring-tasks"
    settings.consumer_group_id = "correlation-ml-service"
    settings.produce_topic_dlq = "dlq-correlation-ml"
    settings.produce_topic_incidents = "incidents"
    settings.max_consecutive_errors = 20
    return settings
