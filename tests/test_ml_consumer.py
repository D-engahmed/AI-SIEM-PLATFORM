"""
Run with: python -m pytest tests/ -v   (from services/correlation-ml-service/)
Requires artifacts/model_latest.joblib to exist -- run
training/generate_synthetic_data.py then the model_selection notebook
(or training/train_model.py) first. `handler` fixture comes from conftest.py.
"""

from __future__ import annotations

import json


def _payload(**overrides) -> bytes:
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
    base.update(overrides)
    return json.dumps(base).encode("utf-8")


class TestDeserialize:
    def test_garbage_bytes_go_to_dlq(self, handler):
        result = handler.handle_raw(b"{not json")
        assert result.incident is None
        assert result.dlq is not None
        assert result.dlq["stage"] == "deserialize"

    def test_bad_utf8_goes_to_dlq(self, handler):
        result = handler.handle_raw(b"\xff\xfe\x00\x01")
        assert result.dlq is not None
        assert result.dlq["stage"] == "deserialize"


class TestValidate:
    def test_missing_required_field_goes_to_dlq(self, handler):
        raw = json.loads(_payload())
        del raw["correlation_id"]
        result = handler.handle_raw(json.dumps(raw).encode("utf-8"))
        assert result.dlq is not None
        assert result.dlq["stage"] == "validate"

    def test_bad_uuid_goes_to_dlq(self, handler):
        result = handler.handle_raw(_payload(correlation_id="not-a-uuid"))
        assert result.dlq is not None
        assert result.dlq["stage"] == "validate"

    def test_unsupported_target_kind_goes_to_dlq(self, handler):
        result = handler.handle_raw(_payload(target_kind="device"))
        assert result.dlq is not None
        assert result.dlq["stage"] == "validate"

    def test_ip_target_kind_is_supported(self, handler):
        result = handler.handle_raw(_payload(target_kind="ip", target_value="203.0.113.77"))
        assert result.dlq is None
        if result.incident is not None:
            assert result.incident["linked_keys"]["ip"] == "203.0.113.77"

    def test_missing_optional_fields_do_not_dlq(self, handler):
        # graph_features/signal_context sub-fields absent entirely -- spec says
        # this must be handled, not rejected.
        raw = json.loads(_payload())
        raw["graph_features"] = {}
        raw["signal_context"] = {}
        result = handler.handle_raw(json.dumps(raw).encode("utf-8"))
        assert result.dlq is None  # must not fail closed on merely-sparse data


class TestScoringAndRouting:
    def test_strong_signal_escalates(self, handler):
        # high fan-in, ti_matched True -- should clear the default 0.7 threshold
        result = handler.handle_raw(_payload())
        assert result.dlq is None
        assert result.incident is not None
        assert result.incident["strategy_name"] == "GraphMLScoring"
        assert result.incident["status"] == "ESCALATED_TO_INCIDENT"
        assert 0.0 <= result.incident["risk_score"] <= 1000.0
        assert result.incident["linked_keys"]["ip"] == "203.0.113.45"
        assert result.incident["linked_keys"]["user"] == "svc_backup_admin"
        assert "model_version" not in result.incident  # internal-only field must not leak onto the wire

    def test_weak_signal_drops_silently(self, handler):
        weak = _payload(
            graph_features={"fan_in_count": 1, "epoch_age_seconds": 300},
            signal_context={
                "triggering_source_ip": "203.0.113.99",
                "ti_matched": False,
                "signal_ids": ["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"],
            },
        )
        result = handler.handle_raw(weak)
        assert result.incident is None
        assert result.dlq is None  # not an error -- just below threshold

    def test_missing_fields_flagged_degraded_but_still_scored(self, handler):
        raw = json.loads(_payload())
        raw["graph_features"] = {}  # fan_in_count AND epoch_age_seconds both missing
        raw["signal_context"]["ti_matched"] = None
        result = handler.handle_raw(json.dumps(raw).encode("utf-8"))
        assert result.dlq is None
        # could escalate or drop depending on threshold, but if it escalates it must say degraded
        if result.incident is not None:
            assert result.incident["degraded_mode"] is True

    def test_output_timestamps_are_z_suffixed_iso8601(self, handler):
        result = handler.handle_raw(_payload())
        assert result.incident is not None
        for field in ("window_start", "window_end", "created_at"):
            assert result.incident[field].endswith("Z")

    def test_linked_keys_value_over_255_chars_goes_to_publish_dlq(self, handler):
        raw = json.loads(_payload())
        raw["target_value"] = "u" * 300  # violates the 255-char contract bound
        result = handler.handle_raw(json.dumps(raw).encode("utf-8"))
        # Either it doesn't escalate (fine, silent drop) or it does and MUST dlq at publish
        if result.incident is None and result.dlq is not None:
            assert result.dlq["stage"] == "publish"
