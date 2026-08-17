"""
Tests for the outbound `incidents` contract in schemas.py -- the bounds
and formatting rules the spec states explicitly, checked directly rather
than only incidentally through ml_consumer.py's tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schemas import MAX_KEY_LEN, MAX_LINKED_KEYS, MAX_VALUE_LEN, IncidentEvent


def _valid_incident(**overrides) -> dict:
    base = dict(
        correlation_id="11111111-1111-1111-1111-111111111111",
        linked_keys={"ip": "203.0.113.1", "user": "alice"},
        signal_ids=["sig-1"],
        risk_score=742.0,
        degraded_mode=False,
        window_start=datetime(2026, 8, 12, 10, 15, tzinfo=timezone.utc),
        window_end=datetime(2026, 8, 12, 10, 25, tzinfo=timezone.utc),
        created_at=datetime(2026, 8, 12, 10, 25, 3, tzinfo=timezone.utc),
        model_version="test-version",
    )
    base.update(overrides)
    return base


class TestRiskScoreBounds:
    @pytest.mark.parametrize("score", [0.0, 500.0, 1000.0])
    def test_in_range_accepted(self, score):
        IncidentEvent(**_valid_incident(risk_score=score))

    @pytest.mark.parametrize("score", [-0.01, 1000.01, -500.0, 5000.0])
    def test_out_of_range_rejected(self, score):
        with pytest.raises(ValidationError):
            IncidentEvent(**_valid_incident(risk_score=score))


class TestLinkedKeysBounds:
    def test_too_many_keys_rejected(self):
        too_many = {f"k{i}": "v" for i in range(MAX_LINKED_KEYS + 1)}
        with pytest.raises(ValidationError):
            IncidentEvent(**_valid_incident(linked_keys=too_many))

    def test_max_keys_exactly_accepted(self):
        exactly_max = {f"k{i}": "v" for i in range(MAX_LINKED_KEYS)}
        IncidentEvent(**_valid_incident(linked_keys=exactly_max))

    def test_key_over_max_len_rejected(self):
        with pytest.raises(ValidationError):
            IncidentEvent(**_valid_incident(linked_keys={"k" * (MAX_KEY_LEN + 1): "v"}))

    def test_value_over_max_len_rejected(self):
        with pytest.raises(ValidationError):
            IncidentEvent(**_valid_incident(linked_keys={"user": "u" * (MAX_VALUE_LEN + 1)}))

    def test_value_at_max_len_accepted(self):
        IncidentEvent(**_valid_incident(linked_keys={"user": "u" * MAX_VALUE_LEN}))


class TestWireFormat:
    def test_strategy_name_defaults_correctly(self):
        incident = IncidentEvent(**_valid_incident())
        assert incident.to_wire_dict()["strategy_name"] == "GraphMLScoring"

    def test_status_defaults_to_escalated(self):
        incident = IncidentEvent(**_valid_incident())
        assert incident.to_wire_dict()["status"] == "ESCALATED_TO_INCIDENT"

    def test_model_version_excluded_from_wire_dict(self):
        # Internal audit field -- must never leak onto the published topic.
        incident = IncidentEvent(**_valid_incident(model_version="secret-internal-tag"))
        wire = incident.to_wire_dict()
        assert "model_version" not in wire
        assert "secret-internal-tag" not in str(wire)

    def test_timestamps_are_z_suffixed_with_microseconds(self):
        incident = IncidentEvent(**_valid_incident())
        wire = incident.to_wire_dict()
        for field in ("window_start", "window_end", "created_at"):
            value = wire[field]
            assert value.endswith("Z")
            assert "+00:00" not in value  # matches the observed strategy_name-tagged
            assert "." in value           # incident wire format, not Python's default isoformat()

    def test_updated_at_null_when_not_set(self):
        incident = IncidentEvent(**_valid_incident())
        assert incident.to_wire_dict()["updated_at"] is None

    def test_naive_datetime_treated_as_utc(self):
        # Defensive: if something upstream forgets a tzinfo, we must not
        # silently reinterpret the clock time as a different zone.
        naive = datetime(2026, 8, 12, 10, 25, 3)
        incident = IncidentEvent(**_valid_incident(created_at=naive))
        assert incident.to_wire_dict()["created_at"] == "2026-08-12T10:25:03.000000Z"
