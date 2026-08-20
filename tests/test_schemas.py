"""
Tests for the outbound `incidents` contract in schemas.py -- the bounds
and formatting rules the spec (AD-061, see schemas.py module docstring)
states explicitly, checked directly rather than only incidentally through
ml_consumer.py's tests.

2026-08-17: rewritten for the AD-061 root-field shape (title/source_ip/
username/severity/risk_score at root, strategy-specific data nested in
metadata). The pre-rewrite version of this file tested the old 0-1000
risk_score range, root-level linked_keys/degraded_mode/window_start/
window_end, and a to_wire_dict() method that no longer exists -- none of
that is a valid contract anymore, not a case of the new code being wrong.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schemas import (
    MAX_KEY_LEN,
    MAX_LINKED_KEYS,
    MAX_VALUE_LEN,
    SEVERITY_VALUES,
    IncidentEvent,
    severity_for_score,
)


def _valid_incident(**overrides) -> dict:
    base = dict(
        title="Graph ML Anomaly for user: alice",
        source_ip="203.0.113.1",
        username="alice",
        protocol=None,
        severity="HIGH",
        risk_score=74,
        correlation_id="11111111-1111-1111-1111-111111111111",
        signal_ids=["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
        tags=[],
        metadata={
            "strategy_name": "GraphMLScoring",
            "linked_keys": {"ip": "203.0.113.1", "user": "alice"},
            "degraded_mode": False,
            "window_start": datetime(2026, 8, 12, 10, 15, tzinfo=timezone.utc),
            "window_end": datetime(2026, 8, 12, 10, 25, tzinfo=timezone.utc),
            "model_version": "test-version",
        },
        created_at=datetime(2026, 8, 12, 10, 25, 3, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return base


class TestRiskScoreBounds:
    @pytest.mark.parametrize("score", [0, 50, 100])
    def test_in_range_accepted(self, score):
        IncidentEvent(**_valid_incident(risk_score=score))

    @pytest.mark.parametrize("score", [-1, 101, -50, 1000])
    def test_out_of_range_rejected(self, score):
        with pytest.raises(ValidationError):
            IncidentEvent(**_valid_incident(risk_score=score))


class TestSeverityEnum:
    @pytest.mark.parametrize("value", SEVERITY_VALUES)
    def test_valid_values_accepted(self, value):
        IncidentEvent(**_valid_incident(severity=value))

    def test_invalid_value_rejected(self):
        with pytest.raises(ValidationError):
            IncidentEvent(**_valid_incident(severity="URGENT"))

    def test_lowercase_rejected(self):
        # Enum is case-sensitive on purpose -- silently normalizing case
        # would hide a real producer bug on whichever side sends it wrong.
        with pytest.raises(ValidationError):
            IncidentEvent(**_valid_incident(severity="high"))


class TestSeverityForScore:
    # PLACEHOLDER mapping (see schemas.py docstring) -- these tests pin the
    # band edges so a change is a visible diff here, not a silent drift.
    @pytest.mark.parametrize(
        "score,expected",
        [(0, "LOW"), (24, "LOW"), (25, "MEDIUM"), (49, "MEDIUM"),
         (50, "HIGH"), (74, "HIGH"), (75, "CRITICAL"), (100, "CRITICAL")],
    )
    def test_band_edges(self, score, expected):
        assert severity_for_score(score) == expected


class TestLinkedKeysBounds:
    # linked_keys now lives inside metadata; bounds are enforced by
    # IncidentEvent._metadata_linked_keys_bounds, not a dedicated field.
    def _incident_with_linked_keys(self, linked_keys: dict) -> dict:
        base = _valid_incident()
        base["metadata"] = {**base["metadata"], "linked_keys": linked_keys}
        return base

    def test_too_many_keys_rejected(self):
        too_many = {f"k{i}": "v" for i in range(MAX_LINKED_KEYS + 1)}
        with pytest.raises(ValidationError):
            IncidentEvent(**self._incident_with_linked_keys(too_many))

    def test_max_keys_exactly_accepted(self):
        exactly_max = {f"k{i}": "v" for i in range(MAX_LINKED_KEYS)}
        IncidentEvent(**self._incident_with_linked_keys(exactly_max))

    def test_key_over_max_len_rejected(self):
        with pytest.raises(ValidationError):
            IncidentEvent(**self._incident_with_linked_keys({"k" * (MAX_KEY_LEN + 1): "v"}))

    def test_value_over_max_len_rejected(self):
        with pytest.raises(ValidationError):
            IncidentEvent(
                **self._incident_with_linked_keys({"user": "u" * (MAX_VALUE_LEN + 1)})
            )

    def test_value_at_max_len_accepted(self):
        IncidentEvent(**self._incident_with_linked_keys({"user": "u" * MAX_VALUE_LEN}))

    def test_missing_linked_keys_does_not_error(self):
        # metadata without a linked_keys entry at all must not crash the
        # validator -- e.g. a DLQ-adjacent or degraded-only record.
        base = _valid_incident()
        del base["metadata"]["linked_keys"]
        IncidentEvent(**base)


class TestRootVsMetadataPlacement:
    def test_strategy_specific_fields_not_required_at_root(self):
        # AD-061: these must NOT be root kwargs anymore. Passing them at
        # root should either be ignored (extra='ignore' default) or raise --
        # either way, a real IncidentEvent built without them at root must
        # validate fine using only the documented root fields.
        incident = IncidentEvent(**_valid_incident())
        assert incident.metadata["strategy_name"] == "GraphMLScoring"
        assert incident.metadata["linked_keys"] == {"ip": "203.0.113.1", "user": "alice"}
        assert incident.metadata["degraded_mode"] is False

    def test_username_maps_from_target_value_max_length(self):
        with pytest.raises(ValidationError):
            IncidentEvent(**_valid_incident(username="u" * 256))

    def test_protocol_defaults_to_none(self):
        base = _valid_incident()
        del base["protocol"]
        incident = IncidentEvent(**base)
        assert incident.protocol is None

    def test_tags_default_to_empty_list(self):
        base = _valid_incident()
        del base["tags"]
        incident = IncidentEvent(**base)
        assert incident.tags == []

    def test_status_defaults_to_escalated(self):
        incident = IncidentEvent(**_valid_incident())
        assert incident.status == "ESCALATED_TO_INCIDENT"

    def test_updated_at_null_when_not_set(self):
        incident = IncidentEvent(**_valid_incident())
        assert incident.updated_at is None

    def test_naive_datetime_treated_as_utc(self):
        # Defensive: if something upstream forgets a tzinfo, we must not
        # silently reinterpret the clock time as a different zone.
        naive = datetime(2026, 8, 12, 10, 25, 3)
        incident = IncidentEvent(**_valid_incident(created_at=naive))
        assert incident.created_at.tzinfo is None  # pydantic keeps it naive as given
        # The "must render as UTC Z" guarantee now lives in the producer's
        # orjson serialization (OPT_UTC_Z), not here -- see
        # tests/test_wire_serialization.py.
