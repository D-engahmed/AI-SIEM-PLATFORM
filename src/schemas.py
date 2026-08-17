"""
Wire contracts for correlation-ml-service.

ASSUMPTION FLAG (read this before trusting the input model):
--------------------------------------------------------------
The task spec describes the `ml-scoring-tasks` payload only in prose
(graph_features, signal_context, fan_in_count, epoch_age_seconds,
ti_matched, target_kind). No real sample message was provided -- the
uploaded `all_attack_incidents.json` is downstream *output* from other
rule-based strategies (RiskBasedAlertingStrategy, BruteThenLoginPattern,
etc.), not an ml-scoring-tasks input example, and it contains zero
GraphPivotStrategy / GraphMLScoring records.

So the exact field names below (`triggering_source_ip`, the nesting of
`graph_features` / `signal_context`, etc.) are a reasonable reconstruction
from the prose spec, not a verified contract. Every upstream field name is
declared in ONE place (this file) specifically so that when you show me a
real sample message, fixing it is a one-file diff, not a rewrite.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_LINKED_KEYS = 10
MAX_KEY_LEN = 50
MAX_VALUE_LEN = 255

STRATEGY_NAME = "GraphMLScoring"
ESCALATED_STATUS = "ESCALATED_TO_INCIDENT"


# --------------------------------------------------------------------------
# Inbound: ml-scoring-tasks
# --------------------------------------------------------------------------

class GraphFeatures(BaseModel):
    """
    Graph-side signal from the upstream GraphPivotStrategy.

    fan_in_count and epoch_age_seconds are the two fields the spec calls
    out explicitly. Everything here is Optional on purpose -- the spec
    says upstream does not always populate every field, and a strict
    model would turn a normal partial payload into a DLQ event.
    """

    model_config = ConfigDict(extra="allow")  # never crash on unknown upstream fields

    fan_in_count: Optional[int] = Field(
        default=None, description="Distinct source IPs seen touching this target so far."
    )
    epoch_age_seconds: Optional[float] = Field(
        default=None, description="Seconds since this target node was created in the graph (resets at 24h)."
    )

    @field_validator("fan_in_count")
    @classmethod
    def _non_negative_fan_in(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("fan_in_count cannot be negative")
        return v

    @field_validator("epoch_age_seconds")
    @classmethod
    def _non_negative_age(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("epoch_age_seconds cannot be negative")
        return v


class SignalContext(BaseModel):
    """
    Signal-side context for the convergence event.

    ASSUMPTION: `triggering_source_ip` is the IP that pushed fan-in to a
    new high -- I could not find this field name confirmed anywhere. If
    the real field is named differently, update it here only.
    """

    model_config = ConfigDict(extra="allow")

    triggering_source_ip: Optional[str] = None
    ti_matched: Optional[bool] = Field(
        default=None, description="True/False from threat-intel feed match. None = no TI check performed."
    )
    signal_ids: list[str] = Field(default_factory=list)
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None


class MLScoringTask(BaseModel):
    """Top-level message consumed from ml-scoring-tasks."""

    model_config = ConfigDict(extra="allow")

    correlation_id: str
    target_kind: str
    target_value: str
    graph_features: GraphFeatures = Field(default_factory=GraphFeatures)
    signal_context: SignalContext = Field(default_factory=SignalContext)

    @field_validator("correlation_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str:
        # Validate shape without forcing UUID type -- keep it a plain str
        # downstream so we never accidentally reject a valid-but-unusual id.
        UUID(v)
        return v


# --------------------------------------------------------------------------
# Outbound: incidents
# --------------------------------------------------------------------------

class IncidentEvent(BaseModel):
    """
    Exact output contract for the `incidents` topic. Field order here
    matches the spec's example so serialized output is easy to diff
    against it by eye.
    """

    correlation_id: str
    strategy_name: str = STRATEGY_NAME
    linked_keys: dict[str, str]
    signal_ids: list[str]
    risk_score: float
    status: str = ESCALATED_STATUS
    degraded_mode: bool
    window_start: datetime
    window_end: datetime
    created_at: datetime
    updated_at: Optional[datetime] = None
    model_version: str = Field(exclude=True)  # internal audit field, NOT part of wire schema

    @field_validator("risk_score")
    @classmethod
    def _score_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1000.0):
            raise ValueError(f"risk_score {v} out of [0, 1000]")
        return v

    @field_validator("linked_keys")
    @classmethod
    def _linked_keys_bounds(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > MAX_LINKED_KEYS:
            raise ValueError(f"linked_keys has {len(v)} entries, max {MAX_LINKED_KEYS}")
        for k, val in v.items():
            if len(k) > MAX_KEY_LEN:
                raise ValueError(f"linked_keys key '{k[:20]}...' exceeds {MAX_KEY_LEN} chars")
            if len(val) > MAX_VALUE_LEN:
                raise ValueError(f"linked_keys value for '{k}' exceeds {MAX_VALUE_LEN} chars")
        return v

    def to_wire_dict(self) -> dict:
        """JSON-safe dict matching the exact published schema (Z-suffixed UTC ISO8601)."""
        d = self.model_dump(mode="json", exclude={"model_version"})
        for field in ("window_start", "window_end", "created_at", "updated_at"):
            if d.get(field) is not None:
                d[field] = _iso_z(getattr(self, field))
        return d


def _iso_z(dt: datetime) -> str:
    """Match the observed wire format exactly: microseconds + literal 'Z' (not +00:00)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


class DLQEnvelope(BaseModel):
    """Everything routed to dlq-correlation-ml gets wrapped like this."""

    original_payload: dict
    error_type: str
    error_message: str
    stage: str  # "deserialize" | "validate" | "inference" | "publish"
    consumer_group: str
    failed_at: datetime

    def to_wire_dict(self) -> dict:
        d = self.model_dump(mode="json")
        d["failed_at"] = _iso_z(self.failed_at)
        return d
