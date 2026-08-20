"""
Wire contracts for correlation-ml-service.

ASSUMPTION FLAG -- INBOUND SCHEMA (read before trusting this):
--------------------------------------------------------------
The reviewer's first email (2026-08-17) claimed the ml-scoring-tasks
payload has source_ip / signal_id / window_start / window_end at the
ROOT level. That directly contradicts docs/ML_AI.md Section 2, which
this service was originally built against, and describes a nested
graph_features / signal_context shape instead. The reviewer's follow-up
reply -- the one that answered every other open question in detail --
never came back to this one. No real sample ml-scoring-tasks message has
been provided either time.

Given a confirmed written spec on one side and an unconfirmed, internally
disputed claim on the other, the INBOUND shape below is UNCHANGED from
the original spec-based reconstruction. Do not flatten this to root-level
fields without an actual sample message to check it against -- every
upstream field name is declared in ONE place (this file) specifically so
that fix is a one-file diff when that sample shows up.

OUTBOUND SCHEMA (2026-08-17 rewrite):
--------------------------------------------------------------
This DID change, on the strength of much stronger evidence: every one of
the 65 "rich schema" records sampled from docs/all_attack_incidents.json
nests strategy_name/linked_keys/degraded_mode/window_start/window_end
inside `metadata`, and every one has metadata.raw_risk_score exactly
10x its root-level risk_score. That's a live, consistent pattern, not
just a claim in an email -- see IncidentEvent below.

Two things this rewrite implements on a WEAKER footing, flagged so
they're easy to find and revisit:
  - risk_score is capped to a 0-100 int per the reviewer's explicit
    instruction. There's a real open question (raised, not yet
    answered) about whether correlation-ml-service should instead
    publish on the SAME raw/uncapped scale other strategies apparently
    use at the point they leave the strategy (RiskBasedAlertingStrategy
    reaches 300 in the sampled data) and let whatever already does the
    /10 conversion for everyone else do it here too. Implemented per
    the explicit instruction; not independently confirmed.
  - severity's LOW/MEDIUM/HIGH/CRITICAL score bands below are a
    placeholder quartile split -- no real bands were ever given. Confirm
    before shipping.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_LINKED_KEYS = 10
MAX_KEY_LEN = 50
MAX_VALUE_LEN = 255
MAX_SIGNAL_IDS = 500

STRATEGY_NAME = "GraphMLScoring"
ESCALATED_STATUS = "ESCALATED_TO_INCIDENT"

SEVERITY_VALUES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def severity_for_score(risk_score: int) -> str:
    """
    PLACEHOLDER -- no score->severity mapping was ever given (asked for,
    not answered). This is a plain quartile split of the 0-100 range.
    doc3 sec 3 says SOAR keys auto-containment off risk_score directly,
    so severity may only be advisory downstream -- but confirm that
    rather than assume it, since "advisory" and "unused" are very
    different amounts of risk if this guess is wrong.
    """
    if risk_score >= 75:
        return "CRITICAL"
    if risk_score >= 50:
        return "HIGH"
    if risk_score >= 25:
        return "MEDIUM"
    return "LOW"


# --------------------------------------------------------------------------
# Inbound: ml-scoring-tasks (UNCHANGED -- see module docstring)
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
# Outbound: incidents (AD-061 shape -- see module docstring)
# --------------------------------------------------------------------------

class IncidentEvent(BaseModel):
    """
    Root-field contract per AD-061: strategy-specific data lives in
    `metadata`, not at the root. `status` and `incident_id` were both
    observed at root in real sampled records; `incident_id` is not
    generated here on the assumption it's assigned by whatever persists
    this (unconfirmed -- flagged, not guessed at further).
    """

    title: str
    source_ip: str
    username: Optional[str] = Field(default=None, max_length=255)
    protocol: Optional[str] = Field(default=None, max_length=20)
    severity: str
    risk_score: int
    status: str = ESCALATED_STATUS
    correlation_id: Optional[UUID] = None
    signal_ids: list[UUID] = Field(default_factory=list, max_length=MAX_SIGNAL_IDS)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: Optional[datetime] = None

    @field_validator("risk_score")
    @classmethod
    def _score_range(cls, v: int) -> int:
        if not (0 <= v <= 100):
            raise ValueError(f"risk_score {v} out of [0, 100]")
        return v

    @field_validator("severity")
    @classmethod
    def _severity_enum(cls, v: str) -> str:
        if v not in SEVERITY_VALUES:
            raise ValueError(f"severity {v!r} not in {SEVERITY_VALUES}")
        return v

    @field_validator("metadata")
    @classmethod
    def _metadata_linked_keys_bounds(cls, v: dict[str, Any]) -> dict[str, Any]:
        linked_keys = v.get("linked_keys")
        if not isinstance(linked_keys, dict):
            return v
        if len(linked_keys) > MAX_LINKED_KEYS:
            raise ValueError(f"linked_keys has {len(linked_keys)} entries, max {MAX_LINKED_KEYS}")
        for k, val in linked_keys.items():
            if len(k) > MAX_KEY_LEN:
                raise ValueError(f"linked_keys key '{k[:20]}...' exceeds {MAX_KEY_LEN} chars")
            if len(str(val)) > MAX_VALUE_LEN:
                raise ValueError(f"linked_keys value for '{k}' exceeds {MAX_VALUE_LEN} chars")
        return v
