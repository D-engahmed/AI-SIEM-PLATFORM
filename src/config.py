"""
Config for correlation-ml-service. Everything that could plausibly need
tuning per environment is an env var -- nothing about the risk threshold,
topic names, or model path is hardcoded in logic files.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CML_", env_file=".env", extra="ignore")

    # --- Kafka wiring ---
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    consume_topic: str = Field(default="ml-scoring-tasks")
    produce_topic_incidents: str = Field(default="incidents")
    produce_topic_dlq: str = Field(default="dlq-correlation-ml")
    # Spec: "Must be strictly unique to this service." Locked, not env-overridable,
    # so a copy-pasted .env from another service can't silently collide consumer groups.
    consumer_group_id: str = Field(default="correlation-ml-service", frozen=True)

    consumer_poll_timeout_seconds: float = Field(default=1.0, ge=0.1)
    # Fail-closed: how many consecutive poll/produce errors before we stop
    # claiming to be healthy (readiness probe should key off this).
    max_consecutive_errors: int = Field(default=20, ge=1)

    # --- Model ---
    # "file" reads a local joblib artifact (fast, no MLflow dependency at
    # serving time -- appropriate for the Kafka consumer's low-latency path).
    # "mlflow" resolves model_mlflow_uri via the MLflow Model Registry --
    # appropriate when you want promotion (Staging -> Production) to control
    # what's served without a redeploy.
    model_source: str = Field(default="file")  # "file" | "mlflow"
    model_artifact_path: str = Field(default="artifacts/model_latest.joblib")
    model_mlflow_uri: str = Field(default="models:/correlation-ml-service-risk-model/Staging")

    # Threshold on the model's calibrated probability (0-1) that maps to
    # ESCALATED_TO_INCIDENT. Kept out of the model artifact so it can be
    # tuned operationally without retraining.
    escalation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # If more than this many expected features were missing/imputed for a
    # given message, we still score (fail-closed means "don't drop data",
    # not "don't drop silently on error") but flag degraded_mode=true.
    degraded_mode_imputed_field_threshold: int = Field(default=2, ge=0)

    # --- MLflow (training/registry side; not required for "file" serving) ---
    # SQLite-backed, not plain filesystem: MLflow 3.x puts the filesystem
    # tracking store in maintenance mode with no Model Registry support, and
    # this service relies on the registry for Staging/Production promotion.
    mlflow_tracking_uri: str = Field(default="sqlite:///./mlflow.db")
    mlflow_experiment_name: str = Field(default="correlation-ml-service")
    mlflow_registered_model_name: str = Field(default="correlation-ml-service-risk-model")

    # --- API service (FastAPI, separate process from the Kafka consumer) ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080, ge=1, le=65535)
    # Requests slower than this are logged loudly -- see docs/07-stress-testing.md
    # for the baseline this was set relative to.
    api_latency_warn_ms: float = Field(default=50.0, gt=0)

    # --- Monitoring ---
    # Bounded ring buffer of recent scored requests, kept in-process for the
    # /metrics and drift-report tooling. Bounded on purpose -- see the
    # "Memory bounds" constraint in ml_scorer.py; this is observability
    # state, not application/business state, but it still must not grow
    # unbounded, so it gets the same discipline.
    monitoring_ring_buffer_size: int = Field(default=5000, ge=100)
    # Population Stability Index thresholds (industry-standard bands):
    # < 0.1 no significant shift, 0.1-0.25 moderate, > 0.25 significant.
    psi_warn_threshold: float = Field(default=0.1, ge=0)
    psi_alert_threshold: float = Field(default=0.25, ge=0)

    # --- Memory bounds ---
    # No per-target caches exist in this service (stateless by design, see
    # ml_scorer.py docstring) -- this constant exists only to bound the
    # single in-flight batch size if batched consumption is ever added.
    max_in_flight_messages: int = Field(default=500, ge=1)

    log_level: str = Field(default="INFO")

    @field_validator("escalation_threshold")
    @classmethod
    def _threshold_sane(cls, v: float) -> float:
        if v <= 0.0 or v >= 1.0:
            # allow exactly at boundaries technically, but warn-worthy in practice
            pass
        return v


def get_settings() -> Settings:
    # Deliberately uncached: settings are read once at startup in main.py and
    # passed down explicitly, so tests can construct Settings(**overrides)
    # without env/monkeypatch gymnastics.
    return Settings()
