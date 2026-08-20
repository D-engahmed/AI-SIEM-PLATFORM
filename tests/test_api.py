"""
Tests for the /healthz and /metrics FastAPI endpoints in api.py -- this
had zero coverage before 2026-08-17. Direct TestClient hits, no live
server/port needed.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api import app
from metrics import DEGRADED_MESSAGES_TOTAL
from ml_consumer import MLScoringHandler


class TestHealthz:
    def test_returns_ok(self):
        with TestClient(app) as client:
            resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestScoreEndpoint:
    def _client(self, handler) -> TestClient:
        client = TestClient(app)
        app.state.handler = handler
        return client

    def _payload(self, **overrides) -> dict:
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
        return base

    def test_no_handler_loaded_returns_503(self):
        client = TestClient(app)
        app.state.handler = None
        resp = client.post("/score", json=self._payload())
        assert resp.status_code == 503

    def test_strong_signal_escalates(self, handler):
        client = self._client(handler)
        resp = client.post("/score", json=self._payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["decision"] == "escalated"
        assert body["incident"]["status"] == "ESCALATED_TO_INCIDENT"
        assert body["incident"]["created_at"].endswith("Z")  # wire-format timestamp, not FastAPI's default

    def test_weak_signal_drops(self, handler):
        client = self._client(handler)
        resp = client.post("/score", json=self._payload(
            graph_features={"fan_in_count": 1, "epoch_age_seconds": 300},
            signal_context={"triggering_source_ip": "203.0.113.99", "ti_matched": False,
                             "signal_ids": ["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"]},
        ))
        assert resp.status_code == 200
        assert resp.json()["decision"] == "dropped"

    def test_invalid_payload_is_rejected_not_500(self, handler):
        client = self._client(handler)
        raw = self._payload()
        del raw["correlation_id"]
        resp = client.post("/score", json=raw)
        assert resp.status_code == 200  # a bad SCORING payload, not a bad HTTP request
        body = resp.json()
        assert body["decision"] == "rejected"
        assert body["stage"] == "validate"

    def test_score_call_does_not_increment_degraded_counter(self, handler):
        # The whole point of the 2026-08-18 fix (see ml_consumer.py) --
        # an ad-hoc test call through this endpoint must not pollute the
        # counter that describes real topic traffic data quality.
        client = self._client(handler)
        before = DEGRADED_MESSAGES_TOTAL._value.get()
        client.post("/score", json=self._payload(graph_features={}))  # missing fields -> degraded
        after = DEGRADED_MESSAGES_TOTAL._value.get()
        assert after == before

    def test_score_does_not_require_kafka(self, handler, monkeypatch):
        # Side-effect-free means side-effect-free -- fail loudly if /score
        # ever starts reaching for a producer that doesn't exist here.
        import shared.kafka.base_producer as base_producer

        def _boom(*a, **kw):
            raise AssertionError("POST /score must never touch the Kafka producer")

        monkeypatch.setattr(base_producer.BaseProducer, "produce", _boom)
        client = self._client(handler)
        resp = client.post("/score", json=self._payload())
        assert resp.status_code == 200


class TestDriftEndpoint:
    def test_no_monitor_loaded_returns_503(self):
        client = TestClient(app)
        app.state.monitor = None
        resp = client.get("/monitoring/drift")
        assert resp.status_code == 503

    def test_returns_per_feature_psi_report(self):
        import numpy as np
        from prometheus_client import CollectorRegistry

        from monitoring import FeatureDriftMonitor

        reference = {"fan_in_count": np.random.RandomState(1).normal(size=500)}
        monitor = FeatureDriftMonitor(
            reference=reference, ring_buffer_size=100,
            psi_warn_threshold=0.1, psi_alert_threshold=0.25, registry=CollectorRegistry(),
        )
        client = TestClient(app)
        app.state.monitor = monitor
        resp = client.get("/monitoring/drift")
        assert resp.status_code == 200
        body = resp.json()
        assert "fan_in_count" in body["features"]
        assert body["features"]["fan_in_count"]["status"] == "INSUFFICIENT_DATA"
