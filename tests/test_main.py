"""
Tests for main.py's entrypoint fail-fast logic -- this had zero coverage
before 2026-08-17. Only covers what's cheap and valuable to test without a
real Kafka broker: that a bad model artifact stops startup before anything
Kafka-related is even constructed, and that a good one proceeds.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import main


class TestFailFast:
    def test_missing_model_artifact_returns_1_and_never_builds_consumer(self):
        with patch("main.get_settings", return_value=_settings()), \
             patch("main.ModelScorer") as mock_scorer_cls, \
             patch("main.CorrelationMLConsumer") as mock_consumer_cls, \
             patch("main.asyncio.run") as mock_run:
            mock_scorer_cls.load.side_effect = FileNotFoundError("no artifact at that path")

            rc = main.main()

            assert rc == 1
            mock_consumer_cls.assert_not_called()
            mock_run.assert_not_called()

    def test_invalid_model_artifact_returns_1_and_never_builds_consumer(self):
        with patch("main.get_settings", return_value=_settings()), \
             patch("main.ModelScorer") as mock_scorer_cls, \
             patch("main.CorrelationMLConsumer") as mock_consumer_cls, \
             patch("main.asyncio.run") as mock_run:
            mock_scorer_cls.load.side_effect = ValueError("feature schema mismatch")

            rc = main.main()

            assert rc == 1
            mock_consumer_cls.assert_not_called()
            mock_run.assert_not_called()

    def test_successful_load_proceeds_to_run_and_returns_0(self):
        # side_effect closes the real coroutine main() builds instead of
        # actually running it (no live Kafka broker in this test env) --
        # avoids a "coroutine was never awaited" ResourceWarning while still
        # never touching the event loop.
        def _fake_run(coro):
            coro.close()

        with patch("main.get_settings", return_value=_settings()), \
             patch("main.ModelScorer") as mock_scorer_cls, \
             patch("main.CorrelationMLConsumer") as mock_consumer_cls, \
             patch("main.asyncio.run", side_effect=_fake_run) as mock_run:
            mock_scorer_cls.load.return_value = MagicMock()

            rc = main.main()

            assert rc == 0
            mock_consumer_cls.assert_called_once()
            mock_run.assert_called_once()

    def test_unhandled_exception_in_run_returns_1(self):
        def _fake_run_then_raise(coro):
            coro.close()
            raise RuntimeError("kafka unreachable")

        with patch("main.get_settings", return_value=_settings()), \
             patch("main.ModelScorer") as mock_scorer_cls, \
             patch("main.CorrelationMLConsumer"), \
             patch("main.asyncio.run", side_effect=_fake_run_then_raise):
            mock_scorer_cls.load.return_value = MagicMock()

            rc = main.main()

            assert rc == 1


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.log_level = "INFO"
    settings.consumer_group_id = "correlation-ml-service"
    settings.model_artifact_path = "/app/artifacts/model_latest.joblib"
    settings.escalation_threshold = 0.5
    settings.degraded_mode_imputed_field_threshold = 2
    settings.api_host = "0.0.0.0"
    settings.api_port = 9100
    return settings
