"""
Entrypoint for correlation-ml-service.

Order matters here and is deliberate:
  1. load config (fail fast on bad env)
  2. load model ONCE (fail fast if artifact missing/corrupt/mismatched)
  3. only then start consuming -- we never want to accept traffic we
     can't score.
"""

from __future__ import annotations

import logging
import sys

from config import get_settings
from ml_consumer import MLScoringHandler, build_runner
from ml_scorer import ModelScorer


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )


def main() -> int:
    settings = get_settings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger("correlation_ml.main")

    logger.info("starting correlation-ml-service consumer_group=%s", settings.consumer_group_id)

    try:
        scorer = ModelScorer.load(
            artifact_path=settings.model_artifact_path,
            threshold=settings.escalation_threshold,
            degraded_field_threshold=settings.degraded_mode_imputed_field_threshold,
        )
    except (FileNotFoundError, ValueError) as e:
        # Fail fast and loud -- an orchestrator should crash-loop this and
        # page someone, not start a consumer that can't score anything.
        logger.critical("failed to load model artifact, refusing to start: %s", e)
        return 1

    handler = MLScoringHandler(scorer=scorer, consumer_group=settings.consumer_group_id)

    try:
        runner = build_runner(settings, handler)
    except (RuntimeError, NotImplementedError) as e:
        logger.critical("failed to build Kafka runner: %s", e)
        return 1

    try:
        runner.run()
    except Exception:
        logger.exception("consumer loop exited with an unhandled error")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
