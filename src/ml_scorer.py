"""
Feature engineering + scoring for correlation-ml-service.

Design constraints this file is answerable to (from the spec):
  - Statelessness: no Redis/Postgres, everything in-process. Enforced here
    by construction -- FeatureEngineer.transform() and ModelScorer.score()
    are pure functions of a single MLScoringTask. There is no dict, cache,
    or counter anywhere in this module keyed by target_value / source_ip /
    correlation_id. Do not add one; that's what upstream GraphPivotStrategy
    is for, and the spec explicitly says fan_in_count already encodes the
    history you'd otherwise be tempted to cache.
  - Startup cost: model loads once (ModelScorer.load), never per-message.
  - Interpretable model only: XGBoost here, restricted with monotonic
    constraints on the features with a defensible causal direction, which
    is what "interpretable" buys you with a tree ensemble (You can say,
    truthfully: "risk never decreases if fan-in count goes up, all else
    equal" -- a logistic regression gives you that for free; a constrained
    GBM gives it to you deliberately).
  - Security: target_value / source_ip are treated as opaque untrusted
    strings. They are only ever used as dict values in the output payload,
    never in file paths, SQL, shell commands, or format strings that build
    such things.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb

from schemas import GraphFeatures, MLScoringTask, SignalContext

logger = logging.getLogger("correlation_ml.scorer")

# Feature order is load-bearing: the trained model expects columns in
# exactly this order. Both training and serving import this constant so
# they can never silently drift apart.
FEATURE_NAMES: list[str] = [
    "fan_in_count",
    "fan_in_count_missing",
    "epoch_age_seconds_log1p",
    "epoch_age_seconds_missing",
    "fan_in_rate_log1p",
    "low_and_slow_ratio",
    "ti_matched_positive",
    "ti_matched_known",
    "signal_id_count",
]

# +1  = risk must not decrease as this feature increases (all else equal)
# -1  = risk must not increase as this feature increases
#  0  = no constraint, let the tree decide
# Kept conservative: only constrain what the spec gives us a clear causal
# story for. Everything else is left to the data instead of my priors.
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "fan_in_count": 1,
    "fan_in_count_missing": 0,
    "epoch_age_seconds_log1p": 0,   # ambiguous alone; direction depends on fan_in, left to interactions
    "epoch_age_seconds_missing": 0,
    "fan_in_rate_log1p": 1,
    "low_and_slow_ratio": 1,
    "ti_matched_positive": 1,
    "ti_matched_known": 0,
    "signal_id_count": 0,           # semantics of multi-signal payloads unconfirmed, left unconstrained
}

MIN_EPOCH_AGE_SECONDS = 1.0  # floor to avoid div-by-zero on brand-new nodes


@dataclass
class FeatureResult:
    vector: np.ndarray
    imputed_fields: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return len(self.imputed_fields) > 0


class FeatureEngineer:
    """Pure transform: MLScoringTask -> feature vector. No I/O, no state."""

    @staticmethod
    def transform(task: MLScoringTask) -> FeatureResult:
        gf: GraphFeatures = task.graph_features
        sc: SignalContext = task.signal_context
        imputed: list[str] = []

        # --- fan_in_count ---
        if gf.fan_in_count is None:
            fan_in_count = 0
            fan_in_count_missing = 1
            imputed.append("graph_features.fan_in_count")
        else:
            fan_in_count = gf.fan_in_count
            fan_in_count_missing = 0

        # --- epoch_age_seconds ---
        if gf.epoch_age_seconds is None:
            # Conservative imputation: treat as "just born". A missing age
            # must not let a message sneak past low_and_slow detection by
            # *looking* fresh -- the missing flag carries that risk instead.
            epoch_age_seconds = MIN_EPOCH_AGE_SECONDS
            epoch_age_missing = 1
            imputed.append("graph_features.epoch_age_seconds")
        else:
            epoch_age_seconds = max(gf.epoch_age_seconds, MIN_EPOCH_AGE_SECONDS)
            epoch_age_missing = 0

        epoch_age_log1p = math.log1p(epoch_age_seconds)

        # --- rate / low-and-slow features ---
        fan_in_rate = fan_in_count / epoch_age_seconds
        fan_in_rate_log1p = math.log1p(fan_in_rate)
        # Deliberately NOT just fan_in_rate inverted: this rewards sustained
        # *volume* over a long window even when the instantaneous rate looks
        # tiny, which is the actual "low and slow" failure mode the spec
        # calls out -- a rate-only feature would score it identically to
        # a target with trivially few, recent pivots.
        low_and_slow_ratio = fan_in_count / epoch_age_log1p if epoch_age_log1p > 0 else float(fan_in_count)

        # --- threat intel ---
        if sc.ti_matched is None:
            ti_matched_positive = 0
            ti_matched_known = 0
            imputed.append("signal_context.ti_matched")
        else:
            ti_matched_positive = 1 if sc.ti_matched else 0
            ti_matched_known = 1

        signal_id_count = len(sc.signal_ids) if sc.signal_ids else 0
        if not sc.signal_ids:
            imputed.append("signal_context.signal_ids")

        vector = np.array(
            [
                fan_in_count,
                fan_in_count_missing,
                epoch_age_log1p,
                epoch_age_missing,
                fan_in_rate_log1p,
                low_and_slow_ratio,
                ti_matched_positive,
                ti_matched_known,
                signal_id_count,
            ],
            dtype=np.float64,
        ).reshape(1, -1)

        return FeatureResult(vector=vector, imputed_fields=imputed)


@dataclass
class ScoringResult:
    risk_score: float           # 0-1000, wire scale
    probability: float          # 0-1, raw model output
    status: str                 # ESCALATED_TO_INCIDENT | OPEN (dropped, not published)
    degraded_mode: bool
    model_version: str
    imputed_fields: list[str]
    feature_vector: dict[str, float]  # for audit logging only, never published


class ModelScorer:
    """
    Loads an XGBoost booster + metadata exactly once (see main.py) and
    scores one message at a time. No mutable state across calls other
    than the immutable model itself.
    """

    def __init__(self, booster: xgb.Booster, model_version: str, threshold: float,
                 degraded_field_threshold: int):
        self._booster = booster
        self._model_version = model_version
        self._threshold = threshold
        self._degraded_field_threshold = degraded_field_threshold

    @classmethod
    def load(cls, artifact_path: str, threshold: float, degraded_field_threshold: int) -> "ModelScorer":
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {path}. Run training/train_model.py first, "
                f"or point CML_MODEL_ARTIFACT_PATH at a real artifact."
            )
        bundle = joblib.load(path)
        booster: xgb.Booster = bundle["booster"]
        model_version: str = bundle["model_version"]
        saved_features: list[str] = bundle["feature_names"]
        if saved_features != FEATURE_NAMES:
            # This is exactly the kind of silent-corruption bug that fail-closed
            # is meant to prevent. Refuse to start rather than score with a
            # column mismatch nobody would notice until the numbers looked odd.
            raise ValueError(
                "Model artifact feature order does not match serving FEATURE_NAMES. "
                f"artifact={saved_features} serving={FEATURE_NAMES}"
            )
        logger.info("Loaded model artifact version=%s from %s", model_version, path)
        return cls(booster, model_version, threshold, degraded_field_threshold)

    def score(self, task: MLScoringTask) -> ScoringResult:
        feats = FeatureEngineer.transform(task)
        dmatrix = xgb.DMatrix(feats.vector, feature_names=FEATURE_NAMES)
        # predict() on a single row is cheap; batching is not offered because
        # the consumer processes one Kafka message at a time by design (see
        # ml_consumer.py) and batching would reintroduce the cross-message
        # coupling statelessness explicitly forbids.
        probability = float(self._booster.predict(dmatrix)[0])
        probability = min(max(probability, 0.0), 1.0)  # clip, defense in depth

        degraded = len(feats.imputed_fields) >= self._degraded_field_threshold
        risk_score = round(probability * 1000.0, 2)
        status = "ESCALATED_TO_INCIDENT" if probability >= self._threshold else "OPEN"

        return ScoringResult(
            risk_score=risk_score,
            probability=probability,
            status=status,
            degraded_mode=degraded,
            model_version=self._model_version,
            imputed_fields=feats.imputed_fields,
            feature_vector=dict(zip(FEATURE_NAMES, feats.vector[0].tolist())),
        )
