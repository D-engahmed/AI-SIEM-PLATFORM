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

OPEN ML QUESTION (2026-08-17, deliberately NOT implemented here):
The first audit email also asked to (a) remove the labeling noise in
training/generate_synthetic_data.py's ambiguous_mid/benign_shared_nat
scenarios and (b) make benign_shared_nat/low_and_slow mathematically
separable on fan_in_count, and (c) add scale_pos_weight in training. None
of that was addressed in the reviewer's later, more careful reply, and
each has a real argument against it (benign_shared_nat's noise level was
mischaracterized in that first email -- it's 4%, not 40%, and it's the
one scenario stress-testing this service's worst false-positive mode;
the low_and_slow/benign_shared_nat "overlap" is what low_and_slow_ratio
and fan_in_rate_log1p exist to resolve, not a data bug). Not implemented
pending explicit confirmation -- only the epoch_age_seconds imputation
fix below was made, since it stands on its own regardless of that dispute.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
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
        # FIX (2026-08-17, flagged not confirmed -- see module docstring
        # "OPEN ML QUESTION"): a missing age used to impute straight to
        # MIN_EPOCH_AGE_SECONDS and then feed that into fan_in_rate below,
        # which artificially inflated the rate (dividing by ~1 second) and,
        # combined with fan_in_rate_log1p's +1 monotonic constraint, meant
        # the model could never down-score a message just because its age
        # was missing. epoch_age_seconds still gets the floor value (so
        # log1p() below doesn't blow up) but fan_in_rate is no longer
        # computed from it -- epoch_age_seconds_missing (unconstrained)
        # carries the "age unknown" risk signal instead, same as it always
        # did for the other imputed fields.
        if gf.epoch_age_seconds is None:
            epoch_age_seconds = MIN_EPOCH_AGE_SECONDS
            epoch_age_missing = 1
            rate_is_known = False
            imputed.append("graph_features.epoch_age_seconds")
        else:
            epoch_age_seconds = max(gf.epoch_age_seconds, MIN_EPOCH_AGE_SECONDS)
            epoch_age_missing = 0
            rate_is_known = True

        epoch_age_log1p = math.log1p(epoch_age_seconds)

        # --- rate / low-and-slow features ---
        if rate_is_known:
            fan_in_rate = fan_in_count / epoch_age_seconds
            fan_in_rate_log1p = math.log1p(fan_in_rate)
        else:
            fan_in_rate_log1p = 0.0  # neutral -- do not let a fabricated age drive this feature
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
    risk_score: int              # 0-100 int, per AD-061 (see schemas.py docstring
                                  # for the open question about whether this should
                                  # instead stay on the raw/uncapped scale)
    probability: float          # 0-1, raw model output
    status: str                 # ESCALATED_TO_INCIDENT | OPEN (dropped, not published)
    degraded_mode: bool
    model_version: str
    imputed_fields: list[str]
    feature_vector: dict[str, float]  # for audit logging only, never published


SUPPORTED_MODEL_FAMILIES = ("xgboost", "lightgbm")


class ModelScorer:
    """
    Loads a booster + metadata exactly once (see main.py) and scores one
    message at a time. No mutable state across calls other than the
    immutable model itself.

    BUG FIXED (2026-08-18): this class used to build an xgb.DMatrix and
    call .predict() on it UNCONDITIONALLY -- correct only for XGBoost.
    That was invisible as long as model_selection.ipynb always picked
    xgboost, but section 5's hyperparameter tuning (added the same day)
    means the winner is now a genuine, data-dependent outcome -- it could
    just as easily be lightgbm on a future retrain. A lightgbm artifact
    would have loaded fine (bundle format doesn't care) and then thrown
    at the first real message, in production, the first time tuning
    picked a different winner. Fixed by storing model_family in the
    artifact bundle and dispatching .score() on it instead of assuming.
    """

    def __init__(self, booster, model_family: str, model_version: str, threshold: float,
                 degraded_field_threshold: int):
        if model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(f"model_family={model_family!r} not in {SUPPORTED_MODEL_FAMILIES}")
        self._booster = booster
        self._model_family = model_family
        self._model_version = model_version
        self._threshold = threshold
        self._degraded_field_threshold = degraded_field_threshold

    @classmethod
    def load(cls, artifact_path: str, threshold: float, degraded_field_threshold: int) -> "ModelScorer":
        """Fast path: local joblib artifact, no MLflow dependency at serving time."""
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {path}. Run training/train_model.py first, "
                f"or point CML_MODEL_ARTIFACT_PATH at a real artifact."
            )
        bundle = joblib.load(path)
        return cls._from_bundle(bundle, threshold, degraded_field_threshold, source=str(path))

    @classmethod
    def load_from_mlflow(
        cls,
        tracking_uri: str,
        model_uri: str,
        threshold: float,
        degraded_field_threshold: int,
    ) -> "ModelScorer":
        """
        Registry path (CML_MODEL_SOURCE=mlflow): resolves model_uri (e.g.
        "models:/correlation-ml-service-risk-model/Staging") via the MLflow
        Model Registry. Lets Staging->Production promotion control what's
        served without a redeploy -- see config.py's model_source docstring.

        Requires the model to have been logged with a "model_family" tag
        (set by _build_notebook.py's registration cell) so this can tell
        which flavor-specific loader to use -- MLflow's generic
        mlflow.pyfunc.load_model() would hide that distinction and reproduce
        the exact bug this class was just fixed for.
        """
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()

        # model_uri is a stage alias ("models:/name/Staging"), not a fixed
        # version -- resolve it to a concrete version first so we can read
        # that version's tags (get_model_version needs a version number,
        # not a stage string).
        model_name, stage = model_uri.removeprefix("models:/").split("/", 1)
        versions = client.search_model_versions(f"name='{model_name}'")
        matching = [v for v in versions if v.current_stage == stage]
        if not matching:
            raise ValueError(f"No model version of '{model_name}' is in stage '{stage}'")
        version = max(matching, key=lambda v: int(v.version))
        model_family = version.tags.get("model_family")
        if model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(
                f"MLflow model version {model_name}/{version.version} has no valid "
                f"'model_family' tag (got {model_family!r}) -- refusing to guess which "
                f"flavor-specific loader to use."
            )

        resolved_uri = f"models:/{model_name}/{version.version}"
        if model_family == "xgboost":
            import mlflow.xgboost
            booster = mlflow.xgboost.load_model(resolved_uri)
        else:
            import mlflow.lightgbm
            booster = mlflow.lightgbm.load_model(resolved_uri)

        model_version = version.tags.get("model_version", f"{model_name}-v{version.version}")
        bundle = {"booster": booster, "model_family": model_family,
                  "model_version": model_version, "feature_names": FEATURE_NAMES}
        return cls._from_bundle(bundle, threshold, degraded_field_threshold, source=resolved_uri)

    @classmethod
    def _from_bundle(cls, bundle: dict, threshold: float, degraded_field_threshold: int,
                      *, source: str) -> "ModelScorer":
        booster = bundle["booster"]
        # Back-compat: artifacts saved before 2026-08-18 don't have
        # model_family -- every one of those was xgboost (it was the only
        # option before tuning could pick a different winner).
        model_family: str = bundle.get("model_family", "xgboost")
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
        logger.info(
            "Loaded model artifact version=%s family=%s from %s",
            model_version, model_family, source,
        )
        return cls(booster, model_family, model_version, threshold, degraded_field_threshold)

    def _predict_proba(self, vector: np.ndarray) -> float:
        if self._model_family == "xgboost":
            dmatrix = xgb.DMatrix(vector, feature_names=FEATURE_NAMES)
            return float(self._booster.predict(dmatrix)[0])
        # lightgbm.Booster.predict takes a plain ndarray directly, no
        # DMatrix-equivalent wrapper needed.
        return float(self._booster.predict(vector)[0])

    def score(self, task: MLScoringTask) -> ScoringResult:
        feats = FeatureEngineer.transform(task)
        # predict() on a single row is cheap; batching is not offered because
        # the consumer processes one Kafka message at a time by design (see
        # ml_consumer.py) and batching would reintroduce the cross-message
        # coupling statelessness explicitly forbids.
        probability = self._predict_proba(feats.vector)
        probability = min(max(probability, 0.0), 1.0)  # clip, defense in depth

        degraded = len(feats.imputed_fields) >= self._degraded_field_threshold
        risk_score = int(round(probability * 100.0))
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
