"""
Training script for correlation-ml-service's risk classifier.

Deliberately a plain script, not a notebook: this is a service that has
to reproduce a byte-identical feature pipeline in production, and
notebooks are exactly where serving/training skew creeps in (cell run
out of order, a variable redefined halfway down, etc). It imports
FeatureEngineer from src/ml_scorer.py directly, so training and serving
are provably the same code, not just "supposed to match".

Usage:
    python training/generate_synthetic_data.py
    python training/train_model.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from ml_scorer import FEATURE_NAMES, MONOTONE_CONSTRAINTS, FeatureEngineer  # noqa: E402
from schemas import GraphFeatures, MLScoringTask, SignalContext  # noqa: E402

DATA_PATH = Path(__file__).parent / "synthetic_ml_scoring_tasks.csv"
ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_VERSION = "gml-svc-0.1.0-synthetic"
# Reference distribution for src/monitoring.py's PSI drift detection --
# capped per-feature sample size, not the full training set. PSI's binning
# is quantile-based, so beyond a few thousand samples more rows buy
# smoother quantile estimates, not a meaningfully different reference --
# not worth shipping a multi-MB artifact for.
REFERENCE_SAMPLE_SIZE = 5000


def load_rows() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run generate_synthetic_data.py first.")
    with DATA_PATH.open() as f:
        return list(csv.DictReader(f))


def row_to_task(row: dict) -> MLScoringTask:
    def _opt_int(v):
        return None if v in ("", "None", None) else int(float(v))

    def _opt_float(v):
        return None if v in ("", "None", None) else float(v)

    def _opt_bool(v):
        if v in ("", "None", None):
            return None
        return str(v).strip().lower() == "true"

    return MLScoringTask(
        correlation_id=row["correlation_id"],
        target_kind=row["target_kind"],
        target_value=row["target_value"],
        graph_features=GraphFeatures(
            fan_in_count=_opt_int(row["fan_in_count"]),
            epoch_age_seconds=_opt_float(row["epoch_age_seconds"]),
        ),
        signal_context=SignalContext(
            triggering_source_ip=row["triggering_source_ip"],
            ti_matched=_opt_bool(row["ti_matched"]),
            signal_ids=[f"sig-{i}" for i in range(int(row["signal_id_count"]))],
        ),
    )


def build_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    scenarios = []
    for i, row in enumerate(rows):
        task = row_to_task(row)
        feats = FeatureEngineer.transform(task)
        X[i, :] = feats.vector[0]
        y[i] = float(row["label"])
        scenarios.append(row["scenario"])
    return X, y, scenarios


def evaluate_low_and_slow_trap(booster: xgb.Booster) -> dict:
    """
    Targeted regression check, not a summary metric: build two payloads
    with the SAME instantaneous fan-in rate but very different absolute
    fan_in_count/epoch_age, and confirm the model doesn't treat them as
    equivalent just because rate matches. This is the exact failure mode
    a rate-only feature set would have -- if this check ever starts
    failing after a feature/model change, that's a real regression, not
    noise.
    """
    trap_high = MLScoringTask(
        correlation_id="00000000-0000-0000-0000-000000000001",
        target_kind="user",
        target_value="probe_user_1",
        graph_features=GraphFeatures(fan_in_count=18, epoch_age_seconds=18 * 3600),
        signal_context=SignalContext(triggering_source_ip="203.0.113.9", ti_matched=None, signal_ids=["s1"]),
    )
    trap_low = MLScoringTask(
        correlation_id="00000000-0000-0000-0000-000000000002",
        target_kind="user",
        target_value="probe_user_2",
        graph_features=GraphFeatures(fan_in_count=2, epoch_age_seconds=2 * 3600),
        signal_context=SignalContext(triggering_source_ip="203.0.113.10", ti_matched=None, signal_ids=["s1"]),
    )
    probs = {}
    for name, task in (("low_and_slow_attack_pattern", trap_high), ("benign_low_rate", trap_low)):
        vec = FeatureEngineer.transform(task).vector
        dm = xgb.DMatrix(vec, feature_names=FEATURE_NAMES)
        probs[name] = float(booster.predict(dm)[0])

    passed = probs["low_and_slow_attack_pattern"] > probs["benign_low_rate"]
    return {"probabilities": probs, "passed": passed}


def save_reference_distribution(X: np.ndarray, out_path: Path, seed: int = 13) -> None:
    """
    Saves a capped per-feature sample of the TRAINING feature matrix as the
    baseline src/monitoring.py compares live traffic against. Sampling from
    the full training set (not just X_train after the split, deliberately --
    the reference should describe "what the model was built to expect",
    which is the whole labeled dataset, not one split of it) keeps the
    artifact small and the reference honest about the data's actual shape.
    """
    rng = np.random.RandomState(seed)
    n = min(REFERENCE_SAMPLE_SIZE, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    sample = X[idx]
    np.savez(out_path, **{name: sample[:, i] for i, name in enumerate(FEATURE_NAMES)})


def main():
    rows = load_rows()
    X, y, scenarios = build_matrix(rows)

    X_train, X_test, y_train, y_test, scen_train, scen_test = train_test_split(
        X, y, scenarios, test_size=0.25, random_state=13, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_NAMES)

    monotone = "(" + ",".join(str(MONOTONE_CONSTRAINTS[f]) for f in FEATURE_NAMES) + ")"

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "monotone_constraints": monotone,
        "seed": 13,
    }

    evals_result = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=20,
        evals_result=evals_result,
        verbose_eval=False,
    )

    preds = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    auc = roc_auc_score(y_test, preds)
    ap = average_precision_score(y_test, preds)

    trap_result = evaluate_low_and_slow_trap(booster)

    importance = booster.get_score(importance_type="gain")

    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "roc_auc": round(float(auc), 4),
        "avg_precision": round(float(ap), 4),
        "best_iteration": int(booster.best_iteration),
        "low_and_slow_trap_check": trap_result,
        "feature_importance_gain": {k: round(v, 2) for k, v in importance.items()},
        "feature_names": FEATURE_NAMES,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    artifact_path = ARTIFACT_DIR / "model_latest.joblib"
    joblib.dump(
        {
            "booster": booster,
            "model_family": "xgboost",  # this script only ever trains xgboost -- see model_selection.ipynb for tuned cross-family comparison
            "model_version": MODEL_VERSION,
            "feature_names": FEATURE_NAMES,
        },
        artifact_path,
    )
    metrics_path = ARTIFACT_DIR / "model_latest.metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    reference_path = ARTIFACT_DIR / "feature_reference_distribution.npz"
    save_reference_distribution(X, reference_path)

    print(json.dumps(metrics, indent=2))
    if not trap_result["passed"]:
        print(
            "\n*** WARNING: low_and_slow_trap_check FAILED. The model scored a "
            "sustained low-and-slow pattern lower than a genuinely low-volume "
            "benign case. Do not ship this artifact as-is. ***",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nSaved model artifact -> {artifact_path}")
    print(f"Saved metrics         -> {metrics_path}")
    print(f"Saved drift reference  -> {reference_path}")


if __name__ == "__main__":
    main()
