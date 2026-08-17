"""
Shared fixtures. Import path setup happens here once instead of being
copy-pasted at the top of every test file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

ARTIFACT_PATH = ROOT / "artifacts" / "model_latest.joblib"
SYNTHETIC_DATA_PATH = ROOT / "training" / "synthetic_ml_scoring_tasks.csv"


@pytest.fixture(scope="session")
def artifact_path() -> Path:
    if not ARTIFACT_PATH.exists():
        pytest.skip(
            "no model artifact at artifacts/model_latest.joblib -- run "
            "training/generate_synthetic_data.py then the model_selection "
            "notebook (or training/train_model.py) first"
        )
    return ARTIFACT_PATH


@pytest.fixture(scope="session")
def scorer(artifact_path):
    from ml_scorer import ModelScorer

    return ModelScorer.load(str(artifact_path), threshold=0.7, degraded_field_threshold=2)


@pytest.fixture(scope="session")
def handler(scorer):
    from ml_consumer import MLScoringHandler

    return MLScoringHandler(scorer=scorer, consumer_group="correlation-ml-service")


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    if not SYNTHETIC_DATA_PATH.exists():
        pytest.skip("no synthetic dataset -- run training/generate_synthetic_data.py first")
    df = pd.read_csv(SYNTHETIC_DATA_PATH, parse_dates=["event_time"])
    df["drifted_period"] = df["drifted_period"].astype(bool)
    return df


def make_task(**overrides):
    """Build a valid MLScoringTask with sane defaults, for tests that need one directly."""
    from schemas import GraphFeatures, MLScoringTask, SignalContext

    base = dict(
        correlation_id="11111111-1111-1111-1111-111111111111",
        target_kind="user",
        target_value="probe_user",
        graph_features=GraphFeatures(fan_in_count=5, epoch_age_seconds=3600),
        signal_context=SignalContext(
            triggering_source_ip="203.0.113.1", ti_matched=None, signal_ids=["sig-1"],
        ),
    )
    base.update(overrides)
    return MLScoringTask(**base)
