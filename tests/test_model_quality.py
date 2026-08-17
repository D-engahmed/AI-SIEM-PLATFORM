"""
End-to-end model quality gate. Scores a sample of
training/synthetic_ml_scoring_tasks.csv through the REAL ModelScorer
(same code path as ml_consumer.py / any future API), not the raw
booster, and checks metric floors + per-scenario sensitivity/specificity
sanity.

This is a smoke-level regression gate, not a substitute for
model_selection.ipynb's held-out evaluation: it scores the whole
dataset (including rows the artifact may have trained on), so treat a
failure here as "something broke badly enough to catch even on
training-adjacent data", and treat a pass as necessary, not sufficient.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from schemas import GraphFeatures, MLScoringTask, SignalContext

SAMPLE_SIZE = 3000
RANDOM_SEED = 7


def _row_to_task(row) -> MLScoringTask:
    import pandas as pd

    fan_in = None if pd.isna(row["fan_in_count"]) else int(row["fan_in_count"])
    age = None if pd.isna(row["epoch_age_seconds"]) else float(row["epoch_age_seconds"])
    ti = row["ti_matched"]
    ti_val = None if pd.isna(ti) else bool(ti)
    return MLScoringTask(
        correlation_id=row["correlation_id"],
        target_kind=row["target_kind"],
        target_value=row["target_value"],
        graph_features=GraphFeatures(fan_in_count=fan_in, epoch_age_seconds=age),
        signal_context=SignalContext(
            triggering_source_ip=row["triggering_source_ip"],
            ti_matched=ti_val,
            signal_ids=[f"sig-{i}" for i in range(int(row["signal_id_count"]))],
        ),
    )


@pytest.fixture(scope="module")
def scored_sample(synthetic_df, scorer):
    sample = synthetic_df.sample(n=min(SAMPLE_SIZE, len(synthetic_df)), random_state=RANDOM_SEED)
    probs, labels, scenarios = [], [], []
    for _, row in sample.iterrows():
        task = _row_to_task(row)
        result = scorer.score(task)
        probs.append(result.probability)
        labels.append(row["label"])
        scenarios.append(row["scenario"])
    sample = sample.copy()
    sample["_probability"] = probs
    return sample


class TestMetricFloors:
    def test_roc_auc_above_floor(self, scored_sample):
        auc = roc_auc_score(scored_sample["label"], scored_sample["_probability"])
        assert auc > 0.75, f"ROC-AUC {auc:.4f} below regression floor -- investigate before shipping"

    def test_average_precision_above_floor(self, scored_sample):
        ap = average_precision_score(scored_sample["label"], scored_sample["_probability"])
        assert ap > 0.70, f"Average precision {ap:.4f} below regression floor"


class TestPerScenarioSensitivitySpecificity:
    """
    Uses the service's actual default escalation_threshold (0.7, from
    config.py) rather than an arbitrary 0.5, since that's what determines
    real ESCALATED_TO_INCIDENT behavior.
    """

    THRESHOLD = 0.7

    @pytest.mark.parametrize("scenario,max_escalation_rate", [
        ("benign_quiet", 0.10),
        ("benign_slow_natural", 0.10),
        ("benign_shared_nat", 0.20),  # the hardest benign case on purpose -- looser bound
    ])
    def test_benign_scenarios_rarely_escalate(self, scored_sample, scenario, max_escalation_rate):
        subset = scored_sample[scored_sample["scenario"] == scenario]
        if len(subset) < 10:
            pytest.skip(f"too few sampled rows for {scenario} to be meaningful")
        rate = (subset["_probability"] >= self.THRESHOLD).mean()
        assert rate <= max_escalation_rate, (
            f"{scenario} escalation rate {rate:.1%} exceeds {max_escalation_rate:.0%} -- "
            f"false-positive regression"
        )

    @pytest.mark.parametrize("scenario,min_escalation_rate", [
        ("loud_burst", 0.70),
        ("low_and_slow", 0.60),
        ("ti_confirmed", 0.50),
    ])
    def test_attack_scenarios_mostly_escalate(self, scored_sample, scenario, min_escalation_rate):
        subset = scored_sample[scored_sample["scenario"] == scenario]
        if len(subset) < 10:
            pytest.skip(f"too few sampled rows for {scenario} to be meaningful")
        rate = (subset["_probability"] >= self.THRESHOLD).mean()
        assert rate >= min_escalation_rate, (
            f"{scenario} escalation rate {rate:.1%} below {min_escalation_rate:.0%} -- "
            f"sensitivity regression (attacks getting missed)"
        )
