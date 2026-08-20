"""
Model/feature-level tests. These test claims the training notebook and
ml_scorer.py's docstrings make explicitly -- monotonicity, imputation
behavior, degraded_mode thresholds -- against the ACTUAL artifact in
artifacts/model_latest.joblib, not just at training time. If someone
retrains with different data or a different model type, these should
catch a regression before it reaches ml_consumer.py or an API.
"""
from __future__ import annotations

import math

import joblib
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import make_task
from ml_scorer import FEATURE_NAMES, MIN_EPOCH_AGE_SECONDS, FeatureEngineer, ModelScorer
from schemas import GraphFeatures, MLScoringTask, SignalContext
from pydantic import ValidationError


# --------------------------------------------------------------------------
# FeatureEngineer: pure, no model needed
# --------------------------------------------------------------------------

class TestFeatureEngineerImputation:
    def test_fully_populated_row_imputes_nothing(self):
        task = make_task(
            graph_features=GraphFeatures(fan_in_count=10, epoch_age_seconds=3600),
            signal_context=SignalContext(triggering_source_ip="203.0.113.1", ti_matched=True, signal_ids=["s1"]),
        )
        result = FeatureEngineer.transform(task)
        assert result.imputed_fields == []
        assert result.degraded is False

    def test_missing_fan_in_count_is_flagged(self):
        task = make_task(graph_features=GraphFeatures(fan_in_count=None, epoch_age_seconds=3600))
        result = FeatureEngineer.transform(task)
        assert "graph_features.fan_in_count" in result.imputed_fields
        # imputed to 0, not silently dropped from the vector
        idx = FEATURE_NAMES.index("fan_in_count")
        assert result.vector[0, idx] == 0.0
        idx_flag = FEATURE_NAMES.index("fan_in_count_missing")
        assert result.vector[0, idx_flag] == 1.0

    def test_missing_epoch_age_floors_conservatively(self):
        # Missing age must NOT create a fake "high-rate" node. The model must
        # learn from the missing-age flag rather than a fabricated 1-second age.
        task = make_task(graph_features=GraphFeatures(fan_in_count=10, epoch_age_seconds=None))
        result = FeatureEngineer.transform(task)
        assert "graph_features.epoch_age_seconds" in result.imputed_fields
        idx = FEATURE_NAMES.index("fan_in_rate_log1p")
        assert result.vector[0, idx] < math.log1p(2.0)
        idx_age = FEATURE_NAMES.index("epoch_age_seconds_log1p")
        # PRE-EXISTING BUG FIXED (2026-08-17): this assertion previously read
        # `math.log1p(max(10.0, MIN_EPOCH_AGE_SECONDS))` -- the 10.0 there was
        # fan_in_count leaking into an epoch-age assertion by copy-paste, not
        # a real relationship (fan_in_count has nothing to do with how
        # epoch_age_seconds gets imputed). Confirmed via `git show HEAD` that
        # this line predates every change made in this session -- the actual
        # imputed age has always been exactly MIN_EPOCH_AGE_SECONDS, not
        # max(fan_in_count, MIN_EPOCH_AGE_SECONDS). Fixed to assert that.
        assert result.vector[0, idx_age] == pytest.approx(math.log1p(MIN_EPOCH_AGE_SECONDS))

    def test_missing_epoch_age_does_not_inflate_fan_in_rate(self):
        # The actual bug the first audit email caught (2026-08-17 fix):
        # imputing epoch_age to a 1-second floor used to feed straight into
        # fan_in_rate = fan_in_count / epoch_age_seconds, producing an
        # artificially huge rate for ANY fan_in_count when age was simply
        # unreported -- not because the node was actually fast-moving.
        # fan_in_rate_log1p carries a +1 monotonic constraint, so that
        # inflation could never be down-scored by the model. Fixed by making
        # fan_in_rate_log1p neutral (0.0) whenever age is imputed, so only
        # the (unconstrained) missing-flag carries that signal now.
        missing_age = make_task(graph_features=GraphFeatures(fan_in_count=50, epoch_age_seconds=None))
        known_age = make_task(graph_features=GraphFeatures(fan_in_count=50, epoch_age_seconds=3600.0))
        idx = FEATURE_NAMES.index("fan_in_rate_log1p")
        missing_result = FeatureEngineer.transform(missing_age)
        known_result = FeatureEngineer.transform(known_age)
        assert missing_result.vector[0, idx] == 0.0
        assert known_result.vector[0, idx] > 0.0

    def test_ti_matched_none_is_flagged_and_distinct_from_false(self):
        task_none = make_task(signal_context=SignalContext(triggering_source_ip="1.1.1.1", ti_matched=None, signal_ids=["s1"]))
        task_false = make_task(signal_context=SignalContext(triggering_source_ip="1.1.1.1", ti_matched=False, signal_ids=["s1"]))

        r_none = FeatureEngineer.transform(task_none)
        r_false = FeatureEngineer.transform(task_false)

        known_idx = FEATURE_NAMES.index("ti_matched_known")
        positive_idx = FEATURE_NAMES.index("ti_matched_positive")

        assert r_none.vector[0, known_idx] == 0.0   # "we don't know" ...
        assert r_false.vector[0, known_idx] == 1.0  # ... is not the same as "checked, no match"
        assert r_none.vector[0, positive_idx] == 0.0
        assert r_false.vector[0, positive_idx] == 0.0
        assert "signal_context.ti_matched" in r_none.imputed_fields
        assert "signal_context.ti_matched" not in r_false.imputed_fields

    def test_empty_signal_ids_is_flagged(self):
        task = make_task(signal_context=SignalContext(triggering_source_ip="1.1.1.1", ti_matched=True, signal_ids=[]))
        result = FeatureEngineer.transform(task)
        assert "signal_context.signal_ids" in result.imputed_fields

    def test_everything_missing_does_not_crash(self):
        task = make_task(
            graph_features=GraphFeatures(fan_in_count=None, epoch_age_seconds=None),
            signal_context=SignalContext(triggering_source_ip=None, ti_matched=None, signal_ids=[]),
        )
        result = FeatureEngineer.transform(task)
        assert result.degraded is True
        assert np.isfinite(result.vector).all()

    def test_zero_epoch_age_does_not_divide_by_zero(self):
        task = make_task(graph_features=GraphFeatures(fan_in_count=5, epoch_age_seconds=0.0))
        result = FeatureEngineer.transform(task)
        assert np.isfinite(result.vector).all()

    def test_large_fan_in_count_does_not_overflow(self):
        task = make_task(graph_features=GraphFeatures(fan_in_count=1_000_000, epoch_age_seconds=1.0))
        result = FeatureEngineer.transform(task)
        assert np.isfinite(result.vector).all()

    def test_feature_vector_matches_declared_names_length(self):
        task = make_task()
        result = FeatureEngineer.transform(task)
        assert result.vector.shape == (1, len(FEATURE_NAMES))


class TestSchemaRejectsInvalidInput:
    def test_negative_fan_in_count_rejected_at_construction(self):
        with pytest.raises(ValidationError):
            GraphFeatures(fan_in_count=-1, epoch_age_seconds=10)

    def test_negative_epoch_age_rejected_at_construction(self):
        with pytest.raises(ValidationError):
            GraphFeatures(fan_in_count=1, epoch_age_seconds=-10)


# --------------------------------------------------------------------------
# ModelScorer: needs the real trained artifact
# --------------------------------------------------------------------------

class TestModelScorerDegradedMode:
    def test_below_threshold_missing_fields_not_degraded(self, scorer):
        # exactly 1 missing field (fan_in_count) -- explicitly supply a known
        # ti_matched so the fixture's own default (None) doesn't add a second
        # imputed field and mask what this test is checking.
        task = make_task(
            graph_features=GraphFeatures(fan_in_count=None, epoch_age_seconds=3600),
            signal_context=SignalContext(triggering_source_ip="203.0.113.1", ti_matched=True, signal_ids=["sig-1"]),
        )
        result = scorer.score(task)
        assert len(result.imputed_fields) == 1
        assert result.degraded_mode is False

    def test_at_threshold_missing_fields_is_degraded(self, scorer):
        task = make_task(
            graph_features=GraphFeatures(fan_in_count=None, epoch_age_seconds=None),
        )
        result = scorer.score(task)
        assert len(result.imputed_fields) >= 2
        assert result.degraded_mode is True

    def test_risk_score_always_in_wire_range(self, scorer):
        task = make_task(graph_features=GraphFeatures(fan_in_count=999999, epoch_age_seconds=0.001))
        result = scorer.score(task)
        assert 0.0 <= result.risk_score <= 150.0
        assert 0.0 <= result.probability <= 1.0


class TestModelScorerMonotonicity:
    """
    Promotes the notebook's monotonicity story from a training-time
    inspection into a property test that runs against the shipped
    artifact. See ml_scorer.py's MONOTONE_CONSTRAINTS table for which
    features this is and isn't expected to hold for -- these tests only
    assert what the constraint set actually guarantees, not more.
    """

    @given(
        epoch_age=st.floats(min_value=1.0, max_value=86400.0, allow_nan=False),
        fan_in_lo=st.integers(min_value=0, max_value=50),
        fan_in_delta=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=60, deadline=None)
    def test_probability_non_decreasing_in_fan_in_count(self, scorer, epoch_age, fan_in_lo, fan_in_delta):
        # fan_in_count feeds three constrained-non-decreasing features at once
        # (itself, fan_in_rate_log1p, low_and_slow_ratio) with epoch_age held
        # fixed, so the composition is guaranteed non-decreasing.
        fan_in_hi = fan_in_lo + fan_in_delta
        task_lo = make_task(graph_features=GraphFeatures(fan_in_count=fan_in_lo, epoch_age_seconds=epoch_age))
        task_hi = make_task(graph_features=GraphFeatures(fan_in_count=fan_in_hi, epoch_age_seconds=epoch_age))

        p_lo = scorer.score(task_lo).probability
        p_hi = scorer.score(task_hi).probability

        assert p_hi >= p_lo - 1e-6, (
            f"probability decreased as fan_in_count went {fan_in_lo} -> {fan_in_hi} "
            f"at epoch_age={epoch_age}: {p_lo} -> {p_hi}"
        )

    @given(epoch_age=st.floats(min_value=1.0, max_value=86400.0, allow_nan=False),
           fan_in=st.integers(min_value=0, max_value=50))
    @settings(max_examples=40, deadline=None)
    def test_probability_non_decreasing_when_ti_matched_flips_true(self, scorer, epoch_age, fan_in):
        # Only ti_matched_positive changes (0 -> 1); ti_matched_known stays
        # 1 either way, since both False and True are "a check was done".
        task_false = make_task(
            graph_features=GraphFeatures(fan_in_count=fan_in, epoch_age_seconds=epoch_age),
            signal_context=SignalContext(triggering_source_ip="1.1.1.1", ti_matched=False, signal_ids=["s1"]),
        )
        task_true = make_task(
            graph_features=GraphFeatures(fan_in_count=fan_in, epoch_age_seconds=epoch_age),
            signal_context=SignalContext(triggering_source_ip="1.1.1.1", ti_matched=True, signal_ids=["s1"]),
        )
        p_false = scorer.score(task_false).probability
        p_true = scorer.score(task_true).probability
        assert p_true >= p_false - 1e-6

    def test_low_and_slow_trap_case(self, scorer):
        """The specific regression check from train_model.py / the notebook, kept
        here too so `pytest` alone (no notebook run) still catches a regression."""
        attack = make_task(graph_features=GraphFeatures(fan_in_count=18, epoch_age_seconds=18 * 3600))
        benign = make_task(graph_features=GraphFeatures(fan_in_count=2, epoch_age_seconds=2 * 3600))
        p_attack = scorer.score(attack).probability
        p_benign = scorer.score(benign).probability
        assert p_attack > p_benign, (
            f"low-and-slow trap FAILED: sustained pattern scored {p_attack:.4f}, "
            f"low-volume benign scored {p_benign:.4f}"
        )


class TestArtifactIntegrity:
    def test_reload_produces_bit_identical_predictions(self, artifact_path, scorer):
        # Round-trip through joblib again (independent of the original save)
        # and confirm nothing about deserialization silently perturbs scores.
        bundle = joblib.load(artifact_path)
        reloaded = ModelScorer(
            booster=bundle["booster"], model_family=bundle.get("model_family", "xgboost"),
            model_version=bundle["model_version"], threshold=0.7, degraded_field_threshold=2,
        )
        rng = np.random.RandomState(7)
        for _ in range(20):
            fan_in = int(rng.randint(0, 40))
            age = float(rng.uniform(1, 86400))
            ti = bool(rng.random() < 0.5) if rng.random() < 0.8 else None
            task = make_task(
                graph_features=GraphFeatures(fan_in_count=fan_in, epoch_age_seconds=age),
                signal_context=SignalContext(triggering_source_ip="1.1.1.1", ti_matched=ti, signal_ids=["s1"]),
            )
            p1 = scorer.score(task).probability
            p2 = reloaded.score(task).probability
            assert p1 == p2

    def test_load_rejects_feature_order_mismatch(self, tmp_path, artifact_path):
        bundle = joblib.load(artifact_path)
        corrupted = dict(bundle)
        corrupted["feature_names"] = list(reversed(bundle["feature_names"]))
        bad_path = tmp_path / "corrupted.joblib"
        joblib.dump(corrupted, bad_path)

        with pytest.raises(ValueError, match="feature order"):
            ModelScorer.load(str(bad_path), threshold=0.7, degraded_field_threshold=2)

    def test_load_missing_file_raises_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModelScorer.load(str(tmp_path / "does_not_exist.joblib"), threshold=0.7, degraded_field_threshold=2)
