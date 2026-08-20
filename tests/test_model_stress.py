"""
Model robustness / stress tests -- "does the model handle stress" (asked
for directly), interpreted as: extreme, malformed, and adversarial-ish
inputs must never crash the scoring path, and the model's OUTPUT must
stay within its contract even on inputs nowhere near the training
distribution. This is about resilience, not accuracy -- these tests don't
assert the model is *right* about weird inputs, only that it doesn't fall
over, produce NaN/inf, or violate its own output bounds.

For THROUGHPUT/latency stress (a different kind of "stress"), see
stress/run_load_test.py -- that's a standalone script, not a pytest
module, since it's meant to run for a while against a live process, not
as part of the normal fast test suite.
"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from ml_scorer import FeatureEngineer, ModelScorer
from schemas import GraphFeatures, MLScoringTask, SignalContext


def make_task(**overrides) -> MLScoringTask:
    base = dict(
        correlation_id="00000000-0000-0000-0000-000000000099",
        target_kind="user",
        target_value="stress_probe",
        graph_features=GraphFeatures(),
        signal_context=SignalContext(triggering_source_ip="203.0.113.1", signal_ids=["s1"]),
    )
    base.update(overrides)
    return MLScoringTask(**base)


class TestExtremeNumericInputs:
    """fan_in_count and epoch_age_seconds are upstream-controlled --
    GraphPivotStrategy is a different service this one doesn't validate
    against. If it ever sends something absurd, this service must degrade
    gracefully, not crash."""

    @pytest.mark.parametrize("fan_in", [0, 1, 10_000, 1_000_000])
    def test_extreme_fan_in_count_does_not_crash(self, scorer, fan_in):
        task = make_task(graph_features=GraphFeatures(fan_in_count=fan_in, epoch_age_seconds=3600))
        result = scorer.score(task)
        assert 0 <= result.risk_score <= 100
        assert 0.0 <= result.probability <= 1.0
        assert not math.isnan(result.probability)

    @pytest.mark.parametrize("age", [0.0, 0.001, 1e9, 1e15])
    def test_extreme_epoch_age_does_not_crash(self, scorer, age):
        task = make_task(graph_features=GraphFeatures(fan_in_count=5, epoch_age_seconds=age))
        result = scorer.score(task)
        assert 0 <= result.risk_score <= 100
        assert not math.isnan(result.probability)

    def test_negative_fan_in_count_rejected_at_schema(self):
        # CORRECTED (2026-08-18): this test originally assumed pydantic
        # didn't constrain fan_in_count to >= 0 and asserted the pass-
        # through value instead -- wrong; there's a _non_negative_fan_in
        # validator already. Checked the actual source before trusting my
        # own assumption, since that's exactly the kind of thing worth
        # verifying rather than guessing.
        with pytest.raises(ValidationError, match="cannot be negative"):
            GraphFeatures(fan_in_count=-5, epoch_age_seconds=3600)

    def test_negative_epoch_age_rejected_at_schema(self):
        # Same correction as above -- also validated, not passed through.
        with pytest.raises(ValidationError, match="cannot be negative"):
            GraphFeatures(fan_in_count=5, epoch_age_seconds=-100.0)

    def test_huge_signal_id_count_does_not_crash(self, scorer):
        task = make_task(signal_context=SignalContext(
            triggering_source_ip="203.0.113.1", signal_ids=[f"s{i}" for i in range(50_000)],
        ))
        result = scorer.score(task)
        assert 0 <= result.risk_score <= 100


class TestAllFieldsMissing:
    def test_completely_empty_graph_features_and_signal_context(self, scorer):
        task = make_task(graph_features=GraphFeatures(), signal_context=SignalContext())
        result = scorer.score(task)
        assert 0 <= result.risk_score <= 100
        assert result.degraded_mode is True  # every optional field missing
        assert len(result.imputed_fields) >= 3

    def test_empty_target_value(self, scorer):
        task = make_task(target_value="")
        result = scorer.score(task)
        assert 0 <= result.risk_score <= 100

    def test_unicode_and_control_characters_in_target_value(self, scorer):
        # target_value is opaque/untrusted per ml_scorer.py's own security
        # note -- confirms that note holds under actually adversarial input,
        # not just the well-behaved strings every other test uses.
        for weird in ["用户\x00\x01", "'; DROP TABLE users; --", "a" * 10_000, "\u202e\u0000"]:
            task = make_task(target_value=weird)
            result = scorer.score(task)
            assert 0 <= result.risk_score <= 100


class TestRepeatedScoringIsStateless:
    def test_identical_input_scored_many_times_gives_identical_output(self, scorer):
        # Statelessness is a hard design constraint (ml_scorer.py) -- confirms
        # it under repetition, not just "works once".
        task = make_task(graph_features=GraphFeatures(fan_in_count=12, epoch_age_seconds=7200))
        scores = [scorer.score(task).probability for _ in range(200)]
        assert len(set(scores)) == 1

    def test_alternating_extreme_and_normal_inputs_do_not_leak_state(self, scorer):
        extreme = make_task(graph_features=GraphFeatures(fan_in_count=1_000_000, epoch_age_seconds=0.001))
        normal = make_task(graph_features=GraphFeatures(fan_in_count=5, epoch_age_seconds=3600))
        baseline = scorer.score(normal).probability
        for _ in range(50):
            scorer.score(extreme)
        after = scorer.score(normal).probability
        assert baseline == after  # normal input's score must be unaffected by extreme inputs in between


class TestBatchOfRandomizedInputs:
    """A property-based-flavored sweep, not a single hand-picked case --
    marked slow (see pytest.ini) since it scores several thousand
    variations. Purely a crash/bounds sweep, not an accuracy check."""

    @pytest.mark.slow
    def test_large_random_sweep_never_crashes_or_violates_bounds(self, scorer):
        import random

        rng = random.Random(13)
        for _ in range(5000):
            fan_in = rng.choice([None, 0, rng.randint(0, 100_000)])
            # Schema already rejects negative values (see the two
            # dedicated tests above) -- this sweep stays within the
            # valid, if extreme, range: 0 up to a very large age.
            age = rng.choice([None, 0.0, rng.uniform(0, 1e12)])
            ti = rng.choice([None, True, False])
            n_signals = rng.randint(0, 500)
            task = make_task(
                graph_features=GraphFeatures(fan_in_count=fan_in, epoch_age_seconds=age),
                signal_context=SignalContext(
                    triggering_source_ip=f"203.0.113.{rng.randint(0, 255)}",
                    ti_matched=ti,
                    signal_ids=[f"s{i}" for i in range(n_signals)],
                ),
            )
            result = scorer.score(task)
            assert 0 <= result.risk_score <= 100
            assert 0.0 <= result.probability <= 1.0
            assert not math.isnan(result.probability)
            assert not math.isinf(result.probability)
