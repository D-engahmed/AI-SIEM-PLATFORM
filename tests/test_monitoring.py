"""
Tests for src/monitoring.py's PSI drift detection -- had zero coverage
when first added (2026-08-18).
"""
from __future__ import annotations

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from monitoring import MIN_SAMPLES_FOR_PSI, FeatureDriftMonitor, compute_psi


class TestComputePsi:
    def test_identical_distributions_score_near_zero(self):
        rng = np.random.RandomState(1)
        reference = rng.normal(size=2000)
        current = rng.normal(size=2000)  # same distribution, different draw
        psi = compute_psi(reference, current)
        assert psi < 0.05  # sampling noise only, well under the 0.1 warn band

    def test_shifted_distribution_scores_high(self):
        rng = np.random.RandomState(1)
        reference = rng.normal(loc=0, scale=1, size=2000)
        current = rng.normal(loc=3, scale=1, size=2000)  # fully shifted
        psi = compute_psi(reference, current)
        assert psi > 0.25  # well past the alert band

    def test_moderate_shift_lands_between_bands(self):
        rng = np.random.RandomState(1)
        reference = rng.normal(loc=0, scale=1, size=3000)
        current = rng.normal(loc=0.5, scale=1, size=3000)
        psi = compute_psi(reference, current)
        assert 0.05 < psi < 1.0  # some real signal, not a full regime change

    def test_empty_arrays_return_zero(self):
        assert compute_psi(np.array([]), np.array([1.0, 2.0])) == 0.0
        assert compute_psi(np.array([1.0, 2.0]), np.array([])) == 0.0

    def test_degenerate_constant_reference_does_not_crash(self):
        # A near-constant feature (e.g. an always-0 flag) can't be
        # meaningfully quantile-binned -- must return 0.0, not raise.
        reference = np.zeros(500)
        current = np.zeros(500)
        assert compute_psi(reference, current) == 0.0

    def test_bin_edges_come_from_reference_not_current(self):
        # Same two distributions, swapped -- PSI is NOT symmetric in
        # general because bin edges are fixed by whichever array is
        # passed as `reference`. This pins that asymmetry is intentional,
        # not a bug, so a future refactor doesn't "fix" it.
        rng = np.random.RandomState(3)
        a = rng.exponential(scale=1.0, size=2000)
        b = rng.exponential(scale=2.0, size=2000)
        psi_ab = compute_psi(a, b)
        psi_ba = compute_psi(b, a)
        assert psi_ab != pytest.approx(psi_ba, rel=0.05)


class TestFeatureDriftMonitorLoad:
    def test_missing_reference_file_returns_none_not_exception(self, tmp_path):
        result = FeatureDriftMonitor.load(
            reference_path=str(tmp_path / "does_not_exist.npz"),
            ring_buffer_size=1000, psi_warn_threshold=0.1, psi_alert_threshold=0.25,
            registry=CollectorRegistry(),
        )
        assert result is None

    def test_loads_real_reference_artifact(self):
        # Uses the actual artifact training/train_model.py or the notebook
        # produces -- confirms the on-disk format and this loader agree.
        from pathlib import Path
        path = Path(__file__).parent.parent / "artifacts" / "feature_reference_distribution.npz"
        if not path.exists():
            pytest.skip("feature_reference_distribution.npz not built in this environment")
        monitor = FeatureDriftMonitor.load(
            reference_path=str(path), ring_buffer_size=1000,
            psi_warn_threshold=0.1, psi_alert_threshold=0.25, registry=CollectorRegistry(),
        )
        assert monitor is not None


class TestFeatureDriftMonitorBehavior:
    @pytest.fixture
    def monitor(self):
        rng = np.random.RandomState(13)
        reference = {"feat_a": rng.normal(size=1000), "feat_b": rng.uniform(size=1000)}
        return FeatureDriftMonitor(
            reference=reference, ring_buffer_size=200,
            psi_warn_threshold=0.1, psi_alert_threshold=0.25, registry=CollectorRegistry(),
        )

    def test_insufficient_samples_reports_status_not_ok(self, monitor):
        for _ in range(MIN_SAMPLES_FOR_PSI - 1):
            monitor.record({"feat_a": 0.0, "feat_b": 0.5})
        result = monitor.compute_all()
        assert result["feat_a"]["status"] == "INSUFFICIENT_DATA"
        assert result["feat_a"]["psi"] == 0.0

    def test_matching_distribution_reports_ok(self, monitor):
        rng = np.random.RandomState(99)
        for _ in range(500):
            monitor.record({"feat_a": float(rng.normal()), "feat_b": float(rng.uniform())})
        result = monitor.compute_all()
        assert result["feat_a"]["status"] == "OK"
        assert result["feat_a"]["n_samples"] == 200  # capped at ring_buffer_size

    def test_shifted_distribution_reports_alert(self, monitor):
        for _ in range(500):
            monitor.record({"feat_a": 10.0, "feat_b": 0.5})  # wildly off reference
        result = monitor.compute_all()
        assert result["feat_a"]["status"] == "ALERT"

    def test_ring_buffer_is_bounded(self, monitor):
        for i in range(10_000):
            monitor.record({"feat_a": float(i), "feat_b": 0.5})
        assert len(monitor._buffers["feat_a"]) == 200  # never exceeds ring_buffer_size

    def test_unknown_feature_name_is_ignored_not_an_error(self, monitor):
        monitor.record({"totally_unrecognized_feature": 1.0})  # must not raise
        assert "totally_unrecognized_feature" not in monitor._buffers
