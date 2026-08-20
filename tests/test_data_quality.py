"""
Tests the DATA, not the model. Motivated directly by this project's
opening mistake: the first uploaded "sample data" turned out to be the
wrong shape entirely, and nothing caught that until it was inspected by
hand. These tests are the automated version of that inspection, run
against whatever is currently at
training/synthetic_ml_scoring_tasks.csv -- so if that file is ever
swapped for real historical data, this suite is most of what you'd want
to rerun first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

REQUIRED_COLUMNS = {
    "scenario", "drifted_period", "event_time", "correlation_id", "target_kind",
    "target_value", "fan_in_count", "epoch_age_seconds", "triggering_source_ip",
    "ti_matched", "signal_id_count", "label",
}

EXPECTED_SCENARIOS = {
    "benign_quiet", "benign_slow_natural", "benign_shared_nat",
    "loud_burst", "low_and_slow", "ti_confirmed", "ambiguous_mid",
}


class TestSchema:
    def test_required_columns_present(self, synthetic_df):
        assert REQUIRED_COLUMNS.issubset(set(synthetic_df.columns))

    def test_label_is_binary(self, synthetic_df):
        assert set(synthetic_df["label"].unique()) <= {0, 1}

    def test_target_kind_is_always_user(self, synthetic_df):
        # Matches the spec's stated constraint -- if this ever fails, the
        # generator (or a real-data swap) introduced a target_kind the
        # service isn't built to handle (see SUPPORTED_TARGET_KINDS in
        # ml_consumer.py).
        assert set(synthetic_df["target_kind"].unique()) == {"user"}

    def test_scenario_values_match_known_set(self, synthetic_df):
        assert set(synthetic_df["scenario"].unique()) == EXPECTED_SCENARIOS

    def test_correlation_ids_are_unique(self, synthetic_df):
        assert synthetic_df["correlation_id"].is_unique

    def test_non_negative_where_present(self, synthetic_df):
        assert (synthetic_df["fan_in_count"].dropna() >= 0).all()
        assert (synthetic_df["epoch_age_seconds"].dropna() >= 0).all()
        assert (synthetic_df["signal_id_count"] >= 0).all()


class TestMissingFieldRate:
    def test_missing_field_rate_near_documented_target(self, synthetic_df):
        # generate_synthetic_data.py's MISSING_FIELD_RATE = 0.12 -- assert
        # the realized rate is in the same neighborhood, not exactly equal
        # (it's stochastic).
        fan_in_missing_rate = synthetic_df["fan_in_count"].isna().mean()
        age_missing_rate = synthetic_df["epoch_age_seconds"].isna().mean()
        assert 0.06 < fan_in_missing_rate < 0.20
        assert 0.06 < age_missing_rate < 0.20


class TestClassBalance:
    def test_overall_positive_rate_is_reasonable(self, synthetic_df):
        # Not claiming this matches real-world base rates (it can't --
        # there's no real data yet, see docs/README). Only guards against
        # a generator bug that makes the problem trivial (e.g. 99%+ or <1%
        # positive), which would make every metric in the notebook meaningless.
        rate = synthetic_df["label"].mean()
        assert 0.2 < rate < 0.8

    def test_every_scenario_has_both_classes_represented(self, synthetic_df):
        # ambiguous_mid and the noise floors on benign_* scenarios exist
        # specifically so this holds -- a scenario with a perfectly pure
        # label lets a model shortcut on `scenario` instead of learning
        # the actual features (scenario itself isn't in FEATURE_NAMES, but
        # a pure-label scenario is still a sign the simulation is too easy).
        counts = synthetic_df.groupby("scenario")["label"].nunique()
        assert (counts == 2).all(), counts[counts != 2]


class TestInjectedDriftIsActuallyPresent:
    """
    generate_synthetic_data.py claims to inject a distribution shift in the
    last 20% of the time window. These tests check that claim against the
    data rather than trusting the docstring -- if someone changes the
    generator and this stops being true, the monitoring notebook section
    and monitoring/ tooling (when built) would be validated against a drift
    that no longer exists.
    """

    def test_drifted_period_is_roughly_a_fifth_of_rows(self, synthetic_df):
        frac = synthetic_df["drifted_period"].mean()
        assert 0.15 < frac < 0.25

    def test_drifted_rows_are_strictly_later_in_time(self, synthetic_df):
        drift_start = synthetic_df.loc[synthetic_df["drifted_period"], "event_time"].min()
        non_drift_end = synthetic_df.loc[~synthetic_df["drifted_period"], "event_time"].max()
        assert non_drift_end <= drift_start

    def test_loud_burst_fan_in_increases_in_drift_period(self, synthetic_df):
        loud = synthetic_df[synthetic_df["scenario"] == "loud_burst"]
        before = loud.loc[~loud["drifted_period"], "fan_in_count"].mean()
        after = loud.loc[loud["drifted_period"], "fan_in_count"].mean()
        assert after > before, (
            f"loud_burst fan_in_count did not increase in the drift window "
            f"(before={before:.2f}, after={after:.2f}) -- the injected drift "
            f"this dataset advertises isn't actually there."
        )

    def test_benign_shared_nat_fan_in_increases_in_drift_period(self, synthetic_df):
        nat = synthetic_df[synthetic_df["scenario"] == "benign_shared_nat"]
        before = nat.loc[~nat["drifted_period"], "fan_in_count"].mean()
        after = nat.loc[nat["drifted_period"], "fan_in_count"].mean()
        assert after > before


class TestScenarioShapesAreDistinguishable:
    """Sanity checks that the scenarios encode genuinely different situations,
    not the same distribution with different labels bolted on."""

    def test_benign_shared_nat_has_higher_fan_in_than_benign_quiet(self, synthetic_df):
        quiet = synthetic_df.loc[synthetic_df["scenario"] == "benign_quiet", "fan_in_count"].mean()
        nat = synthetic_df.loc[synthetic_df["scenario"] == "benign_shared_nat", "fan_in_count"].mean()
        assert nat > quiet

    def test_low_and_slow_has_longer_age_than_loud_burst(self, synthetic_df):
        low_slow_age = synthetic_df.loc[synthetic_df["scenario"] == "low_and_slow", "epoch_age_seconds"].mean()
        loud_age = synthetic_df.loc[synthetic_df["scenario"] == "loud_burst", "epoch_age_seconds"].mean()
        assert low_slow_age > loud_age

    def test_ti_confirmed_always_has_ti_matched_true(self, synthetic_df):
        ti_rows = synthetic_df[synthetic_df["scenario"] == "ti_confirmed"]
        assert (ti_rows["ti_matched"] == True).all()  # noqa: E712


class TestDriftDetectionRealism:
    """
    Pins the corrected claim in generate_synthetic_data.py's module
    docstring (2026-08-18): the injected drift is real (per the tests
    above) but is NOT detectable by pooled whole-dataset PSI -- only by
    PSI restricted to the two scenarios that actually shift. Runs
    monitoring/generate_drift_report.py's actual code path (imported, not
    reimplemented) so this stays true if that script ever changes.
    """

    def test_pooled_drift_report_does_not_alert(self, tmp_path):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "monitoring"))
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from generate_drift_report import _load_features_split  # noqa: E402
        from monitoring import compute_psi  # noqa: E402
        from ml_scorer import FEATURE_NAMES  # noqa: E402

        csv_path = Path(__file__).parent.parent / "training" / "synthetic_ml_scoring_tasks.csv"
        X_ref, X_cur = _load_features_split(csv_path, "drifted_period")
        psis = [compute_psi(X_ref[:, i], X_cur[:, i]) for i in range(len(FEATURE_NAMES))]
        assert max(psis) < 0.1, (
            "Pooled PSI now detects the drift -- either the data generator or the dilution "
            "story in generate_synthetic_data.py's docstring is stale and needs updating, "
            "not this test."
        )

    def test_scenario_restricted_drift_report_does_alert(self, tmp_path):
        import csv
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "monitoring"))
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from generate_drift_report import _load_features_split  # noqa: E402
        from monitoring import compute_psi  # noqa: E402
        from ml_scorer import FEATURE_NAMES  # noqa: E402

        full_path = Path(__file__).parent.parent / "training" / "synthetic_ml_scoring_tasks.csv"
        rows = list(csv.DictReader(full_path.open()))
        affected = [r for r in rows if r["scenario"] in ("loud_burst", "benign_shared_nat")]
        restricted_path = tmp_path / "affected_only.csv"
        with restricted_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(affected)

        X_ref, X_cur = _load_features_split(restricted_path, "drifted_period")
        psis = {name: compute_psi(X_ref[:, i], X_cur[:, i]) for i, name in enumerate(FEATURE_NAMES)}
        assert psis["fan_in_count"] >= 0.25, (
            f"Restricting to the two scenarios that actually shift no longer alerts "
            f"(fan_in_count PSI={psis['fan_in_count']:.4f}) -- the drift-detection math itself "
            f"may be broken now, unlike the pooled case above which is expected to stay quiet."
        )
