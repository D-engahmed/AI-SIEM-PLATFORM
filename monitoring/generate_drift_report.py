"""
Offline batch drift report: compares the feature distribution of two CSVs
(same schema as training/synthetic_ml_scoring_tasks.csv) via the same PSI
math src/monitoring.py uses online, for cases where you want a report
without running the service -- e.g. "did last week's traffic dump drift
from the training baseline" as a scheduled job, not a live metric.

Reuses row_to_task()/FeatureEngineer from the actual serving code (via
train_model.py's helpers) so this report can never silently diverge from
what the running service would have computed for the same rows.

USAGE:
    # Compare two separate files (e.g. training baseline vs. a fresh export):
    python monitoring/generate_drift_report.py \\
        --reference training/synthetic_ml_scoring_tasks.csv \\
        --current path/to/last_weeks_export.csv

    # Self-check against the KNOWN injected drift in the synthetic data
    # (see generate_synthetic_data.py's module docstring -- this invocation
    # is what proves the drift-detection math itself isn't broken, since the
    # shift here is by construction, not a claim):
    python monitoring/generate_drift_report.py \\
        --reference training/synthetic_ml_scoring_tasks.csv \\
        --split-column drifted_period

Exit code is 1 if any feature is in ALERT status (psi >= --alert-threshold),
so this is usable as a CI/cron gate, not just a printout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from ml_scorer import FEATURE_NAMES, FeatureEngineer  # noqa: E402
from monitoring import compute_psi  # noqa: E402
from train_model import build_matrix, load_rows  # noqa: E402


def _load_features(csv_path: Path) -> np.ndarray:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    X, _, _ = build_matrix(rows)
    return X


def _load_features_split(csv_path: Path, split_column: str) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    reference_rows = [r for r in rows if str(r[split_column]).strip().lower() != "true"]
    current_rows = [r for r in rows if str(r[split_column]).strip().lower() == "true"]
    if not reference_rows or not current_rows:
        raise ValueError(
            f"--split-column {split_column!r} did not produce two non-empty groups "
            f"(reference={len(reference_rows)}, current={len(current_rows)})"
        )
    X_ref, _, _ = build_matrix(reference_rows)
    X_cur, _, _ = build_matrix(current_rows)
    return X_ref, X_cur


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, type=Path, help="Baseline CSV (same schema as synthetic_ml_scoring_tasks.csv)")
    parser.add_argument("--current", type=Path, help="Comparison CSV. Mutually exclusive with --split-column.")
    parser.add_argument("--split-column", type=str, help="Boolean column in --reference to split reference(False)/current(True) instead of using a second file.")
    parser.add_argument("--warn-threshold", type=float, default=0.1)
    parser.add_argument("--alert-threshold", type=float, default=0.25)
    parser.add_argument("--json-out", type=Path, help="Optional path to also write the report as JSON.")
    args = parser.parse_args()

    if bool(args.current) == bool(args.split_column):
        parser.error("exactly one of --current or --split-column is required")

    if args.split_column:
        X_ref, X_cur = _load_features_split(args.reference, args.split_column)
    else:
        X_ref = _load_features(args.reference)
        X_cur = _load_features(args.current)

    report = {}
    n_alert = 0
    print(f"{'feature':28s} {'psi':>8s} {'status':>12s}   n_ref={len(X_ref)} n_cur={len(X_cur)}")
    print("-" * 66)
    for i, feat in enumerate(FEATURE_NAMES):
        psi = compute_psi(X_ref[:, i], X_cur[:, i])
        if psi >= args.alert_threshold:
            status = "ALERT"
            n_alert += 1
        elif psi >= args.warn_threshold:
            status = "WARN"
        else:
            status = "OK"
        report[feat] = {"psi": round(psi, 4), "status": status}
        print(f"{feat:28s} {psi:8.4f} {status:>12s}")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json_out}")

    if n_alert:
        print(f"\n{n_alert} feature(s) in ALERT status.", file=sys.stderr)
        return 1
    print("\nNo features in ALERT status.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
