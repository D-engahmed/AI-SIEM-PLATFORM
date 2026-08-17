"""
Synthetic training data for correlation-ml-service.

WHY THIS FILE EXISTS: no real ml-scoring-tasks samples or training
notebook were available at build time -- see src/schemas.py docstring
and docs/01-data-and-features.md for the full gap list. This generator
is a documented, inspectable stand-in, not a claim that it matches
production traffic. Swap it for real historical GraphPivotStrategy
payloads as soon as you have them; nothing downstream (train_model.py,
model_selection.ipynb, evaluate_model.py) cares where the CSV came from,
they only care that it has these columns.

Scenarios encoded (each is a labeled HYPOTHESIS about what "risky" means
here, not a certainty -- every one of these is something a real security
engineer should sign off on, not something to trust because a model was
trained on it):

  - benign_quiet:         few distinct IPs, short-lived node -> label 0
  - benign_slow_natural:   few distinct IPs but spread over a long window
                           (e.g. legit users roaming across NAT/VPN egress
                           over a day) -> label 0. Exists specifically to
                           stop the model from learning "old node = bad".
  - benign_shared_nat:     MODERATE fan-in (5-9 distinct IPs) over a short-
                           to-medium window, from many legitimate users
                           sharing one corporate NAT/VPN egress IP -> label
                           0. This is a well-known real false-positive
                           source for "distinct source IP" fan-in detectors
                           specifically, so it's included to stress-test
                           exactly the failure mode this service is most
                           at risk of (flagging a shared corporate egress
                           as a coordinated attack).
  - loud_burst:            many distinct IPs in a short window (classic
                           fast brute-force fan-in) -> label 1
  - low_and_slow:          many distinct IPs accumulated over a long
                           window, so the *rate* looks unremarkable but
                           the *count* is high -> label 1. This is the
                           pattern the spec calls out explicitly.
  - ti_confirmed:          threat-intel hit regardless of fan-in shape ->
                           label 1 (ti_matched is stated as a strong prior)
  - ambiguous_mid:         genuinely in between, label assigned with
                           noise, so the model isn't trained on a
                           perfectly separable toy problem.

Time axis + injected drift:
  Every row gets an `event_time` spread across a 45-day synthetic window.
  In the final DRIFT_WINDOW_FRACTION of that window, two things shift on
  purpose: loud_burst fan-in counts scale up (simulating attacker
  adaptation to a higher noise floor) and the benign_shared_nat volume
  increases (simulating a growing office). This isn't meant to be a
  realistic drift *cause* -- it exists so docs/06-monitoring.md and
  monitoring/generate_drift_report.py have a KNOWN, verifiable shift to
  detect. If the drift report doesn't flag DRIFT_WINDOW_FRACTION, that's
  a real bug in the monitoring code, not noise.
"""

from __future__ import annotations

import csv
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

random.seed(13)

OUT_PATH = Path(__file__).parent / "synthetic_ml_scoring_tasks.csv"

N_PER_SCENARIO = 2500
MISSING_FIELD_RATE = 0.12  # fraction of rows with a null'd out field, per the spec's "don't assume fully populated"

WINDOW_DAYS = 45
DRIFT_WINDOW_FRACTION = 0.2  # last 20% of the time window is the injected-drift period
WINDOW_START = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _maybe_none(value, rate=MISSING_FIELD_RATE):
    return None if random.random() < rate else value


def _event_time(drifted: bool) -> datetime:
    """drifted=True samples only from the last DRIFT_WINDOW_FRACTION of the window."""
    if drifted:
        start_day = WINDOW_DAYS * (1 - DRIFT_WINDOW_FRACTION)
        day_offset = random.uniform(start_day, WINDOW_DAYS)
    else:
        day_offset = random.uniform(0, WINDOW_DAYS * (1 - DRIFT_WINDOW_FRACTION))
    # day_offset is already continuous (sub-day precision) -- do NOT also add
    # a separate random seconds-of-day on top, or rows near the boundary can
    # spill up to a full day across it (caught by
    # tests/test_data_quality.py::test_drifted_rows_are_strictly_later_in_time).
    return WINDOW_START + timedelta(days=day_offset)


def gen_benign_quiet(drifted: bool):
    fan_in = random.randint(1, 3)
    age = random.uniform(30, 3600)
    ti = False if random.random() < 0.9 else None
    label = 1 if random.random() < 0.02 else 0
    return fan_in, age, ti, label


def gen_benign_slow_natural(drifted: bool):
    fan_in = random.randint(2, 4)
    age = random.uniform(6 * 3600, 23 * 3600)
    ti = False if random.random() < 0.95 else None
    label = 1 if random.random() < 0.03 else 0
    return fan_in, age, ti, label


def gen_benign_shared_nat(drifted: bool):
    # Office growth in the drift window: more people behind the same
    # egress IP, so the fan-in count creeps up -- still benign.
    lo, hi = (7, 12) if drifted else (5, 9)
    fan_in = random.randint(lo, hi)
    age = random.uniform(1800, 10 * 3600)
    ti = False if random.random() < 0.97 else None
    label = 1 if random.random() < 0.04 else 0  # small noise floor -- this is the hardest benign case on purpose
    return fan_in, age, ti, label


def gen_loud_burst(drifted: bool):
    # Attacker-adaptation drift: bigger bursts in the later window.
    lo, hi = (14, 50) if drifted else (8, 35)
    fan_in = random.randint(lo, hi)
    age = random.uniform(5, 300)
    ti = random.random() < 0.3
    label = 1 if random.random() < 0.97 else 0
    return fan_in, age, ti, label


def gen_low_and_slow(drifted: bool):
    fan_in = random.randint(6, 22)
    age = random.uniform(4 * 3600, 23.5 * 3600)
    ti = random.random() < 0.35
    label = 1 if random.random() < 0.95 else 0
    return fan_in, age, ti, label


def gen_ti_confirmed(drifted: bool):
    fan_in = random.randint(1, 10)
    age = random.uniform(30, 20 * 3600)
    ti = True
    label = 1 if random.random() < 0.9 else 0
    return fan_in, age, ti, label


def gen_ambiguous_mid(drifted: bool):
    fan_in = random.randint(4, 7)
    age = random.uniform(1800, 5 * 3600)
    ti = random.random() < 0.15
    label = 1 if random.random() < 0.4 else 0
    return fan_in, age, ti, label


SCENARIOS = {
    "benign_quiet": gen_benign_quiet,
    "benign_slow_natural": gen_benign_slow_natural,
    "benign_shared_nat": gen_benign_shared_nat,
    "loud_burst": gen_loud_burst,
    "low_and_slow": gen_low_and_slow,
    "ti_confirmed": gen_ti_confirmed,
    "ambiguous_mid": gen_ambiguous_mid,
}


def main():
    rows = []
    for scenario_name, gen in SCENARIOS.items():
        for i in range(N_PER_SCENARIO):
            drifted = (i / N_PER_SCENARIO) >= (1 - DRIFT_WINDOW_FRACTION)
            fan_in, age, ti, label = gen(drifted)
            n_signals = random.randint(1, 3)
            event_time = _event_time(drifted)
            rows.append(
                {
                    "scenario": scenario_name,
                    "drifted_period": drifted,
                    "event_time": event_time.isoformat(),
                    "correlation_id": str(uuid.uuid4()),
                    "target_kind": "user",
                    "target_value": f"user_{random.randint(1, 5000)}",
                    "fan_in_count": _maybe_none(fan_in),
                    "epoch_age_seconds": _maybe_none(round(age, 2)),
                    "triggering_source_ip": f"203.0.113.{random.randint(1, 254)}",
                    "ti_matched": ti,
                    "signal_id_count": n_signals,
                    "label": label,
                }
            )

    rows.sort(key=lambda r: r["event_time"])  # chronological, for time-aware splitting/plots
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows to {OUT_PATH}")
    pos = sum(r["label"] for r in rows)
    print(f"Positive rate: {pos}/{len(rows)} = {pos/len(rows):.1%}")
    print(f"Scenarios: {list(SCENARIOS.keys())}")
    print(f"Time window: {WINDOW_START.date()} .. {(WINDOW_START + timedelta(days=WINDOW_DAYS)).date()}")
    print(f"Drift injected in last {DRIFT_WINDOW_FRACTION:.0%} of window (loud_burst scale-up, benign_shared_nat growth)")


if __name__ == "__main__":
    main()
