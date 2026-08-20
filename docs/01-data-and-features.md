# 01 — Data and features

## Why synthetic data exists at all

No real `ml-scoring-tasks` traffic sample was available when this service was
built (see `src/schemas.py`'s module docstring). `training/generate_synthetic_data.py`
is a documented, inspectable stand-in — not a claim that it matches production
traffic. Swap it for real historical `GraphPivotStrategy` output as soon as you
have it; nothing downstream (`train_model.py`, `model_selection.ipynb`,
`monitoring/generate_drift_report.py`) cares where the CSV came from, they only
care that it has the same columns.

Current volume: **56,000 rows** (8,000 per scenario × 7 scenarios), raised from
17,500 on 2026-08-18 by increasing `N_PER_SCENARIO` — not by adding scenarios or
widening ranges. More rows from an already-reviewed generative process is a
reasonable way to get "more data"; rows from an invented 8th scenario nobody
signed off on would not be.

## The 7 scenarios

| Scenario | Label | Why it exists |
|---|---|---|
| `benign_quiet` | 0 | Baseline: few IPs, short-lived node. |
| `benign_slow_natural` | 0 | Few IPs spread over a long window — stops the model learning "old node = bad". |
| `benign_shared_nat` | 0 | Moderate fan-in from many legitimate users behind one corporate NAT/VPN egress IP. This is a well-known real false-positive source for distinct-source-IP fan-in detectors specifically — included to stress-test the exact failure mode this service is most at risk of. Has a deliberate 4% label-noise floor (not 40% — an early external review mischaracterized this; see `training/generate_synthetic_data.py`'s docstring history). |
| `loud_burst` | 1 | Classic fast brute-force fan-in — many IPs in a short window. |
| `low_and_slow` | 1 | Many IPs accumulated over a long window, so the *rate* looks unremarkable but the *count* is high. The pattern the spec calls out explicitly by name. |
| `ti_confirmed` | 1 | Threat-intel hit regardless of fan-in shape. |
| `ambiguous_mid` | noisy (40/60) | Genuinely in-between, labeled with real noise on purpose — the model is not trained on a perfectly separable toy problem. |

## Time axis and injected drift

Every row gets an `event_time` across a 45-day window. In the final 20%
(`DRIFT_WINDOW_FRACTION`), `loud_burst` fan-in scales up and `benign_shared_nat`
volume grows — a known, verifiable shift for `docs/06-monitoring.md` and
`monitoring/generate_drift_report.py` to detect. **Important, corrected
finding**: this drift is only detectable if you restrict the comparison to the
two affected scenarios — pooled across all 7, it's diluted below the alert
threshold. See `docs/06-monitoring.md` for the full story; this is a real PSI
limitation, not a bug, and the original code comment claiming otherwise was
wrong and has been fixed.

## Feature engineering (`src/ml_scorer.py::FeatureEngineer`)

Nine features, computed from `graph_features` + `signal_context`:

- `fan_in_count`, `fan_in_count_missing`
- `epoch_age_seconds_log1p`, `epoch_age_seconds_missing`
- `fan_in_rate_log1p` — **fixed 2026-08-17**: missing age used to impute to a
  1-second floor and feed straight into this rate, artificially inflating it
  for any fan-in count. Combined with this feature's `+1` monotonic
  constraint, the model could never down-score a message just because its age
  was unknown. Now neutral (0.0) when age is imputed; the (unconstrained)
  missing-flag carries that signal instead.
- `low_and_slow_ratio` — `fan_in_count / log1p(epoch_age_seconds)`, deliberately
  not just inverse rate, so it rewards sustained *volume* over a long window
  even when the instantaneous rate looks tiny.
- `ti_matched_positive`, `ti_matched_known`
- `signal_id_count`

Feature engineering is imported directly from `src/ml_scorer.py` by
`train_model.py`, `model_selection.ipynb`, and `monitoring/generate_drift_report.py`
— never reimplemented — so there is no train/serve/monitor skew by construction.

## Known, deliberately unresolved gaps

- The **inbound** schema (`MLScoringTask`/`GraphFeatures`/`SignalContext`) is a
  reconstruction from prose, never verified against a real message. A prior
  external review's proposed root-level restructuring was never actually
  re-confirmed and still contradicts the original written spec — left
  unchanged pending real confirmation.
- `generate_synthetic_data.py`'s noise levels and scenario boundary overlaps
  (`benign_shared_nat` vs `low_and_slow`) were flagged by an early review as
  "bugs" to fix; both have real arguments against "fixing" them (see that
  file's docstring) and were deliberately left alone.
