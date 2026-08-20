# 06 — Monitoring

## What this is and isn't

`src/monitoring.py` implements Population Stability Index (PSI) feature-drift
detection — "does recent traffic, as a whole, look like the training data".
It is **not** anomaly detection on individual entities: the ring buffer holds
aggregate feature values only, never anything keyed by `target_value`,
`source_ip`, or `correlation_id` (would violate the statelessness constraint
in `ml_scorer.py`). It cannot answer "is *this* user suspicious" — only
"has the population shifted".

**Why PSI and not something heavier** (Evidently, WhyLabs, a KS-test service):
PSI is the industry-standard metric for exactly this question and needs
nothing beyond `numpy` — consistent with this service's whole design
philosophy of no heavy runtime dependencies, no external state store. A
heavier drift platform is a legitimate future upgrade, not a default.

## How it works

1. `training/train_model.py` (and the notebook's registration cell) saves a
   capped 5,000-sample-per-feature reference distribution to
   `artifacts/feature_reference_distribution.npz`, drawn from the full
   labeled dataset (not just the training split — the reference should
   describe what the model was built to expect overall).
2. `FeatureDriftMonitor.load()` reads that file at service startup. **Missing
   file is not fatal** — logs loudly, monitoring disabled, scoring continues
   (`monitor=None`). A service that can score but can't self-monitor is
   degraded, not broken.
3. Every real Kafka message scored gets its feature vector recorded into a
   bounded per-feature ring buffer (`CML_MONITORING_RING_BUFFER_SIZE`, default
   5,000) — **not** via `POST /score`, which is deliberately side-effect-free
   (see `docs/05-api-and-service-endpoints.md`).
4. PSI is computed **on demand** (when `/metrics` or `/monitoring/drift` is
   hit), not per-message — cheap enough (a few numpy histogram calls) that
   there's no reason to do it more often than something actually reads it.
5. Below 30 samples in the ring buffer, PSI is reported as
   `INSUFFICIENT_DATA` (not `0.0`) — a few dozen samples compared against a
   10-bin histogram produces huge PSI values from sampling noise alone, not
   real drift; reporting `0.0` there would look like "confirmed no drift"
   instead of "can't say yet".

Thresholds (industry-standard bands): `< 0.1` OK, `0.1–0.25` WARN,
`> 0.25` ALERT.

## `monitoring/generate_drift_report.py`

Offline batch equivalent — compares two CSVs (or one CSV split by a boolean
column) via the same PSI math, for "did last week's traffic dump drift from
the training baseline" as a scheduled job rather than a live metric. Exit
code 1 on any ALERT, so it's usable as a CI/cron gate.

## A real finding from actually running the self-check, not just building it

`generate_synthetic_data.py`'s drift injection (see `docs/01-data-and-features.md`)
only shifts 2 of the 7 scenarios (`loud_burst`, `benign_shared_nat`). The
original code comment claimed "if the drift report doesn't flag
`DRIFT_WINDOW_FRACTION`, that's a real bug in the monitoring code, not
noise." Running that exact check surfaced **no alert** — comparing
`drifted_period=True` vs `False` pooled across all 7 scenarios keeps every
feature's PSI under 0.05.

Investigated rather than shrugged off: restricting the same comparison to
just the 2 affected scenarios **does** alert (`fan_in_count` PSI = 2.37, well
past 0.25). So the drift is real and the detector genuinely works — the
pooled aggregate simply dilutes a shift that's concentrated in 2 of 7
scenarios below the alert threshold, since the other 5 scenarios contribute
unshifted data to both sides of the comparison. This is a well-known, genuine
PSI limitation (aggregate population-level drift detection can miss a real
shift confined to a subpopulation), not a bug in `compute_psi` — that
function is separately unit-tested against synthetic shifts of known size and
correctly flags them.

**The original comment's expectation was simply wrong and has been corrected**
in `training/generate_synthetic_data.py`'s module docstring. Both behaviors
(pooled = quiet, scenario-restricted = alert) are now pinned as regression
tests in `tests/test_data_quality.py::TestDriftDetectionRealism`, so this
stays true and documented rather than silently drifting back into being an
unverified claim.

**Practical implication for reading `cml_feature_psi` in production**: a
quiet aggregate PSI does not rule out a real, meaningful shift in one
customer segment, one `target_kind`, or one time-of-day pattern — it only
rules out a shift large enough to move the *whole* population. If you have a
reason to suspect a specific subpopulation is drifting, `generate_drift_report.py`
against a filtered CSV (like the `--split-column` self-check above) is the
tool for that, not the aggregate `/metrics` gauge.
