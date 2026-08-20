# 03 — Evaluation and testing

## Test suite layout

155 tests across `tests/`, run with `python -m pytest tests/ -v` (needs
`PYTHONPATH=src:.` and `artifacts/model_latest.joblib` to exist — run
`training/generate_synthetic_data.py` then `training/train_model.py` or the
notebook first). 89% line coverage on `src/`.

| File | Covers |
|---|---|
| `test_schemas.py` | Outbound `IncidentEvent` contract: risk_score bounds, severity enum, linked_keys bounds, root-vs-metadata field placement. Rewritten 2026-08-17 for the AD-061 shape — the pre-rewrite version tested a 0–1000 range and fields that no longer exist. |
| `test_wire_serialization.py` | The producer's actual orjson serialization — timestamp UTC conversion, UUID formatting. Added after finding `OPT_UTC_Z` doesn't convert non-UTC offsets (see `docs/04-deployment.md`). |
| `test_ml_scorer.py` | Feature engineering, the `epoch_age_seconds` imputation fix, artifact reload determinism. |
| `test_ml_consumer.py` | Validation → inference → publish pipeline; DLQ routing per stage; the degraded-mode counter's placement (consumer, not the shared handler — see `docs/05-api-and-service-endpoints.md`). |
| `test_data_quality.py` | Sanity checks on the synthetic data itself — scenario shapes are actually distinguishable, drift is really injected, **and the drift-dilution finding is pinned as a regression test** (see `docs/06-monitoring.md`). |
| `test_model_quality.py` | Model-level checks against the trained artifact (not just the feature pipeline). |
| `test_model_stress.py` | Robustness under extreme/adversarial input — see below. |
| `test_config.py`, `test_main.py`, `test_api.py`, `test_monitoring.py` | Had **zero coverage** before 2026-08-17/18 — config validation, the fail-fast entrypoint, the FastAPI endpoints, PSI drift math. |

## The two checks that matter more than aggregate metrics

Both run against every candidate in the model-selection notebook, not just the
eventual winner (see `docs/02-model-creation-and-selection.md`):

- **Low-and-slow trap**: same instantaneous rate, very different absolute
  count/age. A rate-only model scores these identically; this service exists
  specifically to not do that.
- **`benign_shared_nat` false-positive check**: moderate fan-in from a shared
  corporate egress IP — the realistic 3am-page failure mode.

`training/train_model.py` also runs the trap check automatically and
**exits non-zero if it fails** — a CI gate, not just a printed warning.

## Model robustness (`test_model_stress.py`)

"Does the model handle stress" interpreted as: extreme, malformed, and
adversarial-ish inputs must never crash the scoring path or produce
NaN/inf/out-of-bounds output. These tests don't assert the model is *right*
about weird inputs — only that it doesn't fall over.

- Extreme `fan_in_count` (0 to 1,000,000) and `epoch_age_seconds` (0 to 1e15).
- **Negative values**: assumed unconstrained when first writing these tests —
  checked the actual source before trusting that assumption, and found
  `GraphFeatures` already has `_non_negative_fan_in`/`_non_negative_age`
  validators. Tests corrected to assert the real (better) behavior:
  rejection at the schema layer, not silent pass-through.
- All-fields-missing, empty target values, adversarial strings (SQL injection
  attempts, null bytes, unicode direction overrides, 10k-char strings) in
  `target_value` — confirms `ml_scorer.py`'s security note (treated as opaque,
  untrusted) holds under actually adversarial input.
- Statelessness under repetition: identical input scored 200 times gives
  identical output; alternating extreme and normal inputs doesn't leak state
  between calls.
- A `@pytest.mark.slow` 5,000-iteration randomized sweep (deselect with
  `-m "not slow"` for the fast path) — property-based-flavored crash/bounds
  check, not an accuracy check.

## Throughput/latency stress

Different kind of "stress" — needs a live running process, not part of the
normal pytest suite. See `docs/07-stress-testing.md`.
