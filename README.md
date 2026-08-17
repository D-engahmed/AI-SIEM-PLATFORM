# correlation-ml-service

## What's actually verified vs. what's assumed

Read this before trusting the artifact in `artifacts/`. Three things were
requested that I did not have: a real sample of the `ml-scoring-tasks`
payload, a training notebook, and the `shared.kafka` interface. I built
working stand-ins for all three, clearly marked, rather than silently
guessing and shipping it as if it were verified. Specifically:

| Piece | Status |
|---|---|
| `schemas.py` request/response shape | **Verified against spec text**, but the exact upstream field names (`graph_features.fan_in_count`, `signal_context.triggering_source_ip`, etc.) are a **reconstruction**, not confirmed against a real message. All upstream-facing field names live in this one file for that reason. |
| Output schema (`incidents` topic) | **Verified two ways**: matches the spec's example byte-for-byte, and matches the `strategy_name`-tagged records actually observed in `all_attack_incidents.json` (timestamp format, field set, `Z`-suffixed ISO8601 with microseconds). |
| Feature engineering + model | Trained and **tested** (11/11 unit tests pass, including a targeted regression check for the "low and slow" pattern), but trained on **synthetic data** (`training/generate_synthetic_data.py`), not real traffic. AUC/AP numbers in `artifacts/model_latest.metrics.json` describe how well it learned the synthetic labels I wrote, not real-world performance. |
| `ml_consumer.py` business logic (`MLScoringHandler`) | **Verified**: pure function, unit tested against garbage bytes, missing fields, threshold boundaries, contract-violating output. This part should survive the `shared.kafka` swap unchanged. |
| `ml_consumer.py` outer Kafka loop (`StandaloneRunner`) | **Not verified against `shared.kafka`** — that library's interface wasn't available. This is a working `confluent_kafka`-based loop instead, built specifically so it's a ~20-line adapter to swap out, not a rewrite. See the module docstring. |
| `Dockerfile` | Reviewed, not build-tested — no Docker daemon in this environment. |

## Before this goes anywhere near production

1. **Send me (or replace) a real `ml-scoring-tasks` message.** Everything about the model's actual usefulness rests on the feature-to-field mapping in `schemas.py` being right. If `signal_context` doesn't actually carry a single "triggering IP," `linked_keys.ip` in the output will be wrong for every incident this service ever publishes.
2. **Get real labeled history from GraphPivotStrategy**, even a few hundred rows, and retrain. `training/train_model.py` already imports the exact serving feature code, so swapping the CSV source is the only change needed.
3. **Resolve `shared.kafka`.** Point me at it and I'll collapse `StandaloneRunner` into a real subclass.
4. **Decide the `risk_score` scale story.** This is worth flagging loudly: every `strategy_name` currently in `all_attack_incidents.json` (`RiskBasedAlertingStrategy`, `BruteThenLoginPattern`, etc.) tops out around **90–150**, while this service's contract allows up to **1000**. If anything downstream (a dashboard sort, a severity bucket, an auto-triage rule) assumes `risk_score` is roughly comparable across `strategy_name` values, `GraphMLScoring` will dominate every ranking by construction, not because it's finding worse attacks. Either that's already known and handled downstream, or it needs a decision (e.g. rescale to match, or make consumers aware scales differ per strategy).
5. **Decide `escalation_threshold` deliberately.** It defaults to `0.7` (`CML_ESCALATION_THRESHOLD`) purely because it's a round number — it is not calibrated against any real cost-of-false-positive vs cost-of-false-negative analysis, because there's no real labeled data yet to calibrate against.

## Design decisions worth knowing about (in case they're wrong for your setup)

- **`degraded_mode`** is set when `>= 2` expected fields were missing/imputed for a message (`CML_DEGRADED_MODE_IMPUTED_FIELD_THRESHOLD`). The service still scores and can still escalate a degraded message — "fail closed" was read as "never silently lose a signal," not "never score uncertain input." If you want degraded messages routed to DLQ instead of scored, that's a different design and a small change in `ml_consumer.py`.
- **At-least-once, not exactly-once.** Offsets commit only after the produce (incident or DLQ) is confirmed delivered. A crash between produce-confirmed and commit can redeliver and reproduce the same `correlation_id`. This was accepted rather than engineered around, because the spec's statelessness constraint rules out the usual fix (an idempotency cache), and a duplicate incident with the same `correlation_id` is a minor annoyance for a downstream dedup step, not data loss.
- **No cross-message state anywhere**, on purpose — see the docstring at the top of `ml_scorer.py`. It would be easy to "improve" this service by caching recent scores per target, and that's exactly the thing the spec's statelessness/memory-bounds constraints rule out.
- **Monotonic constraints** are only applied to features with a defensible direction from the spec text (`fan_in_count`, `fan_in_rate_log1p`, `low_and_slow_ratio`, `ti_matched_positive` → risk non-decreasing). `epoch_age_seconds` alone is left unconstrained because its direction is genuinely ambiguous outside the ratio features.

## Running it

```bash
cd services/correlation-ml-service
pip install -r requirements.txt

# 1. Train (uses synthetic data until you have real samples)
python training/generate_synthetic_data.py
python training/train_model.py     # fails loudly if the low-and-slow trap check regresses

# 2. Test
python -m pytest tests/ -v

# 3. Run (needs a real Kafka broker reachable at CML_KAFKA_BOOTSTRAP_SERVERS)
python src/main.py
```
