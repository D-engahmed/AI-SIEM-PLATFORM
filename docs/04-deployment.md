# 04 — Deployment

## Process layout

One container, two concurrent asyncio tasks sharing one event loop
(`src/main.py::_run`):

1. The Kafka consumer (`CorrelationMLConsumer`, subclassing the local
   `shared/kafka` stand-in — see "Kafka library" below).
2. A FastAPI app (`src/api.py`) serving `/healthz`, `/metrics`, `/score`, and
   `/monitoring/drift` on `CML_API_PORT` (default 9100, per AD-041).

Startup order is deliberate and fail-fast (`main.py::main`):

1. Load config — fail fast on bad env.
2. Load the model **once** (`ModelScorer.load` or `.load_from_mlflow`) — fail
   fast if the artifact is missing, corrupt, or has a feature-order mismatch.
   Refuses to start rather than silently score with a mismatched pipeline.
3. Load the drift-monitoring reference distribution — **not** fail-fast: a
   missing reference artifact logs loudly and disables monitoring
   (`monitor=None`), since a service that can score but can't self-monitor is
   degraded, not broken.
4. Only then start consuming.

## Model source: file vs MLflow registry

`CML_MODEL_SOURCE=file` (default) reads `artifacts/model_latest.joblib`
directly — no MLflow dependency at serving time, appropriate for the
low-latency Kafka path.

`CML_MODEL_SOURCE=mlflow` resolves `CML_MODEL_MLFLOW_URI` (default
`models:/correlation-ml-service-risk-model/Staging`) via the MLflow Model
Registry — lets Staging→Production promotion control what's served without a
redeploy. Requires the model version to carry a `model_family` tag (set by the
notebook's registration cell); refuses to guess which flavor-specific loader
(`mlflow.xgboost` vs `mlflow.lightgbm`) to use otherwise. Verified working
end-to-end against a real registered model during this build.

## Kafka library

`shared/kafka/` is a **local stand-in**, not the actual platform
`shared.kafka` package — a deliberate decision (not a workaround for not
having the real one yet) to avoid depending on internal infrastructure this
repo doesn't have access to. Built on `aiokafka` (matching the platform's
`issue-008` migration off `kafka-python`) and `orjson` (`AD-032`).

**Real risk of this choice, stated plainly**: inheriting from the actual
`shared.kafka.BaseConsumer`/`BaseProducer` would get its real retry logic,
serialization behavior, and DLQ envelope format for free, with future patches
arriving automatically. This local reimplementation is a best-effort guess at
the described interface (`process_message(self, payload: dict)`,
`route_to_dlq(payload, error)`) and can silently drift from the real thing —
the DLQ envelope shape specifically is the sharpest risk, since any
platform-wide tooling that reads `dlq-*` topics uniformly only needs one
field-name difference to break against this service's topic.

## Real bug found and fixed in the local producer

`orjson.OPT_UTC_Z` only Z-suffixes datetimes that are **already** UTC — it
does not convert other offsets first. A non-UTC aware datetime would have
silently serialized as `...+03:00` instead of being converted, breaking the
"always UTC Z" wire format every consumer of this topic expects (verified
against `docs/all_attack_incidents.json` samples). Fixed by handling datetime
conversion explicitly via `OPT_PASSTHROUGH_DATETIME` + a custom `default`
function in `shared/kafka/base_producer.py`. Caught by
`tests/test_wire_serialization.py`, written specifically because the
timestamp-format test coverage was otherwise lost when the old
`to_wire_dict()`/`_iso_z()` methods were removed.

## Docker

- `HEALTHCHECK` + `curl` installed in the runtime stage (Standard 32).
- `COPY shared/ /app/shared/` — note this copies the **local stand-in**
  above, not the real platform package (see "Kafka library").
- `PYTHONPATH` includes `/app/src` and `/app` so both `api`/`config`/etc. and
  `shared.kafka.*` resolve without a package install step.
- Runtime image currently installs the **full** `requirements.txt** (including
  training-only packages — scikit-learn, lightgbm, shap, mlflow, jupyter,
  pandas, pytest) into the container, per the build stage's
  `pip install --target=/deps -r requirements.txt`. Not split into
  runtime-vs-training requirement files here — real image-size opportunity,
  not urgent, flagged rather than silently left undocumented.

## Config reference

All settings are `CML_`-prefixed env vars (`src/config.py`, `Settings`). Key
ones not obvious from name alone:

| Var | Default | Note |
|---|---|---|
| `CML_ESCALATION_THRESHOLD` | `0.5` | Confirmed explicitly 2026-08-17 (a reviewer offered "0.4 or 0.5 until we have real labeled data" — explicitly provisional). Hard-rejects exactly `0.0`/`1.0` (100% FP / 100% FN respectively) — a **fixed** bug: the validator used to just `pass`, though the Field-level `ge=0.0, le=1.0` already independently blocked genuinely invalid values like `5.0`. |
| `CML_DEGRADED_MODE_IMPUTED_FIELD_THRESHOLD` | `2` | How many missing/imputed fields before `degraded_mode=true`. |
| `CML_API_PORT` | `9100` | AD-041. |
| `CML_FEATURE_REFERENCE_PATH` | `artifacts/feature_reference_distribution.npz` | See `docs/06-monitoring.md`. |
| `CML_MONITORING_RING_BUFFER_SIZE` | `5000` | Per-feature, bounded on purpose — observability state, same memory discipline as everywhere else in this service. |
