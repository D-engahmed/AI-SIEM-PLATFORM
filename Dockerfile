# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt


FROM python:3.11-slim AS runtime

# Standard 32: curl is required in the runtime stage because HEALTHCHECK
# below uses it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 svc
WORKDIR /app

COPY --from=builder /deps /app/deps
COPY src/ /app/src/
# Standard 32: any service importing from shared/ must explicitly COPY it.
COPY shared/ /app/shared/
COPY artifacts/model_latest.joblib /app/artifacts/model_latest.joblib
# NOTE: the artifact baked in here (if present at build time) was trained
# on training/generate_synthetic_data.py's synthetic data -- see README.
# Overwrite artifacts/model_latest.joblib with a real artifact, or mount
# one at runtime via CML_MODEL_ARTIFACT_PATH, before running this in
# production.

ENV PYTHONPATH="/app/deps:/app/src:/app" \
    PYTHONUNBUFFERED=1 \
    CML_MODEL_ARTIFACT_PATH="/app/artifacts/model_latest.joblib" \
    CML_API_PORT="9100"

EXPOSE 9100

# Standard 32 / AD-041: /healthz on the same port as /metrics (9100).
# start-period is generous because model loading (ModelScorer.load) plus
# aiokafka broker connect both happen before the API server starts serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:9100/healthz || exit 1

USER svc

ENTRYPOINT ["python", "/app/src/main.py"]
