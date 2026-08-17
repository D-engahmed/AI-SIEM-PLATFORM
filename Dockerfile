# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /build

# confluent-kafka installs from a prebuilt manylinux wheel for standard
# platforms; build-essential is kept only as a fallback for platforms
# where pip has to compile against librdkafka. Comment it out if your
# CI confirms wheels resolve cleanly and you want a smaller/faster build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt


FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 10001 svc
WORKDIR /app

COPY --from=builder /deps /app/deps
COPY src/ /app/src/
COPY artifacts/model_latest.joblib /app/artifacts/model_latest.joblib
# NOTE: the artifact baked in here (if present at build time) was trained
# on training/generate_synthetic_data.py's synthetic data -- see README.
# Overwrite artifacts/model_latest.joblib with a real artifact, or mount
# one at runtime via CML_MODEL_ARTIFACT_PATH, before running this in
# production.

ENV PYTHONPATH="/app/deps:/app/src" \
    PYTHONUNBUFFERED=1 \
    CML_MODEL_ARTIFACT_PATH="/app/artifacts/model_latest.joblib"

USER svc

ENTRYPOINT ["python", "/app/src/main.py"]
