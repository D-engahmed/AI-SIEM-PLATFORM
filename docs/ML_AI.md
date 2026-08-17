**ROLE:** Senior ML Engineer \& Python Backend Developer

**TASK:** Implement `correlation-ml-service` (Standalone Microservice)



I am providing you with a file containing raw training logs generated from our attack simulator, along with the strict architectural requirements below. 



You are building a decoupled ML microservice that sits downstream of an existing rule-based `GraphPivotStrategy`. It consumes graph-based feature payloads (multi-entity pivoting against a target), scores the convergence risk using an ML model, and publishes actionable incidents. 



### 1. Kafka Contract

*   **Consumer Group:** `correlation-ml-service` (Must be strictly unique to this service).

*   **Consume Topic:** `ml-scoring-tasks` (Expect 1 JSON message per triggering event).

*   **Produce Topic (Success):** `incidents` (Keyed by `correlation\_id` string).

*   **Produce Topic (Failure):** `dlq-correlation-ml` (Route ALL deserialization, validation, or inference errors here along with the original payload and exception. Do NOT reuse upstream DLQs).



### 2. Upstream Context \& Feature Engineering (CRITICAL)

The upstream graph tracks distinct source IPs touching the same target. Nodes expire hard after 24 hours (epoch). You will receive a payload containing `graph\_features` and `signal\_context`. 

*   **Alert Dedup (DO NOT IGNORE):** `GraphPivotStrategy` ONLY publishes when a target's fan-in degree reaches a NEW HIGH. You will NOT get a steady stream of signals per log. `fan\_in\_count` for the same target will always strictly increase. **Do NOT build features assuming message frequency is meaningful** (low frequency might just mean the upstream dedup is silencing repeats).

*   `target\_kind`: Will only ever be `"user"` in practice right now. Do NOT build device-specific feature logic, as device identity tracking was disabled upstream due to false-positive patterns.

*   `epoch\_age\_seconds`: How long the target node has been alive in the graph without a full reset\[cite: 2]. Engineer rate/time-decay features using this (e.g., fan-in count relative to epoch age) to catch "low and slow" behavior\[cite: 2].

*   `ti\_matched`: Treat as a strong prior (the IP hit a threat-intel feed independently)\[cite: 2].

*   Handle missing or null fields in `graph\_features` or `signal\_context` explicitly in your pipeline; upstream sources don't always populate everything\[cite: 2].



### 3. Output Payload (`incidents`)

Your output must perfectly match this exact schema\[cite: 2]:

```json

{

&#x20; "correlation\_id": "uuid-string",

&#x20; "strategy\_name": "GraphMLScoring",

&#x20; "linked\_keys": {"ip": "203.0.113.45", "user": "svc\_backup\_admin"},

&#x20; "signal\_ids": \["uuid-string"],

&#x20; "risk\_score": 742.0,

&#x20; "status": "ESCALATED\_TO\_INCIDENT",

&#x20; "degraded\_mode": false,

&#x20; "window\_start": "2026-08-12T10:15:00Z",

&#x20; "window\_end": "2026-08-12T10:25:00Z",

&#x20; "created\_at": "2026-08-12T10:25:03Z",

&#x20; "updated\_at": null

}
```



&#x20;   risk\_score: Float between 0.0 and 1000.0 inclusive\[cite: 2].



&#x20;   strategy\_name: MUST be exactly "GraphMLScoring"\[cite: 2].



&#x20;   status: ONLY publish to incidents when the status is "ESCALATED\_TO\_INCIDENT" based on your model's threshold. Drop the message silently otherwise\[cite: 2].



&#x20;   linked\_keys: Max 10 keys (≤50 chars/key, ≤255 chars/value)\[cite: 2].



&#x20;   All timestamps must be UTC, ISO 8601\[cite: 2].



4\. ML Strategy \& Non-Negotiables



&#x20;   Model Type: Start with a simple, interpretable classifier (XGBoost, LightGBM, Logistic Regression)\[cite: 2]. NO deep learning, and NO unsupervised anomaly detection (like Isolation Forest) as the primary model\[cite: 2].



&#x20;   Statelessness: No Redis or Postgres connections\[cite: 2]. Everything must be held in process memory\[cite: 2].



&#x20;   Startup Cost: The model MUST load exactly once at process startup, not per-message\[cite: 2].



&#x20;   Security: Treat every string field (e.g., target\_value, source\_ip) as untrusted, attacker-controlled input. Do not build filesystem paths or SQL from them\[cite: 2].



&#x20;   Memory Bounds: Bound your own memory use. No unbounded queues or buffers keyed by attacker-influenced values\[cite: 2].



&#x20;   Fail-Closed: If Kafka/model backend fails, route to DLQ. Never block the consumer loop\[cite: 2].



&#x20;   Auditability: Version your model artifact and log which model\_version produced each score internally\[cite: 2].



Deliverables Required (in services/correlation-ml-service/src/)



Please provide the complete, production-ready Python codebase for:



&#x20;   main.py (App init, model load, Kafka wiring, graceful shutdown).



&#x20;   config.py (Pydantic BaseSettings).



&#x20;   ml\_scorer.py (Feature engineering pipeline + interpretable ML logic).



&#x20;   ml\_consumer.py (Inherits from shared.kafka, parses payload, calls scorer, routes to incidents or DLQ).



&#x20;   Dockerfile (Python 3.11-slim).

