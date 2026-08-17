# AI-SIEM PLATFORM — PROJECT MEMORY FILE
# ============================================================
# PURPOSE : Persistent memory — paste at start of every session
# VERSION : Session 4 Final — Phase 1 COMPLETE
# UPDATED : 2026-03-20
# ============================================================

---

## PROJECT IDENTITY

```
Name      : AI-Driven SIEM / SOAR Platform
Location  : E:\AI-SIEM-PLATFORM
Stack     : Python 3.11, FastAPI, Kafka, PostgreSQL 16, Redis 7.2, Docker, React, GPT-4o
Status    : Phase 1 COMPLETE — syslog-receiver + normalizer-worker running + DB writes confirmed
```

---

## ARCHITECTURAL DECISIONS

### AD-001 — Microservice Decomposition Strategy
```
Decision  : Hybrid Tiered Decomposition (Option C)
Result    : 9 deployable microservices
```

### AD-002 — Event Communication Pattern
```
Decision  : Kafka async pipeline + HTTP for sync calls only
Topics    : 9 topics
```

### AD-003 — Database Ownership Model
```
Decision  : Each service owns its tables — no cross-service writes
Read      : api-gateway + dashboard-backend via role_readonly
```

### AD-004 — Correlation State Management
```
Decision  : Redis sliding windows (5-min, keyed by source_ip) TTL 10 min
```

### AD-005 — AI Validation Pipeline
```
Decision  : 5-step validation before accepting LLM output
Steps     : JSON schema → evidence grounding → entity containment
            → confidence threshold (>=0.65) → MITRE validation
Fallback  : Deterministic report if any step fails
```

### AD-006 — Incident Deduplication
```
Decision  : Redis deduplication — 1-hour window per source_ip
```

### AD-007 — Syslog Port Standard
```
Decision  : Port 514 — cap_add: NET_BIND_SERVICE in Docker
```

### AD-008 — Zookeeper Healthcheck
```
Decision  : echo srvr | nc -w 2 localhost 2181 | grep -q Mode
```

### AD-009 — Docker Compose Strategy
```
Decision  : docker-compose.yml (prod base) + override.yml (dev)
```

### AD-010 — Password Isolation Per Service
```
Decision  : Unique password per infrastructure service
```

### AD-011 — PostgreSQL Table Partitioning
```
Decision  : RANGE partitioning by day on events + signals
Status    : IMPLEMENTED Session 3
```

### AD-012 — Kafka DLQ Segmentation
```
Decision  : dlq-normalizer / dlq-detection / dlq-soar
Status    : IMPLEMENTED Session 3
```

### AD-013 — Database Roles & Grants (Least Privilege)
```
Decision  : 8 PostgreSQL roles — one per service
Status    : IMPLEMENTED Session 3
```

### AD-014 — Threat Intelligence via Redis
```
Decision  : Malicious IP lookup (prefix: ti:malicious:{ip}) TTL 24h
Source    : AlienVault OTX
```

### AD-015 — API Rate Limiting
```
Decision  : 100 req/min (authenticated) / 10 req/min (unauthenticated)
```

### AD-016 — High Availability (Deferred)
```
Decision  : Single node dev — HA deferred post-launch
```

### AD-017 — Cold Storage (Deferred)
```
Decision  : MinIO S3 for data > 30 days — deferred Phase 5
```

### AD-018 — Detection Engine Architecture
```
Decision  : Rule-Based + TI + Selective Multi-key Correlation
Redis Budget (512MB):
  Correlation : 200MB | TI : 100MB | Dedup : 100MB | Baseline : 100MB
```

### AD-019 — UUID Primary Keys
```
Decision  : gen_random_uuid() for ALL PKs — no SERIAL
Status    : IMPLEMENTED Session 3
```

### AD-020 — TIMESTAMPTZ Everywhere
```
Decision  : ALL time fields → TIMESTAMPTZ (UTC)
Status    : IMPLEMENTED Session 3
```

### AD-021 — GIN Index on JSONB
```
Decision  : GIN index on every JSONB column
Status    : IMPLEMENTED Session 3
```

### AD-022 — Bcrypt Hashed Password in Seed
```
Decision  : crypt(password, gen_salt('bf', 12)) at INSERT time
Status    : IMPLEMENTED Session 3
```

### AD-023 — REVOKE PUBLIC Privileges
```
Decision  : REVOKE ALL ON SCHEMA public FROM PUBLIC
Status    : IMPLEMENTED Session 3
```

### AD-024 — Kafka Explicit Retention
```
Decision  :
  raw-events / normalized-events → 86400000  (24h)
  signals                        → 259200000 (72h)
  incidents / actions / dlq-*    → 604800000 (7d)
Status    : IMPLEMENTED Session 3
```

### AD-025 — updated_at Trigger
```
Decision  : DB-level trigger fn_set_updated_at() on siem_incidents.incidents
Status    : IMPLEMENTED Session 3
```

### AD-026 — Kafka 3 Partitions Hot Topics
```
Decision  : raw-events, normalized-events, signals → 3 partitions
            All others → 1 partition
Status    : IMPLEMENTED Session 3
```

### AD-027 — Per-Service DB DSNs
```
Decision  : DB_DSN_INGEST ... DB_DSN_READONLY in .env
Status    : IMPLEMENTED Session 3
```

### AD-028 — protocol Column (First-Class Field)
```
Decision  : protocol VARCHAR(10) in events, signals, correlations, incidents
Values    : TCP, UDP, ICMP, ICMP6, ESP, GRE, OTHER
Status    : IMPLEMENTED Session 3
```

### AD-029 — log_format Column
```
Decision  : log_format VARCHAR(30) DEFAULT 'RFC3164'
Status    : IMPLEMENTED Session 3
```

### AD-030 — Parser Registry
```
Decision  : registry.get(log_format, device_type) → parser instance
            Phase 1: RFC3164 + RFC5424
Status    : IMPLEMENTED Session 4
```

### AD-031 — UDP Socket Buffer
```
Decision  : SO_RCVBUF = 32MB via Python socket.setsockopt() ONLY
            sysctls removed — not supported on Docker Desktop / WSL2
Status    : IMPLEMENTED Session 4
```

### AD-032 — orjson Mandatory
```
Decision  : orjson replaces json built-in in ALL services
Status    : IMPLEMENTED Session 4
```

### AD-033 — UTC Enforcement
```
Decision  : Every timestamp → UTC immediately in parser
Status    : IMPLEMENTED Session 4
```

### AD-034 — device_type Column
```
Decision  : device_type VARCHAR(30) DEFAULT 'GENERIC'
Status    : IMPLEMENTED Session 3 + Session 4
```

### AD-035 — Docker Build Context Strategy
```
Decision  : context: . (root) for all services using shared/
            dockerfile: services/<n>/Dockerfile
            COPY services/<n>/requirements.txt ./requirements.txt
Status    : IMPLEMENTED Session 4
```

### AD-036 — .env Password URL Encoding
```
Decision  : All special chars in DSN passwords URL-encoded
            @ → %40 | # → %23 | $ → %24 | : → %3A
            Dollar signs NOT escaped as $$ (that was compose-level only)
Status    : IMPLEMENTED Session 4
```

### AD-037 — Docker Pull Policy
```
Decision  : pull_policy: never on all locally-built services
Status    : IMPLEMENTED Session 4
```

### AD-038 — ON CONFLICT Removed from Partitioned Table INSERT
```
Decision  : repository.py uses plain INSERT without ON CONFLICT
Rationale : PostgreSQL partitioned tables do not support ON CONFLICT
            without a matching unique constraint on partition key
Status    : IMPLEMENTED Session 4
```

### AD-039 — Kafka Topics Created via CLI (Confluent Image)
```
Decision  : Topics created via /usr/bin/kafka-topics (Confluent cp-kafka:7.6.0)
            Not via bitnami path or create_topics.sh
            Command: docker exec siem-kafka /usr/bin/kafka-topics --create ...
Status    : IMPLEMENTED Session 4
```

### AD-040 — GRANT CONNECT on siem_db
```
Decision  : All 8 service roles need explicit GRANT CONNECT ON DATABASE
            Added to 03_seed.sql permanently
Status    : IMPLEMENTED Session 4
```

---

## SYSTEM ARCHITECTURE

### Data Flow
```
Network Devices (TCP/UDP :514)
  → syslog-receiver              ✅ RUNNING
  → [raw-events]                 ✅ 3 partitions
  → normalizer-worker            ✅ RUNNING + DB writes confirmed
  → [normalized-events]          ✅ 3 partitions
  → detection-engine             ⬜ Phase 2
  → [signals]                    ⬜ Phase 2
  → correlation-engine           ⬜ Phase 2
  → [incidents]                  ⬜ Phase 2
  → incident-service             ⬜ Phase 2
  → [incidents-updated]          ⬜ Phase 2
  → soar + ai + dashboard        ⬜ Phase 3/4
```

---

## MICROSERVICES

| Service            | Phase | Status   | Port  | Produces          | Consumes          | DB Schema        |
|--------------------|-------|----------|-------|-------------------|-------------------|------------------|
| syslog-receiver    | 1     | RUNNING  | 514   | raw-events        | —                 | none             |
| normalizer-worker  | 1     | RUNNING  | —     | normalized-events | raw-events        | siem_ingest      |
| detection-engine   | 2     | PENDING  | —     | signals           | normalized-events | siem_detection   |
| correlation-engine | 2     | PENDING  | —     | incidents         | signals           | siem_correlation |
| incident-service   | 2     | PENDING  | int.  | incidents-updated | incidents         | siem_incidents   |
| soar-service       | 3     | PENDING  | int.  | actions           | incidents-updated | siem_soar        |
| ai-service         | 3     | PENDING  | —     | —                 | incidents-updated | siem_ai          |
| api-gateway        | 4     | PENDING  | 8000  | —                 | —                 | siem_auth        |
| dashboard-backend  | 4     | PENDING  | 8001  | —                 | incidents-updated | read-only        |

---

## KAFKA TOPICS (all created and verified)

| Topic             | Parts | Retention | Status |
|-------------------|-------|-----------|--------|
| raw-events        | 3     | 24h       | ✅     |
| normalized-events | 3     | 24h       | ✅     |
| signals           | 3     | 72h       | ✅     |
| incidents         | 1     | 7d        | ✅     |
| incidents-updated | 1     | 7d        | ✅     |
| actions           | 1     | 7d        | ✅     |
| dlq-normalizer    | 1     | 7d        | ✅     |
| dlq-detection     | 1     | 7d        | ✅     |
| dlq-soar          | 1     | 7d        | ✅     |

---

## DATABASE SCHEMAS

| Schema           | Tables                              | Status |
|------------------|-------------------------------------|--------|
| siem_ingest      | events (PARTITIONED daily)          | ✅     |
| siem_detection   | signals (PARTITIONED daily), rules  | ✅     |
| siem_correlation | correlations                        | ✅     |
| siem_incidents   | incidents, incident_signals         | ✅     |
| siem_soar        | actions, audit_log                  | ✅     |
| siem_ai          | ai_reports                          | ✅     |
| siem_auth        | users, sessions                     | ✅     |

---

## ENVIRONMENT CONFIGURATION

```env
# E:\AI-SIEM-PLATFORM\.env
# Passwords are URL-encoded (AD-036): @ → %40 | # → %23 | $ → %24

POSTGRES_PASSWORD=HgbynbZM1kII44BAufX-fR_pSdoWB_oB84rUsI_junE
POSTGRES_DSN=postgresql://siem_user:HgbynbZM1kII44BAufX-fR_pSdoWB_oB84rUsI_junE@postgres:5432/siem_db

DB_DSN_INGEST=postgresql://role_ingest:Kx9%23mP2%24vL8nQ4%40wR7yT1uJ6eH3bG5cF@postgres:5432/siem_db
DB_DSN_DETECTION=postgresql://role_detection:Zt5%24kM8%40nB2vX9%23pW4qY7cL1dR6eN3mJ@postgres:5432/siem_db
DB_DSN_CORRELATION=postgresql://role_correlation:Hq3%40yN7%24bP5vK2%23xM8wT4fG1rC9eL6jQ@postgres:5432/siem_db
DB_DSN_INCIDENTS=postgresql://role_incidents:Wm6%23cR4%24zX1vN9%40kP7bG3yL8qT2eM5jH@postgres:5432/siem_db
DB_DSN_SOAR=postgresql://role_soar:Pf8%24nG5%40wK3vZ7%23mL2xQ4yR9bC1eT6jN@postgres:5432/siem_db
DB_DSN_AI=postgresql://role_ai:Bj2%40xT9%24hN6vM4%23cQ8pK1wR5eY7gL3zF@postgres:5432/siem_db
DB_DSN_AUTH=postgresql://role_auth:Yk4%23mZ8%24cL1vP6%40nR3xW9qT7eG2bJ5hN@postgres:5432/siem_db
DB_DSN_READONLY=postgresql://role_readonly:Rf7%24kQ2%40bN5vG9%23mX4wL8yC3eP1zT6jW@postgres:5432/siem_db

REDIS_PASSWORD=iwbSnEf0QdhSdOw0nfC4aL0lomI9m7sFoeeeiuRPCz8
REDIS_URL=redis://:iwbSnEf0QdhSdOw0nfC4aL0lomI9m7sFoeeeiuRPCz8@redis:6379/0
GRAFANA_PASSWORD=LywRdrVDZb0nK_0y9-IqL9cW0YkYeZu45VAjPZgzJfo
JWT_SECRET=7c7e912c43ea19ffbc2482de747c192cb658c028a5f877c2c31867c9e1a0089b
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
SYSLOG_PORT=514
REDIS_TI_TTL=86400
REDIS_WINDOW_TTL=600
GEOIP_DB_PATH=/app/data/GeoLite2-City.mmdb
TI_FEED_URL=https://otx.alienvault.com/api/v1/indicators/export
LLM_API_KEY=placeholder
LLM_MODEL=gpt-4o
```

---

## IMPLEMENTATION PROGRESS

```
[OK] Session 1 — Architecture, Scaffold, VSCode, .env
[OK] Session 2 — docker-compose.yml, Infrastructure running
[OK] Session 3 — DB + Kafka + Patches + Verification complete
[OK] Phase 1   — syslog-receiver + normalizer-worker RUNNING + DB writes confirmed
[ ] Phase 2   — detection-engine, correlation-engine, incident-service
[ ] Phase 3   — soar-service, ai-service
[ ] Phase 4   — api-gateway, dashboard-backend, observability
```

---

## SESSION LOG

### Session 1 — 2026-03-16
```
Decisions : AD-001 → AD-006
```

### Session 2 — 2026-03-17
```
Decisions : AD-007 → AD-018
```

### Session 3 — 2026-03-18
```
Decisions : AD-019 → AD-034
Completed : Full DB schema, Kafka topics, seed data, indexes
```

### Session 4 — 2026-03-19/20
```
Decisions : AD-035 → AD-040

Completed :
  shared/schemas/events.py          RawEvent + NormalizedEvent
  shared/schemas/signals.py         Signal skeleton
  shared/schemas/incidents.py       Incident skeleton
  shared/kafka/base_producer.py     retry + orjson
  shared/kafka/base_consumer.py     DLQ + manual commit

  syslog-receiver/src/              all files complete
  normalizer-worker/src/            all files complete
  parsers/rfc3164.py + rfc5424.py   complete

  docker-compose.yml:
    context: . (root)
    dockerfile: full path
    COPY paths fixed
    sysctls removed (WSL2 incompatible)
    pull_policy: never
    normalizer-worker block added

  DB fixes:
    log_format + protocol columns added via ALTER TABLE
    GRANT CONNECT ON DATABASE for all 8 roles
    ON CONFLICT removed from partitioned table INSERT

  Kafka:
    9 topics created via /usr/bin/kafka-topics (Confluent)

  .env:
    All DSN passwords URL-encoded (@ # $ : all encoded)

E2E Verified:
  UDP syslog → syslog-receiver → raw-events → normalizer-worker
  → event_normalized logged
  → siem_ingest.events row confirmed
  PIPELINE END-TO-END WORKING
```

---

## KNOWN ISSUES (Start of Session 5)

```
ISSUE-001 — ROOT duplicate files in syslog-receiver/ (cosmetic)
  Files   : device_registry.py, main.py, metrics.py, producer.py, validator.py
  Action  : Delete after confirming Phase 2 builds clean
  Risk    : None — src/ is authoritative

ISSUE-002 — syslog-receiver Prometheus port mismatch
  docker-compose has METRICS_PORT: 9100
  src/config.py default is 9101
  Action  : Verify config.py reads METRICS_PORT from env correctly
```

---

## SESSION 5 CHECKLIST

```
STEP 1 — Verify E2E pipeline still working
  Send UDP syslog → check siem_ingest.events

STEP 2 — Fix syslog-receiver Prometheus port (ISSUE-002)

STEP 3 — Clean ROOT duplicate files (ISSUE-001)

STEP 4 — Begin Phase 2: detection-engine
  Files to write:
    services/detection-engine/src/config.py
    services/detection-engine/src/main.py
    services/detection-engine/src/core/engine.py
    services/detection-engine/src/core/rules/*.py  (7 rules)
    services/detection-engine/src/db/repository.py
    services/detection-engine/src/kafka/consumer.py
    services/detection-engine/src/kafka/producer.py

STEP 5 — Begin Phase 2: correlation-engine

STEP 6 — Begin Phase 2: incident-service

STEP 7 — Update PROJECT_MEMORY.md
```

---

## PROJECT STANDARDS

```
1.  Zero-Trust: unique password per service, per-service DB DSN
2.  Network: siem-network 172.28.0.0/16
3.  Kafka flow strictly enforced
4.  JWT: 256-bit, Access 15min, Refresh 7d
5.  Pydantic strict → DLQ on failure, never silent drop
6.  Prometheus + healthcheck in every service from day one
7.  orjson mandatory — no json built-in
8.  UTC everywhere, local TZ only in dashboard
9.  protocol is first-class field
10. device_type + log_format = exact parser, GENERIC fallback
11. Docker context: . (root), dockerfile: full path
12. .env passwords: URL-encoded (@ # $ : all percent-encoded)
13. pull_policy: never on local images
14. No ON CONFLICT on partitioned tables
15. Kafka topics: /usr/bin/kafka-topics (Confluent cp-kafka:7.6.0)
16. GRANT CONNECT ON DATABASE required for all service roles
```

---

## HOW TO USE THIS FILE

```
1. Paste entire file at start of every new session
2. At end of session update:
   - IMPLEMENTATION PROGRESS
   - SESSION LOG
   - KNOWN ISSUES
   - NEXT SESSION CHECKLIST
3. Save to E:\AI-SIEM-PLATFORM\PROJECT_MEMORY.md
```
