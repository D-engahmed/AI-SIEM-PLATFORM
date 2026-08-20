"""
Throughput/latency stress test for POST /score against a LIVE running
instance of correlation-ml-service. Different kind of "stress" from
tests/test_model_stress.py (which checks the model doesn't crash on weird
inputs, in-process, as part of the normal pytest suite) -- this measures
whether the service holds up under concurrent load, over the network,
which needs a real running process and shouldn't be part of the normal
fast test suite.

USAGE:
    # In one terminal: start the service (needs a real or fake Kafka
    # broker reachable, or just tolerate the consumer failing to connect --
    # POST /score doesn't depend on it)
    python src/main.py

    # In another:
    python stress/run_load_test.py --url http://localhost:9100 \\
        --concurrency 50 --duration 30

Reports p50/p95/p99 latency and throughput, and flags a FAIL if p95
latency exceeds --latency-budget-ms -- see config.py's api_latency_warn_ms
(default 50.0) for the baseline this is meant to be checked against; pass
the same number here to actually enforce it rather than just eyeball it.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field

import httpx


def _random_payload() -> dict:
    # Deliberately varied, not one fixed payload repeated -- a service
    # that's fast on one cached-friendly input but slow on realistic
    # variety would give a misleadingly rosy number otherwise.
    rng = random.Random()
    return {
        "correlation_id": str(uuid.uuid4()),
        "target_kind": "user",
        "target_value": f"user_{rng.randint(1, 5000)}",
        "graph_features": {
            "fan_in_count": rng.choice([None, rng.randint(1, 40)]),
            "epoch_age_seconds": rng.choice([None, rng.uniform(1, 86400)]),
        },
        "signal_context": {
            "triggering_source_ip": f"203.0.113.{rng.randint(1, 254)}",
            "ti_matched": rng.choice([None, True, False]),
            "signal_ids": [str(uuid.uuid4()) for _ in range(rng.randint(1, 3))],
        },
    }


@dataclass
class Results:
    latencies_ms: list = field(default_factory=list)
    errors: int = 0
    status_counts: dict = field(default_factory=dict)


async def _worker(client: httpx.AsyncClient, url: str, stop_at: float, results: Results, lock: asyncio.Lock) -> None:
    while time.monotonic() < stop_at:
        start = time.perf_counter()
        try:
            resp = await client.post(url, json=_random_payload(), timeout=10.0)
            elapsed_ms = (time.perf_counter() - start) * 1000
            async with lock:
                results.latencies_ms.append(elapsed_ms)
                results.status_counts[resp.status_code] = results.status_counts.get(resp.status_code, 0) + 1
        except Exception:
            async with lock:
                results.errors += 1


async def run(base_url: str, concurrency: int, duration_s: float) -> Results:
    url = base_url.rstrip("/") + "/score"
    results = Results()
    lock = asyncio.Lock()
    stop_at = time.monotonic() + duration_s

    async with httpx.AsyncClient() as client:
        workers = [asyncio.create_task(_worker(client, url, stop_at, results, lock)) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    return results


def _percentile(data: list, p: float) -> float:
    if not data:
        return float("nan")
    data = sorted(data)
    k = (len(data) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:9100")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--duration", type=float, default=15.0, help="seconds")
    parser.add_argument("--latency-budget-ms", type=float, default=50.0,
                         help="p95 above this fails (see config.py's api_latency_warn_ms)")
    args = parser.parse_args()

    print(f"Running {args.concurrency} concurrent workers against {args.url}/score for {args.duration}s ...")
    results = asyncio.run(run(args.url, args.concurrency, args.duration))

    n = len(results.latencies_ms)
    if n == 0:
        print("No successful requests completed -- is the service running and reachable?")
        return 1

    throughput = n / args.duration
    p50 = _percentile(results.latencies_ms, 50)
    p95 = _percentile(results.latencies_ms, 95)
    p99 = _percentile(results.latencies_ms, 99)

    print(f"\nRequests completed: {n}   Errors (connection/timeout): {results.errors}")
    print(f"Status codes: {results.status_counts}")
    print(f"Throughput: {throughput:.1f} req/s")
    print(f"Latency (ms): p50={p50:.2f}  p95={p95:.2f}  p99={p99:.2f}  max={max(results.latencies_ms):.2f}")

    non_200 = sum(v for k, v in results.status_counts.items() if k != 200)
    if non_200:
        print(f"\n*** {non_200} non-200 responses -- treat this run's latency numbers with suspicion. ***")

    if p95 > args.latency_budget_ms:
        print(f"\nFAIL: p95 latency {p95:.2f}ms exceeds --latency-budget-ms {args.latency_budget_ms}")
        return 1
    print(f"\nPASS: p95 latency within {args.latency_budget_ms}ms budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
