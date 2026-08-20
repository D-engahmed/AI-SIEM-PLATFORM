"""
Local stand-in for the platform's `shared.kafka` package.

WHY THIS EXISTS (read before touching anything in this package):
--------------------------------------------------------------
As of 2026-08-17, `correlation-ml-service` does not have access to the
real platform `shared.kafka` package -- it isn't in this repo, isn't
pip-installable from anywhere this container can reach, and wasn't
provided as source. The reviewer's spec ("Inherit from
`shared.kafka.base_consumer.BaseConsumer` and
`shared.kafka.base_producer.BaseProducer`") assumes it exists and is
importable.

Decision made 2026-08-17: rather than block on that, this package is a
**local reimplementation** matching the method signatures the reviewer
gave us (`process_message(self, payload: dict) -> None`,
`route_to_dlq(payload, error)`), built on `aiokafka` (per issue-008:
aiokafka 0.11.0 is the platform-wide standard) and `orjson` (per
AD-032).

This is a GUESS at the real interface's behavior (retry policy, exact
DLQ envelope shape, commit semantics), not a verified copy of it. If a
platform-wide DLQ replay/monitoring tool expects a specific envelope
shape, this package's envelope may not match it -- that risk was raised
explicitly before this was built and accepted as a tradeoff, not missed.

It deliberately lives at the import path `shared.kafka.*` so that
swapping in the real package later means deleting this directory and
installing the real one -- `correlation-ml-service`'s own code
(`ml_consumer.py`) only depends on the `BaseConsumer`/`BaseProducer`
contract below, not on anything inside it.
"""
