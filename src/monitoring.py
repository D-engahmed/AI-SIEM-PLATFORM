"""
Real-life model monitoring for correlation-ml-service: Population
Stability Index (PSI) feature drift detection, computed over a bounded
in-memory ring buffer of recently scored feature vectors, against a
fixed reference distribution captured at training time.

Design constraints this file is answerable to:
  - Memory bounds: the ring buffer is capped at
    config.monitoring_ring_buffer_size (default 5000) per feature -- same
    discipline as everywhere else in this service (see ml_scorer.py's
    "Memory bounds" note). This is observability state, not application
    state, but it still must not grow unbounded.
  - Statelessness of SCORING: this module tracks aggregate feature
    distributions, never anything keyed by target_value/source_ip/
    correlation_id. It cannot answer "is THIS entity suspicious" -- only
    "does recent traffic AS A WHOLE look like training data". Do not
    repurpose it into a per-entity cache.
  - No new runtime dependency beyond numpy (already required for
    ml_scorer.py) and prometheus_client (already required for metrics.py).
    Deliberately NOT pandas -- the reference distribution is loaded from a
    small .npz, not re-derived from the training CSV at service startup.

WHY PSI AND NOT SOMETHING FANCIER (e.g. KS-test, Evidently, WhyLabs):
PSI is the industry-standard metric for exactly this ("has feature X's
distribution shifted since training") and needs nothing beyond numpy --
appropriate for a service whose whole design philosophy is "no heavy
runtime dependencies, no external state store" (see ml_scorer.py). A
heavier drift-monitoring platform is a legitimate future upgrade, not
something to default to.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from prometheus_client import CollectorRegistry, Gauge

logger = logging.getLogger("correlation_ml.monitoring")

DEFAULT_PSI_BINS = 10
# Below this many recent samples, PSI is noise, not signal -- a ring buffer
# that's 3% full comparing against a 10-bin histogram will report enormous
# PSI values from sampling variance alone, not real drift. Suppressed
# rather than reported as 0.0 (which would look like "confirmed no drift"
# instead of "not enough data to say").
MIN_SAMPLES_FOR_PSI = 30


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = DEFAULT_PSI_BINS) -> float:
    """
    PSI for one feature. Bin edges come from the REFERENCE distribution's
    quantiles, not current's -- PSI measures how far `current` has drifted
    from `reference`, so the bins must be fixed by the baseline. Standard
    formula: sum over bins of (actual% - expected%) * ln(actual% / expected%).
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # Degenerate reference (near-constant feature, e.g. a flag that's
        # almost always 0) -- quantile binning can't say anything useful.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Floor at a small epsilon, not zero -- an empty bin would otherwise
    # produce log(0) or divide-by-zero and blow up the whole PSI value from
    # one sparse bin, which is exactly the kind of noise MIN_SAMPLES_FOR_PSI
    # already guards against at the caller level; this is defense in depth.
    ref_pct = np.maximum(ref_counts / len(reference), 1e-6)
    cur_pct = np.maximum(cur_counts / len(current), 1e-6)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class FeatureDriftMonitor:
    """
    Owns the reference distribution (loaded once at startup) and the
    per-feature ring buffers (mutated on every scored message via
    .record()). PSI is computed on demand (.compute_all()), not on every
    message -- cheap enough to run at /metrics scrape time (a few numpy
    histogram calls over a few thousand floats), no reason to do it more
    often than something actually reads it.
    """

    def __init__(
        self,
        reference: dict[str, np.ndarray],
        ring_buffer_size: int,
        psi_warn_threshold: float,
        psi_alert_threshold: float,
        registry: CollectorRegistry,
    ) -> None:
        self._reference = reference
        self._ring_buffer_size = ring_buffer_size
        self._psi_warn_threshold = psi_warn_threshold
        self._psi_alert_threshold = psi_alert_threshold
        self._buffers: dict[str, deque] = {
            name: deque(maxlen=ring_buffer_size) for name in reference
        }
        self._psi_gauge = Gauge(
            "cml_feature_psi",
            "Population Stability Index of each scored feature vs. the training reference distribution",
            labelnames=["feature"],
            registry=registry,
        )
        self._buffer_fill_gauge = Gauge(
            "cml_monitoring_buffer_fill_ratio",
            "Fraction of the monitoring ring buffer currently filled (0-1); PSI is suppressed below "
            f"{MIN_SAMPLES_FOR_PSI} samples regardless of this ratio.",
            registry=registry,
        )

    @classmethod
    def load(
        cls,
        reference_path: str,
        ring_buffer_size: int,
        psi_warn_threshold: float,
        psi_alert_threshold: float,
        registry: CollectorRegistry,
    ) -> Optional["FeatureDriftMonitor"]:
        """
        Returns None (not an exception) if the reference artifact doesn't
        exist -- unlike the model artifact, monitoring is not fail-fast:
        a service that can score but can't detect its own drift is
        degraded, not broken. main.py logs this loudly and continues.
        """
        path = Path(reference_path)
        if not path.exists():
            logger.warning(
                "No feature reference distribution at %s -- drift monitoring disabled. "
                "Run training/train_model.py (or the model_selection notebook) to generate it.",
                path,
            )
            return None
        npz = np.load(path)
        reference = {name: npz[name] for name in npz.files}
        logger.info(
            "Loaded feature drift reference distribution from %s (%d features, %d samples/feature)",
            path, len(reference), len(next(iter(reference.values()))) if reference else 0,
        )
        return cls(reference, ring_buffer_size, psi_warn_threshold, psi_alert_threshold, registry)

    def record(self, feature_vector: dict[str, float]) -> None:
        for name, value in feature_vector.items():
            if name in self._buffers:
                self._buffers[name].append(value)

    def compute_all(self) -> dict[str, dict]:
        """Returns {feature_name: {"psi": float, "status": str, "n_samples": int}}."""
        result = {}
        sizes = [len(buf) for buf in self._buffers.values()]
        self._buffer_fill_gauge.set(
            (min(sizes) / self._ring_buffer_size) if sizes else 0.0
        )
        for name, buf in self._buffers.items():
            n = len(buf)
            if n < MIN_SAMPLES_FOR_PSI:
                psi = 0.0
                status = "INSUFFICIENT_DATA"
            else:
                psi = compute_psi(self._reference[name], np.array(buf))
                status = self._status_for(psi)
            self._psi_gauge.labels(feature=name).set(psi)
            result[name] = {"psi": round(psi, 4), "status": status, "n_samples": n}
        return result

    def _status_for(self, psi: float) -> str:
        if psi >= self._psi_alert_threshold:
            return "ALERT"
        if psi >= self._psi_warn_threshold:
            return "WARN"
        return "OK"
