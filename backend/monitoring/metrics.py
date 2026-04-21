"""
monitoring/metrics.py
══════════════════════════════════════════════════════════════════════
Prometheus metrics for Training Models  (Day 7)

All metric objects live here as module-level singletons.
Import and increment from anywhere:

    from monitoring.metrics import (
        FL_ROUNDS_TOTAL, FL_ROUND_LOSS, REQUEST_LATENCY,
        record_training_complete,
    )

Metric categories
─────────────────
  FL metrics
    fl_rounds_completed_total      Counter   — rounds finished, by model_type
    fl_round_loss                  Gauge     — latest round loss, by experiment
    fl_global_accuracy             Gauge     — latest global accuracy, by experiment
    fl_clients_per_round           Histogram — clients that submitted per round
    fl_training_duration_seconds   Histogram — wall time per training job
    fl_experiments_total           Counter   — total experiments started, by type
    fl_experiments_failed_total    Counter   — failed experiments, by type
    fl_dp_epsilon_spent            Gauge     — current epsilon spend, by experiment
    fl_secagg_overhead_ms          Histogram — SecAgg masking overhead per round
    fl_secagg_verified_total       Counter   — rounds where mask cancellation verified

  Encryption / upload metrics
    upload_total                   Counter   — uploads, by encrypted(true/false)
    decrypt_failures_total         Counter   — AES-GCM auth tag failures
    key_fetch_latency_seconds      Histogram — time to serve /api/pubkey

  HTTP request metrics
    http_requests_total            Counter   — all requests, by method+endpoint+status
    http_request_duration_seconds  Histogram — latency, by method+endpoint
    http_requests_in_flight        Gauge     — concurrent requests

  Celery / queue metrics
    celery_tasks_total             Counter   — tasks dispatched, by task_name+status
    celery_task_duration_seconds   Histogram — task wall time, by task_name
    celery_queue_depth             Gauge     — pending tasks (polled externally)

  gRPC metrics
    grpc_requests_total            Counter   — gRPC calls, by method+status
    grpc_request_duration_seconds  Histogram — gRPC latency, by method

Environment variables
─────────────────────
  PROMETHEUS_ENABLED   — set to "false" to use no-op stubs (default: true)
  PROMETHEUS_MULTIPROC_DIR — directory for multiprocess mode (gunicorn)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"

# ── Optional prometheus_client import ────────────────────────────────────
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, CollectorRegistry,
        CONTENT_TYPE_LATEST, generate_latest,
        multiprocess, REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed — metrics disabled.\n"
        "Install with: pip install prometheus-client"
    )


# ════════════════════════════════════════════════════════════════════════════
#  No-op stubs (used when prometheus_client is absent or PROMETHEUS_ENABLED=false)
# ════════════════════════════════════════════════════════════════════════════

class _NoOpMetric:
    """Drop-in replacement for any Prometheus metric when prom is unavailable."""
    def __init__(self, *a, **kw): pass
    def inc(self, *a, **kw):    pass
    def dec(self, *a, **kw):    pass
    def set(self, *a, **kw):    pass
    def observe(self, *a, **kw): pass
    def labels(self, **kw):     return self
    def time(self):
        @contextmanager
        def _ctx():
            yield
        return _ctx()


def _metric(cls, *args, **kwargs):
    """Create a metric or a no-op stub depending on availability."""
    if _PROM_AVAILABLE and PROMETHEUS_ENABLED:
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            logger.warning("Could not create metric %s: %s", args[0] if args else "?", e)
            return _NoOpMetric()
    return _NoOpMetric()


# ════════════════════════════════════════════════════════════════════════════
#  FL metrics
# ════════════════════════════════════════════════════════════════════════════

FL_ROUNDS_COMPLETED = _metric(
    Counter,
    "fl_rounds_completed_total",
    "Total FL rounds completed",
    ["model_type"],
)

FL_ROUND_LOSS = _metric(
    Gauge,
    "fl_round_loss",
    "Latest round loss for an experiment",
    ["experiment_id", "model_type"],
)

FL_GLOBAL_ACCURACY = _metric(
    Gauge,
    "fl_global_accuracy",
    "Latest global model accuracy for an experiment",
    ["experiment_id", "model_type"],
)

FL_CLIENTS_PER_ROUND = _metric(
    Histogram,
    "fl_clients_per_round",
    "Number of clients that submitted updates per round",
    ["model_type"],
    buckets=[1, 2, 3, 5, 10, 20, 50],
)

FL_TRAINING_DURATION = _metric(
    Histogram,
    "fl_training_duration_seconds",
    "Wall-clock time for a complete training job",
    ["model_type"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

FL_EXPERIMENTS_STARTED = _metric(
    Counter,
    "fl_experiments_total",
    "Total experiments created",
    ["model_type"],
)

FL_EXPERIMENTS_FAILED = _metric(
    Counter,
    "fl_experiments_failed_total",
    "Total experiments that failed",
    ["model_type"],
)

FL_DP_EPSILON_SPENT = _metric(
    Gauge,
    "fl_dp_epsilon_spent",
    "Current epsilon spend for a DP experiment",
    ["experiment_id"],
)

FL_SECAGG_OVERHEAD = _metric(
    Histogram,
    "fl_secagg_overhead_ms",
    "SecAgg masking/unmasking overhead per round (milliseconds)",
    ["experiment_id"],
    buckets=[0.1, 0.5, 1, 5, 10, 50, 100, 500],
)

FL_SECAGG_VERIFIED = _metric(
    Counter,
    "fl_secagg_verified_total",
    "SecAgg rounds where mask cancellation was successfully verified",
    ["experiment_id"],
)


# ════════════════════════════════════════════════════════════════════════════
#  Upload / encryption metrics
# ════════════════════════════════════════════════════════════════════════════

UPLOAD_TOTAL = _metric(
    Counter,
    "upload_total",
    "Total CSV uploads received",
    ["encrypted"],   # "true" or "false"
)

DECRYPT_FAILURES = _metric(
    Counter,
    "decrypt_failures_total",
    "AES-GCM decryption / auth-tag verification failures",
    [],
)

KEY_FETCH_LATENCY = _metric(
    Histogram,
    "key_fetch_latency_seconds",
    "Latency to serve the RSA public key (/api/pubkey)",
    [],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)


# ════════════════════════════════════════════════════════════════════════════
#  HTTP request metrics
# ════════════════════════════════════════════════════════════════════════════

HTTP_REQUESTS_TOTAL = _metric(
    Counter,
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

HTTP_REQUEST_DURATION = _metric(
    Histogram,
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

HTTP_IN_FLIGHT = _metric(
    Gauge,
    "http_requests_in_flight",
    "Number of HTTP requests currently being processed",
    [],
)


# ════════════════════════════════════════════════════════════════════════════
#  Celery / queue metrics
# ════════════════════════════════════════════════════════════════════════════

CELERY_TASKS_TOTAL = _metric(
    Counter,
    "celery_tasks_total",
    "Total Celery tasks dispatched",
    ["task_name", "status"],   # status: started | success | failure
)

CELERY_TASK_DURATION = _metric(
    Histogram,
    "celery_task_duration_seconds",
    "Celery task execution time",
    ["task_name"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

CELERY_QUEUE_DEPTH = _metric(
    Gauge,
    "celery_queue_depth",
    "Approximate number of pending tasks in the training queue",
    [],
)


# ════════════════════════════════════════════════════════════════════════════
#  gRPC metrics
# ════════════════════════════════════════════════════════════════════════════

GRPC_REQUESTS_TOTAL = _metric(
    Counter,
    "grpc_requests_total",
    "Total gRPC calls",
    ["method", "status"],   # status: ok | error
)

GRPC_REQUEST_DURATION = _metric(
    Histogram,
    "grpc_request_duration_seconds",
    "gRPC call latency",
    ["method"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)


# ════════════════════════════════════════════════════════════════════════════
#  Convenience recording functions
# ════════════════════════════════════════════════════════════════════════════

def record_training_complete(result_payload: dict, model_type: str,
                             exp_id: str, duration_s: float) -> None:
    """
    Record all FL metrics for a completed training run.
    Call this from mark_completed() alongside MLflow tracking.
    """
    try:
        loss_history = result_payload.get("lossHistory", [])
        tm           = result_payload.get("trainMetrics", {})
        te           = result_payload.get("testMetrics",  {})
        n_rounds     = len(loss_history)

        # Round-level counters
        FL_ROUNDS_COMPLETED.labels(model_type=model_type).inc(n_rounds)

        # Per-experiment gauges
        if loss_history:
            FL_ROUND_LOSS.labels(
                experiment_id=exp_id, model_type=model_type
            ).set(loss_history[-1])

        if te.get("accuracy") is not None:
            FL_GLOBAL_ACCURACY.labels(
                experiment_id=exp_id, model_type=model_type
            ).set(te["accuracy"])

        # Training duration
        FL_TRAINING_DURATION.labels(model_type=model_type).observe(duration_s)

        # DP epsilon
        privacy = result_payload.get("privacy") or {}
        if privacy.get("current_epsilon") is not None:
            FL_DP_EPSILON_SPENT.labels(experiment_id=exp_id).set(
                privacy["current_epsilon"]
            )

        # SecAgg overhead
        secagg = result_payload.get("secagg") or {}
        if secagg:
            per_round = secagg.get("per_round_log", [])
            for rnd in per_round:
                overhead = rnd.get("overhead_ms", 0)
                FL_SECAGG_OVERHEAD.labels(experiment_id=exp_id).observe(overhead)
                if rnd.get("verified"):
                    FL_SECAGG_VERIFIED.labels(experiment_id=exp_id).inc()

        # Client count (from numClients or per-round survivor counts)
        n_clients = result_payload.get("numClients")
        if n_clients:
            FL_CLIENTS_PER_ROUND.labels(model_type=model_type).observe(n_clients)

    except Exception as exc:
        logger.warning("record_training_complete failed (non-fatal): %s", exc)


def record_experiment_started(model_type: str) -> None:
    """Increment the experiment-started counter."""
    try:
        FL_EXPERIMENTS_STARTED.labels(model_type=model_type).inc()
    except Exception as exc:
        logger.warning("record_experiment_started failed: %s", exc)


def record_experiment_failed(model_type: str) -> None:
    """Increment the experiment-failed counter."""
    try:
        FL_EXPERIMENTS_FAILED.labels(model_type=model_type).inc()
    except Exception as exc:
        logger.warning("record_experiment_failed failed: %s", exc)


def record_upload(encrypted: bool) -> None:
    """Increment the upload counter."""
    try:
        UPLOAD_TOTAL.labels(encrypted=str(encrypted).lower()).inc()
    except Exception as exc:
        logger.warning("record_upload failed: %s", exc)


def record_decrypt_failure() -> None:
    """Increment the decryption failure counter."""
    try:
        DECRYPT_FAILURES.inc()
    except Exception as exc:
        logger.warning("record_decrypt_failure failed: %s", exc)


def record_grpc_call(method: str, status: str, duration_s: float) -> None:
    """Record a completed gRPC call."""
    try:
        GRPC_REQUESTS_TOTAL.labels(method=method, status=status).inc()
        GRPC_REQUEST_DURATION.labels(method=method).observe(duration_s)
    except Exception as exc:
        logger.warning("record_grpc_call failed: %s", exc)


# ════════════════════════════════════════════════════════════════════════════
#  Flask middleware
# ════════════════════════════════════════════════════════════════════════════

def init_flask_metrics(app) -> None:
    """
    Register before/after request hooks on a Flask app to record
    HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION, and HTTP_IN_FLIGHT.

    Call once at app startup:
        from monitoring.metrics import init_flask_metrics
        init_flask_metrics(app)
    """
    import time as _time
    from flask import request, g

    @app.before_request
    def _before():
        g._prom_start = _time.time()
        HTTP_IN_FLIGHT.inc()

    @app.after_request
    def _after(response):
        duration = _time.time() - getattr(g, "_prom_start", _time.time())
        endpoint = request.endpoint or "unknown"
        HTTP_REQUESTS_TOTAL.labels(
            method   = request.method,
            endpoint = endpoint,
            status   = str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method   = request.method,
            endpoint = endpoint,
        ).observe(duration)
        HTTP_IN_FLIGHT.dec()
        return response

    logger.info("Prometheus Flask middleware initialised")


# ════════════════════════════════════════════════════════════════════════════
#  Metrics exposition
# ════════════════════════════════════════════════════════════════════════════

def generate_metrics_output() -> tuple[bytes, str]:
    """
    Generate Prometheus text exposition format.
    Returns (content_bytes, content_type_str).

    Handles multiprocess mode (multiple gunicorn workers) automatically
    when PROMETHEUS_MULTIPROC_DIR is set.
    """
    if not (_PROM_AVAILABLE and PROMETHEUS_ENABLED):
        return b"# Prometheus metrics disabled\n", "text/plain; version=0.0.4"

    mp_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if mp_dir and os.path.isdir(mp_dir):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest(REGISTRY)

    return data, CONTENT_TYPE_LATEST


def metrics_health() -> dict:
    """Return Prometheus status for /api/health."""
    if not _PROM_AVAILABLE:
        return {"prometheus": "unavailable", "reason": "package not installed"}
    if not PROMETHEUS_ENABLED:
        return {"prometheus": "disabled"}
    return {"prometheus": "ok", "scrape_endpoint": "/metrics"}
