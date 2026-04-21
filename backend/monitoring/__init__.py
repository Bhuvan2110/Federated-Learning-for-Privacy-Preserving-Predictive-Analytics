"""
monitoring/__init__.py
Prometheus metrics package  (Day 7)
"""
from monitoring.metrics import (
    # FL metrics
    FL_ROUNDS_COMPLETED, FL_ROUND_LOSS, FL_GLOBAL_ACCURACY,
    FL_CLIENTS_PER_ROUND, FL_TRAINING_DURATION,
    FL_EXPERIMENTS_STARTED, FL_EXPERIMENTS_FAILED,
    FL_DP_EPSILON_SPENT, FL_SECAGG_OVERHEAD, FL_SECAGG_VERIFIED,
    # Upload / encryption
    UPLOAD_TOTAL, DECRYPT_FAILURES, KEY_FETCH_LATENCY,
    # HTTP
    HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION, HTTP_IN_FLIGHT,
    # Celery
    CELERY_TASKS_TOTAL, CELERY_TASK_DURATION, CELERY_QUEUE_DEPTH,
    # gRPC
    GRPC_REQUESTS_TOTAL, GRPC_REQUEST_DURATION,
    # Helpers
    record_training_complete, record_experiment_started,
    record_experiment_failed, record_upload, record_decrypt_failure,
    record_grpc_call, init_flask_metrics,
    generate_metrics_output, metrics_health,
    PROMETHEUS_ENABLED, _PROM_AVAILABLE,
)

__all__ = [
    "FL_ROUNDS_COMPLETED", "FL_ROUND_LOSS", "FL_GLOBAL_ACCURACY",
    "FL_CLIENTS_PER_ROUND", "FL_TRAINING_DURATION",
    "FL_EXPERIMENTS_STARTED", "FL_EXPERIMENTS_FAILED",
    "FL_DP_EPSILON_SPENT", "FL_SECAGG_OVERHEAD", "FL_SECAGG_VERIFIED",
    "UPLOAD_TOTAL", "DECRYPT_FAILURES", "KEY_FETCH_LATENCY",
    "HTTP_REQUESTS_TOTAL", "HTTP_REQUEST_DURATION", "HTTP_IN_FLIGHT",
    "CELERY_TASKS_TOTAL", "CELERY_TASK_DURATION", "CELERY_QUEUE_DEPTH",
    "GRPC_REQUESTS_TOTAL", "GRPC_REQUEST_DURATION",
    "record_training_complete", "record_experiment_started",
    "record_experiment_failed", "record_upload", "record_decrypt_failure",
    "record_grpc_call", "init_flask_metrics",
    "generate_metrics_output", "metrics_health",
    "PROMETHEUS_ENABLED", "_PROM_AVAILABLE",
]
