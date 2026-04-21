"""
celery_app.py
══════════════════════════════════════════════════════════════════════
Celery application factory — single source of truth for the Celery
instance used by both the Flask app and the worker process.

Import this module wherever you need the Celery instance:
    from celery_app import celery

Environment variables
─────────────────────
    CELERY_BROKER_URL    — default: redis://localhost:6379/0
    CELERY_RESULT_BACKEND — default: redis://localhost:6379/1
                            (separate DB from broker for clarity)
    CELERY_TASK_SOFT_TIME_LIMIT — seconds before SoftTimeLimitExceeded (default 600)
    CELERY_TASK_TIME_LIMIT      — hard kill after this many seconds (default 660)

Architecture
────────────
    Flask route  →  .delay() / .apply_async()  →  Redis broker
                                                        ↓
                                               Celery worker picks up task
                                                        ↓
                                          Runs training, writes result to Postgres
                                                        ↓
                                       Result also stored in Redis result backend
                                                        ↓
    Client polls  GET /api/jobs/<task_id>  ←  Flask reads Redis result backend
"""

import os
from celery import Celery

BROKER_URL     = os.getenv("CELERY_BROKER_URL",     "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "600"))   # 10 min
HARD_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT",      "660"))   # 11 min

celery = Celery(
    "training_models",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

celery.conf.update(
    # ── Serialisation ──────────────────────────────────────────────────────
    task_serializer          = "json",
    result_serializer        = "json",
    accept_content           = ["json"],

    # ── Time limits ────────────────────────────────────────────────────────
    task_soft_time_limit     = SOFT_TIME_LIMIT,
    task_time_limit          = HARD_TIME_LIMIT,

    # ── Result expiry ──────────────────────────────────────────────────────
    result_expires           = 86400,    # keep results in Redis for 24 hours

    # ── Worker behaviour ───────────────────────────────────────────────────
    worker_prefetch_multiplier  = 1,     # one task at a time per worker slot
    task_acks_late              = True,  # ack only after task finishes (safe re-queue on crash)
    task_reject_on_worker_lost  = True,

    # ── Routing ────────────────────────────────────────────────────────────
    # All training tasks go to the "training" queue.
    # Future queues (e.g. "privacy", "aggregation") can be added here.
    task_routes = {
        "tasks.training_tasks.run_central_training":   {"queue": "training"},
        "tasks.training_tasks.run_federated_training": {"queue": "training"},
    },

    # ── Beat schedule (placeholder for Phase 3 periodic jobs) ─────────────
    # beat_schedule = {
    #     "model-drift-check": {
    #         "task": "tasks.monitoring_tasks.check_model_drift",
    #         "schedule": crontab(hour="*/6"),
    #     },
    # },
)
