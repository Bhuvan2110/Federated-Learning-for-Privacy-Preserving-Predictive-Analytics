"""
tasks/training_tasks.py
══════════════════════════════════════════════════════════════════════
Celery tasks for async training.

Each task:
  1. Transitions experiment status: pending → running
  2. Runs the training (CPU-bound, can take seconds to minutes)
  3. Persists result and transitions: running → completed / failed
  4. Returns a result dict that Celery stores in Redis
     (client polls GET /api/jobs/<task_id>)

State machine
─────────────
    PENDING  →  RUNNING  →  COMPLETED
                         →  FAILED   (exception during training)
                         →  REVOKED  (worker killed / task cancelled)

Tasks are imported by the worker at startup via tasks/__init__.py.
They are submitted by Flask routes via .apply_async().
"""

import logging
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from celery_app import celery
from database import get_db, ExperimentRepo, ExperimentStatus

# ML engine lives in the shared ml module (extracted from app.py)
from ml.engine import central_train, federated_train
from ml.dp_engine import dp_federated_train
from ml.secagg_engine import secagg_federated_train

logger = logging.getLogger(__name__)


# ── Base task class with shared error handling ─────────────────────────────

class TrainingTask(Task):
    """
    Base class for training tasks.
    on_failure() marks the experiment as FAILED in Postgres even when
    the worker is killed by the OS (OOM, SIGKILL) rather than a Python
    exception — because task_reject_on_worker_lost=True re-queues the
    task, and on the second attempt the status guard prevents double-run.
    """

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        exp_id = kwargs.get("experiment_id") or (args[0] if args else None)
        if exp_id:
            try:
                with get_db() as db:
                    ExperimentRepo.mark_failed(db, exp_id, str(exc))
            except Exception as db_exc:
                logger.error("Could not mark experiment %s failed: %s", exp_id, db_exc)
        super().on_failure(exc, task_id, args, kwargs, einfo)


# ── Central training task ──────────────────────────────────────────────────

@celery.task(
    bind=True,
    base=TrainingTask,
    name="tasks.training_tasks.run_central_training",
    max_retries=0,          # training is not idempotent — don't auto-retry
    track_started=True,
)
def run_central_training(
    self,
    experiment_id: str,
    rows: list,
    headers: list,
    target_col_index: int,
    feature_types: dict,
    epochs: int,
    lr: float,
    algo: str = "logistic",
) -> dict:
    """
    Async Celery task: run central (non-federated) training.

    Returns a dict with training results that Celery stores in Redis.
    The Flask poll endpoint reads this dict to return to the client.
    """
    logger.info("[%s] Starting central training  exp=%s  algo=%s  epochs=%d  lr=%.4f",
                self.request.id, experiment_id, algo, epochs, lr)

    # Guard: only run if still pending/running (prevents double-run on re-queue)
    with get_db() as db:
        exp = ExperimentRepo.get_by_id(db, experiment_id)
        if exp and exp.status == ExperimentStatus.COMPLETED:
            logger.warning("[%s] Experiment %s already completed — skipping",
                           self.request.id, experiment_id)
            return {"skipped": True, "reason": "already completed"}
        ExperimentRepo.mark_running(db, experiment_id)

    try:
        result = central_train(rows, headers, target_col_index, feature_types, epochs, lr, algo=algo)
    except SoftTimeLimitExceeded:
        err = f"Training exceeded soft time limit ({self.app.conf.task_soft_time_limit}s)"
        logger.error("[%s] %s", self.request.id, err)
        with get_db() as db:
            ExperimentRepo.mark_failed(db, experiment_id, err)
        raise
    except Exception as exc:
        logger.exception("[%s] Central training failed: %s", self.request.id, exc)
        with get_db() as db:
            ExperimentRepo.mark_failed(db, experiment_id, str(exc))
        raise

    # Persist result to Postgres
    with get_db() as db:
        ExperimentRepo.mark_completed(db, experiment_id, result)

    result["experimentId"] = experiment_id
    result["taskId"]       = self.request.id

    logger.info("[%s] Central training complete  acc=%.4f  loss=%.6f",
                self.request.id,
                result["testMetrics"]["accuracy"],
                result["finalLoss"])
    return result


# ── Federated training task ────────────────────────────────────────────────

@celery.task(
    bind=True,
    base=TrainingTask,
    name="tasks.training_tasks.run_federated_training",
    max_retries=0,
    track_started=True,
)
def run_federated_training(
    self,
    experiment_id: str,
    rows: list,
    headers: list,
    target_col_index: int,
    feature_types: dict,
    rounds: int,
    local_epochs: int,
    lr: float,
    num_clients: int,
    algo: str = "logistic",
) -> dict:
    """
    Async Celery task: run federated averaging (FedAvg).

    Same lifecycle as run_central_training — see that docstring.
    """
    logger.info("[%s] Starting federated training  exp=%s  algo=%s  rounds=%d  clients=%d",
                self.request.id, experiment_id, algo, rounds, num_clients)

    with get_db() as db:
        exp = ExperimentRepo.get_by_id(db, experiment_id)
        if exp and exp.status == ExperimentStatus.COMPLETED:
            logger.warning("[%s] Experiment %s already completed — skipping",
                           self.request.id, experiment_id)
            return {"skipped": True, "reason": "already completed"}
        ExperimentRepo.mark_running(db, experiment_id)

    try:
        result = federated_train(
            rows, headers, target_col_index, feature_types,
            rounds, local_epochs, lr, num_clients, algo=algo,
        )
    except SoftTimeLimitExceeded:
        err = f"Training exceeded soft time limit ({self.app.conf.task_soft_time_limit}s)"
        logger.error("[%s] %s", self.request.id, err)
        with get_db() as db:
            ExperimentRepo.mark_failed(db, experiment_id, err)
        raise
    except Exception as exc:
        logger.exception("[%s] Federated training failed: %s", self.request.id, exc)
        with get_db() as db:
            ExperimentRepo.mark_failed(db, experiment_id, str(exc))
        raise

    with get_db() as db:
        ExperimentRepo.mark_completed(db, experiment_id, result)

    result["experimentId"] = experiment_id
    result["taskId"]       = self.request.id

    logger.info("[%s] Federated training complete  acc=%.4f  loss=%.6f",
                self.request.id,
                result["testMetrics"]["accuracy"],
                result["finalLoss"])
    return result


# ── DP Federated training task ─────────────────────────────────────────────

@celery.task(
    bind=True,
    base=TrainingTask,
    name="tasks.training_tasks.run_dp_federated_training",
    max_retries=0,
    track_started=True,
)
def run_dp_federated_training(
    self,
    experiment_id: str,
    rows: list,
    headers: list,
    target_col_index: int,
    feature_types: dict,
    rounds: int,
    local_epochs: int,
    lr: float,
    num_clients: int,
    target_epsilon: float,
    delta: float,
    clip_threshold: float,
    noise_multiplier: float | None,
) -> dict:
    """
    Async Celery task: DP-FedAvg with Gaussian noise + Moments Accountant.

    Lifecycle identical to run_federated_training — see that docstring.
    The result dict includes a 'privacy' sub-dict with full budget accounting.
    """
    logger.info(
        "[%s] Starting DP-federated training  exp=%s  rounds=%d  "
        "clients=%d  ε_target=%.2f  δ=%.2e",
        self.request.id, experiment_id, rounds, num_clients,
        target_epsilon, delta,
    )

    with get_db() as db:
        exp = ExperimentRepo.get_by_id(db, experiment_id)
        if exp and exp.status == ExperimentStatus.COMPLETED:
            logger.warning(
                "[%s] Experiment %s already completed — skipping",
                self.request.id, experiment_id,
            )
            return {"skipped": True, "reason": "already completed"}
        ExperimentRepo.mark_running(db, experiment_id)

    try:
        result = dp_federated_train(
            rows, headers, target_col_index, feature_types,
            rounds=rounds,
            local_epochs=local_epochs,
            lr=lr,
            num_clients=num_clients,
            target_epsilon=target_epsilon,
            delta=delta,
            clip_threshold=clip_threshold,
            noise_multiplier=noise_multiplier,
        )
    except SoftTimeLimitExceeded:
        err = f"DP training exceeded soft time limit ({self.app.conf.task_soft_time_limit}s)"
        logger.error("[%s] %s", self.request.id, err)
        with get_db() as db:
            ExperimentRepo.mark_failed(db, experiment_id, err)
        raise
    except Exception as exc:
        logger.exception("[%s] DP-federated training failed: %s", self.request.id, exc)
        with get_db() as db:
            ExperimentRepo.mark_failed(db, experiment_id, str(exc))
        raise

    with get_db() as db:
        ExperimentRepo.mark_completed(db, experiment_id, result)

    result["experimentId"] = experiment_id
    result["taskId"]       = self.request.id

    privacy = result.get("privacy", {})
    logger.info(
        "[%s] DP-federated training complete  acc=%.4f  ε_final=%.4f  rounds=%d",
        self.request.id,
        result["testMetrics"]["accuracy"],
        privacy.get("current_epsilon", 0),
        privacy.get("rounds_consumed", rounds),
    )
    return result


# ── SecAgg federated training task (Day 4) ────────────────────────────────

@celery.task(
    bind=True,
    base=TrainingTask,
    name="tasks.training_tasks.run_secagg_federated_training",
    max_retries=0,
    track_started=True,
)
def run_secagg_federated_training(
    self,
    experiment_id: str,
    rows: list,
    headers: list,
    target_col_index: int,
    feature_types: dict,
    rounds: int,
    local_epochs: int,
    lr: float,
    num_clients: int,
    dropout_rate: float,
    secagg_threshold: int,
) -> dict:
    """
    Async Celery task: FedAvg with Secure Aggregation (SecAgg).

    The server only ever learns the *sum* of client updates — never
    any individual client's gradient.  Lifecycle identical to other
    training tasks; result dict includes a 'secagg' sub-dict with
    per-round audit log and verification status.
    """
    logger.info(
        "[%s] Starting SecAgg-federated training  exp=%s  rounds=%d  "
        "clients=%d  dropout=%.0f%%  threshold=%d",
        self.request.id, experiment_id, rounds, num_clients,
        dropout_rate * 100, secagg_threshold,
    )

    with get_db() as db:
        exp = ExperimentRepo.get_by_id(db, experiment_id)
        if exp and exp.status == ExperimentStatus.COMPLETED:
            logger.warning(
                "[%s] Experiment %s already completed — skipping",
                self.request.id, experiment_id,
            )
            return {"skipped": True, "reason": "already completed"}
        ExperimentRepo.mark_running(db, experiment_id)

    try:
        result = secagg_federated_train(
            rows, headers, target_col_index, feature_types,
            rounds=rounds,
            local_epochs=local_epochs,
            lr=lr,
            num_clients=num_clients,
            dropout_rate=dropout_rate,
            secagg_threshold=secagg_threshold,
        )
    except SoftTimeLimitExceeded:
        err = f"SecAgg training exceeded soft time limit ({self.app.conf.task_soft_time_limit}s)"
        logger.error("[%s] %s", self.request.id, err)
        with get_db() as db:
            ExperimentRepo.mark_failed(db, experiment_id, err)
        raise
    except Exception as exc:
        logger.exception("[%s] SecAgg training failed: %s", self.request.id, exc)
        with get_db() as db:
            ExperimentRepo.mark_failed(db, experiment_id, str(exc))
        raise

    with get_db() as db:
        ExperimentRepo.mark_completed(db, experiment_id, result)

    result["experimentId"] = experiment_id
    result["taskId"]       = self.request.id

    secagg = result.get("secagg", {})
    logger.info(
        "[%s] SecAgg training complete  acc=%.4f  verified=%s  overhead=%.0fms",
        self.request.id,
        result["testMetrics"]["accuracy"],
        secagg.get("all_rounds_verified"),
        secagg.get("total_overhead_ms", 0),
    )
    return result
