"""
database/repository.py
══════════════════════════════════════════════════════════════════════
Repository pattern — one class per model, all queries here.
Routes and business logic stay SQL-free.

Pattern
-------
    from database.repository import UserRepo, FileRepo, ExperimentRepo
    from database.db import get_db

    with get_db() as db:
        user = UserRepo.get_by_username(db, "alice")
        exp  = ExperimentRepo.create(db, user_id=user.id, ...)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from database.models import (
    User, UserRole,
    UploadedFile,
    Experiment, ExperimentStatus, ModelType,
    ExperimentResult,
)

# MLflow tracking (Day 6) — imported lazily so DB layer works without mlflow
def _get_tracker():
    try:
        from tracking.mlflow_tracker import get_tracker
        return get_tracker()
    except Exception:
        return None

# Prometheus metrics (Day 7) — imported lazily
def _record_complete(result_payload, model_type, exp_id, duration_s):
    try:
        from monitoring.metrics import record_training_complete
        record_training_complete(result_payload, model_type, exp_id, duration_s)
    except Exception:
        pass

def _record_failed(model_type):
    try:
        from monitoring.metrics import record_experiment_failed
        record_experiment_failed(model_type)
    except Exception:
        pass

logger = logging.getLogger(__name__)


# ─── UserRepo ─────────────────────────────────────────────────────────────

class UserRepo:

    @staticmethod
    def create(db: Session, *, username: str, email: str,
               role: UserRole = UserRole.TRAINER) -> User:
        user = User(username=username, email=email, role=role)
        db.add(user)
        db.flush()   # get the generated id without committing
        logger.info("Created user %s (%s)", username, user.id)
        return user

    @staticmethod
    def get_by_id(db: Session, user_id: str) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def list_all(db: Session, *, limit: int = 100, offset: int = 0) -> list[User]:
        return db.query(User).order_by(User.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    def deactivate(db: Session, user_id: str) -> Optional[User]:
        user = UserRepo.get_by_id(db, user_id)
        if user:
            user.is_active = False
            db.flush()
        return user


# ─── FileRepo ─────────────────────────────────────────────────────────────

class FileRepo:

    @staticmethod
    def create(db: Session, *,
               filename: str,
               total_rows: int,
               total_cols: int,
               headers: list,
               column_stats: list,
               encrypted_upload: bool = False,
               encryption_method: Optional[str] = None,
               user_id: Optional[str] = None) -> UploadedFile:

        f = UploadedFile(
            filename=filename,
            total_rows=total_rows,
            total_cols=total_cols,
            headers=headers,
            column_stats=column_stats,
            encrypted_upload=encrypted_upload,
            encryption_method=encryption_method,
            user_id=user_id,
        )
        db.add(f)
        db.flush()
        logger.info("Recorded upload %s → %s rows, enc=%s", filename, total_rows, encrypted_upload)
        return f

    @staticmethod
    def get_by_id(db: Session, file_id: str) -> Optional[UploadedFile]:
        return db.query(UploadedFile).filter(UploadedFile.id == file_id).first()

    @staticmethod
    def list_for_user(db: Session, user_id: str, *,
                      limit: int = 50, offset: int = 0) -> list[UploadedFile]:
        return (db.query(UploadedFile)
                .filter(UploadedFile.user_id == user_id)
                .order_by(UploadedFile.uploaded_at.desc())
                .offset(offset).limit(limit).all())

    @staticmethod
    def list_all(db: Session, *, limit: int = 100, offset: int = 0) -> list[UploadedFile]:
        return (db.query(UploadedFile)
                .order_by(UploadedFile.uploaded_at.desc())
                .offset(offset).limit(limit).all())


# ─── ExperimentRepo ───────────────────────────────────────────────────────

class ExperimentRepo:

    @staticmethod
    def create(db: Session, *,
               model_type: ModelType,
               hyperparameters: dict,
               target_col_index: int,
               feature_types: dict,
               name: str = "Untitled experiment",
               user_id: Optional[str] = None,
               file_id: Optional[str] = None,
               # DP fields (Day 3)
               dp_enabled: bool = False,
               dp_target_epsilon: Optional[float] = None,
               dp_delta: Optional[float] = None,
               dp_clip_threshold: Optional[float] = None,
               dp_noise_multiplier: Optional[float] = None,
               # SecAgg fields (Day 4)
               secagg_enabled: bool = False,
               secagg_threshold: Optional[int] = None,
               secagg_dropout: Optional[float] = None) -> Experiment:

        exp = Experiment(
            name=name,
            model_type=model_type,
            status=ExperimentStatus.PENDING,
            hyperparameters=hyperparameters,
            target_col_index=target_col_index,
            feature_types=feature_types,
            user_id=user_id,
            file_id=file_id,
            dp_enabled=dp_enabled,
            dp_target_epsilon=dp_target_epsilon,
            dp_delta=dp_delta,
            dp_clip_threshold=dp_clip_threshold,
            dp_noise_multiplier=dp_noise_multiplier,
            secagg_enabled=secagg_enabled,
            secagg_threshold=secagg_threshold,
            secagg_dropout=secagg_dropout,
        )
        db.add(exp)
        db.flush()
        logger.info("Created experiment %s (%s)", name, exp.id)
        return exp

    @staticmethod
    def get_by_id(db: Session, exp_id: str) -> Optional[Experiment]:
        return db.query(Experiment).filter(Experiment.id == exp_id).first()

    @staticmethod
    def list_all(db: Session, *, limit: int = 100, offset: int = 0,
                 status: Optional[ExperimentStatus] = None) -> list[Experiment]:
        q = db.query(Experiment)
        if status:
            q = q.filter(Experiment.status == status)
        return q.order_by(Experiment.created_at.desc()).offset(offset).limit(limit).all()

    @staticmethod
    def list_for_user(db: Session, user_id: str, *,
                      limit: int = 50, offset: int = 0) -> list[Experiment]:
        return (db.query(Experiment)
                .filter(Experiment.user_id == user_id)
                .order_by(Experiment.created_at.desc())
                .offset(offset).limit(limit).all())

    @staticmethod
    def mark_running(db: Session, exp_id: str) -> Optional[Experiment]:
        exp = ExperimentRepo.get_by_id(db, exp_id)
        if exp:
            exp.status     = ExperimentStatus.RUNNING
            exp.started_at = datetime.now(timezone.utc)
            db.flush()
        return exp

    @staticmethod
    def mark_completed(db: Session, exp_id: str,
                       result_payload: dict) -> Optional[Experiment]:
        """
        Transition experiment to COMPLETED and persist the result blob.
        result_payload is the dict returned by central_train(),
        federated_train(), or dp_federated_train().
        """
        exp = ExperimentRepo.get_by_id(db, exp_id)
        if not exp:
            return None

        exp.status        = ExperimentStatus.COMPLETED
        exp.completed_at  = datetime.now(timezone.utc)
        exp.target_col    = result_payload.get("targetCol")
        exp.feature_cols  = result_payload.get("featureCols")

        # Persist actual noise_multiplier used (may have been auto-calibrated)
        privacy = result_payload.get("privacy") or {}
        if privacy:
            exp.dp_noise_multiplier = privacy.get("noise_multiplier", exp.dp_noise_multiplier)

        tm = result_payload.get("trainMetrics", {})
        te = result_payload.get("testMetrics",  {})

        result = ExperimentResult(
            experiment_id    = exp.id,
            train_accuracy   = tm.get("accuracy"),
            test_accuracy    = te.get("accuracy"),
            train_precision  = tm.get("precision"),
            test_precision   = te.get("precision"),
            train_recall     = tm.get("recall"),
            test_recall      = te.get("recall"),
            train_f1         = tm.get("f1"),
            test_f1          = te.get("f1"),
            final_loss       = result_payload.get("finalLoss"),
            loss_history     = result_payload.get("lossHistory"),
            conf_matrix      = te.get("confMatrix"),
            train_metrics    = tm,
            test_metrics     = te,
            train_samples    = result_payload.get("trainSamples"),
            test_samples     = result_payload.get("testSamples"),
            unique_labels    = result_payload.get("uniqueLabels"),
            training_time_ms = result_payload.get("trainingTimeMs"),
            # DP accounting (Day 3) — None for non-DP runs
            dp_epsilon_final   = privacy.get("current_epsilon"),
            dp_epsilon_history = privacy.get("epsilon_history"),
            dp_noise_sigma     = privacy.get("noise_sigma"),
            dp_budget_summary  = privacy if privacy else None,
            # SecAgg audit (Day 4) — None for non-SecAgg runs
            secagg_summary      = result_payload.get("secagg"),
            secagg_all_verified = (result_payload["secagg"].get("all_rounds_verified")
                                   if result_payload.get("secagg") else None),
        )
        db.add(result)
        db.flush()
        secagg = result_payload.get("secagg") or {}
        logger.info(
            "Experiment %s completed — test_acc=%.4f  loss=%.6f%s%s",
            exp_id,
            te.get("accuracy", 0),
            result_payload.get("finalLoss", 0),
            f"  ε={privacy['current_epsilon']:.4f}" if privacy else "",
            f"  secagg_verified={secagg.get('all_rounds_verified')}" if secagg else "",
        )

        # MLflow tracking (Day 6) — non-blocking, runs after DB commit
        try:
            tracker = _get_tracker()
            if tracker:
                tracker.log_run(exp, result_payload)
        except Exception as mlflow_exc:
            logger.warning("MLflow tracking failed (non-fatal): %s", mlflow_exc)

        # Prometheus metrics (Day 7)
        started_at = exp.started_at
        duration_s = (
            (exp.completed_at - started_at).total_seconds()
            if started_at and exp.completed_at else 0.0
        )
        _record_complete(result_payload, exp.model_type.value, exp_id, duration_s)

        return exp

    @staticmethod
    def mark_failed(db: Session, exp_id: str, error: str) -> Optional[Experiment]:
        exp = ExperimentRepo.get_by_id(db, exp_id)
        if exp:
            exp.status        = ExperimentStatus.FAILED
            exp.completed_at  = datetime.now(timezone.utc)
            exp.error_message = error
            db.flush()
            logger.error("Experiment %s failed: %s", exp_id, error)
            _record_failed(exp.model_type.value)
        return exp


# ─── ResultRepo ───────────────────────────────────────────────────────────

class ResultRepo:

    @staticmethod
    def get_for_experiment(db: Session, exp_id: str) -> Optional[ExperimentResult]:
        return (db.query(ExperimentResult)
                .filter(ExperimentResult.experiment_id == exp_id)
                .first())

    @staticmethod
    def top_by_test_accuracy(db: Session, *, limit: int = 10) -> list[ExperimentResult]:
        return (db.query(ExperimentResult)
                .order_by(ExperimentResult.test_accuracy.desc())
                .limit(limit).all())
