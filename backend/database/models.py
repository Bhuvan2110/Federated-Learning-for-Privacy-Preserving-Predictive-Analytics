"""
database/models.py
══════════════════════════════════════════════════════════════════════
SQLAlchemy ORM models for Training Models v4.0

Tables
------
  users          — registered users (auth placeholder for Phase 2)
  experiments    — one row per training run
  experiment_results — serialised metrics / loss history per experiment
  uploaded_files — metadata for CSV uploads (no raw data stored)

All timestamps are stored as UTC. UUIDs are used as primary keys so
that IDs are safe to expose in API responses.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, Text, ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


# ─── Helpers ──────────────────────────────────────────────────────────────

def _now_utc():
    return datetime.now(timezone.utc)

def _uuid():
    return str(uuid.uuid4())


# ─── Enums ────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    ADMIN   = "admin"
    TRAINER = "trainer"
    VIEWER  = "viewer"


class ExperimentStatus(str, enum.Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class ModelType(str, enum.Enum):
    CENTRAL          = "central"
    FEDERATED        = "federated"
    DP_FEDERATED     = "dp_federated"      # Day 3: differentially private FedAvg
    SECAGG_FEDERATED = "secagg_federated"  # Day 4: SecAgg-protected FedAvg


# ─── Models ───────────────────────────────────────────────────────────────

class User(Base):
    """
    User table extended with OAuth 2.0 / OIDC fields  (Day 8).
    Passwords are NOT stored — authentication is delegated to the
    OAuth provider (Auth0 / Keycloak).  The external_id + provider
    pair uniquely identifies a user across providers.
    """
    __tablename__ = "users"

    id         = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    username   = Column(String(64), unique=True, nullable=False, index=True)
    email      = Column(String(256), unique=True, nullable=False, index=True)
    role       = Column(SAEnum(UserRole), nullable=False, default=UserRole.TRAINER)
    is_active  = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=_now_utc)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_now_utc,
                        onupdate=_now_utc)

    # ── OAuth 2.0 / OIDC fields (Day 8) ──────────────────────────────────
    # external_id: subject claim (sub) from the OIDC provider
    external_id    = Column(String(256), nullable=True, index=True)
    provider       = Column(String(64),  nullable=True)   # "auth0" | "keycloak" | "local"
    # ABAC attribute dict — stored as JSON, indexed on role for queries
    abac_attributes = Column(JSON, nullable=True, default=dict)
    # e.g. {"department": "cardiology", "dp_clearance": true, "clearance_level": 2}
    last_login_at  = Column(DateTime(timezone=True), nullable=True)
    last_login_ip  = Column(String(64), nullable=True)

    # relationships
    experiments    = relationship("Experiment", back_populates="user",
                                  cascade="all, delete-orphan")
    uploaded_files = relationship("UploadedFile", back_populates="user",
                                  cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username!r} role={self.role.value}>"

    def to_dict(self):
        return {
            "id":               self.id,
            "username":         self.username,
            "email":            self.email,
            "role":             self.role.value,
            "is_active":        self.is_active,
            "provider":         self.provider,
            "abac_attributes":  self.abac_attributes,
            "last_login_at":    self.last_login_at.isoformat() if self.last_login_at else None,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
        }


# ─── AuditLog ──────────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Tamper-evident audit trail.  Every security-relevant event is recorded
    here with a SHA-256 hash chain linking consecutive entries.
    Deleting or modifying any row breaks all subsequent hashes.
    """
    __tablename__ = "audit_logs"

    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    created_at    = Column(DateTime(timezone=True), nullable=False,
                           default=_now_utc, index=True)

    # Actor
    user_id       = Column(String(256), nullable=True, index=True)  # may be "anonymous"
    email         = Column(String(256), nullable=True)
    role          = Column(String(64),  nullable=True)
    ip_address    = Column(String(64),  nullable=True)

    # Event
    action        = Column(String(64),  nullable=False, index=True)  # AuditAction value
    outcome       = Column(String(16),  nullable=False)              # SUCCESS / FAILURE / DENIED

    # Resource
    resource_id   = Column(String(256), nullable=True, index=True)
    resource_type = Column(String(64),  nullable=True)

    # Payload
    details       = Column(JSON, nullable=True)

    # Hash chain
    prev_hash     = Column(String(64), nullable=False, default="0" * 64)
    entry_hash    = Column(String(64), nullable=False, default="", index=True)

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "created_at":    self.created_at.isoformat() if self.created_at else None,
            "user_id":       self.user_id,
            "email":         self.email,
            "role":          self.role,
            "ip_address":    self.ip_address,
            "action":        self.action,
            "outcome":       self.outcome,
            "resource_id":   self.resource_id,
            "resource_type": self.resource_type,
            "details":       self.details,
            "entry_hash":    self.entry_hash,
            "prev_hash":     self.prev_hash,
        }


class UploadedFile(Base):
    """
    Metadata record for each CSV upload.
    Raw bytes are never persisted to disk — only stats/schema info.
    """
    __tablename__ = "uploaded_files"

    id               = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id          = Column(UUID(as_uuid=False), ForeignKey("users.id",
                               ondelete="CASCADE"), nullable=True, index=True)
    filename         = Column(String(255), nullable=False)
    total_rows       = Column(Integer, nullable=False)
    total_cols       = Column(Integer, nullable=False)
    headers          = Column(JSON, nullable=False)          # list[str]
    column_stats     = Column(JSON, nullable=False)          # list[dict]
    encrypted_upload = Column(Boolean, nullable=False, default=False)
    encryption_method = Column(String(128), nullable=True)
    uploaded_at      = Column(DateTime(timezone=True), nullable=False, default=_now_utc)

    # relationships
    user        = relationship("User", back_populates="uploaded_files")
    experiments = relationship("Experiment", back_populates="uploaded_file")

    def __repr__(self):
        return f"<UploadedFile {self.filename!r} rows={self.total_rows}>"

    def to_dict(self):
        return {
            "id":               self.id,
            "filename":         self.filename,
            "total_rows":       self.total_rows,
            "total_cols":       self.total_cols,
            "headers":          self.headers,
            "column_stats":     self.column_stats,
            "encrypted_upload": self.encrypted_upload,
            "encryption_method": self.encryption_method,
            "uploaded_at":      self.uploaded_at.isoformat() if self.uploaded_at else None,
        }


class Experiment(Base):
    """
    One row per training run (central or federated).
    Hyperparameters are stored in a JSONB column so we can add new
    algorithm types (FedProx, SCAFFOLD, etc.) in Phase 3 without
    schema migrations.
    """
    __tablename__ = "experiments"

    id               = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id          = Column(UUID(as_uuid=False), ForeignKey("users.id",
                               ondelete="SET NULL"), nullable=True, index=True)
    file_id          = Column(UUID(as_uuid=False), ForeignKey("uploaded_files.id",
                               ondelete="SET NULL"), nullable=True, index=True)
    name             = Column(String(256), nullable=False, default="Untitled experiment")
    model_type       = Column(SAEnum(ModelType), nullable=False)
    status           = Column(SAEnum(ExperimentStatus), nullable=False,
                               default=ExperimentStatus.PENDING, index=True)

    # Hyperparameters — flexible JSON blob
    hyperparameters  = Column(JSON, nullable=False, default=dict)
    # e.g. { "epochs": 100, "lr": 0.1, "rounds": 25,
    #         "local_epochs": 5, "num_clients": 5 }

    target_col       = Column(String(256), nullable=True)
    feature_cols     = Column(JSON, nullable=True)           # list[str]
    feature_types    = Column(JSON, nullable=True)           # dict[str, str]
    target_col_index = Column(Integer, nullable=True)

    # ── Differential Privacy settings (Day 3) ─────────────────────────────
    # Stored flat (not inside hyperparameters JSON) so they can be
    # queried/indexed directly and shown in the experiments list.
    dp_enabled          = Column(Boolean, nullable=False, default=False)
    dp_target_epsilon   = Column(Float, nullable=True)   # requested ε budget
    dp_delta            = Column(Float, nullable=True)   # failure probability δ
    dp_clip_threshold   = Column(Float, nullable=True)   # gradient clipping C
    dp_noise_multiplier = Column(Float, nullable=True)   # z = σ/C (auto or manual)

    # ── Secure Aggregation settings (Day 4) ────────────────────────────────
    secagg_enabled   = Column(Boolean, nullable=False, default=False)
    secagg_threshold = Column(Integer, nullable=True)   # Shamir threshold t
    secagg_dropout   = Column(Float,   nullable=True)   # simulated dropout rate

    error_message    = Column(Text, nullable=True)
    started_at       = Column(DateTime(timezone=True), nullable=True)
    completed_at     = Column(DateTime(timezone=True), nullable=True)
    created_at       = Column(DateTime(timezone=True), nullable=False, default=_now_utc)
    updated_at       = Column(DateTime(timezone=True), nullable=False, default=_now_utc,
                               onupdate=_now_utc)

    # relationships
    user          = relationship("User", back_populates="experiments")
    uploaded_file = relationship("UploadedFile", back_populates="experiments")
    result        = relationship("ExperimentResult", back_populates="experiment",
                                 uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return (f"<Experiment {self.name!r} type={self.model_type.value} "
                f"status={self.status.value}>")

    def to_dict(self, include_result=False):
        d = {
            "id":              self.id,
            "name":            self.name,
            "model_type":      self.model_type.value,
            "status":          self.status.value,
            "hyperparameters": self.hyperparameters,
            "target_col":      self.target_col,
            "feature_cols":    self.feature_cols,
            "error_message":   self.error_message,
            "started_at":      self.started_at.isoformat() if self.started_at else None,
            "completed_at":    self.completed_at.isoformat() if self.completed_at else None,
            "created_at":      self.created_at.isoformat() if self.created_at else None,
            # DP fields
            "dp_enabled":          self.dp_enabled,
            "dp_target_epsilon":   self.dp_target_epsilon,
            "dp_delta":            self.dp_delta,
            "dp_clip_threshold":   self.dp_clip_threshold,
            "dp_noise_multiplier": self.dp_noise_multiplier,
            # SecAgg fields
            "secagg_enabled":      self.secagg_enabled,
            "secagg_threshold":    self.secagg_threshold,
            "secagg_dropout":      self.secagg_dropout,
        }
        if include_result and self.result:
            d["result"] = self.result.to_dict()
        return d


class ExperimentResult(Base):
    """
    Metrics and loss history for a completed experiment.
    Stored separately so the experiments table stays lean and we can
    query experiment lists without pulling large JSON blobs.
    """
    __tablename__ = "experiment_results"

    id            = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    experiment_id = Column(UUID(as_uuid=False), ForeignKey("experiments.id",
                            ondelete="CASCADE"), nullable=False, unique=True, index=True)

    # Core metrics
    train_accuracy  = Column(Float, nullable=True)
    test_accuracy   = Column(Float, nullable=True)
    train_precision = Column(Float, nullable=True)
    test_precision  = Column(Float, nullable=True)
    train_recall    = Column(Float, nullable=True)
    test_recall     = Column(Float, nullable=True)
    train_f1        = Column(Float, nullable=True)
    test_f1         = Column(Float, nullable=True)
    final_loss      = Column(Float, nullable=True)

    # Full history / matrix — large blobs
    loss_history    = Column(JSON, nullable=True)    # list[float]
    conf_matrix     = Column(JSON, nullable=True)    # {tp, fp, fn, tn}
    train_metrics   = Column(JSON, nullable=True)    # full dict
    test_metrics    = Column(JSON, nullable=True)    # full dict

    # Dataset info
    train_samples   = Column(Integer, nullable=True)
    test_samples    = Column(Integer, nullable=True)
    unique_labels   = Column(JSON, nullable=True)    # list[str]

    training_time_ms = Column(Integer, nullable=True)
    recorded_at      = Column(DateTime(timezone=True), nullable=False, default=_now_utc)

    # ── Differential Privacy accounting (Day 3) ────────────────────────────
    # Populated only when dp_enabled=True on the parent Experiment.
    dp_epsilon_final    = Column(Float, nullable=True)   # actual ε spent
    dp_epsilon_history  = Column(JSON,  nullable=True)   # list[float] per round
    dp_noise_sigma      = Column(Float, nullable=True)   # σ actually used
    dp_budget_summary   = Column(JSON,  nullable=True)   # full PrivacyBudget.to_dict()

    # ── Secure Aggregation audit log (Day 4) ───────────────────────────────
    # Populated only when secagg_enabled=True on the parent Experiment.
    secagg_summary      = Column(JSON,    nullable=True)  # full SecAgg summary dict
    secagg_all_verified = Column(Boolean, nullable=True)  # True if all rounds verified

    # relationship
    experiment = relationship("Experiment", back_populates="result")

    def __repr__(self):
        return (f"<ExperimentResult exp={self.experiment_id} "
                f"test_acc={self.test_accuracy}>")

    def to_dict(self):
        return {
            "id":               self.id,
            "experiment_id":    self.experiment_id,
            "train_accuracy":   self.train_accuracy,
            "test_accuracy":    self.test_accuracy,
            "train_f1":         self.train_f1,
            "test_f1":          self.test_f1,
            "final_loss":       self.final_loss,
            "loss_history":     self.loss_history,
            "conf_matrix":      self.conf_matrix,
            "train_metrics":    self.train_metrics,
            "test_metrics":     self.test_metrics,
            "train_samples":    self.train_samples,
            "test_samples":     self.test_samples,
            "unique_labels":    self.unique_labels,
            "training_time_ms": self.training_time_ms,
            "recorded_at":      self.recorded_at.isoformat() if self.recorded_at else None,
            # DP accounting
            "dp_epsilon_final":   self.dp_epsilon_final,
            "dp_epsilon_history": self.dp_epsilon_history,
            "dp_noise_sigma":     self.dp_noise_sigma,
            "dp_budget_summary":  self.dp_budget_summary,
            # SecAgg audit
            "secagg_summary":      self.secagg_summary,
            "secagg_all_verified": self.secagg_all_verified,
        }
