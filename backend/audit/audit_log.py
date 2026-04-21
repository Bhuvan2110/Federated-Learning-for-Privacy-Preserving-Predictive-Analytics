"""
audit/audit_log.py
══════════════════════════════════════════════════════════════════════
Tamper-evident audit trail  (Day 8)

Every security-relevant event is recorded with:
  • Timestamp (UTC, millisecond precision)
  • Actor (user_id, email, role, IP address)
  • Action (LOGIN, LOGOUT, TOKEN_REFRESH, TRAIN_SUBMITTED, …)
  • Resource (experiment_id, file_id, etc.)
  • Outcome (SUCCESS / FAILURE)
  • Hash chain (each entry hashes the previous entry's hash, making
    the log tamper-evident — any modification breaks the chain)

Hash chain
──────────
  entry_hash = SHA-256(prev_hash || timestamp || user_id || action || outcome || details_json)

  The genesis entry uses prev_hash = "0" * 64.
  Any deletion or modification of a row breaks all subsequent hashes.
  Verification: `AuditLogger.verify_chain(since=datetime)` returns False
  if any link is broken.

Storage: Postgres `audit_logs` table (see database/models.py).

Environment variables
─────────────────────
  AUDIT_ENABLED   — set to "false" to disable (default: true)
  AUDIT_HASH_CHAIN — set to "false" to skip chaining (faster, less secure)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

AUDIT_ENABLED    = os.getenv("AUDIT_ENABLED",    "true").lower() == "true"
AUDIT_HASH_CHAIN = os.getenv("AUDIT_HASH_CHAIN", "true").lower() == "true"


# ════════════════════════════════════════════════════════════════════════════
#  Audit event types
# ════════════════════════════════════════════════════════════════════════════

class AuditAction(str, Enum):
    # Authentication
    LOGIN             = "LOGIN"
    LOGOUT            = "LOGOUT"
    TOKEN_REFRESH     = "TOKEN_REFRESH"
    TOKEN_REVOKED     = "TOKEN_REVOKED"
    AUTH_FAILED       = "AUTH_FAILED"
    ACCESS_DENIED     = "ACCESS_DENIED"

    # Data uploads
    UPLOAD_ENCRYPTED  = "UPLOAD_ENCRYPTED"
    UPLOAD_PLAIN      = "UPLOAD_PLAIN"
    DECRYPT_FAILED    = "DECRYPT_FAILED"

    # Training
    TRAIN_SUBMITTED   = "TRAIN_SUBMITTED"
    TRAIN_COMPLETED   = "TRAIN_COMPLETED"
    TRAIN_FAILED      = "TRAIN_FAILED"
    TRAIN_CANCELLED   = "TRAIN_CANCELLED"

    # Data access
    EXPERIMENT_READ   = "EXPERIMENT_READ"
    AUDIT_LOG_READ    = "AUDIT_LOG_READ"

    # Admin
    USER_CREATED      = "USER_CREATED"
    USER_DEACTIVATED  = "USER_DEACTIVATED"
    ROLE_CHANGED      = "ROLE_CHANGED"


class AuditOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DENIED  = "DENIED"


# ════════════════════════════════════════════════════════════════════════════
#  Hash chain helper
# ════════════════════════════════════════════════════════════════════════════

GENESIS_HASH = "0" * 64


def _compute_hash(
    prev_hash: str,
    timestamp: str,
    user_id: str,
    action: str,
    outcome: str,
    details_json: str,
) -> str:
    """SHA-256 of the canonical string representation of an audit entry."""
    canonical = "|".join([
        prev_hash, timestamp, user_id, action, outcome, details_json
    ])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════════════════════
#  AuditLogger
# ════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Records audit events to the Postgres `audit_logs` table.
    All methods are non-blocking — failures are logged as warnings and
    never propagate to the caller (audit must not break the main flow).

    Usage:
        AuditLogger.log(
            action=AuditAction.TRAIN_SUBMITTED,
            outcome=AuditOutcome.SUCCESS,
            user_id="uuid",
            email="alice@example.com",
            role="trainer",
            ip_address="1.2.3.4",
            resource_id="exp-uuid",
            resource_type="experiment",
            details={"model_type": "dp_federated", "rounds": 25},
        )
    """

    @staticmethod
    def log(
        action:        AuditAction,
        outcome:       AuditOutcome,
        user_id:       str               = "anonymous",
        email:         str               = "",
        role:          str               = "",
        ip_address:    str               = "",
        resource_id:   Optional[str]     = None,
        resource_type: Optional[str]     = None,
        details:       Optional[dict]    = None,
    ) -> None:
        """
        Record one audit event.  Non-blocking — errors are swallowed.
        """
        if not AUDIT_ENABLED:
            return

        try:
            AuditLogger._write(
                action=action.value,
                outcome=outcome.value,
                user_id=user_id,
                email=email,
                role=role,
                ip_address=ip_address,
                resource_id=resource_id,
                resource_type=resource_type,
                details=details or {},
            )
        except Exception as exc:
            logger.warning("Audit log write failed (non-fatal): %s", exc)

    @staticmethod
    def _write(
        action: str, outcome: str,
        user_id: str, email: str, role: str, ip_address: str,
        resource_id: Optional[str], resource_type: Optional[str],
        details: dict,
    ) -> None:
        """Internal write — imports DB lazily so audit works without DB."""
        from database import get_db
        from database.models import AuditLog

        timestamp_str = datetime.now(timezone.utc).isoformat()
        details_json  = json.dumps(details, default=str, sort_keys=True)

        # Compute hash chain
        prev_hash = GENESIS_HASH
        if AUDIT_HASH_CHAIN:
            prev_hash = AuditLogger._get_last_hash()

        entry_hash = _compute_hash(
            prev_hash, timestamp_str, user_id, action, outcome, details_json
        )

        with get_db() as db:
            entry = AuditLog(
                action        = action,
                outcome       = outcome,
                user_id       = user_id,
                email         = email,
                role          = role,
                ip_address    = ip_address,
                resource_id   = resource_id,
                resource_type = resource_type,
                details       = details,
                prev_hash     = prev_hash,
                entry_hash    = entry_hash,
            )
            db.add(entry)

        logger.debug(
            "AUDIT: %s %s user=%s resource=%s/%s",
            action, outcome, user_id, resource_type, resource_id,
        )

    @staticmethod
    def _get_last_hash() -> str:
        """Return the entry_hash of the most recent audit log entry."""
        try:
            from database import get_db
            from database.models import AuditLog
            from sqlalchemy import desc
            with get_db() as db:
                last = (db.query(AuditLog)
                        .order_by(desc(AuditLog.created_at))
                        .first())
                return last.entry_hash if last else GENESIS_HASH
        except Exception:
            return GENESIS_HASH

    @staticmethod
    def verify_chain(limit: int = 1000) -> tuple[bool, Optional[str]]:
        """
        Verify the hash chain integrity for the most recent `limit` entries.
        Returns (True, None) if chain is intact.
        Returns (False, "reason") if tampering is detected.
        """
        try:
            from database import get_db
            from database.models import AuditLog
            with get_db() as db:
                entries = (db.query(AuditLog)
                           .order_by(AuditLog.created_at.asc())
                           .limit(limit).all())

            if not entries:
                return True, None

            prev = GENESIS_HASH
            for entry in entries:
                details_json = json.dumps(entry.details or {}, default=str, sort_keys=True)
                expected = _compute_hash(
                    prev,
                    entry.created_at.isoformat(),
                    entry.user_id or "",
                    entry.action  or "",
                    entry.outcome or "",
                    details_json,
                )
                if expected != entry.entry_hash:
                    return False, (
                        f"Hash mismatch at entry id={entry.id} "
                        f"created_at={entry.created_at}"
                    )
                prev = entry.entry_hash

            return True, None

        except Exception as exc:
            return False, f"Verification error: {exc}"

    @staticmethod
    def list_recent(limit: int = 100, user_id: Optional[str] = None) -> list:
        """Return recent audit entries as dicts."""
        try:
            from database import get_db
            from database.models import AuditLog
            from sqlalchemy import desc
            with get_db() as db:
                q = db.query(AuditLog).order_by(desc(AuditLog.created_at))
                if user_id:
                    q = q.filter(AuditLog.user_id == user_id)
                entries = q.limit(limit).all()
            return [e.to_dict() for e in entries]
        except Exception as exc:
            logger.warning("Audit list failed: %s", exc)
            return []


# ════════════════════════════════════════════════════════════════════════════
#  Convenience: log from Flask request context
# ════════════════════════════════════════════════════════════════════════════

def log_from_request(
    action: AuditAction,
    outcome: AuditOutcome,
    resource_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    details: Optional[dict] = None,
) -> None:
    """
    Log an audit event, automatically extracting user info from flask.g.claims
    and the IP from the request context.

    Safe to call outside a request context (does nothing).
    """
    try:
        from flask import g, request as flask_request
        claims = getattr(g, "claims", None)
        ip     = (flask_request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                  or flask_request.remote_addr or "")

        AuditLogger.log(
            action        = action,
            outcome       = outcome,
            user_id       = getattr(claims, "user_id", "anonymous") if claims else "anonymous",
            email         = getattr(claims, "email",   "") if claims else "",
            role          = getattr(claims, "role",    "") if claims else "",
            ip_address    = ip,
            resource_id   = resource_id,
            resource_type = resource_type,
            details       = details,
        )
    except RuntimeError:
        # Outside request context — log without request data
        AuditLogger.log(
            action=action, outcome=outcome,
            resource_id=resource_id, resource_type=resource_type,
            details=details,
        )
    except Exception as exc:
        logger.warning("log_from_request failed: %s", exc)
