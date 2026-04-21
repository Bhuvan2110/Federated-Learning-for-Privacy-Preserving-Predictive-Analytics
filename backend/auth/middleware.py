"""
auth/middleware.py
══════════════════════════════════════════════════════════════════════
Flask authentication + authorisation decorators  (Day 8)

Decorators
──────────
  @require_auth
    Validates the Bearer token in Authorization header.
    Sets flask.g.claims (TokenClaims) on success.
    Returns 401 on missing/invalid/expired token.

  @require_role(*roles)
    Builds on @require_auth.  Returns 403 if the user's role is not
    in the allowed set.
    Example: @require_role("admin", "trainer")

  @require_experiment_access(action)
    Builds on @require_auth.  Loads the experiment from Postgres,
    runs the ABAC policy engine, returns 403 on denial.
    The experiment_id must be in the URL as <exp_id> or <experiment_id>.
    Example: @require_experiment_access(Action.TRAIN)

  AUTH_OPTIONAL env var
    Set AUTH_OPTIONAL=true to make all auth decorators pass through
    without validation.  Used for local dev without a token issuer.
    NEVER set this in production.

Token extraction
────────────────
  Reads from Authorization: Bearer <token>
  Falls back to ?token=<token> query parameter (for WebSocket / SSE).
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Optional

from flask import request, jsonify, g

from auth.jwt_handler import decode_access_token, TokenError, TokenClaims
from auth.abac import Action, Decision, evaluate as abac_evaluate

logger = logging.getLogger(__name__)

AUTH_OPTIONAL = os.getenv("AUTH_OPTIONAL", "false").lower() == "true"
AUTH_ENABLED  = os.getenv("AUTH_ENABLED",  "true").lower()  == "true"

if AUTH_OPTIONAL:
    logger.warning(
        "⚠️  AUTH_OPTIONAL=true — all auth checks bypassed. "
        "NEVER use this in production!"
    )


# ════════════════════════════════════════════════════════════════════════════
#  Token extraction
# ════════════════════════════════════════════════════════════════════════════

def _extract_token() -> Optional[str]:
    """
    Extract Bearer token from:
      1. Authorization: Bearer <token>   (preferred)
      2. ?token=<token>                  (query param fallback for SSE/WS)
    """
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return request.args.get("token") or None


def _anonymous_claims() -> TokenClaims:
    """Return a minimal anonymous claims object for AUTH_OPTIONAL mode."""
    class _AnonClaims:
        user_id    = "anonymous"
        email      = "anonymous@localhost"
        role       = "admin"   # full access in dev mode
        attributes = {"dp_clearance": True}
        jti        = "anonymous"
        token_type = "access"
        raw        = {}
        def is_admin(self)   -> bool: return True
        def is_trainer(self) -> bool: return True
        def is_viewer(self)  -> bool: return True
        def __repr__(self): return "TokenClaims(anonymous)"
    return _AnonClaims()


# ════════════════════════════════════════════════════════════════════════════
#  @require_auth
# ════════════════════════════════════════════════════════════════════════════

def require_auth(fn):
    """
    Decorator: validate Bearer JWT.  Sets g.claims on success.

    On failure, returns:
      401  {"error": "...", "code": "invalid_token" | "token_expired" | ...}
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if AUTH_OPTIONAL or not AUTH_ENABLED:
            g.claims = _anonymous_claims()
            return fn(*args, **kwargs)

        token = _extract_token()
        if not token:
            return jsonify({
                "error": "Missing authentication token",
                "code":  "missing_token",
            }), 401

        try:
            g.claims = decode_access_token(token)
        except TokenError as exc:
            return jsonify({"error": str(exc), "code": exc.code}), 401
        except Exception as exc:
            logger.exception("Unexpected error in require_auth: %s", exc)
            return jsonify({"error": "Authentication service error", "code": "auth_error"}), 500

        return fn(*args, **kwargs)
    return wrapper


# ════════════════════════════════════════════════════════════════════════════
#  @require_role
# ════════════════════════════════════════════════════════════════════════════

def require_role(*allowed_roles: str):
    """
    Decorator factory: only allow users whose role is in allowed_roles.
    Must be combined with @require_auth (applied after — closer to the fn).

    Example:
        @app.route("/api/admin/users")
        @require_auth
        @require_role("admin")
        def admin_users(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            claims: TokenClaims = getattr(g, "claims", None)
            if claims is None:
                return jsonify({"error": "Not authenticated", "code": "not_authenticated"}), 401

            if AUTH_OPTIONAL or not AUTH_ENABLED:
                return fn(*args, **kwargs)

            if claims.role not in allowed_roles:
                return jsonify({
                    "error": f"Role '{claims.role}' is not authorised for this endpoint. "
                             f"Required: {list(allowed_roles)}",
                    "code":  "insufficient_role",
                }), 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════
#  @require_experiment_access
# ════════════════════════════════════════════════════════════════════════════

def require_experiment_access(action: Action):
    """
    Decorator factory: run ABAC policy for the given action on an experiment.

    The experiment UUID is read from the URL kwargs as:
      <exp_id>, <experiment_id>

    If the experiment does not exist, returns 404.
    If access is denied, returns 403 with the policy reason.

    Example:
        @app.route("/api/train/central", methods=["POST"])
        @require_auth
        @require_experiment_access(Action.TRAIN)
        def api_train_central(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if AUTH_OPTIONAL or not AUTH_ENABLED:
                return fn(*args, **kwargs)

            claims: TokenClaims = getattr(g, "claims", None)
            if claims is None:
                return jsonify({"error": "Not authenticated", "code": "not_authenticated"}), 401

            # Extract experiment id from URL kwargs or request JSON
            exp_id = (
                kwargs.get("exp_id")
                or kwargs.get("experiment_id")
                or (request.get_json(silent=True) or {}).get("experimentId")
            )

            # Build resource dict
            resource: dict = {"type": "experiment", "owner_id": None, "dp_enabled": False}

            if exp_id:
                try:
                    from database import get_db, ExperimentRepo
                    with get_db() as db:
                        exp = ExperimentRepo.get_by_id(db, exp_id)
                    if exp is None:
                        return jsonify({"error": "Experiment not found"}), 404
                    resource = {
                        "type":       "experiment",
                        "owner_id":   exp.user_id,
                        "dp_enabled": exp.dp_enabled,
                        "model_type": exp.model_type.value,
                        "exp_id":     exp_id,
                    }
                except Exception as exc:
                    logger.warning("Could not load experiment for ABAC: %s", exc)

            # Build subject dict from claims
            subject = {
                "user_id":   claims.user_id,
                "email":     claims.email,
                "role":      claims.role,
                "attrs":     claims.attributes,
                "is_active": True,
            }

            decision: Decision = abac_evaluate(subject, resource, action)

            if not decision.allowed:
                logger.info(
                    "ABAC denied: user=%s role=%s action=%s rule=%s reason=%r",
                    claims.user_id, claims.role, action.value,
                    decision.rule, decision.reason,
                )
                return jsonify({
                    "error":  decision.reason,
                    "code":   "access_denied",
                    "action": action.value,
                    "rule":   decision.rule,
                }), 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════
#  get_current_user  (convenience)
# ════════════════════════════════════════════════════════════════════════════

def get_current_user() -> Optional[TokenClaims]:
    """Return the TokenClaims stored in flask.g, or None if not authenticated."""
    return getattr(g, "claims", None)
