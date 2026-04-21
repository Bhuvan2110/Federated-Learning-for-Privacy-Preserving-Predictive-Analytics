"""
auth/jwt_handler.py
══════════════════════════════════════════════════════════════════════
JWT access + refresh token management  (Day 8)

Token design
────────────
  Access token  — short-lived (default 15 min), carries identity + roles
  Refresh token — longer-lived (default 7 days), used to get new access tokens

  Both are signed JWTs (HS256 by default, RS256 supported for OIDC providers).

  Access token payload:
    sub   — user UUID (database primary key)
    email — user email
    role  — UserRole value string
    attrs — ABAC attribute dict  {department, data_tier, clearance_level, …}
    type  — "access"
    iat, exp, jti (unique token ID for revocation)

  Refresh token payload:
    sub, type="refresh", jti, iat, exp
    (minimal — never carries sensitive claims)

Revocation
──────────
  Token revocation uses a Redis blocklist keyed by `jti`.
  On logout or role change, the jti is added to the blocklist with
  TTL = remaining token lifetime.  If Redis is unavailable, revocation
  is skipped with a warning (fail-open for availability).

Environment variables
─────────────────────
  JWT_SECRET_KEY        — HMAC signing secret (required in production)
  JWT_ALGORITHM         — "HS256" (default) or "RS256" for OIDC
  JWT_ACCESS_TOKEN_TTL  — access token lifetime in seconds (default 900 = 15 min)
  JWT_REFRESH_TOKEN_TTL — refresh token lifetime in seconds (default 604800 = 7 days)
  REDIS_URL             — for token revocation blocklist (default redis://localhost:6379/2)
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────

JWT_SECRET_KEY        = os.getenv("JWT_SECRET_KEY", "INSECURE_DEV_SECRET_CHANGE_IN_PROD")
JWT_ALGORITHM         = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_TTL  = int(os.getenv("JWT_ACCESS_TOKEN_TTL",  "900"))    # 15 min
JWT_REFRESH_TOKEN_TTL = int(os.getenv("JWT_REFRESH_TOKEN_TTL", "604800")) # 7 days
REDIS_BLOCKLIST_DB    = int(os.getenv("JWT_REDIS_BLOCKLIST_DB", "2"))

_WARN_DEFAULT_SECRET = JWT_SECRET_KEY == "INSECURE_DEV_SECRET_CHANGE_IN_PROD"

# ── Optional imports ──────────────────────────────────────────────────────
try:
    import jwt as pyjwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False
    logger.warning("PyJWT not installed — JWT auth disabled. pip install PyJWT")

try:
    import redis as redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
#  Redis blocklist (optional)
# ════════════════════════════════════════════════════════════════════════════

_redis_client: Optional[object] = None

def _get_redis():
    global _redis_client
    if _redis_client is None and _REDIS_AVAILABLE:
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            _redis_client = redis_lib.from_url(
                redis_url, db=REDIS_BLOCKLIST_DB, decode_responses=True
            )
            _redis_client.ping()
        except Exception as exc:
            logger.warning("JWT blocklist Redis unavailable: %s", exc)
            _redis_client = None
    return _redis_client


# ════════════════════════════════════════════════════════════════════════════
#  Token creation
# ════════════════════════════════════════════════════════════════════════════

class TokenPair:
    """Returned by create_token_pair(); holds both tokens and metadata."""
    def __init__(self, access_token: str, refresh_token: str,
                 access_jti: str, refresh_jti: str,
                 access_expires: datetime, refresh_expires: datetime) -> None:
        self.access_token    = access_token
        self.refresh_token   = refresh_token
        self.access_jti      = access_jti
        self.refresh_jti     = refresh_jti
        self.access_expires  = access_expires
        self.refresh_expires = refresh_expires

    def to_dict(self) -> dict:
        return {
            "access_token":    self.access_token,
            "refresh_token":   self.refresh_token,
            "token_type":      "Bearer",
            "expires_in":      JWT_ACCESS_TOKEN_TTL,
            "access_expires":  self.access_expires.isoformat(),
            "refresh_expires": self.refresh_expires.isoformat(),
        }


def create_token_pair(
    user_id: str,
    email: str,
    role: str,
    attributes: Optional[dict] = None,
) -> TokenPair:
    """
    Create an access + refresh token pair for a user.

    Parameters
    ──────────
    user_id    — database UUID (becomes `sub` claim)
    email      — user email (carried in access token)
    role       — UserRole value string (admin / trainer / viewer)
    attributes — ABAC attribute dict, e.g. {department: "cardiology"}

    Returns TokenPair or raises RuntimeError if JWT unavailable.
    """
    if not _JWT_AVAILABLE:
        raise RuntimeError("PyJWT not installed — cannot create tokens")
    if _WARN_DEFAULT_SECRET:
        logger.warning(
            "JWT_SECRET_KEY is the insecure development default. "
            "Set a strong secret in production!"
        )

    now             = datetime.now(timezone.utc)
    access_jti      = str(uuid.uuid4())
    refresh_jti     = str(uuid.uuid4())
    access_expires  = now + timedelta(seconds=JWT_ACCESS_TOKEN_TTL)
    refresh_expires = now + timedelta(seconds=JWT_REFRESH_TOKEN_TTL)

    access_payload = {
        "sub":   user_id,
        "email": email,
        "role":  role,
        "attrs": attributes or {},
        "type":  "access",
        "jti":   access_jti,
        "iat":   now,
        "exp":   access_expires,
    }

    refresh_payload = {
        "sub":  user_id,
        "type": "refresh",
        "jti":  refresh_jti,
        "iat":  now,
        "exp":  refresh_expires,
    }

    access_token  = pyjwt.encode(access_payload,  JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    refresh_token = pyjwt.encode(refresh_payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    return TokenPair(
        access_token    = access_token,
        refresh_token   = refresh_token,
        access_jti      = access_jti,
        refresh_jti     = refresh_jti,
        access_expires  = access_expires,
        refresh_expires = refresh_expires,
    )


# ════════════════════════════════════════════════════════════════════════════
#  Token validation
# ════════════════════════════════════════════════════════════════════════════

class TokenClaims:
    """Decoded, validated token payload."""
    def __init__(self, payload: dict) -> None:
        self._p = payload

    @property
    def user_id(self)    -> str:            return self._p["sub"]
    @property
    def email(self)      -> str:            return self._p.get("email", "")
    @property
    def role(self)       -> str:            return self._p.get("role", "viewer")
    @property
    def attributes(self) -> dict:           return self._p.get("attrs", {})
    @property
    def jti(self)        -> str:            return self._p.get("jti", "")
    @property
    def token_type(self) -> str:            return self._p.get("type", "access")
    @property
    def raw(self)        -> dict:           return dict(self._p)

    def is_admin(self)   -> bool:           return self.role == "admin"
    def is_trainer(self) -> bool:           return self.role in ("admin", "trainer")
    def is_viewer(self)  -> bool:           return True  # all roles can view

    def __repr__(self) -> str:
        return f"TokenClaims(sub={self.user_id!r}, role={self.role!r})"


class TokenError(Exception):
    """Raised when token validation fails — maps to HTTP 401."""
    def __init__(self, message: str, code: str = "invalid_token") -> None:
        super().__init__(message)
        self.code = code


def decode_access_token(token: str) -> TokenClaims:
    """
    Decode and validate an access token.

    Raises TokenError on:
      - Expired token         (code: "token_expired")
      - Invalid signature     (code: "invalid_token")
      - Wrong token type      (code: "wrong_token_type")
      - Token revoked         (code: "token_revoked")
      - PyJWT not installed   (code: "auth_unavailable")
    """
    if not _JWT_AVAILABLE:
        raise TokenError("Authentication service unavailable", "auth_unavailable")

    try:
        payload = pyjwt.decode(
            token, JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "jti", "exp", "type"]},
        )
    except pyjwt.ExpiredSignatureError:
        raise TokenError("Access token has expired", "token_expired")
    except pyjwt.InvalidTokenError as exc:
        raise TokenError(f"Invalid token: {exc}", "invalid_token")

    if payload.get("type") != "access":
        raise TokenError("Expected access token, got refresh token", "wrong_token_type")

    # Blocklist check
    jti = payload.get("jti", "")
    if jti and _is_revoked(jti):
        raise TokenError("Token has been revoked", "token_revoked")

    return TokenClaims(payload)


def decode_refresh_token(token: str) -> TokenClaims:
    """Decode and validate a refresh token."""
    if not _JWT_AVAILABLE:
        raise TokenError("Authentication service unavailable", "auth_unavailable")

    try:
        payload = pyjwt.decode(
            token, JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["sub", "jti", "exp", "type"]},
        )
    except pyjwt.ExpiredSignatureError:
        raise TokenError("Refresh token has expired", "token_expired")
    except pyjwt.InvalidTokenError as exc:
        raise TokenError(f"Invalid refresh token: {exc}", "invalid_token")

    if payload.get("type") != "refresh":
        raise TokenError("Expected refresh token", "wrong_token_type")

    jti = payload.get("jti", "")
    if jti and _is_revoked(jti):
        raise TokenError("Refresh token has been revoked", "token_revoked")

    return TokenClaims(payload)


# ════════════════════════════════════════════════════════════════════════════
#  Revocation
# ════════════════════════════════════════════════════════════════════════════

def _is_revoked(jti: str) -> bool:
    """Check if a jti is in the Redis revocation blocklist."""
    r = _get_redis()
    if r is None:
        return False
    try:
        return bool(r.get(f"revoked_jti:{jti}"))
    except Exception:
        return False


def revoke_token(jti: str, ttl_seconds: int) -> bool:
    """
    Add a jti to the revocation blocklist with the given TTL.
    Returns True if successfully revoked, False if Redis unavailable.
    """
    r = _get_redis()
    if r is None:
        logger.warning("Token revocation skipped — Redis unavailable (fail-open)")
        return False
    try:
        r.setex(f"revoked_jti:{jti}", ttl_seconds, "1")
        return True
    except Exception as exc:
        logger.warning("Token revocation failed: %s", exc)
        return False


def revoke_all_user_tokens(user_id: str) -> None:
    """
    Revoke all tokens for a user by setting a per-user revocation timestamp.
    Any token issued before this timestamp is considered revoked.
    This is a simpler alternative to tracking individual JTIs.
    """
    r = _get_redis()
    if r is None:
        logger.warning("Bulk token revocation skipped — Redis unavailable")
        return
    try:
        key = f"revoked_user:{user_id}"
        r.set(key, datetime.now(timezone.utc).isoformat(), ex=JWT_REFRESH_TOKEN_TTL)
    except Exception as exc:
        logger.warning("Bulk token revocation failed: %s", exc)
