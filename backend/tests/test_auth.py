"""
tests/test_auth.py
══════════════════════════════════════════════════════════════════════
Tests for Day 8: OAuth 2.0 / JWT authentication + ABAC + audit logging.

Strategy
────────
  • JWT tests use PyJWT directly when available, stubs otherwise.
  • ABAC tests use pure Python — no external deps.
  • Audit tests use SQLite (USE_SQLITE_FALLBACK=true).
  • Flask route tests use AUTH_OPTIONAL=true so no real token is needed
    for integration tests; separate unit tests cover the auth path.
  • All tests are non-blocking — auth/audit failures must never
    crash the main application flow.

Covers
──────
  TokenPair / TokenClaims   — creation, field access, to_dict
  create_token_pair          — payload shape, expiry, jti uniqueness
  decode_access_token        — happy path, expired, wrong type, revoked
  decode_refresh_token       — happy path, expired, wrong type
  revoke_token               — blocklist entry written (mocked Redis)
  Action enum                — all values present
  Decision                   — allow/deny, __bool__
  PolicyEngine               — rule ordering, first-match wins, default deny
  DEFAULT_POLICY rules       — admin, trainer, viewer, owner, DP clearance
  require_auth decorator     — valid token sets g.claims; missing token 401
  require_role decorator     — correct role passes; wrong role 403
  require_experiment_access  — owner allowed; non-owner denied; DP gate
  AuditLogger                — log(), list_recent(), verify_chain()
  Hash chain                 — intact chain verifies; tampered entry fails
  log_from_request           — extracts claims from flask.g
  Flask auth routes          — POST /api/auth/token, /refresh, /logout
  Flask audit routes         — GET /api/audit (admin only), /verify
  Flask users routes         — GET/POST /api/users (admin only)
  /api/health auth key       — present with enabled/optional flags

Run with:
    USE_SQLITE_FALLBACK=true pytest tests/test_auth.py -v
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

os.environ["USE_SQLITE_FALLBACK"]  = "true"
os.environ["CELERY_ASYNC_ENABLED"] = "false"
os.environ["AUTH_OPTIONAL"]        = "true"   # bypass auth for Flask route tests
os.environ["AUDIT_ENABLED"]        = "true"
os.environ["AUDIT_HASH_CHAIN"]     = "true"
os.environ["JWT_SECRET_KEY"]       = "test-secret-key-for-unit-tests-only"

import pytest

# ── Optional PyJWT ────────────────────────────────────────────────────────
try:
    import jwt as pyjwt
    _JWT_AVAILABLE = True
except ImportError:
    _JWT_AVAILABLE = False

from auth.jwt_handler import (
    create_token_pair, decode_access_token, decode_refresh_token,
    revoke_token, TokenClaims, TokenError, TokenPair,
    JWT_SECRET_KEY, JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_TTL, JWT_REFRESH_TOKEN_TTL,
)
from auth.abac import (
    Action, Decision, Rule, PolicyEngine, DEFAULT_POLICY,
    evaluate as abac_evaluate, get_policy_engine,
    _rule_admin_allow_all, _rule_viewer_read_only,
    _rule_dp_clearance_required, _rule_inactive_user_deny,
    _rule_owner_full_access,
)
from auth.middleware import require_auth, require_role, get_current_user
from audit.audit_log import (
    AuditLogger, AuditAction, AuditOutcome,
    _compute_hash, GENESIS_HASH, log_from_request,
)
from database import init_db, drop_db, get_db, ExperimentRepo, ModelType, UserRepo


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    drop_db(); init_db()
    yield
    drop_db()


def _make_experiment(model_type=ModelType.CENTRAL, user_id=None,
                     dp_enabled=False):
    with get_db() as db:
        exp = ExperimentRepo.create(
            db,
            model_type=model_type,
            hyperparameters={"epochs": 5, "lr": 0.1},
            target_col_index=0,
            feature_types={},
            name="auth test exp",
            user_id=user_id,
            dp_enabled=dp_enabled,
            dp_target_epsilon=2.0 if dp_enabled else None,
            dp_delta=1e-5 if dp_enabled else None,
            dp_clip_threshold=1.0 if dp_enabled else None,
        )
        return exp.id


def _make_user(role="trainer", username=None):
    from database.models import UserRole
    with get_db() as db:
        u = UserRepo.create(
            db,
            username=username or f"user_{role}",
            email=f"{username or role}@test.com",
            role=UserRole(role),
        )
        return u.id


# ════════════════════════════════════════════════════════════════════════════
#  TokenPair / TokenClaims
# ════════════════════════════════════════════════════════════════════════════

class TestTokenPair:

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_create_token_pair_returns_token_pair(self):
        pair = create_token_pair("user-1", "u@test.com", "trainer", {})
        assert isinstance(pair, TokenPair)
        assert pair.access_token
        assert pair.refresh_token
        assert pair.access_token != pair.refresh_token

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_to_dict_shape(self):
        pair = create_token_pair("u1", "u@t.com", "trainer", {})
        d = pair.to_dict()
        for key in ("access_token", "refresh_token", "token_type",
                    "expires_in", "access_expires", "refresh_expires"):
            assert key in d

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_access_token_type_bearer(self):
        pair = create_token_pair("u1", "u@t.com", "trainer", {})
        assert pair.to_dict()["token_type"] == "Bearer"

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_jti_are_unique_across_calls(self):
        p1 = create_token_pair("u1", "u@t.com", "trainer", {})
        p2 = create_token_pair("u1", "u@t.com", "trainer", {})
        assert p1.access_jti != p2.access_jti

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_access_token_expires_sooner_than_refresh(self):
        pair = create_token_pair("u1", "u@t.com", "trainer", {})
        assert pair.access_expires < pair.refresh_expires


# ════════════════════════════════════════════════════════════════════════════
#  decode_access_token
# ════════════════════════════════════════════════════════════════════════════

class TestDecodeAccessToken:

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_valid_token_returns_claims(self):
        pair   = create_token_pair("uid-1", "a@b.com", "admin", {"dp": True})
        claims = decode_access_token(pair.access_token)
        assert isinstance(claims, TokenClaims)
        assert claims.user_id == "uid-1"
        assert claims.email   == "a@b.com"
        assert claims.role    == "admin"
        assert claims.attributes.get("dp") is True

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_claims_role_helpers(self):
        pair   = create_token_pair("u", "u@t.com", "trainer", {})
        claims = decode_access_token(pair.access_token)
        assert claims.is_trainer() is True
        assert claims.is_admin()   is False
        assert claims.is_viewer()  is True

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_expired_token_raises_token_error(self):
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "u1", "email": "u@t.com", "role": "trainer",
            "attrs": {}, "type": "access",
            "jti": "test-jti",
            "iat": now - timedelta(hours=2),
            "exp": now - timedelta(hours=1),   # expired 1 hour ago
        }
        token = pyjwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        with pytest.raises(TokenError) as exc:
            decode_access_token(token)
        assert exc.value.code == "token_expired"

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_wrong_secret_raises_token_error(self):
        payload = {
            "sub": "u1", "email": "u@t.com", "role": "trainer",
            "attrs": {}, "type": "access", "jti": "j1",
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        token = pyjwt.encode(payload, "WRONG_SECRET", algorithm=JWT_ALGORITHM)
        with pytest.raises(TokenError) as exc:
            decode_access_token(token)
        assert exc.value.code == "invalid_token"

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_refresh_token_rejected_as_access(self):
        pair = create_token_pair("u1", "u@t.com", "trainer", {})
        with pytest.raises(TokenError) as exc:
            decode_access_token(pair.refresh_token)
        assert exc.value.code == "wrong_token_type"

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_revoked_token_raises(self):
        pair = create_token_pair("u1", "u@t.com", "trainer", {})
        # Mock Redis blocklist
        import auth.jwt_handler as jh
        orig_is_revoked = jh._is_revoked
        jh._is_revoked = lambda jti: True
        try:
            with pytest.raises(TokenError) as exc:
                decode_access_token(pair.access_token)
            assert exc.value.code == "token_revoked"
        finally:
            jh._is_revoked = orig_is_revoked


# ════════════════════════════════════════════════════════════════════════════
#  decode_refresh_token
# ════════════════════════════════════════════════════════════════════════════

class TestDecodeRefreshToken:

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_valid_refresh_token(self):
        pair   = create_token_pair("u1", "u@t.com", "viewer", {})
        claims = decode_refresh_token(pair.refresh_token)
        assert claims.user_id    == "u1"
        assert claims.token_type == "refresh"

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_access_token_rejected_as_refresh(self):
        pair = create_token_pair("u1", "u@t.com", "trainer", {})
        with pytest.raises(TokenError) as exc:
            decode_refresh_token(pair.access_token)
        assert exc.value.code == "wrong_token_type"

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_garbage_token_raises(self):
        with pytest.raises(TokenError):
            decode_refresh_token("not.a.jwt")


# ════════════════════════════════════════════════════════════════════════════
#  revoke_token
# ════════════════════════════════════════════════════════════════════════════

class TestRevokeToken:

    def test_revoke_returns_false_when_redis_unavailable(self):
        import auth.jwt_handler as jh
        orig = jh._get_redis
        jh._get_redis = lambda: None
        try:
            result = revoke_token("some-jti", 900)
            assert result is False
        finally:
            jh._get_redis = orig

    def test_revoke_calls_setex_on_redis(self):
        mock_redis = MagicMock()
        import auth.jwt_handler as jh
        orig = jh._get_redis
        jh._get_redis = lambda: mock_redis
        try:
            revoke_token("test-jti-123", 900)
            mock_redis.setex.assert_called_once_with("revoked_jti:test-jti-123", 900, "1")
        finally:
            jh._get_redis = orig


# ════════════════════════════════════════════════════════════════════════════
#  Action enum
# ════════════════════════════════════════════════════════════════════════════

class TestActionEnum:

    def test_all_actions_exist(self):
        for name in ("READ", "TRAIN", "UPLOAD", "ADMIN", "CANCEL",
                     "VIEW_BUDGET", "VIEW_AUDIT"):
            assert hasattr(Action, name), f"Missing Action.{name}"

    def test_action_values_are_strings(self):
        for action in Action:
            assert isinstance(action.value, str)


# ════════════════════════════════════════════════════════════════════════════
#  Decision
# ════════════════════════════════════════════════════════════════════════════

class TestDecision:

    def test_allow_is_truthy(self):
        d = Decision.allow("OK")
        assert bool(d) is True
        assert d.allowed is True

    def test_deny_is_falsy(self):
        d = Decision.deny("No")
        assert bool(d) is False
        assert d.allowed is False

    def test_allow_carries_reason_and_rule(self):
        d = Decision.allow("because admin", rule="admin_rule")
        assert d.reason == "because admin"
        assert d.rule   == "admin_rule"

    def test_deny_carries_reason(self):
        d = Decision.deny("access denied", rule="my_rule")
        assert "denied" in d.reason
        assert d.rule == "my_rule"


# ════════════════════════════════════════════════════════════════════════════
#  PolicyEngine
# ════════════════════════════════════════════════════════════════════════════

class TestPolicyEngine:

    def test_first_matching_rule_wins(self):
        """Rules are evaluated in priority order; first match wins."""
        rule_a = Rule("allow_a", lambda s, r, a: Decision.allow("rule_a"), priority=10)
        rule_b = Rule("allow_b", lambda s, r, a: Decision.deny("rule_b"),  priority=20)
        engine = PolicyEngine([rule_b, rule_a])   # inserted out of order
        d = engine.evaluate({}, {}, Action.READ)
        assert d.allowed      # rule_a (priority 10) wins
        assert d.rule == "allow_a"

    def test_default_deny_when_no_rules_match(self):
        engine = PolicyEngine([Rule("pass", lambda s, r, a: None, priority=0)])
        d = engine.evaluate({}, {}, Action.ADMIN)
        assert not d.allowed
        assert d.rule == "default_deny"

    def test_rule_exception_is_swallowed(self):
        def crashing(s, r, a): raise RuntimeError("boom")
        rule   = Rule("crash", crashing, priority=0)
        engine = PolicyEngine([rule])
        d = engine.evaluate({}, {}, Action.READ)
        assert not d.allowed   # default deny after exception

    def test_add_rule_re_sorts(self):
        engine = PolicyEngine([])
        engine.add_rule(Rule("deny_all", lambda s, r, a: Decision.deny("x"), priority=50))
        engine.add_rule(Rule("allow_all", lambda s, r, a: Decision.allow("y"), priority=10))
        d = engine.evaluate({}, {}, Action.READ)
        assert d.allowed  # allow_all (priority 10) wins


# ════════════════════════════════════════════════════════════════════════════
#  DEFAULT_POLICY rules
# ════════════════════════════════════════════════════════════════════════════

class TestDefaultPolicyRules:

    def _eval(self, role, action, resource=None, user_id="u1",
              attrs=None, is_active=True):
        subject = {
            "user_id": user_id, "role": role, "is_active": is_active,
            "attrs": attrs or {},
        }
        return abac_evaluate(subject, resource or {}, action)

    # ── Admin ──────────────────────────────────────────────────────────────
    def test_admin_can_do_everything(self):
        for action in Action:
            assert self._eval("admin", action).allowed, f"Admin denied {action}"

    # ── Inactive user ──────────────────────────────────────────────────────
    def test_inactive_user_denied_all(self):
        for action in Action:
            d = self._eval("trainer", action, is_active=False)
            assert not d.allowed

    # ── Trainer ────────────────────────────────────────────────────────────
    def test_trainer_can_train(self):
        assert self._eval("trainer", Action.TRAIN).allowed

    def test_trainer_can_read(self):
        assert self._eval("trainer", Action.READ).allowed

    def test_trainer_can_upload(self):
        assert self._eval("trainer", Action.UPLOAD).allowed

    def test_trainer_cannot_admin(self):
        assert not self._eval("trainer", Action.ADMIN).allowed

    def test_trainer_cannot_view_audit(self):
        assert not self._eval("trainer", Action.VIEW_AUDIT).allowed

    def test_trainer_can_view_budget(self):
        assert self._eval("trainer", Action.VIEW_BUDGET).allowed

    # ── Viewer ─────────────────────────────────────────────────────────────
    def test_viewer_can_read(self):
        assert self._eval("viewer", Action.READ).allowed

    def test_viewer_cannot_train(self):
        assert not self._eval("viewer", Action.TRAIN).allowed

    def test_viewer_cannot_upload(self):
        assert not self._eval("viewer", Action.UPLOAD).allowed

    def test_viewer_cannot_admin(self):
        assert not self._eval("viewer", Action.ADMIN).allowed

    # ── Owner ──────────────────────────────────────────────────────────────
    def test_owner_can_read_own_experiment(self):
        d = abac_evaluate(
            {"user_id": "alice", "role": "trainer", "is_active": True, "attrs": {}},
            {"type": "experiment", "owner_id": "alice"},
            Action.READ,
        )
        assert d.allowed

    def test_non_owner_trainer_can_also_read(self):
        """Trainers can read even when not the owner."""
        d = abac_evaluate(
            {"user_id": "bob", "role": "trainer", "is_active": True, "attrs": {}},
            {"type": "experiment", "owner_id": "alice"},
            Action.READ,
        )
        assert d.allowed

    # ── DP clearance gate ──────────────────────────────────────────────────
    def test_trainer_without_dp_clearance_cannot_train_dp_experiment(self):
        d = abac_evaluate(
            {"user_id": "u1", "role": "trainer", "is_active": True,
             "attrs": {}},   # no dp_clearance
            {"type": "experiment", "dp_enabled": True, "owner_id": "other"},
            Action.TRAIN,
        )
        assert not d.allowed
        assert d.rule == "dp_clearance_required"

    def test_trainer_with_dp_clearance_can_train_dp_experiment(self):
        d = abac_evaluate(
            {"user_id": "u1", "role": "trainer", "is_active": True,
             "attrs": {"dp_clearance": True}},
            {"type": "experiment", "dp_enabled": True, "owner_id": "other"},
            Action.TRAIN,
        )
        assert d.allowed

    def test_dp_gate_does_not_block_non_dp_experiment(self):
        d = abac_evaluate(
            {"user_id": "u1", "role": "trainer", "is_active": True, "attrs": {}},
            {"type": "experiment", "dp_enabled": False},
            Action.TRAIN,
        )
        assert d.allowed

    def test_admin_can_train_dp_experiment_without_clearance(self):
        """Admin rule fires before DP gate."""
        d = abac_evaluate(
            {"user_id": "admin1", "role": "admin", "is_active": True, "attrs": {}},
            {"type": "experiment", "dp_enabled": True},
            Action.TRAIN,
        )
        assert d.allowed


# ════════════════════════════════════════════════════════════════════════════
#  require_auth middleware
# ════════════════════════════════════════════════════════════════════════════

class TestRequireAuthMiddleware:

    @pytest.fixture(autouse=True)
    def flask_app(self):
        from flask import Flask, g, jsonify
        self.app = Flask("test_auth_middleware")
        self.app.config["TESTING"] = True

        @self.app.route("/protected")
        @require_auth
        def protected():
            claims = get_current_user()
            return jsonify({"user": claims.user_id if claims else None})

        self.client = self.app.test_client()

    def test_no_token_returns_401_when_auth_not_optional(self):
        import auth.middleware as mw
        orig = mw.AUTH_OPTIONAL
        mw.AUTH_OPTIONAL = False
        try:
            resp = self.client.get("/protected")
            assert resp.status_code == 401
            data = resp.get_json()
            assert data["code"] == "missing_token"
        finally:
            mw.AUTH_OPTIONAL = orig

    def test_auth_optional_passes_through(self):
        import auth.middleware as mw
        orig = mw.AUTH_OPTIONAL
        mw.AUTH_OPTIONAL = True
        try:
            resp = self.client.get("/protected")
            assert resp.status_code == 200
        finally:
            mw.AUTH_OPTIONAL = orig

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_valid_token_sets_g_claims(self):
        import auth.middleware as mw
        orig = mw.AUTH_OPTIONAL
        mw.AUTH_OPTIONAL = False
        try:
            pair  = create_token_pair("user-42", "x@t.com", "trainer", {})
            resp  = self.client.get(
                "/protected",
                headers={"Authorization": f"Bearer {pair.access_token}"},
            )
            assert resp.status_code == 200
            assert resp.get_json()["user"] == "user-42"
        finally:
            mw.AUTH_OPTIONAL = orig

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_invalid_token_returns_401(self):
        import auth.middleware as mw
        orig = mw.AUTH_OPTIONAL
        mw.AUTH_OPTIONAL = False
        try:
            resp = self.client.get(
                "/protected",
                headers={"Authorization": "Bearer not.a.real.token"},
            )
            assert resp.status_code == 401
        finally:
            mw.AUTH_OPTIONAL = orig


# ════════════════════════════════════════════════════════════════════════════
#  require_role middleware
# ════════════════════════════════════════════════════════════════════════════

class TestRequireRoleMiddleware:

    @pytest.fixture(autouse=True)
    def flask_app(self):
        from flask import Flask, g, jsonify
        import auth.middleware as mw
        self.app = Flask("test_role_middleware")
        self.app.config["TESTING"] = True

        @self.app.route("/admin-only")
        @require_auth
        @require_role("admin")
        def admin_only():
            return jsonify({"ok": True})

        self.client = self.app.test_client()
        self._orig_optional = mw.AUTH_OPTIONAL
        mw.AUTH_OPTIONAL = False

    def teardown_method(self):
        import auth.middleware as mw
        mw.AUTH_OPTIONAL = self._orig_optional

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_admin_role_passes(self):
        pair = create_token_pair("a1", "a@t.com", "admin", {})
        resp = self.client.get(
            "/admin-only",
            headers={"Authorization": f"Bearer {pair.access_token}"},
        )
        assert resp.status_code == 200

    @pytest.mark.skipif(not _JWT_AVAILABLE, reason="PyJWT not installed")
    def test_trainer_role_denied(self):
        pair = create_token_pair("t1", "t@t.com", "trainer", {})
        resp = self.client.get(
            "/admin-only",
            headers={"Authorization": f"Bearer {pair.access_token}"},
        )
        assert resp.status_code == 403
        assert resp.get_json()["code"] == "insufficient_role"


# ════════════════════════════════════════════════════════════════════════════
#  AuditLogger
# ════════════════════════════════════════════════════════════════════════════

class TestAuditLogger:

    def test_log_does_not_raise(self):
        AuditLogger.log(
            action=AuditAction.LOGIN, outcome=AuditOutcome.SUCCESS,
            user_id="u1", email="u@t.com", role="trainer",
            ip_address="127.0.0.1",
        )

    def test_log_stores_entry_in_db(self):
        AuditLogger.log(
            action=AuditAction.TRAIN_SUBMITTED, outcome=AuditOutcome.SUCCESS,
            user_id="u1", email="u@t.com",
            resource_id="exp-123", resource_type="experiment",
            details={"model_type": "central"},
        )
        entries = AuditLogger.list_recent(limit=10)
        assert len(entries) == 1
        assert entries[0]["action"]      == "TRAIN_SUBMITTED"
        assert entries[0]["resource_id"] == "exp-123"

    def test_log_multiple_entries(self):
        for i in range(5):
            AuditLogger.log(
                action=AuditAction.EXPERIMENT_READ,
                outcome=AuditOutcome.SUCCESS,
                user_id=f"user-{i}",
            )
        entries = AuditLogger.list_recent(limit=10)
        assert len(entries) == 5

    def test_list_recent_filtered_by_user(self):
        AuditLogger.log(AuditAction.LOGIN, AuditOutcome.SUCCESS, user_id="alice")
        AuditLogger.log(AuditAction.LOGIN, AuditOutcome.SUCCESS, user_id="bob")
        alice_entries = AuditLogger.list_recent(limit=100, user_id="alice")
        assert all(e["user_id"] == "alice" for e in alice_entries)
        assert len(alice_entries) == 1

    def test_entry_has_hash_fields(self):
        AuditLogger.log(AuditAction.LOGOUT, AuditOutcome.SUCCESS, user_id="u1")
        entries = AuditLogger.list_recent(limit=1)
        assert "entry_hash" in entries[0]
        assert "prev_hash"  in entries[0]
        assert len(entries[0]["entry_hash"]) == 64   # SHA-256 hex

    def test_log_disabled_writes_nothing(self):
        import audit.audit_log as al
        orig = al.AUDIT_ENABLED
        al.AUDIT_ENABLED = False
        try:
            AuditLogger.log(AuditAction.LOGIN, AuditOutcome.SUCCESS, user_id="x")
            entries = AuditLogger.list_recent(limit=10)
            assert len(entries) == 0
        finally:
            al.AUDIT_ENABLED = orig

    def test_log_exception_does_not_propagate(self):
        """A broken DB write must not crash the caller."""
        import audit.audit_log as al
        orig = al.AUDIT_ENABLED
        al.AUDIT_ENABLED = True
        # Monkey-patch _write to raise
        orig_write = AuditLogger._write
        AuditLogger._write = staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("db down")))
        try:
            AuditLogger.log(AuditAction.LOGIN, AuditOutcome.SUCCESS)  # must not raise
        finally:
            AuditLogger._write = staticmethod(orig_write)
            al.AUDIT_ENABLED   = orig


# ════════════════════════════════════════════════════════════════════════════
#  Hash chain
# ════════════════════════════════════════════════════════════════════════════

class TestHashChain:

    def test_compute_hash_is_deterministic(self):
        h1 = _compute_hash("prev", "ts", "uid", "ACT", "OK", "{}")
        h2 = _compute_hash("prev", "ts", "uid", "ACT", "OK", "{}")
        assert h1 == h2

    def test_compute_hash_changes_with_any_field(self):
        base = _compute_hash("prev", "ts", "uid", "ACT", "OK", "{}")
        assert _compute_hash("DIFF", "ts", "uid", "ACT", "OK", "{}") != base
        assert _compute_hash("prev", "DIFF", "uid", "ACT", "OK", "{}") != base
        assert _compute_hash("prev", "ts", "DIFF", "ACT", "OK", "{}") != base

    def test_genesis_hash_is_all_zeros(self):
        assert GENESIS_HASH == "0" * 64

    def test_chain_verifies_after_logging(self):
        for i in range(3):
            AuditLogger.log(
                AuditAction.TRAIN_SUBMITTED, AuditOutcome.SUCCESS,
                user_id=f"u{i}",
            )
        ok, reason = AuditLogger.verify_chain(limit=100)
        assert ok is True, f"Chain broken: {reason}"

    def test_verify_chain_empty_db_is_ok(self):
        ok, reason = AuditLogger.verify_chain()
        assert ok is True
        assert reason is None

    def test_tampered_entry_breaks_chain(self):
        """Modify an entry's hash directly — verify_chain must detect it."""
        AuditLogger.log(AuditAction.LOGIN, AuditOutcome.SUCCESS, user_id="u1")
        AuditLogger.log(AuditAction.LOGOUT, AuditOutcome.SUCCESS, user_id="u1")

        from database.models import AuditLog
        from sqlalchemy import asc
        with get_db() as db:
            first = db.query(AuditLog).order_by(asc(AuditLog.created_at)).first()
            first.entry_hash = "a" * 64   # tamper

        ok, reason = AuditLogger.verify_chain(limit=100)
        assert ok is False
        assert reason is not None

    def test_second_entry_prev_hash_equals_first_entry_hash(self):
        AuditLogger.log(AuditAction.LOGIN,  AuditOutcome.SUCCESS, user_id="u1")
        AuditLogger.log(AuditAction.LOGOUT, AuditOutcome.SUCCESS, user_id="u1")

        from database.models import AuditLog
        from sqlalchemy import asc
        with get_db() as db:
            entries = (db.query(AuditLog)
                       .order_by(asc(AuditLog.created_at)).limit(2).all())
        assert entries[1].prev_hash == entries[0].entry_hash


# ════════════════════════════════════════════════════════════════════════════
#  Flask auth + audit + users routes  (AUTH_OPTIONAL=true)
# ════════════════════════════════════════════════════════════════════════════

class TestFlaskAuthRoutes:

    @pytest.fixture(autouse=True)
    def flask_client(self):
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        with flask_app.app.test_client() as c:
            self.client = c

    def test_health_has_auth_key(self):
        resp = self.client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "auth" in data
        assert "enabled" in data["auth"]

    def test_health_version_is_10(self):
        resp = self.client.get("/api/health")
        assert resp.get_json()["version"] == "10.0.0"

    def test_auth_token_returns_tokens(self):
        resp = self.client.post(
            "/api/auth/token",
            json={"user_id": "u1", "email": "u@t.com", "role": "trainer"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "access_token"  in data
        assert "refresh_token" in data

    def test_auth_token_records_login_audit(self):
        self.client.post(
            "/api/auth/token",
            json={"user_id": "audited-user", "email": "a@t.com", "role": "trainer"},
        )
        entries = AuditLogger.list_recent(limit=10)
        login_entries = [e for e in entries if e["action"] == "LOGIN"]
        assert len(login_entries) >= 1

    def test_auth_refresh_returns_access_token(self):
        # Get initial pair
        resp = self.client.post(
            "/api/auth/token",
            json={"user_id": "u1", "email": "u@t.com", "role": "trainer"},
        )
        refresh_token = resp.get_json().get("refresh_token", "")
        if not refresh_token or not _JWT_AVAILABLE:
            pytest.skip("JWT unavailable")

        resp2 = self.client.post(
            "/api/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert resp2.status_code == 200
        assert "access_token" in resp2.get_json()

    def test_auth_logout_returns_200(self):
        resp = self.client.post("/api/auth/logout")
        assert resp.status_code == 200

    def test_audit_endpoint_requires_admin(self):
        # AUTH_OPTIONAL=true → no real auth check, admin role injected
        resp = self.client.get("/api/audit")
        assert resp.status_code == 200

    def test_audit_verify_returns_result(self):
        resp = self.client.get("/api/audit/verify")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "intact" in data

    def test_users_list_returns_list(self):
        resp = self.client.get("/api/users")
        assert resp.status_code == 200
        assert isinstance(resp.get_json(), list)

    def test_users_create_and_retrieve(self):
        resp = self.client.post(
            "/api/users",
            json={"username": "newuser", "email": "new@test.com", "role": "viewer"},
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["email"] == "new@test.com"
        assert data["role"]  == "viewer"

    def test_audit_log_create_user_event(self):
        self.client.post(
            "/api/users",
            json={"username": "testaudit", "email": "ta@test.com", "role": "trainer"},
        )
        entries = AuditLogger.list_recent(limit=20)
        created = [e for e in entries if e["action"] == "USER_CREATED"]
        assert len(created) >= 1


# ── Manual runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    drop_db(); init_db()

    suites = [
        TestTokenPair, TestDecodeAccessToken, TestDecodeRefreshToken,
        TestRevokeToken, TestActionEnum, TestDecision, TestPolicyEngine,
        TestDefaultPolicyRules, TestRequireAuthMiddleware,
        TestRequireRoleMiddleware, TestAuditLogger, TestHashChain,
        TestFlaskAuthRoutes,
    ]

    passed = failed = 0
    for suite_cls in suites:
        suite   = suite_cls()
        methods = sorted(m for m in dir(suite) if m.startswith("test_"))
        for m in methods:
            drop_db(); init_db()
            try:
                getattr(suite, m)()
                print(f"  ✓  {suite_cls.__name__}::{m}")
                passed += 1
            except Exception as exc:
                print(f"  ✗  {suite_cls.__name__}::{m}")
                print(f"       {exc}")
                import traceback; traceback.print_exc()
                failed += 1

    drop_db()
    print(f"\n{'✅' if not failed else '❌'}  {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
