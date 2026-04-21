"""auth/__init__.py — Authentication + authorisation package  (Day 8)"""
from auth.jwt_handler import (
    create_token_pair, decode_access_token, decode_refresh_token,
    revoke_token, revoke_all_user_tokens,
    TokenPair, TokenClaims, TokenError,
    JWT_ACCESS_TOKEN_TTL, JWT_REFRESH_TOKEN_TTL, _JWT_AVAILABLE,
)
from auth.abac import (
    Action, Decision, Rule, PolicyEngine,
    DEFAULT_POLICY, get_policy_engine, evaluate as abac_evaluate,
)
from auth.middleware import (
    require_auth, require_role, require_experiment_access, get_current_user,
    AUTH_ENABLED, AUTH_OPTIONAL,
)

__all__ = [
    "create_token_pair", "decode_access_token", "decode_refresh_token",
    "revoke_token", "revoke_all_user_tokens",
    "TokenPair", "TokenClaims", "TokenError",
    "JWT_ACCESS_TOKEN_TTL", "JWT_REFRESH_TOKEN_TTL", "_JWT_AVAILABLE",
    "Action", "Decision", "Rule", "PolicyEngine",
    "DEFAULT_POLICY", "get_policy_engine", "abac_evaluate",
    "require_auth", "require_role", "require_experiment_access", "get_current_user",
    "AUTH_ENABLED", "AUTH_OPTIONAL",
]
