"""
auth/abac.py
══════════════════════════════════════════════════════════════════════
Attribute-Based Access Control  (Day 8)

Replaces the 3-tier role system (admin/trainer/viewer) with a
flexible policy engine where access decisions are based on:
  • Subject attributes  — who is the user? (role, department, clearance)
  • Resource attributes — what are they accessing? (experiment, dp_enabled)
  • Action              — what are they trying to do? (read, train, admin)
  • Environment         — context (time, IP, etc.) — extensible

Policy structure
────────────────
  A Policy is a list of Rules.  A Rule matches a subject+resource+action
  triple and returns ALLOW or DENY.  Rules are evaluated in order;
  the first matching rule wins (deny-by-default if no rule matches).

Built-in policies
─────────────────
  DEFAULT_POLICY — implements the original 3-tier RBAC as ABAC rules:
    • admin  → ALLOW all actions on all resources
    • trainer → ALLOW train + read; DENY admin actions
    • viewer  → ALLOW read only

  PER_EXPERIMENT_POLICY — experiment-level access:
    • owner (user_id == experiment.user_id) → ALLOW all
    • trainer with matching department → ALLOW train
    • viewer → ALLOW read
    • dp_required experiments → DENY if user lacks dp_clearance attribute

Usage
─────
    from auth.abac import PolicyEngine, Action, DEFAULT_POLICY

    engine = PolicyEngine(DEFAULT_POLICY)
    decision = engine.evaluate(
        subject={"role": "trainer", "department": "cardiology"},
        resource={"type": "experiment", "model_type": "dp_federated",
                  "dp_enabled": True, "owner_id": "uuid-..."},
        action=Action.TRAIN,
    )
    if not decision.allowed:
        abort(403, decision.reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Actions
# ════════════════════════════════════════════════════════════════════════════

class Action(str, Enum):
    READ        = "read"        # GET /api/experiments, /api/files, etc.
    TRAIN       = "train"       # POST /api/train/*
    UPLOAD      = "upload"      # POST /api/upload/*
    ADMIN       = "admin"       # user management, audit log access
    CANCEL      = "cancel"      # POST /api/jobs/*/cancel
    VIEW_BUDGET = "view_budget" # GET /api/privacy/budget/*
    VIEW_AUDIT  = "view_audit"  # GET /api/audit/*


# ════════════════════════════════════════════════════════════════════════════
#  Decision
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Decision:
    allowed: bool
    reason:  str
    rule:    Optional[str] = None   # name of the rule that matched

    @classmethod
    def allow(cls, reason: str = "Allowed by policy",
              rule: Optional[str] = None) -> "Decision":
        return cls(allowed=True, reason=reason, rule=rule)

    @classmethod
    def deny(cls, reason: str = "Denied by policy",
             rule: Optional[str] = None) -> "Decision":
        return cls(allowed=False, reason=reason, rule=rule)

    def __bool__(self) -> bool:
        return self.allowed


# ════════════════════════════════════════════════════════════════════════════
#  Rule
# ════════════════════════════════════════════════════════════════════════════

RuleFunc = Callable[[dict, dict, Action], Optional[Decision]]

@dataclass
class Rule:
    """
    A single ABAC rule.

    fn(subject, resource, action) → Decision | None
      Return None to pass (let the next rule evaluate).
      Return Decision to short-circuit (allow or deny).
    """
    name:     str
    fn:       RuleFunc
    priority: int = 0   # lower = higher priority; evaluated in ascending order


# ════════════════════════════════════════════════════════════════════════════
#  PolicyEngine
# ════════════════════════════════════════════════════════════════════════════

class PolicyEngine:
    """
    Evaluates a list of Rules in priority order.
    The first rule that returns a Decision wins.
    If no rule matches, defaults to DENY (secure by default).

    Usage:
        engine   = PolicyEngine(rules)
        decision = engine.evaluate(subject, resource, action)
    """

    def __init__(self, rules: List[Rule]) -> None:
        self._rules = sorted(rules, key=lambda r: r.priority)

    def evaluate(
        self,
        subject:  dict,
        resource: dict,
        action:   Action,
    ) -> Decision:
        """
        Evaluate all rules and return the first matching Decision.
        Defaults to DENY if no rule matches.

        subject  — dict with keys: role, user_id, department, attrs, …
        resource — dict with keys: type, owner_id, dp_enabled, model_type, …
        action   — Action enum value
        """
        for rule in self._rules:
            try:
                result = rule.fn(subject, resource, action)
                if result is not None:
                    logger.debug(
                        "ABAC: rule=%r  action=%s  allowed=%s  reason=%r",
                        rule.name, action.value, result.allowed, result.reason,
                    )
                    return result
            except Exception as exc:
                logger.warning("ABAC rule %r raised exception: %s", rule.name, exc)
                continue

        logger.debug("ABAC: no rule matched — deny by default  action=%s", action.value)
        return Decision.deny(
            "No matching policy rule — deny by default",
            rule="default_deny",
        )

    def add_rule(self, rule: Rule) -> None:
        """Dynamically add a rule and re-sort."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)


# ════════════════════════════════════════════════════════════════════════════
#  Built-in rules
# ════════════════════════════════════════════════════════════════════════════

def _rule_admin_allow_all(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Admins can do everything."""
    if subject.get("role") == "admin":
        return Decision.allow("Admin role: unrestricted access", "admin_allow_all")
    return None


def _rule_inactive_user_deny(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Inactive users are denied everything."""
    if subject.get("is_active") is False:
        return Decision.deny("User account is inactive", "inactive_user_deny")
    return None


def _rule_trainer_can_train(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Trainers can submit training jobs and upload files."""
    if subject.get("role") == "trainer" and action in (Action.TRAIN, Action.UPLOAD):
        return Decision.allow("Trainer role: training allowed", "trainer_can_train")
    return None


def _rule_trainer_can_read(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Trainers can read experiments and results."""
    if subject.get("role") == "trainer" and action == Action.READ:
        return Decision.allow("Trainer role: read allowed", "trainer_can_read")
    return None


def _rule_trainer_can_cancel_own(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Trainers can cancel their own jobs."""
    if (subject.get("role") == "trainer"
            and action == Action.CANCEL
            and subject.get("user_id") == resource.get("owner_id")):
        return Decision.allow("Trainer: cancel own job", "trainer_cancel_own")
    return None


def _rule_viewer_read_only(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Viewers can only read."""
    if subject.get("role") == "viewer":
        if action == Action.READ:
            return Decision.allow("Viewer role: read-only access", "viewer_read")
        return Decision.deny(
            f"Viewer role cannot perform action: {action.value}",
            "viewer_deny_write",
        )
    return None


def _rule_trainer_deny_admin(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Trainers cannot perform admin actions."""
    if subject.get("role") == "trainer" and action == Action.ADMIN:
        return Decision.deny("Trainer role: admin action denied", "trainer_deny_admin")
    return None


def _rule_dp_clearance_required(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """
    DP experiments require dp_clearance attribute on the subject.
    Trainers without dp_clearance cannot train DP experiments.
    """
    if (action == Action.TRAIN
            and resource.get("dp_enabled") is True
            and subject.get("role") == "trainer"):
        attrs = subject.get("attrs") or subject.get("attributes") or {}
        if not attrs.get("dp_clearance", False):
            return Decision.deny(
                "DP-enabled experiment requires dp_clearance attribute",
                "dp_clearance_required",
            )
    return None


def _rule_owner_full_access(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Experiment owner has full access to their own experiment."""
    if (subject.get("user_id")
            and subject.get("user_id") == resource.get("owner_id")
            and action in (Action.READ, Action.TRAIN, Action.CANCEL,
                           Action.VIEW_BUDGET, Action.VIEW_AUDIT)):
        return Decision.allow("Resource owner: full access", "owner_full_access")
    return None


def _rule_view_budget_trainer_or_owner(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Privacy budget is readable by trainers and owners."""
    if action == Action.VIEW_BUDGET:
        if subject.get("role") in ("admin", "trainer"):
            return Decision.allow("Trainer/admin can view privacy budget", "view_budget_trainer")
    return None


def _rule_view_audit_admin_only(subject: dict, resource: dict, action: Action) -> Optional[Decision]:
    """Audit logs are admin-only."""
    if action == Action.VIEW_AUDIT:
        if subject.get("role") != "admin":
            return Decision.deny("Audit log access requires admin role", "audit_admin_only")
    return None


# ════════════════════════════════════════════════════════════════════════════
#  Default policy (ordered)
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_POLICY: List[Rule] = [
    Rule("inactive_user_deny",      _rule_inactive_user_deny,        priority=0),
    Rule("admin_allow_all",         _rule_admin_allow_all,           priority=10),
    Rule("owner_full_access",       _rule_owner_full_access,         priority=20),
    Rule("dp_clearance_required",   _rule_dp_clearance_required,     priority=25),
    Rule("trainer_deny_admin",      _rule_trainer_deny_admin,        priority=30),
    Rule("view_audit_admin_only",   _rule_view_audit_admin_only,     priority=35),
    Rule("view_budget_trainer",     _rule_view_budget_trainer_or_owner, priority=40),
    Rule("trainer_can_train",       _rule_trainer_can_train,         priority=50),
    Rule("trainer_can_read",        _rule_trainer_can_read,          priority=55),
    Rule("trainer_can_cancel_own",  _rule_trainer_can_cancel_own,    priority=60),
    Rule("viewer_read_only",        _rule_viewer_read_only,          priority=70),
]

# Module-level singleton engine using the default policy
_default_engine: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    """Return the module-level policy engine singleton."""
    global _default_engine
    if _default_engine is None:
        _default_engine = PolicyEngine(DEFAULT_POLICY)
    return _default_engine


def evaluate(subject: dict, resource: dict, action: Action) -> Decision:
    """Convenience wrapper using the default engine."""
    return get_policy_engine().evaluate(subject, resource, action)
