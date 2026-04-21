"""
ml/privacy.py
══════════════════════════════════════════════════════════════════════
Differential Privacy primitives — pure Python, zero extra dependencies.

Implements
──────────
  GaussianMechanism   — calibrate σ from (ε, δ, sensitivity) and inject noise
  GradientClipper     — per-sample L2 clipping to bound sensitivity
  MomentsAccountant   — tight ε accounting across T rounds via RDP composition
  PrivacyBudget       — mutable spend tracker (current_epsilon, remaining)

Theory
──────
  DP guarantee: A randomised mechanism M is (ε, δ)-DP if for any two
  adjacent datasets D, D' differing in one record, and any output S:

      Pr[M(D) ∈ S]  ≤  exp(ε) · Pr[M(D') ∈ S]  +  δ

  Gaussian mechanism: adding N(0, σ²I) to a function f with L2-sensitivity
  Δf satisfies (ε, δ)-DP when:

      σ  ≥  Δf · √(2 ln(1.25/δ)) / ε          [analytic formula]

  Moments Accountant (Abadi et al. 2016):
  Tracks the Rényi Differential Privacy (RDP) guarantee at order α and
  converts to (ε, δ)-DP via:

      ε(δ)  =  min_α [ RDP_α  −  log(δ) / (α − 1) ]

  For the subsampled Gaussian mechanism (sampling rate q = batch/N,
  noise multiplier z = σ/Δf), the RDP at order α is bounded by:

      RDP_α  ≈  (1/α−1) · log[ (1−q)^(α−1)(1−q+(α−1)q·exp((α−1)/(2z²)))
                               + (α−1)(1−q) q^α exp((α²−α)/(2z²)) ]

  This is the simplified bound from Mironov (2017) "Rényi Differential
  Privacy of the Gaussian Mechanism".

Notation used throughout
────────────────────────
  epsilon (ε) — privacy loss budget (lower = more private)
  delta   (δ) — failure probability (typically 1/N²)
  sigma   (σ) — Gaussian noise standard deviation
  C           — gradient clipping threshold (= L2-sensitivity Δf)
  q           — sampling rate (lot_size / dataset_size)
  T           — number of rounds
  z           — noise multiplier = σ / C
"""

from __future__ import annotations

import math
import random
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────

# RDP orders to evaluate when converting RDP → (ε, δ)-DP.
# More orders = tighter bound at the cost of more compute.
_RDP_ORDERS: list[float] = [
    1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
    12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0,
]


# ════════════════════════════════════════════════════════════════════════════
#  Gaussian Mechanism
# ════════════════════════════════════════════════════════════════════════════

class GaussianMechanism:
    """
    Calibrated Gaussian noise for (ε, δ)-DP.

    Usage
    ─────
        gm = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        noisy_gradient = gm.privatise(gradient_vector)
        print(gm.sigma)   # noise std actually used
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be > 0, got {sensitivity}")

        self.epsilon     = epsilon
        self.delta       = delta
        self.sensitivity = sensitivity
        self.sigma       = self._calibrate_sigma()

    def _calibrate_sigma(self) -> float:
        """
        Analytic Gaussian mechanism formula:
            σ  =  Δf · √(2 · ln(1.25 / δ)) / ε
        """
        return self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    def privatise(self, vector: list[float]) -> list[float]:
        """Add calibrated Gaussian noise to every coordinate."""
        return [v + random.gauss(0.0, self.sigma) for v in vector]

    def privatise_scalar(self, value: float) -> float:
        """Add calibrated Gaussian noise to a scalar."""
        return value + random.gauss(0.0, self.sigma)

    def __repr__(self) -> str:
        return (
            f"GaussianMechanism(ε={self.epsilon}, δ={self.delta}, "
            f"Δf={self.sensitivity}, σ={self.sigma:.4f})"
        )


# ════════════════════════════════════════════════════════════════════════════
#  Gradient Clipper
# ════════════════════════════════════════════════════════════════════════════

class GradientClipper:
    """
    Per-sample L2 gradient clipping.

    Clips each gradient vector so its L2 norm ≤ C.
    This bounds the L2-sensitivity of the gradient sum to C,
    which is the Δf value fed into GaussianMechanism.

    For a batch of gradients: clip each individually, then sum.
    The sum has sensitivity C (adding/removing one sample changes
    the sum by at most C in L2 norm).

    Usage
    ─────
        clipper = GradientClipper(clip_threshold=1.0)
        clipped = clipper.clip(gradient_vector)
        norm    = clipper.l2_norm(gradient_vector)
    """

    def __init__(self, clip_threshold: float = 1.0) -> None:
        if clip_threshold <= 0:
            raise ValueError(f"clip_threshold must be > 0, got {clip_threshold}")
        self.C = clip_threshold

    def l2_norm(self, vector: list[float]) -> float:
        return math.sqrt(sum(v * v for v in vector))

    def clip(self, vector: list[float]) -> list[float]:
        """Clip vector to L2 norm ≤ C. Returns unchanged if norm ≤ C."""
        norm = self.l2_norm(vector)
        if norm <= self.C:
            return list(vector)
        scale = self.C / norm
        return [v * scale for v in vector]

    def clip_and_sum(self, gradients: list[list[float]]) -> list[float]:
        """
        Clip each gradient in the batch, then sum.
        Returns the aggregated (clipped) gradient vector.
        """
        if not gradients:
            raise ValueError("Empty gradient list.")
        n_features = len(gradients[0])
        agg = [0.0] * n_features
        for g in gradients:
            clipped = self.clip(g)
            for j in range(n_features):
                agg[j] += clipped[j]
        return agg

    def __repr__(self) -> str:
        return f"GradientClipper(C={self.C})"


# ════════════════════════════════════════════════════════════════════════════
#  Moments Accountant  (RDP composition → (ε, δ)-DP)
# ════════════════════════════════════════════════════════════════════════════

def _rdp_gaussian_subsampled(alpha: float, noise_multiplier: float, q: float) -> float:
    """
    RDP guarantee at order α for the subsampled Gaussian mechanism.

    Implements the bound from Mironov 2017 + Wang et al. 2019.
    For α = 1 the limit gives the KL-divergence bound; we handle α > 1.

    Parameters
    ──────────
    alpha          : RDP order (> 1)
    noise_multiplier : z = σ / C  (larger = more private)
    q              : sampling rate ∈ (0, 1]
    """
    if alpha <= 1:
        raise ValueError("RDP order alpha must be > 1")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be > 0")
    if not (0 < q <= 1):
        raise ValueError("sampling rate q must be in (0, 1]")

    # No subsampling — full-batch Gaussian RDP
    if q == 1.0:
        return alpha / (2.0 * noise_multiplier ** 2)

    # Subsampled Gaussian bound (tight for moderate α)
    # RDP_α = (1/(α-1)) · log(E[exp((α-1)·privacy_loss)])
    # Simplified first-order bound (safe and standard):
    #
    #   log[ (1 + q(exp(μ) - 1))^α ] where μ = (α-1)/(2z²)  (single-step)
    #
    # We use the tighter two-term bound from Mironov (2017) Theorem 8:
    z    = noise_multiplier
    a    = alpha
    term1 = (1.0 - q) ** (a - 1.0) * (
                (1.0 - q) + a * q * math.exp((a - 1.0) / (2.0 * z * z))
            )
    # Protect against overflow for large alpha or small noise
    try:
        exp2  = math.exp((a * a - a) / (2.0 * z * z))
    except OverflowError:
        exp2  = float("inf")

    term2 = (a - 1.0) * (1.0 - q) * (q ** a) * exp2

    try:
        log_moment = math.log(term1 + term2)
    except (ValueError, OverflowError):
        # Fallback: non-subsampled bound, always valid (conservative)
        return a / (2.0 * z * z)

    return log_moment / (a - 1.0)


def _rdp_to_dp(rdp_alpha: float, rdp_epsilon: float, delta: float) -> float:
    """
    Convert RDP(α, ε_rdp) to (ε, δ)-DP via the standard conversion:

        ε  =  ε_rdp  +  log(1 - 1/α)  −  log(δ) / (α − 1)

    (Proposition 3, Balle et al. 2020 — tighter than the original Mironov bound)
    """
    if rdp_alpha <= 1:
        return float("inf")
    return (
        rdp_epsilon
        + math.log(1.0 - 1.0 / rdp_alpha)
        - math.log(delta) / (rdp_alpha - 1.0)
    )


class MomentsAccountant:
    """
    Privacy accountant based on Rényi Differential Privacy composition.

    Tracks cumulative privacy spend across T rounds of subsampled
    Gaussian mechanism and converts to (ε, δ)-DP on demand.

    Usage
    ─────
        acc = MomentsAccountant(
            noise_multiplier=1.1,
            sampling_rate=0.01,
            delta=1e-5,
        )
        for round in range(100):
            acc.step()
        epsilon = acc.get_epsilon()
        print(f"After 100 rounds: ε = {epsilon:.4f}")
    """

    def __init__(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        delta: float,
        orders: Optional[list[float]] = None,
    ) -> None:
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be > 0")
        if not (0 < sampling_rate <= 1):
            raise ValueError("sampling_rate must be in (0, 1]")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")

        self.noise_multiplier = noise_multiplier
        self.sampling_rate    = sampling_rate
        self.delta            = delta
        self.orders           = orders or _RDP_ORDERS
        self.steps            = 0      # number of mechanism invocations so far

        # Accumulated RDP at each order
        self._rdp: dict[float, float] = {a: 0.0 for a in self.orders}

    def step(self, num_steps: int = 1) -> None:
        """Record one (or more) mechanism invocations."""
        for _ in range(num_steps):
            for alpha in self.orders:
                self._rdp[alpha] += _rdp_gaussian_subsampled(
                    alpha, self.noise_multiplier, self.sampling_rate
                )
        self.steps += num_steps

    def get_epsilon(self) -> float:
        """
        Convert accumulated RDP to (ε, δ)-DP.
        Returns the tightest ε across all tracked orders.
        """
        best = float("inf")
        for alpha in self.orders:
            eps = _rdp_to_dp(alpha, self._rdp[alpha], self.delta)
            if eps < best:
                best = eps
        return best

    def get_rdp_curve(self) -> dict[float, float]:
        """Return RDP at each tracked order (for diagnostics)."""
        return dict(self._rdp)

    def summary(self) -> dict:
        eps = self.get_epsilon()
        return {
            "steps":           self.steps,
            "noise_multiplier": self.noise_multiplier,
            "sampling_rate":   self.sampling_rate,
            "delta":           self.delta,
            "epsilon":         round(eps, 6),
            "rdp_curve":       {str(a): round(v, 8) for a, v in self._rdp.items()},
        }

    def __repr__(self) -> str:
        return (
            f"MomentsAccountant(z={self.noise_multiplier}, "
            f"q={self.sampling_rate}, δ={self.delta}, "
            f"steps={self.steps}, ε≈{self.get_epsilon():.4f})"
        )


# ════════════════════════════════════════════════════════════════════════════
#  Privacy Budget Tracker
# ════════════════════════════════════════════════════════════════════════════

class PrivacyBudget:
    """
    Mutable per-experiment privacy budget tracker.

    Wraps a MomentsAccountant and adds:
      - a hard budget cap (raises BudgetExhaustedError if exceeded)
      - per-round epsilon snapshots for visualisation
      - a serialisable summary for Postgres storage

    Usage
    ─────
        budget = PrivacyBudget(
            target_epsilon=2.0,
            delta=1e-5,
            noise_multiplier=1.1,
            sampling_rate=0.01,
        )
        for round_idx in range(num_rounds):
            budget.consume_round()          # raises if over budget
        print(budget.current_epsilon)       # final spend
        print(budget.to_dict())             # for DB storage
    """

    class BudgetExhaustedError(Exception):
        pass

    def __init__(
        self,
        target_epsilon: float,
        delta: float,
        noise_multiplier: float,
        sampling_rate: float,
    ) -> None:
        self.target_epsilon   = target_epsilon
        self.delta            = delta
        self.noise_multiplier = noise_multiplier
        self.sampling_rate    = sampling_rate

        self._accountant = MomentsAccountant(
            noise_multiplier=noise_multiplier,
            sampling_rate=sampling_rate,
            delta=delta,
        )
        self._epsilon_history: list[float] = []   # epsilon after each round

    @property
    def current_epsilon(self) -> float:
        if self._accountant.steps == 0:
            return 0.0
        return self._accountant.get_epsilon()

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.target_epsilon - self.current_epsilon)

    @property
    def rounds_consumed(self) -> int:
        return self._accountant.steps

    @property
    def budget_fraction_used(self) -> float:
        if self.target_epsilon <= 0:
            return 1.0
        return min(1.0, self.current_epsilon / self.target_epsilon)

    def consume_round(self) -> float:
        """
        Record one FL round of privacy spend.
        Returns current epsilon after this round.
        Raises BudgetExhaustedError if budget was already exceeded.
        """
        # Check BEFORE consuming (allow the round that pushes just over,
        # but reject any round after the budget is clearly blown)
        if self._accountant.steps > 0 and self.current_epsilon > self.target_epsilon:
            raise PrivacyBudget.BudgetExhaustedError(
                f"Privacy budget exhausted: ε={self.current_epsilon:.4f} "
                f"> target ε={self.target_epsilon}"
            )
        self._accountant.step()
        eps = self._accountant.get_epsilon()
        self._epsilon_history.append(round(eps, 6))
        return eps

    def to_dict(self) -> dict:
        return {
            "target_epsilon":      self.target_epsilon,
            "current_epsilon":     round(self.current_epsilon, 6),
            "remaining_epsilon":   round(self.remaining_epsilon, 6),
            "budget_fraction_used": round(self.budget_fraction_used, 4),
            "delta":               self.delta,
            "noise_multiplier":    self.noise_multiplier,
            "sampling_rate":       self.sampling_rate,
            "rounds_consumed":     self.rounds_consumed,
            "epsilon_history":     self._epsilon_history,
        }

    def __repr__(self) -> str:
        return (
            f"PrivacyBudget(ε_target={self.target_epsilon}, "
            f"ε_current={self.current_epsilon:.4f}, "
            f"remaining={self.remaining_epsilon:.4f}, "
            f"rounds={self.rounds_consumed})"
        )


# ════════════════════════════════════════════════════════════════════════════
#  Convenience: recommend noise multiplier given budget + rounds
# ════════════════════════════════════════════════════════════════════════════

def recommended_noise_multiplier(
    target_epsilon: float,
    delta: float,
    num_rounds: int,
    sampling_rate: float,
    lo: float = 0.1,
    hi: float = 10.0,
    tol: float = 1e-3,
) -> float:
    """
    Binary-search for the smallest noise_multiplier z such that
    T rounds of (z, q)-subsampled Gaussian satisfies (ε, δ)-DP.

    Returns z (noise multiplier = σ/C).
    """
    for _ in range(64):   # 64 bisection steps → precision < 1e-18
        mid = (lo + hi) / 2.0
        acc = MomentsAccountant(
            noise_multiplier=mid,
            sampling_rate=sampling_rate,
            delta=delta,
        )
        acc.step(num_rounds)
        eps = acc.get_epsilon()
        if eps <= target_epsilon:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi
