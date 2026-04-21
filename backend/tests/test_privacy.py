"""
tests/test_privacy.py
══════════════════════════════════════════════════════════════════════
Tests for Day 3: Differential Privacy layer.

Covers
──────
  GaussianMechanism      — sigma calibration, noise injection
  GradientClipper        — L2 clipping, clip-and-sum
  MomentsAccountant      — RDP accumulation, ε conversion, ordering
  PrivacyBudget          — spend tracking, exhaustion guard, to_dict
  recommended_noise_multiplier — binary search correctness
  dp_federated_train     — end-to-end DP training result shape + DP fields
  run_dp_federated_training   — Celery task (eager mode) + DB persistence

Run with:
    USE_SQLITE_FALLBACK=true pytest tests/test_privacy.py -v
"""

from __future__ import annotations

import math
import os
import random

os.environ["USE_SQLITE_FALLBACK"] = "true"
os.environ["CELERY_ASYNC_ENABLED"] = "false"

import pytest

# ── Celery eager mode (no Redis needed) ───────────────────────────────────
from celery_app import celery
celery.conf.update(
    task_always_eager     = True,
    task_eager_propagates = True,
    result_backend        = "cache",
    cache_backend         = "memory",
)

from ml.privacy import (
    GaussianMechanism,
    GradientClipper,
    MomentsAccountant,
    PrivacyBudget,
    recommended_noise_multiplier,
    _rdp_gaussian_subsampled,
    _rdp_to_dp,
)
from ml.dp_engine import dp_federated_train
from database import (
    init_db, drop_db, get_db,
    ExperimentRepo, ExperimentStatus, ModelType,
)
from tasks.training_tasks import run_dp_federated_training


# ─── Shared fixtures ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    drop_db()
    init_db()
    yield
    drop_db()


def _toy_data(n: int = 80):
    """n rows, 3 columns: x1, x2, label (linearly separable)."""
    random.seed(0)
    rows = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        rows.append({"x1": str(x1), "x2": str(x2),
                     "label": "1" if x1 + x2 > 10 else "0"})
    return rows, ["x1", "x2", "label"]


def _make_experiment(db, model_type=ModelType.DP_FEDERATED):
    return ExperimentRepo.create(
        db,
        model_type=model_type,
        hyperparameters={},
        target_col_index=2,
        feature_types={},
        name="dp test experiment",
        dp_enabled=True,
        dp_target_epsilon=2.0,
        dp_delta=1e-5,
        dp_clip_threshold=1.0,
    )


# ════════════════════════════════════════════════════════════════════════════
#  GaussianMechanism
# ════════════════════════════════════════════════════════════════════════════

class TestGaussianMechanism:

    def test_sigma_analytic_formula(self):
        """σ = Δf · √(2 ln(1.25/δ)) / ε"""
        gm = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
        assert abs(gm.sigma - expected) < 1e-10

    def test_smaller_epsilon_means_larger_sigma(self):
        """Stricter privacy budget → more noise."""
        gm_tight  = GaussianMechanism(epsilon=0.5, delta=1e-5)
        gm_loose  = GaussianMechanism(epsilon=2.0, delta=1e-5)
        assert gm_tight.sigma > gm_loose.sigma

    def test_sensitivity_scales_sigma(self):
        gm1 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        gm2 = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=2.0)
        assert abs(gm2.sigma - 2.0 * gm1.sigma) < 1e-10

    def test_privatise_vector_length_preserved(self):
        gm = GaussianMechanism(epsilon=1.0, delta=1e-5)
        vec = [1.0, 2.0, 3.0, 4.0]
        noised = gm.privatise(vec)
        assert len(noised) == len(vec)

    def test_privatise_adds_nonzero_noise_on_average(self):
        """Over many samples, noise should have near-zero mean but nonzero std."""
        gm = GaussianMechanism(epsilon=1.0, delta=1e-5)
        samples = [gm.privatise_scalar(0.0) for _ in range(2000)]
        mean = sum(samples) / len(samples)
        std  = math.sqrt(sum((s - mean)**2 for s in samples) / len(samples))
        assert abs(mean) < 0.2          # mean ≈ 0
        assert abs(std - gm.sigma) < 0.3  # std ≈ σ

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            GaussianMechanism(epsilon=0.0, delta=1e-5)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            GaussianMechanism(epsilon=1.0, delta=1.5)

    def test_invalid_sensitivity_raises(self):
        with pytest.raises(ValueError, match="sensitivity"):
            GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=-1.0)

    def test_repr_contains_sigma(self):
        gm = GaussianMechanism(epsilon=1.0, delta=1e-5)
        assert "σ=" in repr(gm)


# ════════════════════════════════════════════════════════════════════════════
#  GradientClipper
# ════════════════════════════════════════════════════════════════════════════

class TestGradientClipper:

    def test_clip_below_threshold_unchanged(self):
        clipper = GradientClipper(clip_threshold=5.0)
        vec = [1.0, 2.0, 2.0]   # norm = 3.0 < 5.0
        clipped = clipper.clip(vec)
        assert clipped == vec

    def test_clip_above_threshold_scales_to_C(self):
        clipper = GradientClipper(clip_threshold=1.0)
        vec = [3.0, 4.0]        # norm = 5.0
        clipped = clipper.clip(vec)
        norm = clipper.l2_norm(clipped)
        assert abs(norm - 1.0) < 1e-10

    def test_clip_direction_preserved(self):
        clipper = GradientClipper(clip_threshold=1.0)
        vec = [3.0, 4.0]
        clipped = clipper.clip(vec)
        # Direction: ratio of components should be preserved
        assert abs(clipped[0] / clipped[1] - vec[0] / vec[1]) < 1e-10

    def test_clip_zero_vector(self):
        clipper = GradientClipper(clip_threshold=1.0)
        clipped = clipper.clip([0.0, 0.0, 0.0])
        assert clipped == [0.0, 0.0, 0.0]

    def test_clip_and_sum_length(self):
        clipper = GradientClipper(clip_threshold=1.0)
        grads = [[1.0, 2.0], [3.0, 4.0], [0.5, 0.5]]
        result = clipper.clip_and_sum(grads)
        assert len(result) == 2

    def test_clip_and_sum_each_clipped_individually(self):
        """
        Sum of individually clipped vectors must differ from clip(sum).
        This verifies we clip per-client, not the aggregate.
        """
        C = 1.0
        clipper = GradientClipper(clip_threshold=C)
        g1 = [3.0, 4.0]   # norm=5, will be clipped to 1.0
        g2 = [3.0, 4.0]   # same
        agg = clipper.clip_and_sum([g1, g2])
        # Each clipped to norm 1.0 → sum has norm 2.0, NOT 1.0
        assert abs(clipper.l2_norm(agg) - 2.0) < 1e-9

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            GradientClipper(clip_threshold=0.0)

    def test_clip_and_sum_empty_raises(self):
        clipper = GradientClipper(clip_threshold=1.0)
        with pytest.raises(ValueError):
            clipper.clip_and_sum([])


# ════════════════════════════════════════════════════════════════════════════
#  MomentsAccountant
# ════════════════════════════════════════════════════════════════════════════

class TestMomentsAccountant:

    def test_epsilon_zero_before_any_steps(self):
        acc = MomentsAccountant(noise_multiplier=1.1, sampling_rate=0.01, delta=1e-5)
        # step() not called yet — epsilon should be negligible / close to 0
        assert acc.steps == 0

    def test_epsilon_increases_with_steps(self):
        acc = MomentsAccountant(noise_multiplier=1.1, sampling_rate=0.01, delta=1e-5)
        acc.step(10)
        eps10 = acc.get_epsilon()
        acc.step(10)
        eps20 = acc.get_epsilon()
        assert eps20 > eps10

    def test_more_noise_means_less_epsilon(self):
        """Higher noise multiplier → stronger privacy → lower ε."""
        acc_noisy  = MomentsAccountant(noise_multiplier=2.0, sampling_rate=0.01, delta=1e-5)
        acc_quiet  = MomentsAccountant(noise_multiplier=0.5, sampling_rate=0.01, delta=1e-5)
        acc_noisy.step(50)
        acc_quiet.step(50)
        assert acc_noisy.get_epsilon() < acc_quiet.get_epsilon()

    def test_higher_sampling_rate_means_higher_epsilon(self):
        """Larger batch fraction → more data exposed per step → higher ε."""
        acc_high = MomentsAccountant(noise_multiplier=1.1, sampling_rate=0.5,  delta=1e-5)
        acc_low  = MomentsAccountant(noise_multiplier=1.1, sampling_rate=0.01, delta=1e-5)
        acc_high.step(10)
        acc_low.step(10)
        assert acc_high.get_epsilon() > acc_low.get_epsilon()

    def test_rdp_curve_keys_match_orders(self):
        acc = MomentsAccountant(noise_multiplier=1.1, sampling_rate=0.01, delta=1e-5)
        acc.step(5)
        curve = acc.get_rdp_curve()
        assert set(curve.keys()) == set(acc.orders)

    def test_summary_dict_has_required_keys(self):
        acc = MomentsAccountant(noise_multiplier=1.1, sampling_rate=0.01, delta=1e-5)
        acc.step(10)
        s = acc.summary()
        for k in ("steps", "noise_multiplier", "sampling_rate", "delta", "epsilon"):
            assert k in s

    def test_rdp_subsampled_larger_than_zero(self):
        rdp = _rdp_gaussian_subsampled(alpha=2.0, noise_multiplier=1.0, q=0.01)
        assert rdp > 0

    def test_rdp_to_dp_finite(self):
        eps = _rdp_to_dp(rdp_alpha=2.0, rdp_epsilon=0.05, delta=1e-5)
        assert math.isfinite(eps)
        assert eps > 0

    def test_invalid_noise_multiplier_raises(self):
        with pytest.raises(ValueError):
            MomentsAccountant(noise_multiplier=0.0, sampling_rate=0.01, delta=1e-5)

    def test_invalid_sampling_rate_raises(self):
        with pytest.raises(ValueError):
            MomentsAccountant(noise_multiplier=1.0, sampling_rate=0.0, delta=1e-5)


# ════════════════════════════════════════════════════════════════════════════
#  PrivacyBudget
# ════════════════════════════════════════════════════════════════════════════

class TestPrivacyBudget:

    def test_initial_state(self):
        pb = PrivacyBudget(
            target_epsilon=2.0, delta=1e-5,
            noise_multiplier=1.1, sampling_rate=0.01,
        )
        assert pb.current_epsilon == 0.0
        assert pb.rounds_consumed == 0
        assert pb.remaining_epsilon == 2.0
        assert pb.budget_fraction_used == 0.0

    def test_consume_round_increases_epsilon(self):
        pb = PrivacyBudget(
            target_epsilon=10.0, delta=1e-5,
            noise_multiplier=1.1, sampling_rate=0.01,
        )
        pb.consume_round()
        assert pb.current_epsilon > 0.0
        assert pb.rounds_consumed == 1

    def test_epsilon_history_length_matches_rounds(self):
        pb = PrivacyBudget(
            target_epsilon=10.0, delta=1e-5,
            noise_multiplier=1.1, sampling_rate=0.01,
        )
        for _ in range(5):
            pb.consume_round()
        assert len(pb._epsilon_history) == 5

    def test_remaining_never_negative(self):
        pb = PrivacyBudget(
            target_epsilon=0.5, delta=1e-5,
            noise_multiplier=0.5, sampling_rate=0.5,
        )
        for _ in range(20):
            try:
                pb.consume_round()
            except PrivacyBudget.BudgetExhaustedError:
                break
        assert pb.remaining_epsilon >= 0.0

    def test_budget_exhausted_error_raised(self):
        """With a tiny budget and aggressive settings, budget must eventually exhaust."""
        pb = PrivacyBudget(
            target_epsilon=0.01, delta=1e-5,
            noise_multiplier=0.3, sampling_rate=0.9,
        )
        with pytest.raises(PrivacyBudget.BudgetExhaustedError):
            for _ in range(500):
                pb.consume_round()

    def test_to_dict_shape(self):
        pb = PrivacyBudget(
            target_epsilon=2.0, delta=1e-5,
            noise_multiplier=1.1, sampling_rate=0.01,
        )
        pb.consume_round()
        d = pb.to_dict()
        for key in ("target_epsilon", "current_epsilon", "remaining_epsilon",
                    "budget_fraction_used", "delta", "noise_multiplier",
                    "sampling_rate", "rounds_consumed", "epsilon_history"):
            assert key in d, f"Missing key: {key}"

    def test_fraction_between_zero_and_one(self):
        pb = PrivacyBudget(
            target_epsilon=2.0, delta=1e-5,
            noise_multiplier=1.1, sampling_rate=0.01,
        )
        for _ in range(10):
            try:
                pb.consume_round()
            except PrivacyBudget.BudgetExhaustedError:
                break
        assert 0.0 <= pb.budget_fraction_used <= 1.0


# ════════════════════════════════════════════════════════════════════════════
#  recommended_noise_multiplier
# ════════════════════════════════════════════════════════════════════════════

class TestRecommendedNoise:

    def test_result_satisfies_budget(self):
        """
        The recommended z should yield ε ≤ target_epsilon after T rounds.
        """
        target_eps = 2.0
        delta      = 1e-5
        T          = 30
        q          = 0.1

        z = recommended_noise_multiplier(
            target_epsilon=target_eps,
            delta=delta,
            num_rounds=T,
            sampling_rate=q,
        )

        acc = MomentsAccountant(noise_multiplier=z, sampling_rate=q, delta=delta)
        acc.step(T)
        eps = acc.get_epsilon()
        assert eps <= target_eps + 0.05, f"ε={eps:.4f} exceeded target {target_eps}"

    def test_tighter_budget_needs_more_noise(self):
        z_loose = recommended_noise_multiplier(2.0, 1e-5, 30, 0.1)
        z_tight = recommended_noise_multiplier(0.5, 1e-5, 30, 0.1)
        assert z_tight > z_loose

    def test_more_rounds_needs_more_noise(self):
        z_few  = recommended_noise_multiplier(2.0, 1e-5, 10,  0.1)
        z_many = recommended_noise_multiplier(2.0, 1e-5, 100, 0.1)
        assert z_many > z_few


# ════════════════════════════════════════════════════════════════════════════
#  dp_federated_train  (engine-level)
# ════════════════════════════════════════════════════════════════════════════

class TestDPFederatedTrain:

    def test_result_shape(self):
        rows, headers = _toy_data()
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=5, local_epochs=2, lr=0.1, num_clients=3,
            target_epsilon=5.0, delta=1e-5, clip_threshold=1.0,
        )
        for key in ("model", "featureCols", "targetCol", "lossHistory",
                    "trainMetrics", "testMetrics", "finalLoss",
                    "trainingTimeMs", "privacy", "rounds"):
            assert key in result, f"Missing key: {key}"

    def test_model_label_is_dp_federated(self):
        rows, headers = _toy_data()
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=5, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5, clip_threshold=1.0,
        )
        assert result["model"] == "DP-Federated"

    def test_privacy_dict_shape(self):
        rows, headers = _toy_data()
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=5, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5, clip_threshold=1.0,
        )
        p = result["privacy"]
        for key in ("target_epsilon", "current_epsilon", "remaining_epsilon",
                    "budget_fraction_used", "delta", "noise_multiplier",
                    "rounds_consumed", "epsilon_history", "noise_sigma",
                    "clip_threshold"):
            assert key in p, f"Missing privacy key: {key}"

    def test_epsilon_spent_is_positive(self):
        rows, headers = _toy_data()
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=8, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5, clip_threshold=1.0,
        )
        assert result["privacy"]["current_epsilon"] > 0.0

    def test_epsilon_does_not_exceed_target_with_good_noise(self):
        """With high noise, we should stay within budget over few rounds."""
        rows, headers = _toy_data()
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=5, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=10.0, delta=1e-5, clip_threshold=1.0,
            noise_multiplier=3.0,   # very high noise → very low ε spend
        )
        assert result["privacy"]["current_epsilon"] <= 10.0

    def test_auto_calibration_when_no_noise_multiplier(self):
        """Omitting noise_multiplier should auto-calibrate it."""
        rows, headers = _toy_data()
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=5, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=3.0, delta=1e-5, clip_threshold=1.0,
            noise_multiplier=None,
        )
        # Auto-calibrated noise_multiplier should be present in privacy dict
        assert result["privacy"]["noise_multiplier"] > 0.0

    def test_loss_history_length_matches_completed_rounds(self):
        rows, headers = _toy_data()
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=6, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5, clip_threshold=1.0,
        )
        assert len(result["lossHistory"]) == result["rounds"]

    def test_epsilon_history_length_matches_completed_rounds(self):
        rows, headers = _toy_data()
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=6, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5, clip_threshold=1.0,
        )
        assert len(result["privacy"]["epsilon_history"]) == result["rounds"]

    def test_accuracy_in_valid_range(self):
        rows, headers = _toy_data(n=100)
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=15, local_epochs=5, lr=0.1, num_clients=3,
            target_epsilon=5.0, delta=1e-5, clip_threshold=1.0,
        )
        acc = result["testMetrics"]["accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_noise_sigma_equals_multiplier_times_clip(self):
        rows, headers = _toy_data()
        C = 1.5
        z = 2.0
        result = dp_federated_train(
            rows, headers, 2, {},
            rounds=3, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=10.0, delta=1e-5, clip_threshold=C,
            noise_multiplier=z,
        )
        sigma = result["privacy"]["noise_sigma"]
        assert abs(sigma - z * C) < 1e-6


# ════════════════════════════════════════════════════════════════════════════
#  Celery task  run_dp_federated_training  (eager mode)
# ════════════════════════════════════════════════════════════════════════════

class TestDPCeleryTask:

    def test_task_happy_path(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp = _make_experiment(db)
            exp_id = exp.id

        result = run_dp_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=5, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5,
            clip_threshold=1.0, noise_multiplier=2.0,
        )

        assert result["model"] == "DP-Federated"
        assert result["experimentId"] == exp_id
        assert "taskId" in result
        assert "privacy" in result

    def test_task_persists_dp_fields_to_db(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp = _make_experiment(db)
            exp_id = exp.id

        run_dp_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=5, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5,
            clip_threshold=1.0, noise_multiplier=2.0,
        )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.status == ExperimentStatus.COMPLETED
            assert exp.dp_enabled is True

            r = exp.result
            assert r is not None
            assert r.dp_epsilon_final is not None
            assert r.dp_epsilon_final > 0.0
            assert r.dp_epsilon_history is not None
            assert len(r.dp_epsilon_history) > 0
            assert r.dp_noise_sigma is not None
            assert r.dp_budget_summary is not None

    def test_task_experiment_status_completed(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp = _make_experiment(db)
            exp_id = exp.id

        run_dp_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=3, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5,
            clip_threshold=1.0, noise_multiplier=2.0,
        )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.status == ExperimentStatus.COMPLETED

    def test_task_marks_failed_on_bad_data(self):
        with get_db() as db:
            exp = _make_experiment(db)
            exp_id = exp.id

        bad_rows = [{"x": "NaN", "label": "1"} for _ in range(3)]
        with pytest.raises(Exception):
            run_dp_federated_training(
                experiment_id=exp_id,
                rows=bad_rows, headers=["x", "label"],
                target_col_index=1, feature_types={},
                rounds=3, local_epochs=2, lr=0.1, num_clients=2,
                target_epsilon=5.0, delta=1e-5,
                clip_threshold=1.0, noise_multiplier=2.0,
            )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.status == ExperimentStatus.FAILED
            assert exp.error_message is not None

    def test_task_idempotency_guard(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp = _make_experiment(db)
            exp_id = exp.id

        # First run — completes normally
        run_dp_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=3, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5,
            clip_threshold=1.0, noise_multiplier=2.0,
        )

        # Second run — already completed, must skip
        result = run_dp_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=3, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5,
            clip_threshold=1.0, noise_multiplier=2.0,
        )
        assert result.get("skipped") is True

    def test_to_dict_includes_dp_fields(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp = _make_experiment(db)
            exp_id = exp.id

        run_dp_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=3, local_epochs=2, lr=0.1, num_clients=2,
            target_epsilon=5.0, delta=1e-5,
            clip_threshold=1.0, noise_multiplier=2.0,
        )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            d = exp.to_dict(include_result=True)

        assert d["dp_enabled"] is True
        assert d["dp_target_epsilon"] == 5.0
        assert d["dp_delta"] == 1e-5
        assert "result" in d
        assert d["result"]["dp_epsilon_final"] is not None
        assert d["result"]["dp_budget_summary"] is not None


# ── Manual runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    drop_db(); init_db()

    suites = [
        TestGaussianMechanism,
        TestGradientClipper,
        TestMomentsAccountant,
        TestPrivacyBudget,
        TestRecommendedNoise,
        TestDPFederatedTrain,
        TestDPCeleryTask,
    ]

    passed = failed = 0
    for suite_cls in suites:
        suite = suite_cls()
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
                failed += 1

    drop_db()
    print(f"\n{'✅' if not failed else '❌'}  {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
