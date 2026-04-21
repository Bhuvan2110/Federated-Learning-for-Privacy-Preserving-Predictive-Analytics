"""
tests/test_secagg.py
══════════════════════════════════════════════════════════════════════
Tests for Day 4: Secure Aggregation (SecAgg) layer.

Covers
──────
  ShamirSecretSharing   — split/reconstruct correctness, threshold property
  SecAggClient          — masking, pairwise mask cancellation
  SecAggServer          — full round aggregation, mask recovery, dropout
  secagg_federated_train — end-to-end training result shape + secagg fields
  run_secagg_federated_training — Celery task (eager mode) + DB persistence

Run with:
    USE_SQLITE_FALLBACK=true pytest tests/test_secagg.py -v
"""

from __future__ import annotations

import os
import random

os.environ["USE_SQLITE_FALLBACK"] = "true"
os.environ["CELERY_ASYNC_ENABLED"] = "false"

import pytest

# ── Celery eager mode ─────────────────────────────────────────────────────
from celery_app import celery
celery.conf.update(
    task_always_eager     = True,
    task_eager_propagates = True,
    result_backend        = "cache",
    cache_backend         = "memory",
)

from ml.secagg import (
    ShamirSecretSharing,
    SecAggClient,
    SecAggServer,
    _prg_vector,
    _derive_pairwise_seed,
    _derive_self_seed,
)
from ml.secagg_engine import secagg_federated_train
from database import (
    init_db, drop_db, get_db,
    ExperimentRepo, ExperimentStatus, ModelType,
)
from tasks.training_tasks import run_secagg_federated_training


# ─── Shared fixtures ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    drop_db()
    init_db()
    yield
    drop_db()


def _toy_data(n: int = 80):
    random.seed(1)
    rows = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        rows.append({"x1": str(x1), "x2": str(x2),
                     "label": "1" if x1 + x2 > 10 else "0"})
    return rows, ["x1", "x2", "label"]


def _make_experiment(db, secagg_enabled=True):
    return ExperimentRepo.create(
        db,
        model_type=ModelType.SECAGG_FEDERATED,
        hyperparameters={},
        target_col_index=2,
        feature_types={},
        name="secagg test experiment",
        secagg_enabled=secagg_enabled,
        secagg_threshold=3,
        secagg_dropout=0.0,
    )


# ════════════════════════════════════════════════════════════════════════════
#  PRG helpers
# ════════════════════════════════════════════════════════════════════════════

class TestPRGHelpers:

    def test_prg_vector_is_deterministic(self):
        v1 = _prg_vector(seed=42, length=10)
        v2 = _prg_vector(seed=42, length=10)
        assert v1 == v2

    def test_prg_vector_different_seeds_differ(self):
        v1 = _prg_vector(seed=1, length=10)
        v2 = _prg_vector(seed=2, length=10)
        assert v1 != v2

    def test_prg_vector_correct_length(self):
        assert len(_prg_vector(seed=0, length=50)) == 50

    def test_pairwise_seed_symmetric(self):
        """seed(i,j) must equal seed(j,i) for masks to cancel."""
        base = 99999
        for i, j in [(0,1),(2,3),(0,4),(3,7)]:
            assert _derive_pairwise_seed(base, i, j) == _derive_pairwise_seed(base, j, i)

    def test_pairwise_seed_distinct_pairs(self):
        """Different pairs must (overwhelmingly) get different seeds."""
        base = 12345
        seeds = {_derive_pairwise_seed(base, i, j)
                 for i in range(5) for j in range(i+1, 5)}
        assert len(seeds) == 10   # all 10 pairs have distinct seeds

    def test_self_seed_unique_per_client(self):
        base = 99
        seeds = {_derive_self_seed(base, i) for i in range(10)}
        assert len(seeds) == 10


# ════════════════════════════════════════════════════════════════════════════
#  Shamir Secret Sharing
# ════════════════════════════════════════════════════════════════════════════

class TestShamirSecretSharing:

    def test_reconstruct_with_exactly_t_shares(self):
        secret = 123456789
        n, t   = 5, 3
        shares = ShamirSecretSharing.split(secret, n, t)
        # Use exactly t shares
        recon = ShamirSecretSharing.reconstruct(shares[:t])
        assert recon == secret % ShamirSecretSharing.PRIME

    def test_reconstruct_with_all_n_shares(self):
        secret = 987654321
        shares = ShamirSecretSharing.split(secret, 5, 3)
        assert ShamirSecretSharing.reconstruct(shares) == secret % ShamirSecretSharing.PRIME

    def test_reconstruct_with_different_t_subsets(self):
        """Any t shares reconstruct the same secret."""
        secret = 55555
        n, t   = 6, 3
        shares = ShamirSecretSharing.split(secret, n, t)
        p      = ShamirSecretSharing.PRIME
        # Three different subsets of size t
        results = set()
        for i in range(n - t + 1):
            subset = shares[i:i+t]
            results.add(ShamirSecretSharing.reconstruct(subset))
        assert len(results) == 1   # all give the same secret

    def test_threshold_2_of_2(self):
        secret = 1
        shares = ShamirSecretSharing.split(secret, 2, 2)
        assert ShamirSecretSharing.reconstruct(shares) == secret % ShamirSecretSharing.PRIME

    def test_threshold_1_of_n(self):
        secret = 42
        shares = ShamirSecretSharing.split(secret, 4, 1)
        # With t=1, even one share reconstructs it (trivially)
        assert ShamirSecretSharing.reconstruct(shares[:1]) == secret % ShamirSecretSharing.PRIME

    def test_split_returns_n_shares(self):
        shares = ShamirSecretSharing.split(100, n=7, t=4)
        assert len(shares) == 7

    def test_shares_are_distinct_x_values(self):
        shares = ShamirSecretSharing.split(100, n=5, t=3)
        xs = [s[0] for s in shares]
        assert sorted(xs) == list(range(1, 6))


# ════════════════════════════════════════════════════════════════════════════
#  SecAggClient
# ════════════════════════════════════════════════════════════════════════════

class TestSecAggClient:

    def _make_clients(self, n=4, nf=6, seed=1234):
        clients = [
            SecAggClient(client_id=i, n_clients=n, n_features=nf,
                         threshold=max(2,(n+1)//2), round_seed=seed)
            for i in range(n)
        ]
        for c in clients:
            c.setup()
        return clients

    def test_pairwise_masks_cancel_in_sum(self):
        """
        Core SecAgg invariant: Σ pairwise_mask_i = 0 for all j.
        This means if all clients mask their updates, the pairwise
        contributions sum to zero and only the true sum survives.
        """
        n  = 5
        nf = 8
        clients = self._make_clients(n, nf)
        update = [0.0] * nf   # zero update so masked = just masks

        masked = [c.mask(update) for c in clients]

        # Sum of all masked vectors = sum of all self-masks
        # (pairwise masks must have cancelled)
        total = [sum(masked[i][k] for i in range(n)) for k in range(nf)]

        # Sum of self-masks
        self_mask_sum = [0.0] * nf
        for c in clients:
            sm = _prg_vector(c._self_seed, nf)
            for k in range(nf):
                self_mask_sum[k] += sm[k]

        for k in range(nf):
            assert abs(total[k] - self_mask_sum[k]) < 1e-10, (
                f"Pairwise masks did not cancel at index {k}: "
                f"total={total[k]:.6e}  self_mask_sum={self_mask_sum[k]:.6e}"
            )

    def test_masked_differs_from_original(self):
        clients = self._make_clients(n=3, nf=5)
        update = [1.0, 2.0, 3.0, 4.0, 5.0]
        masked = clients[0].mask(update)
        assert masked != update

    def test_mask_preserves_length(self):
        clients = self._make_clients(n=3, nf=10)
        update = [random.random() for _ in range(10)]
        masked = clients[0].mask(update)
        assert len(masked) == 10

    def test_self_seed_shares_count(self):
        clients = self._make_clients(n=5, nf=4)
        shares = clients[0].get_all_self_seed_shares()
        assert len(shares) == 5   # n shares total


# ════════════════════════════════════════════════════════════════════════════
#  SecAggServer
# ════════════════════════════════════════════════════════════════════════════

class TestSecAggServer:

    def _random_updates(self, n, nf, seed=42):
        rng = random.Random(seed)
        weights = [[rng.gauss(0, 1) for _ in range(nf)] for _ in range(n)]
        biases  = [rng.gauss(0, 0.1) for _ in range(n)]
        return weights, biases

    def test_aggregate_recovers_true_sum_no_dropout(self):
        """SecAgg result must equal plaintext sum of all updates."""
        n, nf = 4, 6
        server  = SecAggServer(n_clients=n, n_features=nf, dropout_rate=0.0)
        w_upd, b_upd = self._random_updates(n, nf)

        result = server.aggregate_round(w_upd, b_upd, round_seed=777)

        true_w = [sum(w_upd[i][k] for i in range(n)) for k in range(nf)]
        true_b = sum(b_upd)

        assert result.verified, "SecAgg verification flag should be True"
        for k in range(nf):
            assert abs(result.aggregated_weights[k] - true_w[k]) < 1e-9, (
                f"Weight mismatch at k={k}"
            )
        assert abs(result.aggregated_bias - true_b) < 1e-9

    def test_aggregate_verified_flag_true(self):
        n, nf = 3, 4
        server = SecAggServer(n_clients=n, n_features=nf)
        w, b   = self._random_updates(n, nf)
        result = server.aggregate_round(w, b)
        assert result.verified is True

    def test_no_clients_dropped_when_dropout_zero(self):
        server = SecAggServer(n_clients=5, n_features=4, dropout_rate=0.0)
        w, b   = self._random_updates(5, 4)
        result = server.aggregate_round(w, b)
        assert result.n_clients_dropped == 0
        assert result.n_clients_survived == 5

    def test_dropout_reduces_surviving_count(self):
        n = 5
        server = SecAggServer(n_clients=n, n_features=4, dropout_rate=0.2)
        w, b   = self._random_updates(n, 4)
        result = server.aggregate_round(w, b)
        assert result.n_clients_dropped == 1
        assert result.n_clients_survived == 4

    def test_aggregate_with_dropout_recovers_survived_sum(self):
        """With 1 dropout, SecAgg must recover sum of surviving 4 clients."""
        n, nf = 5, 6
        server = SecAggServer(n_clients=n, n_features=nf,
                              dropout_rate=0.2, threshold=3)
        w_upd, b_upd = self._random_updates(n, nf, seed=99)

        result = server.aggregate_round(w_upd, b_upd, round_seed=42)

        # Survived = clients 0..3  (last 1 dropped)
        survived = list(range(n - result.n_clients_dropped))
        true_w = [sum(w_upd[i][k] for i in survived) for k in range(nf)]
        true_b = sum(b_upd[i] for i in survived)

        assert result.verified
        for k in range(nf):
            assert abs(result.aggregated_weights[k] - true_w[k]) < 1e-9

        assert abs(result.aggregated_bias - true_b) < 1e-9

    def test_two_clients_minimum(self):
        """SecAgg is defined only for ≥ 2 clients."""
        with pytest.raises(ValueError, match="at least 2"):
            SecAggServer(n_clients=1, n_features=4)

    def test_threshold_exceeds_n_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            SecAggServer(n_clients=3, n_features=4, threshold=5)

    def test_too_many_dropouts_raises(self):
        """If dropout leaves fewer survivors than threshold, raise RuntimeError."""
        n = 4
        server = SecAggServer(n_clients=n, n_features=4,
                              dropout_rate=0.75,   # drops 3/4
                              threshold=4)          # needs all 4
        w, b = self._random_updates(n, 4)
        with pytest.raises(RuntimeError, match="Too many dropouts"):
            server.aggregate_round(w, b)

    def test_round_seed_reproducibility(self):
        """Same seed → same aggregated result (deterministic protocol)."""
        n, nf = 3, 5
        server = SecAggServer(n_clients=n, n_features=nf)
        w, b   = self._random_updates(n, nf)
        r1 = server.aggregate_round(w, b, round_seed=12345)
        r2 = server.aggregate_round(w, b, round_seed=12345)
        assert r1.aggregated_weights == r2.aggregated_weights

    def test_secagg_summary_keys(self):
        server = SecAggServer(n_clients=3, n_features=4)
        w, b   = self._random_updates(3, 4)
        result = server.aggregate_round(w, b)
        for key in ("protocol", "n_clients", "n_survived", "n_dropped",
                    "threshold", "masking_overhead_ms", "verified"):
            assert key in result.secagg_summary

    def test_large_n_clients(self):
        """Protocol should work with 20 clients."""
        n, nf = 20, 10
        server = SecAggServer(n_clients=n, n_features=nf)
        w, b   = self._random_updates(n, nf, seed=5)
        result = server.aggregate_round(w, b)
        assert result.verified

    def test_masking_overhead_positive(self):
        server = SecAggServer(n_clients=4, n_features=6)
        w, b   = self._random_updates(4, 6)
        result = server.aggregate_round(w, b)
        assert result.masking_overhead_ms > 0


# ════════════════════════════════════════════════════════════════════════════
#  secagg_federated_train  (engine-level)
# ════════════════════════════════════════════════════════════════════════════

class TestSecAggFederatedTrain:

    def test_result_shape(self):
        rows, headers = _toy_data()
        result = secagg_federated_train(
            rows, headers, 2, {},
            rounds=5, local_epochs=2, lr=0.1, num_clients=4,
        )
        for key in ("model", "featureCols", "targetCol", "lossHistory",
                    "trainMetrics", "testMetrics", "finalLoss",
                    "trainingTimeMs", "secagg", "rounds"):
            assert key in result, f"Missing key: {key}"

    def test_model_label(self):
        rows, headers = _toy_data()
        result = secagg_federated_train(
            rows, headers, 2, {},
            rounds=3, local_epochs=2, lr=0.1, num_clients=3,
        )
        assert result["model"] == "SecAgg-Federated"

    def test_secagg_dict_shape(self):
        rows, headers = _toy_data()
        result = secagg_federated_train(
            rows, headers, 2, {},
            rounds=3, local_epochs=2, lr=0.1, num_clients=3,
        )
        s = result["secagg"]
        for key in ("protocol", "n_clients", "threshold", "dropout_rate",
                    "rounds_completed", "rounds_aborted",
                    "all_rounds_verified", "total_overhead_ms", "per_round_log"):
            assert key in s, f"Missing secagg key: {key}"

    def test_all_rounds_verified(self):
        rows, headers = _toy_data()
        result = secagg_federated_train(
            rows, headers, 2, {},
            rounds=5, local_epochs=2, lr=0.1, num_clients=4,
        )
        assert result["secagg"]["all_rounds_verified"] is True

    def test_per_round_log_length(self):
        rows, headers = _toy_data()
        result = secagg_federated_train(
            rows, headers, 2, {},
            rounds=4, local_epochs=2, lr=0.1, num_clients=3,
        )
        assert len(result["secagg"]["per_round_log"]) == 4

    def test_loss_history_length_matches_rounds(self):
        rows, headers = _toy_data()
        result = secagg_federated_train(
            rows, headers, 2, {},
            rounds=6, local_epochs=2, lr=0.1, num_clients=3,
        )
        assert len(result["lossHistory"]) == result["rounds"]

    def test_accuracy_in_valid_range(self):
        rows, headers = _toy_data(n=100)
        result = secagg_federated_train(
            rows, headers, 2, {},
            rounds=10, local_epochs=3, lr=0.1, num_clients=4,
        )
        assert 0.0 <= result["testMetrics"]["accuracy"] <= 1.0

    def test_with_dropout(self):
        """Training still completes with 1 dropped client per round."""
        rows, headers = _toy_data(n=100)
        result = secagg_federated_train(
            rows, headers, 2, {},
            rounds=4, local_epochs=2, lr=0.1,
            num_clients=5, dropout_rate=0.2, secagg_threshold=3,
        )
        per_round = result["secagg"]["per_round_log"]
        assert all(e["n_dropped"] == 1 for e in per_round)
        assert all(e["n_survived"] == 4 for e in per_round)

    def test_training_equivalent_to_plain_fedavg_structure(self):
        """
        SecAgg is a privacy wrapper — training accuracy should be
        in the same ballpark as plain FedAvg (not catastrophically worse).
        """
        from ml.engine import federated_train
        rows, headers = _toy_data(n=100)
        kwargs = dict(rows=rows, headers=headers, target_idx=2, ftypes={},
                      rounds=10, local_epochs=3, lr=0.1, num_clients=4)

        plain  = federated_train(**kwargs)
        secagg = secagg_federated_train(**kwargs)

        # SecAgg accuracy should be within 20 percentage points of plain FedAvg
        acc_diff = abs(secagg["testMetrics"]["accuracy"] -
                       plain["testMetrics"]["accuracy"])
        assert acc_diff < 0.20, (
            f"SecAgg accuracy {secagg['testMetrics']['accuracy']:.3f} "
            f"too far from FedAvg {plain['testMetrics']['accuracy']:.3f}"
        )

    def test_overhead_ms_positive(self):
        rows, headers = _toy_data()
        result = secagg_federated_train(
            rows, headers, 2, {},
            rounds=3, local_epochs=2, lr=0.1, num_clients=3,
        )
        assert result["secagg"]["total_overhead_ms"] > 0


# ════════════════════════════════════════════════════════════════════════════
#  Celery task  run_secagg_federated_training  (eager mode)
# ════════════════════════════════════════════════════════════════════════════

class TestSecAggCeleryTask:

    def test_task_happy_path(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp    = _make_experiment(db)
            exp_id = exp.id

        result = run_secagg_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=4, local_epochs=2, lr=0.1,
            num_clients=4, dropout_rate=0.0, secagg_threshold=3,
        )

        assert result["model"] == "SecAgg-Federated"
        assert result["experimentId"] == exp_id
        assert "taskId" in result
        assert "secagg" in result

    def test_task_persists_secagg_fields(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp    = _make_experiment(db)
            exp_id = exp.id

        run_secagg_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=4, local_epochs=2, lr=0.1,
            num_clients=4, dropout_rate=0.0, secagg_threshold=3,
        )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.status == ExperimentStatus.COMPLETED
            assert exp.secagg_enabled is True
            assert exp.secagg_threshold == 3

            r = exp.result
            assert r is not None
            assert r.secagg_summary is not None
            assert r.secagg_all_verified is True

    def test_task_marks_failed_on_bad_data(self):
        with get_db() as db:
            exp    = _make_experiment(db)
            exp_id = exp.id

        bad_rows = [{"x": "bad", "label": "1"} for _ in range(3)]
        with pytest.raises(Exception):
            run_secagg_federated_training(
                experiment_id=exp_id,
                rows=bad_rows, headers=["x", "label"],
                target_col_index=1, feature_types={},
                rounds=3, local_epochs=2, lr=0.1,
                num_clients=2, dropout_rate=0.0, secagg_threshold=2,
            )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.status == ExperimentStatus.FAILED
            assert exp.error_message is not None

    def test_task_idempotency_guard(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp    = _make_experiment(db)
            exp_id = exp.id

        run_secagg_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=3, local_epochs=2, lr=0.1,
            num_clients=3, dropout_rate=0.0, secagg_threshold=2,
        )

        result = run_secagg_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=3, local_epochs=2, lr=0.1,
            num_clients=3, dropout_rate=0.0, secagg_threshold=2,
        )
        assert result.get("skipped") is True

    def test_to_dict_includes_secagg_fields(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp    = _make_experiment(db)
            exp_id = exp.id

        run_secagg_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=3, local_epochs=2, lr=0.1,
            num_clients=3, dropout_rate=0.0, secagg_threshold=2,
        )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            d   = exp.to_dict(include_result=True)

        assert d["secagg_enabled"] is True
        assert d["secagg_threshold"] == 3
        assert "result" in d
        assert d["result"]["secagg_summary"] is not None
        assert d["result"]["secagg_all_verified"] is True

    def test_with_dropout_task(self):
        """Task still completes with dropout=0.2 on 5 clients."""
        rows, headers = _toy_data(n=100)
        with get_db() as db:
            exp    = _make_experiment(db)
            exp_id = exp.id

        result = run_secagg_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=3, local_epochs=2, lr=0.1,
            num_clients=5, dropout_rate=0.2, secagg_threshold=3,
        )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.status == ExperimentStatus.COMPLETED
            assert exp.result.secagg_summary["dropout_rate"] == 0.2


# ── Manual runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    drop_db(); init_db()

    suites = [
        TestPRGHelpers,
        TestShamirSecretSharing,
        TestSecAggClient,
        TestSecAggServer,
        TestSecAggFederatedTrain,
        TestSecAggCeleryTask,
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
                failed += 1

    drop_db()
    print(f"\n{'✅' if not failed else '❌'}  {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
