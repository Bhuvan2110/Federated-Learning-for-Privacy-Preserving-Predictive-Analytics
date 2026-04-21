"""
tests/test_celery_tasks.py
══════════════════════════════════════════════════════════════════════
Tests for the Celery task layer.

Approach: run tasks in EAGER mode (task_always_eager=True) so no
Redis or worker process is needed. The task function executes
synchronously in-process, exactly as a worker would run it, but
without any broker/backend round-trip.

Run with:
    USE_SQLITE_FALLBACK=true pytest tests/test_celery_tasks.py -v
"""

import os
os.environ["USE_SQLITE_FALLBACK"] = "true"
os.environ["CELERY_ASYNC_ENABLED"] = "false"

import pytest

# ── Patch Celery into eager (synchronous) mode before importing tasks ──────
from celery_app import celery
celery.conf.update(
    task_always_eager       = True,   # run tasks inline, no worker needed
    task_eager_propagates   = True,   # re-raise exceptions so pytest sees them
    result_backend          = "cache",
    cache_backend           = "memory",
)

from database import init_db, drop_db, get_db, ExperimentRepo, ExperimentStatus, ModelType
from tasks.training_tasks import run_central_training, run_federated_training


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    drop_db(); init_db()
    yield
    drop_db()


def _make_experiment(db, model_type=ModelType.CENTRAL):
    return ExperimentRepo.create(
        db,
        model_type=model_type,
        hyperparameters={},
        target_col_index=2,
        feature_types={},
        name="test experiment",
    )


def _toy_data():
    """50 rows, 3 columns: x1 (numeric), x2 (numeric), label (binary)."""
    import random
    random.seed(42)
    rows = []
    for i in range(50):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        label = "1" if x1 + x2 > 10 else "0"
        rows.append({"x1": str(x1), "x2": str(x2), "label": label})
    return rows, ["x1", "x2", "label"]


# ── ML engine unit tests ───────────────────────────────────────────────────

class TestMLEngine:

    def test_central_train_returns_expected_keys(self):
        from ml.engine import central_train
        rows, headers = _toy_data()
        result = central_train(rows, headers, 2, {}, epochs=20, lr=0.1)
        for key in ("model", "featureCols", "targetCol", "lossHistory",
                    "trainMetrics", "testMetrics", "finalLoss", "trainingTimeMs"):
            assert key in result, f"Missing key: {key}"

    def test_central_train_loss_decreases(self):
        from ml.engine import central_train
        rows, headers = _toy_data()
        result = central_train(rows, headers, 2, {}, epochs=50, lr=0.1)
        hist = result["lossHistory"]
        # Loss should be lower at end than at start for a separable problem
        assert hist[-1] < hist[0], "Loss did not decrease over training"

    def test_federated_train_returns_expected_keys(self):
        from ml.engine import federated_train
        rows, headers = _toy_data()
        result = federated_train(rows, headers, 2, {}, rounds=10,
                                 local_epochs=3, lr=0.1, num_clients=3)
        for key in ("model", "numClients", "rounds", "lossHistory",
                    "trainMetrics", "testMetrics", "finalLoss"):
            assert key in result, f"Missing key: {key}"

    def test_metrics_perfect_classifier(self):
        from ml.engine import metrics
        y_pred = [1, 1, 0, 0]
        y_true = [1, 1, 0, 0]
        m = metrics(y_pred, y_true)
        assert m["accuracy"]  == 1.0
        assert m["precision"] == 1.0
        assert m["recall"]    == 1.0
        assert m["f1"]        == 1.0

    def test_metrics_all_wrong(self):
        from ml.engine import metrics
        y_pred = [0, 0, 1, 1]
        y_true = [1, 1, 0, 0]
        m = metrics(y_pred, y_true)
        assert m["accuracy"]  == 0.0
        assert m["confMatrix"]["tp"] == 0
        assert m["confMatrix"]["tn"] == 0

    def test_prepare_data_raises_on_too_few_rows(self):
        from ml.engine import prepare_data
        rows = [{"x": "1", "y": "0"} for _ in range(5)]
        with pytest.raises(ValueError, match="≥10"):
            prepare_data(rows, ["x", "y"], 1, {})

    def test_normalise_zero_std_column(self):
        """A constant column should not produce division-by-zero."""
        from ml.engine import normalise
        X = [[5.0, i * 1.0] for i in range(10)]   # col0 is constant
        Xn, means, stds = normalise(X)
        assert stds[0] == 1.0           # clamped to 1 to avoid /0
        assert all(row[0] == 0.0 for row in Xn)  # (5-5)/1 = 0


# ── Celery task tests (eager mode) ─────────────────────────────────────────

class TestCeleryTasks:

    def test_central_task_happy_path(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp = _make_experiment(db)
            exp_id = exp.id

        result = run_central_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            epochs=20, lr=0.1,
        )

        assert result["model"] == "Central"
        assert result["experimentId"] == exp_id
        assert "taskId" in result

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.status == ExperimentStatus.COMPLETED
            assert exp.result is not None
            assert exp.result.test_accuracy is not None

    def test_federated_task_happy_path(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp = _make_experiment(db, ModelType.FEDERATED)
            exp_id = exp.id

        result = run_federated_training(
            experiment_id=exp_id,
            rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=5, local_epochs=2, lr=0.1, num_clients=3,
        )

        assert result["model"] == "Federated"
        assert result["experimentId"] == exp_id

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.status == ExperimentStatus.COMPLETED

    def test_central_task_marks_failed_on_bad_data(self):
        with get_db() as db:
            exp = _make_experiment(db)
            exp_id = exp.id

        bad_rows = [{"x": "not-a-number", "label": "1"} for _ in range(3)]
        with pytest.raises(Exception):
            run_central_training(
                experiment_id=exp_id,
                rows=bad_rows, headers=["x", "label"],
                target_col_index=1, feature_types={},
                epochs=5, lr=0.1,
            )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.status == ExperimentStatus.FAILED
            assert exp.error_message is not None

    def test_central_task_skips_if_already_completed(self):
        """Idempotency guard — re-queued tasks must not overwrite results."""
        rows, headers = _toy_data()
        with get_db() as db:
            exp = _make_experiment(db)
            exp_id = exp.id

        # First run — completes normally
        run_central_training(
            experiment_id=exp_id, rows=rows, headers=headers,
            target_col_index=2, feature_types={}, epochs=5, lr=0.1,
        )

        # Second run — should detect COMPLETED and skip
        result = run_central_training(
            experiment_id=exp_id, rows=rows, headers=headers,
            target_col_index=2, feature_types={}, epochs=5, lr=0.1,
        )
        assert result.get("skipped") is True

    def test_federated_task_result_persisted_correctly(self):
        rows, headers = _toy_data()
        with get_db() as db:
            exp = _make_experiment(db, ModelType.FEDERATED)
            exp_id = exp.id

        run_federated_training(
            experiment_id=exp_id, rows=rows, headers=headers,
            target_col_index=2, feature_types={},
            rounds=3, local_epochs=2, lr=0.1, num_clients=2,
        )

        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            r   = exp.result
            assert r.train_samples  > 0
            assert r.test_samples   > 0
            assert r.loss_history   is not None
            assert len(r.loss_history) == 3   # one entry per round
            assert r.training_time_ms > 0


# ── Manual runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    drop_db(); init_db()
    suites = [TestMLEngine, TestCeleryTasks]
    passed = failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        methods = [m for m in dir(suite) if m.startswith("test_")]
        for m in methods:
            drop_db(); init_db()
            try:
                getattr(suite, m)()
                print(f"  ✓  {suite_cls.__name__}::{m}")
                passed += 1
            except Exception as exc:
                print(f"  ✗  {suite_cls.__name__}::{m}  →  {exc}")
                failed += 1
    print(f"\n{'✅' if not failed else '❌'}  {passed} passed, {failed} failed")
    drop_db()
    sys.exit(1 if failed else 0)
