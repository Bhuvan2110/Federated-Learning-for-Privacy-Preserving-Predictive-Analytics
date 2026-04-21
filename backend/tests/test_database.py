"""
tests/test_database.py
══════════════════════════════════════════════════════════════════════
Smoke tests for the Day 1 database layer.

Run with:
    USE_SQLITE_FALLBACK=true pytest tests/test_database.py -v

No Postgres required — SQLite is used via the fallback flag.
"""

import os
os.environ["USE_SQLITE_FALLBACK"] = "true"   # must be set before importing db

import pytest
from database import (
    init_db, drop_db, get_db,
    UserRole, ModelType, ExperimentStatus,
    UserRepo, FileRepo, ExperimentRepo, ResultRepo,
)


@pytest.fixture(autouse=True)
def fresh_db():
    """Drop and recreate schema before every test."""
    drop_db()
    init_db()
    yield
    drop_db()


# ─── User tests ───────────────────────────────────────────────────────────

def test_create_and_fetch_user():
    with get_db() as db:
        user = UserRepo.create(db, username="alice", email="alice@example.com")
    with get_db() as db:
        fetched = UserRepo.get_by_username(db, "alice")
        assert fetched is not None
        assert fetched.email == "alice@example.com"
        assert fetched.role == UserRole.TRAINER
        assert fetched.is_active is True


def test_user_deactivate():
    with get_db() as db:
        user = UserRepo.create(db, username="bob", email="bob@example.com")
        uid = user.id
    with get_db() as db:
        UserRepo.deactivate(db, uid)
    with get_db() as db:
        user = UserRepo.get_by_id(db, uid)
        assert user.is_active is False


# ─── File upload tests ────────────────────────────────────────────────────

def test_create_and_fetch_file():
    with get_db() as db:
        f = FileRepo.create(
            db,
            filename="test.csv",
            total_rows=100,
            total_cols=5,
            headers=["a","b","c","d","target"],
            column_stats=[{"col":"a","type":"numeric"}],
            encrypted_upload=True,
            encryption_method="RSA-2048-OAEP + AES-256-GCM",
        )
        fid = f.id

    with get_db() as db:
        fetched = FileRepo.get_by_id(db, fid)
        assert fetched is not None
        assert fetched.filename == "test.csv"
        assert fetched.total_rows == 100
        assert fetched.encrypted_upload is True


def test_list_files():
    with get_db() as db:
        for i in range(3):
            FileRepo.create(
                db, filename=f"file{i}.csv", total_rows=10, total_cols=2,
                headers=["x","y"], column_stats=[],
            )
    with get_db() as db:
        files = FileRepo.list_all(db, limit=10)
        assert len(files) == 3


# ─── Experiment tests ─────────────────────────────────────────────────────

def test_experiment_full_lifecycle():
    """pending → running → completed with result."""
    with get_db() as db:
        exp = ExperimentRepo.create(
            db,
            model_type=ModelType.CENTRAL,
            hyperparameters={"epochs": 10, "lr": 0.1},
            target_col_index=4,
            feature_types={"a":"numeric"},
            name="Test central run",
        )
        exp_id = exp.id
        assert exp.status == ExperimentStatus.PENDING

    with get_db() as db:
        ExperimentRepo.mark_running(db, exp_id)
    with get_db() as db:
        exp = ExperimentRepo.get_by_id(db, exp_id)
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.started_at is not None

    fake_result = {
        "targetCol": "target",
        "featureCols": ["a","b","c","d"],
        "uniqueLabels": ["0","1"],
        "trainSamples": 80,
        "testSamples": 20,
        "finalLoss": 0.3412,
        "lossHistory": [0.7, 0.5, 0.4, 0.3412],
        "trainMetrics": {"accuracy":0.91,"precision":0.89,"recall":0.90,"f1":0.895,
                          "confMatrix":{"tp":40,"fp":4,"fn":4,"tn":32}},
        "testMetrics":  {"accuracy":0.85,"precision":0.83,"recall":0.84,"f1":0.835,
                          "confMatrix":{"tp":9,"fp":2,"fn":1,"tn":8}},
        "trainingTimeMs": 1234,
    }

    with get_db() as db:
        ExperimentRepo.mark_completed(db, exp_id, fake_result)
    with get_db() as db:
        exp = ExperimentRepo.get_by_id(db, exp_id)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.completed_at is not None
        assert exp.target_col == "target"

        result = ResultRepo.get_for_experiment(db, exp_id)
        assert result is not None
        assert result.test_accuracy == pytest.approx(0.85)
        assert result.final_loss    == pytest.approx(0.3412)
        assert result.train_samples == 80


def test_experiment_failure():
    with get_db() as db:
        exp = ExperimentRepo.create(
            db, model_type=ModelType.FEDERATED,
            hyperparameters={"rounds":5}, target_col_index=0,
            feature_types={},
        )
        exp_id = exp.id
    with get_db() as db:
        ExperimentRepo.mark_failed(db, exp_id, "Not enough data rows")
    with get_db() as db:
        exp = ExperimentRepo.get_by_id(db, exp_id)
        assert exp.status == ExperimentStatus.FAILED
        assert "Not enough data" in exp.error_message


def test_experiment_to_dict_includes_result():
    with get_db() as db:
        exp = ExperimentRepo.create(
            db, model_type=ModelType.CENTRAL,
            hyperparameters={"epochs":5}, target_col_index=0, feature_types={},
        )
        exp_id = exp.id
        ExperimentRepo.mark_completed(db, exp_id, {
            "targetCol":"t","featureCols":["x"],"uniqueLabels":["0","1"],
            "trainSamples":8,"testSamples":2,"finalLoss":0.1,"lossHistory":[0.5,0.1],
            "trainMetrics":{"accuracy":0.9,"precision":0.9,"recall":0.9,"f1":0.9,
                             "confMatrix":{"tp":4,"fp":0,"fn":1,"tn":3}},
            "testMetrics":{"accuracy":0.8,"precision":0.8,"recall":0.8,"f1":0.8,
                            "confMatrix":{"tp":1,"fp":0,"fn":0,"tn":1}},
            "trainingTimeMs":100,
        })

    with get_db() as db:
        exp = ExperimentRepo.get_by_id(db, exp_id)
        d = exp.to_dict(include_result=True)
        assert "result" in d
        assert d["result"]["test_accuracy"] == pytest.approx(0.8)


def test_list_experiments():
    with get_db() as db:
        for i in range(5):
            ExperimentRepo.create(
                db, model_type=ModelType.CENTRAL,
                hyperparameters={}, target_col_index=0, feature_types={},
                name=f"Exp {i}",
            )
    with get_db() as db:
        exps = ExperimentRepo.list_all(db, limit=10)
        assert len(exps) == 5


if __name__ == "__main__":
    # Quick manual run without pytest
    import sys
    os.environ["USE_SQLITE_FALLBACK"] = "true"
    drop_db(); init_db()
    try:
        test_create_and_fetch_user(); print("✓ create/fetch user")
        test_user_deactivate();       print("✓ deactivate user")
        test_create_and_fetch_file(); print("✓ create/fetch file")
        test_list_files();            print("✓ list files")
        drop_db(); init_db()
        test_experiment_full_lifecycle(); print("✓ experiment lifecycle")
        drop_db(); init_db()
        test_experiment_failure();    print("✓ experiment failure")
        drop_db(); init_db()
        test_experiment_to_dict_includes_result(); print("✓ to_dict with result")
        drop_db(); init_db()
        test_list_experiments();      print("✓ list experiments")
        print("\n✅  All smoke tests passed!")
    except Exception as exc:
        print(f"\n❌  Test failed: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        drop_db()
