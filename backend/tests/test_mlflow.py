"""
tests/test_mlflow.py
══════════════════════════════════════════════════════════════════════
Tests for Day 6: MLflow experiment tracking.

Strategy
────────
  MLflow is tested without a real tracking server by patching
  the mlflow module with lightweight stubs.  This lets us verify:
    • Every log_params / log_metrics / log_artifacts / set_tags call
      is made with the correct values
    • The tracker is non-blocking (MLflow errors don't crash training)
    • LogisticRegressionWrapper predict() is correct
    • Model card markdown contains all required sections
    • mark_completed() calls log_run() and survives MLflow failure
    • All four Flask /api/mlflow/* routes respond correctly

Run with:
    USE_SQLITE_FALLBACK=true pytest tests/test_mlflow.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch, call

os.environ["USE_SQLITE_FALLBACK"] = "true"
os.environ["CELERY_ASYNC_ENABLED"] = "false"
os.environ["MLFLOW_ENABLED"] = "true"

import pytest

# ── Force _MLFLOW_AVAILABLE = True via mock before tracker import ─────────
import sys

# Build a minimal mlflow stub so the tracker can import without the real package
_mock_mlflow = MagicMock()

# mlflow.start_run() must work as a context manager
class _FakeRun:
    class _Info:
        run_id = "fake-run-id-001"
    info = _Info()

class _FakeRunCtx:
    def __enter__(self): return _FakeRun()
    def __exit__(self, *_): pass

_mock_mlflow.start_run.return_value = _FakeRunCtx()
_mock_mlflow.MlflowClient.return_value = MagicMock()

# Patch mlflow into sys.modules before importing tracker
sys.modules.setdefault("mlflow", _mock_mlflow)
sys.modules.setdefault("mlflow.sklearn", MagicMock())

# Now import tracking — _MLFLOW_AVAILABLE will be True because import succeeded
import importlib
import tracking.mlflow_tracker as _tracker_module
# Force _MLFLOW_AVAILABLE = True so tests exercise the full code path
_tracker_module._MLFLOW_AVAILABLE = True
_tracker_module.MLFLOW_ENABLED    = True

from tracking.mlflow_tracker import (
    MLflowTracker,
    LogisticRegressionWrapper,
    mlflow_health,
    get_tracker,
    _MLFLOW_AVAILABLE,
)
from database import init_db, drop_db, get_db, ExperimentRepo, ModelType


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    drop_db(); init_db()
    yield
    drop_db()


@pytest.fixture(autouse=True)
def reset_mlflow_mock():
    """Reset all mock call counts between tests."""
    _mock_mlflow.reset_mock()
    _mock_mlflow.start_run.return_value = _FakeRunCtx()
    yield


def _make_experiment(model_type=ModelType.CENTRAL, hp=None, **kwargs):
    with get_db() as db:
        exp = ExperimentRepo.create(
            db,
            model_type=model_type,
            hyperparameters=hp or {"epochs": 50, "lr": 0.1},
            target_col_index=2,
            feature_types={},
            name="mlflow test experiment",
            **kwargs,
        )
        return exp.id


def _central_result(n_feat=4) -> dict:
    return {
        "model":        "Central",
        "featureCols":  [f"f{i}" for i in range(n_feat)],
        "targetCol":    "label",
        "uniqueLabels": ["0", "1"],
        "trainSamples": 80,
        "testSamples":  20,
        "epochs":       50,
        "lr":           0.1,
        "lossHistory":  [0.693, 0.612, 0.54, 0.48, 0.43],
        "trainMetrics": {"accuracy":0.91,"precision":0.90,"recall":0.89,"f1":0.895,
                          "confMatrix":{"tp":40,"fp":4,"fn":4,"tn":32}},
        "testMetrics":  {"accuracy":0.85,"precision":0.83,"recall":0.84,"f1":0.835,
                          "confMatrix":{"tp":9,"fp":2,"fn":1,"tn":8}},
        "finalLoss":    0.43,
        "trainingTimeMs": 1200,
    }


def _federated_result(n_feat=4) -> dict:
    base = _central_result(n_feat)
    base.update({
        "model":       "Federated",
        "numClients":  5,
        "rounds":      10,
        "localEpochs": 3,
        "lossHistory": [0.7, 0.65, 0.60, 0.55, 0.51,
                        0.48, 0.46, 0.44, 0.43, 0.42],
    })
    return base


def _dp_result(n_feat=4) -> dict:
    base = _federated_result(n_feat)
    base.update({
        "model": "DP-Federated",
        "privacy": {
            "target_epsilon":      2.0,
            "current_epsilon":     1.74,
            "remaining_epsilon":   0.26,
            "budget_fraction_used": 0.87,
            "delta":               1e-5,
            "noise_multiplier":    1.1,
            "sampling_rate":       0.02,
            "rounds_consumed":     10,
            "epsilon_history":     [0.17*i for i in range(1, 11)],
            "noise_sigma":         1.1,
            "clip_threshold":      1.0,
        },
    })
    return base


def _secagg_result(n_feat=4) -> dict:
    base = _federated_result(n_feat)
    base.update({
        "model": "SecAgg-Federated",
        "secagg": {
            "protocol":            "SecAgg (Bonawitz et al. 2017)",
            "n_clients":           5,
            "threshold":           3,
            "dropout_rate":        0.2,
            "rounds_completed":    10,
            "rounds_aborted":      0,
            "all_rounds_verified": True,
            "total_overhead_ms":   12.4,
            "per_round_log":       [{"round": i, "n_survived": 4,
                                     "n_dropped": 1, "verified": True}
                                    for i in range(1, 11)],
        },
    })
    return base


# ════════════════════════════════════════════════════════════════════════════
#  LogisticRegressionWrapper
# ════════════════════════════════════════════════════════════════════════════

class TestLogisticRegressionWrapper:

    def test_predict_all_zeros_bias_minus_ten(self):
        """With large negative bias, all predictions should be 0."""
        model = LogisticRegressionWrapper(
            weights=[0.0, 0.0], bias=-10.0,
            feature_cols=["x1","x2"], target_col="y",
        )
        preds = model.predict([[1.0, 1.0], [2.0, 3.0]])
        assert all(p == 0 for p in preds)

    def test_predict_all_ones_bias_plus_ten(self):
        """With large positive bias, all predictions should be 1."""
        model = LogisticRegressionWrapper(
            weights=[0.0, 0.0], bias=10.0,
            feature_cols=["x1","x2"], target_col="y",
        )
        preds = model.predict([[1.0, 1.0], [-1.0, -1.0]])
        assert all(p == 1 for p in preds)

    def test_predict_linearly_separable(self):
        """Simple separable case: x1 > 0 → class 1."""
        model = LogisticRegressionWrapper(
            weights=[5.0, 0.0], bias=0.0,
            feature_cols=["x1","x2"], target_col="y",
        )
        assert model.predict([[1.0, 0.0]])[0] == 1
        assert model.predict([[-1.0, 0.0]])[0] == 0

    def test_predict_output_length(self):
        model = LogisticRegressionWrapper(
            weights=[1.0], bias=0.0,
            feature_cols=["x"], target_col="y",
        )
        rows = [[i * 0.1] for i in range(10)]
        assert len(model.predict(rows)) == 10

    def test_to_dict_round_trip(self):
        model = LogisticRegressionWrapper(
            weights=[1.0, -2.0, 0.5], bias=0.3,
            feature_cols=["a","b","c"], target_col="out",
            means=[0.5, 1.0, -0.5], stds=[1.0, 2.0, 0.5],
        )
        d     = model.to_dict()
        model2 = LogisticRegressionWrapper.from_dict(d)
        assert model2.weights      == model.weights
        assert model2.bias         == model.bias
        assert model2.feature_cols == model.feature_cols
        assert model2.means        == model.means
        assert model2.stds         == model.stds

    def test_normalisation_applied_when_means_given(self):
        """With means/stds set, normalisation shifts predictions."""
        weights = [1.0]
        # Without norm: sigmoid(1.0*5 + 0) → 1
        # With norm:    sigmoid(1.0*(5-5)/1 + 0) = sigmoid(0) → 0 or 1 depending on threshold
        model = LogisticRegressionWrapper(
            weights=weights, bias=0.0,
            feature_cols=["x"], target_col="y",
            means=[5.0], stds=[1.0],
        )
        # (5 - 5) / 1 = 0 → sigmoid(0) = 0.5 → class 1 (≥0.5)
        assert model.predict([[5.0]])[0] == 1
        # (0 - 5) / 1 = -5 → sigmoid(-5) ≈ 0.007 → class 0
        assert model.predict([[0.0]])[0] == 0


# ════════════════════════════════════════════════════════════════════════════
#  MLflowTracker — _log_tags
# ════════════════════════════════════════════════════════════════════════════

class TestLogTags:

    def _run_tags(self, model_type, result, **exp_kwargs):
        tracker = MLflowTracker()
        exp_id  = _make_experiment(model_type, **exp_kwargs)
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            tracker._log_tags(exp, result)
        return _mock_mlflow.set_tags.call_args_list

    def test_base_tags_always_set(self):
        calls = self._run_tags(ModelType.CENTRAL, _central_result())
        all_tags = {}
        for c in calls:
            all_tags.update(c[0][0])
        for key in ("experiment_id","experiment_name","model_type",
                    "target_col","n_features","dp_enabled","secagg_enabled"):
            assert key in all_tags, f"Missing tag: {key}"

    def test_dp_tags_set_when_dp_enabled(self):
        calls = self._run_tags(
            ModelType.DP_FEDERATED, _dp_result(),
            dp_enabled=True, dp_target_epsilon=2.0,
            dp_delta=1e-5, dp_clip_threshold=1.0,
        )
        all_tags = {}
        for c in calls:
            all_tags.update(c[0][0])
        assert "dp_target_epsilon" in all_tags
        assert "dp_delta"          in all_tags
        assert "dp_clip_threshold" in all_tags

    def test_dp_tags_absent_when_dp_disabled(self):
        calls = self._run_tags(ModelType.CENTRAL, _central_result())
        all_tags = {}
        for c in calls:
            all_tags.update(c[0][0])
        assert "dp_target_epsilon" not in all_tags

    def test_secagg_tags_set_when_secagg_enabled(self):
        calls = self._run_tags(
            ModelType.SECAGG_FEDERATED, _secagg_result(),
            secagg_enabled=True, secagg_threshold=3, secagg_dropout=0.2,
        )
        all_tags = {}
        for c in calls:
            all_tags.update(c[0][0])
        assert "secagg_threshold" in all_tags
        assert "secagg_dropout"   in all_tags


# ════════════════════════════════════════════════════════════════════════════
#  MLflowTracker — _log_params
# ════════════════════════════════════════════════════════════════════════════

class TestLogParams:

    def _run_params(self, model_type, result, hp=None, **exp_kwargs):
        tracker = MLflowTracker()
        exp_id  = _make_experiment(model_type, hp=hp, **exp_kwargs)
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            tracker._log_params(exp, result)
        calls = _mock_mlflow.log_params.call_args_list
        merged = {}
        for c in calls:
            merged.update(c[0][0])
        return merged

    def test_central_params_include_epochs_and_lr(self):
        params = self._run_params(
            ModelType.CENTRAL, _central_result(),
            hp={"epochs": 50, "lr": 0.1},
        )
        assert "epochs"     in params
        assert "lr"         in params
        assert "model_type" in params

    def test_federated_params_include_rounds_and_clients(self):
        params = self._run_params(
            ModelType.FEDERATED, _federated_result(),
            hp={"rounds": 10, "local_epochs": 3, "lr": 0.1, "num_clients": 5},
        )
        assert "rounds"       in params
        assert "local_epochs" in params
        assert "num_clients"  in params

    def test_dp_params_include_epsilon_and_delta(self):
        params = self._run_params(
            ModelType.DP_FEDERATED, _dp_result(),
            hp={"rounds": 10, "local_epochs": 3, "lr": 0.1, "num_clients": 5,
                "target_epsilon": 2.0, "delta": 1e-5, "clip_threshold": 1.0},
            dp_enabled=True, dp_target_epsilon=2.0, dp_delta=1e-5,
            dp_clip_threshold=1.0,
        )
        assert "dp_target_epsilon" in params
        assert "dp_delta"          in params
        assert "dp_clip_threshold" in params

    def test_empty_params_stripped(self):
        """Empty-string params must not be logged (clutter MLflow UI)."""
        params = self._run_params(
            ModelType.CENTRAL, _central_result(),
            hp={"epochs": 50, "lr": 0.1},
        )
        assert all(v != "" for v in params.values())


# ════════════════════════════════════════════════════════════════════════════
#  MLflowTracker — _log_metrics
# ════════════════════════════════════════════════════════════════════════════

class TestLogMetrics:

    def _run_metrics(self, result):
        tracker = MLflowTracker()
        tracker._log_metrics(result)
        # Gather all log_metrics calls (summary) and log_metric calls (series)
        summary_calls = _mock_mlflow.log_metrics.call_args_list
        series_calls  = _mock_mlflow.log_metric.call_args_list
        merged = {}
        for c in summary_calls:
            merged.update(c[0][0])
        return merged, series_calls

    def test_summary_metrics_logged_for_central(self):
        merged, _ = self._run_metrics(_central_result())
        for key in ("train_accuracy","test_accuracy","train_f1","test_f1",
                    "train_precision","test_precision","final_loss",
                    "training_time_ms"):
            assert key in merged, f"Missing metric: {key}"

    def test_loss_history_logged_as_series(self):
        result = _central_result()
        _, series = self._run_metrics(result)
        # log_metric is called once per loss history entry (step=N)
        assert len(series) >= len(result["lossHistory"])

    def test_dp_metrics_logged_for_dp_result(self):
        merged, series = self._run_metrics(_dp_result())
        assert "dp_epsilon_final"        in merged
        assert "dp_budget_fraction_used" in merged
        # Epsilon history series
        eps_calls = [c for c in series if c[0] and c[0][0] == "dp_epsilon"]
        assert len(eps_calls) == 10   # 10 rounds

    def test_secagg_metrics_logged_for_secagg_result(self):
        merged, _ = self._run_metrics(_secagg_result())
        assert "secagg_total_overhead_ms" in merged
        assert "secagg_rounds_completed"  in merged
        assert "secagg_rounds_aborted"    in merged

    def test_accuracy_values_match_result(self):
        result = _central_result()
        merged, _ = self._run_metrics(result)
        assert merged["test_accuracy"]  == pytest.approx(0.85)
        assert merged["train_accuracy"] == pytest.approx(0.91)
        assert merged["final_loss"]     == pytest.approx(0.43)


# ════════════════════════════════════════════════════════════════════════════
#  MLflowTracker — _log_artifacts  (model card content)
# ════════════════════════════════════════════════════════════════════════════

class TestLogArtifacts:

    def _capture_model_card(self, exp, result):
        """Write model card to a temp dir and return its content."""
        tracker = MLflowTracker()
        with tempfile.TemporaryDirectory() as tmpdir:
            card_path = os.path.join(tmpdir, "model_card.md")
            tracker._write_model_card(card_path, exp, result)
            return open(card_path).read()

    def test_model_card_has_all_sections(self):
        exp_id = _make_experiment(ModelType.CENTRAL)
        with get_db() as db:
            exp  = ExperimentRepo.get_by_id(db, exp_id)
            card = self._capture_model_card(exp, _central_result())
        for section in ("## Model Overview", "## Performance",
                        "## Training Data", "## Intended Use", "## Limitations"):
            assert section in card, f"Missing section: {section}"

    def test_model_card_includes_accuracy(self):
        exp_id = _make_experiment(ModelType.CENTRAL)
        with get_db() as db:
            exp  = ExperimentRepo.get_by_id(db, exp_id)
            card = self._capture_model_card(exp, _central_result())
        assert "0.8500" in card   # test accuracy
        assert "0.4300" in card   # final loss

    def test_model_card_includes_dp_section_when_dp(self):
        exp_id = _make_experiment(
            ModelType.DP_FEDERATED,
            dp_enabled=True, dp_target_epsilon=2.0,
            dp_delta=1e-5, dp_clip_threshold=1.0,
        )
        with get_db() as db:
            exp  = ExperimentRepo.get_by_id(db, exp_id)
            card = self._capture_model_card(exp, _dp_result())
        assert "## Differential Privacy" in card
        assert "1.74"  in card   # current epsilon
        assert "87.0%" in card   # budget used

    def test_model_card_includes_secagg_section_when_secagg(self):
        exp_id = _make_experiment(
            ModelType.SECAGG_FEDERATED,
            secagg_enabled=True, secagg_threshold=3, secagg_dropout=0.2,
        )
        with get_db() as db:
            exp  = ExperimentRepo.get_by_id(db, exp_id)
            card = self._capture_model_card(exp, _secagg_result())
        assert "## Secure Aggregation" in card
        assert "True"   in card   # all_rounds_verified

    def test_model_card_no_dp_section_for_central(self):
        exp_id = _make_experiment(ModelType.CENTRAL)
        with get_db() as db:
            exp  = ExperimentRepo.get_by_id(db, exp_id)
            card = self._capture_model_card(exp, _central_result())
        assert "## Differential Privacy" not in card

    def test_artifacts_dir_logged(self):
        tracker = MLflowTracker()
        exp_id  = _make_experiment(ModelType.CENTRAL)
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
        tracker._log_artifacts(exp, _central_result())
        assert _mock_mlflow.log_artifacts.called

    def test_privacy_budget_artifact_logged_for_dp(self):
        tracker = MLflowTracker()
        exp_id  = _make_experiment(
            ModelType.DP_FEDERATED,
            dp_enabled=True, dp_target_epsilon=2.0, dp_delta=1e-5,
            dp_clip_threshold=1.0,
        )
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
        # Capture files written to tmpdir
        written = {}
        original_write = tracker._write_json
        def capturing_write(path, data):
            written[os.path.basename(path)] = data
        tracker._write_json = capturing_write
        tracker._write_model_card = lambda *a, **kw: None  # skip card
        tracker._log_artifacts(exp, _dp_result())
        assert "privacy_budget.json" in written
        assert "result_payload.json" in written

    def test_secagg_audit_artifact_logged_for_secagg(self):
        tracker = MLflowTracker()
        exp_id  = _make_experiment(
            ModelType.SECAGG_FEDERATED,
            secagg_enabled=True, secagg_threshold=3, secagg_dropout=0.0,
        )
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
        written = {}
        def capturing_write(path, data):
            written[os.path.basename(path)] = data
        tracker._write_json = capturing_write
        tracker._write_model_card = lambda *a, **kw: None
        tracker._log_artifacts(exp, _secagg_result())
        assert "secagg_audit.json" in written


# ════════════════════════════════════════════════════════════════════════════
#  MLflowTracker — log_run  (full integration)
# ════════════════════════════════════════════════════════════════════════════

class TestLogRun:

    def test_log_run_returns_run_id(self):
        tracker = MLflowTracker()
        exp_id  = _make_experiment(ModelType.CENTRAL)
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
        run_id = tracker.log_run(exp, _central_result())
        assert run_id == "fake-run-id-001"

    def test_log_run_calls_set_experiment(self):
        tracker = MLflowTracker()
        exp_id  = _make_experiment(ModelType.CENTRAL)
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
        tracker.log_run(exp, _central_result())
        assert _mock_mlflow.set_experiment.called

    def test_log_run_calls_start_run(self):
        tracker = MLflowTracker()
        exp_id  = _make_experiment(ModelType.FEDERATED,
                                   hp={"rounds":5,"local_epochs":2,
                                       "lr":0.1,"num_clients":3})
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
        tracker.log_run(exp, _federated_result())
        assert _mock_mlflow.start_run.called

    def test_log_run_survives_mlflow_exception(self):
        """MLflow errors must NOT propagate to training code."""
        tracker = MLflowTracker()
        _mock_mlflow.set_experiment.side_effect = RuntimeError("Server down")
        exp_id  = _make_experiment(ModelType.CENTRAL)
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
        result = tracker.log_run(exp, _central_result())
        assert result is None   # graceful None, not exception

    def test_log_run_disabled_returns_none(self):
        orig = _tracker_module.MLFLOW_ENABLED
        _tracker_module.MLFLOW_ENABLED = False
        tracker = MLflowTracker()
        tracker._enabled = False
        exp_id  = _make_experiment(ModelType.CENTRAL)
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
        result = tracker.log_run(exp, _central_result())
        assert result is None
        _tracker_module.MLFLOW_ENABLED = orig


# ════════════════════════════════════════════════════════════════════════════
#  Repository integration — mark_completed calls MLflow
# ════════════════════════════════════════════════════════════════════════════

class TestRepositoryMLflowIntegration:

    def test_mark_completed_triggers_mlflow(self):
        """mark_completed must call MLflow tracker after DB write."""
        exp_id = _make_experiment(ModelType.CENTRAL)

        log_run_calls = []
        original_get_tracker = _tracker_module.get_tracker

        def fake_get_tracker():
            class FakeTracker:
                def log_run(self, exp, payload):
                    log_run_calls.append((exp.id, payload.get("model")))
                    return "fake-run"
            return FakeTracker()

        import database.repository as repo_module
        orig = repo_module._get_tracker
        repo_module._get_tracker = fake_get_tracker

        try:
            with get_db() as db:
                ExperimentRepo.mark_completed(db, exp_id, _central_result())
        finally:
            repo_module._get_tracker = orig

        assert len(log_run_calls) == 1
        assert log_run_calls[0][0] == exp_id
        assert log_run_calls[0][1] == "Central"

    def test_mark_completed_survives_tracker_exception(self):
        """DB write must succeed even when MLflow tracker crashes."""
        exp_id = _make_experiment(ModelType.CENTRAL)

        import database.repository as repo_module
        orig = repo_module._get_tracker

        def crashing_tracker():
            class BrokenTracker:
                def log_run(self, *a, **kw):
                    raise RuntimeError("MLflow exploded")
            return BrokenTracker()

        repo_module._get_tracker = crashing_tracker
        try:
            with get_db() as db:
                exp = ExperimentRepo.mark_completed(db, exp_id, _central_result())
            # DB write must have succeeded
            assert exp is not None
            with get_db() as db:
                from database import ExperimentStatus
                exp2 = ExperimentRepo.get_by_id(db, exp_id)
                assert exp2.status == ExperimentStatus.COMPLETED
        finally:
            repo_module._get_tracker = orig

    def test_mark_completed_stores_dp_fields(self):
        """DP result fields must reach DB even when MLflow is mocked."""
        exp_id = _make_experiment(
            ModelType.DP_FEDERATED,
            hp={"rounds":5,"local_epochs":2,"lr":0.1,"num_clients":3,
                "target_epsilon":2.0,"delta":1e-5,"clip_threshold":1.0},
            dp_enabled=True, dp_target_epsilon=2.0,
            dp_delta=1e-5, dp_clip_threshold=1.0,
        )
        with get_db() as db:
            ExperimentRepo.mark_completed(db, exp_id, _dp_result())
        with get_db() as db:
            exp = ExperimentRepo.get_by_id(db, exp_id)
            assert exp.result.dp_epsilon_final  == pytest.approx(1.74)
            assert exp.result.dp_budget_summary is not None


# ════════════════════════════════════════════════════════════════════════════
#  mlflow_health()
# ════════════════════════════════════════════════════════════════════════════

class TestMLflowHealth:

    def test_health_ok_when_available(self):
        _tracker_module._MLFLOW_AVAILABLE = True
        _tracker_module.MLFLOW_ENABLED    = True
        _mock_mlflow.MlflowClient.return_value.search_experiments.return_value = []
        result = mlflow_health()
        assert result["mlflow"] == "ok"

    def test_health_disabled_when_flag_false(self):
        orig = _tracker_module.MLFLOW_ENABLED
        _tracker_module.MLFLOW_ENABLED = False
        result = mlflow_health()
        assert result["mlflow"] == "disabled"
        _tracker_module.MLFLOW_ENABLED = orig

    def test_health_unavailable_when_no_package(self):
        orig = _tracker_module._MLFLOW_AVAILABLE
        _tracker_module._MLFLOW_AVAILABLE = False
        result = mlflow_health()
        assert result["mlflow"] == "unavailable"
        _tracker_module._MLFLOW_AVAILABLE = orig

    def test_health_error_on_connection_failure(self):
        _tracker_module._MLFLOW_AVAILABLE = True
        _tracker_module.MLFLOW_ENABLED    = True
        _mock_mlflow.MlflowClient.return_value.search_experiments.side_effect = \
            ConnectionRefusedError("Server unreachable")
        result = mlflow_health()
        assert result["mlflow"] == "error"
        assert "detail" in result
        _mock_mlflow.MlflowClient.return_value.search_experiments.side_effect = None


# ════════════════════════════════════════════════════════════════════════════
#  Flask /api/mlflow/* routes
# ════════════════════════════════════════════════════════════════════════════

class TestMLflowRoutes:

    @pytest.fixture(autouse=True)
    def flask_client(self):
        import os
        os.environ["USE_SQLITE_FALLBACK"] = "true"
        os.environ["CELERY_ASYNC_ENABLED"] = "false"
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        with flask_app.app.test_client() as client:
            self.client = client

    def test_mlflow_status_route_200(self):
        resp = self.client.get("/api/mlflow/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "enabled"      in data
        assert "tracking_uri" in data

    def test_mlflow_experiments_returns_list(self):
        _mock_mlflow.MlflowClient.return_value.search_experiments.return_value = []
        resp = self.client.get("/api/mlflow/experiments")
        # Either 200 (mlflow available) or 503 (not installed)
        assert resp.status_code in (200, 503)

    def test_mlflow_runs_returns_list(self):
        _mock_mlflow.MlflowClient.return_value.search_runs.return_value = []
        resp = self.client.get("/api/mlflow/runs/some-exp-uuid")
        assert resp.status_code in (200, 503)

    def test_mlflow_registry_returns_list(self):
        _mock_mlflow.MlflowClient.return_value.search_registered_models.return_value = []
        resp = self.client.get("/api/mlflow/registry")
        assert resp.status_code in (200, 503)

    def test_health_includes_mlflow_key(self):
        resp = self.client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "mlflow" in data
        assert "version" in data
        assert data["version"] == "8.0.0"


# ── Manual runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    drop_db(); init_db()

    suites = [
        TestLogisticRegressionWrapper,
        TestLogTags,
        TestLogParams,
        TestLogMetrics,
        TestLogArtifacts,
        TestLogRun,
        TestRepositoryMLflowIntegration,
        TestMLflowHealth,
    ]

    passed = failed = 0
    for suite_cls in suites:
        suite   = suite_cls()
        methods = sorted(m for m in dir(suite) if m.startswith("test_"))
        for m in methods:
            drop_db(); init_db()
            _mock_mlflow.reset_mock()
            _mock_mlflow.start_run.return_value = _FakeRunCtx()
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
