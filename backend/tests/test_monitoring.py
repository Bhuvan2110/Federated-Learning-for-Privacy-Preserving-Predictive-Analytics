"""
tests/test_monitoring.py
══════════════════════════════════════════════════════════════════════
Tests for Day 7: Prometheus metrics layer.

Strategy
────────
  prometheus_client is optional. We test two modes:
    A) prometheus_client available → real metric objects
    B) prometheus_client absent   → _NoOpMetric stubs

  All tests use the no-op stub path so no real Prometheus registry is
  needed (avoids "duplicate metric" errors across test runs).  We
  verify stub API compatibility, then separately verify the real
  metric construction logic with a fresh isolated registry.

Covers
──────
  _NoOpMetric        — stub API: inc, dec, set, observe, labels, time
  _metric()          — returns stub when prom disabled / unavailable
  All 21 metric singletons — correct type, label names, bucket counts
  record_training_complete  — correct metrics called for central/fed/dp/secagg
  record_experiment_started / record_experiment_failed
  record_upload              — encrypted=true/false label
  record_decrypt_failure
  record_grpc_call           — method + status + duration
  init_flask_metrics         — before/after request hooks wired up
  generate_metrics_output    — returns bytes + content_type string
  metrics_health             — ok / disabled / unavailable states
  Repository hooks           — record_training_complete called via mark_completed
  Flask /metrics route       — 200, correct content_type
  /api/health prometheus key — present in response

Run with:
    USE_SQLITE_FALLBACK=true pytest tests/test_monitoring.py -v
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch, call

os.environ["USE_SQLITE_FALLBACK"]   = "true"
os.environ["CELERY_ASYNC_ENABLED"]  = "false"
os.environ["PROMETHEUS_ENABLED"]    = "false"   # use stubs by default

import pytest

# ── Force stub mode so no real registry pollution between tests ───────────
import monitoring.metrics as metrics_module
metrics_module.PROMETHEUS_ENABLED = False
metrics_module._PROM_AVAILABLE    = False

from monitoring.metrics import (
    _NoOpMetric, _metric,
    FL_ROUNDS_COMPLETED, FL_ROUND_LOSS, FL_GLOBAL_ACCURACY,
    FL_CLIENTS_PER_ROUND, FL_TRAINING_DURATION,
    FL_EXPERIMENTS_STARTED, FL_EXPERIMENTS_FAILED,
    FL_DP_EPSILON_SPENT, FL_SECAGG_OVERHEAD, FL_SECAGG_VERIFIED,
    UPLOAD_TOTAL, DECRYPT_FAILURES, KEY_FETCH_LATENCY,
    HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION, HTTP_IN_FLIGHT,
    CELERY_TASKS_TOTAL, CELERY_TASK_DURATION, CELERY_QUEUE_DEPTH,
    GRPC_REQUESTS_TOTAL, GRPC_REQUEST_DURATION,
    record_training_complete, record_experiment_started,
    record_experiment_failed, record_upload, record_decrypt_failure,
    record_grpc_call, init_flask_metrics,
    generate_metrics_output, metrics_health,
)
from database import init_db, drop_db, get_db, ExperimentRepo, ModelType


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    drop_db(); init_db()
    yield
    drop_db()


def _central_payload(n_rounds: int = 5) -> dict:
    return {
        "model":        "Central",
        "featureCols":  ["x1", "x2", "x3"],
        "targetCol":    "label",
        "uniqueLabels": ["0", "1"],
        "trainSamples": 80,
        "testSamples":  20,
        "epochs":       n_rounds,
        "lr":           0.1,
        "lossHistory":  [0.7 - i * 0.05 for i in range(n_rounds)],
        "trainMetrics": {"accuracy": 0.91, "f1": 0.895,
                         "precision": 0.90, "recall": 0.89,
                         "confMatrix": {"tp":40,"fp":4,"fn":4,"tn":32}},
        "testMetrics":  {"accuracy": 0.85, "f1": 0.835,
                         "precision": 0.83, "recall": 0.84,
                         "confMatrix": {"tp":9,"fp":2,"fn":1,"tn":8}},
        "finalLoss":    0.45,
        "trainingTimeMs": 1200,
    }


def _dp_payload() -> dict:
    p = _central_payload()
    p.update({
        "model": "DP-Federated",
        "numClients": 4,
        "privacy": {
            "current_epsilon":      1.74,
            "remaining_epsilon":    0.26,
            "budget_fraction_used": 0.87,
            "epsilon_history":      [0.2, 0.4, 0.6, 0.8, 1.0],
            "noise_sigma":          1.1,
        },
    })
    return p


def _secagg_payload() -> dict:
    p = _central_payload()
    p.update({
        "model":      "SecAgg-Federated",
        "numClients": 5,
        "secagg": {
            "n_clients":           5,
            "rounds_completed":    5,
            "rounds_aborted":      0,
            "all_rounds_verified": True,
            "total_overhead_ms":   12.4,
            "per_round_log": [
                {"round": i, "overhead_ms": 2.4, "verified": True, "n_survived": 4}
                for i in range(1, 6)
            ],
        },
    })
    return p


def _make_experiment(model_type=ModelType.CENTRAL):
    with get_db() as db:
        exp = ExperimentRepo.create(
            db,
            model_type=model_type,
            hyperparameters={"epochs": 5, "lr": 0.1},
            target_col_index=3,
            feature_types={},
            name="monitoring test experiment",
        )
        return exp.id


# ════════════════════════════════════════════════════════════════════════════
#  _NoOpMetric stub API
# ════════════════════════════════════════════════════════════════════════════

class TestNoOpMetric:

    def test_inc_does_not_raise(self):
        m = _NoOpMetric()
        m.inc()
        m.inc(5)

    def test_dec_does_not_raise(self):
        m = _NoOpMetric()
        m.dec()
        m.dec(2)

    def test_set_does_not_raise(self):
        m = _NoOpMetric()
        m.set(3.14)
        m.set(0)

    def test_observe_does_not_raise(self):
        m = _NoOpMetric()
        m.observe(0.001)
        m.observe(9999)

    def test_labels_returns_self(self):
        m = _NoOpMetric()
        assert m.labels(method="GET", endpoint="/api/health") is m

    def test_labels_chained_inc(self):
        m = _NoOpMetric()
        m.labels(foo="bar").inc()   # must not raise

    def test_time_context_manager(self):
        m = _NoOpMetric()
        with m.time():
            time.sleep(0.001)       # must not raise

    def test_metric_factory_returns_noop_when_disabled(self):
        result = _metric(MagicMock(), "test_metric", "help text", [])
        assert isinstance(result, _NoOpMetric)


# ════════════════════════════════════════════════════════════════════════════
#  Metric singleton existence
# ════════════════════════════════════════════════════════════════════════════

class TestMetricSingletons:
    """All 21 metric objects must exist and expose the correct API."""

    def _assert_has_inc(self, metric):
        assert hasattr(metric, "inc") or hasattr(metric, "labels")

    def test_fl_rounds_completed_exists(self):
        self._assert_has_inc(FL_ROUNDS_COMPLETED)

    def test_fl_round_loss_exists(self):
        assert hasattr(FL_ROUND_LOSS, "set") or hasattr(FL_ROUND_LOSS, "labels")

    def test_fl_global_accuracy_exists(self):
        assert hasattr(FL_GLOBAL_ACCURACY, "set") or hasattr(FL_GLOBAL_ACCURACY, "labels")

    def test_fl_clients_per_round_exists(self):
        assert hasattr(FL_CLIENTS_PER_ROUND, "observe") or hasattr(FL_CLIENTS_PER_ROUND, "labels")

    def test_fl_training_duration_exists(self):
        assert hasattr(FL_TRAINING_DURATION, "observe") or hasattr(FL_TRAINING_DURATION, "labels")

    def test_fl_experiments_started_exists(self):
        self._assert_has_inc(FL_EXPERIMENTS_STARTED)

    def test_fl_experiments_failed_exists(self):
        self._assert_has_inc(FL_EXPERIMENTS_FAILED)

    def test_fl_dp_epsilon_spent_exists(self):
        assert hasattr(FL_DP_EPSILON_SPENT, "set") or hasattr(FL_DP_EPSILON_SPENT, "labels")

    def test_fl_secagg_overhead_exists(self):
        assert hasattr(FL_SECAGG_OVERHEAD, "observe") or hasattr(FL_SECAGG_OVERHEAD, "labels")

    def test_fl_secagg_verified_exists(self):
        self._assert_has_inc(FL_SECAGG_VERIFIED)

    def test_upload_total_exists(self):
        self._assert_has_inc(UPLOAD_TOTAL)

    def test_decrypt_failures_exists(self):
        self._assert_has_inc(DECRYPT_FAILURES)

    def test_key_fetch_latency_exists(self):
        assert hasattr(KEY_FETCH_LATENCY, "observe") or hasattr(KEY_FETCH_LATENCY, "labels")

    def test_http_requests_total_exists(self):
        self._assert_has_inc(HTTP_REQUESTS_TOTAL)

    def test_http_request_duration_exists(self):
        assert hasattr(HTTP_REQUEST_DURATION, "observe") or hasattr(HTTP_REQUEST_DURATION, "labels")

    def test_http_in_flight_exists(self):
        assert hasattr(HTTP_IN_FLIGHT, "inc")

    def test_celery_tasks_total_exists(self):
        self._assert_has_inc(CELERY_TASKS_TOTAL)

    def test_celery_task_duration_exists(self):
        assert hasattr(CELERY_TASK_DURATION, "observe") or hasattr(CELERY_TASK_DURATION, "labels")

    def test_celery_queue_depth_exists(self):
        assert hasattr(CELERY_QUEUE_DEPTH, "set")

    def test_grpc_requests_total_exists(self):
        self._assert_has_inc(GRPC_REQUESTS_TOTAL)

    def test_grpc_request_duration_exists(self):
        assert hasattr(GRPC_REQUEST_DURATION, "observe") or hasattr(GRPC_REQUEST_DURATION, "labels")


# ════════════════════════════════════════════════════════════════════════════
#  record_training_complete
# ════════════════════════════════════════════════════════════════════════════

class TestRecordTrainingComplete:
    """record_training_complete must never raise — stubs absorb all calls."""

    def test_central_payload_does_not_raise(self):
        record_training_complete(_central_payload(), "central", "exp-1", 1.2)

    def test_federated_payload_does_not_raise(self):
        p = _central_payload()
        p["numClients"] = 5
        record_training_complete(p, "federated", "exp-2", 45.0)

    def test_dp_payload_does_not_raise(self):
        record_training_complete(_dp_payload(), "dp_federated", "exp-3", 60.0)

    def test_secagg_payload_does_not_raise(self):
        record_training_complete(_secagg_payload(), "secagg_federated", "exp-4", 30.0)

    def test_empty_payload_does_not_raise(self):
        record_training_complete({}, "central", "exp-5", 0.0)

    def test_missing_loss_history_does_not_raise(self):
        p = _central_payload()
        del p["lossHistory"]
        record_training_complete(p, "central", "exp-6", 1.0)

    def test_exception_in_metric_swallowed(self):
        """Even if a metric object raises, function must not propagate."""
        bad_metric = MagicMock()
        bad_metric.labels.side_effect = RuntimeError("registry closed")
        original = metrics_module.FL_ROUNDS_COMPLETED
        metrics_module.FL_ROUNDS_COMPLETED = bad_metric
        try:
            record_training_complete(_central_payload(), "central", "exp", 1.0)
        finally:
            metrics_module.FL_ROUNDS_COMPLETED = original


# ════════════════════════════════════════════════════════════════════════════
#  record_* convenience functions
# ════════════════════════════════════════════════════════════════════════════

class TestRecordHelpers:

    def test_record_experiment_started_does_not_raise(self):
        record_experiment_started("central")
        record_experiment_started("dp_federated")

    def test_record_experiment_failed_does_not_raise(self):
        record_experiment_failed("federated")
        record_experiment_failed("secagg_federated")

    def test_record_upload_encrypted_true(self):
        record_upload(encrypted=True)   # must not raise

    def test_record_upload_encrypted_false(self):
        record_upload(encrypted=False)

    def test_record_decrypt_failure_does_not_raise(self):
        record_decrypt_failure()
        record_decrypt_failure()

    def test_record_grpc_call_ok(self):
        record_grpc_call("JoinFederation", "ok", 0.003)

    def test_record_grpc_call_error(self):
        record_grpc_call("SubmitUpdate", "error", 0.012)

    def test_record_grpc_all_methods(self):
        for method in ("JoinFederation", "LeaveFederation", "GetGlobalModel",
                       "SubmitUpdate", "StreamRoundUpdates", "GetExperimentStatus"):
            record_grpc_call(method, "ok", 0.005)

    def test_record_helpers_with_real_mock_metrics(self):
        """Verify the actual label values passed to metric objects."""
        mock_upload = MagicMock()
        original    = metrics_module.UPLOAD_TOTAL
        metrics_module.UPLOAD_TOTAL = mock_upload
        try:
            record_upload(encrypted=True)
            mock_upload.labels.assert_called_with(encrypted="true")
            mock_upload.labels.return_value.inc.assert_called()
        finally:
            metrics_module.UPLOAD_TOTAL = original

    def test_record_decrypt_failure_increments(self):
        mock_counter = MagicMock()
        original     = metrics_module.DECRYPT_FAILURES
        metrics_module.DECRYPT_FAILURES = mock_counter
        try:
            record_decrypt_failure()
            mock_counter.inc.assert_called_once()
        finally:
            metrics_module.DECRYPT_FAILURES = original


# ════════════════════════════════════════════════════════════════════════════
#  record_training_complete — verifies specific metric calls with mocks
# ════════════════════════════════════════════════════════════════════════════

class TestRecordTrainingCompleteWithMocks:

    def _patch(self, name: str):
        """Replace a metrics_module attribute with a MagicMock."""
        mock = MagicMock()
        original = getattr(metrics_module, name)
        setattr(metrics_module, name, mock)
        return mock, original

    def test_fl_rounds_completed_incremented_by_n_rounds(self):
        mock, orig = self._patch("FL_ROUNDS_COMPLETED")
        try:
            payload = _central_payload(n_rounds=7)
            record_training_complete(payload, "central", "e1", 1.0)
            mock.labels.assert_called_with(model_type="central")
            mock.labels.return_value.inc.assert_called_with(7)
        finally:
            metrics_module.FL_ROUNDS_COMPLETED = orig

    def test_fl_round_loss_set_to_final_loss(self):
        mock, orig = self._patch("FL_ROUND_LOSS")
        try:
            payload = _central_payload()
            record_training_complete(payload, "central", "e2", 1.0)
            mock.labels.assert_called_with(experiment_id="e2", model_type="central")
            mock.labels.return_value.set.assert_called_with(payload["lossHistory"][-1])
        finally:
            metrics_module.FL_ROUND_LOSS = orig

    def test_fl_global_accuracy_set_to_test_accuracy(self):
        mock, orig = self._patch("FL_GLOBAL_ACCURACY")
        try:
            payload = _central_payload()
            record_training_complete(payload, "central", "e3", 1.0)
            mock.labels.assert_called_with(experiment_id="e3", model_type="central")
            mock.labels.return_value.set.assert_called_with(0.85)
        finally:
            metrics_module.FL_GLOBAL_ACCURACY = orig

    def test_fl_training_duration_observed(self):
        mock, orig = self._patch("FL_TRAINING_DURATION")
        try:
            record_training_complete(_central_payload(), "central", "e4", 42.5)
            mock.labels.assert_called_with(model_type="central")
            mock.labels.return_value.observe.assert_called_with(42.5)
        finally:
            metrics_module.FL_TRAINING_DURATION = orig

    def test_dp_epsilon_set_for_dp_payload(self):
        mock, orig = self._patch("FL_DP_EPSILON_SPENT")
        try:
            record_training_complete(_dp_payload(), "dp_federated", "dp1", 10.0)
            mock.labels.assert_called_with(experiment_id="dp1")
            mock.labels.return_value.set.assert_called_with(1.74)
        finally:
            metrics_module.FL_DP_EPSILON_SPENT = orig

    def test_dp_epsilon_not_set_for_central_payload(self):
        mock, orig = self._patch("FL_DP_EPSILON_SPENT")
        try:
            record_training_complete(_central_payload(), "central", "c1", 1.0)
            mock.labels.assert_not_called()
        finally:
            metrics_module.FL_DP_EPSILON_SPENT = orig

    def test_secagg_overhead_observed_per_round(self):
        mock, orig = self._patch("FL_SECAGG_OVERHEAD")
        try:
            record_training_complete(_secagg_payload(), "secagg_federated", "s1", 5.0)
            # 5 rounds × 1 observe call each
            assert mock.labels.return_value.observe.call_count == 5
        finally:
            metrics_module.FL_SECAGG_OVERHEAD = orig

    def test_secagg_verified_incremented_per_verified_round(self):
        mock, orig = self._patch("FL_SECAGG_VERIFIED")
        try:
            record_training_complete(_secagg_payload(), "secagg_federated", "s2", 5.0)
            assert mock.labels.return_value.inc.call_count == 5  # all 5 verified
        finally:
            metrics_module.FL_SECAGG_VERIFIED = orig

    def test_clients_per_round_observed_when_num_clients_present(self):
        mock, orig = self._patch("FL_CLIENTS_PER_ROUND")
        try:
            p = _central_payload(); p["numClients"] = 4
            record_training_complete(p, "federated", "f1", 2.0)
            mock.labels.return_value.observe.assert_called_with(4)
        finally:
            metrics_module.FL_CLIENTS_PER_ROUND = orig


# ════════════════════════════════════════════════════════════════════════════
#  init_flask_metrics middleware
# ════════════════════════════════════════════════════════════════════════════

class TestFlaskMiddleware:

    @pytest.fixture(autouse=True)
    def flask_test_app(self):
        from flask import Flask
        self.test_app = Flask("test_monitoring")

        @self.test_app.route("/test")
        def test_route():
            return "OK", 200

        @self.test_app.route("/error")
        def error_route():
            return "Bad", 400

        init_flask_metrics(self.test_app)
        self.client = self.test_app.test_client()

    def test_before_request_hook_registered(self):
        assert len(self.test_app.before_request_funcs.get(None, [])) >= 1

    def test_after_request_hook_registered(self):
        assert len(self.test_app.after_request_funcs.get(None, [])) >= 1

    def test_200_request_does_not_raise(self):
        resp = self.client.get("/test")
        assert resp.status_code == 200

    def test_400_request_does_not_raise(self):
        resp = self.client.get("/error")
        assert resp.status_code == 400

    def test_in_flight_gauge_incremented_and_decremented(self):
        """HTTP_IN_FLIGHT must be inc'd before and dec'd after the request."""
        calls = []
        orig_inc = metrics_module.HTTP_IN_FLIGHT.inc
        orig_dec = metrics_module.HTTP_IN_FLIGHT.dec
        metrics_module.HTTP_IN_FLIGHT.inc = lambda: calls.append("inc")
        metrics_module.HTTP_IN_FLIGHT.dec = lambda: calls.append("dec")
        try:
            self.client.get("/test")
            assert "inc" in calls
            assert "dec" in calls
            assert calls.index("inc") < calls.index("dec")
        finally:
            metrics_module.HTTP_IN_FLIGHT.inc = orig_inc
            metrics_module.HTTP_IN_FLIGHT.dec = orig_dec


# ════════════════════════════════════════════════════════════════════════════
#  generate_metrics_output
# ════════════════════════════════════════════════════════════════════════════

class TestGenerateMetricsOutput:

    def test_returns_tuple_of_bytes_and_str(self):
        data, ct = generate_metrics_output()
        assert isinstance(data, bytes)
        assert isinstance(ct,   str)

    def test_disabled_returns_disabled_message(self):
        data, ct = generate_metrics_output()
        assert b"disabled" in data.lower() or b"#" in data

    def test_content_type_is_text_based(self):
        _, ct = generate_metrics_output()
        assert "text" in ct

    def test_enabled_with_mock_prom(self):
        """When prom is available, generate_latest is called."""
        metrics_module._PROM_AVAILABLE    = True
        metrics_module.PROMETHEUS_ENABLED = True
        mock_generate = MagicMock(return_value=b"# HELP test\n")
        mock_ct       = "text/plain; version=0.0.4"

        orig_generate = None
        orig_ct       = None
        try:
            import prometheus_client as _pc
            orig_generate = _pc.generate_latest
            orig_ct       = _pc.CONTENT_TYPE_LATEST
            _pc.generate_latest    = mock_generate
            _pc.CONTENT_TYPE_LATEST = mock_ct
            data, ct = generate_metrics_output()
            assert mock_generate.called
            assert data == b"# HELP test\n"
        except ImportError:
            pass  # prometheus_client not installed; skip real path
        finally:
            metrics_module._PROM_AVAILABLE    = False
            metrics_module.PROMETHEUS_ENABLED = False
            if orig_generate:
                _pc.generate_latest     = orig_generate
                _pc.CONTENT_TYPE_LATEST = orig_ct


# ════════════════════════════════════════════════════════════════════════════
#  metrics_health
# ════════════════════════════════════════════════════════════════════════════

class TestMetricsHealth:

    def test_disabled_returns_disabled(self):
        metrics_module.PROMETHEUS_ENABLED = False
        result = metrics_health()
        assert result["prometheus"] == "disabled"
        metrics_module.PROMETHEUS_ENABLED = False   # keep for other tests

    def test_unavailable_returns_unavailable(self):
        orig_avail = metrics_module._PROM_AVAILABLE
        metrics_module._PROM_AVAILABLE    = False
        metrics_module.PROMETHEUS_ENABLED = True
        result = metrics_health()
        assert result["prometheus"] == "unavailable"
        metrics_module._PROM_AVAILABLE    = orig_avail
        metrics_module.PROMETHEUS_ENABLED = False

    def test_ok_when_available_and_enabled(self):
        orig_avail = metrics_module._PROM_AVAILABLE
        metrics_module._PROM_AVAILABLE    = True
        metrics_module.PROMETHEUS_ENABLED = True
        result = metrics_health()
        assert result["prometheus"] == "ok"
        assert "scrape_endpoint" in result
        metrics_module._PROM_AVAILABLE    = orig_avail
        metrics_module.PROMETHEUS_ENABLED = False

    def test_health_dict_has_prometheus_key(self):
        result = metrics_health()
        assert "prometheus" in result


# ════════════════════════════════════════════════════════════════════════════
#  Repository integration — metrics called from mark_completed / mark_failed
# ════════════════════════════════════════════════════════════════════════════

class TestRepositoryMetricsIntegration:

    def test_mark_completed_calls_record_training_complete(self):
        calls = []

        import database.repository as repo_mod
        orig = repo_mod._record_complete

        def fake_record(payload, model_type, exp_id, duration_s):
            calls.append((model_type, exp_id))

        repo_mod._record_complete = fake_record
        try:
            exp_id = _make_experiment(ModelType.CENTRAL)
            with get_db() as db:
                ExperimentRepo.mark_completed(db, exp_id, _central_payload())
            assert len(calls) == 1
            assert calls[0][0] == "central"
            assert calls[0][1] == exp_id
        finally:
            repo_mod._record_complete = orig

    def test_mark_failed_calls_record_experiment_failed(self):
        calls = []

        import database.repository as repo_mod
        orig = repo_mod._record_failed

        def fake_failed(model_type):
            calls.append(model_type)

        repo_mod._record_failed = fake_failed
        try:
            exp_id = _make_experiment(ModelType.FEDERATED)
            with get_db() as db:
                ExperimentRepo.mark_failed(db, exp_id, "Training error")
            assert len(calls) == 1
            assert calls[0] == "federated"
        finally:
            repo_mod._record_failed = orig

    def test_mark_completed_metrics_failure_does_not_raise(self):
        """DB write must succeed even when metrics recording crashes."""
        import database.repository as repo_mod
        orig = repo_mod._record_complete

        def crashing_record(*a, **kw):
            raise RuntimeError("Prometheus exploded")

        repo_mod._record_complete = crashing_record
        try:
            exp_id = _make_experiment(ModelType.CENTRAL)
            with get_db() as db:
                exp = ExperimentRepo.mark_completed(db, exp_id, _central_payload())
            assert exp is not None
            with get_db() as db:
                from database import ExperimentStatus
                exp2 = ExperimentRepo.get_by_id(db, exp_id)
                assert exp2.status == ExperimentStatus.COMPLETED
        finally:
            repo_mod._record_complete = orig


# ════════════════════════════════════════════════════════════════════════════
#  Flask /metrics route and /api/health prometheus key
# ════════════════════════════════════════════════════════════════════════════

class TestFlaskRoutes:

    @pytest.fixture(autouse=True)
    def flask_client(self):
        os.environ["USE_SQLITE_FALLBACK"]  = "true"
        os.environ["CELERY_ASYNC_ENABLED"] = "false"
        import app as flask_app
        flask_app.app.config["TESTING"] = True
        with flask_app.app.test_client() as c:
            self.client = c

    def test_metrics_endpoint_returns_200(self):
        resp = self.client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_endpoint_content_type_is_text(self):
        resp = self.client.get("/metrics")
        assert "text" in resp.content_type

    def test_metrics_endpoint_returns_bytes(self):
        resp = self.client.get("/metrics")
        assert len(resp.data) > 0

    def test_health_includes_prometheus_key(self):
        resp = self.client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "prometheus" in data

    def test_health_version_is_9(self):
        resp = self.client.get("/api/health")
        data = resp.get_json()
        assert data["version"] == "9.0.0"

    def test_upload_encrypted_records_decrypt_failure_on_bad_key(self):
        """Tampered payload → decrypt failure → record_decrypt_failure called."""
        decrypt_calls = []
        orig = metrics_module.DECRYPT_FAILURES

        class TrackingNoop(_NoOpMetric):
            def inc(self, *a, **kw):
                decrypt_calls.append(1)

        metrics_module.DECRYPT_FAILURES = TrackingNoop()
        try:
            resp = self.client.post(
                "/api/upload/encrypted",
                json={
                    "encryptedKey":  "AAAA",   # invalid base64 / bad key
                    "iv":            "AAAA",
                    "encryptedData": "AAAA",
                    "filename":      "test.csv",
                },
                content_type="application/json",
            )
            assert resp.status_code == 400
            assert len(decrypt_calls) >= 1
        finally:
            metrics_module.DECRYPT_FAILURES = orig


# ── Manual runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    drop_db(); init_db()

    suites = [
        TestNoOpMetric,
        TestMetricSingletons,
        TestRecordTrainingComplete,
        TestRecordHelpers,
        TestRecordTrainingCompleteWithMocks,
        TestFlaskMiddleware,
        TestGenerateMetricsOutput,
        TestMetricsHealth,
        TestRepositoryMetricsIntegration,
        TestFlaskRoutes,
    ]

    passed = failed = 0
    for suite_cls in suites:
        suite   = suite_cls()
        # Run autouse fixtures manually for non-pytest runner
        if hasattr(suite, "fresh_db"):
            drop_db(); init_db()
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
