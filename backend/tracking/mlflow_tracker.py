"""
tracking/mlflow_tracker.py
══════════════════════════════════════════════════════════════════════
MLflow experiment tracking integration  (Day 6)

What it does
────────────
  Every completed training run is logged to MLflow with:
    • Parameters  — hyperparameters (epochs, lr, rounds, ε, C, …)
    • Metrics     — accuracy, F1, loss per round (as a time series)
    • Tags        — model type, target column, feature count, DP/SecAgg flags
    • Artifacts   — model weights JSON, confusion matrix, privacy budget summary
    • Model       — registered in the MLflow Model Registry under a versioned name

Architecture
────────────
  MLflow is configured via MLFLOW_TRACKING_URI (default: file-based
  local store at ./mlruns for dev, remote tracking server in production).
  The tracker is intentionally non-blocking: if MLflow is unavailable,
  logging is skipped with a warning and training results are still
  saved to Postgres.

  Tracking is invoked from ExperimentRepo.mark_completed() — a single
  call to log_run(experiment_row, result_payload) after the DB write.

  Model registry uses the pattern:
    model name = "tm-{model_type}-{target_col}"
    version     = auto-incremented by MLflow
    stage       = None (use MLflow UI to promote to Staging/Production)

Environment variables
─────────────────────
  MLFLOW_TRACKING_URI   — MLflow server URI (default: ./mlruns)
  MLFLOW_EXPERIMENT_NAME — MLflow experiment name prefix (default: training-models)
  MLFLOW_REGISTRY_URI   — Model registry URI (default: same as tracking URI)
  MLFLOW_ENABLED        — set to "false" to disable entirely (default: true)
  MLFLOW_REGISTER_MODELS — set to "false" to skip model registry (default: true)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI",    "./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "training-models")
MLFLOW_ENABLED         = os.getenv("MLFLOW_ENABLED",         "true").lower() == "true"
MLFLOW_REGISTER_MODELS = os.getenv("MLFLOW_REGISTER_MODELS", "true").lower() == "true"

# ── Optional MLflow import ────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn  # for model flavour (we use pyfunc for custom models)
    _MLFLOW_AVAILABLE = True
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning(
        "mlflow not installed — experiment tracking disabled.\n"
        "Install with: pip install mlflow"
    )


# ════════════════════════════════════════════════════════════════════════════
#  Model wrapper for MLflow pyfunc flavour
# ════════════════════════════════════════════════════════════════════════════

class LogisticRegressionWrapper:
    """
    MLflow pyfunc-compatible wrapper for the pure-Python logistic regression.

    Stores weights and bias as numpy-free JSON so the model can be logged
    and reloaded without any ML framework dependency.

    In production (Phase 3) this would wrap MLP, GBT, etc.
    """

    def __init__(self, weights: list[float], bias: float,
                 feature_cols: list[str], target_col: str,
                 means: list[float] | None = None,
                 stds:  list[float] | None = None) -> None:
        self.weights      = weights
        self.bias         = bias
        self.feature_cols = feature_cols
        self.target_col   = target_col
        self.means        = means or []
        self.stds         = stds  or []

    def predict(self, data: list[list[float]]) -> list[int]:
        """Binary logistic regression inference."""
        import math

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

        def dot(a: list[float], b: list[float]) -> float:
            return sum(ai * bi for ai, bi in zip(a, b))

        def normalise_row(row: list[float]) -> list[float]:
            if not self.means:
                return row
            return [(row[j] - self.means[j]) / (self.stds[j] or 1.0)
                    for j in range(len(row))]

        return [
            1 if sigmoid(dot(normalise_row(row), self.weights) + self.bias) >= 0.5 else 0
            for row in data
        ]

    def to_dict(self) -> dict:
        return {
            "weights":      self.weights,
            "bias":         self.bias,
            "feature_cols": self.feature_cols,
            "target_col":   self.target_col,
            "means":        self.means,
            "stds":         self.stds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticRegressionWrapper":
        return cls(
            weights      = d["weights"],
            bias         = d["bias"],
            feature_cols = d.get("feature_cols", []),
            target_col   = d.get("target_col", ""),
            means        = d.get("means", []),
            stds         = d.get("stds",  []),
        )


# ════════════════════════════════════════════════════════════════════════════
#  MLflow Tracker
# ════════════════════════════════════════════════════════════════════════════

class MLflowTracker:
    """
    Logs a completed training run to MLflow.

    Usage
    ─────
        tracker = MLflowTracker()
        tracker.log_run(experiment_row, result_payload)

    Parameters
    ──────────
    experiment_row  — SQLAlchemy Experiment ORM object (from DB)
    result_payload  — the dict returned by central_train() / federated_train() / …

    Returns the MLflow run_id (str) or None if MLflow is unavailable.
    """

    def __init__(self) -> None:
        self._enabled = MLFLOW_ENABLED and _MLFLOW_AVAILABLE

    # ── Public entry point ────────────────────────────────────────────────

    def log_run(
        self,
        experiment_row: Any,            # Experiment ORM object
        result_payload: dict,
    ) -> Optional[str]:
        """
        Log a completed experiment to MLflow.
        Returns MLflow run_id or None on failure/disabled.
        """
        if not self._enabled:
            logger.debug("MLflow tracking disabled — skipping log_run")
            return None

        try:
            return self._log_run_inner(experiment_row, result_payload)
        except Exception as exc:
            logger.warning("MLflow log_run failed (non-fatal): %s", exc)
            return None

    # ── Inner implementation ──────────────────────────────────────────────

    def _log_run_inner(self, exp, result_payload: dict) -> str:
        mlflow_exp_name = f"{MLFLOW_EXPERIMENT_NAME}/{exp.model_type.value}"
        mlflow.set_experiment(mlflow_exp_name)

        with mlflow.start_run(run_name=exp.name) as run:
            run_id = run.info.run_id

            # ── Tags ──────────────────────────────────────────────────────
            self._log_tags(exp, result_payload)

            # ── Parameters ────────────────────────────────────────────────
            self._log_params(exp, result_payload)

            # ── Metrics ───────────────────────────────────────────────────
            self._log_metrics(result_payload)

            # ── Artifacts ─────────────────────────────────────────────────
            self._log_artifacts(exp, result_payload)

            # ── Model registry ────────────────────────────────────────────
            if MLFLOW_REGISTER_MODELS:
                self._register_model(exp, result_payload, run_id)

            logger.info(
                "MLflow run logged: exp=%s  run_id=%s  uri=%s",
                exp.id, run_id, MLFLOW_TRACKING_URI,
            )
            return run_id

    # ── Tags ──────────────────────────────────────────────────────────────

    def _log_tags(self, exp, result_payload: dict) -> None:
        fc = result_payload.get("featureCols") or []
        mlflow.set_tags({
            # Identity
            "experiment_id":   exp.id,
            "experiment_name": exp.name,
            "model_type":      exp.model_type.value,
            # Data
            "target_col":      result_payload.get("targetCol", ""),
            "n_features":      len(fc),
            "feature_cols":    json.dumps(fc),
            "unique_labels":   json.dumps(result_payload.get("uniqueLabels", [])),
            # Privacy flags
            "dp_enabled":      str(exp.dp_enabled),
            "secagg_enabled":  str(exp.secagg_enabled),
        })

        # DP-specific tags
        if exp.dp_enabled:
            mlflow.set_tags({
                "dp_target_epsilon":   str(exp.dp_target_epsilon),
                "dp_delta":            str(exp.dp_delta),
                "dp_clip_threshold":   str(exp.dp_clip_threshold),
            })

        # SecAgg-specific tags
        if exp.secagg_enabled:
            mlflow.set_tags({
                "secagg_threshold": str(exp.secagg_threshold),
                "secagg_dropout":   str(exp.secagg_dropout),
            })

    # ── Parameters ────────────────────────────────────────────────────────

    def _log_params(self, exp, result_payload: dict) -> None:
        hp = exp.hyperparameters or {}

        # Common params
        params = {
            "model_type":    exp.model_type.value,
            "lr":            hp.get("lr", result_payload.get("lr", "")),
            "train_samples": result_payload.get("trainSamples", ""),
            "test_samples":  result_payload.get("testSamples", ""),
        }

        # Model-type-specific params
        mt = exp.model_type.value
        if mt == "central":
            params["epochs"] = hp.get("epochs", result_payload.get("epochs", ""))

        elif mt in ("federated", "dp_federated", "secagg_federated"):
            params["rounds"]       = hp.get("rounds", result_payload.get("rounds", ""))
            params["local_epochs"] = hp.get("local_epochs", result_payload.get("localEpochs", ""))
            params["num_clients"]  = hp.get("num_clients", result_payload.get("numClients", ""))

        if mt == "dp_federated":
            privacy = result_payload.get("privacy") or {}
            params["dp_target_epsilon"]   = hp.get("target_epsilon", "")
            params["dp_delta"]            = hp.get("delta", "")
            params["dp_clip_threshold"]   = hp.get("clip_threshold", "")
            params["dp_noise_multiplier"] = privacy.get("noise_multiplier",
                                             hp.get("noise_multiplier", ""))

        if mt == "secagg_federated":
            params["secagg_threshold"] = hp.get("secagg_threshold", "")
            params["secagg_dropout"]   = hp.get("dropout_rate", "")

        # Strip empty strings so MLflow UI isn't cluttered
        mlflow.log_params({k: v for k, v in params.items() if v != ""})

    # ── Metrics ───────────────────────────────────────────────────────────

    def _log_metrics(self, result_payload: dict) -> None:
        tm = result_payload.get("trainMetrics", {})
        te = result_payload.get("testMetrics",  {})

        # Summary metrics (step=0 → appear on the run overview)
        mlflow.log_metrics({
            "train_accuracy":  tm.get("accuracy",  0.0),
            "test_accuracy":   te.get("accuracy",  0.0),
            "train_f1":        tm.get("f1",         0.0),
            "test_f1":         te.get("f1",         0.0),
            "train_precision": tm.get("precision",  0.0),
            "test_precision":  te.get("precision",  0.0),
            "train_recall":    tm.get("recall",     0.0),
            "test_recall":     te.get("recall",     0.0),
            "final_loss":      result_payload.get("finalLoss", 0.0),
            "training_time_ms": result_payload.get("trainingTimeMs", 0),
        })

        # Per-round loss history as a time series (step = round/epoch index)
        loss_history = result_payload.get("lossHistory", [])
        for step, loss in enumerate(loss_history):
            mlflow.log_metric("loss", loss, step=step)

        # DP per-round epsilon history
        privacy = result_payload.get("privacy") or {}
        eps_history = privacy.get("epsilon_history", [])
        for step, eps in enumerate(eps_history):
            mlflow.log_metric("dp_epsilon", eps, step=step)

        # DP summary metrics
        if privacy:
            mlflow.log_metrics({
                "dp_epsilon_final":       privacy.get("current_epsilon",    0.0),
                "dp_epsilon_remaining":   privacy.get("remaining_epsilon",  0.0),
                "dp_budget_fraction_used": privacy.get("budget_fraction_used", 0.0),
            })

        # SecAgg overhead
        secagg = result_payload.get("secagg") or {}
        if secagg:
            mlflow.log_metrics({
                "secagg_total_overhead_ms": secagg.get("total_overhead_ms", 0.0),
                "secagg_rounds_completed":  secagg.get("rounds_completed",  0),
                "secagg_rounds_aborted":    secagg.get("rounds_aborted",    0),
            })

    # ── Artifacts ─────────────────────────────────────────────────────────

    def _log_artifacts(self, exp, result_payload: dict) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Full result payload (for reproducibility)
            self._write_json(
                os.path.join(tmpdir, "result_payload.json"),
                result_payload,
            )

            # 2. Confusion matrix
            te = result_payload.get("testMetrics", {})
            cm = te.get("confMatrix")
            if cm:
                self._write_json(
                    os.path.join(tmpdir, "confusion_matrix.json"), cm
                )

            # 3. Privacy budget summary
            privacy = result_payload.get("privacy")
            if privacy:
                self._write_json(
                    os.path.join(tmpdir, "privacy_budget.json"), privacy
                )

            # 4. SecAgg audit log
            secagg = result_payload.get("secagg")
            if secagg:
                self._write_json(
                    os.path.join(tmpdir, "secagg_audit.json"), secagg
                )

            # 5. Model card (human-readable markdown)
            card_path = os.path.join(tmpdir, "model_card.md")
            self._write_model_card(card_path, exp, result_payload)

            mlflow.log_artifacts(tmpdir)

    def _write_json(self, path: str, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _write_model_card(self, path: str, exp, result_payload: dict) -> None:
        te   = result_payload.get("testMetrics", {})
        fc   = result_payload.get("featureCols", [])
        privacy = result_payload.get("privacy") or {}
        secagg  = result_payload.get("secagg")  or {}

        lines = [
            f"# Model Card: {exp.name}",
            "",
            "## Model Overview",
            f"- **Type**: {exp.model_type.value}",
            f"- **Target column**: {result_payload.get('targetCol', 'N/A')}",
            f"- **Features**: {len(fc)} ({', '.join(fc[:5])}{'…' if len(fc)>5 else ''})",
            f"- **Experiment ID**: `{exp.id}`",
            "",
            "## Performance",
            f"- Test accuracy:  `{te.get('accuracy',  0):.4f}`",
            f"- Test F1:        `{te.get('f1',         0):.4f}`",
            f"- Test precision: `{te.get('precision',  0):.4f}`",
            f"- Test recall:    `{te.get('recall',     0):.4f}`",
            f"- Final loss:     `{result_payload.get('finalLoss', 0):.6f}`",
            "",
            "## Training Data",
            f"- Train samples: {result_payload.get('trainSamples', 'N/A')}",
            f"- Test samples:  {result_payload.get('testSamples',  'N/A')}",
            f"- Labels:        {result_payload.get('uniqueLabels', [])}",
            "",
        ]

        if privacy:
            lines += [
                "## Differential Privacy",
                f"- Target ε: `{privacy.get('target_epsilon', 'N/A')}`",
                f"- Actual ε: `{privacy.get('current_epsilon', 'N/A'):.4f}`",
                f"- δ:        `{privacy.get('delta', 'N/A')}`",
                f"- Noise σ:  `{privacy.get('noise_sigma', 'N/A'):.4f}`",
                f"- Budget used: `{privacy.get('budget_fraction_used', 0)*100:.1f}%`",
                "",
            ]

        if secagg:
            lines += [
                "## Secure Aggregation",
                f"- Protocol: {secagg.get('protocol', 'SecAgg')}",
                f"- Clients:  {secagg.get('n_clients', 'N/A')}",
                f"- Threshold: {secagg.get('threshold', 'N/A')}",
                f"- All rounds verified: `{secagg.get('all_rounds_verified', 'N/A')}`",
                f"- Total overhead: `{secagg.get('total_overhead_ms', 0):.1f}ms`",
                "",
            ]

        lines += [
            "## Intended Use",
            "Binary classification on tabular CSV data.",
            "Not validated for production medical or safety-critical decisions.",
            "",
            "## Limitations",
            "- Logistic regression only (Phase 3 adds MLP, GBT).",
            "- Simulated federated environment (Phase 6 adds real node federation).",
        ]

        with open(path, "w") as f:
            f.write("\n".join(lines))

    # ── Model Registry ────────────────────────────────────────────────────

    def _register_model(self, exp, result_payload: dict, run_id: str) -> None:
        """
        Register the trained model weights in the MLflow Model Registry.

        Model name follows: tm-{model_type}-{target_col}
        Each registration creates a new version; use the MLflow UI to
        transition versions to Staging or Production.
        """
        target_col = (result_payload.get("targetCol") or "unknown").replace(" ", "_")
        model_name = f"tm-{exp.model_type.value}-{target_col}"

        try:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=model_name,
            )
            logger.info("Model registered: %s (run=%s)", model_name, run_id)
        except Exception as exc:
            # Registry may not be configured (e.g. file:// backend without DB)
            logger.warning("Model registration skipped: %s", exc)


# ════════════════════════════════════════════════════════════════════════════
#  Convenience: log model weights as MLflow artifact + pyfunc
# ════════════════════════════════════════════════════════════════════════════

def log_model_weights(
    weights: list[float],
    bias: float,
    feature_cols: list[str],
    target_col: str,
    run_id: Optional[str] = None,
) -> None:
    """
    Log model weights as an MLflow artifact (weights.json) and as a
    pyfunc model so it can be loaded with mlflow.pyfunc.load_model().

    Called from log_run() automatically; exposed separately for ad-hoc use.
    """
    if not (_MLFLOW_AVAILABLE and MLFLOW_ENABLED):
        return

    model = LogisticRegressionWrapper(
        weights=weights, bias=bias,
        feature_cols=feature_cols, target_col=target_col,
    )

    ctx = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run()
    with ctx:
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "weights.json")
            with open(weights_path, "w") as f:
                json.dump(model.to_dict(), f, indent=2)
            mlflow.log_artifact(weights_path, artifact_path="model")


# ════════════════════════════════════════════════════════════════════════════
#  MLflow health check (used by /api/health)
# ════════════════════════════════════════════════════════════════════════════

def mlflow_health() -> dict:
    """
    Return MLflow connectivity status for inclusion in /api/health.
    """
    if not _MLFLOW_AVAILABLE:
        return {"mlflow": "unavailable", "reason": "package not installed"}
    if not MLFLOW_ENABLED:
        return {"mlflow": "disabled"}
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        # Lightweight check: list experiments (creates the store if file://)
        client.search_experiments(max_results=1)
        return {"mlflow": "ok", "tracking_uri": MLFLOW_TRACKING_URI}
    except Exception as exc:
        return {"mlflow": "error", "detail": str(exc)}


# ── Module-level singleton ────────────────────────────────────────────────
_tracker: Optional[MLflowTracker] = None


def get_tracker() -> MLflowTracker:
    """Return the module-level MLflowTracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker()
    return _tracker
