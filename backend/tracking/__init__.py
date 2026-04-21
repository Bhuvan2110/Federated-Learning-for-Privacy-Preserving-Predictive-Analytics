"""
tracking/__init__.py
MLflow experiment tracking package  (Day 6)
"""
from tracking.mlflow_tracker import (
    MLflowTracker,
    LogisticRegressionWrapper,
    mlflow_health,
    get_tracker,
    log_model_weights,
    MLFLOW_ENABLED,
    MLFLOW_TRACKING_URI,
    MLFLOW_REGISTER_MODELS,
    _MLFLOW_AVAILABLE,
)

__all__ = [
    "MLflowTracker",
    "LogisticRegressionWrapper",
    "mlflow_health",
    "get_tracker",
    "log_model_weights",
    "MLFLOW_ENABLED",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_REGISTER_MODELS",
    "_MLFLOW_AVAILABLE",
]
