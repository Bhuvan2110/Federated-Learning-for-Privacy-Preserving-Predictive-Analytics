"""
tasks/__init__.py
Celery task package.  Import tasks here so the worker auto-discovers them.
"""
from tasks.training_tasks import (
    run_central_training,
    run_federated_training,
    run_dp_federated_training,        # Day 3: differentially private FedAvg
    run_secagg_federated_training,    # Day 4: SecAgg-protected FedAvg
)

__all__ = [
    "run_central_training",
    "run_federated_training",
    "run_dp_federated_training",
    "run_secagg_federated_training",
]
