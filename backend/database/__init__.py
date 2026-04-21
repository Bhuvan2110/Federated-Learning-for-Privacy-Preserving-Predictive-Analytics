"""
database/__init__.py
Expose the most commonly imported names so routes can do:
    from database import get_db, init_db, UserRepo, ExperimentRepo
"""

from database.db import get_db, init_db, drop_db, health_check, engine
from database.models import (
    Base,
    User, UserRole,
    UploadedFile,
    Experiment, ExperimentStatus, ModelType,
    ExperimentResult,
)
from database.repository import UserRepo, FileRepo, ExperimentRepo, ResultRepo

__all__ = [
    # db helpers
    "get_db", "init_db", "drop_db", "health_check", "engine",
    # models
    "Base",
    "User", "UserRole",
    "UploadedFile",
    "Experiment", "ExperimentStatus", "ModelType",
    "ExperimentResult",
    # repos
    "UserRepo", "FileRepo", "ExperimentRepo", "ResultRepo",
]
