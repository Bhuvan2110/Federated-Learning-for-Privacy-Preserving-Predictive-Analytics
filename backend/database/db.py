"""
database/db.py
══════════════════════════════════════════════════════════════════════
Engine setup, session factory, and helper context manager.

Usage in Flask routes
─────────────────────
    from database.db import get_db

    with get_db() as db:
        user = db.query(User).filter_by(username="alice").first()

Environment variables
─────────────────────
    DATABASE_URL   — full SQLAlchemy connection string
                     default: postgresql://tm_user:tm_pass@localhost:5432/tm_db

    DB_POOL_SIZE   — number of persistent connections (default: 5)
    DB_MAX_OVERFLOW — extra connections beyond pool_size (default: 10)
    DB_ECHO        — set to "true" to log every SQL statement (default: false)
"""

import os
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://tm_user:tm_pass@localhost:5432/tm_db",
)

# SQLite fallback for local dev or simple cloud demos without Postgres
_SQLITE_FALLBACK = os.getenv("USE_SQLITE_FALLBACK", "false").lower() == "true"

# Auto-enable SQLite if we are on Render and no DB is provided
if os.getenv("RENDER") and DATABASE_URL.startswith("postgresql://tm_user:tm_pass@localhost"):
    _SQLITE_FALLBACK = True

if _SQLITE_FALLBACK:
    DATABASE_URL = "sqlite:///./tm_dev.db"
    logger.warning("⚠️  Using SQLite fallback — NOT suitable for production persistence!")

POOL_SIZE    = int(os.getenv("DB_POOL_SIZE", "5"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
ECHO_SQL     = os.getenv("DB_ECHO", "false").lower() == "true"

# ─── Engine ───────────────────────────────────────────────────────────────

_engine_kwargs = {
    "echo":          ECHO_SQL,
    "future":        True,           # SQLAlchemy 2.x style
}

if not DATABASE_URL.startswith("sqlite"):
    _engine_kwargs.update({
        "pool_size":         POOL_SIZE,
        "max_overflow":      MAX_OVERFLOW,
        "pool_pre_ping":     True,   # recycle stale connections automatically
        "pool_recycle":      1800,   # recycle connections after 30 minutes
    })

engine = create_engine(DATABASE_URL, **_engine_kwargs)

# ─── Session factory ──────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,   # safe to read attributes after commit
)


# ─── Context manager ──────────────────────────────────────────────────────

@contextmanager
def get_db() -> Session:
    """
    Yield a SQLAlchemy session and commit on exit, or rollback on error.

    Usage:
        with get_db() as db:
            db.add(some_model)
        # automatically committed here
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─── Schema management ────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables that don't exist yet.
    Call this once on application startup.
    For production schema migrations use Alembic instead.
    """
    logger.info("Initialising database schema …")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ready ✓")


def drop_db() -> None:
    """
    Drop ALL tables. DESTRUCTIVE — only use in tests or dev resets.
    """
    logger.warning("⚠️  Dropping all database tables!")
    Base.metadata.drop_all(bind=engine)


def health_check() -> dict:
    """
    Lightweight connectivity check used by /api/health.
    Returns {"db": "ok", "url": "<sanitised>"} or {"db": "error", "detail": "..."}.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        # Strip credentials from URL for the health response
        safe_url = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
        return {"db": "ok", "host": safe_url}
    except Exception as exc:
        return {"db": "error", "detail": str(exc)}
