import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.core.config import settings

log = logging.getLogger("backend.db")


def _get_engine():
    db_url = settings.database_url
    if db_url.startswith("sqlite"):
        return create_engine(db_url, connect_args={"check_same_thread": False})

    try:
        eng = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        with eng.connect():
            pass
        return eng
    except Exception as e:
        log.warning(
            f"PostgreSQL connection failed ({e}). "
            "Falling back to local SQLite database (sqlite:///./neurostage.db) for local development."
        )
        settings.database_url = "sqlite:///./neurostage.db"
        return create_engine(settings.database_url, connect_args={"check_same_thread": False})


engine = _get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a database session and closes it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
