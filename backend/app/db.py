from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.db_url, connect_args={"check_same_thread": False})
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _session_factory


def init_db() -> None:
    from . import models  # noqa: F401 - register mappings

    engine = get_engine()
    Base.metadata.create_all(engine)
    _migrate_sqlite(engine)


def _migrate_sqlite(engine) -> None:
    """Idempotent column additions for databases created by older versions."""
    from sqlalchemy import text

    with engine.begin() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(meetings)"))}
        if "speaker_names_json" not in columns:
            conn.execute(text(
                "ALTER TABLE meetings ADD COLUMN speaker_names_json TEXT NOT NULL DEFAULT '{}'"
            ))


def new_session():
    return get_session_factory()()
