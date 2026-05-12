from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend import settings


class Base(DeclarativeBase):
    pass


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


_engine: Engine | None = None
_engine_url: str | None = None


def engine() -> Engine:
    global _engine, _engine_url
    url = settings.sqlalchemy_database_url()
    if _engine is not None and _engine_url == url:
        return _engine

    dbp = settings.db_path()
    _ensure_parent_dir(dbp)
    _engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        future=True,
    )
    _engine_url = url
    return _engine


def session_factory():
    return sessionmaker(bind=engine(), autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = session_factory()()
    try:
        yield db
    finally:
        db.close()


def _reset_engine_for_tests() -> None:
    global _engine, _engine_url
    _engine = None
    _engine_url = None
