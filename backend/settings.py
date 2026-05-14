import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    # Single-workstation deployment: keep data adjacent to repo by default.
    p = Path(os.getenv("TB_DATA_DIR", str(repo_root() / "data"))).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    return Path(os.getenv("TB_DB_PATH", str(data_dir() / "tb_cdss.sqlite3"))).resolve()


def sqlalchemy_database_url() -> str:
    # Use sqlite+pysqlite so SQLAlchemy doesn't try aiosqlite.
    return f"sqlite+pysqlite:///{db_path()}"


def max_upload_bytes() -> int:
    # 25MB default; override via env for larger DICOM-derived exports.
    return int(os.getenv("TB_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
