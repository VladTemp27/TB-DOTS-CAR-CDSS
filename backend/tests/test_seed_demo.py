from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "data"
    monkeypatch.setenv("TB_DATA_DIR", str(d))
    monkeypatch.setenv("TB_DB_PATH", str(d / "test.sqlite3"))
    return d


def _run_seed(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    e = os.environ.copy()
    if env:
        e.update(env)
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, "-m", "backend.seed_demo", *args],
        check=False,
        text=True,
        capture_output=True,
        env=e,
        cwd=str(repo_root),
    )


def test_seed_demo_requires_explicit_reset_guard(tmp_data_dir: Path):
    # Reset is dev-only and must be explicitly enabled.
    r = _run_seed("--reset")
    assert r.returncode != 0
    assert "TB_ALLOW_DEV_RESET=1" in (r.stdout + r.stderr)


def test_seed_demo_seeds_and_is_repeatable_with_reset(tmp_data_dir: Path):
    r1 = _run_seed("--reset", "--include-xrays", env={"TB_ALLOW_DEV_RESET": "1"})
    assert r1.returncode == 0, r1.stdout + r1.stderr
    assert "Seeded 3 demo patients" in r1.stdout

    db_path = Path(os.environ["TB_DB_PATH"])
    assert db_path.exists()
    xrays_dir = tmp_data_dir / "xrays"
    assert xrays_dir.exists()
    # Expect at least one seeded file.
    assert any(p.suffix == ".jpg" for p in xrays_dir.iterdir())

    # Re-run without reset should refuse to seed into a non-empty DB.
    r2 = _run_seed()
    assert r2.returncode != 0
    assert "DB already has patients" in (r2.stdout + r2.stderr)

    # Re-run with reset should succeed again.
    r3 = _run_seed("--reset", env={"TB_ALLOW_DEV_RESET": "1"})
    assert r3.returncode == 0, r3.stdout + r3.stderr
