from __future__ import annotations

import os
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


REQUIRED_FORM_FIELDS = (
    "name",
    "medicalId",
    "age",
    "sex",
    "province",
    "registrationGroup",
    "anatomicalSite",
    "bacteriologicStatus",
    "microscopyResult",
    "sourceOfPatient",
    "type",
)

REQUIRED_FEATURE_KEYS = (
    "age",
    "daysToTreatment",
    "year",
    "sex",
    "anatomicalSite",
    "registrationGroup",
    "bacteriologicStatus",
    "microscopyResult",
    "sourceOfPatient",
    "type",
    "province",
    "cityMunicipality",
    "treatmentHealthFacility",
    "screeningDiagnosingHealthFacility",
)

EXPECTED_CONTRIBUTION_FEATURES = {
    "Age",
    "Days to Treatment",
    "Year",
    "Sex",
    "Anatomical Site",
    "Registration Group",
    "Bacteriologic Status",
    "Microscopy Result",
    "Source of Patient",
    "Type",
    "Province",
    "City/Municipality",
    "Treatment Facility",
    "Screening Facility",
}


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


def test_seeded_rows_include_required_form_fields(tmp_data_dir: Path):
    r = _run_seed("--reset", env={"TB_ALLOW_DEV_RESET": "1"})
    assert r.returncode == 0, r.stdout + r.stderr

    db_path = Path(os.environ["TB_DB_PATH"])
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT id, name, medical_id, features FROM patients").fetchall()
    finally:
        conn.close()

    assert rows
    for pid, name, medical_id, features_json in rows:
        assert str(name).strip()
        assert str(medical_id).strip()
        features = json.loads(features_json)

        for key in REQUIRED_FEATURE_KEYS:
            assert key in features, f"{pid}: missing feature key {key}"

        for key in REQUIRED_FORM_FIELDS:
            value = name if key == "name" else (medical_id if key == "medicalId" else features[key])
            assert value is not None
            if isinstance(value, str):
                assert value.strip(), f"{pid}: empty required field {key}"


def test_seeded_predictions_include_full_feature_contributions(tmp_data_dir: Path):
    r = _run_seed("--reset", env={"TB_ALLOW_DEV_RESET": "1"})
    assert r.returncode == 0, r.stdout + r.stderr

    db_path = Path(os.environ["TB_DB_PATH"])
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT patient_id, contributions FROM predictions").fetchall()
    finally:
        conn.close()

    assert rows
    for patient_id, contributions_json in rows:
        contributions = json.loads(contributions_json)
        assert len(contributions) == len(EXPECTED_CONTRIBUTION_FEATURES), (
            f"{patient_id}: expected {len(EXPECTED_CONTRIBUTION_FEATURES)} contributions, "
            f"got {len(contributions)}"
        )

        names = {c["feature"] for c in contributions}
        assert names == EXPECTED_CONTRIBUTION_FEATURES, (
            f"{patient_id}: contribution features mismatch: {names}"
        )

        assert all(c.get("direction") in {"risk", "protective"} for c in contributions), (
            f"{patient_id}: contribution direction values must be risk/protective"
        )

        assert all(
            isinstance(c.get("delta"), (int, float)) and float(c["delta"]) >= 0
            for c in contributions
        ), f"{patient_id}: contribution deltas must be non-negative numeric values"


def test_seeded_prediction_features_used_include_required_keys(tmp_data_dir: Path):
    r = _run_seed("--reset", env={"TB_ALLOW_DEV_RESET": "1"})
    assert r.returncode == 0, r.stdout + r.stderr

    db_path = Path(os.environ["TB_DB_PATH"])
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT patient_id, features_used FROM predictions").fetchall()
    finally:
        conn.close()

    assert rows
    for patient_id, features_used_json in rows:
        used = json.loads(features_used_json)
        for key in REQUIRED_FEATURE_KEYS:
            assert key in used, f"{patient_id}: features_used missing key {key}"

        assert isinstance(used.get("age"), int) and used["age"] > 0
        assert isinstance(used.get("daysToTreatment"), int) and used["daysToTreatment"] >= 0
