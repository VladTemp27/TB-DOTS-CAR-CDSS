from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "data"
    monkeypatch.setenv("TB_DATA_DIR", str(d))
    monkeypatch.setenv("TB_DB_PATH", str(d / "test.sqlite3"))
    return d


def _run_migrations() -> None:
    from alembic import command
    from alembic.config import Config
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parents[1]
    cfg = Config(str(backend_dir / "alembic.ini"))
    # Use an absolute script_location so tests pass regardless of CWD.
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    command.upgrade(cfg, "head")


def test_migrations_apply_on_clean_db(tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    # Ensure a clean DB
    db_path = Path(os.environ["TB_DB_PATH"])
    if db_path.exists():
        db_path.unlink()

    _run_migrations()
    assert db_path.exists()


def test_patient_create_and_fetch_shape(tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _run_migrations()

    # Import after env vars are set so the backend uses the test DB.
    from backend.db import _reset_engine_for_tests
    _reset_engine_for_tests()
    from backend.main import app

    client = TestClient(app)

    body = {
        "id": "MED-2026-0001",
        "name": "Alice",
        "medicalId": "MED-2026-0001",
        "features": {
            "age": 35,
            "daysToTreatment": 7,
            "year": 2026,
            "sex": "M",
            "anatomicalSite": "P",
            "registrationGroup": "NEW",
            "bacteriologicStatus": "POS",
            "microscopyResult": "POS",
            "sourceOfPatient": "Walk-in",
            "type": "New",
            "province": "CAR",
            "cityMunicipality": "Baguio",
            "treatmentHealthFacility": "",
            "screeningDiagnosingHealthFacility": "",
        },
        "createdAt": 1710000000000,
        "predictions": [],
        "monthlyRecords": [],
    }

    # POST /api/patients expects PatientCreate (no predictions/monthlyRecords)
    create_body = {
        "id": body["id"],
        "name": body["name"],
        "medicalId": body["medicalId"],
        "features": body["features"],
        "createdAt": body["createdAt"],
    }

    res = client.post("/api/patients", json=create_body)
    assert res.status_code == 200
    created = res.json()

    assert created["id"] == "MED-2026-0001"
    assert created["medicalId"] == "MED-2026-0001"
    assert created["features"]["daysToTreatment"] == 7
    assert created["predictions"] == []
    assert created["monthlyRecords"] == []
    assert created.get("intakeXrayIds") in (None, [])

    res2 = client.get("/api/patients/MED-2026-0001")
    assert res2.status_code == 200
    fetched = res2.json()
    assert fetched["id"] == "MED-2026-0001"


def test_xray_upload_and_file_fetch(tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _run_migrations()
    from backend.db import _reset_engine_for_tests
    _reset_engine_for_tests()
    from backend.main import app

    client = TestClient(app)

    # Create patient
    res = client.post(
        "/api/patients",
        json={
            "id": "MED-2026-0002",
            "name": "Bob",
            "medicalId": "MED-2026-0002",
            "features": {
                "age": 40,
                "daysToTreatment": 7,
                "year": 2026,
                "sex": "M",
                "anatomicalSite": "P",
                "registrationGroup": "NEW",
                "bacteriologicStatus": "POS",
                "microscopyResult": "POS",
                "sourceOfPatient": "Walk-in",
                "type": "New",
                "province": "CAR",
                "cityMunicipality": "Baguio",
                "treatmentHealthFacility": "",
                "screeningDiagnosingHealthFacility": "",
            },
            "createdAt": 1710000000000,
        },
    )
    assert res.status_code == 200

    # Minimal valid JPEG: SOI + EOI.
    jpeg_bytes = b"\xff\xd8\xff\xd9"
    files = {"file": ("test.jpg", jpeg_bytes, "image/jpeg")}
    res2 = client.post(
        "/api/xrays?patient_id=MED-2026-0002&kind=intake",
        files=files,
    )
    assert res2.status_code == 200
    x = res2.json()
    assert x["id"].startswith("xr_")
    assert x["mime"] == "image/jpeg"
    assert x["sizeBytes"] == len(jpeg_bytes)

    # Fetch file
    res3 = client.get(f"/api/xrays/{x['id']}/file")
    assert res3.status_code == 200
    assert res3.headers["content-type"].startswith("image/jpeg")
    assert res3.content == jpeg_bytes
