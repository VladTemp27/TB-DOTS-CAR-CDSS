from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend import settings
from backend.db import engine, session_factory
from backend.models import MonthlyRecord as MonthlyRecordRow
from backend.models import Patient as PatientRow
from backend.models import Prediction as PredictionRow
from backend.models import Xray as XrayRow
from backend.xray_store import new_xray_id


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

CATEGORICAL_FEATURE_TO_ENCODING = {
    "sex": "Sex",
    "anatomicalSite": "Anatomical_Site",
    "registrationGroup": "Registration_Group",
    "bacteriologicStatus": "Bacteriologic_Status",
    "microscopyResult": "Microscopy_Result",
    "sourceOfPatient": "Source_of_Patient",
    "type": "Type",
    "province": "Province",
    "cityMunicipality": "City_Municipality",
    "treatmentHealthFacility": "Treatment_Health_Facility",
    "screeningDiagnosingHealthFacility": "Screening_Diagnosing_Health_Facility",
}


def _backend_dir() -> Path:
    return Path(__file__).resolve().parent


def _run_migrations() -> None:
    cfg = Config(str(_backend_dir() / "alembic.ini"))
    # Use an absolute script_location so it works regardless of CWD.
    cfg.set_main_option("script_location", str(_backend_dir() / "alembic"))
    command.upgrade(cfg, "head")


def _require_dev_reset_guard() -> None:
    # Dev-only safety: force an explicit opt-in env var.
    if os.getenv("TB_ALLOW_DEV_RESET") != "1":
        raise SystemExit(
            "Refusing to reset without TB_ALLOW_DEV_RESET=1 (dev-only safety guard)."
        )


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _reset_storage(*, reset_db: bool, reset_xrays: bool) -> None:
    _require_dev_reset_guard()

    data = settings.data_dir()
    dbp = settings.db_path()
    xray_dir = data / "xrays"

    # Safety: only allow deletes inside TB_DATA_DIR.
    if reset_db and not _is_within(dbp, data):
        raise SystemExit(f"Refusing to delete DB outside TB_DATA_DIR: {dbp}")
    if reset_xrays and not _is_within(xray_dir, data):
        raise SystemExit(f"Refusing to delete X-ray dir outside TB_DATA_DIR: {xray_dir}")

    if reset_db:
        if dbp.exists():
            dbp.unlink()
        # Reset SQLAlchemy engine cache in case this runs in-process.
        # Import locally to avoid import cycles at module import time.
        from backend.db import _reset_engine_for_tests

        _reset_engine_for_tests()

    if reset_xrays:
        if xray_dir.exists():
            shutil.rmtree(xray_dir)
        xray_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class _DemoPatient:
    id: str
    name: str
    features: dict
    treatment_regimen: str
    treatment_start_date: str
    created_at: int
    predictions: list[dict]
    monthly_records: list[dict]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _demo_patients() -> list[_DemoPatient]:
    # Deterministic, non-PHI demo dataset.
    # Goal: non-empty dashboard + varied risk + trends + adherence.
    base = _now_ms() - 10 * 24 * 60 * 60 * 1000

    def contribs(used: dict, prob: float) -> list[dict]:
        # Mirror production shape: one entry per model feature so the
        # FeatureContribution page looks complete for seeded records.
        items: list[dict] = [
            {
                "feature": "Age",
                "delta": 0.05 if int(used.get("age", 0)) >= 45 else 0.02,
                "direction": "risk" if int(used.get("age", 0)) >= 45 else "protective",
            },
            {
                "feature": "Days to Treatment",
                "delta": 0.10 if int(used.get("daysToTreatment", 7)) > 7 else 0.03,
                "direction": "risk" if int(used.get("daysToTreatment", 7)) > 7 else "protective",
            },
            {
                "feature": "Year",
                "delta": 0.01,
                "direction": "protective",
            },
            {
                "feature": "Sex",
                "delta": 0.02,
                "direction": "risk" if str(used.get("sex", "")).upper() == "M" else "protective",
            },
            {
                "feature": "Anatomical Site",
                "delta": 0.04,
                "direction": "risk" if str(used.get("anatomicalSite", "")).upper() == "P" else "protective",
            },
            {
                "feature": "Registration Group",
                "delta": 0.11 if str(used.get("registrationGroup", "")).upper() != "NEW" else 0.03,
                "direction": "risk" if str(used.get("registrationGroup", "")).upper() != "NEW" else "protective",
            },
            {
                "feature": "Bacteriologic Status",
                "delta": 0.08
                if str(used.get("bacteriologicStatus", "")).startswith("Bacteriologically-confirmed")
                else 0.03,
                "direction": "risk"
                if str(used.get("bacteriologicStatus", "")).startswith("Bacteriologically-confirmed")
                else "protective",
            },
            {
                "feature": "Microscopy Result",
                "delta": 0.12 if str(used.get("microscopyResult", "")) in {"1+", "2+", "3+"} else 0.05,
                "direction": "risk"
                if str(used.get("microscopyResult", "")) in {"1+", "2+", "3+"}
                else "protective",
            },
            {
                "feature": "Source of Patient",
                "delta": 0.06
                if str(used.get("sourceOfPatient", "")).upper() != "PUBLIC HEALTH CENTER"
                else 0.02,
                "direction": "risk"
                if str(used.get("sourceOfPatient", "")).upper() != "PUBLIC HEALTH CENTER"
                else "protective",
            },
            {
                "feature": "Type",
                "delta": 0.09 if str(used.get("type", "")).upper() == "DRTB" else 0.03,
                "direction": "risk" if str(used.get("type", "")).upper() == "DRTB" else "protective",
            },
            {
                "feature": "Province",
                "delta": 0.01,
                "direction": "protective",
            },
            {
                "feature": "City/Municipality",
                "delta": 0.01,
                "direction": "protective",
            },
            {
                "feature": "Treatment Facility",
                "delta": 0.02,
                "direction": "protective",
            },
            {
                "feature": "Screening Facility",
                "delta": 0.02,
                "direction": "protective",
            },
        ]

        # Nudge top factors by risk profile so cards look realistic.
        if prob >= 0.7:
            for item in items:
                if item["feature"] in {
                    "Microscopy Result",
                    "Days to Treatment",
                    "Registration Group",
                    "Type",
                }:
                    item["delta"] = round(float(item["delta"]) + 0.04, 3)

        items.sort(key=lambda x: float(x["delta"]), reverse=True)
        return items

    def pred(*, label: int, prob: float, ts: int, used: dict) -> dict:
        return {
            "label": int(label),
            "failure_probability": float(prob),
            "contributions": contribs(used, prob),
            "features_used": used,
            "timestamp": int(ts),
        }

    def mr(*, month: int, prob: float, ts: int, adherence: str, weight: float | None = None, smear: str | None = None) -> dict:
        return {
            "month": int(month),
            "weight": float(weight) if weight is not None else None,
            "smear_result": smear,
            "adherence": adherence,
            "failure_probability": float(prob),
            "timestamp": int(ts),
        }

    common = {
        "year": 2026,
        "province": "BENGUET",
        "cityMunicipality": "BAGUIO CITY",
        "treatmentHealthFacility": "BAGUIO GENERAL HOSPITAL AND MEDICAL CENTER - IDOTS",
        "screeningDiagnosingHealthFacility": "BAGUIO GENERAL HOSPITAL AND MEDICAL CENTER - IDOTS",
    }

    # Note: keep feature keys as client expects (camelCase), because we store JSON and
    # serve it back to the web-app.
    p1_features = {
        **common,
        "age": 29,
        "daysToTreatment": 2,
        "sex": "F",
        "anatomicalSite": "P",
        "registrationGroup": "NEW",
        "bacteriologicStatus": "Bacteriologically-confirmed TB",
        "microscopyResult": "3+",
        "sourceOfPatient": "PUBLIC HEALTH CENTER",
        "type": "DSTB",
    }
    p1_used_m1 = {**p1_features, "microscopyResult": "Negative"}

    p2_features = {
        **common,
        "age": 48,
        "daysToTreatment": 12,
        "sex": "M",
        "anatomicalSite": "P",
        "registrationGroup": "RELAPSE",
        "bacteriologicStatus": "Bacteriologically-confirmed TB",
        "microscopyResult": "2+",
        "sourceOfPatient": "OTHER PUBLIC FACILITY",
        "type": "DRTB",
    }

    p3_features = {
        **common,
        "age": 36,
        "daysToTreatment": 7,
        "sex": "M",
        "anatomicalSite": "EP",
        "registrationGroup": "NEW",
        "bacteriologicStatus": "Clinically-diagnosed TB",
        "microscopyResult": "Negative",
        "sourceOfPatient": "PUBLIC HEALTH CENTER",
        "type": "DSTB",
    }

    return [
        _DemoPatient(
            id="MED-2026-DEMO-0001",
            name="Demo Patient A",
            features=p1_features,
            treatment_regimen="hrze",
            treatment_start_date="2026-04-01",
            created_at=base,
            predictions=[
                pred(label=1, prob=0.62, ts=base + 1000, used=p1_features),
                pred(label=0, prob=0.41, ts=base + 35 * 24 * 60 * 60 * 1000, used=p1_used_m1),
            ],
            monthly_records=[
                mr(month=1, prob=0.41, ts=base + 35 * 24 * 60 * 60 * 1000, adherence="full", weight=53.2, smear="NEG"),
            ],
        ),
        _DemoPatient(
            id="MED-2026-DEMO-0002",
            name="Demo Patient B",
            features=p2_features,
            treatment_regimen="mdr",
            treatment_start_date="2026-03-15",
            created_at=base + 2000,
            predictions=[
                pred(label=1, prob=0.78, ts=base + 3000, used=p2_features),
                pred(label=1, prob=0.83, ts=base + 40 * 24 * 60 * 60 * 1000, used=p2_features),
            ],
            monthly_records=[
                mr(month=1, prob=0.83, ts=base + 40 * 24 * 60 * 60 * 1000, adherence="poor", weight=50.1, smear="POS"),
            ],
        ),
        _DemoPatient(
            id="MED-2026-DEMO-0003",
            name="Demo Patient C",
            features=p3_features,
            treatment_regimen="hrze",
            treatment_start_date="2026-04-20",
            created_at=base + 4000,
            predictions=[
                pred(label=0, prob=0.18, ts=base + 5000, used=p3_features),
            ],
            monthly_records=[],
        ),
    ]


def _validate_demo_patients(demo: list[_DemoPatient]) -> None:
    enc_path = _backend_dir().parent / "web-app" / "src" / "data" / "feature_encodings.json"
    ui_choices: dict[str, set[str]] = {}
    if enc_path.exists():
        import json

        enc = json.loads(enc_path.read_text())
        for feature_key, enc_key in CATEGORICAL_FEATURE_TO_ENCODING.items():
            ui_choices[feature_key] = set(enc.get(enc_key, {}).keys())

    for d in demo:
        if not d.name.strip():
            raise SystemExit(f"Invalid demo patient {d.id}: missing name")

        # medicalId is required by the form/API shape; we mirror id for seeded records.
        if not d.id.strip():
            raise SystemExit("Invalid demo patient: missing id/medicalId")

        missing = [k for k in REQUIRED_FEATURE_KEYS if k not in d.features]
        if missing:
            raise SystemExit(f"Invalid demo patient {d.id}: missing required feature keys: {missing}")

        for key in REQUIRED_FORM_FIELDS:
            value = d.name if key == "name" else (d.id if key == "medicalId" else d.features.get(key))
            if value is None:
                raise SystemExit(f"Invalid demo patient {d.id}: required field {key!r} is null")
            if isinstance(value, str) and not value.strip():
                raise SystemExit(f"Invalid demo patient {d.id}: required field {key!r} is empty")

        for feature_key, choices in ui_choices.items():
            value = d.features.get(feature_key)
            if value not in choices:
                raise SystemExit(
                    f"Invalid demo patient {d.id}: feature {feature_key!r} value {value!r} "
                    "is not selectable from UI choices"
                )

        age = d.features.get("age")
        if not isinstance(age, int) or age <= 0:
            raise SystemExit(f"Invalid demo patient {d.id}: age must be a positive integer")

        dtt = d.features.get("daysToTreatment")
        if not isinstance(dtt, int) or dtt < 0:
            raise SystemExit(
                f"Invalid demo patient {d.id}: daysToTreatment must be a non-negative integer"
            )

        for idx, pr in enumerate(d.predictions, start=1):
            used = pr.get("features_used")
            if not isinstance(used, dict):
                raise SystemExit(
                    f"Invalid demo patient {d.id} prediction #{idx}: features_used missing or invalid"
                )

            missing_used = [k for k in REQUIRED_FEATURE_KEYS if k not in used]
            if missing_used:
                raise SystemExit(
                    f"Invalid demo patient {d.id} prediction #{idx}: "
                    f"features_used missing required keys: {missing_used}"
                )

            for feature_key, choices in ui_choices.items():
                value = used.get(feature_key)
                if value not in choices:
                    raise SystemExit(
                        f"Invalid demo patient {d.id} prediction #{idx}: features_used {feature_key!r} "
                        f"value {value!r} is not selectable from UI choices"
                    )

            used_age = used.get("age")
            if not isinstance(used_age, int) or used_age <= 0:
                raise SystemExit(
                    f"Invalid demo patient {d.id} prediction #{idx}: age must be a positive integer"
                )

            used_dtt = used.get("daysToTreatment")
            if not isinstance(used_dtt, int) or used_dtt < 0:
                raise SystemExit(
                    f"Invalid demo patient {d.id} prediction #{idx}: "
                    "daysToTreatment must be a non-negative integer"
                )


def _seed_db(db: Session, *, include_xrays: bool) -> int:
    # Refuse to seed into a non-empty DB unless user reset first.
    existing = db.scalar(select(PatientRow.id).limit(1))
    if existing is not None:
        raise SystemExit(
            "DB already has patients. Re-run with --reset (and TB_ALLOW_DEV_RESET=1) to seed from scratch."
        )

    demo = _demo_patients()
    _validate_demo_patients(demo)
    for d in demo:
        p = PatientRow(
            id=d.id,
            name=d.name,
            medical_id=d.id,
            features=d.features,
            treatment_regimen=d.treatment_regimen,
            treatment_start_date=d.treatment_start_date,
            created_at=d.created_at,
        )
        for pr in d.predictions:
            p.predictions.append(
                PredictionRow(
                    label=pr["label"],
                    failure_probability=pr["failure_probability"],
                    contributions=pr["contributions"],
                    features_used=pr.get("features_used"),
                    timestamp=pr["timestamp"],
                )
            )
        for r in d.monthly_records:
            p.monthly_records.append(
                MonthlyRecordRow(
                    month=r["month"],
                    weight=r.get("weight"),
                    smear_result=r.get("smear_result"),
                    adherence=r["adherence"],
                    failure_probability=r["failure_probability"],
                    timestamp=r["timestamp"],
                )
            )
        db.add(p)

    if include_xrays:
        _seed_demo_xrays(db)

    db.commit()
    return len(demo)


def _seed_demo_xrays(db: Session) -> None:
    # Keep bytes tiny so this is fast and repo-agnostic.
    # Minimal valid JPEG: SOI + EOI.
    jpeg_bytes = b"\xff\xd8\xff\xd9"
    mime = "image/jpeg"
    ext = ".jpg"

    data = settings.data_dir()
    xray_dir = data / "xrays"
    xray_dir.mkdir(parents=True, exist_ok=True)

    def write_xray(*, patient_id: str, kind: str, month: int | None, name: str) -> str:
        x_id = new_xray_id()
        dest = xray_dir / f"{x_id}{ext}"
        dest.write_bytes(jpeg_bytes)
        sha = hashlib.sha256(jpeg_bytes).hexdigest()
        now = _now_ms()
        db.add(
            XrayRow(
                id=x_id,
                patient_id=patient_id,
                kind=kind,
                month=month,
                mime=mime,
                name=name,
                sha256=sha,
                size_bytes=len(jpeg_bytes),
                rel_path=dest.name,
                created_at=now,
            )
        )
        return x_id

    # One intake image for patient A.
    write_xray(
        patient_id="MED-2026-DEMO-0001",
        kind="intake",
        month=None,
        name="demo_intake.jpg",
    )
    # One monthly image for patient B at month 1.
    write_xray(
        patient_id="MED-2026-DEMO-0002",
        kind="monthly",
        month=1,
        name="demo_month1.jpg",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.seed_demo",
        description="Seed demo patient data into the backend SQLite database.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe DB file and X-ray directory before seeding (requires TB_ALLOW_DEV_RESET=1).",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Wipe DB file before seeding (requires TB_ALLOW_DEV_RESET=1).",
    )
    parser.add_argument(
        "--reset-xrays",
        action="store_true",
        help="Wipe X-ray directory before seeding (requires TB_ALLOW_DEV_RESET=1).",
    )
    parser.add_argument(
        "--include-xrays",
        action="store_true",
        help="Also seed a small number of demo X-ray files via the filesystem store.",
    )

    args = parser.parse_args(argv)

    reset_db = bool(args.reset or args.reset_db)
    reset_xrays = bool(args.reset or args.reset_xrays)
    if args.reset and not (reset_db and reset_xrays):
        # defensive, but should always be true
        reset_db = True
        reset_xrays = True

    if reset_db or reset_xrays:
        _reset_storage(reset_db=reset_db, reset_xrays=reset_xrays)

    # Create DB file + tables if needed.
    engine()  # ensure path created
    _run_migrations()

    db = session_factory()()
    try:
        seeded = _seed_db(db, include_xrays=bool(args.include_xrays))
    finally:
        db.close()

    data = settings.data_dir()
    print(f"Seeded {seeded} demo patients")
    print(f"DB: {settings.db_path()}")
    print(f"X-rays dir: {data / 'xrays'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
