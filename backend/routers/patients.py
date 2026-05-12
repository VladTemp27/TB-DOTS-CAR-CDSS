import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import MonthlyRecord as MonthlyRecordRow
from backend.models import Patient as PatientRow
from backend.models import Prediction as PredictionRow
from backend.models import Xray as XrayRow
from backend.schemas import (
    MonthlyRecordCreate,
    Patient,
    PatientCreate,
    PredictionCreate,
)

router = APIRouter()


def _patient_to_api(db: Session, p: PatientRow) -> dict:
    xrays = db.scalars(select(XrayRow).where(XrayRow.patient_id == p.id)).all()
    intake_ids = [x.id for x in xrays if x.kind == "intake"]

    records = db.scalars(
        select(MonthlyRecordRow)
        .where(MonthlyRecordRow.patient_id == p.id)
        .order_by(MonthlyRecordRow.month.asc())
    ).all()
    record_id_to_xray_ids: dict[int, list[str]] = {}
    for x in xrays:
        if x.kind == "monthly" and x.month is not None:
            record_id_to_xray_ids.setdefault(x.month, []).append(x.id)

    predictions = db.scalars(
        select(PredictionRow)
        .where(PredictionRow.patient_id == p.id)
        .order_by(PredictionRow.timestamp.asc())
    ).all()

    return Patient(
        id=p.id,
        name=p.name,
        medicalId=p.medical_id,
        features=p.features,
        treatmentRegimen=p.treatment_regimen,
        treatmentStartDate=p.treatment_start_date,
        createdAt=p.created_at,
        predictions=[
            {
                "label": int(r.label),
                "failureProbability": float(r.failure_probability),
                "contributions": r.contributions,
                "featuresUsed": r.features_used,
                "timestamp": int(r.timestamp),
            }
            for r in predictions
        ],
        monthlyRecords=[
            {
                "month": int(r.month),
                "weight": float(r.weight) if r.weight is not None else None,
                "smearResult": r.smear_result,
                "adherence": r.adherence,
                "failureProbability": float(r.failure_probability),
                "timestamp": int(r.timestamp),
                "xrayIds": record_id_to_xray_ids.get(int(r.month)) or None,
            }
            for r in records
        ],
        intakeXrayIds=intake_ids or None,
    ).model_dump(by_alias=True)


@router.get("/api/patients")
def list_patients(db: Annotated[Session, Depends(get_db)]):
    patients = db.scalars(select(PatientRow).order_by(PatientRow.created_at.desc())).all()
    return [_patient_to_api(db, p) for p in patients]


@router.post("/api/patients")
def create_patient(body: PatientCreate, db: Annotated[Session, Depends(get_db)]):
    pid = body.id or f"MED-{time.gmtime().tm_year}-{int.from_bytes(os.urandom(2), 'big'):04d}"
    created_at = body.created_at or int(time.time() * 1000)

    existing = db.get(PatientRow, pid)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Patient id already exists")

    row = PatientRow(
        id=pid,
        name=body.name,
        medical_id=body.medical_id,
        features=body.features.model_dump(by_alias=True),
        treatment_regimen=body.treatment_regimen,
        treatment_start_date=body.treatment_start_date,
        created_at=created_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _patient_to_api(db, row)


@router.put("/api/patients/{patient_id}")
def upsert_patient(
    patient_id: str, body: PatientCreate, db: Annotated[Session, Depends(get_db)]
):
    created_at = body.created_at or int(time.time() * 1000)

    row = db.get(PatientRow, patient_id)
    if row is None:
        row = PatientRow(
            id=patient_id,
            name=body.name,
            medical_id=body.medical_id,
            features=body.features.model_dump(by_alias=True),
            treatment_regimen=body.treatment_regimen,
            treatment_start_date=body.treatment_start_date,
            created_at=created_at,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _patient_to_api(db, row)

    row.name = body.name
    row.medical_id = body.medical_id
    row.features = body.features.model_dump(by_alias=True)
    row.treatment_regimen = body.treatment_regimen
    row.treatment_start_date = body.treatment_start_date
    row.created_at = row.created_at or created_at
    db.commit()
    return _patient_to_api(db, row)


@router.get("/api/patients/{patient_id}")
def get_patient(patient_id: str, db: Annotated[Session, Depends(get_db)]):
    p = db.get(PatientRow, patient_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return _patient_to_api(db, p)


@router.post("/api/patients/{patient_id}/monthly-records")
def add_monthly_record(
    patient_id: str, body: MonthlyRecordCreate, db: Annotated[Session, Depends(get_db)]
):
    p = db.get(PatientRow, patient_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    row = MonthlyRecordRow(
        patient_id=patient_id,
        month=body.month,
        weight=body.weight,
        smear_result=body.smear_result,
        adherence=body.adherence,
        failure_probability=body.failure_probability,
        timestamp=body.timestamp,
    )
    db.add(row)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=409, detail="Monthly record already exists")
    return {"ok": True}


@router.post("/api/patients/{patient_id}/predictions")
def add_prediction(
    patient_id: str, body: PredictionCreate, db: Annotated[Session, Depends(get_db)]
):
    p = db.get(PatientRow, patient_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    row = PredictionRow(
        patient_id=patient_id,
        label=int(body.label),
        failure_probability=float(body.failure_probability),
        contributions=[c.model_dump() for c in body.contributions],
        features_used=body.features_used,
        timestamp=int(body.timestamp),
    )
    db.add(row)
    db.commit()
    return {"ok": True}
