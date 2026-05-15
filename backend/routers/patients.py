import os
import time
from typing import Annotated, Any

import numpy as np

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
    TemporalRiskRequest,
)

from backend.temporal_lstm import load_artifacts, predict_success_probability

router = APIRouter()


def _epoch_days(ms_or_iso: str | int | float | None) -> float | None:
    import datetime

    if ms_or_iso is None:
        return None
    if isinstance(ms_or_iso, (int, float)):
        # assume milliseconds
        return float(ms_or_iso) / 1000.0 / 86400.0
    s = str(ms_or_iso).strip()
    if not s:
        return None
    try:
        # ISO date
        dt = datetime.datetime.fromisoformat(s)
    except ValueError:
        return None
    epoch = datetime.datetime(1970, 1, 1, tzinfo=dt.tzinfo)
    return (dt - epoch).total_seconds() / 86400.0


def _build_static_vector(p: PatientRow, static_names: list[str]) -> np.ndarray:
    """Best-effort mapping from stored patient.features into the 94-dim static vector."""

    f = p.features or {}
    idx = {name: i for i, name in enumerate(static_names)}
    out = np.zeros((len(static_names),), dtype=np.float32)
    explicit_missing_flags: set[str] = set()

    def setv(name: str, value: float | int | None):
        i = idx.get(name)
        if i is None or value is None:
            return
        out[i] = float(value)

    def set_missing(name: str, missing: bool):
        i = idx.get(name)
        if i is None:
            return
        explicit_missing_flags.add(name)
        out[i] = 1.0 if missing else 0.0

    # Core numeric features.
    setv("age", f.get("age"))
    setv("weight_kg", f.get("baselineWeightKg"))
    setv("height_cm", f.get("baselineHeightCm"))
    setv("bp_systolic", f.get("bpSystolic"))
    setv("bp_diastolic", f.get("bpDiastolic"))
    setv("heart_rate", f.get("heartRate"))
    setv("o2_sat", f.get("o2Sat"))

    # Binary lab at baseline if present.
    def _as01(v: Any) -> int | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return 1 if float(v) >= 0.5 else 0
        s = str(v).strip().lower()
        if s in {"1", "pos", "positive", "true", "yes"}:
            return 1
        if s in {"0", "neg", "negative", "false", "no"}:
            return 0
        return None

    setv("xpert_mtb_rif", _as01(f.get("xpertMtbRif")))
    setv("smear_microscopy", _as01(f.get("microscopyResult")))

    # Date-like numeric columns use days since epoch.
    tx_start_days = _epoch_days(p.treatment_start_date)
    setv("treatment_start_date", tx_start_days)
    setv("intensive_phase_start_date", tx_start_days)

    days_to_tx = f.get("daysToTreatment")
    try:
        days_to_tx_n = float(days_to_tx) if days_to_tx is not None else None
    except Exception:
        days_to_tx_n = None
    if tx_start_days is not None and days_to_tx_n is not None:
        setv("date_of_diagnosis", tx_start_days - days_to_tx_n)
        setv("date_of_notification", tx_start_days - max(0.0, days_to_tx_n - 1.0))

    # One-hot features.
    sex = (f.get("sex") or "").strip().lower()
    if sex in {"m", "male"} and "sex_Male" in idx:
        out[idx["sex_Male"]] = 1.0
    if sex in {"f", "female"} and "sex_Female" in idx:
        out[idx["sex_Female"]] = 1.0

    diagnosis = f.get("diagnosis")
    if isinstance(diagnosis, str):
        key = f"diagnosis_{diagnosis.strip()}"
        if key in idx:
            out[idx[key]] = 1.0

    bact = f.get("bacteriologicStatus")
    if isinstance(bact, str):
        key = f"bacteriologic_status_{bact.strip()}"
        if key in idx:
            out[idx[key]] = 1.0

    regimen = p.treatment_regimen
    if regimen in {"hrze", "mdr", "xdr"}:
        label = "Regimen 1" if regimen == "hrze" else "Regimen 2"
        key = f"treatment_regimen_{label}"
        if key in idx:
            out[idx[key]] = 1.0

    com = f.get("coMorbidities")
    if isinstance(com, str) and com.strip():
        for part in [p.strip() for p in com.split(",") if p.strip()]:
            key = f"co_morbidities_{part}"
            if key in idx:
                out[idx[key]] = 1.0

    # Facility best-effort from treatment facility string.
    fac = (f.get("treatmentHealthFacility") or "").strip().lower()
    if fac:
        for token, col in (
            ("pacdal", "facility_Pacdal"),
            ("quirino", "facility_Quirino Hill"),
            ("pinsao", "facility_Pinsao"),
            ("asin", "facility_Asin"),
            ("atab", "facility_Atab"),
            ("atoko", "facility_Atok Trail"),
            ("at\u014dk", "facility_Atok Trail"),
            ("eng", "facility_Eng Hill"),
            ("city camp", "facility_City Camp"),
            ("scout", "facility_Scout Barrio"),
            ("campo", "facility_Campo Filipino"),
        ):
            if token in fac and col in idx:
                out[idx[col]] = 1.0

    # Missing flags: default to missing if we don't have the value.
    set_missing("is_missing_age", f.get("age") is None)
    set_missing("is_missing_weight_kg", f.get("baselineWeightKg") is None)
    set_missing("is_missing_height_cm", f.get("baselineHeightCm") is None)
    set_missing("is_missing_bp_systolic", f.get("bpSystolic") is None)
    set_missing("is_missing_bp_diastolic", f.get("bpDiastolic") is None)
    set_missing("is_missing_heart_rate", f.get("heartRate") is None)
    set_missing("is_missing_o2_sat", f.get("o2Sat") is None)
    set_missing("is_missing_xpert_mtb_rif", f.get("xpertMtbRif") is None)
    set_missing("is_missing_smear_microscopy", f.get("microscopyResult") is None)
    set_missing("is_missing_diagnosis", f.get("diagnosis") is None)
    set_missing("is_missing_bacteriologic_status", f.get("bacteriologicStatus") is None)
    set_missing("is_missing_treatment_regimen", p.treatment_regimen is None)
    set_missing("is_missing_co_morbidities", f.get("coMorbidities") is None)

    # Any remaining is_missing_* columns not explicitly set are unknown from the app form.
    for name, i in idx.items():
        if name.startswith("is_missing_") and name not in explicit_missing_flags:
            out[i] = 1.0

    return out


def _build_temporal_matrix(
    *,
    temporal_names: list[str],
    records: list[MonthlyRecordRow],
    current_month: int,
) -> tuple[np.ndarray, int]:
    by_month: dict[int, MonthlyRecordRow] = {int(r.month): r for r in records if r.month is not None}
    max_month = current_month
    xt = np.zeros((13, len(temporal_names)), dtype=np.float32)

    # Compute cumulative doses from history if not provided.
    cumulative = 0
    for m in range(0, max_month + 1):
        r = by_month.get(m)
        taken = int(r.monthly_doses_taken) if r and r.monthly_doses_taken is not None else None
        missed = int(r.monthly_missed_doses) if r and r.monthly_missed_doses is not None else None

        if taken is not None:
            cumulative += taken

        weight = float(r.weight) if r and r.weight is not None else None
        height = float(r.height) if r and r.height is not None else None
        smear = int(r.smear_tb_lamp) if r and r.smear_tb_lamp is not None else None
        xpert = int(r.xpert_mtb_rif) if r and r.xpert_mtb_rif is not None else None

        cum = int(r.cumulative_doses_taken) if r and r.cumulative_doses_taken is not None else (cumulative if taken is not None else None)
        pct = float(r.pct_adherence) if r and r.pct_adherence is not None else None

        # If pct is missing but taken/missed are present, derive.
        if pct is None and taken is not None and missed is not None:
            total = taken + missed
            pct = (taken / total) * 100.0 if total > 0 else None

        row = {
            "cumulative_doses_taken": float(cum) if cum is not None else 0.0,
            "height": float(height) if height is not None else 0.0,
            "monthly_doses_taken": float(taken) if taken is not None else 0.0,
            "monthly_missed_doses": float(missed) if missed is not None else 0.0,
            "pct_adherence": float(pct) if pct is not None else 0.0,
            "smear_tb_lamp": float(smear) if smear is not None else 0.0,
            "weight": float(weight) if weight is not None else 0.0,
            "xpert_mtb_rif": float(xpert) if xpert is not None else 0.0,
            "is_missing_cumulative_doses_taken": 1.0 if cum is None else 0.0,
            "is_missing_height": 1.0 if height is None else 0.0,
            "is_missing_monthly_doses_taken": 1.0 if taken is None else 0.0,
            "is_missing_monthly_missed_doses": 1.0 if missed is None else 0.0,
            "is_missing_pct_adherence": 1.0 if pct is None else 0.0,
            "is_missing_smear_tb_lamp": 1.0 if smear is None else 0.0,
            "is_missing_weight": 1.0 if weight is None else 0.0,
            "is_missing_xpert_mtb_rif": 1.0 if xpert is None else 0.0,
        }

        for j, name in enumerate(temporal_names):
            xt[m, j] = float(row.get(name, 0.0))

    seq_len = max_month + 1
    return xt, seq_len


def _monthly_rule_failure_floor(
    *,
    month: int,
    pct_adherence: float | None,
    smear_tb_lamp: int | None,
    xpert_mtb_rif: int | None,
) -> float:
    """Conservative floor while temporal model preprocessing is being corrected.

    The Bi-LSTM output is currently experimental for live UI inputs. Severe
    non-adherence, positive labs, or missing labs should never collapse to 0%.
    """

    floor = 0.0

    if pct_adherence is None:
        if month > 0:
            # Follow-up visits need dose data to justify low risk.
            floor = max(floor, 0.35)
    else:
        if pct_adherence < 50:
            # 0% adherence -> 80% floor, 50% adherence -> 45% floor.
            floor = max(floor, 0.45 + ((50.0 - pct_adherence) / 50.0) * 0.35)
        elif pct_adherence < 80:
            # 50% adherence -> 45% floor, 80% adherence -> 30% floor.
            floor = max(floor, 0.30 + ((80.0 - pct_adherence) / 30.0) * 0.15)
        elif pct_adherence < 90:
            floor = max(floor, 0.20)

    labs_missing = smear_tb_lamp is None and xpert_mtb_rif is None
    any_positive_lab = smear_tb_lamp == 1 or xpert_mtb_rif == 1

    if any_positive_lab:
        floor = max(floor, 0.55)
    if labs_missing:
        floor = max(floor, 0.30)
        if pct_adherence is not None and pct_adherence < 50:
            floor = max(floor, 0.70)

    return min(max(floor, 0.0), 0.95)


def _apply_monthly_drop_limit(current_failure: float, previous_failure: float | None) -> float:
    if previous_failure is None:
        return current_failure
    max_drop = 0.30
    return max(current_failure, previous_failure - max_drop)


def _latest_failure_probability(records: list[PredictionRow]) -> float | None:
    if not records:
        return None
    return float(records[-1].failure_probability)


def _expected_next_month(records: list[MonthlyRecordRow]) -> int:
    months = [int(r.month) for r in records if r.month is not None]
    return 0 if not months else max(months) + 1


@router.post("/api/patients/{patient_id}/temporal-risk")
def temporal_risk_and_save(
    patient_id: str, body: TemporalRiskRequest, db: Annotated[Session, Depends(get_db)]
):
    # Load patient.
    p = db.get(PatientRow, patient_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Ensure artifacts are loadable early.
    art = load_artifacts()

    month = int(body.month)
    if month < 0 or month > 12:
        raise HTTPException(status_code=400, detail="month must be in [0, 12]")
    if month > 0 and (body.monthly_doses_taken is None or body.monthly_missed_doses is None):
        raise HTTPException(
            status_code=400,
            detail="Doses taken and missed doses are required after baseline. Enter 0 if none were missed.",
        )

    # Collect existing records.
    records = db.scalars(
        select(MonthlyRecordRow)
        .where(MonthlyRecordRow.patient_id == patient_id)
        .order_by(MonthlyRecordRow.month.asc())
    ).all()
    if any(int(r.month) == month for r in records):
        raise HTTPException(status_code=409, detail="Monthly record already exists")
    expected_month = _expected_next_month(records)
    if month != expected_month:
        raise HTTPException(
            status_code=400,
            detail=f"Expected month M{expected_month}, got M{month}",
        )

    # Derive adherence and cumulative doses.
    taken = body.monthly_doses_taken
    missed = body.monthly_missed_doses
    pct = None
    if taken is not None and missed is not None:
        total = taken + missed
        pct = (taken / total) * 100.0 if total > 0 else None

    prior_cum = sum(int(r.monthly_doses_taken or 0) for r in records)
    cum = None
    if taken is not None:
        cum = prior_cum + int(taken)

    # Missing flags.
    is_missing_weight = 1 if body.weight is None else 0
    is_missing_height = 1 if body.height is None else 0
    is_missing_smear = 1 if body.smear_tb_lamp is None else 0
    is_missing_xpert = 1 if body.xpert_mtb_rif is None else 0
    is_missing_taken = 1 if taken is None else 0
    is_missing_missed = 1 if missed is None else 0

    adherence = "poor"
    if pct is not None:
        adherence = "full" if pct >= 90 else "partial" if pct >= 50 else "poor"

    # Build model inputs.
    x_static = _build_static_vector(p, art.static_feature_names)
    # Predict using sequence that includes this new month.
    # Add a temp MonthlyRecordRow-like object by inserting computed values.
    temp_row = MonthlyRecordRow(
        patient_id=patient_id,
        month=month,
        weight=body.weight,
        height=body.height,
        smear_tb_lamp=body.smear_tb_lamp,
        xpert_mtb_rif=body.xpert_mtb_rif,
        monthly_doses_taken=taken,
        monthly_missed_doses=missed,
        cumulative_doses_taken=cum,
        pct_adherence=pct,
        adherence=adherence,
        failure_probability=0.0,
        timestamp=int(time.time() * 1000),
        is_missing_weight=is_missing_weight,
        is_missing_height=is_missing_height,
        is_missing_smear_tb_lamp=is_missing_smear,
        is_missing_xpert_mtb_rif=is_missing_xpert,
        is_missing_monthly_doses_taken=is_missing_taken,
        is_missing_monthly_missed_doses=is_missing_missed,
    )
    xt, seq_len = _build_temporal_matrix(
        temporal_names=art.temporal_feature_names,
        records=[*records, temp_row],
        current_month=month,
    )
    months_used = sorted({int(r.month) for r in [*records, temp_row] if r.month is not None})

    prob_success = predict_success_probability(
        x_temporal_unscaled=xt,
        x_static_unscaled=x_static,
        seq_len=seq_len,
    )
    raw_failure = 1.0 - prob_success

    previous_predictions = db.scalars(
        select(PredictionRow)
        .where(PredictionRow.patient_id == patient_id)
        .order_by(PredictionRow.timestamp.asc())
    ).all()
    previous_failure = _latest_failure_probability(previous_predictions)
    rule_floor = _monthly_rule_failure_floor(
        month=month,
        pct_adherence=pct,
        smear_tb_lamp=body.smear_tb_lamp,
        xpert_mtb_rif=body.xpert_mtb_rif,
    )
    # Previous risk is useful for trend display/debugging, but it must not lock
    # future monthly assessments at an old high-risk value.
    prob_failure = max(raw_failure, rule_floor)
    prob_failure = _apply_monthly_drop_limit(prob_failure, previous_failure)
    prob_failure = min(max(prob_failure, 0.0), 0.95)
    display_success = 1.0 - prob_failure
    label = 1 if prob_failure >= 0.50 else 0

    now = int(time.time() * 1000)

    # Save monthly record.
    row = MonthlyRecordRow(
        patient_id=patient_id,
        month=month,
        weight=body.weight,
        height=body.height,
        smear_tb_lamp=body.smear_tb_lamp,
        xpert_mtb_rif=body.xpert_mtb_rif,
        monthly_doses_taken=taken,
        monthly_missed_doses=missed,
        cumulative_doses_taken=cum,
        pct_adherence=pct,
        adherence=adherence,
        failure_probability=float(prob_failure),
        timestamp=now,
        is_missing_weight=is_missing_weight,
        is_missing_height=is_missing_height,
        is_missing_smear_tb_lamp=is_missing_smear,
        is_missing_xpert_mtb_rif=is_missing_xpert,
        is_missing_monthly_doses_taken=is_missing_taken,
        is_missing_monthly_missed_doses=is_missing_missed,
    )
    db.add(row)

    # Save prediction.
    pred_row = PredictionRow(
        patient_id=patient_id,
        label=int(label),
        failure_probability=float(prob_failure),
        contributions=[],
        features_used={
            "model": "hybrid_bi_lstm_best",
            "month": month,
            "successProbability": display_success,
            "rawSuccessProbability": prob_success,
            "rawFailureProbability": raw_failure,
            "ruleFailureFloor": rule_floor,
            "previousFailureProbability": previous_failure,
            "adjustedFailureProbability": prob_failure,
            "threshold": art.threshold,
            "seqLen": seq_len,
            "monthsUsed": months_used,
            "expectedMonth": expected_month,
            "riskPolicy": "conservative_monthly_floor_v1",
        },
        timestamp=now,
    )
    db.add(pred_row)

    db.commit()

    return {
        "label": int(label),
        "failureProbability": float(prob_failure),
        "successProbability": float(display_success),
        "threshold": float(art.threshold),
        "modelName": "hybrid_bi_lstm_best",
        "month": month,
        "monthsUsed": months_used,
    }


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
                "height": float(r.height) if r.height is not None else None,
                "smearTbLamp": int(r.smear_tb_lamp) if r.smear_tb_lamp is not None else None,
                "xpertMtbRif": int(r.xpert_mtb_rif) if r.xpert_mtb_rif is not None else None,
                "monthlyDosesTaken": int(r.monthly_doses_taken) if r.monthly_doses_taken is not None else None,
                "monthlyMissedDoses": int(r.monthly_missed_doses) if r.monthly_missed_doses is not None else None,
                "cumulativeDosesTaken": int(r.cumulative_doses_taken) if r.cumulative_doses_taken is not None else None,
                "pctAdherence": float(r.pct_adherence) if r.pct_adherence is not None else None,
                "adherence": r.adherence,
                "failureProbability": float(r.failure_probability),
                "timestamp": int(r.timestamp),
                "xrayIds": record_id_to_xray_ids.get(int(r.month)) or None,
                "isMissingWeight": int(r.is_missing_weight),
                "isMissingHeight": int(r.is_missing_height),
                "isMissingSmearTbLamp": int(r.is_missing_smear_tb_lamp),
                "isMissingXpertMtbRif": int(r.is_missing_xpert_mtb_rif),
                "isMissingMonthlyDosesTaken": int(r.is_missing_monthly_doses_taken),
                "isMissingMonthlyMissedDoses": int(r.is_missing_monthly_missed_doses),
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
        height=body.height,
        smear_tb_lamp=body.smear_tb_lamp,
        xpert_mtb_rif=body.xpert_mtb_rif,
        monthly_doses_taken=body.monthly_doses_taken,
        monthly_missed_doses=body.monthly_missed_doses,
        cumulative_doses_taken=body.cumulative_doses_taken,
        pct_adherence=body.pct_adherence,
        adherence=body.adherence,
        failure_probability=body.failure_probability,
        timestamp=body.timestamp,
        is_missing_weight=body.is_missing_weight,
        is_missing_height=body.is_missing_height,
        is_missing_smear_tb_lamp=body.is_missing_smear_tb_lamp,
        is_missing_xpert_mtb_rif=body.is_missing_xpert_mtb_rif,
        is_missing_monthly_doses_taken=body.is_missing_monthly_doses_taken,
        is_missing_monthly_missed_doses=body.is_missing_monthly_missed_doses,
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
