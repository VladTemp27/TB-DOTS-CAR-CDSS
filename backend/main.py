import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncGenerator, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import (
    MAX_TOKENS,
    MODEL_PATH,
    N_CTX,
    N_GPU_LAYERS,
    N_THREADS,
    REPEAT_PENALTY,
    TEMPERATURE,
    TOP_P,
)
from backend.db import get_db
from backend.models import MonthlyRecord as MonthlyRecordRow
from backend.models import Patient as PatientRow
from backend.models import Prediction as PredictionRow
from backend.models import Xray as XrayRow
from backend.prompt import build_prompt
from backend.schemas import (
    MonthlyRecordCreate,
    Patient,
    PatientCreate,
    PredictionCreate,
    XrayMetadata,
    XrayUploadResponse,
)
from backend.settings import data_dir, max_upload_bytes
from backend.xray_store import XrayFileStore, new_xray_id

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("medgemma")

# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment,misc]

llm: "Llama | None" = None
_load_error: str | None = None

# Runtime stats — updated during inference, read by /api/stats
_stats: dict = {
    "requests_served": 0,
    "requests_active": 0,
    "last_patient": None,
    "last_tokens": 0,
    "last_elapsed_s": 0.0,
    "model_load_time_s": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, _load_error
    app.state.llm_lock = asyncio.Lock()  # Created inside running event loop

    if Llama is None:
        _load_error = "llama_cpp not installed — run: pip install llama-cpp-python"
        log.error(_load_error)
        yield
        return

    model_path = Path(MODEL_PATH)
    log.info("Model path resolved to: %s", model_path.resolve())

    if not model_path.exists():
        _load_error = f"Model file not found: {model_path.resolve()}"
        log.error(_load_error)
        yield
        return

    log.info(
        "Loading MedGemma (size=%.1f GB, n_ctx=%d, n_threads=%d, n_gpu_layers=%d)...",
        model_path.stat().st_size / 1e9,
        N_CTX,
        N_THREADS,
        N_GPU_LAYERS,
    )
    t0 = time.monotonic()
    try:
        llm = await asyncio.to_thread(
            Llama,
            model_path=str(model_path),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )
        elapsed = time.monotonic() - t0
        _stats["model_load_time_s"] = round(elapsed, 1)
        log.info("MedGemma ready — loaded in %.1fs", elapsed)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        log.exception("Model load failed after %.1fs", time.monotonic() - t0)
    yield


app = FastAPI(title="MedGemma Clinical CDSS API", lifespan=lifespan)

# CORS — allow all origins for development (no credentials needed for SSE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


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
            # Client buckets by month number.
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


def _xray_store() -> XrayFileStore:
    return XrayFileStore(data_dir() / "xrays")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ContributionItem(BaseModel):
    feature: str
    delta: float
    direction: Literal["risk", "protective"]


class InterpretRequest(BaseModel):
    patient_name: str
    age: int
    sex: Literal["M", "F"]
    bacteriologic_status: str
    microscopy_result: str
    anatomical_site: Literal["P", "EP"]
    registration_group: str
    source_of_patient: str
    type: str
    days_to_treatment: int
    failure_probability: float
    contributions: list[ContributionItem]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


async def _generate(prompt: str, lock: asyncio.Lock, patient_name: str, request: Request) -> AsyncGenerator:
    log.debug(
        "Inference queued for %r — prompt length: %d chars", patient_name, len(prompt)
    )
    # Yield before acquiring the lock so the SSE byte stream opens immediately
    # (sse_starlette needs a first yield to finalise response headers).
    # requests_active is incremented inside the lock so it only reflects
    # requests that are actually running inference, not ones waiting to acquire.
    yield {"data": json.dumps({"status": "thinking"})}
    async with lock:
        _stats["requests_active"] += 1
        log.info("Inference started for %r", patient_name)
        t0 = time.monotonic()
        token_count = 0

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()

        def run_inference():
            nonlocal token_count
            try:
                log.debug("llm() call entered for %r", patient_name)
                for chunk in llm(  # type: ignore[misc]
                    prompt,
                    stream=True,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repeat_penalty=REPEAT_PENALTY,
                ):
                    if stop_event.is_set():
                        log.debug("Inference cancelled for %r after %d tokens", patient_name, token_count)
                        break
                    token = chunk.get("choices", [{}])[0].get("text", "")
                    if token:
                        token_count += 1
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                log.exception("Inference error for %r: %s", patient_name, e)
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

        t = threading.Thread(target=run_inference, daemon=True)
        t.start()
        try:
            while True:
                if await request.is_disconnected():
                    log.info("Client disconnected for %r — stopping inference", patient_name)
                    break
                try:
                    kind, value = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue  # re-poll is_disconnected during long prefill phase
                if kind == "token":
                    yield {"data": json.dumps({"token": value})}
                elif kind == "done":
                    _stats["requests_served"] += 1
                    elapsed = time.monotonic() - t0
                    log.info(
                        "Inference done for %r — %d tokens in %.1fs (%.1f tok/s)",
                        patient_name, token_count, elapsed,
                        token_count / elapsed if elapsed > 0 else 0,
                    )
                    yield {"data": json.dumps({"token": "", "done": True})}
                    break
                elif kind == "error":
                    log.error("Inference stream error for %r: %s", patient_name, value)
                    yield {"data": json.dumps({"error": value, "done": True})}
                    break
        finally:
            stop_event.set()
            await asyncio.to_thread(t.join)
            _stats["requests_active"] -= 1
            _stats["last_patient"] = patient_name
            _stats["last_tokens"] = token_count
            _stats["last_elapsed_s"] = round(time.monotonic() - t0, 1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    if _load_error:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "error": _load_error,
                "model": "medgemma-1.5-4b-it-IQ4_XS",
                "n_ctx": N_CTX,
            },
        )
    if llm is None:
        return {"status": "loading", "model": "medgemma-1.5-4b-it-IQ4_XS", "n_ctx": N_CTX}
    return {"status": "ready", "model": "medgemma-1.5-4b-it-IQ4_XS", "n_ctx": N_CTX}


@app.get("/api/stats")
async def stats_endpoint(request: Request):
    import sys, resource as _resource
    lock = request.app.state.llm_lock
    try:
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        # macOS: bytes; Linux: kilobytes
        ram_mb = rss // (1024 * 1024) if sys.platform == "darwin" else rss // 1024
    except Exception:
        ram_mb = 0
    return {
        "status": "error" if _load_error else ("loading" if llm is None else "ready"),
        "error": _load_error,
        "inference_active": lock.locked(),
        "ram_mb": ram_mb,
        **_stats,
    }


@app.post("/api/interpret")
async def interpret(req: InterpretRequest, request: Request):
    log.info(
        "POST /api/interpret — patient=%r failure_prob=%.1f%% contributions=%d",
        req.patient_name,
        req.failure_probability * 100,
        len(req.contributions),
    )

    if llm is None:
        error_msg = _load_error or "Model not loaded"
        log.warning("Rejecting request — model unavailable: %s", error_msg)

        async def err():
            yield {"data": json.dumps({"error": error_msg, "done": True})}

        return EventSourceResponse(err())

    lock = request.app.state.llm_lock
    prompt = build_prompt(req)
    log.debug("Prompt built (%d chars)", len(prompt))
    return EventSourceResponse(_generate(prompt, lock, req.patient_name, request))


@app.get("/api/patients")
def list_patients(db: Annotated[Session, Depends(get_db)]):
    patients = db.scalars(select(PatientRow).order_by(PatientRow.created_at.desc())).all()
    return [_patient_to_api(db, p) for p in patients]


@app.post("/api/patients")
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


@app.put("/api/patients/{patient_id}")
def upsert_patient(
    patient_id: str, body: PatientCreate, db: Annotated[Session, Depends(get_db)]
):
    # Used by the web app to persist edits (regimen, start date, etc.).
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
    # Preserve original created_at if already set.
    row.created_at = row.created_at or created_at
    db.commit()
    return _patient_to_api(db, row)


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str, db: Annotated[Session, Depends(get_db)]):
    p = db.get(PatientRow, patient_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return _patient_to_api(db, p)


@app.post("/api/patients/{patient_id}/monthly-records")
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


@app.post("/api/patients/{patient_id}/predictions")
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


@app.post("/api/xrays")
async def upload_xray(
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile,
    patient_id: str,
    kind: Literal["intake", "monthly"],
    month: int | None = None,
):
    p = db.get(PatientRow, patient_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    if kind == "monthly" and month is None:
        raise HTTPException(status_code=400, detail="month is required for monthly xrays")

    xray_id = new_xray_id()
    store = _xray_store()
    rel_path, sha256, size_bytes, mime = await store.save(
        xray_id=xray_id, file=file, max_bytes=max_upload_bytes()
    )

    row = XrayRow(
        id=xray_id,
        patient_id=patient_id,
        kind=kind,
        month=month,
        mime=mime,
        name=file.filename or xray_id,
        sha256=sha256,
        size_bytes=size_bytes,
        rel_path=rel_path,
        created_at=int(time.time() * 1000),
    )
    db.add(row)
    db.commit()
    return XrayUploadResponse(id=xray_id, sha256=sha256, sizeBytes=size_bytes, mime=mime).model_dump(
        by_alias=True
    )


@app.get("/api/xrays/{xray_id}/file")
def get_xray_file(xray_id: str, db: Annotated[Session, Depends(get_db)]):
    x = db.get(XrayRow, xray_id)
    if x is None:
        raise HTTPException(status_code=404, detail="X-ray not found")

    root = (data_dir() / "xrays").resolve()
    file_path = (root / x.rel_path).resolve()
    if root not in file_path.parents:
        raise HTTPException(status_code=500, detail="Invalid stored path")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="X-ray file missing")
    return FileResponse(path=str(file_path), media_type=x.mime, filename=x.name)


@app.get("/api/patients/{patient_id}/xrays")
def list_patient_xrays(patient_id: str, db: Annotated[Session, Depends(get_db)]):
    p = db.get(PatientRow, patient_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    xrays = db.scalars(select(XrayRow).where(XrayRow.patient_id == patient_id)).all()
    return [
        XrayMetadata(
            id=x.id,
            patientId=x.patient_id,
            kind=x.kind,
            month=x.month,
            mime=x.mime,
            name=x.name,
            sha256=x.sha256,
            sizeBytes=x.size_bytes,
            createdAt=x.created_at,
        ).model_dump(by_alias=True)
        for x in xrays
    ]
