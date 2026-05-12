import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_db
from backend.models import Patient as PatientRow
from backend.models import Xray as XrayRow
from backend.schemas import XrayMetadata, XrayUploadResponse
from backend.settings import data_dir, max_upload_bytes
from backend.xray_store import XrayFileStore, new_xray_id

router = APIRouter()


def _xray_store() -> XrayFileStore:
    return XrayFileStore(data_dir() / "xrays")


@router.post("/api/xrays")
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


@router.get("/api/xrays/{xray_id}/file")
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


@router.get("/api/patients/{patient_id}/xrays")
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
