from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from backend.db import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    medical_id: Mapped[str] = mapped_column(String, nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    treatment_regimen: Mapped[str | None] = mapped_column(String, nullable=True)
    treatment_start_date: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)

    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    monthly_records: Mapped[list[MonthlyRecord]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )
    xrays: Mapped[list[Xray]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False)

    label: Mapped[int] = mapped_column(Integer, nullable=False)
    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
    contributions: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    features_used: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[int] = mapped_column(Integer, nullable=False)

    patient: Mapped[Patient] = relationship(back_populates="predictions")


class MonthlyRecord(Base):
    __tablename__ = "monthly_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False)

    month: Mapped[int] = mapped_column(Integer, nullable=False)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    height: Mapped[float | None] = mapped_column(Float, nullable=True)
    smear_tb_lamp: Mapped[int | None] = mapped_column(Integer, nullable=True)
    xpert_mtb_rif: Mapped[int | None] = mapped_column(Integer, nullable=True)
    monthly_doses_taken: Mapped[int | None] = mapped_column(Integer, nullable=True)
    monthly_missed_doses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cumulative_doses_taken: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pct_adherence: Mapped[float | None] = mapped_column(Float, nullable=True)
    adherence: Mapped[str] = mapped_column(String, nullable=False)
    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[int] = mapped_column(Integer, nullable=False)
    is_missing_weight: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_missing_height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_missing_smear_tb_lamp: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_missing_xpert_mtb_rif: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_missing_monthly_doses_taken: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_missing_monthly_missed_doses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    patient: Mapped[Patient] = relationship(back_populates="monthly_records")


class Xray(Base):
    __tablename__ = "xrays"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), nullable=False)

    kind: Mapped[str] = mapped_column(String, nullable=False)  # intake | monthly
    month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    rel_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)

    patient: Mapped[Patient] = relationship(back_populates="xrays")
