from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CamelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class ContributionItem(CamelModel):
    feature: str
    delta: float
    direction: Literal["risk", "protective"]


class PredictionRecord(CamelModel):
    label: Literal[0, 1]
    failure_probability: float = Field(alias="failureProbability")
    contributions: list[ContributionItem]
    features_used: dict[str, Any] | None = Field(default=None, alias="featuresUsed")
    timestamp: int


class MonthlyRecord(CamelModel):
    month: int
    weight: float | None = None
    height: float | None = None
    smear_tb_lamp: int | None = Field(default=None, alias="smearTbLamp")
    xpert_mtb_rif: int | None = Field(default=None, alias="xpertMtbRif")
    monthly_doses_taken: int | None = Field(default=None, alias="monthlyDosesTaken")
    monthly_missed_doses: int | None = Field(default=None, alias="monthlyMissedDoses")
    cumulative_doses_taken: int | None = Field(default=None, alias="cumulativeDosesTaken")
    pct_adherence: float | None = Field(default=None, alias="pctAdherence")
    adherence: Literal["full", "partial", "poor"]
    failure_probability: float = Field(alias="failureProbability")
    timestamp: int
    xray_ids: list[str] | None = Field(default=None, alias="xrayIds")
    is_missing_weight: int = Field(default=0, alias="isMissingWeight")
    is_missing_height: int = Field(default=0, alias="isMissingHeight")
    is_missing_smear_tb_lamp: int = Field(default=0, alias="isMissingSmearTbLamp")
    is_missing_xpert_mtb_rif: int = Field(default=0, alias="isMissingXpertMtbRif")
    is_missing_monthly_doses_taken: int = Field(default=0, alias="isMissingMonthlyDosesTaken")
    is_missing_monthly_missed_doses: int = Field(default=0, alias="isMissingMonthlyMissedDoses")


class PatientFeatures(CamelModel):
    age: int
    days_to_treatment: int = Field(alias="daysToTreatment")
    year: int
    sex: str
    anatomical_site: str = Field(alias="anatomicalSite")
    registration_group: str = Field(alias="registrationGroup")
    bacteriologic_status: str = Field(alias="bacteriologicStatus")
    microscopy_result: str = Field(alias="microscopyResult")
    source_of_patient: str = Field(alias="sourceOfPatient")
    type: str
    province: str
    city_municipality: str = Field(alias="cityMunicipality")
    treatment_health_facility: str = Field(alias="treatmentHealthFacility")
    screening_diagnosing_health_facility: str = Field(alias="screeningDiagnosingHealthFacility")
    # Extended fields for temporal model — stored in DB but not fed to static ONNX model
    xpert_mtb_rif: str | None = Field(default=None, alias="xpertMtbRif")
    drug_resistance_status: str | None = Field(default=None, alias="drugResistanceStatus")
    baseline_height_cm: float | None = Field(default=None, alias="baselineHeightCm")
    baseline_weight_kg: float | None = Field(default=None, alias="baselineWeightKg")
    bp_systolic: float | None = Field(default=None, alias="bpSystolic")
    bp_diastolic: float | None = Field(default=None, alias="bpDiastolic")
    heart_rate: float | None = Field(default=None, alias="heartRate")
    o2_sat: float | None = Field(default=None, alias="o2Sat")
    co_morbidities: str | None = Field(default=None, alias="coMorbidities")
    chest_x_ray_at_notification: str | None = Field(default=None, alias="chestXRayAtNotification")
    civil_status: str | None = Field(default=None, alias="civilStatus")
    nationality: str | None = Field(default=None, alias="nationality")
    diagnosis: str | None = None


class PatientCreate(CamelModel):
    id: str | None = None
    name: str
    medical_id: str = Field(alias="medicalId")
    features: PatientFeatures
    treatment_regimen: Literal["hrze", "mdr", "xdr"] | None = Field(
        default=None, alias="treatmentRegimen"
    )
    treatment_start_date: str | None = Field(default=None, alias="treatmentStartDate")
    created_at: int | None = Field(default=None, alias="createdAt")


class Patient(CamelModel):
    id: str
    name: str
    medical_id: str = Field(alias="medicalId")
    features: PatientFeatures
    treatment_regimen: Literal["hrze", "mdr", "xdr"] | None = Field(
        default=None, alias="treatmentRegimen"
    )
    treatment_start_date: str | None = Field(default=None, alias="treatmentStartDate")
    created_at: int = Field(alias="createdAt")
    predictions: list[PredictionRecord]
    monthly_records: list[MonthlyRecord] = Field(alias="monthlyRecords")
    intake_xray_ids: list[str] | None = Field(default=None, alias="intakeXrayIds")


class MonthlyRecordCreate(CamelModel):
    month: int
    weight: float | None = None
    height: float | None = None
    smear_tb_lamp: int | None = Field(default=None, alias="smearTbLamp")
    xpert_mtb_rif: int | None = Field(default=None, alias="xpertMtbRif")
    monthly_doses_taken: int | None = Field(default=None, alias="monthlyDosesTaken")
    monthly_missed_doses: int | None = Field(default=None, alias="monthlyMissedDoses")
    cumulative_doses_taken: int | None = Field(default=None, alias="cumulativeDosesTaken")
    pct_adherence: float | None = Field(default=None, alias="pctAdherence")
    adherence: Literal["full", "partial", "poor"]
    failure_probability: float = Field(alias="failureProbability")
    timestamp: int
    is_missing_weight: int = Field(default=0, alias="isMissingWeight")
    is_missing_height: int = Field(default=0, alias="isMissingHeight")
    is_missing_smear_tb_lamp: int = Field(default=0, alias="isMissingSmearTbLamp")
    is_missing_xpert_mtb_rif: int = Field(default=0, alias="isMissingXpertMtbRif")
    is_missing_monthly_doses_taken: int = Field(default=0, alias="isMissingMonthlyDosesTaken")
    is_missing_monthly_missed_doses: int = Field(default=0, alias="isMissingMonthlyMissedDoses")


class PredictionCreate(CamelModel):
    label: Literal[0, 1]
    failure_probability: float = Field(alias="failureProbability")
    contributions: list[ContributionItem]
    features_used: dict[str, Any] | None = Field(default=None, alias="featuresUsed")
    timestamp: int


class XrayUploadResponse(CamelModel):
    id: str
    sha256: str
    size_bytes: int = Field(alias="sizeBytes")
    mime: str


class XrayMetadata(CamelModel):
    id: str
    patient_id: str = Field(alias="patientId")
    kind: Literal["intake", "monthly"]
    month: int | None = None
    mime: str
    name: str
    sha256: str
    size_bytes: int = Field(alias="sizeBytes")
    created_at: int = Field(alias="createdAt")


class InterpretRequest(BaseModel):
    patient_name: str
    age: int
    sex: Literal["M", "F"]
    bacteriologic_status: str
    microscopy_result: str
    anatomical_site: Literal["P", "EP"] | None = None   # Optional; V2 CSV doesn't have this
    registration_group: str
    source_of_patient: str | None = None                 # Optional; V2 CSV doesn't have this
    type: str | None = None                              # Optional; V2 CSV doesn't have this
    days_to_treatment: int
    failure_probability: float
    contributions: list[ContributionItem]
    month_of_prediction: int | None = None               # V2: M0-M12 horizon; None = unspecified
