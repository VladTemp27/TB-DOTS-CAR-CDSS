import pytest
from pathlib import Path
from slm_shap_pipeline.data_loader import PatientRow, TestCohort, load_test_patients
from slm_shap_pipeline.config import PipelineConfig

CSV_PATH = Path("dataset/temporal/output/cleaned_human_readable.csv")
SKIP_IF_MISSING = pytest.mark.skipif(
    not CSV_PATH.exists(), reason="cleaned_human_readable.csv not present"
)


@SKIP_IF_MISSING
def test_load_test_patients_returns_44():
    cfg = PipelineConfig()
    cohort = load_test_patients(cfg)
    assert len(cohort.patients) == 44, f"Expected 44 test patients, got {len(cohort.patients)}"


@SKIP_IF_MISSING
def test_patient_row_has_required_fields():
    cfg = PipelineConfig()
    cohort = load_test_patients(cfg)
    p = cohort.patients[0]
    assert isinstance(p.patient_id, str)
    assert isinstance(p.age, (int, float))
    assert p.sex in ("M", "F"), f"sex must be M or F, got {p.sex!r}"
    assert isinstance(p.days_to_treatment, (int, float))
    assert isinstance(p.registration_group, str)
    assert isinstance(p.bacteriologic_status, str)
    assert isinstance(p.microscopy_result, str)


@SKIP_IF_MISSING
def test_sex_encoding_normalized():
    """CSV stores Male/Female; PatientRow must store M/F."""
    cfg = PipelineConfig()
    cohort = load_test_patients(cfg)
    for r in cohort.patients:
        assert r.sex in ("M", "F"), f"patient {r.patient_id}: sex={r.sex!r} not normalized"


@SKIP_IF_MISSING
def test_sorted_by_patient_id():
    cfg = PipelineConfig()
    cohort = load_test_patients(cfg)
    ids = [r.patient_id for r in cohort.patients]
    assert ids == sorted(ids), "Rows must be sorted by patient_id for deterministic output"


@SKIP_IF_MISSING
def test_idempotent_split():
    """Same 44 patients returned on every call (split is deterministic)."""
    cfg = PipelineConfig()
    ids_a = [r.patient_id for r in load_test_patients(cfg).patients]
    ids_b = [r.patient_id for r in load_test_patients(cfg).patients]
    assert ids_a == ids_b


@SKIP_IF_MISSING
def test_cohort_has_scaler_hashes():
    cfg = PipelineConfig()
    cohort = load_test_patients(cfg)
    assert len(cohort.scaler_static_hash) == 64
    assert len(cohort.scaler_temporal_hash) == 64


@SKIP_IF_MISSING
def test_scaler_hashes_are_deterministic():
    """Running twice must produce identical hashes."""
    cfg = PipelineConfig()
    c1 = load_test_patients(cfg)
    c2 = load_test_patients(cfg)
    assert c1.scaler_static_hash == c2.scaler_static_hash
    assert c1.scaler_temporal_hash == c2.scaler_temporal_hash
