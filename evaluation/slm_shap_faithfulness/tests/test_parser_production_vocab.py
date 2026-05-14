"""Tests that parser recognises production feature vocabulary."""
import pytest
from evaluation.slm_shap_faithfulness.parser import parse_explanation


def test_recognises_age():
    result = parse_explanation("Age strongly increases risk.")
    features = [c["feature"] for c in result["claims"]]
    assert "Age" in features


def test_recognises_registration_group():
    result = parse_explanation("Registration Group decreases risk.")
    features = [c["feature"] for c in result["claims"]]
    assert "Registration Group" in features


def test_recognises_bacteriologic_status():
    result = parse_explanation("Bacteriologic Status strongly increases risk.")
    features = [c["feature"] for c in result["claims"]]
    assert "Bacteriologic Status" in features


def test_recognises_microscopy_result():
    result = parse_explanation("Microscopy Result decreases risk.")
    features = [c["feature"] for c in result["claims"]]
    assert "Microscopy Result" in features


def test_recognises_treatment_adherence():
    result = parse_explanation("Treatment Adherence strongly decreases risk.")
    features = [c["feature"] for c in result["claims"]]
    assert "Treatment Adherence" in features


def test_recognises_days_to_treatment():
    result = parse_explanation("Days To Treatment moderately increases risk.")
    features = [c["feature"] for c in result["claims"]]
    assert "Days To Treatment" in features


def test_recognises_mixed_sign():
    result = parse_explanation("Body Weight had a mixed effect on risk.")
    features = [c["feature"] for c in result["claims"]]
    assert "Body Weight" in features
    claim = next(c for c in result["claims"] if c["feature"] == "Body Weight")
    assert claim["direction"] == "mixed"   # NOTE: use "direction" not "sign" — consistent with existing schema


def test_does_not_recognise_stale_bmi():
    result = parse_explanation("BMI strongly increases risk.")
    features = [c["feature"] for c in result["claims"]]
    assert "BMI" not in features


def test_does_not_recognise_stale_province():
    result = parse_explanation("Province moderately decreases risk.")
    features = [c["feature"] for c in result["claims"]]
    assert "Province" not in features
