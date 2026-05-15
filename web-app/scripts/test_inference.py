#!/usr/bin/env python3
"""
Validate the exported ONNX model with representative patient profiles.

Mirrors the encoding logic in inference.ts:
- Numerics: raw values (Age_Final, Days_To_Treatment, Year)
- Categoricals: integer label codes from feature_encodings.json

Run from project root:
    python web-app/scripts/test_inference.py
"""

import json
import os
import sys
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE, "web-app", "public", "model", "tb_outcome_prediction.onnx")
ENCODINGS_PATH = os.path.join(BASE, "web-app", "src", "data", "feature_encodings.json")

with open(ENCODINGS_PATH) as f:
    encodings = json.load(f)

CATEGORICAL_KEYS = [
    "Sex", "Anatomical_Site", "Registration_Group", "Bacteriologic_Status",
    "Microscopy_Result", "Source_of_Patient", "Type", "Province",
    "City_Municipality", "Treatment_Health_Facility",
    "Screening_Diagnosing_Health_Facility",
]


def encode_cat(col: str, label: str) -> int:
    m = encodings.get(col, {})
    if label in m:
        return int(m[label])
    lower = label.lower()
    for k, v in m.items():
        if k.lower() == lower:
            return int(v)
    return 0


def build_feeds(age, days_to_treatment, year, sex, anatomical_site,
                registration_group, bacteriologic_status, microscopy_result,
                source_of_patient, type_, province, city, treatment_facility,
                screening_facility):
    values = [
        float(age),
        float(days_to_treatment),
        float(year),
        float(encode_cat("Sex", sex)),
        float(encode_cat("Anatomical_Site", anatomical_site)),
        float(encode_cat("Registration_Group", registration_group)),
        float(encode_cat("Bacteriologic_Status", bacteriologic_status)),
        float(encode_cat("Microscopy_Result", microscopy_result)),
        float(encode_cat("Source_of_Patient", source_of_patient)),
        float(encode_cat("Type", type_)),
        float(encode_cat("Province", province)),
        float(encode_cat("City_Municipality", city)),
        float(encode_cat("Treatment_Health_Facility", treatment_facility)),
        float(encode_cat("Screening_Diagnosing_Health_Facility", screening_facility)),
    ]
    names = [
        "Age_Final", "Days_To_Treatment", "Year", "Sex", "Anatomical_Site",
        "Registration_Group", "Bacteriologic_Status", "Microscopy_Result",
        "Source_of_Patient", "Type", "Province", "City_Municipality",
        "Treatment_Health_Facility", "Screening_Diagnosing_Health_Facility",
    ]
    return {n: np.array([[v]], dtype=np.float32) for n, v in zip(names, values)}


def run(sess, feeds):
    out = sess.run(None, feeds)
    # out[0] = label, out[1] = probabilities [batch, 2]
    probs = out[1][0]
    return float(probs[0]), float(probs[1])  # (failure_prob, success_prob)


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}")
        sys.exit(1)

    sess = ort.InferenceSession(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Inputs:  {[i.name for i in sess.get_inputs()]}")
    print(f"Outputs: {[o.name for o in sess.get_outputs()]}")
    print()

    failures = []

    # --- Patient A: high-risk profile ---
    # DRTB type, TX AFTER FAILURE registration, elderly male, long delay
    feeds_a = build_feeds(
        age=65, days_to_treatment=30, year=2022,
        sex="M", anatomical_site="P",
        registration_group="TX AFTER FAILURE",
        bacteriologic_status="Bacteriologically-confirmed TB",
        microscopy_result="3+",
        source_of_patient="PUBLIC HEALTH CENTER",
        type_="DRTB",
        province="BENGUET",
        city="BAGUIO CITY",
        treatment_facility="BAGUIO GENERAL HOSPITAL AND MEDICAL CENTER - IDOTS",
        screening_facility="BAGUIO GENERAL HOSPITAL AND MEDICAL CENTER - IDOTS",
    )
    fail_a, succ_a = run(sess, feeds_a)
    print(f"Patient A (high-risk DRTB):  failure={fail_a:.4f}  success={succ_a:.4f}")

    prob_sum_a = abs(fail_a + succ_a - 1.0)
    if prob_sum_a > 0.01:
        failures.append(f"Patient A: probabilities don't sum to 1 (sum={fail_a+succ_a:.4f})")
    if not (0.0 <= fail_a <= 1.0):
        failures.append(f"Patient A: failure prob out of range ({fail_a})")

    # --- Patient B: low-risk profile ---
    # New DSTB, young female, quick treatment start, bacteriologically confirmed
    feeds_b = build_feeds(
        age=25, days_to_treatment=2, year=2021,
        sex="F", anatomical_site="P",
        registration_group="NEW",
        bacteriologic_status="Bacteriologically-confirmed TB",
        microscopy_result="1+",
        source_of_patient="COMMUNITY",
        type_="DSTB",
        province="BENGUET",
        city="BAGUIO CITY",
        treatment_facility="BAGUIO HEALTH SERVICES - DOTS",
        screening_facility="BAGUIO HEALTH SERVICES - DOTS",
    )
    fail_b, succ_b = run(sess, feeds_b)
    print(f"Patient B (low-risk new DSTB): failure={fail_b:.4f}  success={succ_b:.4f}")

    prob_sum_b = abs(fail_b + succ_b - 1.0)
    if prob_sum_b > 0.01:
        failures.append(f"Patient B: probabilities don't sum to 1 (sum={fail_b+succ_b:.4f})")
    if not (0.0 <= fail_b <= 1.0):
        failures.append(f"Patient B: failure prob out of range ({fail_b})")

    # --- Patient C: edge case — all-zero inputs ---
    names = [i.name for i in sess.get_inputs()]
    feeds_c = {n: np.array([[0.0]], dtype=np.float32) for n in names}
    fail_c, succ_c = run(sess, feeds_c)
    print(f"Patient C (all-zero inputs):  failure={fail_c:.4f}  success={succ_c:.4f}")

    if not (0.0 <= fail_c <= 1.0):
        failures.append(f"Patient C: failure prob out of range ({fail_c})")
    if abs(fail_c + succ_c - 1.0) > 0.01:
        failures.append(f"Patient C: probs don't sum to 1 ({fail_c+succ_c:.4f})")

    # --- Ordering assertion: high-risk should have higher failure prob than low-risk ---
    print()
    if fail_a > fail_b:
        print("PASS: Patient A failure prob > Patient B failure prob (as expected)")
    else:
        failures.append(
            f"ORDERING: expected high-risk > low-risk but got {fail_a:.4f} <= {fail_b:.4f}"
        )

    print()
    if failures:
        print("FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)
    else:
        print("All assertions passed.")


if __name__ == "__main__":
    main()
