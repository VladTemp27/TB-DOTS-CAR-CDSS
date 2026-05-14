def build_prompt(req) -> str:
    """Build the Gemma 3 chat template string from the request data.

    Supports three V2 pipeline extensions:
    - Blind condition: req.contributions == [] → inference-only instruction
    - Missing-field omit: anatomical_site/source_of_patient/type may be None
    - Month context: req.month_of_prediction added to profile when present
    """
    sex_full = "Male" if req.sex == "M" else "Female"
    failure_pct = req.failure_probability * 100

    # Only compute anatomical site display if the field is present
    anatomical_site_full: str | None = None
    if req.anatomical_site is not None:
        anatomical_site_full = (
            "PTB (Pulmonary)" if req.anatomical_site == "P" else "EPTB (Extra-pulmonary)"
        )

    # Build value_map for contribution annotations (only include available fields)
    value_map: dict[str, str] = {
        "Age": f"{req.age} years",
        "Days To Treatment": f"{req.days_to_treatment} days",
        "Sex": sex_full,
        "Registration Group": req.registration_group,
        "Bacteriologic Status": req.bacteriologic_status,
        "Microscopy Result": req.microscopy_result,
    }
    if anatomical_site_full is not None:
        value_map["Anatomical Site"] = anatomical_site_full
    if req.source_of_patient is not None:
        value_map["Source of Patient"] = req.source_of_patient
    if getattr(req, "type", None) is not None:
        value_map["Type"] = req.type

    # Patch 1: Blind vs sighted contributions block
    if req.contributions:
        top_contributions = req.contributions[:5]
        contributions_lines = []
        for c in top_contributions:
            value_note = (
                f" (patient value: {value_map[c.feature]})" if c.feature in value_map else ""
            )
            direction = "increases" if c.delta > 0 else "decreases"
            contributions_lines.append(
                f"- {c.feature}{value_note}: {c.delta * 100:+.1f}% impact ({direction} failure risk)"
            )
        contributions_text = "\n".join(contributions_lines)
    else:
        contributions_text = (
            "Top contributing factors were not provided. Based on the patient "
            "profile and the predicted probability, explicitly list the 3-5 "
            "factors you believe most influenced this prediction, then provide "
            "the three sections below reasoning from those factors."
        )

    # Build profile lines (Patch 2: skip None fields)
    profile_lines = [
        f"- Name: {req.patient_name}",
        f"- Age: {req.age} years | Sex: {sex_full}",
        f"- Bacteriologic Status: {req.bacteriologic_status}",
        f"- Microscopy Result: {req.microscopy_result}",
        f"- Registration Group: {req.registration_group}",
        f"- Days from Diagnosis to Treatment Start: {req.days_to_treatment}",
    ]
    if anatomical_site_full is not None:
        profile_lines.insert(4, f"- Anatomical Site: {anatomical_site_full}")
    if req.source_of_patient is not None:
        profile_lines.append(f"- Source of Patient: {req.source_of_patient}")
    if getattr(req, "type", None) is not None:
        profile_lines.append(f"- TB Type: {req.type}")
    # Patch 3: Month context
    if getattr(req, "month_of_prediction", None) is not None:
        profile_lines.append(
            f"- Months of observation: {req.month_of_prediction} "
            f"(prediction made at end of month {req.month_of_prediction})"
        )

    profile_text = "\n".join(profile_lines)

    content = (
        "You are a clinical decision support assistant for TB treatment in the Philippines "
        "Cordillera Administrative Region (CAR). You interpret ML model predictions for "
        "healthcare workers using DOTS guidelines.\n\n"
        f"Patient profile:\n"
        f"{profile_text}\n\n"
        f"ML model prediction: {failure_pct:.0f}% treatment failure probability\n\n"
        f"Top contributing factors:\n"
        f"{contributions_text}\n\n"
        "Provide a structured clinical interpretation in three sections:\n"
        "1. **Risk Assessment Summary** (1-2 sentences interpreting what the probability means for this patient)\n"
        "2. **Key Risk Factors** — For each top contributing factor, write one short paragraph "
        "in this format: '**[Factor name]** had a [+/−X%] impact. "
        "Because this patient [specific value], [clinical reason why that value raises or lowers risk].'\n"
        "Do not use sub-headers, bullet points, or separate 'Impact:' lines within this section.\n"
        "3. **Why the Model Reached This Conclusion** — Synthesise how the combination of "
        "these specific values and their impacts produced this exact probability. Describe "
        "which factors were most decisive and why this patient's overall profile lands at "
        "this risk level. Do not give treatment recommendations.\n\n"
        "Note: This is a decision support tool to help understand the model's reasoning. "
        "Final clinical judgment rests with the treating physician."
    )

    return (
        "<start_of_turn>user\n"
        f"{content}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
