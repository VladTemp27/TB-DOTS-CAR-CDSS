def build_prompt(req) -> str:
    """Build the Gemma 3 chat template string from the request data."""

    # Expand sex code
    sex_full = "Male" if req.sex == "M" else "Female"

    # Expand anatomical site code
    anatomical_site_full = (
        "PTB (Pulmonary)" if req.anatomical_site == "P" else "EPTB (Extra-pulmonary)"
    )  # req.anatomical_site is Literal["P", "EP"]

    # Failure percentage
    failure_pct = req.failure_probability * 100

    # Build contributions text from top 5
    top_contributions = req.contributions[:5]
    contributions_lines = [
        f"- {c.feature}: {c.delta * 100:+.1f}% ({c.direction})"
        for c in top_contributions
    ]
    contributions_text = "\n".join(contributions_lines)

    content = (
        "You are a clinical decision support assistant for TB treatment in the Philippines "
        "Cordillera Administrative Region (CAR). You interpret ML model predictions for "
        "healthcare workers using DOTS guidelines.\n\n"
        f"Patient profile:\n"
        f"- Name: {req.patient_name}\n"
        f"- Age: {req.age} years | Sex: {sex_full}\n"
        f"- Bacteriologic Status: {req.bacteriologic_status}\n"
        f"- Microscopy Result: {req.microscopy_result}\n"
        f"- Anatomical Site: {anatomical_site_full}\n"
        f"- Registration Group: {req.registration_group}\n"
        f"- Source of Patient: {req.source_of_patient}\n"
        f"- TB Type: {req.type}\n"
        f"- Days from Diagnosis to Treatment Start: {req.days_to_treatment}\n\n"
        f"ML model prediction: {failure_pct:.0f}% treatment failure probability\n\n"
        f"Top contributing factors:\n"
        f"{contributions_text}\n\n"
        "Provide a structured clinical interpretation:\n"
        "1. **Risk Assessment Summary** (1-2 sentences interpreting the probability)\n"
        "2. **Key Risk Factors** (explain top 3-4 contributors in clinical terms)\n"
        "3. **Clinical Considerations** (actionable steps aligned with Philippine NTP guidelines)\n\n"
        "Note: This is a decision support tool. Final clinical judgment rests with the treating physician."
    )

    # Gemma 3 chat template format
    prompt = (
        "<start_of_turn>user\n"
        f"{content}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    return prompt
