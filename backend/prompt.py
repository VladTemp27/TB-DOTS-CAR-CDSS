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

    # Map feature names to their actual patient values for inline annotation
    value_map = {
        "Age": f"{req.age} years",
        "Days to Treatment": f"{req.days_to_treatment} days",
        "Sex": sex_full,
        "Anatomical Site": anatomical_site_full,
        "Registration Group": req.registration_group,
        "Bacteriologic Status": req.bacteriologic_status,
        "Microscopy Result": req.microscopy_result,
        "Source of Patient": req.source_of_patient,
        "Type": req.type,
    }

    # Build contributions text from top 5
    # delta > 0 means the feature pushed toward failure; delta < 0 pushed toward success
    top_contributions = req.contributions[:5]
    contributions_lines = []
    for c in top_contributions:
        value_note = f" (patient value: {value_map[c.feature]})" if c.feature in value_map else ""
        direction = "increases" if c.delta > 0 else "decreases"
        contributions_lines.append(
            f"- {c.feature}{value_note}: {c.delta * 100:+.1f}% impact ({direction} failure risk)"
        )
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

    # Gemma 3 chat template format
    prompt = (
        "<start_of_turn>user\n"
        f"{content}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    return prompt
