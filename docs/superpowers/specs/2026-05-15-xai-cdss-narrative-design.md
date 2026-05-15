# XAI CDSS Narrative Design

Date: 2026-05-15

## Purpose

Strengthen the thesis narrative in `paper/apa/thesis_apa.tex` so explainable AI (XAI) is presented as necessary for clinical adoption of the TB treatment-failure Clinical Decision Support System (CDSS), not as an optional technical add-on. The primary emphasis is clinician trust and adoption. The secondary emphasis is safety and governance through bounded explanation behavior.

## Narrative Thesis

The paper should argue that tuberculosis treatment-failure prediction cannot rely on risk scores alone. A CDSS used by TB-DOTS staff must help clinicians understand why a patient is flagged before they can reasonably use the alert for review, prioritization, or intervention planning. Machine learning supplies the risk estimate; SHAP supplies the model-grounded explanation; the small language model (SLM) improves readability; and the clinician retains final decision authority.

## Approved Narrative Architecture

The revised paper should move through this logic:

1. TB treatment failure is clinically serious and operationally difficult to anticipate.
2. Machine learning can detect complex nonlinear risk patterns better than fragmented single-factor analysis.
3. Prediction alone is insufficient in clinical decision support because clinicians need to understand the basis of a risk alert before acting on it.
4. SHAP provides the primary explanation layer by identifying which patient, diagnostic, treatment, or temporal factors contributed to a model prediction.
5. The SLM provides a secondary communication layer that translates SHAP-grounded outputs into readable clinical text, but it does not generate independent clinical reasoning.
6. Faithfulness testing and safety limits are necessary because readable explanations are useful only if they remain grounded in SHAP and do not overstate causality or replace clinician judgment.

## Scope

This is a narrative-throughout revision. The edit should touch the front-to-back argument where needed, not only one XAI paragraph.

Target sections:

1. Abstract
2. Machine Learning for TB Outcome Prediction
3. Contributions
4. Research Objectives
5. Identification of Variables Associated With Treatment Outcomes
6. Development of a Clinical Decision Support Tool
7. SLM-Based Narrative Explanation Layer
8. Results and Discussion
9. Recommendations
10. Safety Scope and Interpretation Limits

## Objective Handling

Objective 3 should remain broad. It should continue to focus on identifying contributing variables that influence treatment outcomes over the course of therapy. It should not be rewritten as a SHAP-only objective.

Later sections should explain that SHAP and the SLM operationalize Objective 3 inside the predictive framework and CDSS by making contributing variables interpretable, clinically reviewable, and usable in the interface.

Objective 6 should be strengthened to include explainable risk visualization and clinician-readable explanation outputs, not just generic predictive outputs.

## Section-Level Design

### Abstract

Revise the final interpretability and CDSS sentences so SHAP and the SLM support clinician trust and safe use. The abstract should make clear that explanations are tied to individual predictions and are intended to support clinical review, not autonomous treatment decisions.

### Machine Learning for TB Outcome Prediction

Strengthen the interpretability paragraph so it argues that clinical prediction models need patient-level explanations before they can support decision-making. The section should transition from ML performance to clinical usability: high-performing black-box predictions are not enough for TB-DOTS workflows if staff cannot inspect why a patient was flagged.

### Contributions

Revise the fifth contribution into an explainable CDSS contribution. It should include dynamic risk estimates, SHAP-based factor explanations, bounded SLM narrative summaries, and clinician oversight.

### Research Objectives

Keep Objective 3 broad. Revise Objective 6 to state that the decision support tool visualizes risk levels, predictive outputs, and explanation outputs that support clinician interpretation.

### Identification of Variables Associated With Treatment Outcomes

Use this as the methodological home for Objective 3. The subsection should connect variable identification to both global and patient-level interpretability. It should clarify that SHAP explains model behavior by ranking contributors to predictions, while avoiding causal claims.

### Development of a Clinical Decision Support Tool

Clarify the division of responsibilities: the model predicts risk, SHAP explains model contributions, the SLM verbalizes explanations, and clinicians decide. The CDSS should be framed as supporting review and prioritization rather than recommending treatment or diagnosis.

### SLM-Based Narrative Explanation Layer

Make the SLM explicitly secondary to SHAP. It should be described as a constrained readability layer that verbalizes ranked contribution signals. It must not infer new causes, prescribe actions, alter model probabilities, or override SHAP.

### Results and Discussion

Interpret feature importance and SHAP outputs as explanation evidence for clinical review, not only model diagnostics. The discussion should connect explanation outputs to trust, auditability, and workflow adoption.

### Recommendations

Recommend explainability and faithfulness evaluation as prerequisites for clinical deployment. Emphasize that any deployment should include external validation, prospective testing, clinician review, calibration assessment, and SLM faithfulness evaluation.

### Safety Scope and Interpretation Limits

Preserve and strengthen the existing safety boundary: explanations describe how features influenced model predictions, not why disease progression occurred. The CDSS supports clinician judgment and does not replace it.

## Explanation Responsibility Model

| Layer | Role | Boundary |
|---|---|---|
| Predictive model | Estimates treatment failure risk from static and temporal patient data | Does not explain itself directly |
| SHAP/XAI layer | Identifies feature-level contributors to each prediction | Explains model behavior, not clinical causality |
| SLM narrative layer | Converts SHAP-grounded signals into readable clinical summaries | Cannot invent causes, prescribe treatment, or override SHAP |
| Clinician | Interprets risk and explanation in context | Retains final decision authority |

## Revision Rules

1. Keep SHAP as an explanation of model behavior, not proof of clinical causation.
2. Keep Objective 3 broad, then connect it later to SHAP and the SLM as the method for surfacing and communicating contributing variables.
3. Avoid saying the CDSS recommends treatment or diagnoses patients.
4. Prefer language such as supports review, flags risk, surfaces contributing factors, and supports clinician judgment.
5. Keep the SLM secondary to SHAP: it improves readability but cannot invent factors or change model outputs.
6. Keep the narrative consistent across Abstract, Objectives, Method, Results, Discussion, Recommendations, and Safety Scope.

## Success Criteria

1. A reader understands why XAI is necessary for clinical adoption, not optional decoration.
2. The paper clearly separates prediction, explanation, narration, and clinical judgment.
3. Objective 3 remains defensible as contributing-variable identification, with SHAP and the SLM introduced later as operational mechanisms.
4. The final language strengthens the paper without making unsupported deployment, diagnostic, treatment, or causality claims.

## Out of Scope

This design does not include implementation edits to `thesis_apa.tex`. Those edits require a separate implementation plan after the written spec is reviewed and approved.
