# SLM Thesis Placement Design (APA)

## 1) Objective

Update the APA thesis to accurately document the already-implemented SLM explanation layer in the TB CDSS web application, with emphasis on methodological clarity, safety boundaries, and appropriate scholarly framing. The scope is implementation-only (no new model features, no claimed evaluation results beyond current implementation behavior).

## 2) Scope and Boundaries

### In Scope

- Add SLM-focused text in both Method and Discussion sections.
- Position SLM as a constrained natural-language explanation layer over model outputs and feature contributions.
- Describe implemented data flow and operational controls reflected in the codebase.
- Add/adjust citations that are appropriate for SHAP-grounded explanation framing, medical SLM context, and AI-in-health governance/reporting.

### Out of Scope

- Any code or architecture changes to the CDSS, frontend, backend, or model pipeline.
- New experiments, new benchmark claims, or retrospective performance inflation.
- Claims that the SLM performs diagnosis, causal inference, or treatment recommendation.

## 3) Current Implementation Context (Code-Verified)

The current SLM explanation flow is already implemented as follows:

- Frontend request composition: `web-app/src/hooks/useStreamingInterpretation.ts`
  - Builds a constrained interpretation payload from patient context, failure probability, and contribution list.
  - Omits unrelated fields by design.
  - Includes retry behavior and StrictMode-aware debounce/abort handling.
- Frontend transport: `web-app/src/lib/medgemma.ts`
  - Sends `POST /api/interpret` and consumes SSE token stream.
  - Handles stream completion/error events and abort conditions.
- Backend endpoint and inference lifecycle: `backend/main.py`
  - Validates request schema via Pydantic.
  - Builds prompt and streams output tokens using EventSourceResponse.
  - Uses lock-guarded inference path and runtime stats (`requests_active`, `requests_served`, timing/tokens).
- Prompt framing: `backend/prompt.py`
  - Creates structured interpretation prompt from selected fields and top contributions.
  - Frames output as decision support and explicitly places final judgment with clinicians.
- UI rendering: `web-app/src/components/ClinicalInterpretation.tsx`
  - Displays streamed narrative interpretation in clinician-facing view.
  - Embedded in result pages (`DiagnosticResult`, `FeatureContribution`).

## 4) Chosen Placement Strategy

Chosen approach: **Method + Discussion with dedicated standalone subsections** for review clarity under APA heading hierarchy.

### Method Placement

- Add a new Level-2 subsection after `Development of a Clinical Decision Support Tool`:
  - **Proposed title:** `SLM-Based Narrative Explanation Layer`

### Discussion Placement

- Add a new Level-2 subsection in Discussion focused on boundaries and safety:
  - **Proposed title:** `Safety Scope and Interpretation Limits`

Rationale: this preserves readability and examiner traceability while remaining APA-consistent and not over-fragmenting the manuscript.

## 5) Content Design by Section

### 5.1 Method Subsection Content

The subsection will:

- State that SHAP/feature-contribution outputs remain the primary explanation source.
- Define the SLM as a constrained verbalization layer that translates structured outputs into readable narrative text.
- Describe implementation-accurate input constraints: selected patient context fields, failure probability, and top contributions.
- Describe request/response mechanism at a high level: API interpretation endpoint with streamed narrative output.
- Explicitly prohibit over-claims: no independent diagnosis, no causal inference, no treatment recommendation generation.
- Close with decision-support framing: clinician remains final decision-maker.

### 5.2 Discussion Subsection Content

The subsection will:

- Explain practical value: improved interpretability usability for healthcare users.
- Discuss key risk class: generative explanation drift/hallucination risk.
- Tie risk controls to implemented boundaries (constrained payload, prompt role-bounding, explicit authority disclaimers).
- Use non-causal wording for interpretation claims (e.g., "increased/decreased model-predicted risk").
- Avoid claiming completed clinical efficacy validation of SLM narratives.

## 6) Citation Plan

References for insertion should prioritize peer-reviewed/foundational and standards-aligned sources from the project guidance, with emphasis on:

- SHAP foundation and explainability grounding.
- LLM/SLM-assisted explanation translation relevance.
- Medical AI governance/safety/reporting guidance.

Non-academic media examples are excluded as primary evidence in the thesis body.

## 7) Writing Constraints for Final Edits

- Preserve APA-consistent heading usage and tone.
- Keep both new subsections concise and focused.
- Ensure terminology consistency:
  - "model-predicted risk"
  - "decision support"
  - "clinician judgment" / "final clinical judgment"
- Ensure no contradiction with existing Method claims and no scope expansion beyond implemented behavior.

## 8) Acceptance Criteria

The design is considered complete when:

- Two dedicated SLM subsections are integrated (Method and Discussion).
- Method text reflects code-verified implementation behavior without adding unimplemented capabilities.
- Discussion text clearly states safety scope and non-causal interpretation boundaries.
- Citation updates are academically appropriate and aligned with the guidance document.
- The manuscript does not claim independent clinical reasoning by the SLM.

## 9) Risks and Mitigations

- Risk: overclaiming SLM clinical intelligence.
  - Mitigation: explicit role boundary language and non-causal phrasing.
- Risk: subsection bloat in APA structure.
  - Mitigation: concise standalone subsections with strict scope.
- Risk: mismatch between manuscript claims and code behavior.
  - Mitigation: claims anchored to verified files and request/response flow.

## 10) Implementation Transition Note

After user review/approval of this design document, the next step is to invoke `writing-plans` to produce a concrete, ordered editing plan for `paper/apa/thesis_apa.tex` and citation updates.
