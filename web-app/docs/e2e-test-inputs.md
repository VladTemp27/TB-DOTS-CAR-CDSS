# E2E Test Inputs

Two Playwright scripts covering the SHAP temporal engine. Both require:
- Frontend: `npm run dev -- --port 5299` (from `web-app/`)
- Backend: `uvicorn backend.main:app --port 8000`

---

## 1. Smoke Test — `e2e-shap-highrisk.mjs`

Single high-risk patient created at intake (M0 only). Verifies SHAP feature bars are non-zero after the first prediction.

### Patient Profile

| Field | Value | Rationale |
|-------|-------|-----------|
| Name | E2E High Risk SHAP Test | — |
| Age | 72 | Elderly — high-risk static feature |
| Sex | M | — |
| Registration group | RELAPSE | High-risk case type |
| Province | *(first option)* | — |
| Bacteriological status | Bacteriologically-confirmed TB | — |
| Microscopy result | 3+ | High bacillary load |
| Anatomical site | Pulmonary (P) | — |
| Source | Community | — |
| Case type | DRTB | Drug-resistant TB |
| Heart rate | 115 bpm | Tachycardia |
| O₂ saturation | 88% | Hypoxic |
| Weight | 42 kg | Severely underweight |
| Height | 165 cm | — |
| Drug resistance | RR-TB | Rifampicin-resistant |
| X-ray | Skipped | — |

### What it checks

- Feature contribution bars render (non-empty DOM)
- Deltas are printed for all groups present
- Any `[SHAP-PROBE]` console logs are captured and printed

### Run

```bash
cd web-app
node scripts/e2e-shap-highrisk.mjs
```

---

## 2. Longitudinal Acceptance Test — `e2e-longitudinal.mjs`

Four patients, each receiving an intake (M0) and three monthly check-ins (M1–M3). Verifies that risk changes over time and that SHAP produces a meaningful top contribution after the final check-in.

### Pass criteria (per patient)

| Check | Threshold |
|-------|-----------|
| Risk changed M0 → M3 | Must differ (integer %) |
| Top SHAP contribution after M3 | ≥ 5% |

### Patient A — Good adherence, improving

**Intake**

| Field | Value |
|-------|-------|
| Name | Longitudinal A Good |
| Age | 34 |
| Sex | F |
| Registration group | NEW |
| Bacteriological status | Bacteriologically-confirmed TB |
| Microscopy result | 2+ |
| Anatomical site | P |
| Case type | DSTB |
| Heart rate | 78 bpm |
| BP | 118/76 mmHg |
| O₂ saturation | 97% |
| Weight | 54 kg |
| Height | 160 cm |
| Drug resistance | DS-TB |

**Monthly check-ins**

| Month | Weight | Doses taken | Doses missed | Smear |
|-------|--------|-------------|--------------|-------|
| M1 | 54.5 kg | 28 | 0 | Positive (1) |
| M2 | 55.2 kg | 28 | 0 | Positive (1) |
| M3 | 55.8 kg | 28 | 0 | Positive (1) |

**Expected SHAP pattern:** Treatment Adherence strongly protective (−40–50%); other groups collapse to ~0% because perfect adherence dominates the LSTM signal.

---

### Patient B — Poor adherence, declining

**Intake**

| Field | Value |
|-------|-------|
| Name | Longitudinal B Poor |
| Age | 48 |
| Sex | M |
| Registration group | RELAPSE |
| Bacteriological status | Bacteriologically-confirmed TB |
| Microscopy result | 3+ |
| Anatomical site | P |
| Case type | DRTB |
| Heart rate | 92 bpm |
| BP | 130/84 mmHg |
| O₂ saturation | 94% |
| Weight | 50 kg |
| Height | 165 cm |
| Drug resistance | RR-TB |

**Monthly check-ins**

| Month | Weight | Doses taken | Doses missed | Smear |
|-------|--------|-------------|--------------|-------|
| M1 | 49 kg | 10 | 18 | Positive (1) |
| M2 | 48 kg | 8 | 20 | Positive (1) |
| M3 | 47 kg | 6 | 22 | Positive (1) |

**Expected SHAP pattern:** Treatment Adherence strongly risk-increasing (+45–50%); multiple other groups (Smear, Vitals) also fire because poor adherence leaves the model uncertain across dimensions.

---

### Patient C — Mixed adherence, fluctuating

**Intake**

| Field | Value |
|-------|-------|
| Name | Longitudinal C Mixed |
| Age | 28 |
| Sex | F |
| Registration group | NEW |
| Bacteriological status | Bacteriologically-confirmed TB |
| Microscopy result | 1+ |
| Anatomical site | P |
| Case type | DSTB |
| Heart rate | 72 bpm |
| BP | 112/72 mmHg |
| O₂ saturation | 98% |
| Weight | 48 kg |
| Height | 155 cm |
| Drug resistance | DS-TB |

**Monthly check-ins**

| Month | Weight | Doses taken | Doses missed | Smear |
|-------|--------|-------------|--------------|-------|
| M1 | 48.5 kg | 27 | 1 | Positive (1) |
| M2 | 47.8 kg | 12 | 16 | Negative (0) |
| M3 | 48.2 kg | 26 | 2 | Positive (1) |

**Expected SHAP pattern:** Treatment Adherence moderately protective (−25–30%); static groups (Body Height, Vital Signs, Body Weight) more visible than in PA/PD because the mixed adherence history leaves the model sensitive to baseline features.

---

### Patient D — High-risk baseline, improving

**Intake**

| Field | Value |
|-------|-------|
| Name | Longitudinal D HighRisk |
| Age | 65 |
| Sex | M |
| Registration group | RELAPSE |
| Bacteriological status | Bacteriologically-confirmed TB |
| Microscopy result | 3+ |
| Anatomical site | P |
| Case type | DRTB |
| Heart rate | 108 bpm |
| BP | 100/65 mmHg |
| O₂ saturation | 90% |
| Weight | 42 kg |
| Height | 162 cm |
| Drug resistance | MDR-TB |

**Monthly check-ins**

| Month | Weight | Doses taken | Doses missed | Smear |
|-------|--------|-------------|--------------|-------|
| M1 | 42.5 kg | 22 | 6 | Positive (1) |
| M2 | 43.2 kg | 25 | 3 | Positive (1) |
| M3 | 44.0 kg | 26 | 2 | Negative (0) |

**Expected SHAP pattern:** Treatment Adherence protective (−15–20%); high-risk static profile (age, MDR-TB, low O₂) keeps the base failure probability elevated even with improving adherence, so the adherence delta is smaller than in PA.

---

### Run

```bash
cd web-app
PORT=5299 node scripts/e2e-longitudinal.mjs
```

Expected final output:

```
══════════════════════════════════════════════════════════════════
  Overall: ✓ ALL PASSED
══════════════════════════════════════════════════════════════════
```
