# Design: Temporal SHAP via Occlusion for Bi-LSTM Monthly Check-ins

**Date:** 2026-05-17  
**Branch:** fix/temporal-lstm-gate-calibration-platt  
**Status:** Approved

---

## Problem

The Hybrid Bi-LSTM produces a failure probability at every monthly check-in (M1–M6), but the `contributions` array stored alongside each prediction is always empty (`[]`). This means:

- `RiskUpdate` page shows "Top Risk Factors: (nothing)"
- `FeatureContribution` page shows blank bars for all temporal predictions
- Clinicians cannot see *why* risk changed — only that it did

The static model at intake already computes contributions via occlusion in `inference.ts:predictWithContributions`. This design extends the same technique to the temporal model, making contributions visible from the first monthly check-in onwards.

---

## Approach: Occlusion per Clinical Feature Group

Run 4 extra ONNX forward passes after the base inference — one per clinical group. Each pass zeros out the group's columns across all 13 timesteps in the padded temporal tensor. The probability shift (masked − base) is the contribution delta. This mirrors the exact technique already used by the static model.

**Why not Integrated Gradients or SHAP proper?**  
ONNX runtime-web does not expose gradient computation. True KernelSHAP requires ~1000 forward passes. Occlusion with 4 groups is 4 forward passes, runs in <200ms in the browser, and produces output a clinician can act on.

---

## Temporal Feature Groups

The deployed model uses 16 temporal features per timestep (ordered in `temporalFeatureNames` from the metadata JSON):

| Index | Feature name |
|---|---|
| 0 | cumulative_doses_taken |
| 1 | height |
| 2 | monthly_doses_taken |
| 3 | monthly_missed_doses |
| 4 | pct_adherence |
| 5 | smear_tb_lamp |
| 6 | weight |
| 7 | xpert_mtb_rif |
| 8 | is_missing_cumulative_doses_taken |
| 9 | is_missing_height |
| 10 | is_missing_monthly_doses_taken |
| 11 | is_missing_monthly_missed_doses |
| 12 | is_missing_pct_adherence |
| 13 | is_missing_smear_tb_lamp |
| 14 | is_missing_weight |
| 15 | is_missing_xpert_mtb_rif |

These collapse into 4 clinical groups:

| Group label | Indices zeroed | Clinical meaning |
|---|---|---|
| Adherence | 0, 2, 3, 4, 8, 10, 11, 12 | Dose-taking behaviour: monthly doses, missed doses, adherence %, cumulative + all their is_missing flags |
| Weight / Height | 1, 6, 9, 14 | Nutritional / physical status: weight, height + is_missing flags |
| Smear Result | 5, 13 | Bacteriological clearance: smear_tb_lamp + is_missing_smear_tb_lamp |
| Xpert Result | 7, 15 | Drug-resistance marker: xpert_mtb_rif + is_missing_xpert_mtb_rif |

**Direction logic** (same as static model):  
`delta = maskedProb − baseProb`  
- `delta > 0`: removing the group increases failure probability → the group was **protective**  
- `delta < 0`: removing the group decreases failure probability → the group was a **risk driver**

---

## Files Changed (5 files)

### 1. `web-app/src/lib/temporalInference.ts`

Add a constant `TEMPORAL_CONTRIBUTION_GROUPS` defining the 4 groups (indices only — no hardcoded feature names so it stays aligned with the metadata order).

Add `predictTemporalWithContributions(patient, input)`:
1. Call `predictTemporalInBrowser(patient, input)` for base result and to get `xTemporalPadded`
2. For each group, clone `xTemporalPadded`, zero the group's column indices at every timestep stride, run `sess.run(...)`, apply Platt/temperature to get masked probability
3. Compute delta; push `ContributionItem`
4. Sort by `|delta|` descending
5. Return `{ ...baseResult, contributions }`

The function must be refactored slightly so `xTemporalPadded` and `xStatic` are built once and reused across all 5 passes (base + 4 masked) to avoid redundant scaling work.

### 2. `web-app/src/lib/storage.ts`

Add `contributions?: ContributionItem[]` to `TemporalRiskSaveInput`. Import `ContributionItem` from `./inference`.

### 3. `web-app/src/pages/MonthlyCheckin.tsx`

Replace:
```typescript
const browserPrediction = await predictTemporalInBrowser(patient, temporalInput)
const pred = await saveTemporalRiskRecord(id, browserPrediction)
const result = { ..., contributions: [] }
```
With:
```typescript
const browserPrediction = await predictTemporalWithContributions(patient, temporalInput)
const pred = await saveTemporalRiskRecord(id, browserPrediction)
const result = { ..., contributions: browserPrediction.contributions }
```

### 4. `backend/schemas.py`

Add to `TemporalRiskSaveRequest`:
```python
contributions: list[ContributionItem] = Field(default_factory=list)
```
`ContributionItem` is already defined in the same file (line 12).

### 5. `backend/routers/patients.py`

Change line 115 in `save_temporal_risk_record`:
```python
contributions=[]          # before
contributions=[c.model_dump() for c in body.contributions]  # after
```

---

## What Does NOT Change

- `RiskUpdate.tsx` — already renders `result.contributions.slice(0, 3)` from navigate state
- `FeatureContribution.tsx` — already renders all `contributions` from the stored prediction record
- `inference.ts:predictWithContributions` — static model path, unchanged
- The ONNX model, metadata JSON, scalers — unchanged
- Any other page or component

---

## Error Handling

- If any masked inference pass throws, catch per-group and assign `delta = 0` for that group (so the contribution shows as zero rather than crashing the whole check-in).
- Contributions are purely informational — a failure here must never prevent the check-in from being saved.

---

## Verification

1. Submit a monthly check-in for any patient.
2. Open `RiskUpdate` — confirm "Top Risk Factors" section lists 1–3 named factors (not blank).
3. Open `FeatureContribution` (→ "See full breakdown") — confirm all 4 groups appear with coloured bars.
4. Confirm no ONNX runtime errors in DevTools console.
5. Fetch the patient from the backend (`GET /api/patients/:id`) and confirm `predictions.at(-1).contributions` is a non-empty array.
6. Existing static model intake flow: confirm intake contributions still show correctly (no regression).
