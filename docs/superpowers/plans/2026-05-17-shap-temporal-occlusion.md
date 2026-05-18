# Temporal SHAP Occlusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute and persist feature contribution scores for the Hybrid Bi-LSTM monthly check-in predictions so that `RiskUpdate` and `FeatureContribution` pages display "Top Risk Factors" instead of blank bars.

**Architecture:** After the base temporal inference pass, run 4 additional masked ONNX forward passes (one per clinical feature group). Zeroing a group's columns across all timesteps and measuring the probability shift yields a `ContributionItem[]` compatible with the existing display components. Contributions flow through the same save path as the prediction itself: frontend → `saveTemporalRiskRecord` → backend → stored `PredictionRow`.

**Tech Stack:** TypeScript (onnxruntime-web 1.25.1), Python/FastAPI (Pydantic v2), SQLite via SQLAlchemy.

---

## File Map

| File | Change |
|---|---|
| `web-app/src/lib/temporalInference.ts` | Extract 3 internal helpers; add `TEMPORAL_CONTRIBUTION_GROUPS`; add exported `predictTemporalWithContributions` |
| `web-app/src/lib/storage.ts` | Add `contributions?: ContributionItem[]` to `TemporalRiskSaveInput` |
| `web-app/src/pages/MonthlyCheckin.tsx` | Switch call site from `predictTemporalInBrowser` to `predictTemporalWithContributions`; pass contributions through |
| `backend/schemas.py` | Add `contributions: list[ContributionItem]` field to `TemporalRiskSaveRequest` |
| `backend/routers/patients.py` | Replace hardcoded `contributions=[]` with `body.contributions` |

---

## Task 1: Add `contributions` to the backend schema and save endpoint

**Files:**
- Modify: `backend/schemas.py` (around line 164)
- Modify: `backend/routers/patients.py` (line 115)

- [ ] **Step 1.1: Add field to `TemporalRiskSaveRequest`**

Open `backend/schemas.py`. Find `class TemporalRiskSaveRequest(TemporalRiskRequest):` (line 164). Add one field at the end of the class — `ContributionItem` is already defined at line 12 in the same file:

```python
class TemporalRiskSaveRequest(TemporalRiskRequest):
    label: Literal[0, 1]
    failure_probability: float = Field(alias="failureProbability")
    success_probability: float = Field(alias="successProbability")
    raw_failure_probability: float | None = Field(default=None, alias="rawFailureProbability")
    raw_success_probability: float | None = Field(default=None, alias="rawSuccessProbability")
    rule_failure_floor: float | None = Field(default=None, alias="ruleFailureFloor")
    previous_failure_probability: float | None = Field(default=None, alias="previousFailureProbability")
    adjusted_failure_probability: float | None = Field(default=None, alias="adjustedFailureProbability")
    threshold: float | None = None
    model_name: str = Field(default="hybrid_bi_lstm_best_onnx", alias="modelName")
    months_used: list[int] | None = Field(default=None, alias="monthsUsed")
    seq_len: int | None = Field(default=None, alias="seqLen")
    risk_policy: str | None = Field(default=None, alias="riskPolicy")
    contributions: list[ContributionItem] = Field(default_factory=list)
```

- [ ] **Step 1.2: Wire contributions into the save endpoint**

Open `backend/routers/patients.py`. Find the `db.add(PredictionRow(...))` block inside `save_temporal_risk_record` (around line 110). Change line 115 from `contributions=[]` to use the request body:

```python
    db.add(
        PredictionRow(
            patient_id=patient_id,
            label=int(body.label),
            failure_probability=prob_failure,
            contributions=[c.model_dump() for c in body.contributions],  # ← was []
            features_used={
```

- [ ] **Step 1.3: Verify the backend accepts contributions**

Start the backend (`uvicorn backend.main:app --reload` from the project root or however it's normally started).

Run this one-liner to confirm the new field is accepted (replace `<patient_id>` with any existing patient):

```bash
curl -s -X POST http://localhost:8000/api/patients/<patient_id>/temporal-risk-record \
  -H "Content-Type: application/json" \
  -d '{
    "month": 99,
    "label": 0,
    "failureProbability": 0.1,
    "successProbability": 0.9,
    "threshold": 0.05,
    "modelName": "test",
    "contributions": [{"feature":"Adherence","delta":0.05,"direction":"risk"}]
  }'
```

Expected: HTTP 400 (`month must be in [0, 12]`) or HTTP 409 — NOT a 422 Unprocessable Entity. A 422 means the schema rejected the shape; any other error means the field was accepted and the endpoint ran business logic.

- [ ] **Step 1.4: Commit**

```bash
git add backend/schemas.py backend/routers/patients.py
git commit -m "feat(backend): accept contributions in temporal-risk-record endpoint"
```

---

## Task 2: Add `contributions` to the TypeScript save input type

**Files:**
- Modify: `web-app/src/lib/storage.ts` (line 1 import + line 125 type)

- [ ] **Step 2.1: Extend the import and the type**

Open `web-app/src/lib/storage.ts`. Line 1 currently reads:

```typescript
import type { PatientFeatures, ContributionResult } from './inference'
```

Change it to also import `ContributionItem`:

```typescript
import type { PatientFeatures, ContributionResult, ContributionItem } from './inference'
```

Then find `TemporalRiskSaveInput` (around line 125) and add the optional field:

```typescript
export type TemporalRiskSaveInput = TemporalRiskInput & TemporalRiskResult & {
  rawFailureProbability?: number
  rawSuccessProbability?: number
  ruleFailureFloor?: number
  previousFailureProbability?: number
  adjustedFailureProbability?: number
  seqLen?: number
  riskPolicy?: string
  contributions?: ContributionItem[]   // ← add this line
}
```

- [ ] **Step 2.2: Verify TypeScript compiles**

```bash
cd web-app && npx tsc --noEmit
```

Expected: zero errors. If you see "Module has no exported member 'ContributionItem'" — check that `inference.ts` exports it (`export interface ContributionItem`; it does at line 28).

- [ ] **Step 2.3: Commit**

```bash
git add web-app/src/lib/storage.ts
git commit -m "feat(types): add contributions field to TemporalRiskSaveInput"
```

---

## Task 3: Extract helpers and add `predictTemporalWithContributions`

**Files:**
- Modify: `web-app/src/lib/temporalInference.ts`

This is the core task. We extract three private helpers from the existing `predictTemporalInBrowser` body so they can be reused by the new contribution function, then add the feature group constant and the new exported function.

- [ ] **Step 3.1: Add the import for `ContributionItem`**

Open `web-app/src/lib/temporalInference.ts`. Line 2 currently reads:

```typescript
import type { Patient, MonthlyRecord, TemporalRiskInput, TemporalRiskSaveInput } from './storage'
```

Change it to:

```typescript
import type { Patient, MonthlyRecord, TemporalRiskInput, TemporalRiskSaveInput } from './storage'
import type { ContributionItem } from './inference'
```

- [ ] **Step 3.2: Add the feature group constant**

After the `sigmoid` helper (around line 47), add:

```typescript
const TEMPORAL_CONTRIBUTION_GROUPS: { name: string; cols: readonly number[] }[] = [
  { name: 'Adherence',       cols: [0, 2, 3, 4, 8, 10, 11, 12] },
  { name: 'Weight / Height', cols: [1, 6, 9, 14] },
  { name: 'Smear Result',    cols: [5, 13] },
  { name: 'Xpert Result',    cols: [7, 15] },
]
```

Column indices refer to the `temporalFeatureNames` order in the metadata JSON:
```
0=cumulative_doses_taken  1=height  2=monthly_doses_taken  3=monthly_missed_doses
4=pct_adherence  5=smear_tb_lamp  6=weight  7=xpert_mtb_rif
8=is_missing_cumulative_doses_taken  9=is_missing_height  10=is_missing_monthly_doses_taken
11=is_missing_monthly_missed_doses  12=is_missing_pct_adherence  13=is_missing_smear_tb_lamp
14=is_missing_weight  15=is_missing_xpert_mtb_rif
```

- [ ] **Step 3.3: Extract `_buildTemporalTensors`**

Insert this private helper immediately before the `predictTemporalInBrowser` function (around line 301). It consolidates the tensor-building logic that was previously inlined:

```typescript
function _buildTemporalTensors(patient: Patient, input: TemporalRiskInput, meta: Metadata) {
  const priorCumulative = patient.monthlyRecords.reduce(
    (sum, r) => sum + (r.monthlyDosesTaken ?? 0), 0
  )
  const record = currentRecord(input, priorCumulative)
  const records = [...patient.monthlyRecords, record]
  const monthsUsed = [...new Set(records.map(r => r.month))].sort((a, b) => a - b)
  const seqLen = input.month + 1
  const clampedSeqLen = Math.min(TOTAL_MONTHS, Math.max(1, seqLen))
  const temporalFeatureCount = meta.temporalFeatureNames.length

  const xStatic = scale(
    buildStaticVector(patient, meta.staticFeatureNames),
    meta.staticScaler.mean,
    meta.staticScaler.scale,
  )
  const xTemporalKnown = scale(
    buildTemporalMatrix(records, meta.temporalFeatureNames, input.month),
    meta.temporalScaler.mean,
    meta.temporalScaler.scale,
  )
  const xTemporalPadded = new Float32Array(TOTAL_MONTHS * temporalFeatureCount)
  xTemporalPadded.set(xTemporalKnown)
  const seqLens = new BigInt64Array([BigInt(clampedSeqLen)])

  return { xStatic, xTemporalPadded, seqLens, temporalFeatureCount, monthsUsed, seqLen }
}
```

- [ ] **Step 3.4: Extract `_runLogit`**

Insert immediately after `_buildTemporalTensors`:

```typescript
async function _runLogit(
  sess: ort.InferenceSession,
  xTemporalPadded: Float32Array,
  xStatic: Float32Array,
  seqLens: BigInt64Array,
  temporalFeatureCount: number,
): Promise<number> {
  const result = await sess.run({
    x_temporal: new ort.Tensor('float32', xTemporalPadded, [1, TOTAL_MONTHS, temporalFeatureCount]),
    x_static:   new ort.Tensor('float32', xStatic,         [1, xStatic.length]),
    seq_lens:   new ort.Tensor('int64',   seqLens,          [1]),
  })
  return Number((result.logit.data as Float32Array | number[])[0])
}
```

- [ ] **Step 3.5: Extract `_applyCalibration`**

Insert immediately after `_runLogit`:

```typescript
function _applyCalibration(logit: number, meta: Metadata): number {
  return meta.platt
    ? sigmoid(meta.platt.a * logit + meta.platt.b)
    : sigmoid(logit / (
        meta.temperature && Number.isFinite(meta.temperature) && meta.temperature > 1e-6
          ? meta.temperature
          : 1
      ))
}
```

- [ ] **Step 3.6: Refactor `predictTemporalInBrowser` to use the helpers**

Replace the entire body of `predictTemporalInBrowser` with the helper-based version. The function signature stays identical — only the body changes:

```typescript
export async function predictTemporalInBrowser(
  patient: Patient,
  input: TemporalRiskInput,
): Promise<TemporalRiskSaveInput> {
  const meta = await getMetadata()
  const sess = await getSession()
  const { xStatic, xTemporalPadded, seqLens, temporalFeatureCount, monthsUsed, seqLen } =
    _buildTemporalTensors(patient, input, meta)

  const logit      = await _runLogit(sess, xTemporalPadded, xStatic, seqLens, temporalFeatureCount)
  const rawFailure = _applyCalibration(logit, meta)
  const rawSuccess = 1 - rawFailure
  const threshold  = meta.threshold
  const label: 0 | 1 = rawFailure >= threshold ? 1 : 0
  const modelName  = meta.modelName ?? meta.modelType ?? 'hybrid_lstm_temporal_balanced_failure_risk'

  return {
    ...input,
    label,
    failureProbability:    rawFailure,
    successProbability:    rawSuccess,
    rawFailureProbability: rawFailure,
    rawSuccessProbability: rawSuccess,
    threshold,
    modelName,
    monthsUsed,
    seqLen,
    riskPolicy: 'browser_onnx_balanced_failure_risk_v2',
  }
}
```

- [ ] **Step 3.7: Add `predictTemporalWithContributions`**

Append this new exported function at the end of the file:

```typescript
export async function predictTemporalWithContributions(
  patient: Patient,
  input: TemporalRiskInput,
): Promise<TemporalRiskSaveInput & { contributions: ContributionItem[] }> {
  const meta = await getMetadata()
  const sess = await getSession()
  const { xStatic, xTemporalPadded, seqLens, temporalFeatureCount, monthsUsed, seqLen } =
    _buildTemporalTensors(patient, input, meta)

  const baseLogit = await _runLogit(sess, xTemporalPadded, xStatic, seqLens, temporalFeatureCount)
  const baseProb  = _applyCalibration(baseLogit, meta)

  const contributions: ContributionItem[] = []
  for (const group of TEMPORAL_CONTRIBUTION_GROUPS) {
    try {
      const masked = new Float32Array(xTemporalPadded)
      for (let t = 0; t < TOTAL_MONTHS; t++) {
        for (const col of group.cols) {
          masked[t * temporalFeatureCount + col] = 0
        }
      }
      const maskedLogit = await _runLogit(sess, masked, xStatic, seqLens, temporalFeatureCount)
      const maskedProb  = _applyCalibration(maskedLogit, meta)
      const delta       = maskedProb - baseProb
      contributions.push({
        feature:   group.name,
        delta:     Math.abs(delta),
        direction: delta > 0 ? 'protective' : 'risk',
      })
    } catch {
      // A failed masked pass must not prevent the check-in from saving.
      contributions.push({ feature: group.name, delta: 0, direction: 'risk' })
    }
  }
  contributions.sort((a, b) => b.delta - a.delta)

  const threshold = meta.threshold
  const label: 0 | 1 = baseProb >= threshold ? 1 : 0
  const modelName = meta.modelName ?? meta.modelType ?? 'hybrid_lstm_temporal_balanced_failure_risk'

  return {
    ...input,
    label,
    failureProbability:    baseProb,
    successProbability:    1 - baseProb,
    rawFailureProbability: baseProb,
    rawSuccessProbability: 1 - baseProb,
    threshold,
    modelName,
    monthsUsed,
    seqLen,
    riskPolicy: 'browser_onnx_balanced_failure_risk_v2',
    contributions,
  }
}
```

- [ ] **Step 3.8: Verify TypeScript compiles**

```bash
cd web-app && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 3.9: Commit**

```bash
git add web-app/src/lib/temporalInference.ts
git commit -m "feat(inference): add predictTemporalWithContributions via occlusion"
```

---

## Task 4: Wire the new function into `MonthlyCheckin.tsx`

**Files:**
- Modify: `web-app/src/pages/MonthlyCheckin.tsx`

- [ ] **Step 4.1: Update the import**

Find the import line that pulls from `../lib/temporalInference`. It currently imports `predictTemporalInBrowser`. Add `predictTemporalWithContributions`:

```typescript
import { predictTemporalInBrowser, predictTemporalWithContributions } from '../lib/temporalInference'
```

- [ ] **Step 4.2: Replace the call site**

Find the block inside `handleSubmit` that calls `predictTemporalInBrowser` (around line 161). Replace it so contributions flow through:

**Before:**
```typescript
const browserPrediction = await predictTemporalInBrowser(patient, temporalInput)
const pred = await saveTemporalRiskRecord(id, browserPrediction)

const result = {
  label: pred.label,
  failureProbability: pred.failureProbability,
  successProbability: pred.successProbability,
  contributions: [],
}
```

**After:**
```typescript
const browserPrediction = await predictTemporalWithContributions(patient, temporalInput)
const pred = await saveTemporalRiskRecord(id, browserPrediction)

const result = {
  label: pred.label,
  failureProbability: pred.failureProbability,
  successProbability: pred.successProbability,
  contributions: browserPrediction.contributions,
}
```

`predictTemporalInBrowser` is no longer called from this file, but keep the import in case other call sites exist — or remove it if unused (TypeScript will warn with `--noUnusedLocals`).

- [ ] **Step 4.3: Verify TypeScript compiles**

```bash
cd web-app && npx tsc --noEmit
```

Expected: zero errors. If you see "unused import" for `predictTemporalInBrowser`, remove it from the import line.

- [ ] **Step 4.4: Commit**

```bash
git add web-app/src/pages/MonthlyCheckin.tsx
git commit -m "feat(ui): use predictTemporalWithContributions in MonthlyCheckin"
```

---

## Task 5: End-to-end verification

No code changes — this is a manual browser test using the existing Playwright scripts.

- [ ] **Step 5.1: Start both servers**

```bash
# Terminal 1 — backend
uvicorn backend.main:app --reload

# Terminal 2 — frontend
cd web-app && npm run dev
```

- [ ] **Step 5.2: Run the live inference smoke test**

```bash
cd web-app && node playwright-live-inference-test.mjs
```

Expected output: `Results: 5 passed, 0 failed` (same as before — regression check).

- [ ] **Step 5.3: Run the monthly check-in test (3 profiles, 3 months each)**

```bash
cd web-app && node playwright-monthly-checkin-test.mjs
```

Expected: table shows 9 rows with non-null `P(failure)` values.

- [ ] **Step 5.4: Manually verify contributions appear on RiskUpdate**

After the test completes, open the browser at `http://localhost:5175`. Pick any patient that just had a check-in. Navigate to their profile → click the latest month → confirm the "Top Risk Factors" section on `RiskUpdate` lists at least one factor (not blank).

- [ ] **Step 5.5: Manually verify contributions appear on FeatureContribution**

From `RiskUpdate`, click "See full breakdown →". Confirm the `FeatureContribution` page shows coloured bars for the 4 groups (Adherence, Weight / Height, Smear Result, Xpert Result).

- [ ] **Step 5.6: Verify contributions are persisted in the backend**

```bash
curl -s http://localhost:8000/api/patients/<any_patient_id> | python -m json.tool | grep -A 20 '"contributions"'
```

Expected: at least one prediction record with `contributions` as a non-empty array, e.g.:
```json
"contributions": [
  {"feature": "Adherence", "delta": 0.12, "direction": "risk"},
  ...
]
```

- [ ] **Step 5.7: Verify static model intake contributions still work (regression)**

Add a new test patient through the intake form. After clicking "Generate Diagnosis →", confirm the `DiagnosticResult` page shows feature bars (the static model contribution — 14 features). This must still work.

- [ ] **Step 5.8: Final commit if any cleanup was done**

```bash
git add -A
git commit -m "test: verify temporal SHAP contributions end-to-end"
```
