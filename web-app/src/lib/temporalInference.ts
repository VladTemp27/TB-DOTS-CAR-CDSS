import * as ort from 'onnxruntime-web'
import type { Patient, MonthlyRecord, TemporalRiskInput, TemporalRiskSaveInput } from './storage'
import type { ContributionItem } from './inference'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/'

const MODEL_URL = '/model/hybrid_lstm_temporal.onnx'
const METADATA_URL = '/model/hybrid_lstm_temporal_metadata.json'

const TOTAL_MONTHS = 13

type Metadata = {
  modelName?: string
  modelType?: string
  outputProbability?: string
  logitMeaning?: string
  deploymentStatus?: string
  temperature?: number
  platt?: { a: number; b: number }
  threshold: number
  staticFeatureNames: string[]
  temporalFeatureNames: string[]
  staticScaler: { mean: number[]; scale: number[] }
  temporalScaler: { mean: number[]; scale: number[] }
}

let session: ort.InferenceSession | null = null
let metadata: Metadata | null = null

async function getSession() {
  if (!session) {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] })
  }
  return session
}

async function getMetadata() {
  if (!metadata) {
    const res = await fetch(METADATA_URL)
    if (!res.ok) throw new Error(`Failed to load temporal metadata: ${res.statusText}`)
    metadata = await res.json() as Metadata
  }
  return metadata
}

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x))
}

const TEMPORAL_CONTRIBUTION_GROUPS: { name: string; cols: readonly number[] }[] = [
  { name: 'Adherence',       cols: [0, 2, 3, 4, 8, 10, 11, 12] },
  { name: 'Weight / Height', cols: [1, 6, 9, 14] },
  { name: 'Smear Result',    cols: [5, 13] },
  { name: 'Xpert Result',    cols: [7, 15] },
]

function as01(value: unknown): 0 | 1 | undefined {
  if (value == null) return undefined
  if (typeof value === 'number') return value >= 0.5 ? 1 : 0
  const s = String(value).trim().toLowerCase()
  if (['1', 'pos', 'positive', 'true', 'yes', '3+', '2+', '+'].includes(s)) return 1
  if (['0', 'neg', 'negative', 'false', 'no'].includes(s)) return 0
  return undefined
}

function epochDays(value?: string) {
  if (!value) return undefined
  const ms = Date.parse(value)
  return Number.isFinite(ms) ? ms / 1000 / 86400 : undefined
}

function buildStaticVector(patient: Patient, names: string[]) {
  const out = new Float32Array(names.length)
  const idx = new Map(names.map((name, i) => [name, i]))
  const explicitMissing = new Set<string>()
  const setValues = new Set<string>()
  const f = patient.features

  const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v))

  const setv = (name: string, value: number | undefined | null) => {
    const i = idx.get(name)
    if (i == null || value == null || !Number.isFinite(value)) return
    out[i] = value
    setValues.add(name)
  }
  const setMissing = (name: string, missing: boolean) => {
    const i = idx.get(name)
    if (i == null) return
    explicitMissing.add(name)
    out[i] = missing ? 1 : 0
  }
  const oneHot = (name: string) => {
    const i = idx.get(name)
    if (i != null) out[i] = 1
    return i != null
  }

  // Training-time preprocessing adds explicit "<col>__missing" indicators for
  // any static column with NA in the training set. Mirror that in inference.
  const setTrainMissingIndicator = (col: string, missing: boolean) => {
    const name = `${col}__missing`
    const i = idx.get(name)
    if (i == null) return
    out[i] = missing ? 1 : 0
  }

  // Align inference-time bounds with training-time clipping.
  setv('age', f.age == null ? f.age : clamp(f.age, 0, 110))
  setv('weight_kg', f.baselineWeightKg == null ? f.baselineWeightKg : clamp(f.baselineWeightKg, 10, 180))
  setv('height_cm', f.baselineHeightCm == null ? f.baselineHeightCm : clamp(f.baselineHeightCm, 80, 220))
  setv('bp_systolic', f.bpSystolic == null ? f.bpSystolic : clamp(f.bpSystolic, 60, 220))
  setv('bp_diastolic', f.bpDiastolic == null ? f.bpDiastolic : clamp(f.bpDiastolic, 30, 140))
  setv('heart_rate', f.heartRate == null ? f.heartRate : clamp(f.heartRate, 20, 250))
  setv('o2_sat', f.o2Sat == null ? f.o2Sat : clamp(f.o2Sat, 70, 100))
  setv('xpert_mtb_rif', as01(f.xpertMtbRif))
  setv('smear_microscopy', as01(f.microscopyResult))

  const txStart = epochDays(patient.treatmentStartDate)
  setv('treatment_start_date', txStart)
  setv('intensive_phase_start_date', txStart)
  if (txStart != null && f.daysToTreatment != null) {
    setv('date_of_diagnosis', txStart - f.daysToTreatment)
    setv('date_of_notification', txStart - Math.max(0, f.daysToTreatment - 1))
  }

  const sex = String(f.sex || '').toLowerCase()
  const sexCat = (sex === 'm' || sex === 'male') ? 'Male' : (sex === 'f' || sex === 'female') ? 'Female' : undefined
  const sexMissing = sexCat == null
  // Mode-fill used during training; default to Male when missing/unrecognized.
  oneHot(`sex_${sexCat ?? 'Male'}`)
  setMissing('is_missing_sex', sexMissing)
  setTrainMissingIndicator('sex', sexMissing)

  const diagnosis = f.diagnosis?.trim()
  const diagnosisMissing = !diagnosis
  // Mode-fill used during training; default to TB Disease when missing.
  oneHot(`diagnosis_${diagnosisMissing ? 'TB Disease' : diagnosis}`)
  setMissing('is_missing_diagnosis', diagnosisMissing)
  setTrainMissingIndicator('diagnosis', diagnosisMissing)

  const bacterio = f.bacteriologicStatus?.toLowerCase().includes('bacteriologically')
    ? 'Bacteriologically Confirmed'
    : f.bacteriologicStatus?.toLowerCase().includes('clinically')
      ? 'Clinically Diagnosed'
      : undefined
  const bacterioMissing = bacterio == null
  // Mode-fill used during training; default to Clinically Diagnosed when missing.
  oneHot(`bacteriologic_status_${bacterio ?? 'Clinically Diagnosed'}`)
  setMissing('is_missing_bacteriologic_status', bacterioMissing)
  setTrainMissingIndicator('bacteriologic_status', bacterioMissing)

  const regRaw = f.registrationGroup ? String(f.registrationGroup) : ''
  const regLower = regRaw.trim().toLowerCase()
  const regGroup = regLower.includes('relapse') ? 'Relapse' : regLower.includes('new') ? 'New' : undefined
  const regMissing = regGroup == null
  // Mode-fill used during training; default to New when missing/unrecognized.
  oneHot(`case_registration_group_${regGroup ?? 'New'}`)
  setMissing('is_missing_case_registration_group', regMissing)
  setTrainMissingIndicator('case_registration_group', regMissing)

  const regimenMissing = patient.treatmentRegimen == null
  const regimen = patient.treatmentRegimen === 'hrze' ? 'Regimen 1' : patient.treatmentRegimen ? 'Regimen 2' : 'Regimen 1'
  oneHot(`treatment_regimen_${regimen}`)
  setMissing('is_missing_treatment_regimen', regimenMissing)
  setTrainMissingIndicator('treatment_regimen', regimenMissing)

  const comorbRaw = f.coMorbidities?.trim() || ''
  const comorbMissing = comorbRaw.length === 0
  // Mode-fill used during training; default to No Known Comorbidity.
  const comorbToEncode = comorbMissing ? 'No Known Comorbidity' : comorbRaw
  const comorbOk = oneHot(`co_morbidities_${comorbToEncode}`)
  if (!comorbOk) {
    // If it doesn't exist in the model's feature space, treat as missing.
    oneHot('co_morbidities_No Known Comorbidity')
    setTrainMissingIndicator('co_morbidities', true)
  } else {
    setTrainMissingIndicator('co_morbidities', comorbMissing)
  }
  setMissing('is_missing_co_morbidities', comorbMissing)

  const facility = String(f.treatmentHealthFacility || '').toLowerCase()
  const facilityMissing = facility.trim().length === 0
  for (const [token, col] of [
    ['pacdal', 'facility_Pacdal'], ['quirino', 'facility_Quirino Hill'], ['pinsao', 'facility_Pinsao'],
    ['asin', 'facility_Asin'], ['atab', 'facility_Atab'], ['atok', 'facility_Atok Trail'],
    ['eng', 'facility_Eng Hill'], ['city camp', 'facility_City Camp'], ['scout', 'facility_Scout Barrio'],
    ['campo', 'facility_Campo Filipino'],
  ] as const) {
    if (facility.includes(token)) oneHot(col)
  }

  // These fields are dropped as raw strings during training, but their missing
  // indicators remain as features.
  setMissing('is_missing_name_of_treatment_unit', facilityMissing)
  setMissing('is_missing_name_of_diagnosing_facility', true)

  setMissing('is_missing_age', f.age == null)
  setMissing('is_missing_weight_kg', f.baselineWeightKg == null)
  setMissing('is_missing_height_cm', f.baselineHeightCm == null)
  setMissing('is_missing_bp_systolic', f.bpSystolic == null)
  setMissing('is_missing_bp_diastolic', f.bpDiastolic == null)
  setMissing('is_missing_heart_rate', f.heartRate == null)
  setMissing('is_missing_o2_sat', f.o2Sat == null)
  setMissing('is_missing_xpert_mtb_rif', f.xpertMtbRif == null)
  setMissing('is_missing_smear_microscopy', f.microscopyResult == null)
  // The categorical missing flags above are set using the same semantics as the
  // one-hot encoding (missing/unrecognized => missing=1, plus mode-fill one-hot).

  for (const [name, i] of idx) {
    if (name.startsWith('is_missing_') && !explicitMissing.has(name)) out[i] = 1
  }

  const continuousFeatures = [
    'age', 'weight_kg', 'height_cm', 'bp_systolic', 'bp_diastolic', 'heart_rate', 'o2_sat',
    'date_of_diagnosis', 'date_of_notification', 'treatment_start_date', 'intensive_phase_start_date',
    'xpert_mtb_rif', 'smear_microscopy',
  ]
  for (const name of continuousFeatures) {
    if (!setValues.has(name)) {
      const i = idx.get(name)
      if (i != null) out[i] = NaN
    }
  }

  return out
}

function currentRecord(input: TemporalRiskInput, priorCumulative: number): MonthlyRecord {
  const taken = input.monthlyDosesTaken
  const missed = input.monthlyMissedDoses
  const total = (taken ?? 0) + (missed ?? 0)
  const pct = taken != null && missed != null && total > 0 ? taken / total : undefined
  return {
    month: input.month,
    weight: input.weight,
    height: input.height,
    smearTbLamp: input.smearTbLamp,
    xpertMtbRif: input.xpertMtbRif,
    monthlyDosesTaken: taken,
    monthlyMissedDoses: missed,
    cumulativeDosesTaken: taken != null ? priorCumulative + taken : undefined,
    pctAdherence: pct,
    adherence: pct == null ? 'poor' : pct >= 0.9 ? 'full' : pct >= 0.5 ? 'partial' : 'poor',
    failureProbability: 0,
    timestamp: Date.now(),
  }
}

function buildTemporalMatrix(records: MonthlyRecord[], names: string[], currentMonth: number) {
  const byMonth = new Map(records.map(record => [record.month, record]))
  const out = new Float32Array((currentMonth + 1) * names.length)
  let cumulative = 0

  const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v))
  for (let month = 0; month <= currentMonth; month++) {
    const r = byMonth.get(month)
    // Keep inference-time bounds aligned with training-time clinical clipping.
    const takenRaw = r?.monthlyDosesTaken
    const missedRaw = r?.monthlyMissedDoses
    const taken = takenRaw == null ? undefined : clamp(takenRaw, 0, 31)
    const missed = missedRaw == null ? undefined : clamp(missedRaw, 0, 31)
    if (taken != null) cumulative += taken
    const total = (taken ?? 0) + (missed ?? 0)
    const rawPct = r?.pctAdherence
    const storedPct = rawPct != null && rawPct > 1.0 ? rawPct / 100 : rawPct
    const pctUnclamped = storedPct ?? (taken != null && missed != null && total > 0 ? taken / total : undefined)
    const pct = pctUnclamped == null ? undefined : clamp(pctUnclamped, 0, 1)
    const cum = r?.cumulativeDosesTaken ?? (taken != null ? cumulative : undefined)

    const weightRaw = r?.weight
    const heightRaw = r?.height
    const weight = weightRaw == null ? undefined : clamp(weightRaw, 10, 180)
    const height = heightRaw == null ? undefined : clamp(heightRaw, 80, 220)

    const smear = r?.smearTbLamp
    const xpert = r?.xpertMtbRif
    const row: Record<string, number> = {
      cumulative_doses_taken: cum == null ? NaN : cum,
      height: height == null ? NaN : height,
      monthly_doses_taken: taken == null ? NaN : taken,
      monthly_missed_doses: missed == null ? NaN : missed,
      pct_adherence: pct == null ? NaN : pct,
      smear_tb_lamp: smear == null ? NaN : smear,
      weight: weight == null ? NaN : weight,
      xpert_mtb_rif: xpert == null ? NaN : xpert,
      is_missing_cumulative_doses_taken: cum == null ? 1 : 0,
      is_missing_height: height == null ? 1 : 0,
      is_missing_monthly_doses_taken: taken == null ? 1 : 0,
      is_missing_monthly_missed_doses: missed == null ? 1 : 0,
      is_missing_pct_adherence: pct == null ? 1 : 0,
      is_missing_smear_tb_lamp: smear == null ? 1 : 0,
      is_missing_weight: weight == null ? 1 : 0,
      is_missing_xpert_mtb_rif: xpert == null ? 1 : 0,
    }
    names.forEach((name, j) => { out[month * names.length + j] = row[name] ?? 0 })
  }
  return out
}

function scale(values: Float32Array, mean: number[], scaleValues: number[]) {
  const out = new Float32Array(values.length)
  for (let i = 0; i < values.length; i++) {
    out[i] = Number.isNaN(values[i]) ? 0 : (values[i] - mean[i % mean.length]) / scaleValues[i % scaleValues.length]
  }
  return out
}

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

function _applyCalibration(logit: number, meta: Metadata): number {
  return meta.platt
    ? sigmoid(meta.platt.a * logit + meta.platt.b)
    : sigmoid(logit / (
        meta.temperature && Number.isFinite(meta.temperature) && meta.temperature > 1e-6
          ? meta.temperature
          : 1
      ))
}

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
