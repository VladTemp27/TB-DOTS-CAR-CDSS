import * as ort from 'onnxruntime-web'
import type { Patient, MonthlyRecord, TemporalRiskInput, TemporalRiskSaveInput } from './storage'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/'

const MODEL_URL = '/model/hybrid_lstm_temporal.onnx'
const METADATA_URL = '/model/hybrid_lstm_temporal_metadata.json'

type Metadata = {
  modelName: string
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
  const f = patient.features

  const setv = (name: string, value: number | undefined | null) => {
    const i = idx.get(name)
    if (i == null || value == null || !Number.isFinite(value)) return
    out[i] = value
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
  }

  setv('age', f.age)
  setv('weight_kg', f.baselineWeightKg)
  setv('height_cm', f.baselineHeightCm)
  setv('bp_systolic', f.bpSystolic)
  setv('bp_diastolic', f.bpDiastolic)
  setv('heart_rate', f.heartRate)
  setv('o2_sat', f.o2Sat)
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
  if (sex === 'm' || sex === 'male') oneHot('sex_Male')
  if (sex === 'f' || sex === 'female') oneHot('sex_Female')

  const diagnosis = f.diagnosis?.trim()
  if (diagnosis) oneHot(`diagnosis_${diagnosis}`)

  const bacterio = f.bacteriologicStatus?.toLowerCase().includes('bacteriologically')
    ? 'Bacteriologically Confirmed'
    : f.bacteriologicStatus?.toLowerCase().includes('clinically')
      ? 'Clinically Diagnosed'
      : undefined
  if (bacterio) oneHot(`bacteriologic_status_${bacterio}`)

  const regGroup = f.registrationGroup?.toLowerCase().includes('relapse') ? 'Relapse' : f.registrationGroup ? 'New' : undefined
  if (regGroup) oneHot(`case_registration_group_${regGroup}`)

  if (patient.treatmentRegimen) oneHot(`treatment_regimen_${patient.treatmentRegimen === 'hrze' ? 'Regimen 1' : 'Regimen 2'}`)
  if (f.coMorbidities?.trim()) oneHot(`co_morbidities_${f.coMorbidities.trim()}`)

  const facility = String(f.treatmentHealthFacility || '').toLowerCase()
  for (const [token, col] of [
    ['pacdal', 'facility_Pacdal'], ['quirino', 'facility_Quirino Hill'], ['pinsao', 'facility_Pinsao'],
    ['asin', 'facility_Asin'], ['atab', 'facility_Atab'], ['atok', 'facility_Atok Trail'],
    ['eng', 'facility_Eng Hill'], ['city camp', 'facility_City Camp'], ['scout', 'facility_Scout Barrio'],
    ['campo', 'facility_Campo Filipino'],
  ] as const) {
    if (facility.includes(token)) oneHot(col)
  }

  setMissing('is_missing_age', f.age == null)
  setMissing('is_missing_weight_kg', f.baselineWeightKg == null)
  setMissing('is_missing_height_cm', f.baselineHeightCm == null)
  setMissing('is_missing_bp_systolic', f.bpSystolic == null)
  setMissing('is_missing_bp_diastolic', f.bpDiastolic == null)
  setMissing('is_missing_heart_rate', f.heartRate == null)
  setMissing('is_missing_o2_sat', f.o2Sat == null)
  setMissing('is_missing_xpert_mtb_rif', f.xpertMtbRif == null)
  setMissing('is_missing_smear_microscopy', f.microscopyResult == null)
  setMissing('is_missing_diagnosis', f.diagnosis == null)
  setMissing('is_missing_bacteriologic_status', f.bacteriologicStatus == null)
  setMissing('is_missing_treatment_regimen', patient.treatmentRegimen == null)
  setMissing('is_missing_co_morbidities', f.coMorbidities == null)

  for (const [name, i] of idx) {
    if (name.startsWith('is_missing_') && !explicitMissing.has(name)) out[i] = 1
  }
  return out
}

function currentRecord(input: TemporalRiskInput, priorCumulative: number): MonthlyRecord {
  const taken = input.monthlyDosesTaken
  const missed = input.monthlyMissedDoses
  const total = (taken ?? 0) + (missed ?? 0)
  const pct = taken != null && missed != null && total > 0 ? (taken / total) * 100 : undefined
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
    adherence: pct == null ? 'poor' : pct >= 90 ? 'full' : pct >= 50 ? 'partial' : 'poor',
    failureProbability: 0,
    timestamp: Date.now(),
  }
}

function buildTemporalMatrix(records: MonthlyRecord[], names: string[], currentMonth: number) {
  const byMonth = new Map(records.map(record => [record.month, record]))
  const out = new Float32Array((currentMonth + 1) * names.length)
  let cumulative = 0
  for (let month = 0; month <= currentMonth; month++) {
    const r = byMonth.get(month)
    const taken = r?.monthlyDosesTaken
    const missed = r?.monthlyMissedDoses
    if (taken != null) cumulative += taken
    const total = (taken ?? 0) + (missed ?? 0)
    const pct = r?.pctAdherence ?? (taken != null && missed != null && total > 0 ? (taken / total) * 100 : undefined)
    const cum = r?.cumulativeDosesTaken ?? (taken != null ? cumulative : undefined)
    const row: Record<string, number> = {
      cumulative_doses_taken: cum ?? 0,
      height: r?.height ?? 0,
      monthly_doses_taken: taken ?? 0,
      monthly_missed_doses: missed ?? 0,
      pct_adherence: pct ?? 0,
      smear_tb_lamp: r?.smearTbLamp ?? 0,
      weight: r?.weight ?? 0,
      xpert_mtb_rif: r?.xpertMtbRif ?? 0,
      is_missing_cumulative_doses_taken: cum == null ? 1 : 0,
      is_missing_height: r?.height == null ? 1 : 0,
      is_missing_monthly_doses_taken: taken == null ? 1 : 0,
      is_missing_monthly_missed_doses: missed == null ? 1 : 0,
      is_missing_pct_adherence: pct == null ? 1 : 0,
      is_missing_smear_tb_lamp: r?.smearTbLamp == null ? 1 : 0,
      is_missing_weight: r?.weight == null ? 1 : 0,
      is_missing_xpert_mtb_rif: r?.xpertMtbRif == null ? 1 : 0,
    }
    names.forEach((name, j) => { out[month * names.length + j] = row[name] ?? 0 })
  }
  return out
}

function scale(values: Float32Array, mean: number[], scaleValues: number[]) {
  const out = new Float32Array(values.length)
  for (let i = 0; i < values.length; i++) out[i] = (values[i] - mean[i % mean.length]) / scaleValues[i % scaleValues.length]
  return out
}

function ruleFloor(month: number, pct: number | undefined, smear: 0 | 1 | undefined, xpert: 0 | 1 | undefined) {
  let floor = 0
  if (pct == null) {
    if (month > 0) floor = Math.max(floor, 0.35)
  } else if (pct < 50) floor = Math.max(floor, 0.45 + ((50 - pct) / 50) * 0.35)
  else if (pct < 80) floor = Math.max(floor, 0.30 + ((80 - pct) / 30) * 0.15)
  else if (pct < 90) floor = Math.max(floor, 0.20)
  const labsMissing = smear == null && xpert == null
  if (smear === 1 || xpert === 1) floor = Math.max(floor, 0.55)
  if (labsMissing) {
    floor = Math.max(floor, 0.30)
    if (pct != null && pct < 50) floor = Math.max(floor, 0.70)
  }
  return Math.min(Math.max(floor, 0), 0.95)
}

export async function predictTemporalInBrowser(patient: Patient, input: TemporalRiskInput): Promise<TemporalRiskSaveInput> {
  const meta = await getMetadata()
  const sess = await getSession()
  const priorCumulative = patient.monthlyRecords.reduce((sum, record) => sum + (record.monthlyDosesTaken ?? 0), 0)
  const record = currentRecord(input, priorCumulative)
  const records = [...patient.monthlyRecords, record]
  const monthsUsed = [...new Set(records.map(r => r.month))].sort((a, b) => a - b)
  const seqLen = input.month + 1

  const xStatic = scale(buildStaticVector(patient, meta.staticFeatureNames), meta.staticScaler.mean, meta.staticScaler.scale)
  const xTemporal = scale(buildTemporalMatrix(records, meta.temporalFeatureNames, input.month), meta.temporalScaler.mean, meta.temporalScaler.scale)
  const result = await sess.run({
    x_temporal: new ort.Tensor('float32', xTemporal, [1, seqLen, meta.temporalFeatureNames.length]),
    x_static: new ort.Tensor('float32', xStatic, [1, meta.staticFeatureNames.length]),
  })
  const logit = Number((result.logit.data as Float32Array | number[])[0])
  const rawSuccess = sigmoid(logit)
  const rawFailure = 1 - rawSuccess
  const floor = ruleFloor(input.month, record.pctAdherence, input.smearTbLamp, input.xpertMtbRif)
  const previous = patient.predictions.at(-1)?.failureProbability
  const current = Math.max(rawFailure, floor)
  const adjusted = Math.min(Math.max(previous == null ? current : Math.max(current, previous - 0.30), 0), 0.95)
  return {
    ...input,
    label: adjusted >= 0.5 ? 1 : 0,
    failureProbability: adjusted,
    successProbability: 1 - adjusted,
    rawFailureProbability: rawFailure,
    rawSuccessProbability: rawSuccess,
    ruleFailureFloor: floor,
    previousFailureProbability: previous,
    adjustedFailureProbability: adjusted,
    threshold: meta.threshold,
    modelName: meta.modelName,
    monthsUsed,
    seqLen,
    riskPolicy: 'browser_onnx_conservative_monthly_floor_v1',
  }
}
