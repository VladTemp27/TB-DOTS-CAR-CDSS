import type { PatientFeatures, ContributionResult, ContributionItem } from './inference'

export interface MonthlyRecord {
  month: number
  weight?: number
  height?: number
  smearTbLamp?: 0 | 1
  xpertMtbRif?: 0 | 1
  monthlyDosesTaken?: number
  monthlyMissedDoses?: number
  cumulativeDosesTaken?: number
  pctAdherence?: number
  adherence: 'full' | 'partial' | 'poor'
  failureProbability: number
  timestamp: number
  xrayIds?: string[]
  isMissingWeight?: 0 | 1
  isMissingHeight?: 0 | 1
  isMissingSmearTbLamp?: 0 | 1
  isMissingXpertMtbRif?: 0 | 1
  isMissingMonthlyDosesTaken?: 0 | 1
  isMissingMonthlyMissedDoses?: 0 | 1
}

export interface PredictionRecord {
  label: 0 | 1
  failureProbability: number
  contributions: ContributionResult['contributions']
  featuresUsed?: PatientFeatures
  timestamp: number
}

export interface Patient {
  id: string
  name: string
  medicalId: string
  features: PatientFeatures
  treatmentRegimen?: 'hrze' | 'mdr' | 'xdr'
  treatmentStartDate?: string
  createdAt: number
  predictions: PredictionRecord[]
  monthlyRecords: MonthlyRecord[]
  // TODO(server): these are IndexedDB keys today; replace with server-side asset IDs
  intakeXrayIds?: string[]
}

async function json<T>(res: Response): Promise<T> {
  const text = await res.text()
  return text ? (JSON.parse(text) as T) : (null as T)
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(body || res.statusText || `HTTP ${res.status}`)
  }
  return json<T>(res)
}

export async function getAllPatients(): Promise<Patient[]> {
  const patients = await api<Patient[]>('/api/patients')
  return patients.sort((a, b) => {
    const aRisk = a.predictions.at(-1)?.failureProbability ?? 0
    const bRisk = b.predictions.at(-1)?.failureProbability ?? 0
    return bRisk - aRisk
  })
}

export async function getPatient(id: string): Promise<Patient | null> {
  try {
    return await api<Patient>(`/api/patients/${encodeURIComponent(id)}`)
  } catch (e) {
    if (e instanceof Error && e.message.includes('404')) return null
    throw e
  }
}

export async function savePatient(patient: Patient): Promise<void> {
  await api<Patient>(`/api/patients/${encodeURIComponent(patient.id)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patient),
    },
  )
}

export async function addPrediction(id: string, record: PredictionRecord): Promise<void> {
  await api<{ ok: true }>(`/api/patients/${encodeURIComponent(id)}/predictions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(record),
  })
}

export async function addMonthlyRecord(id: string, record: MonthlyRecord): Promise<void> {
  await api<{ ok: true }>(`/api/patients/${encodeURIComponent(id)}/monthly-records`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(record),
  })
}

export type TemporalRiskInput = {
  month: number
  weight?: number
  height?: number
  smearTbLamp?: 0 | 1
  xpertMtbRif?: 0 | 1
  monthlyDosesTaken?: number
  monthlyMissedDoses?: number
}

export type TemporalRiskResult = {
  label: 0 | 1
  failureProbability: number
  successProbability: number
  threshold: number
  modelName: string
  month: number
  monthsUsed?: number[]
}

export type TemporalRiskSaveInput = TemporalRiskInput & TemporalRiskResult & {
  rawFailureProbability?: number
  rawSuccessProbability?: number
  ruleFailureFloor?: number
  previousFailureProbability?: number
  adjustedFailureProbability?: number
  seqLen?: number
  riskPolicy?: string
  contributions?: ContributionItem[]
}

export async function saveTemporalRiskRecord(patientId: string, input: TemporalRiskSaveInput): Promise<TemporalRiskResult> {
  return api<TemporalRiskResult>(`/api/patients/${encodeURIComponent(patientId)}/temporal-risk-record`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  })
}

export function generateId(): string {
  return `MED-${new Date().getFullYear()}-${Math.floor(Math.random() * 9000 + 1000)}`
}

export async function attachIntakeXrays(patientId: string, files: File[]): Promise<void> {
  if (files.length === 0) return

  // Ensure patient exists.
  const p = await getPatient(patientId)
  if (!p) return

  await Promise.all(
    files.map(async file => {
      const form = new FormData()
      form.append('file', file)
      await api(`/api/xrays?patient_id=${encodeURIComponent(patientId)}&kind=intake`, {
        method: 'POST',
        body: form,
      })
    }),
  )
}

export async function attachMonthlyXrays(patientId: string, month: number, files: File[]): Promise<void> {
  if (files.length === 0) return

  // Ensure patient and record exist.
  const p = await getPatient(patientId)
  const record = p?.monthlyRecords.find(r => r.month === month)
  if (!p || !record) return

  await Promise.all(
    files.map(async file => {
      const form = new FormData()
      form.append('file', file)
      await api(
        `/api/xrays?patient_id=${encodeURIComponent(patientId)}&kind=monthly&month=${encodeURIComponent(String(month))}`,
        {
          method: 'POST',
          body: form,
        },
      )
    }),
  )
}
