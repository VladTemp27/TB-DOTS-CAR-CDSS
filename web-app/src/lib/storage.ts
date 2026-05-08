import type { PatientFeatures, ContributionResult } from './inference'
import { putXray } from './xrayStore'

export interface MonthlyRecord {
  month: number
  weight?: number
  smearResult?: string
  adherence: 'full' | 'partial' | 'poor'
  failureProbability: number
  timestamp: number
  // TODO(server): these are IndexedDB keys today; replace with server-side asset IDs
  xrayIds?: string[]
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

const STORAGE_KEY = 'tb_patients'

function load(): Patient[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

// SECURITY NOTE: Patient structured data is stored unencrypted in localStorage.
// Deploy only on a dedicated, access-controlled clinical workstation. See xrayStore.ts.
function save(patients: Patient[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(patients))
  } catch (e) {
    // QuotaExceededError — alert clinician so data loss is visible
    console.error('localStorage quota exceeded — patient data may not have been saved.', e)
    alert('Storage limit reached. Some patient data could not be saved. Please export records and clear storage.')
  }
}

export function getAllPatients(): Patient[] {
  return load().sort((a, b) => {
    const aRisk = a.predictions.at(-1)?.failureProbability ?? 0
    const bRisk = b.predictions.at(-1)?.failureProbability ?? 0
    return bRisk - aRisk
  })
}

export function getPatient(id: string): Patient | null {
  return load().find(p => p.id === id) ?? null
}

export function savePatient(patient: Patient): void {
  const patients = load()
  const idx = patients.findIndex(p => p.id === patient.id)
  if (idx >= 0) {
    patients[idx] = patient
  } else {
    patients.push(patient)
  }
  save(patients)
}

export function addPrediction(id: string, record: PredictionRecord): void {
  const patients = load()
  const p = patients.find(p => p.id === id)
  if (p) {
    p.predictions.push(record)
    save(patients)
  }
}

export function addMonthlyRecord(id: string, record: MonthlyRecord): void {
  const patients = load()
  const p = patients.find(p => p.id === id)
  if (p) {
    p.monthlyRecords.push(record)
    save(patients)
  }
}

export function generateId(): string {
  return `MED-${new Date().getFullYear()}-${Math.floor(Math.random() * 9000 + 1000)}`
}

// TODO(server): replace putXray calls below with server upload; see xrayStore.ts

export async function attachIntakeXrays(patientId: string, files: File[]): Promise<void> {
  if (files.length === 0) return
  // Load and verify patient exists before writing blobs to IndexedDB
  const patients = load()
  const p = patients.find(p => p.id === patientId)
  if (!p) return
  const ids = await Promise.all(files.map(f => putXray(f)))
  p.intakeXrayIds = [...(p.intakeXrayIds ?? []), ...ids]
  save(patients)
}

export async function attachMonthlyXrays(patientId: string, month: number, files: File[]): Promise<void> {
  if (files.length === 0) return
  // Load and verify both patient and monthly record exist before writing blobs
  const patients = load()
  const p = patients.find(p => p.id === patientId)
  const record = p?.monthlyRecords.find(r => r.month === month)
  if (!p || !record) return
  const ids = await Promise.all(files.map(f => putXray(f)))
  record.xrayIds = [...(record.xrayIds ?? []), ...ids]
  save(patients)
}
