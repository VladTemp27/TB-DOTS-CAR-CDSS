import type { PatientFeatures, ContributionResult } from './inference'

export interface MonthlyRecord {
  month: number
  weight?: number
  smearResult?: string
  adherence: 'full' | 'partial' | 'poor'
  failureProbability: number
  timestamp: number
}

export interface PredictionRecord {
  label: 0 | 1
  failureProbability: number
  contributions: ContributionResult['contributions']
  timestamp: number
}

export interface Patient {
  id: string
  name: string
  medicalId: string
  features: PatientFeatures
  treatmentRegimen?: string
  treatmentStartDate?: string
  createdAt: number
  predictions: PredictionRecord[]
  monthlyRecords: MonthlyRecord[]
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

function save(patients: Patient[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(patients))
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
