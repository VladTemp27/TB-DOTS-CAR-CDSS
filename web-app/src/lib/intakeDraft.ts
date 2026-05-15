import type { ContributionResult, PatientFeatures } from './inference'
import { addPrediction, savePatient } from './storage'
import type { Patient } from './storage'

export const INTAKE_DRAFT_KEY = 'tb_intake_draft'

export interface IntakeDraft {
  draftPatientId?: string
  name?: string
  age?: number
  sex?: string
  medicalId?: string
  province?: string
  city?: string
  registrationGroup?: string
  civilStatus?: string
  nationality?: string
  bacteriologicStatus?: string
  microscopyResult?: string
  anatomicalSite?: string
  sourceOfPatient?: string
  type?: string
  treatmentFacility?: string
  screeningFacility?: string
  dateStartedTx?: string
  dateOfDiagnosis?: string
  // Extended temporal-model fields
  xpertMtbRif?: string
  drugResistanceStatus?: string
  baselineHeightCm?: number
  baselineWeightKg?: number
  bpSystolic?: number
  bpDiastolic?: number
  heartRate?: number
  o2Sat?: number
  coMorbidities?: string
  chestXRayAtNotification?: string
  diagnosis?: string
  features?: PatientFeatures
  result?: ContributionResult
}

export type CommitReadyIntakeDraft = IntakeDraft & {
  draftPatientId: string
  features: PatientFeatures
  result: ContributionResult
}

export function createDraftPatientId(): string {
  return `MED-${new Date().getFullYear()}-${Math.floor(Math.random() * 9000 + 1000)}`
}

export function loadIntakeDraft(): IntakeDraft {
  try {
    return JSON.parse(sessionStorage.getItem(INTAKE_DRAFT_KEY) || '{}') as IntakeDraft
  } catch {
    return {}
  }
}

export function saveIntakeDraft(draft: IntakeDraft): void {
  sessionStorage.setItem(INTAKE_DRAFT_KEY, JSON.stringify(draft))
}

export function clearIntakeDraft(): void {
  sessionStorage.removeItem(INTAKE_DRAFT_KEY)
}

export function hasCommitReadyDraft(draft: IntakeDraft): draft is CommitReadyIntakeDraft {
  return Boolean(draft.draftPatientId && draft.features && draft.result)
}

export async function commitIntakePatient(draft: CommitReadyIntakeDraft): Promise<{ id: string; result: ContributionResult }> {
  const patient: Patient = {
    id: draft.draftPatientId,
    name: draft.name ?? 'Unknown',
    medicalId: draft.medicalId ?? draft.draftPatientId,
    features: draft.features,
    treatmentStartDate: draft.dateStartedTx,
    createdAt: Date.now(),
    predictions: [],
    monthlyRecords: [],
  }

  await savePatient(patient)
  await addPrediction(draft.draftPatientId, {
    label: draft.result.label,
    failureProbability: draft.result.failureProbability,
    contributions: draft.result.contributions,
    featuresUsed: draft.features,
    timestamp: Date.now(),
  })

  clearIntakeDraft()
  return { id: draft.draftPatientId, result: draft.result }
}
