import type { ContributionResult, PatientFeatures } from './inference'
import { savePatient, saveTemporalRiskRecord } from './storage'
import type { Patient } from './storage'
import { predictTemporalWithContributions } from './temporalInference'

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

function labTo01(value?: string): 0 | 1 | undefined {
  if (!value) return undefined
  const normalized = value.trim().toLowerCase()
  if (['positive', 'pos', '1', '1+', '2+', '3+'].includes(normalized)) return 1
  if (['negative', 'neg', '0'].includes(normalized)) return 0
  return undefined
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
  const baselinePrediction = await predictTemporalWithContributions(patient, {
    month: 0,
    weight: draft.features.baselineWeightKg,
    height: draft.features.baselineHeightCm,
    smearTbLamp: labTo01(draft.features.microscopyResult),
    xpertMtbRif: labTo01(draft.features.xpertMtbRif),
  })
  const savedBaseline = await saveTemporalRiskRecord(draft.draftPatientId, baselinePrediction)
  const result: ContributionResult = {
    label: savedBaseline.label,
    failureProbability: savedBaseline.failureProbability,
    successProbability: savedBaseline.successProbability,
    contributions: baselinePrediction.contributions,
  }

  clearIntakeDraft()
  return { id: draft.draftPatientId, result }
}
