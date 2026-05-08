import { useState, useEffect, useCallback } from 'react'
import type { Patient } from '../lib/storage'
import type { ContributionResult } from '../lib/inference'
import { streamInterpretation } from '../lib/medgemma'
import type { InterpretRequest } from '../lib/medgemma'

interface UseStreamingResult {
  text: string
  isStreaming: boolean
  isComplete: boolean
  error: string | null
  retry: () => void
}

export function useStreamingInterpretation(
  patient: Patient | null,
  result: ContributionResult | null,
): UseStreamingResult {
  const [text, setText] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [isComplete, setIsComplete] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retryCount, setRetryCount] = useState(0)

  // Fix 1: Use scalar IDs as deps to prevent re-firing on parent re-render
  // patient.id is unique per patient; failureProbability is the prediction result.
  // On DiagnosticResult page both are set once from route state, so this is stable.
  useEffect(() => {
    if (patient === null || result === null) return

    setIsStreaming(true)
    setIsComplete(false)
    setError(null)
    setText('')

    // Note: year, province, cityMunicipality, treatmentHealthFacility, screeningDiagnosingHealthFacility
    // are intentionally omitted — the backend prompt only needs the clinically relevant fields
    // for the narrative interpretation.
    const req: InterpretRequest = {
      patient_name: patient.name,
      age: patient.features.age,
      sex: patient.features.sex,
      bacteriologic_status: patient.features.bacteriologicStatus,
      microscopy_result: patient.features.microscopyResult,
      anatomical_site: patient.features.anatomicalSite,
      registration_group: patient.features.registrationGroup,
      source_of_patient: patient.features.sourceOfPatient,
      type: patient.features.type,
      days_to_treatment: patient.features.daysToTreatment,
      failure_probability: result.failureProbability,
      contributions: result.contributions,
    }

    const controller = streamInterpretation(
      req,
      (token) => setText((prev) => prev + token),
      () => {
        setIsStreaming(false)
        setIsComplete(true)
      },
      (err) => {
        setIsStreaming(false)
        setError(err.message)
      },
    )

    // Fix 4: Set isStreaming=false in cleanup so AbortError path clears streaming state
    return () => {
      controller.abort()
      setIsStreaming(false)
    }
  }, [patient?.id, result?.failureProbability, retryCount])

  // Fix 3: Clear error immediately on retry click and optimistically set streaming
  const retry = useCallback(() => {
    setError(null)
    setIsStreaming(true)
    setRetryCount((c) => c + 1)
  }, [])

  return { text, isStreaming, isComplete, error, retry }
}
