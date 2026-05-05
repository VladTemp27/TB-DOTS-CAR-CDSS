import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { getChoices, predictWithContributions } from '../lib/inference'
import { savePatient, generateId } from '../lib/storage'
import type { PatientFeatures } from '../lib/inference'

const DRAFT_KEY = 'tb_intake_draft'

function loadDraft() {
  try { return JSON.parse(sessionStorage.getItem(DRAFT_KEY) || '{}') } catch { return {} }
}

export function PatientIntakeStep2() {
  const navigate = useNavigate()
  const draft = loadDraft()

  const [bacteriologicStatus, setBacteriologicStatus] = useState(draft.bacteriologicStatus ?? '')
  const [microscopyResult, setMicroscopyResult] = useState(draft.microscopyResult ?? '')
  const [anatomicalSite, setAnatomicalSite] = useState(draft.anatomicalSite ?? '')
  const [sourceOfPatient, setSourceOfPatient] = useState(draft.sourceOfPatient ?? '')
  const [type, setType] = useState(draft.type ?? '')
  const [treatmentFacility, setTreatmentFacility] = useState(draft.treatmentFacility ?? '')
  const [screeningFacility, setScreeningFacility] = useState(draft.screeningFacility ?? '')
  const [dateStartedTx, setDateStartedTx] = useState(draft.dateStartedTx ?? '')
  const [dateOfDiagnosis, setDateOfDiagnosis] = useState(draft.dateOfDiagnosis ?? '')
  const [loading, setLoading] = useState(false)

  const bactChoices = getChoices('Bacteriologic_Status')
  const microChoices = getChoices('Microscopy_Result')
  const siteChoices = getChoices('Anatomical_Site')
  const sourceChoices = getChoices('Source_of_Patient')
  const typeChoices = getChoices('Type')
  const treatmentFacChoices = getChoices('Treatment_Health_Facility')
  const screeningFacChoices = getChoices('Screening_Diagnosing_Health_Facility')

  const valid = bacteriologicStatus && microscopyResult && anatomicalSite && sourceOfPatient && type

  async function handleGenerate() {
    if (!valid) return
    setLoading(true)

    const daysToTreatment = dateStartedTx && dateOfDiagnosis
      ? Math.max(0, (new Date(dateStartedTx).getTime() - new Date(dateOfDiagnosis).getTime()) / 86400000)
      : 7 // median default

    const features: PatientFeatures = {
      age: draft.age ?? 35,
      daysToTreatment,
      year: new Date().getFullYear(),
      sex: draft.sex ?? 'M',
      anatomicalSite,
      registrationGroup: draft.registrationGroup ?? 'NEW',
      bacteriologicStatus,
      microscopyResult,
      sourceOfPatient,
      type,
      province: draft.province ?? '',
      cityMunicipality: draft.city ?? '',
      treatmentHealthFacility: treatmentFacility,
      screeningDiagnosingHealthFacility: screeningFacility,
    }

    try {
      console.log('Triggering inference with features:', features)
      const result = await predictWithContributions(features)
      console.log('Inference complete, result:', result)
      const id = generateId()
      savePatient({
        id,
        name: draft.name ?? 'Unknown',
        medicalId: draft.medicalId ?? id,
        features,
        treatmentStartDate: dateStartedTx,
        createdAt: Date.now(),
        predictions: [{
          label: result.label,
          failureProbability: result.failureProbability,
          contributions: result.contributions,
          timestamp: Date.now(),
        }],
        monthlyRecords: [],
      })
      sessionStorage.removeItem(DRAFT_KEY)
      navigate(`/patient/${id}/result`, { state: { result, fresh: true } })
    } catch (err) {
      console.error(err)
      alert('Inference failed. Check the browser console for details.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <AppHeader />
      <StepProgress steps={4} current={2} />

      <main className="flex-1 px-4 pb-24 space-y-4">
        <div>
          <h2 className="font-semibold text-gray-900 text-base">Lab Results & Diagnosis</h2>
          <p className="text-sm text-gray-500">Enter laboratory and clinical results</p>
        </div>

        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Bacteriologic Status *</label>
            <select value={bacteriologicStatus} onChange={e => setBacteriologicStatus(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white">
              <option value="">Select</option>
              {bactChoices.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Microscopy / Smear Result *</label>
            <select value={microscopyResult} onChange={e => setMicroscopyResult(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white">
              <option value="">Select</option>
              {microChoices.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Anatomical Site *</label>
            <select value={anatomicalSite} onChange={e => setAnatomicalSite(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white">
              <option value="">Select</option>
              {siteChoices.map(v => <option key={v} value={v}>{v === 'P' ? 'PTB (Pulmonary)' : 'EPTB (Extra-pulmonary)'}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Source of Patient *</label>
            <select value={sourceOfPatient} onChange={e => setSourceOfPatient(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white">
              <option value="">Select</option>
              {sourceChoices.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Case Type *</label>
            <select value={type} onChange={e => setType(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white">
              <option value="">Select</option>
              {typeChoices.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Treatment Health Facility</label>
            <select value={treatmentFacility} onChange={e => setTreatmentFacility(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white">
              <option value="">Select (optional)</option>
              {treatmentFacChoices.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Screening / Diagnosing Facility</label>
            <select value={screeningFacility} onChange={e => setScreeningFacility(e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white">
              <option value="">Select (optional)</option>
              {screeningFacChoices.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-1">Date of Diagnosis</label>
              <input type="date" value={dateOfDiagnosis} onChange={e => setDateOfDiagnosis(e.target.value)}
                className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary" />
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-1">Date Started Tx</label>
              <input type="date" value={dateStartedTx} onChange={e => setDateStartedTx(e.target.value)}
                className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary" />
            </div>
          </div>
        </div>

        <div className="bg-blue-50 border border-blue-100 rounded-xl p-3 text-xs text-blue-700">
          TB Diagnostic Guide: TB diagnosis requires at least 2 positive confirmatory tests or clinical + radiological evidence with positive bacteriology.
        </div>
      </main>

      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-100 px-4 py-4 flex gap-3">
        <button onClick={() => navigate(-1)}
          className="flex-1 border border-gray-200 text-gray-700 py-3.5 rounded-xl font-semibold text-sm active:bg-gray-50">
          ← Back
        </button>
        <button
          onClick={handleGenerate}
          disabled={!valid || loading}
          className="flex-[2] bg-primary text-white py-3.5 rounded-xl font-semibold text-sm disabled:opacity-40 active:bg-primary-dark"
        >
          {loading ? 'Running model…' : 'Generate Diagnosis →'}
        </button>
      </div>
    </div>
  )
}
