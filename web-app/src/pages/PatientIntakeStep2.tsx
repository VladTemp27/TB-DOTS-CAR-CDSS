import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Info } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { getChoices, predictWithContributions } from '../lib/inference'
import { savePatient, generateId, addPrediction } from '../lib/storage'
import type { PatientFeatures } from '../lib/inference'
import { PageFooter } from '../components/PageFooter'

const DRAFT_KEY = 'tb_intake_draft'

function loadDraft() {
  try { return JSON.parse(sessionStorage.getItem(DRAFT_KEY) || '{}') } catch { return {} }
}

const inputCls = 'w-full min-h-[44px] border border-border rounded-xl px-3 py-2.5 text-sm text-ink-base bg-surface placeholder-ink-muted focus:border-primary outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1'
const selectCls = `${inputCls} bg-surface`
const labelCls = 'block text-sm font-medium text-ink-secondary mb-1'

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

  const bactChoices         = getChoices('Bacteriologic_Status')
  const microChoices        = getChoices('Microscopy_Result')
  const siteChoices         = getChoices('Anatomical_Site')
  const sourceChoices       = getChoices('Source_of_Patient')
  const typeChoices         = getChoices('Type')
  const treatmentFacChoices = getChoices('Treatment_Health_Facility')
  const screeningFacChoices = getChoices('Screening_Diagnosing_Health_Facility')

  const valid = bacteriologicStatus && microscopyResult && anatomicalSite && sourceOfPatient && type

  async function handleGenerate() {
    if (!valid) return
    setLoading(true)

    const daysToTreatment = dateStartedTx && dateOfDiagnosis
      ? Math.max(0, (new Date(dateStartedTx).getTime() - new Date(dateOfDiagnosis).getTime()) / 86400000)
      : 7

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
      const result = await predictWithContributions(features)
      const id = generateId()
      await savePatient({
        id,
        name: draft.name ?? 'Unknown',
        medicalId: draft.medicalId ?? id,
        features,
        treatmentStartDate: dateStartedTx,
        createdAt: Date.now(),
        predictions: [],
        monthlyRecords: [],
      })
      await addPrediction(id, {
        label: result.label,
        failureProbability: result.failureProbability,
        contributions: result.contributions,
        featuresUsed: features,
        timestamp: dateOfDiagnosis ? new Date(dateOfDiagnosis).getTime() : 0,
      })
      sessionStorage.removeItem(DRAFT_KEY)
      navigate('/patient/new/xray', { state: { id, result } })
    } catch (err) {
      console.error(err)
      alert('Inference failed. Check the browser console for details.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader />
      <StepProgress steps={5} current={2} />

      <div className="flex-1 flex flex-col">
        <div className="flex-1 px-4 lg:px-8 pb-6 pt-4 w-full max-w-3xl mx-auto space-y-4">

          <div>
            <h1 className="font-bold text-ink-base text-base font-display">Lab Results &amp; Diagnosis</h1>
            <p className="text-sm text-ink-secondary">Enter laboratory and clinical results</p>
          </div>

          {/* Bacteriology */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-6 space-y-3">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="bactStatus">Bacteriologic Status <span className="text-red-500">*</span></label>
                <select id="bactStatus" value={bacteriologicStatus} onChange={e => setBacteriologicStatus(e.target.value)} className={selectCls}>
                  <option value="">Select</option>
                  {bactChoices.map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
              <div>
                <label className={labelCls} htmlFor="microResult">Microscopy / Smear Result <span className="text-red-500">*</span></label>
                <select id="microResult" value={microscopyResult} onChange={e => setMicroscopyResult(e.target.value)} className={selectCls}>
                  <option value="">Select</option>
                  {microChoices.map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="anatSite">Anatomical Site <span className="text-red-500">*</span></label>
                <select id="anatSite" value={anatomicalSite} onChange={e => setAnatomicalSite(e.target.value)} className={selectCls}>
                  <option value="">Select</option>
                  {siteChoices.map(v => <option key={v} value={v}>{v === 'P' ? 'PTB (Pulmonary)' : 'EPTB (Extra-pulmonary)'}</option>)}
                </select>
              </div>
              <div>
                <label className={labelCls} htmlFor="source">Source of Patient <span className="text-red-500">*</span></label>
                <select id="source" value={sourceOfPatient} onChange={e => setSourceOfPatient(e.target.value)} className={selectCls}>
                  <option value="">Select</option>
                  {sourceChoices.map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
            </div>

            <div>
              <label className={labelCls} htmlFor="caseType">Case Type <span className="text-red-500">*</span></label>
              <select id="caseType" value={type} onChange={e => setType(e.target.value)} className={selectCls}>
                <option value="">Select</option>
                {typeChoices.map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          </div>

          {/* Facilities & Dates */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-6 space-y-3">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="txFac">Treatment Health Facility</label>
                <select id="txFac" value={treatmentFacility} onChange={e => setTreatmentFacility(e.target.value)} className={selectCls}>
                  <option value="">Select (optional)</option>
                  {treatmentFacChoices.map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
              <div>
                <label className={labelCls} htmlFor="screenFac">Screening / Diagnosing Facility</label>
                <select id="screenFac" value={screeningFacility} onChange={e => setScreeningFacility(e.target.value)} className={selectCls}>
                  <option value="">Select (optional)</option>
                  {screeningFacChoices.map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="diagDate">Date of Diagnosis</label>
                <input id="diagDate" type="date" value={dateOfDiagnosis}
                  onChange={e => setDateOfDiagnosis(e.target.value)}
                  className={inputCls} />
              </div>
              <div>
                <label className={labelCls} htmlFor="txDate">Date Started Treatment</label>
                <input id="txDate" type="date" value={dateStartedTx}
                  onChange={e => setDateStartedTx(e.target.value)}
                  className={inputCls} />
              </div>
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-100 rounded-2xl p-3 flex gap-2.5 items-start">
            <Info size={16} className="text-blue-500 mt-0.5 flex-shrink-0" aria-hidden="true" />
            <p className="text-xs text-blue-700">
              TB diagnosis requires at least 2 positive confirmatory tests or clinical + radiological evidence with positive bacteriology.
            </p>
          </div>
        </div>
      </div>

      <PageFooter>
        <div className="flex gap-3">
          <button onClick={() => navigate(-1)}
            className="flex-1 border border-border text-ink-secondary py-3.5 rounded-xl font-semibold text-sm active:bg-gray-50">
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
      </PageFooter>
    </div>
  )
}
