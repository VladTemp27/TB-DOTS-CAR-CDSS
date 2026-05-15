import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Info } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { getChoices, predictWithContributions } from '../lib/inference'
import type { PatientFeatures } from '../lib/inference'
import { PageFooter } from '../components/PageFooter'
import { createDraftPatientId, loadIntakeDraft, saveIntakeDraft } from '../lib/intakeDraft'

const inputCls = 'w-full min-h-[44px] border border-border rounded-xl px-3 py-2.5 text-sm text-ink-base bg-surface placeholder-ink-muted focus:border-primary outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1'
const selectCls = `${inputCls} bg-surface`
const labelCls = 'block text-sm font-medium text-ink-secondary mb-1'

export function PatientIntakeStep2() {
  const navigate = useNavigate()
  const draft = loadIntakeDraft()

  const [bacteriologicStatus, setBacteriologicStatus] = useState(draft.bacteriologicStatus ?? '')
  const [microscopyResult, setMicroscopyResult] = useState(draft.microscopyResult ?? '')
  const [anatomicalSite, setAnatomicalSite] = useState(draft.anatomicalSite ?? '')
  const [sourceOfPatient, setSourceOfPatient] = useState(draft.sourceOfPatient ?? '')
  const [type, setType] = useState(draft.type ?? '')
  const [treatmentFacility, setTreatmentFacility] = useState(draft.treatmentFacility ?? '')
  const [screeningFacility, setScreeningFacility] = useState(draft.screeningFacility ?? '')
  const [dateStartedTx, setDateStartedTx] = useState(draft.dateStartedTx ?? '')
  const [dateOfDiagnosis, setDateOfDiagnosis] = useState(draft.dateOfDiagnosis ?? '')

  // Extended temporal-model fields
  const [xpertMtbRif, setXpertMtbRif] = useState(draft.xpertMtbRif ?? '')
  const [drugResistanceStatus, setDrugResistanceStatus] = useState(draft.drugResistanceStatus ?? '')
  const [baselineHeightCm, setBaselineHeightCm] = useState(draft.baselineHeightCm ? String(draft.baselineHeightCm) : '')
  const [baselineWeightKg, setBaselineWeightKg] = useState(draft.baselineWeightKg ? String(draft.baselineWeightKg) : '')
  const [bpSystolic, setBpSystolic] = useState(draft.bpSystolic ? String(draft.bpSystolic) : '')
  const [bpDiastolic, setBpDiastolic] = useState(draft.bpDiastolic ? String(draft.bpDiastolic) : '')
  const [heartRate, setHeartRate] = useState(draft.heartRate ? String(draft.heartRate) : '')
  const [o2Sat, setO2Sat] = useState(draft.o2Sat ? String(draft.o2Sat) : '')
  const [coMorbidities, setCoMorbidities] = useState(draft.coMorbidities ?? '')
  const [chestXRayAtNotification, setChestXRayAtNotification] = useState(draft.chestXRayAtNotification ?? '')
  const [diagnosis, setDiagnosis] = useState(draft.diagnosis ?? '')
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

    const parsedHeight = parseFloat(baselineHeightCm)
    const parsedWeight = parseFloat(baselineWeightKg)
    const parsedBpSys = parseFloat(bpSystolic)
    const parsedBpDia = parseFloat(bpDiastolic)
    const parsedHr = parseFloat(heartRate)
    const parsedO2 = parseFloat(o2Sat)

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
      // Extended fields
      xpertMtbRif: xpertMtbRif || undefined,
      drugResistanceStatus: drugResistanceStatus || undefined,
      baselineHeightCm: Number.isFinite(parsedHeight) ? parsedHeight : undefined,
      baselineWeightKg: Number.isFinite(parsedWeight) ? parsedWeight : undefined,
      bpSystolic: Number.isFinite(parsedBpSys) ? parsedBpSys : undefined,
      bpDiastolic: Number.isFinite(parsedBpDia) ? parsedBpDia : undefined,
      heartRate: Number.isFinite(parsedHr) ? parsedHr : undefined,
      o2Sat: Number.isFinite(parsedO2) ? parsedO2 : undefined,
      coMorbidities: coMorbidities || undefined,
      chestXRayAtNotification: chestXRayAtNotification || undefined,
      diagnosis: diagnosis || undefined,
    }

    try {
      const result = await predictWithContributions(features)
      const draftPatientId = draft.draftPatientId ?? draft.medicalId ?? createDraftPatientId()
      saveIntakeDraft({
        ...draft,
        draftPatientId,
        bacteriologicStatus,
        microscopyResult,
        anatomicalSite,
        sourceOfPatient,
        type,
        treatmentFacility,
        screeningFacility,
        dateStartedTx,
        dateOfDiagnosis,
        xpertMtbRif: xpertMtbRif || undefined,
        drugResistanceStatus: drugResistanceStatus || undefined,
        baselineHeightCm: Number.isFinite(parsedHeight) ? parsedHeight : undefined,
        baselineWeightKg: Number.isFinite(parsedWeight) ? parsedWeight : undefined,
        bpSystolic: Number.isFinite(parsedBpSys) ? parsedBpSys : undefined,
        bpDiastolic: Number.isFinite(parsedBpDia) ? parsedBpDia : undefined,
        heartRate: Number.isFinite(parsedHr) ? parsedHr : undefined,
        o2Sat: Number.isFinite(parsedO2) ? parsedO2 : undefined,
        coMorbidities: coMorbidities || undefined,
        chestXRayAtNotification: chestXRayAtNotification || undefined,
        diagnosis: diagnosis || undefined,
        features,
        result,
      })
      navigate('/patient/new/xray')
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

            {/* Xpert MTB/RIF */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="xpertMtbRif">Xpert MTB/RIF</label>
                <select id="xpertMtbRif" value={xpertMtbRif} onChange={e => setXpertMtbRif(e.target.value)} className={selectCls}>
                  <option value="">Select (optional)</option>
                  <option value="Positive">Positive</option>
                  <option value="Negative">Negative</option>
                  <option value="Not Done">Not Done</option>
                  <option value="Invalid">Invalid</option>
                </select>
              </div>
              <div>
                <label className={labelCls} htmlFor="drugResistance">Drug Resistance Status</label>
                <select
                  id="drugResistance"
                  value={drugResistanceStatus}
                  onChange={e => setDrugResistanceStatus(e.target.value)}
                  className={inputCls + ' bg-surface'}
                >
                  <option value="">Select Drug Resistance Status</option>
                  <option value="DS-TB">Drug-Susceptible TB (DS-TB)</option>
                  <option value="Hr-TB">Isoniazid-Monoresistant TB (Hr-TB)</option>
                  <option value="RR-TB">Rifampicin-Resistant TB (RR-TB)</option>
                  <option value="MDR-TB">Multi-Drug Resistant TB (MDR-TB)</option>
                  <option value="Pre-XDR-TB">Pre-Extensively Drug-Resistant TB (Pre-XDR)</option>
                  <option value="XDR-TB">Extensively Drug-Resistant TB (XDR-TB)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Baseline Vitals & Measurements */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-6 space-y-3">
            <h3 className="text-sm font-semibold text-ink-base">Baseline Vitals &amp; Measurements</h3>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
              <div>
                <label className={labelCls} htmlFor="baseHeight">Height (cm)</label>
                <input id="baseHeight" type="number" step="0.1" placeholder="e.g. 162.5"
                  value={baselineHeightCm} onChange={e => setBaselineHeightCm(e.target.value)} className={inputCls} />
              </div>
              <div>
                <label className={labelCls} htmlFor="baseWeight">Weight (kg)</label>
                <input id="baseWeight" type="number" step="0.1" placeholder="e.g. 55.0"
                  value={baselineWeightKg} onChange={e => setBaselineWeightKg(e.target.value)} className={inputCls} />
              </div>
              <div>
                <label className={labelCls} htmlFor="heartRate">Heart Rate (bpm)</label>
                <input id="heartRate" type="number" placeholder="e.g. 72"
                  value={heartRate} onChange={e => setHeartRate(e.target.value)} className={inputCls} />
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
              <div>
                <label className={labelCls} htmlFor="bpSys">BP Systolic</label>
                <input id="bpSys" type="number" placeholder="e.g. 120"
                  value={bpSystolic} onChange={e => setBpSystolic(e.target.value)} className={inputCls} />
              </div>
              <div>
                <label className={labelCls} htmlFor="bpDia">BP Diastolic</label>
                <input id="bpDia" type="number" placeholder="e.g. 80"
                  value={bpDiastolic} onChange={e => setBpDiastolic(e.target.value)} className={inputCls} />
              </div>
              <div>
                <label className={labelCls} htmlFor="o2Sat">O2 Saturation (%)</label>
                <input id="o2Sat" type="number" min="0" max="100" placeholder="e.g. 98"
                  value={o2Sat} onChange={e => setO2Sat(e.target.value)} className={inputCls} />
              </div>
            </div>
          </div>

          {/* Clinical Info */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-6 space-y-3">
            <h3 className="text-sm font-semibold text-ink-base">Clinical Information</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className={labelCls} htmlFor="coMorbidities">Co-morbidities</label>
                <select
                  id="coMorbidities"
                  value={coMorbidities}
                  onChange={e => setCoMorbidities(e.target.value)}
                  className={inputCls + ' bg-surface'}
                >
                  <option value="">Select Co-morbidities</option>
                  <option value="None">None</option>
                  <option value="Diabetes">Diabetes</option>
                  <option value="HIV">HIV</option>
                  <option value="LungDisease">Lung Disease</option>
                  <option value="KidneyDisease">Kidney Disease</option>
                  <option value="LiverDisease">Liver Disease</option>
                  <option value="Malnutrition">Malnutrition</option>
                  <option value="OtherImmunosuppression">Other Immunosuppression (Cancer/Steroids)</option>
                  <option value="Multiple">Multiple Comorbidities</option>
                </select>
              </div>
              <div>
                <label className={labelCls} htmlFor="chestXRay">Chest X-ray at Notification</label>
                <select id="chestXRay" value={chestXRayAtNotification} onChange={e => setChestXRayAtNotification(e.target.value)} className={selectCls}>
                  <option value="">Select (optional)</option>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="Married">Married</option>
                  <option value="Divorced">Divorced</option>
                </select>
              </div>
              <div>
                <label className={labelCls} htmlFor="chestXRay">Chest X-ray at Notification</label>
                <select id="chestXRay" value={chestXRayAtNotification} onChange={e => setChestXRayAtNotification(e.target.value)} className={selectCls}>
                  <option value="">Select (optional)</option>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="Not Done">Not Done</option>
                </select>
              </div>
            </div>
            <div>
              <label className={labelCls} htmlFor="diagnosis">Diagnosis</label>
              <select
                  id="diagnosis"
                  value={diagnosis}
                  onChange={e => setDiagnosis(e.target.value)}
                  className={inputCls + ' bg-surface'}
                >
                  <option value="">Select Diagnosis Type</option>
                  <option value="PTB-BacteriologicallyConfirmed">Pulmonary TB - Bacteriologically Confirmed</option>
                  <option value="PTB-ClinicallyDiagnosed">Pulmonary TB - Clinically Diagnosed</option>
                  <option value="EPTB">Extrapulmonary TB (EPTB)</option>
                  <option value="Both">Both Pulmonary and Extrapulmonary TB</option>
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
          <button onClick={() => navigate('/patient/new')}
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
