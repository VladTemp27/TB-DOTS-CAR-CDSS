import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Check, AlertTriangle, X } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { XrayUploadField } from '../components/XrayUploadField'
import { getChoices, predictWithContributions } from '../lib/inference'
import { getPatient, addMonthlyRecord, addPrediction, attachMonthlyXrays } from '../lib/storage'
import { PageFooter } from '../components/PageFooter'
import type { Patient } from '../lib/storage'

type Adherence = 'full' | 'partial' | 'poor'

export function MonthlyCheckin() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const [patient, setPatient] = useState<Patient | null>(null)
  const [loadingPatient, setLoadingPatient] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!id) {
      setLoadingPatient(false)
      setPatient(null)
      return
    }
    setLoadingPatient(true)
    ;(async () => {
      try {
        const p = await getPatient(id)
        if (!cancelled) setPatient(p)
      } catch (e) {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e))
      } finally {
        if (!cancelled) setLoadingPatient(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [id])

  const [weight, setWeight] = useState('')
  const [height, setHeight] = useState('')
  const [smearResult, setSmearResult] = useState('')
  const [smearTbLamp, setSmearTbLamp] = useState('')
  const [xpertMtbRif, setXpertMtbRif] = useState('')
  const [monthlyDosesTaken, setMonthlyDosesTaken] = useState('')
  const [monthlyMissedDoses, setMonthlyMissedDoses] = useState('')
  const [cumulativeDosesTaken, setCumulativeDosesTaken] = useState('')
  const [pctAdherence, setPctAdherence] = useState('')
  const [adherence, setAdherence] = useState<Adherence | ''>('')
  const [xrayFiles, setXrayFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)

  const microChoices  = getChoices('Microscopy_Result')
  const monthNumber   = (patient?.monthlyRecords.length ?? 0) + 1

  async function handleGenerate() {
    if (!patient || !id || !adherence) return
    setLoading(true)

    const updatedFeatures = { ...patient.features }
    if (smearResult) updatedFeatures.microscopyResult = smearResult

    try {
      const result = await predictWithContributions(updatedFeatures)

      const parsedWeight = parseFloat(weight)
      const parsedHeight = parseFloat(height)
      const parsedSmearTbLamp = smearTbLamp === '1' ? 1 : smearTbLamp === '0' ? 0 : undefined
      const parsedXpertMtbRif = xpertMtbRif === '1' ? 1 : xpertMtbRif === '0' ? 0 : undefined
      const parsedMonthlyDoses = parseInt(monthlyDosesTaken, 10)
      const parsedMonthlyMissed = parseInt(monthlyMissedDoses, 10)
      const parsedCumulativeDoses = parseInt(cumulativeDosesTaken, 10)
      const parsedPctAdherence = parseFloat(pctAdherence)
      const now = Date.now()
      await addMonthlyRecord(id, {
        month: monthNumber,
        weight: Number.isFinite(parsedWeight) ? parsedWeight : undefined,
        height: Number.isFinite(parsedHeight) ? parsedHeight : undefined,
        smearResult: smearResult || undefined,
        smearTbLamp: parsedSmearTbLamp as 0 | 1 | undefined,
        xpertMtbRif: parsedXpertMtbRif as 0 | 1 | undefined,
        monthlyDosesTaken: Number.isFinite(parsedMonthlyDoses) ? parsedMonthlyDoses : undefined,
        monthlyMissedDoses: Number.isFinite(parsedMonthlyMissed) ? parsedMonthlyMissed : undefined,
        cumulativeDosesTaken: Number.isFinite(parsedCumulativeDoses) ? parsedCumulativeDoses : undefined,
        pctAdherence: Number.isFinite(parsedPctAdherence) ? parsedPctAdherence : undefined,
        adherence,
        failureProbability: result.failureProbability,
        timestamp: now,
      })

      if (xrayFiles.length > 0) {
        await attachMonthlyXrays(id, monthNumber, xrayFiles)
      }

      await addPrediction(id, {
        label: result.label,
        failureProbability: result.failureProbability,
        contributions: result.contributions,
        featuresUsed: updatedFeatures,
        timestamp: now,
      })

      navigate(`/patient/${id}/risk-update`, { state: { result, monthNumber }, replace: true })
    } catch (err) {
      console.error(err)
      alert('Inference failed.')
    } finally {
      setLoading(false)
    }
  }

  const adherenceOptions: Array<{
    value: Adherence
    label: string
    sublabel: string
    Icon: typeof Check
    selectedCls: string
  }> = [
    { value: 'full',    label: 'Full',    sublabel: 'All doses taken', Icon: Check,         selectedCls: 'border-risk-low bg-risk-low/10 text-risk-low' },
    { value: 'partial', label: 'Partial', sublabel: 'Some missed',     Icon: AlertTriangle, selectedCls: 'border-risk-med bg-risk-med/10 text-risk-med' },
    { value: 'poor',    label: 'Poor',    sublabel: 'Most missed',     Icon: X,             selectedCls: 'border-risk-high bg-risk-high/10 text-risk-high' },
  ]

  const inputCls = 'w-full min-h-[44px] border border-border rounded-xl px-3 py-2.5 text-sm text-ink-base bg-surface placeholder-ink-muted focus:border-primary outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1'

  if (loadingPatient) {
    return (
      <div className="min-h-screen bg-bg flex flex-col">
        <AppHeader backLabel="Back" onBack={() => navigate(-1)} />
        <div className="flex-1 px-4 lg:px-8 pt-10 pb-6 w-full max-w-3xl mx-auto">
          <p className="text-sm text-ink-muted">Loading patient…</p>
        </div>
      </div>
    )
  }

  if (loadError) {
    return (
      <div className="min-h-screen bg-bg flex flex-col">
        <AppHeader backLabel="Back" onBack={() => navigate(-1)} />
        <div className="flex-1 px-4 lg:px-8 pt-10 pb-6 w-full max-w-3xl mx-auto">
          <p className="text-sm text-risk-high">{loadError}</p>
        </div>
      </div>
    )
  }

  if (!patient || !id) {
    return (
      <div className="min-h-screen bg-bg flex flex-col">
        <AppHeader backLabel="Back" onBack={() => navigate(-1)} />
        <div className="flex-1 px-4 lg:px-8 pt-10 pb-6 w-full max-w-3xl mx-auto">
          <p className="text-sm text-ink-muted">Patient not found.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader backLabel="Back" onBack={() => navigate(-1)} />

      <div className="flex-1 flex flex-col">
        <div className="flex-1 px-4 lg:px-8 pb-6 pt-4 w-full max-w-3xl mx-auto space-y-5">

          {/* Progress bar */}
          <div>
            <div className="h-1.5 bg-gray-100 rounded-full">
              <div className="h-full bg-primary rounded-full transition-all"
                style={{ width: `${((monthNumber - 1) / 6) * 100}%` }} />
            </div>
            <p className="text-xs text-ink-muted mt-1">Month {monthNumber} of 6</p>
          </div>

          <div>
            <h2 className="font-bold text-ink-base text-lg font-display">Monthly Follow-up</h2>
            <p className="text-sm text-ink-secondary">Recording Month {monthNumber} data for {patient?.name}</p>
          </div>

          {/* Measurements */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-5 space-y-3">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="weight">
                  Current Weight (kg)
                </label>
                <input
                  id="weight"
                  type="number"
                  step="0.1"
                  placeholder="e.g. 52.5"
                  value={weight}
                  onChange={e => setWeight(e.target.value)}
                  className={inputCls}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="height">
                  Height (cm)
                </label>
                <input
                  id="height"
                  type="number"
                  step="0.1"
                  placeholder="e.g. 162.5"
                  value={height}
                  onChange={e => setHeight(e.target.value)}
                  className={inputCls}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="smear">
                  Sputum Smear Result
                </label>
                <select
                  id="smear"
                  value={smearResult}
                  onChange={e => setSmearResult(e.target.value)}
                  className={inputCls + ' bg-surface'}
                >
                  <option value="">Not performed</option>
                  {microChoices.map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* Dose Tracking */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-5 space-y-3">
            <h3 className="text-sm font-semibold text-ink-base">Dose Tracking</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="monthlyDosesTaken">
                  Monthly Doses Taken
                </label>
                <input
                  id="monthlyDosesTaken"
                  type="number"
                  min="0"
                  placeholder="e.g. 28"
                  value={monthlyDosesTaken}
                  onChange={e => setMonthlyDosesTaken(e.target.value)}
                  className={inputCls}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="monthlyMissedDoses">
                  Monthly Missed Doses
                </label>
                <input
                  id="monthlyMissedDoses"
                  type="number"
                  min="0"
                  placeholder="e.g. 2"
                  value={monthlyMissedDoses}
                  onChange={e => setMonthlyMissedDoses(e.target.value)}
                  className={inputCls}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="cumulativeDosesTaken">
                  Cumulative Doses Taken
                </label>
                <input
                  id="cumulativeDosesTaken"
                  type="number"
                  min="0"
                  placeholder="e.g. 84"
                  value={cumulativeDosesTaken}
                  onChange={e => setCumulativeDosesTaken(e.target.value)}
                  className={inputCls}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="pctAdherence">
                  Adherence (%)
                </label>
                <input
                  id="pctAdherence"
                  type="number"
                  min="0"
                  max="100"
                  step="0.1"
                  placeholder="e.g. 93.3"
                  value={pctAdherence}
                  onChange={e => setPctAdherence(e.target.value)}
                  className={inputCls}
                />
              </div>
            </div>
          </div>

          {/* Lab Results */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-5 space-y-3">
            <h3 className="text-sm font-semibold text-ink-base">Lab Results</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="smearTbLamp">
                  Smear TB LAMP
                </label>
                <select
                  id="smearTbLamp"
                  value={smearTbLamp}
                  onChange={e => setSmearTbLamp(e.target.value)}
                  className={inputCls + ' bg-surface'}
                >
                  <option value="">Not performed</option>
                  <option value="0">Negative (0)</option>
                  <option value="1">Positive (1)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="xpertMtbRif">
                  Xpert MTB/RIF
                </label>
                <select
                  id="xpertMtbRif"
                  value={xpertMtbRif}
                  onChange={e => setXpertMtbRif(e.target.value)}
                  className={inputCls + ' bg-surface'}
                >
                  <option value="">Not performed</option>
                  <option value="0">Negative (0)</option>
                  <option value="1">Positive (1)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Adherence */}
          <div role="group" aria-labelledby="adherence-label">
            <p id="adherence-label" className="block text-sm font-medium text-ink-secondary mb-2">Adherence This Month *</p>
            <div className="flex gap-3">
              {adherenceOptions.map(({ value, label, sublabel, Icon, selectedCls }) => (
                <button
                  key={value}
                  onClick={() => setAdherence(value)}
                  aria-pressed={adherence === value}
                  className={`flex-1 border-2 rounded-2xl p-3 lg:p-4 text-center transition-colors min-h-[80px] flex flex-col items-center justify-center gap-1
                    ${adherence === value ? selectedCls : 'border-border bg-surface text-ink-secondary hover:border-primary/30'}`}
                >
                  <Icon size={18} aria-hidden="true" />
                  <p className="text-xs font-bold mt-0.5">{label}</p>
                  <p className="text-xs opacity-70">{sublabel}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Chest X-rays — optional */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-5 space-y-3">
            <p className="text-sm font-medium text-ink-secondary">Chest X-rays (optional)</p>
            <XrayUploadField onChange={setXrayFiles} />
          </div>

          <div className="bg-blue-50 border border-blue-100 rounded-2xl p-3 text-xs text-blue-700">
            Follow-up Guide: Record weight, height, dose tracking, and lab results at each monthly visit. Adherence gaps and positive lab results are top predictors of treatment failure.
          </div>
        </div>
      </div>

      <PageFooter>
        <div className="flex gap-3">
          <button onClick={() => navigate(-1)}
            className="flex-1 border border-border text-ink-secondary py-3.5 rounded-xl font-semibold text-sm">
            ← Back
          </button>
          <button
            onClick={handleGenerate}
            disabled={!adherence || loading}
            className="flex-[2] bg-primary text-white py-3.5 rounded-xl font-semibold text-sm disabled:opacity-40"
          >
            {loading ? 'Running model…' : 'Generate Prediction →'}
          </button>
        </div>
      </PageFooter>
    </div>
  )
}
