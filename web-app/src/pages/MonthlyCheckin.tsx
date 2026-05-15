import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { XrayUploadField } from '../components/XrayUploadField'
import { predictWithContributions } from '../lib/inference'
import { getPatient, addMonthlyRecord, addPrediction, attachMonthlyXrays } from '../lib/storage'
import { PageFooter } from '../components/PageFooter'
import type { Patient } from '../lib/storage'

type Adherence = 'full' | 'partial' | 'poor'

function computeAdherence(pct?: number): Adherence {
  if (pct == null) return 'poor'
  if (pct >= 90) return 'full'
  if (pct >= 50) return 'partial'
  return 'poor'
}

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
    return () => { cancelled = true }
  }, [id])

  const [weight, setWeight] = useState('')
  const [height, setHeight] = useState('')
  const [smearTbLamp, setSmearTbLamp] = useState('')
  const [xpertMtbRif, setXpertMtbRif] = useState('')
  const [monthlyDosesTaken, setMonthlyDosesTaken] = useState('')
  const [monthlyMissedDoses, setMonthlyMissedDoses] = useState('')
  const [cumulativeDosesTaken, setCumulativeDosesTaken] = useState('')
  const [pctAdherence, setPctAdherence] = useState('')
  const [xrayFiles, setXrayFiles] = useState<File[]>([])
  const [loading, setLoading] = useState(false)

  const monthCount = patient?.monthlyRecords.length ?? 0
  const isM0 = monthCount === 0
  const isAllDone = monthCount >= 13
  const priorCumulative = (patient?.monthlyRecords ?? []).reduce(
    (sum, r) => sum + (r.monthlyDosesTaken ?? 0), 0
  )
  const monthLabel = isM0 ? 'Baseline (M0)' : `Month M${monthCount}`

  useEffect(() => {
    if (isM0) {
      setMonthlyDosesTaken('0')
      setMonthlyMissedDoses('0')
      setCumulativeDosesTaken('0')
      setPctAdherence('')
    }
  }, [isM0])

  useEffect(() => {
    if (isM0) return
    const takenNum = parseInt(monthlyDosesTaken, 10)
    const missedNum = parseInt(monthlyMissedDoses, 10)
    const taken = Number.isFinite(takenNum) ? takenNum : 0
    const missed = Number.isFinite(missedNum) ? missedNum : 0
    const total = taken + missed

    setCumulativeDosesTaken(String(priorCumulative + taken))

    if (total > 0) {
      const pct = Math.min(100, Math.max(0, (taken / total) * 100))
      setPctAdherence(pct.toFixed(1))
    } else {
      setPctAdherence('')
    }
  }, [isM0, monthlyDosesTaken, monthlyMissedDoses, priorCumulative])

  async function handleGenerate() {
    if (!patient || !id) return
    setLoading(true)

    const parsedWeight = parseFloat(weight)
    const parsedHeight = parseFloat(height)

    if (!Number.isFinite(parsedWeight) || parsedWeight <= 0) {
      alert('Weight is required and must be greater than 0.')
      setLoading(false)
      return
    }
    if (!Number.isFinite(parsedHeight) || parsedHeight <= 0) {
      alert('Height is required and must be greater than 0.')
      setLoading(false)
      return
    }
    if (!smearTbLamp && !xpertMtbRif) {
      alert('At least one lab result (Smear TB LAMP or Xpert MTB/RIF) is required.')
      setLoading(false)
      return
    }

    const isMissingWeight = (!weight || parseFloat(weight) <= 0) ? 1 : 0
    const isMissingHeight = (!height || parseFloat(height) <= 0) ? 1 : 0
    const bothLabsBlank = !smearTbLamp && !xpertMtbRif
    const isMissingSmearTbLamp = bothLabsBlank ? 1 : 0
    const isMissingXpertMtbRif = bothLabsBlank ? 1 : 0
    const isMissingDosesTaken = isM0 ? 0 : (!monthlyDosesTaken ? 1 : 0)
    const isMissingMissedDoses = isM0 ? 0 : (!monthlyMissedDoses ? 1 : 0)

    const parsedSmearTbLamp = smearTbLamp === '1' ? 1 : smearTbLamp === '0' ? 0 : undefined
    const parsedXpertMtbRif = xpertMtbRif === '1' ? 1 : xpertMtbRif === '0' ? 0 : undefined
    const parsedMonthlyDoses = isM0 ? 0 : parseInt(monthlyDosesTaken, 10)
    const parsedMonthlyMissed = isM0 ? 0 : parseInt(monthlyMissedDoses, 10)
    const parsedCumulativeDoses = isM0 ? 0 : parseInt(cumulativeDosesTaken, 10)
    const parsedPctAdherence = pctAdherence ? parseFloat(pctAdherence) : undefined
    const adherenceCategory = computeAdherence(parsedPctAdherence)

    const updatedFeatures = { ...patient.features }
    if (parsedSmearTbLamp !== undefined) updatedFeatures.microscopyResult = String(parsedSmearTbLamp)

    try {
      const result = await predictWithContributions(updatedFeatures)
      const now = Date.now()

      await addMonthlyRecord(id, {
        month: monthCount,
        weight: parsedWeight,
        height: parsedHeight,
        smearTbLamp: parsedSmearTbLamp as 0 | 1 | undefined,
        xpertMtbRif: parsedXpertMtbRif as 0 | 1 | undefined,
        monthlyDosesTaken: Number.isFinite(parsedMonthlyDoses) ? parsedMonthlyDoses : undefined,
        monthlyMissedDoses: Number.isFinite(parsedMonthlyMissed) ? parsedMonthlyMissed : undefined,
        cumulativeDosesTaken: Number.isFinite(parsedCumulativeDoses) ? parsedCumulativeDoses : undefined,
        pctAdherence: parsedPctAdherence,
        adherence: adherenceCategory,
        failureProbability: result.failureProbability,
        timestamp: now,
        isMissingWeight: isMissingWeight as 0 | 1,
        isMissingHeight: isMissingHeight as 0 | 1,
        isMissingSmearTbLamp: isMissingSmearTbLamp as 0 | 1,
        isMissingXpertMtbRif: isMissingXpertMtbRif as 0 | 1,
        isMissingMonthlyDosesTaken: isMissingDosesTaken as 0 | 1,
        isMissingMonthlyMissedDoses: isMissingMissedDoses as 0 | 1,
      })

      if (xrayFiles.length > 0) {
        await attachMonthlyXrays(id, monthCount, xrayFiles)
      }

      await addPrediction(id, {
        label: result.label,
        failureProbability: result.failureProbability,
        contributions: result.contributions,
        featuresUsed: updatedFeatures,
        timestamp: now,
      })

      navigate(`/patient/${id}/risk-update`, { state: { result, monthNumber: monthCount }, replace: true })
    } catch (err) {
      console.error(err)
      alert('Inference failed.')
    } finally {
      setLoading(false)
    }
  }

  const inputCls = 'w-full min-h-[44px] border border-border rounded-xl px-3 py-2.5 text-sm text-ink-base bg-surface placeholder-ink-muted focus:border-primary outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1'
  const readOnlyCls = inputCls + ' bg-gray-50 cursor-not-allowed'

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

  if (!patient.treatmentStartDate) {
    return (
      <div className="min-h-screen bg-bg flex flex-col">
        <AppHeader backLabel="Back" onBack={() => navigate(-1)} />
        <div className="flex-1 px-4 lg:px-8 pt-10 pb-6 w-full max-w-3xl mx-auto">
          <p className="text-sm text-ink-muted">A treatment start date must be set before recording monthly check-ins.</p>
        </div>
      </div>
    )
  }

  if (isAllDone) {
    return (
      <div className="min-h-screen bg-bg flex flex-col">
        <AppHeader backLabel="Back" onBack={() => navigate(-1)} />
        <div className="flex-1 px-4 lg:px-8 pt-10 pb-6 w-full max-w-3xl mx-auto">
          <p className="text-sm text-ink-muted">All monthly check-ins have been recorded.</p>
        </div>
      </div>
    )
  }

  const adherencePct = pctAdherence ? parseFloat(pctAdherence) : undefined
  const adherenceValue = computeAdherence(adherencePct)
  const adhColor = adherenceValue === 'full' ? 'text-risk-low bg-risk-low/10 border-risk-low'
    : adherenceValue === 'partial' ? 'text-risk-med bg-risk-med/10 border-risk-med'
    : 'text-risk-high bg-risk-high/10 border-risk-high'

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader backLabel="Back" onBack={() => navigate(-1)} />

      <div className="flex-1 flex flex-col">
        <div className="flex-1 px-4 lg:px-8 pb-6 pt-4 w-full max-w-3xl mx-auto space-y-5">

          <div>
            <div className="h-1.5 bg-gray-100 rounded-full">
              <div className="h-full bg-primary rounded-full transition-all"
                style={{ width: `${Math.min((monthCount / 12) * 100, 100)}%` }} />
            </div>
            <p className="text-xs text-ink-muted mt-1">{monthLabel} of 12</p>
          </div>

          <div>
            <h2 className="font-bold text-ink-base text-lg font-display">Monthly Follow-up</h2>
            <p className="text-sm text-ink-secondary">Recording {monthLabel} data for {patient?.name}</p>
          </div>

          {/* Measurements */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-5 space-y-3">
            <h3 className="text-sm font-semibold text-ink-base">Measurements</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="weight">
                  Weight (kg)
                </label>
                <input
                  id="weight"
                  type="number"
                  step="0.1"
                  min="0"
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
                  min="0"
                  placeholder="e.g. 162.5"
                  value={height}
                  onChange={e => setHeight(e.target.value)}
                  className={inputCls}
                />
              </div>
            </div>
          </div>

          {/* Dose Tracking */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-5 space-y-3">
            <h3 className="text-sm font-semibold text-ink-base">Dose Tracking</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="monthlyDosesTaken">
                  Doses Taken
                </label>
                <input
                  id="monthlyDosesTaken"
                  type="number"
                  min="0"
                  placeholder={isM0 ? '0 (Baseline)' : 'e.g. 28'}
                  value={monthlyDosesTaken}
                  readOnly={isM0}
                  onChange={e => setMonthlyDosesTaken(e.target.value)}
                  className={isM0 ? readOnlyCls : inputCls}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-ink-secondary mb-1" htmlFor="monthlyMissedDoses">
                  Missed Doses
                </label>
                <input
                  id="monthlyMissedDoses"
                  type="number"
                  min="0"
                  placeholder={isM0 ? '0 (Baseline)' : 'e.g. 2'}
                  value={monthlyMissedDoses}
                  readOnly={isM0}
                  onChange={e => setMonthlyMissedDoses(e.target.value)}
                  className={isM0 ? readOnlyCls : inputCls}
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
                  placeholder="—"
                  value={cumulativeDosesTaken}
                  readOnly
                  className={readOnlyCls}
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
                  placeholder={isM0 ? '—' : 'Auto-calculated'}
                  value={pctAdherence}
                  readOnly
                  className={readOnlyCls}
                />
              </div>
            </div>
            {!isM0 && pctAdherence && (
              <div className="flex items-center gap-2 pt-1">
                <span className="text-xs text-ink-muted">Adherence:</span>
                <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${adhColor}`}>
                  {adherenceValue}
                </span>
              </div>
            )}
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

          {/* Chest X-rays — optional */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-5 space-y-3">
            <p className="text-sm font-medium text-ink-secondary">Chest X-rays (optional)</p>
            <XrayUploadField onChange={setXrayFiles} />
          </div>

          <div className="bg-blue-50 border border-blue-100 rounded-2xl p-3 text-xs text-blue-700">
            Follow-up Guide: Record weight, height, dose tracking, and lab results at each monthly visit.
            {isM0 && ' At least one lab result (TB LAMP or Xpert) is required for baseline.'}
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
            disabled={loading}
            className="flex-[2] bg-primary text-white py-3.5 rounded-xl font-semibold text-sm disabled:opacity-40"
          >
            {loading ? 'Running model…' : 'Generate Prediction →'}
          </button>
        </div>
      </PageFooter>
    </div>
  )
}
