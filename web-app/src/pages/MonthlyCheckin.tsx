import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Check, AlertTriangle, X } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { getChoices, predictWithContributions } from '../lib/inference'
import { getPatient, addMonthlyRecord, addPrediction } from '../lib/storage'
import { PageFooter } from '../components/PageFooter'

type Adherence = 'full' | 'partial' | 'poor'

export function MonthlyCheckin() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const patient = id ? getPatient(id) : null

  const [weight, setWeight] = useState('')
  const [smearResult, setSmearResult] = useState('')
  const [adherence, setAdherence] = useState<Adherence | ''>('')
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

      addMonthlyRecord(id, {
        month: monthNumber,
        weight: weight ? Number(weight) : undefined,
        smearResult: smearResult || undefined,
        adherence,
        failureProbability: result.failureProbability,
        timestamp: Date.now(),
      })

      addPrediction(id, {
        label: result.label,
        failureProbability: result.failureProbability,
        contributions: result.contributions,
        timestamp: Date.now(),
      })

      navigate(`/patient/${id}/risk-update`, { state: { result, monthNumber } })
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
                  placeholder="e.g. 52.5"
                  value={weight}
                  onChange={e => setWeight(e.target.value)}
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

          <div className="bg-blue-50 border border-blue-100 rounded-2xl p-3 text-xs text-blue-700">
            Follow-up Guide: Record weight and smear at each monthly visit. Adherence gaps are a top predictor of treatment failure.
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
