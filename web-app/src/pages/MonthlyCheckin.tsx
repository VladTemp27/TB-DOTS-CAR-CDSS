import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { getChoices, predictWithContributions } from '../lib/inference'
import { getPatient, savePatient, addMonthlyRecord, addPrediction } from '../lib/storage'
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

  const microChoices = getChoices('Microscopy_Result')
  const monthNumber = (patient?.monthlyRecords.length ?? 0) + 1

  async function handleGenerate() {
    if (!patient || !id || !adherence) return
    setLoading(true)

    // Update features with new smear result if provided
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

  const adherenceOptions: { value: Adherence; label: string; sublabel: string; icon: string; color: string }[] = [
    { value: 'full', label: 'Full', sublabel: 'All doses taken', icon: '✓', color: 'border-green-400 bg-green-50 text-green-700' },
    { value: 'partial', label: 'Partial', sublabel: 'Some missed', icon: '⚠', color: 'border-orange-400 bg-orange-50 text-orange-700' },
    { value: 'poor', label: 'Poor', sublabel: 'Most missed', icon: '✗', color: 'border-red-400 bg-red-50 text-red-700' },
  ]

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <AppHeader backLabel="Back" onBack={() => navigate(-1)} />

      <div className="px-4 pt-3 pb-2">
        <div className="h-1.5 bg-gray-100 rounded-full">
          <div className="h-full bg-primary rounded-full" style={{ width: `${((monthNumber - 1) / 6) * 100}%` }} />
        </div>
        <p className="text-xs text-gray-500 mt-1">Monthly Check-in</p>
      </div>

      <main className="flex-1 px-4 pb-6 space-y-5">
        <div>
          <h2 className="font-bold text-gray-900 text-lg">Monthly Follow-up</h2>
          <p className="text-sm text-gray-500">Recording Month {monthNumber} data for {patient?.name}</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Current Weight (kg)</label>
          <input
            type="number"
            placeholder="e.g. 52.5"
            value={weight}
            onChange={e => setWeight(e.target.value)}
            className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Sputum Smear Result</label>
          <select
            value={smearResult}
            onChange={e => setSmearResult(e.target.value)}
            className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary bg-white"
          >
            <option value="">Not performed</option>
            {microChoices.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Adherence This Month *</label>
          <div className="flex gap-3">
            {adherenceOptions.map(opt => (
              <button
                key={opt.value}
                onClick={() => setAdherence(opt.value)}
                className={`flex-1 border-2 rounded-xl p-3 text-center transition-colors
                  ${adherence === opt.value ? opt.color + ' border-2' : 'border-gray-200 bg-white'}`}
              >
                <p className="text-lg">{opt.icon}</p>
                <p className="text-xs font-bold mt-1">{opt.label}</p>
                <p className="text-xs text-current opacity-70">{opt.sublabel}</p>
              </button>
            ))}
          </div>
        </div>

        <div className="bg-blue-50 border border-blue-100 rounded-xl p-3 text-xs text-blue-700">
          Follow-up Guide: Record weight and smear at each monthly visit. Adherence gaps are a top predictor of treatment failure.
        </div>
      </main>

      <PageFooter>
        <div className="flex gap-3">
          <button onClick={() => navigate(-1)}
            className="flex-1 border border-gray-200 text-gray-700 py-3.5 rounded-xl font-semibold text-sm">
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
