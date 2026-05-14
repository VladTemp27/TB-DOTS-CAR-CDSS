import { useEffect, useState } from 'react'
import { useNavigate, useParams, useLocation } from 'react-router-dom'
import { CheckCircle2 } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { getPatient, savePatient } from '../lib/storage'
import { PageFooter } from '../components/PageFooter'
import type { Patient } from '../lib/storage'

type RegimenId = 'hrze' | 'mdr' | 'xdr'

const REGIMENS: Array<{ id: RegimenId; name: string; drugs: string; duration: string; successRate: string }> = [
  {
    id: 'hrze',
    name: 'Drug-Susceptible TB (HRZE)',
    drugs: 'Isoniazid, Rifampicin, Pyrazinamide, Ethambutol',
    duration: '6 months',
    successRate: '~85%',
  },
  {
    id: 'mdr',
    name: 'MDR-TB Regimen',
    drugs: 'Fluoroquinolone-based regimen with 2nd-line drugs',
    duration: '9–12 months',
    successRate: '~60%',
  },
  {
    id: 'xdr',
    name: 'XDR-TB Regimen',
    drugs: 'Newer agents: Bedaquiline, Linezolid, Pretomanid',
    duration: '6–9 months',
    successRate: '~50%',
  },
]

export function TreatmentSelection() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const fromIntake = Boolean((location.state as { fromIntake?: boolean } | null)?.fromIntake)
  const [patient, setPatient] = useState<Patient | null>(null)
  const [loadingPatient, setLoadingPatient] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  const [selected, setSelected] = useState<RegimenId | ''>('')
  const [startDate, setStartDate] = useState('')
  const valid = !!selected && !!startDate

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
        if (cancelled) return
        setPatient(p)
        setSelected(((p?.treatmentRegimen as RegimenId | undefined) ?? ''))
        setStartDate(p?.treatmentStartDate ?? '')
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

  async function handleProceed() {
    if (!id || !valid) return
    const p = patient ?? (await getPatient(id))
    if (p) {
      p.treatmentRegimen = selected || undefined
      p.treatmentStartDate = startDate || undefined
      savePatient(p)
    }
    navigate(`/patient/${id}`)
  }

  async function handleSkip() {
    if (!id) return
    const p = patient ?? (await getPatient(id))
    if (p) {
      p.treatmentRegimen = undefined
      p.treatmentStartDate = startDate || undefined
      await savePatient(p)
    }
    navigate(`/patient/${id}`)
  }

  if (loadingPatient) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg">
        <p className="text-ink-muted">Loading patient…</p>
      </div>
    )
  }

  if (loadError) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg">
        <p className="text-ink-muted">{loadError}</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader />
      <StepProgress steps={5} current={5} />

      <div className="flex-1 flex flex-col">
        <div className="flex-1 px-4 lg:px-8 pb-6 pt-4 w-full max-w-4xl mx-auto space-y-4">

          <div>
            <h2 className="font-bold text-ink-base text-base font-display">Select Treatment Regimen</h2>
            <p className="text-sm text-primary mt-0.5">Based on TB diagnosis and drug sensitivity profile</p>
          </div>

          {/* Regimen cards — 3-col on desktop */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            {REGIMENS.map(r => (
              <button
                key={r.id}
                onClick={() => setSelected(r.id)}
                className={`w-full text-left border-2 rounded-2xl p-4 transition-colors relative
                  ${selected === r.id
                    ? 'border-primary bg-primary-light'
                    : 'border-border bg-surface hover:border-primary/30'}`}
                aria-pressed={selected === r.id}
              >
                {selected === r.id && (
                  <CheckCircle2
                    size={18}
                    className="text-primary absolute top-3 right-3"
                    aria-hidden="true"
                  />
                )}
                <p className="font-semibold text-ink-base text-sm pr-6">{r.name}</p>
                <p className="text-xs text-ink-secondary mt-1.5 leading-relaxed">{r.drugs}</p>
                <div className="flex gap-2 mt-2 flex-wrap">
                  <span className="text-xs text-ink-muted bg-gray-100 rounded-full px-2 py-0.5">
                    {r.duration}
                  </span>
                  <span className="text-xs text-risk-low bg-risk-low/10 rounded-full px-2 py-0.5 font-medium">
                    {r.successRate} success
                  </span>
                </div>
              </button>
            ))}
          </div>

          {/* Date input */}
          <div className="bg-surface border border-border rounded-2xl p-4 lg:p-5">
            <label className="block text-sm font-medium text-ink-secondary mb-2" htmlFor="startDate">
              Treatment Start Date <span className="text-red-500">*</span>
            </label>
            <input
              id="startDate"
              type="date"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              className="w-full min-h-[44px] border border-border rounded-xl px-3 py-2.5 text-sm text-ink-base outline-none focus:border-primary lg:max-w-xs"
            />
          </div>
        </div>
      </div>

      <PageFooter>
        <div className="flex gap-3 items-center">
          <button onClick={() => navigate(-1)}
            className="flex-1 border border-border text-ink-secondary py-3.5 rounded-xl font-semibold text-sm">
            ← Back
          </button>
          {fromIntake && (
            <button onClick={handleSkip}
              className="flex-1 text-ink-secondary py-3.5 rounded-xl font-semibold text-sm hover:bg-gray-50 transition-colors">
              Skip for now
            </button>
          )}
          <button
            onClick={handleProceed}
            disabled={!valid}
            className="flex-[2] bg-primary text-white py-3.5 rounded-xl font-semibold text-sm disabled:opacity-40"
          >
            Proceed to Outcome →
          </button>
        </div>
      </PageFooter>
    </div>
  )
}
