import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { CheckCircle2 } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { getPatient, savePatient } from '../lib/storage'
import { PageFooter } from '../components/PageFooter'

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
  const patient = id ? getPatient(id) : null

  const [selected, setSelected] = useState<RegimenId | ''>(patient?.treatmentRegimen ?? '')
  const [startDate, setStartDate] = useState(patient?.treatmentStartDate ?? '')

  function handleProceed() {
    if (!id || !selected) return
    const p = getPatient(id)
    if (p) {
      p.treatmentRegimen = selected || undefined
      p.treatmentStartDate = startDate
      savePatient(p)
    }
    navigate(`/patient/${id}`)
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader />
      <StepProgress steps={4} current={4} />

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
              Treatment Start Date *
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
        <div className="flex gap-3">
          <button onClick={() => navigate(-1)}
            className="flex-1 border border-border text-ink-secondary py-3.5 rounded-xl font-semibold text-sm">
            ← Back
          </button>
          <button
            onClick={handleProceed}
            disabled={!selected}
            className="flex-[2] bg-primary text-white py-3.5 rounded-xl font-semibold text-sm disabled:opacity-40"
          >
            Proceed to Outcome →
          </button>
        </div>
      </PageFooter>
    </div>
  )
}
