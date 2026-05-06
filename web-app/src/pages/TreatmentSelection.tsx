import { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { getPatient, savePatient } from '../lib/storage'
import { PageFooter } from '../components/PageFooter'

const REGIMENS = [
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

  const [selected, setSelected] = useState(patient?.treatmentRegimen ?? '')
  const [startDate, setStartDate] = useState(patient?.treatmentStartDate ?? '')

  function handleProceed() {
    if (!id || !selected) return
    const p = getPatient(id)
    if (p) {
      p.treatmentRegimen = selected
      p.treatmentStartDate = startDate
      savePatient(p)
    }
    navigate(`/patient/${id}`)
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <AppHeader />
      <StepProgress steps={4} current={4} />

      <main className="flex-1 px-4 pb-6 space-y-4 pt-2">
        <div>
          <h2 className="font-bold text-gray-900 text-base">Select Treatment Regimen</h2>
          <p className="text-sm text-primary mt-0.5">Based on TB diagnosis and drug sensitivity profile</p>
        </div>

        <div className="space-y-3">
          {REGIMENS.map(r => (
            <button
              key={r.id}
              onClick={() => setSelected(r.id)}
              className={`w-full text-left bg-white border-2 rounded-xl p-4 transition-colors
                ${selected === r.id ? 'border-primary bg-primary-light' : 'border-gray-100'}`}
            >
              <p className="font-semibold text-gray-900 text-sm">{r.name}</p>
              <p className="text-xs text-gray-500 mt-1">{r.drugs}</p>
              <p className="text-xs text-gray-400 mt-1">Duration: {r.duration} | Success Rate: {r.successRate}</p>
            </button>
          ))}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Treatment Start Date *</label>
          <input
            type="date"
            value={startDate}
            onChange={e => setStartDate(e.target.value)}
            className="w-full border border-gray-200 rounded-lg px-3 py-2.5 text-sm outline-none focus:border-primary"
          />
        </div>
      </main>

      <PageFooter>
        <div className="flex gap-3">
          <button onClick={() => navigate(-1)}
            className="flex-1 border border-gray-200 text-gray-700 py-3.5 rounded-xl font-semibold text-sm">
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
