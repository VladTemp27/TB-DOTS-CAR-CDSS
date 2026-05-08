import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Stethoscope, UserPlus } from 'lucide-react'
import { getAllPatients } from '../lib/storage'
import { AppHeader } from '../components/AppHeader'
import { PatientCard } from '../components/PatientCard'
import { riskLabel } from '../components/RiskBadge'

export function Home() {
  const navigate = useNavigate()
  const patients = getAllPatients()
  const [query, setQuery] = useState('')

  const filtered = query.trim()
    ? patients.filter(p =>
        p.name.toLowerCase().includes(query.toLowerCase()) ||
        p.medicalId.toLowerCase().includes(query.toLowerCase())
      )
    : patients

  const high = patients.filter(p => riskLabel(p.predictions.at(-1)?.failureProbability ?? 0) === 'HIGH').length
  const dueSoon = patients.filter(p => p.monthlyRecords.length < 6).length

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader />

      <div className="flex-1 px-4 lg:px-8 pt-4 pb-24 lg:pb-8 w-full max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-xl font-bold text-ink-base font-display">Patient Overview</h1>
            {patients.length > 0 && (
              <p className="text-sm text-ink-secondary mt-0.5">
                {patients.length} patient{patients.length !== 1 ? 's' : ''} registered
              </p>
            )}
          </div>

          {patients.length > 0 && (
            <div className="flex gap-2 flex-wrap justify-end">
              {high > 0 && (
                <span className="text-xs bg-risk-high/10 text-risk-high font-semibold px-2.5 py-1 rounded-full">
                  {high} High Risk
                </span>
              )}
              {dueSoon > 0 && (
                <span className="text-xs bg-risk-med/10 text-risk-med font-semibold px-2.5 py-1 rounded-full">
                  {dueSoon} Due Check-In
                </span>
              )}
            </div>
          )}
        </div>

        <div className="sticky top-[60px] lg:top-0 z-10 bg-bg pt-1 pb-3 mb-2 -mx-4 px-4 lg:-mx-8 lg:px-8">
          <div className="flex gap-3 items-center">
            <div className="relative flex-1">
              <input
                type="search"
                placeholder="Search by name or Medical ID…"
                value={query}
                onChange={e => setQuery(e.target.value)}
                className="w-full min-h-[44px] bg-surface border border-border rounded-xl px-4 py-2.5 text-sm text-ink-base placeholder-ink-muted focus:border-primary outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-1"
                aria-label="Search patients by name or Medical ID"
              />
            </div>
            <button
              onClick={() => navigate('/patient/new')}
              className="hidden lg:inline-flex items-center gap-2 bg-primary text-white px-5 py-2.5 rounded-xl font-semibold text-sm whitespace-nowrap min-h-[44px] active:bg-primary-dark hover:bg-primary-mid transition-colors"
            >
              <UserPlus size={16} aria-hidden="true" />
              New Patient
            </button>
          </div>
        </div>

        {patients.length === 0 ? (
          <div className="text-center py-20 text-ink-muted">
            <Stethoscope size={56} className="mx-auto mb-4 text-ink-muted/40" aria-hidden="true" />
            <p className="font-semibold text-ink-secondary text-base">No patients yet</p>
            <p className="text-sm mt-1">Use the New button to register a patient</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-20 text-ink-muted">
            <p className="font-semibold text-ink-secondary text-base">No patients found</p>
            <p className="text-sm mt-1">Try a different name or Medical ID</p>
          </div>
        ) : (
          <>
            <p className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-3">
              {query.trim() ? `${filtered.length} result${filtered.length !== 1 ? 's' : ''}` : 'Sorted by Risk'}
            </p>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
              {filtered.map((p) => (
                <PatientCard key={p.id} patient={p} monthNumber={p.monthlyRecords.length} />
              ))}
            </div>
          </>
        )}

      </div>
    </div>
  )
}