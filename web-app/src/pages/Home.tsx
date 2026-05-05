import { useNavigate } from 'react-router-dom'
import { getAllPatients } from '../lib/storage'
import { AppHeader } from '../components/AppHeader'
import { PatientCard } from '../components/PatientCard'
import { riskLabel } from '../components/RiskBadge'

export function Home() {
  const navigate = useNavigate()
  const patients = getAllPatients()

  const high = patients.filter(p => riskLabel(p.predictions.at(-1)?.failureProbability ?? 0) === 'HIGH').length
  const dueSoon = patients.filter(p => p.monthlyRecords.length < 6).length

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <AppHeader />

      <main className="flex-1 px-4 pt-4 pb-24">
        <h2 className="text-lg font-bold text-gray-900 mb-1">Patient Overview</h2>

        {patients.length > 0 && (
          <div className="flex gap-2 mb-4">
            <span className="text-sm text-gray-600">Total: {patients.length}</span>
            {high > 0 && (
              <span className="text-sm bg-red-100 text-red-600 font-semibold px-2 py-0.5 rounded-full">
                High Risk: {high}
              </span>
            )}
            {dueSoon > 0 && (
              <span className="text-sm bg-yellow-100 text-yellow-700 font-semibold px-2 py-0.5 rounded-full">
                Due Check-In: {dueSoon}
              </span>
            )}
          </div>
        )}

        <div className="relative mb-4">
          <input
            type="search"
            placeholder="Search by name or Medical ID"
            className="w-full bg-white border border-gray-200 rounded-xl px-4 py-2.5 text-sm text-gray-700 placeholder-gray-400 outline-none"
            readOnly
          />
        </div>

        {patients.length === 0 ? (
          <div className="text-center py-16 text-gray-400">
            <p className="text-4xl mb-3">🫁</p>
            <p className="font-medium text-gray-500">No patients yet</p>
            <p className="text-sm mt-1">Tap + to add a new patient</p>
          </div>
        ) : (
          <>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              PATIENTS — SORTED BY RISK
            </p>
            <div className="space-y-3">
              {patients.map((p, i) => (
                <PatientCard key={p.id} patient={p} monthNumber={i + 1} />
              ))}
            </div>
          </>
        )}
      </main>

      <div className="fixed bottom-6 right-6">
        <button
          onClick={() => navigate('/patient/new')}
          className="w-14 h-14 bg-primary rounded-full shadow-lg flex items-center justify-center text-white text-2xl font-light active:bg-primary-dark"
        >
          +
        </button>
      </div>
    </div>
  )
}
