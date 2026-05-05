import { useNavigate, useParams, useLocation } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { RiskBadge, riskColor } from '../components/RiskBadge'
import { getPatient } from '../lib/storage'
import type { ContributionResult } from '../lib/inference'

export function RiskUpdate() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const patient = id ? getPatient(id) : null
  const result: ContributionResult | null = location.state?.result ?? null
  const monthNumber: number = location.state?.monthNumber ?? 1

  const prob = result?.failureProbability ?? 0
  const prevProb = patient?.predictions.at(-2)?.failureProbability
  const delta = prevProb != null ? prob - prevProb : null
  const isHigh = prob >= 0.5
  const { text } = riskColor(prob)

  const topContribs = result?.contributions.slice(0, 3) ?? []

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <AppHeader />

      <main className="flex-1 px-4 pb-24 pt-4 space-y-4">
        <div className="flex justify-between items-start">
          <div>
            <p className="font-bold text-gray-900">{patient?.name}</p>
            <p className="text-xs text-gray-500">{patient?.medicalId} · Month {monthNumber}</p>
            <p className="text-xs text-gray-400">{patient?.treatmentRegimen?.toUpperCase() || ''} · {monthNumber} of 6 months</p>
          </div>
        </div>

        <div className={`rounded-xl border-2 p-4 ${isHigh ? 'bg-red-50 border-red-300' : 'bg-green-50 border-green-200'}`}>
          <p className={`font-bold text-base ${isHigh ? 'text-red-700' : 'text-green-700'}`}>
            {isHigh ? 'HIGH RISK' : 'LOW RISK'} — {Math.round(prob * 100)}% Failure Probability
          </p>
          {delta != null && (
            <p className={`text-sm mt-1 ${delta > 0 ? 'text-red-600' : 'text-green-600'}`}>
              {delta > 0 ? '↑' : '↓'} {Math.abs(Math.round(delta * 100))}% {delta > 0 ? 'increased' : 'decreased'} since last month
            </p>
          )}
          <p className="text-sm font-semibold text-gray-800 mt-3 mb-1">Top Risk Factors:</p>
          {topContribs.map(c => (
            <p key={c.feature} className={`text-sm ${c.direction === 'risk' ? 'text-red-600' : 'text-green-600'}`}>
              • {c.feature} — {c.direction === 'risk' ? 'strongest contributor' : 'protective factor'}
            </p>
          ))}
          {id && (
            <button onClick={() => navigate(`/patient/${id}/features`)}
              className="text-primary text-sm font-medium mt-2">
              See full breakdown →
            </button>
          )}
        </div>

        {isHigh && (
          <div className="bg-blue-50 border border-blue-100 rounded-xl p-3 flex gap-2">
            <span className="text-blue-500">ℹ</span>
            <p className="text-xs text-blue-700">
              Clinical Note: Patient shows increasing failure risk. Consider enhanced monitoring and adherence support intervention.
            </p>
          </div>
        )}
      </main>

      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-100 px-4 py-4 flex gap-3">
        <button onClick={() => navigate(-1)}
          className="flex-1 border border-gray-200 text-gray-700 py-3.5 rounded-xl font-semibold text-sm">
          ← Back
        </button>
        <button onClick={() => navigate(`/patient/${id}`)}
          className="flex-[2] bg-primary text-white py-3.5 rounded-xl font-semibold text-sm">
          View Patient Profile
        </button>
      </div>
    </div>
  )
}
