import { useNavigate, useParams } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { RiskBadge, riskColor } from '../components/RiskBadge'
import { getPatient } from '../lib/storage'
import { PageFooter } from '../components/PageFooter'

export function PatientProfile() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const patient = id ? getPatient(id) : null

  if (!patient) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-500">Patient not found.</p>
      </div>
    )
  }

  const latestProb = patient.predictions.at(-1)?.failureProbability ?? 0
  const prevProb = patient.predictions.at(-2)?.failureProbability
  const delta = prevProb != null ? latestProb - prevProb : null
  const { text } = riskColor(latestProb)

  const regimen = patient.treatmentRegimen === 'hrze' ? 'HRZE Regimen'
    : patient.treatmentRegimen === 'mdr' ? 'MDR-TB Regimen'
    : patient.treatmentRegimen === 'xdr' ? 'XDR-TB Regimen'
    : 'No Regimen Selected'

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <AppHeader backLabel="Back to All Patients" onBack={() => navigate('/')} />

      <main className="flex-1 px-4 pb-6 pt-4 space-y-4">
        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
          <p className="font-bold text-gray-900 text-lg">{patient.name}</p>
          <p className="text-sm text-gray-500">
            {patient.features.sex === 'M' ? 'Male' : 'Female'}, {patient.features.age} yrs | {regimen} | Started {patient.treatmentStartDate || 'N/A'}
          </p>
          <p className="text-xs text-gray-400 mt-1">{patient.medicalId}</p>
        </div>

        <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
          <p className={`text-5xl font-extrabold ${text}`}>{Math.round(latestProb * 100)}%</p>
          <div className="flex items-center gap-2 mt-1">
            <p className="text-sm font-semibold text-gray-600">FAILURE RISK</p>
            <RiskBadge probability={latestProb} />
          </div>
          {delta != null && (
            <p className={`text-sm mt-1 ${delta > 0 ? 'text-red-500' : 'text-green-500'}`}>
              {delta > 0 ? '↑' : '↓'} {Math.abs(Math.round(delta * 100))}% from last month
            </p>
          )}

          <div className="mt-3">
            <p className="text-sm font-semibold text-gray-700 mb-1">Top Risk Factors:</p>
            {patient.predictions.at(-1)?.contributions.slice(0, 3).map((c: { feature: string }) => (
              <p key={c.feature} className="text-sm text-gray-600">• {c.feature}</p>
            ))}
            <button
              onClick={() => navigate(`/patient/${id}/features`)}
              className="text-primary text-sm font-medium mt-2"
            >
              See full breakdown →
            </button>
          </div>
        </div>

        {patient.predictions.length > 1 && (
          <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
            <p className="text-sm font-semibold text-gray-800 mb-3">Risk Trend Over Time</p>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {patient.predictions.map((pr, i) => {
                const p = Math.round(pr.failureProbability * 100)
                const isLast = i === patient.predictions.length - 1
                return (
                  <div key={i} className={`flex flex-col items-center min-w-[52px] border rounded-lg p-2
                    ${isLast ? 'border-red-300 bg-red-50' : 'border-gray-100 bg-gray-50'}`}>
                    <p className="text-xs text-gray-500">M{i}</p>
                    <p className={`text-sm font-bold ${riskColor(pr.failureProbability).text}`}>{p}%</p>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {patient.monthlyRecords.length > 0 && (
          <div className="bg-white border border-gray-100 rounded-xl p-4 shadow-sm">
            <p className="text-sm font-semibold text-gray-800 mb-3">Monthly Records</p>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-gray-400 border-b border-gray-100">
                  <th className="text-left pb-2">Month</th>
                  <th className="text-left pb-2">Weight</th>
                  <th className="text-left pb-2">Smear</th>
                  <th className="text-left pb-2">Adherence</th>
                  <th className="text-right pb-2">Risk</th>
                </tr>
              </thead>
              <tbody>
                {patient.monthlyRecords.map((r, i) => (
                  <tr key={i} className="border-b border-gray-50">
                    <td className="py-2 text-gray-700">M{r.month}</td>
                    <td className="py-2 text-gray-700">{r.weight ? `${r.weight} kg` : '—'}</td>
                    <td className="py-2 text-gray-700">{r.smearResult || '—'}</td>
                    <td className={`py-2 capitalize font-medium ${r.adherence === 'full' ? 'text-green-600' : r.adherence === 'partial' ? 'text-orange-500' : 'text-red-500'}`}>
                      {r.adherence}
                    </td>
                    <td className="py-2 text-right text-gray-700">{Math.round(r.failureProbability * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>

      <PageFooter>
        <button
          onClick={() => navigate(`/patient/${id}/checkin`)}
          className="w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm active:bg-primary-dark"
        >
          Log Monthly Update →
        </button>
      </PageFooter>
    </div>
  )
}
