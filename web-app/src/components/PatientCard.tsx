import { useNavigate } from 'react-router-dom'
import type { Patient } from '../lib/storage'
import { RiskBadge, riskColor } from './RiskBadge'

interface Props {
  patient: Patient
  monthNumber?: number
}

export function PatientCard({ patient, monthNumber }: Props) {
  const navigate = useNavigate()
  const latest = patient.predictions.at(-1)
  const prob = latest?.failureProbability ?? 0
  const { text } = riskColor(prob)

  return (
    <button
      onClick={() => navigate(`/patient/${patient.id}`)}
      className="w-full bg-white rounded-xl border border-gray-100 p-4 text-left shadow-sm active:bg-gray-50"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="font-semibold text-gray-900">{patient.name}</p>
          <p className="text-xs text-gray-500 mt-0.5">
            {patient.medicalId}
            {monthNumber != null && ` · Month ${monthNumber} of 6`}
          </p>
        </div>
        <div className="text-right">
          <p className={`text-2xl font-bold ${text}`}>{Math.round(prob * 100)}%</p>
          <RiskBadge probability={prob} size="sm" />
        </div>
      </div>
      {prob >= 0.5 && (
        <div className={`mt-2 w-full h-1 rounded-full bg-gray-100`}>
          <div className="h-full rounded-full bg-red-400" style={{ width: `${Math.round(prob * 100)}%` }} />
        </div>
      )}
    </button>
  )
}
