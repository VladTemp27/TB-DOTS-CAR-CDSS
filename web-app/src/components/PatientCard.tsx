import { useNavigate } from 'react-router-dom'
import type { Patient } from '../lib/storage'
import { RiskBadge, riskColor, riskLabel } from './RiskBadge'

interface Props {
  patient: Patient
  monthNumber?: number
}

const riskEdge: Record<string, string> = {
  HIGH: 'border-l-risk-high',
  MED:  'border-l-risk-med',
  LOW:  'border-l-risk-low',
}

export function PatientCard({ patient, monthNumber }: Props) {
  const navigate = useNavigate()
  const latest = patient.predictions.at(-1)
  const prob = latest?.failureProbability ?? 0
  const { text } = riskColor(prob)
  const label = riskLabel(prob)

  return (
    <button
      onClick={() => navigate(`/patient/${patient.id}/chart`)}
      className={`w-full bg-surface rounded-xl border border-border border-l-4 ${riskEdge[label]} p-4 text-left
        shadow-sm hover:shadow-md hover:border-primary/30 transition-shadow duration-150 active:bg-gray-50`}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="font-semibold text-ink-base">{patient.name}</p>
          <p className="text-xs text-ink-secondary mt-0.5">
            {patient.medicalId}
            {monthNumber != null && ` · Month ${monthNumber} of 6`}
          </p>
        </div>
        <div className="text-right">
          <p className={`text-2xl font-bold font-display tabular-nums ${text}`}>{Math.round(prob * 100)}%</p>
          <RiskBadge probability={prob} size="sm" />
        </div>
      </div>
      <div className="mt-3 w-full h-1.5 rounded-full bg-gray-100">
        <div
          className={`h-full rounded-full transition-all ${label === 'HIGH' ? 'bg-risk-high' : label === 'MED' ? 'bg-risk-med' : 'bg-risk-low'}`}
          style={{ width: `${Math.round(prob * 100)}%` }}
        />
      </div>
    </button>
  )
}
