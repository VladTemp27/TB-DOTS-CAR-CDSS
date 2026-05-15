import { useEffect, useState } from 'react'
import { useNavigate, useParams, useLocation } from 'react-router-dom'
import { ArrowUp, ArrowDown, Info } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { RiskBadge, riskColor } from '../components/RiskBadge'
import { getPatient } from '../lib/storage'
import type { ContributionResult } from '../lib/inference'
import { PageFooter } from '../components/PageFooter'
import type { Patient } from '../lib/storage'

function formatPercent(probability: number): string {
  const value = probability * 100
  if (value > 0 && value < 1) return '<1%'
  return `${Math.round(value)}%`
}

function formatDelta(delta: number): string {
  const value = Math.abs(delta * 100)
  if (value > 0 && value < 1) return '<1%'
  return `${Math.round(value)}%`
}

export function RiskUpdate() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const [patient, setPatient] = useState<Patient | null>(null)
  const [loadingPatient, setLoadingPatient] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const result: ContributionResult | null = location.state?.result ?? null
  const monthNumber: number = location.state?.monthNumber ?? 1

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
        if (!cancelled) setPatient(p)
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

  const latestPrediction = patient?.predictions.at(-1)
  const prob = result?.failureProbability ?? latestPrediction?.failureProbability ?? 0
  const derivedMonthNumber = monthNumber ?? patient?.monthlyRecords.at(-1)?.month ?? 1
  const previousMonthlyRecord = patient
    ? [...patient.monthlyRecords]
      .filter(record => record.month < derivedMonthNumber)
      .sort((a, b) => b.month - a.month)[0]
    : undefined
  const prevProb = previousMonthlyRecord?.failureProbability
  const delta = prevProb != null ? prob - prevProb : null
  const isHigh   = prob >= 0.5
  const { text } = riskColor(prob)

  const topContribs = (result?.contributions ?? latestPrediction?.contributions ?? []).slice(0, 3)

  const riskGradient = isHigh
    ? 'bg-gradient-to-br from-risk-high/5 to-risk-high/15 border-risk-high/40'
    : 'bg-gradient-to-br from-risk-low/5 to-risk-low/15 border-risk-low/40'

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader />

      <div className="flex-1 px-4 lg:px-8 pb-6 pt-4 w-full max-w-3xl mx-auto space-y-4">
        <div className="flex justify-between items-start">
          <div>
            <p className="font-bold text-ink-base font-display">{patient?.name}</p>
            <p className="text-xs text-ink-secondary">{patient?.medicalId} · Month {derivedMonthNumber}</p>
            <p className="text-xs text-ink-muted">{patient?.treatmentRegimen?.toUpperCase() || ''} · {derivedMonthNumber} of 6 months</p>
          </div>
          <RiskBadge probability={prob} size="md" />
        </div>

        <div className={`rounded-2xl border-2 p-5 ${riskGradient}`}>
          <p className={`font-bold text-base ${text} mb-2`}>
            {isHigh ? 'HIGH RISK' : 'LOW RISK'} — {formatPercent(prob)} Failure Probability
          </p>
          {delta != null && (
            <p className={`text-sm mb-3 flex items-center gap-1 tabular-nums
              ${delta > 0 ? 'text-risk-high' : 'text-risk-low'}`}>
              {delta > 0
                ? <ArrowUp size={14} aria-hidden="true" />
                : <ArrowDown size={14} aria-hidden="true" />}
              {formatDelta(delta)}{' '}
              {delta > 0 ? 'increased' : 'decreased'} since last month
            </p>
          )}
          <p className="text-sm font-semibold text-ink-base mb-1">Top Risk Factors:</p>
          {topContribs.map(c => (
            <p key={c.feature} className={`text-sm ${c.direction === 'risk' ? 'text-risk-high' : 'text-risk-low'}`}>
              • {c.feature} — {c.direction === 'risk' ? 'strongest contributor' : 'protective factor'}
            </p>
          ))}
          {id && (
            <button
              onClick={() => navigate(`/patient/${id}/features`)}
              className="text-primary text-sm font-medium mt-2"
            >
              See full breakdown →
            </button>
          )}
        </div>

        {isHigh && (
          <div className="bg-blue-50 border border-blue-100 rounded-2xl p-3 flex gap-2.5 items-start">
            <Info size={16} className="text-blue-500 mt-0.5 flex-shrink-0" aria-hidden="true" />
            <p className="text-xs text-blue-700">
              Clinical Note: Patient shows increasing failure risk. Consider enhanced monitoring and adherence support intervention.
            </p>
          </div>
        )}
      </div>

      <PageFooter>
        <div className="flex gap-3">
          <button onClick={() => navigate(`/patient/${id}/chart`)}
            className="flex-1 border border-border text-ink-secondary py-3.5 rounded-xl font-semibold text-sm">
            ← Back
          </button>
          <button onClick={() => navigate(`/patient/${id}`)}
            className="flex-[2] bg-primary text-white py-3.5 rounded-xl font-semibold text-sm">
            View Patient Profile
          </button>
        </div>
      </PageFooter>
    </div>
  )
}
