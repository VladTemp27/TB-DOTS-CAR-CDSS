import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Info } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { RiskBadge, riskColor } from '../components/RiskBadge'
import { FeatureBar } from '../components/FeatureBar'
import { getPatient } from '../lib/storage'
import { featureKeyFor } from '../lib/inference'
import { PageFooter } from '../components/PageFooter'
import { ClinicalInterpretation } from '../components/ClinicalInterpretation'
import type { ContributionResult } from '../lib/inference'
import type { Patient } from '../lib/storage'

function formatValue(displayName: string, raw: unknown): string | undefined {
  if (raw === undefined || raw === null || raw === '') return undefined
  if (displayName === 'Days to Treatment') return `${raw} days`
  return String(raw)
}

export function FeatureContribution() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const [patient, setPatient] = useState<Patient | null>(null)
  const [loadingPatient, setLoadingPatient] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

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

  const latest  = patient?.predictions.at(-1)

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

  if (!patient || !latest) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg">
        <p className="text-ink-muted">No prediction data available.</p>
      </div>
    )
  }

  const prob          = latest.failureProbability
  const contributions = latest.contributions
  const maxDelta      = Math.max(...contributions.map(c => c.delta), 0.001)
  const featuresUsed  = latest.featuresUsed ?? patient.features

  // Reconstruct a ContributionResult shape from the stored PredictionRecord
  // (successProbability is derived — storage omits it as redundant)
  const result: ContributionResult = {
    label: latest.label,
    failureProbability: latest.failureProbability,
    successProbability: 1 - latest.failureProbability,
    contributions: latest.contributions,
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader backLabel="Back to Patient Profile" onBack={() => navigate(`/patient/${id}`)} />

      <div className="flex-1 px-4 lg:px-8 pb-6 pt-4 w-full max-w-3xl mx-auto space-y-4">
        <div>
          <h2 className="font-bold text-ink-base text-lg font-display">Risk Factor Analysis</h2>
          <p className="text-sm text-ink-secondary">What's driving {patient.name}'s failure risk</p>
        </div>

        <div className="bg-surface border border-border rounded-2xl p-4 shadow-sm flex items-center gap-3">
          <span className="text-sm font-semibold text-ink-secondary">Overall Risk:</span>
          <span className={`text-2xl font-extrabold font-display tabular-nums ${riskColor(prob).text}`}>
            {Math.round(prob * 100)}%
          </span>
          <RiskBadge probability={prob} size="lg" />
        </div>

        <div className="bg-surface border border-border rounded-2xl p-4 lg:p-5 shadow-sm">
          <p className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-4">Feature Contributions</p>
          <div className="space-y-1">
            {contributions.map(c => {
              const key = featureKeyFor(c.feature)
              const value = key ? formatValue(c.feature, featuresUsed[key]) : undefined
              return (
                <FeatureBar
                  key={c.feature}
                  feature={c.feature}
                  delta={c.delta}
                  direction={c.direction}
                  maxDelta={maxDelta}
                  value={value}
                />
              )
            })}
          </div>
          <div className="flex gap-4 mt-4 pt-3 border-t border-border">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm bg-risk-high/70" aria-hidden="true" />
              <span className="text-xs text-ink-secondary">Increases failure risk</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-sm bg-risk-low/70" aria-hidden="true" />
              <span className="text-xs text-ink-secondary">Decreases failure risk</span>
            </div>
          </div>
        </div>

        <ClinicalInterpretation patient={patient} result={result} />

        <div className="bg-blue-50 border border-blue-100 rounded-2xl p-3 flex gap-2.5 items-start">
          <Info size={16} className="text-blue-500 mt-0.5 flex-shrink-0" aria-hidden="true" />
          <p className="text-xs text-blue-700">
            This explanation is generated by the AI model. Final clinical decisions must be made by qualified healthcare professionals.
          </p>
        </div>
      </div>

      <PageFooter>
        <button onClick={() => navigate(`/patient/${id}`)}
          className="w-full border border-border text-ink-secondary py-3.5 rounded-xl font-semibold text-sm">
          ← Back to Patient Profile
        </button>
      </PageFooter>
    </div>
  )
}
