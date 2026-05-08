import { useNavigate, useParams, useLocation } from 'react-router-dom'
import { BarChart3, Info } from 'lucide-react'
import { AppHeader } from '../components/AppHeader'
import { StepProgress } from '../components/StepProgress'
import { RiskBadge } from '../components/RiskBadge'
import { PageFooter } from '../components/PageFooter'
import { getPatient } from '../lib/storage'
import type { ContributionResult } from '../lib/inference'

export function DiagnosticResult() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const location = useLocation()

  const patient = id ? getPatient(id) : null
  const result: ContributionResult | null = location.state?.result ?? null
  const prob = result?.failureProbability ?? patient?.predictions.at(-1)?.failureProbability ?? 0
  const isHighRisk = prob >= 0.6  // matches riskLabel() HIGH threshold in RiskBadge.tsx

  const riskGradient = isHighRisk
    ? 'bg-gradient-to-br from-risk-high/5 to-risk-high/15 border-risk-high/30'
    : 'bg-gradient-to-br from-risk-low/5 to-risk-low/15 border-risk-low/30'
  const riskText = isHighRisk ? 'text-risk-high' : 'text-risk-low'

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader />
      <StepProgress steps={5} current={4} />

      <div className="flex-1 flex flex-col">
        <div className="flex-1 px-4 lg:px-8 pb-6 pt-4 w-full max-w-4xl mx-auto">
          <div className="lg:grid lg:grid-cols-[2fr_3fr] gap-6">

            {/* Left column: patient summary + clinical findings */}
            <div className="space-y-4 mb-4 lg:mb-0">
              <div className="bg-surface border border-border rounded-2xl p-4 shadow-sm">
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-2">Patient Summary</p>
                <p className="text-sm text-ink-base font-medium">{patient?.name ?? 'Unknown'}</p>
                <p className="text-xs text-ink-secondary mt-1">ID: {patient?.medicalId ?? id}</p>
                <p className="text-xs text-ink-secondary">Age: {patient?.features.age} yrs | {patient?.features.sex === 'M' ? 'Male' : 'Female'}</p>
              </div>

              <div className="bg-surface border border-border rounded-2xl p-4 shadow-sm">
                <p className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-3">Clinical Summary</p>
                <ul className="space-y-1.5">
                  <li className="text-sm text-ink-secondary">
                    <span className="font-medium text-ink-base">Bacteriologic: </span>
                    {patient?.features.bacteriologicStatus}
                  </li>
                  <li className="text-sm text-ink-secondary">
                    <span className="font-medium text-ink-base">Microscopy: </span>
                    {patient?.features.microscopyResult}
                  </li>
                  <li className="text-sm text-ink-secondary">
                    <span className="font-medium text-ink-base">Site: </span>
                    {patient?.features.anatomicalSite === 'P' ? 'PTB (Pulmonary)' : 'EPTB'}
                  </li>
                  <li className="text-sm text-ink-secondary">
                    <span className="font-medium text-ink-base">Source: </span>
                    {patient?.features.sourceOfPatient}
                  </li>
                </ul>
                {id && (
                  <button
                    onClick={() => navigate(`/patient/${id}/features`)}
                    className="mt-3 text-primary text-sm font-medium flex items-center gap-1.5"
                  >
                    <BarChart3 size={14} aria-hidden="true" />
                    View feature contributions
                  </button>
                )}
              </div>
            </div>

            {/* Right column: risk callout + CTA */}
            <div className="space-y-4">
              <div className={`rounded-2xl border-2 p-5 ${riskGradient}`}>
                <p className={`font-bold text-base ${riskText} mb-3`}>
                  {isHighRisk
                    ? 'HIGH RISK — Elevated treatment failure probability'
                    : 'LOW RISK — Favorable treatment outlook'}
                </p>
                <div className="flex items-baseline gap-3 mb-2">
                  <span className={`text-5xl font-extrabold font-display tabular-nums ${riskText}`}>
                    {Math.round(prob * 100)}%
                  </span>
                  <RiskBadge probability={prob} size="lg" />
                </div>
                <p className="text-sm text-ink-secondary mt-2">Predicted treatment failure probability</p>
                {isHighRisk && (
                  <p className="text-sm text-risk-high mt-2 font-medium">
                    Recommend immediate treatment initiation and enhanced monitoring.
                  </p>
                )}
              </div>

              <div className="bg-blue-50 border border-blue-100 rounded-2xl p-3 flex gap-2.5 items-start">
                <Info size={16} className="text-blue-500 mt-0.5 flex-shrink-0" aria-hidden="true" />
                <p className="text-xs text-blue-700">
                  This tool provides diagnostic assistance only. Final diagnosis must be made by qualified healthcare professionals considering all clinical findings.
                </p>
              </div>

              {/* CTA visible inline on desktop */}
              <button
                onClick={() => navigate(`/patient/${id}/treatment`)}
                className="hidden lg:block w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm active:bg-primary-dark hover:bg-primary-mid transition-colors"
              >
                Select Treatment Regimen →
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile footer CTA */}
      <div className="lg:hidden">
        <PageFooter>
          <button
            onClick={() => navigate(`/patient/${id}/treatment`)}
            className="w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm active:bg-primary-dark"
          >
            Select Treatment Regimen →
          </button>
        </PageFooter>
      </div>
    </div>
  )
}
