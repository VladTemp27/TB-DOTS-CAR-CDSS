import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { AppHeader } from '../components/AppHeader'
import { RiskBadge, riskColor } from '../components/RiskBadge'
import { getPatient } from '../lib/storage'
import { PageFooter } from '../components/PageFooter'
import type { Patient } from '../lib/storage'

export function PatientProfile() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const [patient, setPatient] = useState<Patient | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!id) {
      setLoading(false)
      setPatient(null)
      return
    }
    setLoading(true)
    ;(async () => {
      try {
        const p = await getPatient(id)
        if (!cancelled) setPatient(p)
      } catch (e) {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e))
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [id])

  if (loading) {
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

  if (!patient) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg">
        <p className="text-ink-muted">Patient not found.</p>
      </div>
    )
  }

  const latestProb = patient.predictions.at(-1)?.failureProbability ?? 0
  const prevProb   = patient.predictions.at(-2)?.failureProbability
  const delta      = prevProb != null ? latestProb - prevProb : null
  const { text }   = riskColor(latestProb)

  const regimen = patient.treatmentRegimen === 'hrze' ? 'HRZE Regimen'
    : patient.treatmentRegimen === 'mdr' ? 'MDR-TB Regimen'
    : patient.treatmentRegimen === 'xdr' ? 'XDR-TB Regimen'
    : 'No Regimen Selected'
  const hasTreatment = !!patient.treatmentRegimen
  const monthlyRecords = [...patient.monthlyRecords].sort((a, b) => a.month - b.month)

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader backLabel="Back to All Patients" onBack={() => navigate('/')} />

      <div className="flex-1 px-4 lg:px-8 pb-6 lg:pb-8 pt-4 w-full max-w-5xl mx-auto">
        <div className="lg:grid lg:grid-cols-[2fr_3fr] gap-6">

          {/* Left column: patient info + risk score + action */}
          <div className="space-y-4 mb-4 lg:mb-0">
            <div className="bg-surface border border-border rounded-2xl p-4 shadow-sm">
              <p className="font-bold text-ink-base text-lg font-display">{patient.name}</p>
              <p className="text-sm text-ink-secondary mt-1">
                {patient.features.sex === 'M' ? 'Male' : 'Female'}, {patient.features.age} yrs
              </p>
              <p className="text-sm text-ink-secondary">{regimen}</p>
              <p className="text-sm text-ink-secondary">Started: {patient.treatmentStartDate || 'N/A'}</p>
              <p className="text-xs text-ink-muted mt-1.5">{patient.medicalId}</p>
            </div>

            <div className="bg-surface border border-border rounded-2xl p-4 shadow-sm">
              <p className={`text-6xl font-extrabold font-display tabular-nums ${text}`}>
                {Math.round(latestProb * 100)}%
              </p>
              <div className="flex items-center gap-2 mt-2">
                <p className="text-sm font-semibold text-ink-secondary uppercase tracking-wide">Failure Risk</p>
                <RiskBadge probability={latestProb} />
              </div>
              {delta != null && (
                <p className={`text-sm mt-1 tabular-nums ${delta > 0 ? 'text-risk-high' : 'text-risk-low'}`}>
                  {delta > 0 ? '↑' : '↓'} {Math.abs(Math.round(delta * 100))}% from last month
                </p>
              )}

              <div className="mt-3">
                <p className="text-sm font-semibold text-ink-secondary mb-1">Top Risk Factors:</p>
                {patient.predictions.at(-1)?.contributions.slice(0, 3).map((c: { feature: string }) => (
                  <p key={c.feature} className="text-sm text-ink-secondary">• {c.feature}</p>
                ))}
                <button
                  onClick={() => navigate(`/patient/${id}/features`)}
                  className="text-primary text-sm font-medium mt-2"
                >
                  See full breakdown →
                </button>
              </div>
            </div>

            {/* Desktop-only action button */}
            {hasTreatment ? (
              <button
                onClick={() => navigate(`/patient/${id}/checkin`)}
                className="hidden lg:block w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm active:bg-primary-dark hover:bg-primary-mid transition-colors"
              >
                Log Monthly Update →
              </button>
            ) : (
              <button
                onClick={() => navigate(`/patient/${id}/treatment`)}
                className="hidden lg:block w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm active:bg-primary-dark hover:bg-primary-mid transition-colors"
              >
                Select Treatment Regimen →
              </button>
            )}

            <button
              onClick={() => navigate(`/patient/${id}/chart`)}
              className="hidden lg:block w-full bg-surface border border-border text-ink-base py-3.5 rounded-xl font-semibold text-sm active:bg-gray-50 hover:border-primary/30 transition-all"
            >
              View Patient Chart →
            </button>
          </div>

          {/* Right column: risk trend + monthly records */}
          <div className="space-y-4">
            {monthlyRecords.length > 0 && (
              <div className="bg-surface border border-border rounded-2xl p-4 shadow-sm">
                <p className="text-sm font-semibold text-ink-base mb-3">Risk Trend Over Time</p>
                <div className="flex gap-2 overflow-x-auto pb-1">
                  {monthlyRecords.map((record, i) => {
                    const p = Math.round(record.failureProbability * 100)
                    const isLast = i === monthlyRecords.length - 1
                    return (
                      <div key={record.timestamp} className={`flex flex-col items-center min-w-[52px] border rounded-xl p-2
                        ${isLast ? 'border-risk-high/40 bg-risk-high/5' : 'border-border bg-gray-50'}`}>
                        <p className="text-xs text-ink-muted">M{record.month}</p>
                        <p className={`text-sm font-bold font-display tabular-nums ${riskColor(record.failureProbability).text}`}>{p}%</p>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {(() => {
              const m0 = patient.monthlyRecords.find(r => r.month === 0)
              return m0 ? (
                <div className="bg-surface border border-border rounded-2xl p-4 shadow-sm">
                  <p className="text-sm font-semibold text-ink-base mb-2">Baseline (M0)</p>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <p className="text-xs text-ink-muted">Weight</p>
                      <p className="font-medium">{m0.weight ?? '—'} kg</p>
                    </div>
                    <div>
                      <p className="text-xs text-ink-muted">Height</p>
                      <p className="font-medium">{m0.height ?? '—'} cm</p>
                    </div>
                    <div>
                      <p className="text-xs text-ink-muted">Smear TB LAMP</p>
                      <p className="font-medium">{m0.smearTbLamp != null ? m0.smearTbLamp : '—'}</p>
                    </div>
                    <div>
                      <p className="text-xs text-ink-muted">Xpert MTB/RIF</p>
                      <p className="font-medium">{m0.xpertMtbRif != null ? m0.xpertMtbRif : '—'}</p>
                    </div>
                  </div>
                </div>
              ) : null
            })()}

            {patient.monthlyRecords.length > 0 && (
              <div className="bg-surface border border-border rounded-2xl p-4 shadow-sm overflow-hidden">
                <p className="text-sm font-semibold text-ink-base mb-3">Monthly Records</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-xs text-ink-muted border-b border-border sticky top-0 bg-surface">
                        <th className="text-left pb-2 font-semibold">Month</th>
                        <th className="text-left pb-2 font-semibold">Wt</th>
                        <th className="text-left pb-2 font-semibold">Ht</th>
                        <th className="text-left pb-2 font-semibold">TB LAMP</th>
                        <th className="text-left pb-2 font-semibold">Xpert</th>
                        <th className="text-left pb-2 font-semibold">Doses</th>
                        <th className="text-left pb-2 font-semibold">Missed</th>
                        <th className="text-left pb-2 font-semibold">Cumul</th>
                        <th className="text-left pb-2 font-semibold">Adh%</th>
                        <th className="text-left pb-2 font-semibold">Adh</th>
                        <th className="text-right pb-2 font-semibold">Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {patient.monthlyRecords.map((r, i) => (
                        <tr key={r.timestamp} className={`border-b border-border/50 ${i % 2 === 0 ? '' : 'bg-gray-50/50'}`}>
                          <td className="py-2 text-ink-secondary">M{r.month}</td>
                          <td className="py-2 text-ink-secondary tabular-nums">{r.weight ?? '—'}</td>
                          <td className="py-2 text-ink-secondary tabular-nums">{r.height ?? '—'}</td>
                          <td className="py-2 text-ink-secondary">{r.smearTbLamp != null ? r.smearTbLamp : '—'}</td>
                          <td className="py-2 text-ink-secondary">{r.xpertMtbRif != null ? r.xpertMtbRif : '—'}</td>
                          <td className="py-2 text-ink-secondary tabular-nums">{r.monthlyDosesTaken ?? '—'}</td>
                          <td className="py-2 text-ink-secondary tabular-nums">{r.monthlyMissedDoses ?? '—'}</td>
                          <td className="py-2 text-ink-secondary tabular-nums">{r.cumulativeDosesTaken ?? '—'}</td>
                          <td className="py-2 text-ink-secondary tabular-nums">{r.pctAdherence != null ? `${r.pctAdherence.toFixed(2)}%` : '—'}</td>
                          <td className={`py-2 capitalize font-medium
                            ${r.adherence === 'full' ? 'text-risk-low'
                              : r.adherence === 'partial' ? 'text-risk-med'
                              : 'text-risk-high'}`}>
                            {r.adherence}
                          </td>
                          <td className="py-2 text-right text-ink-secondary tabular-nums">
                            {Math.round(r.failureProbability * 100)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile footer CTA */}
      <div className="lg:hidden">
        <PageFooter>
          {hasTreatment ? (
            <button
              onClick={() => navigate(`/patient/${id}/checkin`)}
              className="w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm active:bg-primary-dark"
            >
              Log Monthly Update →
            </button>
          ) : (
            <button
              onClick={() => navigate(`/patient/${id}/treatment`)}
              className="w-full bg-primary text-white py-3.5 rounded-xl font-semibold text-sm active:bg-primary-dark"
            >
              Select Treatment Regimen →
            </button>
          )}
        </PageFooter>
      </div>
    </div>
  )
}
