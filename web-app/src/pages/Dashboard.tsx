import { useEffect, useState } from 'react'
import { LayoutDashboard, ArrowDown, ArrowUp, Minus, HelpCircle } from 'lucide-react'
import { getAllPatients } from '../lib/storage'
import { riskLabel, riskColor } from '../components/RiskBadge'
import { AppHeader } from '../components/AppHeader'
import type { Patient } from '../lib/storage'

const DOTS_COURSE_TOTAL_RECORDS = 13

function latestProb(p: Patient): number {
  return p.predictions.at(-1)?.failureProbability ?? 0
}

export function Dashboard() {
  const [patients, setPatients] = useState<Patient[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [adhHintOpen, setAdhHintOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const list = await getAllPatients()
        if (!cancelled) setPatients(list)
      } catch (e) {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  if (patients === null) {
    return (
      <div className="min-h-screen bg-bg flex flex-col">
        <AppHeader title="Dashboard" />
        <div className="flex-1 px-4 lg:px-8 pt-10 pb-8 w-full">
          <p className="text-sm text-ink-muted">Loading dashboard…</p>
          {loadError && <p className="text-sm text-risk-high mt-2">{loadError}</p>}
        </div>
      </div>
    )
  }

  const total = patients.length

  const highRisk = patients.filter(p => riskLabel(latestProb(p)) === 'HIGH').length
  const medRisk  = patients.filter(p => riskLabel(latestProb(p)) === 'MED').length
  const lowRisk  = patients.filter(p => riskLabel(latestProb(p)) === 'LOW').length
  const avgRisk  = total > 0
    ? patients.reduce((sum, p) => sum + latestProb(p), 0) / total
    : null
  const dueCheckin = patients.filter(p => {
    if (p.monthlyRecords.length >= DOTS_COURSE_TOTAL_RECORDS) return false
    if (!p.treatmentStartDate) return false
    const start = new Date(p.treatmentStartDate).getTime()
    const monthsElapsed = Math.floor((Date.now() - start) / (1000 * 60 * 60 * 24 * 30))
    if (monthsElapsed < 1) return false
    return p.monthlyRecords.length < monthsElapsed
  }).length
  const onTreatment = patients.filter(p => p.treatmentRegimen !== undefined).length

  const patientsWithDays = patients.filter(p => p.features.daysToTreatment != null)
  const avgDaysToTreatment = patientsWithDays.length > 0
    ? Math.round(patientsWithDays.reduce((sum, p) => sum + p.features.daysToTreatment, 0) / patientsWithDays.length)
    : null
  // WHO recommends treatment within 1 day of diagnosis; >7 days is a concern
  const daysToTreatmentStatus = avgDaysToTreatment === null ? null
    : avgDaysToTreatment <= 3 ? 'good'
    : avgDaysToTreatment <= 7 ? 'moderate'
    : 'poor'

  const allRecords = patients.flatMap(p => p.monthlyRecords)
  const totalRecords = allRecords.length
  const fullAdh    = allRecords.filter(r => r.adherence === 'full').length
  const partialAdh = allRecords.filter(r => r.adherence === 'partial').length
  const poorAdh    = allRecords.filter(r => r.adherence === 'poor').length
  const fullAdhPct = totalRecords > 0 ? Math.round((fullAdh / totalRecords) * 100) : null

  const regimenGroups = {
    hrze:       patients.filter(p => p.treatmentRegimen === 'hrze').length,
    mdr:        patients.filter(p => p.treatmentRegimen === 'mdr').length,
    xdr:        patients.filter(p => p.treatmentRegimen === 'xdr').length,
    unassigned: patients.filter(p => p.treatmentRegimen === undefined).length,
  }
  const maxRegimen = Math.max(...Object.values(regimenGroups), 1)

  const regimenOutcomes = (['hrze', 'mdr', 'xdr'] as const).map(regimen => {
    const group = patients.filter(p => p.treatmentRegimen === regimen)
    const count = group.length
    const avgProb = count > 0
      ? group.reduce((sum, p) => sum + latestProb(p), 0) / count
      : null
    const improving = group.filter(p => {
      if (p.predictions.length < 2) return false
      return p.predictions.at(-1)!.failureProbability - p.predictions.at(-2)!.failureProbability < -0.05
    }).length
    const worsening = group.filter(p => {
      if (p.predictions.length < 2) return false
      return p.predictions.at(-1)!.failureProbability - p.predictions.at(-2)!.failureProbability > 0.05
    }).length
    return { regimen, count, avgProb, improving, worsening }
  })

  const maleCount   = patients.filter(p => p.features.sex === 'M').length
  const femaleCount = patients.filter(p => p.features.sex === 'F').length
  const sexTotal    = maleCount + femaleCount || 1

  const ageBuckets = [
    { label: '0–17',  count: patients.filter(p => p.features.age <= 17).length },
    { label: '18–34', count: patients.filter(p => p.features.age >= 18 && p.features.age <= 34).length },
    { label: '35–49', count: patients.filter(p => p.features.age >= 35 && p.features.age <= 49).length },
    { label: '50–64', count: patients.filter(p => p.features.age >= 50 && p.features.age <= 64).length },
    { label: '65+',   count: patients.filter(p => p.features.age >= 65).length },
  ]
  const maxAge = Math.max(...ageBuckets.map(b => b.count), 1)

  const withTwoPreds       = patients.filter(p => p.predictions.length >= 2)
  const excludedFromTrends = patients.length - withTwoPreds.length
  const improving = withTwoPreds.filter(p => {
    const delta = p.predictions.at(-1)!.failureProbability - p.predictions.at(-2)!.failureProbability
    return delta < -0.05
  }).length
  const worsening = withTwoPreds.filter(p => {
    const delta = p.predictions.at(-1)!.failureProbability - p.predictions.at(-2)!.failureProbability
    return delta > 0.05
  }).length
  const stable     = withTwoPreds.length - improving - worsening
  const trendTotal = withTwoPreds.length || 1

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <AppHeader title="Dashboard" />

      <div className="flex-1 px-4 lg:px-8 pt-4 pb-8 space-y-5 w-full">
        <div>
          <h1 className="text-xl font-bold text-ink-base font-display">Population Overview</h1>
          <p className="text-sm text-ink-muted mt-0.5">
            {total} registered patient{total !== 1 ? 's' : ''}
          </p>
        </div>

        {total === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center py-24 gap-3">
            <LayoutDashboard size={56} className="text-ink-muted/40 mx-auto" aria-hidden="true" />
            <p className="text-ink-secondary text-sm font-semibold">No patient data yet</p>
            <p className="text-ink-muted text-xs">Register patients to see dashboard analytics.</p>
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <section aria-labelledby="summary-heading">
              <p id="summary-heading" className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-3">Summary</p>
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">

                <div className="bg-surface rounded-2xl border border-border p-4 shadow-sm text-center">
                  <p className="text-2xl font-bold font-display text-primary tabular-nums">{total}</p>
                  <p className="text-xs text-ink-muted mt-1">Total Patients</p>
                </div>

                <div className="bg-risk-high/5 rounded-2xl border border-risk-high/20 p-4 shadow-sm text-center">
                  <p className="text-2xl font-bold font-display text-risk-high tabular-nums">{highRisk}</p>
                  <p className="text-xs text-ink-muted mt-1">High Risk</p>
                </div>

                <div className="bg-risk-med/5 rounded-2xl border border-risk-med/20 p-4 shadow-sm text-center">
                  <p className="text-2xl font-bold font-display text-risk-med tabular-nums">{dueCheckin}</p>
                  <p className="text-xs text-ink-muted mt-1">Due Check-in</p>
                </div>

                <div className="bg-surface rounded-2xl border border-border p-4 shadow-sm text-center">
                  {avgRisk !== null ? (
                    <p className={`text-2xl font-bold font-display tabular-nums ${riskColor(avgRisk).text}`}>
                      {Math.round(avgRisk * 100)}%
                    </p>
                  ) : (
                    <p className="text-2xl font-bold text-ink-muted/30 font-display">—</p>
                  )}
                  <p className="text-xs text-ink-muted mt-1">Avg Risk</p>
                </div>

                <div className="bg-risk-low/5 rounded-2xl border border-risk-low/20 p-4 shadow-sm text-center">
                  {fullAdhPct !== null ? (
                    <p className="text-2xl font-bold font-display text-risk-low tabular-nums">{fullAdhPct}%</p>
                  ) : (
                    <p className="text-2xl font-bold text-ink-muted/30 font-display">—</p>
                  )}
                  <p className="text-xs text-ink-muted mt-1">Full Adherence</p>
                </div>

                <div className="bg-surface rounded-2xl border border-border p-4 shadow-sm text-center">
                  <p className="text-2xl font-bold font-display text-primary tabular-nums">{onTreatment}</p>
                  <p className="text-xs text-ink-muted mt-1">On Treatment</p>
                </div>

                <div className={`rounded-2xl border p-4 shadow-sm text-center
                  ${daysToTreatmentStatus === 'good' ? 'bg-risk-low/5 border-risk-low/20'
                    : daysToTreatmentStatus === 'moderate' ? 'bg-risk-med/5 border-risk-med/20'
                    : daysToTreatmentStatus === 'poor' ? 'bg-risk-high/5 border-risk-high/20'
                    : 'bg-surface border-border'}`}>
                  {avgDaysToTreatment !== null ? (
                    <p className={`text-2xl font-bold font-display tabular-nums
                      ${daysToTreatmentStatus === 'good' ? 'text-risk-low'
                        : daysToTreatmentStatus === 'moderate' ? 'text-risk-med'
                        : 'text-risk-high'}`}>
                      {avgDaysToTreatment}d
                    </p>
                  ) : (
                    <p className="text-2xl font-bold text-ink-muted/30 font-display">—</p>
                  )}
                  <p className="text-xs text-ink-muted mt-1">Avg Days to Tx</p>
                </div>

              </div>
            </section>

            {/* Risk Distribution + Adherence + Regimens */}
            <div className="grid lg:grid-cols-3 gap-4">

              {/* Risk Distribution */}
              <section className="bg-surface rounded-2xl border border-border p-4 lg:p-5 shadow-sm" aria-labelledby="risk-dist-heading">
                <p id="risk-dist-heading" className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-4">Risk Distribution</p>
                <div className="flex h-4 rounded-full overflow-hidden mb-3">
                  <div className="bg-risk-high flex items-center justify-center" style={{ width: `${(highRisk / total) * 100}%` }}>
                    {highRisk / total > 0.15 && <span className="text-white text-xs font-bold">{highRisk}</span>}
                  </div>
                  <div className="bg-risk-med flex items-center justify-center" style={{ width: `${(medRisk / total) * 100}%` }}>
                    {medRisk / total > 0.15 && <span className="text-white text-xs font-bold">{medRisk}</span>}
                  </div>
                  <div className="bg-risk-low flex items-center justify-center" style={{ width: `${(lowRisk / total) * 100}%` }}>
                    {lowRisk / total > 0.15 && <span className="text-white text-xs font-bold">{lowRisk}</span>}
                  </div>
                </div>
                <div className="flex gap-4 flex-wrap">
                  {[
                    { label: 'HIGH', count: highRisk, color: 'bg-risk-high' },
                    { label: 'MED',  count: medRisk,  color: 'bg-risk-med' },
                    { label: 'LOW',  count: lowRisk,  color: 'bg-risk-low' },
                  ].map(({ label, count, color }) => (
                    <span key={label} className="flex items-center gap-1.5 text-xs text-ink-secondary">
                      <span className={`w-2 h-2 rounded-full ${color}`} aria-hidden="true" />
                      {label} — {count}
                    </span>
                  ))}
                </div>
              </section>

              {/* Adherence Overview */}
              <section className="bg-surface rounded-2xl border border-border p-4 lg:p-5 shadow-sm" aria-labelledby="adherence-heading">
                <div className="flex items-center justify-between mb-2">
                  <p id="adherence-heading" className="text-xs font-semibold text-ink-muted uppercase tracking-wider">Adherence Overview</p>
                  <div className="relative group">
                    <button
                      type="button"
                      onClick={() => setAdhHintOpen(v => !v)}
                      aria-label="What do adherence levels mean?"
                      aria-expanded={adhHintOpen}
                      className="w-6 h-6 rounded-full flex items-center justify-center text-ink-muted hover:bg-gray-100 hover:text-ink-secondary transition-colors"
                    >
                      <HelpCircle size={14} aria-hidden="true" />
                    </button>
                    <div
                      role="tooltip"
                      className={`absolute right-0 top-7 z-10 w-64 bg-surface border border-border rounded-xl shadow-lg p-3 space-y-2.5 transition-all duration-150
                        ${adhHintOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}
                        group-hover:opacity-100 group-hover:pointer-events-auto`}
                    >
                      {[
                        { label: 'Full',    color: 'bg-risk-low',  hint: 'All doses taken as prescribed — strongest predictor of treatment success.' },
                        { label: 'Partial', color: 'bg-risk-med',  hint: 'Some doses missed. Repeated gaps significantly raise failure risk.' },
                        { label: 'Poor',    color: 'bg-risk-high', hint: 'Most doses missed. Immediate intervention needed — primary cause of drug resistance.' },
                      ].map(({ label, color, hint }) => (
                        <div key={label} className="flex gap-2.5 items-start">
                          <span className={`w-2 h-2 rounded-full mt-1 flex-shrink-0 ${color}`} aria-hidden="true" />
                          <div>
                            <p className="text-xs font-semibold text-ink-base">{label}</p>
                            <p className="text-xs text-ink-secondary leading-relaxed">{hint}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                {totalRecords === 0 ? (
                  <p className="text-xs text-ink-muted italic mt-3">No monthly check-in records yet.</p>
                ) : (
                  <div className="flex flex-col gap-2.5 mt-3">
                    {[
                      { label: 'Full',    count: fullAdh,    color: 'bg-risk-low' },
                      { label: 'Partial', count: partialAdh, color: 'bg-risk-med' },
                      { label: 'Poor',    count: poorAdh,    color: 'bg-risk-high' },
                    ].map(({ label, count, color }) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs text-ink-secondary w-12">{label}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                          <div className={`h-full rounded-full ${color}`}
                            style={{ width: `${Math.round((count / totalRecords) * 100)}%` }} />
                        </div>
                        <span className="text-xs font-semibold text-ink-secondary tabular-nums w-8 text-right">
                          {Math.round((count / totalRecords) * 100)}%
                        </span>
                      </div>
                    ))}
                    <p className="text-xs text-ink-muted mt-1">{totalRecords} check-in record{totalRecords !== 1 ? 's' : ''}</p>
                  </div>
                )}
              </section>

              {/* Treatment Regimens */}
              <section className="bg-surface rounded-2xl border border-border p-4 lg:p-5 shadow-sm" aria-labelledby="regimens-heading">
                <p id="regimens-heading" className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-4">Treatment Regimens</p>
                <div className="flex flex-col gap-2.5">
                  {[
                    { label: 'HRZE',       count: regimenGroups.hrze,       primary: true },
                    { label: 'MDR-TB',     count: regimenGroups.mdr,        primary: true },
                    { label: 'XDR-TB',     count: regimenGroups.xdr,        primary: true },
                    { label: 'Unassigned', count: regimenGroups.unassigned, primary: false },
                  ].map(({ label, count, primary }) => (
                    <div key={label} className="flex items-center gap-3">
                      <span className={`text-xs w-20 ${primary ? 'text-ink-secondary' : 'text-ink-muted'}`}>{label}</span>
                      <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                        <div className={`h-full rounded-full ${primary ? 'bg-primary' : 'bg-gray-300'}`}
                          style={{ width: `${Math.round((count / maxRegimen) * 100)}%` }} />
                      </div>
                      <span className={`text-xs font-semibold tabular-nums w-4 text-right ${primary ? 'text-ink-secondary' : 'text-ink-muted'}`}>
                        {count}
                      </span>
                    </div>
                  ))}
                </div>
              </section>
            </div>

            {/* Demographics + Risk Trends */}
            <div className="grid lg:grid-cols-2 gap-4">

              {/* Demographics */}
              <section aria-labelledby="demo-heading">
                <p id="demo-heading" className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-3">Demographics</p>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-surface rounded-2xl border border-border p-4 shadow-sm">
                    <p className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-3">Sex</p>
                    <div className="flex h-3 rounded-full overflow-hidden mb-3">
                      <div className="bg-blue-400" style={{ width: `${Math.round((maleCount / sexTotal) * 100)}%` }} />
                      <div className="bg-pink-400" style={{ width: `${Math.round((femaleCount / sexTotal) * 100)}%` }} />
                    </div>
                    <div className="flex justify-between text-xs text-ink-secondary">
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" aria-hidden="true" />M — {maleCount}
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-pink-400 inline-block" aria-hidden="true" />F — {femaleCount}
                      </span>
                    </div>
                  </div>

                  <div className="bg-surface rounded-2xl border border-border p-4 shadow-sm">
                    <p className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-3">Age Groups</p>
                    <div className="flex flex-col gap-2">
                      {ageBuckets.map(({ label, count }) => (
                        <div key={label} className="flex items-center gap-2">
                          <span className="text-xs text-ink-muted w-8">{label}</span>
                          <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                            <div className="h-full rounded-full bg-primary"
                              style={{ width: `${Math.round((count / maxAge) * 100)}%` }} />
                          </div>
                          <span className="text-xs text-ink-muted tabular-nums w-3 text-right">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </section>

              {/* Risk Trends */}
              <section className="bg-surface rounded-2xl border border-border p-4 lg:p-5 shadow-sm" aria-labelledby="trends-heading">
                <p id="trends-heading" className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-4">Risk Trends</p>
                {withTwoPreds.length === 0 ? (
                  <p className="text-xs text-ink-muted italic">
                    Not enough data yet. Risk trends appear after a patient's second prediction.
                  </p>
                ) : (
                  <>
                    <div className="grid grid-cols-3 gap-2 mb-4">
                      <div className="bg-risk-low/5 border border-risk-low/20 rounded-xl p-3 text-center">
                        <p className="text-lg font-bold font-display text-risk-low flex items-center justify-center gap-1">
                          <ArrowDown size={16} aria-hidden="true" />{improving}
                        </p>
                        <p className="text-xs text-ink-muted mt-1">Improving</p>
                      </div>
                      <div className="bg-gray-50 border border-border rounded-xl p-3 text-center">
                        <p className="text-lg font-bold font-display text-ink-muted flex items-center justify-center gap-1">
                          <Minus size={16} aria-hidden="true" />{stable}
                        </p>
                        <p className="text-xs text-ink-muted mt-1">Stable</p>
                      </div>
                      <div className="bg-risk-high/5 border border-risk-high/20 rounded-xl p-3 text-center">
                        <p className="text-lg font-bold font-display text-risk-high flex items-center justify-center gap-1">
                          <ArrowUp size={16} aria-hidden="true" />{worsening}
                        </p>
                        <p className="text-xs text-ink-muted mt-1">Worsening</p>
                      </div>
                    </div>
                    <div className="flex h-3 rounded-full overflow-hidden mb-2">
                      <div className="bg-risk-low" style={{ width: `${(improving / trendTotal) * 100}%` }} />
                      <div className="bg-gray-300"  style={{ width: `${(stable    / trendTotal) * 100}%` }} />
                      <div className="bg-risk-high" style={{ width: `${(worsening / trendTotal) * 100}%` }} />
                    </div>
                    {excludedFromTrends > 0 && (
                      <p className="text-xs text-ink-muted">
                        {excludedFromTrends} patient{excludedFromTrends !== 1 ? 's' : ''} excluded (fewer than 2 predictions).
                      </p>
                    )}
                  </>
                )}
              </section>
            </div>

            {/* Avg Days to Treatment */}
            <section className="bg-surface rounded-2xl border border-border p-4 lg:p-5 shadow-sm" aria-labelledby="days-to-tx-heading">
              <p id="days-to-tx-heading" className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-4">Average Days to Treatment</p>
              {avgDaysToTreatment === null ? (
                <p className="text-xs text-ink-muted italic">No diagnosis-to-treatment data available yet.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                  {/* Big number */}
                  <div className="text-center md:border-r border-border">
                    <p className={`text-5xl font-extrabold font-display tabular-nums
                      ${daysToTreatmentStatus === 'good' ? 'text-risk-low'
                        : daysToTreatmentStatus === 'moderate' ? 'text-risk-med'
                        : 'text-risk-high'}`}>
                      {avgDaysToTreatment}
                    </p>
                    <p className="text-sm text-ink-muted mt-1">days on average</p>
                    <p className={`text-xs font-semibold mt-2 uppercase tracking-wide
                      ${daysToTreatmentStatus === 'good' ? 'text-risk-low'
                        : daysToTreatmentStatus === 'moderate' ? 'text-risk-med'
                        : 'text-risk-high'}`}>
                      {daysToTreatmentStatus === 'good' ? '✓ Within target'
                        : daysToTreatmentStatus === 'moderate' ? '⚠ Moderate delay'
                        : '✕ Significant delay'}
                    </p>
                  </div>

                  {/* Distribution bar */}
                  <div className="md:col-span-2 space-y-3">
                    {[
                      { label: '≤ 3 days', count: patientsWithDays.filter(p => p.features.daysToTreatment <= 3).length, color: 'bg-risk-low' },
                      { label: '4–7 days', count: patientsWithDays.filter(p => p.features.daysToTreatment > 3 && p.features.daysToTreatment <= 7).length, color: 'bg-risk-med' },
                      { label: '> 7 days', count: patientsWithDays.filter(p => p.features.daysToTreatment > 7).length, color: 'bg-risk-high' },
                    ].map(({ label, count, color }) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs text-ink-secondary w-16">{label}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                          <div className={`h-full rounded-full ${color}`}
                            style={{ width: `${patientsWithDays.length > 0 ? Math.round((count / patientsWithDays.length) * 100) : 0}%` }} />
                        </div>
                        <span className="text-xs font-semibold text-ink-secondary tabular-nums w-8 text-right">
                          {count}
                        </span>
                      </div>
                    ))}
                    <p className="text-xs text-ink-muted pt-1">
                      WHO target: treatment within 1 day of diagnosis. Based on {patientsWithDays.length} patient{patientsWithDays.length !== 1 ? 's' : ''}.
                    </p>
                  </div>
                </div>
              )}
            </section>

            {/* Regimen Outcomes */}
            <section className="bg-surface rounded-2xl border border-border p-4 lg:p-5 shadow-sm" aria-labelledby="regimen-outcomes-heading">
              <p id="regimen-outcomes-heading" className="text-xs font-semibold text-ink-muted uppercase tracking-wider mb-4">Regimen Outcomes by Type</p>
              {regimenOutcomes.every(r => r.count === 0) ? (
                <p className="text-xs text-ink-muted italic">No patients with assigned regimens yet.</p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {regimenOutcomes.map(({ regimen, count, avgProb, improving, worsening }) => {
                    const label = regimen === 'hrze' ? 'HRZE' : regimen === 'mdr' ? 'MDR-TB' : 'XDR-TB'
                    const description = regimen === 'hrze'
                      ? 'Drug-Susceptible TB'
                      : regimen === 'mdr'
                      ? 'Multi-Drug Resistant'
                      : 'Extensively Drug Resistant'
                    return (
                      <div key={regimen} className="border border-border rounded-xl p-4 space-y-3">
                        <div className="flex items-start justify-between">
                          <div>
                            <p className="text-sm font-bold text-ink-base">{label}</p>
                            <p className="text-xs text-ink-muted">{description}</p>
                          </div>
                          <span className="text-xs font-semibold bg-primary/10 text-primary px-2 py-0.5 rounded-full">
                            {count} patient{count !== 1 ? 's' : ''}
                          </span>
                        </div>

                        {count === 0 ? (
                          <p className="text-xs text-ink-muted italic">No patients assigned.</p>
                        ) : (
                          <>
                            {/* Avg Risk */}
                            <div>
                              <p className="text-xs text-ink-muted mb-1">Avg Failure Risk</p>
                              <div className="flex items-center gap-2">
                                <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                                  <div
                                    className={`h-full rounded-full ${avgProb! >= 0.6 ? 'bg-risk-high' : avgProb! >= 0.4 ? 'bg-risk-med' : 'bg-risk-low'}`}
                                    style={{ width: `${Math.round((avgProb ?? 0) * 100)}%` }}
                                  />
                                </div>
                                <span className={`text-sm font-bold tabular-nums ${riskColor(avgProb ?? 0).text}`}>
                                  {avgProb !== null ? `${Math.round(avgProb * 100)}%` : '—'}
                                </span>
                              </div>
                            </div>

                            {/* Trend mini-summary */}
                            <div className="flex gap-2">
                              <div className="flex-1 bg-risk-low/5 border border-risk-low/20 rounded-lg p-2 text-center">
                                <p className="text-sm font-bold text-risk-low tabular-nums">{improving}</p>
                                <p className="text-xs text-ink-muted">Improving</p>
                              </div>
                              <div className="flex-1 bg-risk-high/5 border border-risk-high/20 rounded-lg p-2 text-center">
                                <p className="text-sm font-bold text-risk-high tabular-nums">{worsening}</p>
                                <p className="text-xs text-ink-muted">Worsening</p>
                              </div>
                              <div className="flex-1 bg-gray-50 border border-border rounded-lg p-2 text-center">
                                <p className="text-sm font-bold text-ink-muted tabular-nums">
                                  {count - improving - worsening}
                                </p>
                                <p className="text-xs text-ink-muted">Stable</p>
                              </div>
                            </div>
                          </>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </div>
  )
}
