import { getAllPatients } from '../lib/storage'
import { riskLabel, riskColor } from '../components/RiskBadge'
import { AppHeader } from '../components/AppHeader'
import type { Patient } from '../lib/storage'

// ─── helpers ────────────────────────────────────────────────────────────────

function latestProb(p: Patient): number {
  return p.predictions.at(-1)?.failureProbability ?? 0
}

// ─── component ───────────────────────────────────────────────────────────────

export function Dashboard() {
  const patients = getAllPatients()
  const total = patients.length

  // ── summary card values ──────────────────────────────────────────────────
  const highRisk = patients.filter(p => riskLabel(latestProb(p)) === 'HIGH').length
  const medRisk  = patients.filter(p => riskLabel(latestProb(p)) === 'MED').length
  const lowRisk  = patients.filter(p => riskLabel(latestProb(p)) === 'LOW').length
  const avgRisk  = total > 0
    ? patients.reduce((sum, p) => sum + latestProb(p), 0) / total
    : null
  const dueCheckin   = patients.filter(p => p.monthlyRecords.length < 6).length
  const onTreatment  = patients.filter(p => p.treatmentRegimen !== undefined).length

  // ── adherence ────────────────────────────────────────────────────────────
  const allRecords = patients.flatMap(p => p.monthlyRecords)
  const totalRecords   = allRecords.length
  const fullAdh        = allRecords.filter(r => r.adherence === 'full').length
  const partialAdh     = allRecords.filter(r => r.adherence === 'partial').length
  const poorAdh        = allRecords.filter(r => r.adherence === 'poor').length
  const fullAdhPct     = totalRecords > 0 ? Math.round((fullAdh / totalRecords) * 100) : null

  // ── regimens ─────────────────────────────────────────────────────────────
  const regimenGroups = {
    hrze:       patients.filter(p => p.treatmentRegimen === 'hrze').length,
    mdr:        patients.filter(p => p.treatmentRegimen === 'mdr').length,
    xdr:        patients.filter(p => p.treatmentRegimen === 'xdr').length,
    unassigned: patients.filter(p => p.treatmentRegimen === undefined).length,
  }
  const maxRegimen = Math.max(...Object.values(regimenGroups), 1)

  // ── demographics ─────────────────────────────────────────────────────────
  const maleCount   = patients.filter(p => p.features.sex === 'M').length
  const femaleCount = patients.filter(p => p.features.sex === 'F').length

  const ageBuckets = [
    { label: '0–17',  count: patients.filter(p => p.features.age <= 17).length },
    { label: '18–34', count: patients.filter(p => p.features.age >= 18 && p.features.age <= 34).length },
    { label: '35–49', count: patients.filter(p => p.features.age >= 35 && p.features.age <= 49).length },
    { label: '50–64', count: patients.filter(p => p.features.age >= 50 && p.features.age <= 64).length },
    { label: '65+',   count: patients.filter(p => p.features.age >= 65).length },
  ]
  const maxAge = Math.max(...ageBuckets.map(b => b.count), 1)

  // ── risk trends ───────────────────────────────────────────────────────────
  const withTwoPreds = patients.filter(p => p.predictions.length >= 2)
  const singlePred   = patients.length - withTwoPreds.length
  const improving  = withTwoPreds.filter(p => {
    const delta = (p.predictions.at(-1)!.failureProbability) - (p.predictions.at(-2)!.failureProbability)
    return delta < -0.05
  }).length
  const worsening  = withTwoPreds.filter(p => {
    const delta = (p.predictions.at(-1)!.failureProbability) - (p.predictions.at(-2)!.failureProbability)
    return delta > 0.05
  }).length
  const stable     = withTwoPreds.length - improving - worsening
  const trendTotal = withTwoPreds.length || 1

  // ─── render ───────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <AppHeader title="Dashboard" />

      <main className="flex-1 px-4 pt-4 pb-8 flex flex-col gap-4">
        {/* page title */}
        <div>
          <h1 className="text-base font-semibold text-gray-900">Population Overview</h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Based on {total} registered patient{total !== 1 ? 's' : ''}
          </p>
        </div>

        {total === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center py-20 gap-3">
            <span className="text-4xl">📊</span>
            <p className="text-gray-500 text-sm font-medium">No patient data yet</p>
            <p className="text-gray-400 text-xs">Register patients to see dashboard analytics.</p>
          </div>
        ) : (
          <>
            {/* ── [1] Summary Cards ─────────────────────────────────── */}
            <section>
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Summary</p>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">

                <div className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm text-center">
                  <p className="text-2xl font-bold text-primary">{total}</p>
                  <p className="text-xs text-gray-400 mt-1">Total Patients</p>
                </div>

                <div className="bg-red-50 rounded-xl border border-red-200 p-4 shadow-sm text-center">
                  <p className="text-2xl font-bold text-red-600">{highRisk}</p>
                  <p className="text-xs text-gray-400 mt-1">High Risk</p>
                </div>

                <div className="bg-yellow-50 rounded-xl border border-yellow-200 p-4 shadow-sm text-center">
                  <p className="text-2xl font-bold text-yellow-600">{dueCheckin}</p>
                  <p className="text-xs text-gray-400 mt-1">Due Check-in</p>
                </div>

                <div className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm text-center">
                  {avgRisk !== null ? (
                    <p className={`text-2xl font-bold ${riskColor(avgRisk).text}`}>
                      {Math.round(avgRisk * 100)}%
                    </p>
                  ) : (
                    <p className="text-2xl font-bold text-gray-300">—</p>
                  )}
                  <p className="text-xs text-gray-400 mt-1">Avg Risk</p>
                </div>

                <div className="bg-green-50 rounded-xl border border-green-100 p-4 shadow-sm text-center">
                  {fullAdhPct !== null ? (
                    <p className="text-2xl font-bold text-green-600">{fullAdhPct}%</p>
                  ) : (
                    <p className="text-2xl font-bold text-gray-300">—</p>
                  )}
                  <p className="text-xs text-gray-400 mt-1">Full Adherence</p>
                </div>

                <div className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm text-center">
                  <p className="text-2xl font-bold text-primary">{onTreatment}</p>
                  <p className="text-xs text-gray-400 mt-1">On Treatment</p>
                </div>

              </div>
            </section>

            {/* ── [2] Risk Distribution ─────────────────────────────── */}
            <section className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Risk Distribution</p>

              {/* stacked bar */}
              <div className="flex h-4 rounded-full overflow-hidden mb-3">
                <div
                  className="bg-red-400 flex items-center justify-center"
                  style={{ width: `${(highRisk / total) * 100}%` }}
                >
                  {highRisk / total > 0.15 && (
                    <span className="text-white text-xs font-bold">{highRisk}</span>
                  )}
                </div>
                <div
                  className="bg-orange-400 flex items-center justify-center"
                  style={{ width: `${(medRisk / total) * 100}%` }}
                >
                  {medRisk / total > 0.15 && (
                    <span className="text-white text-xs font-bold">{medRisk}</span>
                  )}
                </div>
                <div
                  className="bg-green-400 flex items-center justify-center"
                  style={{ width: `${(lowRisk / total) * 100}%` }}
                >
                  {lowRisk / total > 0.15 && (
                    <span className="text-white text-xs font-bold">{lowRisk}</span>
                  )}
                </div>
              </div>

              {/* legend */}
              <div className="flex gap-4 flex-wrap">
                {[
                  { label: 'HIGH', count: highRisk, color: 'bg-red-400' },
                  { label: 'MED',  count: medRisk,  color: 'bg-orange-400' },
                  { label: 'LOW',  count: lowRisk,  color: 'bg-green-400' },
                ].map(({ label, count, color }) => (
                  <span key={label} className="flex items-center gap-1.5 text-xs text-gray-600">
                    <span className={`w-2 h-2 rounded-full ${color}`} />
                    {label} — {count}
                  </span>
                ))}
              </div>
            </section>

            {/* ── [3] Adherence Overview ────────────────────────────── */}
            <section className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Adherence Overview</p>

              {/* short explanation */}
              <p className="text-xs text-gray-500 leading-relaxed mb-4">
                Tracks whether patients took their medication as prescribed during monthly check-ins.{' '}
                <span className="text-green-600 font-semibold">Full</span> = all doses taken,{' '}
                <span className="text-orange-500 font-semibold">Partial</span> = some missed,{' '}
                <span className="text-red-500 font-semibold">Poor</span> = frequently missed.
              </p>

              {totalRecords === 0 ? (
                <p className="text-xs text-gray-400 italic">No monthly check-in records yet.</p>
              ) : (
                <>
                  <div className="flex flex-col gap-2.5">
                    {[
                      { label: 'Full',    count: fullAdh,    color: 'bg-green-400' },
                      { label: 'Partial', count: partialAdh, color: 'bg-orange-400' },
                      { label: 'Poor',    count: poorAdh,    color: 'bg-red-400' },
                    ].map(({ label, count, color }) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs text-gray-700 w-12">{label}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${color}`}
                            style={{ width: `${Math.round((count / totalRecords) * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs font-semibold text-gray-600 w-8 text-right">
                          {Math.round((count / totalRecords) * 100)}%
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-gray-400 mt-3">
                    Across {totalRecords} total monthly check-in record{totalRecords !== 1 ? 's' : ''}
                  </p>
                </>
              )}
            </section>

            {/* ── [4] Treatment Regimens ────────────────────────────── */}
            <section className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Treatment Regimens</p>

              <div className="flex flex-col gap-2.5">
                {[
                  { label: 'HRZE',       count: regimenGroups.hrze,       primary: true },
                  { label: 'MDR-TB',     count: regimenGroups.mdr,        primary: true },
                  { label: 'XDR-TB',     count: regimenGroups.xdr,        primary: true },
                  { label: 'Unassigned', count: regimenGroups.unassigned, primary: false },
                ].map(({ label, count, primary }) => (
                  <div key={label} className="flex items-center gap-3">
                    <span className={`text-xs w-20 ${primary ? 'text-gray-700' : 'text-gray-400'}`}>{label}</span>
                    <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${primary ? 'bg-primary' : 'bg-gray-300'}`}
                        style={{ width: `${Math.round((count / maxRegimen) * 100)}%` }}
                      />
                    </div>
                    <span className={`text-xs font-semibold w-4 text-right ${primary ? 'text-gray-600' : 'text-gray-400'}`}>
                      {count}
                    </span>
                  </div>
                ))}
              </div>
            </section>

            {/* ── [5] Demographics ─────────────────────────────────── */}
            <section>
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Demographics</p>
              <div className="grid grid-cols-2 gap-3">

                {/* Sex */}
                <div className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Sex</p>
                  <div className="flex h-3 rounded-full overflow-hidden mb-3">
                    <div
                      className="bg-blue-400"
                      style={{ width: `${total > 0 ? Math.round((maleCount / total) * 100) : 50}%` }}
                    />
                    <div
                      className="bg-pink-400"
                      style={{ width: `${total > 0 ? Math.round((femaleCount / total) * 100) : 50}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-gray-600">
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
                      M — {maleCount}
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-pink-400 inline-block" />
                      F — {femaleCount}
                    </span>
                  </div>
                </div>

                {/* Age groups */}
                <div className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
                  <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Age Groups</p>
                  <div className="flex flex-col gap-2">
                    {ageBuckets.map(({ label, count }) => (
                      <div key={label} className="flex items-center gap-2">
                        <span className="text-xs text-gray-500 w-8">{label}</span>
                        <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${Math.round((count / maxAge) * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-500 w-3 text-right">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            </section>

            {/* ── [6] Risk Trends ──────────────────────────────────── */}
            <section className="bg-white rounded-xl border border-gray-100 p-4 shadow-sm">
              <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Risk Trends</p>

              {withTwoPreds.length === 0 ? (
                <p className="text-xs text-gray-400 italic">
                  Not enough data yet. Risk trends appear after a patient's second prediction.
                </p>
              ) : (
                <>
                  {/* mini-cards */}
                  <div className="grid grid-cols-3 gap-2 mb-4">
                    <div className="bg-green-50 border border-green-100 rounded-xl p-3 text-center">
                      <p className="text-lg font-bold text-green-600">↓ {improving}</p>
                      <p className="text-xs text-gray-500 mt-1">Improving</p>
                    </div>
                    <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 text-center">
                      <p className="text-lg font-bold text-gray-500">— {stable}</p>
                      <p className="text-xs text-gray-500 mt-1">Stable</p>
                    </div>
                    <div className="bg-red-50 border border-red-100 rounded-xl p-3 text-center">
                      <p className="text-lg font-bold text-red-600">↑ {worsening}</p>
                      <p className="text-xs text-gray-500 mt-1">Worsening</p>
                    </div>
                  </div>

                  {/* segmented bar */}
                  <div className="flex h-3 rounded-full overflow-hidden mb-2">
                    <div className="bg-green-400" style={{ width: `${(improving / trendTotal) * 100}%` }} />
                    <div className="bg-gray-300"  style={{ width: `${(stable    / trendTotal) * 100}%` }} />
                    <div className="bg-red-400"   style={{ width: `${(worsening / trendTotal) * 100}%` }} />
                  </div>

                  {singlePred > 0 && (
                    <p className="text-xs text-gray-400">
                      {singlePred} patient{singlePred !== 1 ? 's' : ''} excluded (only 1 prediction).
                    </p>
                  )}
                </>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  )
}
