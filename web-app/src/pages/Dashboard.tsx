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

            {/* sections will be added in subsequent tasks */}
          </>
        )}
      </main>
    </div>
  )
}
