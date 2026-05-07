# Dashboard Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `/dashboard` page to the TB-DOTS CDSS web app that shows six sections of patient statistics (risk distribution, adherence, regimens, demographics, risk trends) derived from existing localStorage data.

**Architecture:** Single new `Dashboard.tsx` page component, registered as a route in `App.tsx` and linked from the sidebar in `DesktopLayout.tsx`. All stats are computed inline from `getAllPatients()`. No new dependencies — visualizations use pure CSS/Tailwind horizontal bars with inline `style={{ width: \`${pct}%\` }}`, matching the existing `FeatureBar` pattern.

**Tech Stack:** React 18, TypeScript 5, Vite 5, TailwindCSS 3, React Router v6, `onnxruntime-web` (not touched). No test runner — use `npm run build` (tsc + vite) as type-safety gate and `npm run dev` for visual verification.

**Spec:** `docs/superpowers/specs/2026-05-07-dashboard-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `web-app/src/pages/Dashboard.tsx` | **Create** | Full dashboard page — all sections, stat computation, empty state |
| `web-app/src/App.tsx` | **Modify** | Import `Dashboard`, add `/dashboard` route |
| `web-app/src/components/DesktopLayout.tsx` | **Modify** | Add Dashboard nav item between Patient Overview and New Patient |

---

## Task 1: Wire up route and nav item

**Files:**
- Modify: `web-app/src/App.tsx`
- Modify: `web-app/src/components/DesktopLayout.tsx`
- Create: `web-app/src/pages/Dashboard.tsx` (stub only)

- [ ] **Step 1: Create Dashboard stub**

Create `web-app/src/pages/Dashboard.tsx` with this exact content:

```tsx
export function Dashboard() {
  return <div>Dashboard coming soon</div>
}
```

- [ ] **Step 2: Add route in App.tsx**

Open `web-app/src/App.tsx`. Add the import and route:

```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Home } from './pages/Home'
import { PatientIntakeStep1 } from './pages/PatientIntakeStep1'
import { PatientIntakeStep2 } from './pages/PatientIntakeStep2'
import { DiagnosticResult } from './pages/DiagnosticResult'
import { TreatmentSelection } from './pages/TreatmentSelection'
import { PatientProfile } from './pages/PatientProfile'
import { MonthlyCheckin } from './pages/MonthlyCheckin'
import { RiskUpdate } from './pages/RiskUpdate'
import { FeatureContribution } from './pages/FeatureContribution'
import { Dashboard } from './pages/Dashboard'
import { DesktopLayout } from './components/DesktopLayout'

export default function App() {
  return (
    <BrowserRouter>
      <DesktopLayout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/patient/new" element={<PatientIntakeStep1 />} />
          <Route path="/patient/new/lab" element={<PatientIntakeStep2 />} />
          <Route path="/patient/:id" element={<PatientProfile />} />
          <Route path="/patient/:id/result" element={<DiagnosticResult />} />
          <Route path="/patient/:id/treatment" element={<TreatmentSelection />} />
          <Route path="/patient/:id/checkin" element={<MonthlyCheckin />} />
          <Route path="/patient/:id/risk-update" element={<RiskUpdate />} />
          <Route path="/patient/:id/features" element={<FeatureContribution />} />
        </Routes>
      </DesktopLayout>
    </BrowserRouter>
  )
}
```

- [ ] **Step 3: Add nav item in DesktopLayout.tsx**

Open `web-app/src/components/DesktopLayout.tsx`. Replace the `NAV_ITEMS` array (lines 7-10):

```tsx
const NAV_ITEMS = [
  { path: '/', label: 'Patient Overview', icon: '🫁' },
  { path: '/dashboard', label: 'Dashboard', icon: '📊' },
  { path: '/patient/new', label: 'New Patient', icon: '➕' },
]
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd web-app && npm run build
```

Expected: build succeeds with no type errors.

- [ ] **Step 5: Visual check**

```bash
npm run dev
```

Open http://localhost:5173. Confirm:
- Sidebar shows "Dashboard" between "Patient Overview" and "New Patient"
- Clicking Dashboard highlights it and shows "Dashboard coming soon"

- [ ] **Step 6: Commit**

```bash
cd web-app && git add src/pages/Dashboard.tsx src/App.tsx src/components/DesktopLayout.tsx
git commit -m "feat: add dashboard route and nav item"
```

---

## Task 2: Page shell, data computation, and empty state

**Files:**
- Modify: `web-app/src/pages/Dashboard.tsx`

This task replaces the stub with the full page shell: imports, data loading, all stat computations, and the empty state. No visual sections yet — just the structure that every later task builds on.

- [ ] **Step 1: Replace stub with full shell**

Replace the entire contents of `web-app/src/pages/Dashboard.tsx`:

```tsx
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
            {/* sections will be added in subsequent tasks */}
          </>
        )}
      </main>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd web-app && npm run build
```

Expected: build succeeds. All computed variables are typed correctly.

- [ ] **Step 3: Visual check — empty state**

```bash
npm run dev
```

Open http://localhost:5173/dashboard with empty localStorage (open DevTools → Application → Local Storage → clear `tb_patients`). Confirm:
- AppHeader shows "Dashboard"
- Empty state shows 📊 icon and "No patient data yet"

- [ ] **Step 4: Commit**

```bash
cd web-app && git add src/pages/Dashboard.tsx
git commit -m "feat: dashboard page shell with data computation and empty state"
```

---

## Task 3: Summary stat cards

**Files:**
- Modify: `web-app/src/pages/Dashboard.tsx`

Add the 6-card summary grid inside the `<>...</>` fragment. Each card shows a large bold number and a small gray label.

- [ ] **Step 1: Replace the `<>` fragment placeholder**

In `web-app/src/pages/Dashboard.tsx`, replace:
```tsx
          <>
            {/* sections will be added in subsequent tasks */}
          </>
```

with:

```tsx
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
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd web-app && npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Visual check**

Add a patient through the intake flow (or paste test data into localStorage). Open http://localhost:5173/dashboard. Confirm:
- 6 cards render in a 2-col grid (mobile) / 3-col (desktop ≥1024px)
- High Risk card has red tint, Due Check-in has yellow tint, Full Adherence has green tint
- Avg Risk shows `—` if no predictions exist on any patient

- [ ] **Step 4: Commit**

```bash
cd web-app && git add src/pages/Dashboard.tsx
git commit -m "feat: dashboard summary stat cards"
```

---

## Task 4: Risk distribution section

**Files:**
- Modify: `web-app/src/pages/Dashboard.tsx`

- [ ] **Step 1: Add Risk Distribution section**

In `web-app/src/pages/Dashboard.tsx`, replace the comment `{/* sections will be added in subsequent tasks */}` with:

```tsx
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

            {/* sections will be added in subsequent tasks */}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd web-app && npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Visual check**

Open http://localhost:5173/dashboard. Confirm:
- Stacked bar shows three colored segments proportional to HIGH/MED/LOW counts
- Counts appear inside segments that are wide enough (>15% of total)
- Legend shows colored dots with counts

- [ ] **Step 4: Commit**

```bash
cd web-app && git add src/pages/Dashboard.tsx
git commit -m "feat: dashboard risk distribution section"
```

---

## Task 5: Adherence overview section

**Files:**
- Modify: `web-app/src/pages/Dashboard.tsx`

- [ ] **Step 1: Add Adherence Overview section**

Replace `{/* sections will be added in subsequent tasks */}` with:

```tsx
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

            {/* sections will be added in subsequent tasks */}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd web-app && npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Visual check**

Open http://localhost:5173/dashboard. Confirm:
- Explanation text appears with colored inline labels (green/orange/red)
- Three bar rows (Full / Partial / Poor) render with proportional widths and percentages
- "No monthly check-in records yet." appears when no patients have monthly records

- [ ] **Step 4: Commit**

```bash
cd web-app && git add src/pages/Dashboard.tsx
git commit -m "feat: dashboard adherence overview section"
```

---

## Task 6: Treatment regimens section

**Files:**
- Modify: `web-app/src/pages/Dashboard.tsx`

- [ ] **Step 1: Add Treatment Regimens section**

Replace `{/* sections will be added in subsequent tasks */}` with:

```tsx
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

            {/* sections will be added in subsequent tasks */}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd web-app && npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Visual check**

Open http://localhost:5173/dashboard. Confirm:
- Four rows render (HRZE / MDR-TB / XDR-TB / Unassigned)
- HRZE/MDR/XDR bars use primary green, Unassigned uses gray
- Widths are relative to the largest group (largest fills 100% of track)

- [ ] **Step 4: Commit**

```bash
cd web-app && git add src/pages/Dashboard.tsx
git commit -m "feat: dashboard treatment regimens section"
```

---

## Task 7: Demographics section

**Files:**
- Modify: `web-app/src/pages/Dashboard.tsx`

- [ ] **Step 1: Add Demographics section**

Replace `{/* sections will be added in subsequent tasks */}` with:

```tsx
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

            {/* sections will be added in subsequent tasks */}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd web-app && npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Visual check**

Open http://localhost:5173/dashboard. Confirm:
- Sex card: two-tone bar (blue for M, pink for F) + legend with counts
- Age Groups card: 5 horizontal bar rows (0–17, 18–34, 35–49, 50–64, 65+) with proportional widths
- Both cards are side-by-side in a 2-col grid

- [ ] **Step 4: Commit**

```bash
cd web-app && git add src/pages/Dashboard.tsx
git commit -m "feat: dashboard demographics section"
```

---

## Task 8: Risk trends section and final cleanup

**Files:**
- Modify: `web-app/src/pages/Dashboard.tsx`

- [ ] **Step 1: Add Risk Trends section**

Replace `{/* sections will be added in subsequent tasks */}` with:

```tsx
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
```

- [ ] **Step 2: Remove the placeholder comment**

After adding the Risk Trends section, delete the line:
```tsx
            {/* sections will be added in subsequent tasks */}
```

There should be no placeholder comments remaining in the file.

- [ ] **Step 3: Verify TypeScript compiles cleanly**

```bash
cd web-app && npm run build
```

Expected: build succeeds with no type errors or warnings.

- [ ] **Step 4: Full visual walkthrough**

```bash
npm run dev
```

Test with **empty localStorage:**
- `/dashboard` → empty state (📊 icon + message)
- Sidebar "Dashboard" item is clickable and highlights correctly

Test with **patients in localStorage** (add via intake flow or paste directly):
- All 6 sections render
- Summary cards: correct counts and percentages
- Risk distribution: stacked bar segments proportional, counts inside wide segments
- Adherence: explanation text visible, bars proportional, footer record count
- Regimens: bars relative to largest group, Unassigned is gray
- Demographics: sex two-tone bar, age buckets proportional
- Risk trends: mini-cards with ↓/—/↑, excluded patient count if any

Test **responsive layout:**
- Mobile width (<1024px): summary cards in 2-col grid
- Desktop width (≥1024px): summary cards in 3-col grid, sidebar visible

Test **existing pages still work:**
- `/` Home, `/patient/new`, a patient profile — all render without errors

- [ ] **Step 5: Final commit**

```bash
cd web-app && git add src/pages/Dashboard.tsx
git commit -m "feat: dashboard risk trends section — dashboard complete"
```
