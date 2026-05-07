# Dashboard Page — Design Spec

**Date:** 2026-05-07
**Branch:** dev/feat/dashboard
**Status:** Approved

---

## Context

The TB-DOTS CDSS web app currently has a Home page (`/`) showing a patient list sorted by risk. Healthcare workers have no at-a-glance view of their caseload's aggregate health — how many patients are high risk, how adherence is trending, which regimens are in use, or how the population is distributed by age/sex. This dashboard fills that gap.

**Audience:** Healthcare workers managing their TB caseload.
**Goal:** Actionable population-level overview — "what's the state of my patients right now?"

---

## What We're Building

A new **Dashboard page** at `/dashboard`, accessible from the sidebar nav. It shows six sections of statistics derived entirely from existing `localStorage` patient data. No backend, no new dependencies, no changes to the data model.

---

## Route & Navigation

- **New route:** `/dashboard` → `<Dashboard />` component
- **Sidebar nav item added** between "Patient Overview" and "New Patient":
  ```
  { path: '/dashboard', label: 'Dashboard', icon: '📊' }
  ```
- Active state handled automatically by existing `location.pathname === item.path` logic in `DesktopLayout.tsx`.

---

## Page Structure

```
AppHeader (title="Dashboard")

<main> px-4 pt-4 pb-6, flex-col gap-4

  Page title: "Population Overview"
  Subtitle: "Based on N registered patients"

  [1] Summary Cards       — 3-col grid, 6 cards
  [2] Risk Distribution   — stacked horizontal bar
  [3] Adherence Overview  — explanation text + 3 horizontal bars
  [4] Treatment Regimens  — 4 horizontal bars (HRZE / MDR / XDR / Unassigned)
  [5] Demographics        — 2-col grid: Sex bar + Age group bars
  [6] Risk Trends         — 3 mini-cards + segmented bar

</main>
```

**Empty state:** If `patients.length === 0`, show a centered message ("No patient data yet. Register patients to see dashboard analytics.") and skip all sections.

---

## Section Specs

### Helper

```ts
const latestProb = (p: Patient) => p.predictions.at(-1)?.failureProbability ?? 0
```

---

### [1] Summary Cards

Six cards in a `grid-cols-2 lg:grid-cols-3` grid:

| Card | Value | Accent |
|---|---|---|
| Total Patients | `patients.length` | primary green |
| High Risk | patients where `riskLabel(latestProb(p)) === 'HIGH'` | red (`bg-red-50 border-red-200`) |
| Due Check-in | patients where `monthlyRecords.length < 6` | yellow (`bg-yellow-50 border-yellow-200`) |
| Avg Risk | mean of `latestProb` across all patients, as % — show `—` if no patients | dynamic via `riskColor()` on the computed mean |
| Full Adherence | `fullRecords / totalRecords` as % — show `—` if no monthly records exist | green (`bg-green-50 border-green-100`) |
| On Treatment | patients where `treatmentRegimen !== undefined` | primary green |

Each card: `bg-white rounded-xl border p-4 text-center` with a large bold number and a small gray label below.

Guard: if `patients.length === 0`, all values show `—`.

---

### [2] Risk Distribution

Stacked single horizontal bar (`h-4 rounded-full flex overflow-hidden`) with three segments:
- Red (`bg-red-400`) — HIGH count
- Orange (`bg-orange-400`) — MED count
- Green (`bg-green-400`) — LOW count

Segment widths: `(count / patients.length) * 100`%

Legend below: three colored dots with labels "HIGH — N", "MED — N", "LOW — N".

Counts inside each bar segment (white, `text-xs font-bold`) when segment is wide enough (>15%).

---

### [3] Adherence Overview

**Section header:** `ADHERENCE OVERVIEW`

**Short explanation** (below header, `text-xs text-gray-500 leading-relaxed mb-3`):
> Tracks whether patients took their medication as prescribed during monthly check-ins. **Full** = all doses taken, **Partial** = some missed, **Poor** = frequently missed.

Flatten all `patient.monthlyRecords` across all patients. Count by `adherence` value.

Three bar rows (Full / Partial / Poor):
- Bar colors: green-400 / orange-400 / red-400
- Bar widths: `(count / totalRecords) * 100`%
- Each row: label (w-10) → track → percentage

Footer note: `text-xs text-gray-400` — "Across N total monthly check-in records"

**Empty state:** If `totalRecords === 0`, show "No monthly check-in records yet."

---

### [4] Treatment Regimens

Four bar rows: HRZE / MDR-TB / XDR-TB / Unassigned

Group patients by `patient.treatmentRegimen` (`'hrze'` | `'mdr'` | `'xdr'` | `undefined`).

Bar widths relative to the largest group (largest = 100%). Color: `bg-primary` for assigned, `bg-gray-300` for Unassigned.

Each row: label (w-16) → track → count.

---

### [5] Demographics

Two cards side-by-side (`grid-cols-2 gap-3`):

**Sex:**
- Two-tone bar: blue-400 (Male) / pink-400 (Female)
- Count Male: `p.features.sex === 'M'`, count Female: `p.features.sex === 'F'` (explicit, not assumed)
- Legend below: "M — N" / "F — N"

**Age Groups:**
- Bucket `p.features.age` into: 0–17 / 18–34 / 35–49 / 50–64 / 65+
- Horizontal bars, width relative to largest bucket
- Color: `bg-primary`
- Rows: label → bar → count

---

### [6] Risk Trends

For patients with `predictions.length >= 2`, compute:
```ts
const delta = predictions.at(-1).failureProbability - predictions.at(-2).failureProbability
```

- **Improving:** `delta < -0.05` → green, ↓
- **Stable:** `|delta| <= 0.05` → gray, —
- **Worsening:** `delta > 0.05` → red, ↑

Three mini-cards (`grid-cols-3`) showing count with arrow + color.

Segmented bar below (green / gray / red proportional to counts).

Footer note: `text-xs text-gray-400` — "N patients excluded (only 1 prediction)."

**Empty state:** If no patients have ≥2 predictions, show "Not enough data yet. Risk trends appear after a patient's second prediction."

---

## Design System

Follows existing app patterns exactly:

- **Cards:** `bg-white rounded-xl border border-gray-100 p-4 shadow-sm`
- **Section headers:** `text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3`
- **Bar tracks:** `bg-gray-100 rounded-full h-3 overflow-hidden`
- **Primary color:** `bg-primary` (#0d6b4e), `text-primary`
- **Risk colors:** red-400/red-600, orange-400/orange-600, green-400/green-600
- **Typography:** `text-2xl font-bold` for stat numbers, `text-xs text-gray-500` for labels

No new npm packages. All visualizations are pure CSS/Tailwind with inline `style={{ width: \`${pct}%\` }}`.

---

## Files

| File | Action |
|---|---|
| `web-app/src/pages/Dashboard.tsx` | Create |
| `web-app/src/App.tsx` | Add import + `/dashboard` route |
| `web-app/src/components/DesktopLayout.tsx` | Add Dashboard nav item |

**Reused (no changes):**
- `getAllPatients()` from `lib/storage.ts`
- `riskLabel()`, `riskColor()` from `components/RiskBadge.tsx`
- `AppHeader` from `components/AppHeader.tsx`
- `Patient`, `MonthlyRecord`, `PredictionRecord` types from `lib/storage.ts`

---

## Verification

1. `cd web-app && npm run build` — TypeScript must compile clean
2. `npm run dev` — open browser:
   - With empty localStorage: empty state message shows, no sections
   - Add patients via intake flow, then open `/dashboard`
   - Verify all 6 sections render with correct values
   - Resize to mobile: stat cards show 2-col grid, bars fill full width
   - Resize to desktop: stat cards show 3-col grid, sidebar highlights "Dashboard"
   - Verify existing routes (Home, PatientProfile, etc.) still work
