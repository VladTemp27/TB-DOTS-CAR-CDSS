/**
 * Multi-patient SHAP matrix test.
 *
 * Runs 6 clinically distinct patient profiles and prints a comparison table
 * of displayed risk + top 3 feature contributions for each.
 *
 * Usage: node scripts/e2e-shap-matrix.mjs
 * Requires: dev server on port 5299 + backend on port 8000
 */

import { chromium } from 'playwright'

const BASE = `http://localhost:${process.env.PORT ?? 5299}`

const PATIENTS = [
  {
    label: 'P1 — Low risk (DS-TB, new, good vitals)',
    demographics: { name: 'Matrix P1 Low Risk', age: '35', sex: 'M', regGroup: 'NEW' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB', microResult: '1+', anatSite: 'P',
      source: 'COMMUNITY', caseType: 'DSTB',
      heartRate: '72', o2Sat: '98', baseWeight: '65', baseHeight: '168',
      drugResistance: 'DS-TB',
    },
  },
  {
    label: 'P2 — Moderate risk (DS-TB, relapse, average vitals)',
    demographics: { name: 'Matrix P2 Relapse', age: '50', sex: 'F', regGroup: 'RELAPSE' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB', microResult: '2+', anatSite: 'P',
      source: 'COMMUNITY', caseType: 'DSTB',
      heartRate: '85', o2Sat: '95', baseWeight: '52', baseHeight: '158',
      drugResistance: 'DS-TB',
    },
  },
  {
    label: 'P3 — High risk (RR-TB, relapse, elderly, poor vitals)',
    demographics: { name: 'Matrix P3 High Risk', age: '72', sex: 'M', regGroup: 'RELAPSE' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB', microResult: '3+', anatSite: 'P',
      source: 'COMMUNITY', caseType: 'DRTB',
      heartRate: '115', o2Sat: '88', baseWeight: '42', baseHeight: '165',
      drugResistance: 'RR-TB',
    },
  },
  {
    label: 'P4 — Clinically diagnosed, young female, no vitals',
    demographics: { name: 'Matrix P4 Clinical', age: '22', sex: 'F', regGroup: 'NEW' },
    lab: {
      bactStatus: 'Clinically-diagnosed TB', microResult: 'Negative', anatSite: 'P',
      source: 'COMMUNITY', caseType: 'DSTB',
      heartRate: '', o2Sat: '', baseWeight: '48', baseHeight: '155',
      drugResistance: 'DS-TB',
    },
  },
  {
    label: 'P5 — MDR-TB, middle-aged, tachycardia',
    demographics: { name: 'Matrix P5 MDR', age: '45', sex: 'M', regGroup: 'RELAPSE' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB', microResult: '2+', anatSite: 'P',
      source: 'COMMUNITY', caseType: 'DRTB',
      heartRate: '108', o2Sat: '92', baseWeight: '55', baseHeight: '170',
      drugResistance: 'MDR-TB',
    },
  },
  {
    label: 'P6 — Extrapulmonary, DS-TB, new, healthy vitals',
    demographics: { name: 'Matrix P6 Extrapulm', age: '30', sex: 'F', regGroup: 'NEW' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB', microResult: '1+', anatSite: 'EP',
      source: 'COMMUNITY', caseType: 'DSTB',
      heartRate: '68', o2Sat: '99', baseWeight: '58', baseHeight: '162',
      drugResistance: 'DS-TB',
    },
  },
]

async function runPatient(page, patient) {
  // Login (session persists; will be fast after first time)
  await page.goto(`${BASE}/login`)
  try {
    await page.fill('#username', 'admin123')
    await page.fill('#password', 'password123')
    await page.click('button:has-text("Sign In")')
    await page.waitForURL(`${BASE}/`, { timeout: 8000 })
  } catch {
    // Already logged in
  }

  // Demographics
  await page.goto(`${BASE}/patient/new`)
  await page.waitForSelector('#name')
  await page.fill('#name', patient.demographics.name)
  await page.fill('#age', patient.demographics.age)
  await page.selectOption('#sex', patient.demographics.sex)
  await page.selectOption('#regGroup', patient.demographics.regGroup)
  await page.selectOption('#province', { index: 1 })
  await page.click('button:has-text("Continue to Lab Results")')
  await page.waitForURL(`${BASE}/patient/new/lab`, { timeout: 10000 })

  // Lab
  await page.waitForSelector('#bactStatus')
  await page.selectOption('#bactStatus', patient.lab.bactStatus)
  await page.selectOption('#microResult', patient.lab.microResult)
  await page.selectOption('#anatSite', patient.lab.anatSite)
  await page.selectOption('#source', patient.lab.source)
  await page.selectOption('#caseType', patient.lab.caseType)
  if (patient.lab.heartRate) await page.fill('#heartRate', patient.lab.heartRate)
  if (patient.lab.o2Sat) await page.fill('#o2Sat', patient.lab.o2Sat)
  if (patient.lab.baseWeight) await page.fill('#baseWeight', patient.lab.baseWeight)
  if (patient.lab.baseHeight) await page.fill('#baseHeight', patient.lab.baseHeight)
  try { await page.selectOption('#drugResistance', patient.lab.drugResistance) } catch {}
  await page.click('button:has-text("Generate Diagnosis")')
  await page.waitForURL(`${BASE}/patient/new/xray`, { timeout: 10000 })

  // Skip X-ray, commit
  await page.waitForSelector('button:has-text("Skip")')
  await page.click('button:has-text("Skip")')
  await page.waitForURL(`${BASE}/patient/*/result`, { timeout: 60000 })

  // Read overall risk from result page
  const riskText = await page.locator('[class*="font-extrabold"][class*="tabular-nums"]').first().textContent({ timeout: 3000 }).catch(() => '?')

  // Navigate to feature contributions
  await page.click('button:has-text("View feature contributions")')
  await page.waitForURL(`${BASE}/patient/*/features`, { timeout: 10000 })
  await page.waitForTimeout(2000)

  const rows = await page.locator('.flex.items-center.gap-3.py-1\\.5').all()
  const contributions = []
  for (const row of rows) {
    const name = (await row.locator('.text-sm.text-ink-secondary').first().textContent({ timeout: 500 }).catch(() => '?')).trim()
    const delta = (await row.locator('span.tabular-nums').textContent({ timeout: 500 }).catch(() => '?')).trim()
    const val = (await row.locator('.text-xs.text-ink-muted').textContent({ timeout: 300 }).catch(() => '')).trim()
    if (name && delta !== '?') contributions.push({ name, delta, val })
  }

  // Navigate away to close any open SSE / LLM streams before the next patient.
  await page.goto(`${BASE}/`)
  await page.waitForTimeout(1500)

  return { riskText: riskText?.trim(), contributions }
}

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage()

console.log('\n══════════════════════════════════════════════════════════════════')
console.log('  MULTI-PATIENT SHAP MATRIX — Fresh DB')
console.log('══════════════════════════════════════════════════════════════════\n')

for (const patient of PATIENTS) {
  process.stdout.write(`Running ${patient.label}... `)
  try {
    const { riskText, contributions } = await runPatient(page, patient)
    console.log(`Risk: ${riskText}`)
    const top = contributions.slice(0, 5)
    for (const c of top) {
      const valStr = c.val ? `   (${c.val})` : ''
      console.log(`    ${c.name.padEnd(32)} ${c.delta.padStart(7)}${valStr}`)
    }
  } catch (err) {
    console.log(`FAILED — ${err.message}`)
  }
  console.log()
}

await browser.close()
console.log('══════════════════════════════════════════════════════════════════')
