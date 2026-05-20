/**
 * Fully-populated patient matrix test.
 *
 * Tests 4 patients with ALL optional fields filled in (BP, X-ray, co-morbidities,
 * facility, dates, Xpert result) to see if risk prediction changes from 55%.
 *
 * Usage: node scripts/e2e-shap-fullyloaded.mjs
 * Requires: dev server on port 5299 + backend on port 8000
 */

import { chromium } from 'playwright'

const BASE = `http://localhost:${process.env.PORT ?? 5299}`

// Today and plausible treatment dates
const today = new Date().toISOString().slice(0, 10)
const twoWeeksAgo = new Date(Date.now() - 14 * 86400000).toISOString().slice(0, 10)

const PATIENTS = [
  {
    label: 'FA — Fully loaded low-risk (DS-TB, new, excellent vitals)',
    demographics: { name: 'Full A Low Risk', age: '28', sex: 'F', regGroup: 'NEW' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB',
      microResult: '1+', anatSite: 'P', source: 'COMMUNITY', caseType: 'DSTB',
      heartRate: '68', bpSys: '110', bpDia: '70', o2Sat: '99',
      baseWeight: '55', baseHeight: '160',
      drugResistance: 'DS-TB',
      xpertMtbRif: 'Negative',
      chestXRay: 'Suggestive of PTB',
      coMorbidities: 'None',
      txFac: 'PACDAL RURAL HEALTH UNIT - IDOTS',
      diagDate: twoWeeksAgo, txDate: twoWeeksAgo,
    },
  },
  {
    label: 'FB — Fully loaded moderate-risk (DS-TB, relapse, mild tachycardia)',
    demographics: { name: 'Full B Moderate', age: '52', sex: 'M', regGroup: 'RELAPSE' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB',
      microResult: '2+', anatSite: 'P', source: 'COMMUNITY', caseType: 'DSTB',
      heartRate: '92', bpSys: '130', bpDia: '85', o2Sat: '94',
      baseWeight: '55', baseHeight: '165',
      drugResistance: 'DS-TB',
      xpertMtbRif: 'Positive',
      chestXRay: 'Suggestive of PTB',
      coMorbidities: 'Diabetes',
      txFac: 'QUIRINO HILL RURAL HEALTH UNIT - IDOTS',
      diagDate: twoWeeksAgo, txDate: twoWeeksAgo,
    },
  },
  {
    label: 'FC — Fully loaded high-risk (RR-TB, relapse, elderly, poor vitals, CKD)',
    demographics: { name: 'Full C High Risk', age: '70', sex: 'M', regGroup: 'RELAPSE' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB',
      microResult: '3+', anatSite: 'P', source: 'COMMUNITY', caseType: 'DRTB',
      heartRate: '118', bpSys: '95', bpDia: '60', o2Sat: '86',
      baseWeight: '40', baseHeight: '162',
      drugResistance: 'RR-TB',
      xpertMtbRif: 'Positive',
      chestXRay: 'Suggestive of PTB',
      coMorbidities: 'KidneyDisease',
      txFac: 'BAGUIO GENERAL HOSPITAL AND MEDICAL CENTER - PMDT TC',
      diagDate: twoWeeksAgo, txDate: twoWeeksAgo,
    },
  },
  {
    label: 'FD — Fully loaded very high-risk (MDR-TB, relapse, HIV, hypoxic)',
    demographics: { name: 'Full D Very High', age: '45', sex: 'M', regGroup: 'RELAPSE' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB',
      microResult: '3+', anatSite: 'P', source: 'COMMUNITY', caseType: 'DRTB',
      heartRate: '125', bpSys: '90', bpDia: '55', o2Sat: '82',
      baseWeight: '38', baseHeight: '168',
      drugResistance: 'MDR-TB',
      xpertMtbRif: 'Positive',
      chestXRay: 'Suggestive of PTB',
      coMorbidities: 'HIV',
      txFac: 'BAGUIO GENERAL HOSPITAL AND MEDICAL CENTER - PMDT TC',
      diagDate: twoWeeksAgo, txDate: twoWeeksAgo,
    },
  },
]

async function runPatient(page, patient) {
  await page.goto(`${BASE}/login`)
  try {
    await page.fill('#username', 'admin123')
    await page.fill('#password', 'password123')
    await page.click('button:has-text("Sign In")')
    await page.waitForURL(`${BASE}/`, { timeout: 10000 })
  } catch { /* already logged in */ }

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

  // Lab — fill every available field
  await page.waitForSelector('#bactStatus')
  const lab = patient.lab
  await page.selectOption('#bactStatus', lab.bactStatus)
  await page.selectOption('#microResult', lab.microResult)
  await page.selectOption('#anatSite', lab.anatSite)
  await page.selectOption('#source', lab.source)
  await page.selectOption('#caseType', lab.caseType)
  await page.fill('#heartRate', lab.heartRate)
  await page.fill('#bpSys', lab.bpSys)
  await page.fill('#bpDia', lab.bpDia)
  await page.fill('#o2Sat', lab.o2Sat)
  await page.fill('#baseWeight', lab.baseWeight)
  await page.fill('#baseHeight', lab.baseHeight)
  await page.selectOption('#drugResistance', lab.drugResistance).catch(() => {})
  await page.selectOption('#xpertMtbRif', lab.xpertMtbRif).catch(() => {})
  await page.selectOption('#chestXRay', lab.chestXRay).catch(() => {})
  await page.selectOption('#coMorbidities', lab.coMorbidities).catch(() => {})
  await page.selectOption('#txFac', lab.txFac).catch(() => {})
  await page.fill('#diagDate', lab.diagDate).catch(() => {})
  await page.fill('#txDate', lab.txDate).catch(() => {})

  await page.click('button:has-text("Generate Diagnosis")')
  await page.waitForURL(`${BASE}/patient/new/xray`, { timeout: 10000 })

  // Skip X-ray
  await page.waitForSelector('button:has-text("Skip")')
  await page.click('button:has-text("Skip")')
  await page.waitForURL(`${BASE}/patient/*/result`, { timeout: 60000 })

  // Read overall risk
  const riskText = await page.locator('[class*="font-extrabold"][class*="tabular-nums"]').first().textContent({ timeout: 3000 }).catch(() => '?')

  // Feature contributions
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

  // Navigate away to close SSE streams before next patient
  await page.goto(`${BASE}/`)
  await page.waitForTimeout(1500)

  return { riskText: riskText?.trim(), contributions }
}

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage()

console.log('\n══════════════════════════════════════════════════════════════════')
console.log('  FULLY-POPULATED PATIENT TEST — All Optional Fields Filled')
console.log('══════════════════════════════════════════════════════════════════\n')

for (const patient of PATIENTS) {
  process.stdout.write(`Running ${patient.label}...\n  `)
  try {
    const { riskText, contributions } = await runPatient(page, patient)
    console.log(`Risk: ${riskText}`)
    for (const c of contributions.slice(0, 6)) {
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
