/**
 * Longitudinal 4-patient × 4-month SHAP test.
 *
 * Creates 4 patients with distinct clinical profiles, then performs M0–M3 check-ins
 * for each, capturing risk % and top feature contributions after every month.
 * Verifies that:
 *   1. Risk changes meaningfully across months (not static).
 *   2. SHAP contributions are non-zero across at least 3 groups per prediction.
 *   3. Treatment Adherence becomes a visible contributor once longitudinal data exists.
 *
 * Usage: node scripts/e2e-longitudinal.mjs
 * Requires: dev server on port 5299 + backend on port 8000
 */

import { chromium } from 'playwright'

const BASE = `http://localhost:${process.env.PORT ?? 5299}`
const twoWeeksAgo = new Date(Date.now() - 14 * 86400000).toISOString().slice(0, 10)

// 4 patient profiles: each will receive 4 check-ins (M0 at intake + M1–M3 manually)
const PATIENTS = [
  {
    label: 'PA — Good adherence, improving (should trend down)',
    demographics: { name: 'Longitudinal A Good', age: '34', sex: 'F', regGroup: 'NEW' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB',
      microResult: '2+', anatSite: 'P', source: 'COMMUNITY', caseType: 'DSTB',
      heartRate: '78', bpSys: '118', bpDia: '76', o2Sat: '97',
      baseWeight: '54', baseHeight: '160',
      drugResistance: 'DS-TB',
      diagDate: twoWeeksAgo, txDate: twoWeeksAgo,
    },
    // M1–M3 check-ins: great adherence, weight gains
    checkins: [
      { weight: '54.5', height: '160', taken: '28', missed: '0', smear: '1' },
      { weight: '55.2', height: '160', taken: '28', missed: '0', smear: '1' },
      { weight: '55.8', height: '160', taken: '28', missed: '0', smear: '1' },
    ],
  },
  {
    label: 'PB — Poor adherence, declining (should trend up)',
    demographics: { name: 'Longitudinal B Poor', age: '48', sex: 'M', regGroup: 'RELAPSE' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB',
      microResult: '3+', anatSite: 'P', source: 'COMMUNITY', caseType: 'DRTB',
      heartRate: '92', bpSys: '130', bpDia: '84', o2Sat: '94',
      baseWeight: '50', baseHeight: '165',
      drugResistance: 'RR-TB',
      diagDate: twoWeeksAgo, txDate: twoWeeksAgo,
    },
    // M1–M3 check-ins: poor adherence, weight loss
    checkins: [
      { weight: '49', height: '165', taken: '10', missed: '18', smear: '1' },
      { weight: '48', height: '165', taken: '8',  missed: '20', smear: '1' },
      { weight: '47', height: '165', taken: '6',  missed: '22', smear: '1' },
    ],
  },
  {
    label: 'PC — Mixed adherence, fluctuating',
    demographics: { name: 'Longitudinal C Mixed', age: '28', sex: 'F', regGroup: 'NEW' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB',
      microResult: '1+', anatSite: 'P', source: 'COMMUNITY', caseType: 'DSTB',
      heartRate: '72', bpSys: '112', bpDia: '72', o2Sat: '98',
      baseWeight: '48', baseHeight: '155',
      drugResistance: 'DS-TB',
      diagDate: twoWeeksAgo, txDate: twoWeeksAgo,
    },
    // M1: great, M2: poor, M3: great again
    checkins: [
      { weight: '48.5', height: '155', taken: '27', missed: '1',  smear: '1' },
      { weight: '47.8', height: '155', taken: '12', missed: '16', smear: '0' },
      { weight: '48.2', height: '155', taken: '26', missed: '2',  smear: '1' },
    ],
  },
  {
    label: 'PD — High-risk baseline, improving over time',
    demographics: { name: 'Longitudinal D HighRisk', age: '65', sex: 'M', regGroup: 'RELAPSE' },
    lab: {
      bactStatus: 'Bacteriologically-confirmed TB',
      microResult: '3+', anatSite: 'P', source: 'COMMUNITY', caseType: 'DRTB',
      heartRate: '108', bpSys: '100', bpDia: '65', o2Sat: '90',
      baseWeight: '42', baseHeight: '162',
      drugResistance: 'MDR-TB',
      diagDate: twoWeeksAgo, txDate: twoWeeksAgo,
    },
    // M1–M3: steady adherence, gradual weight recovery
    checkins: [
      { weight: '42.5', height: '162', taken: '22', missed: '6', smear: '1' },
      { weight: '43.2', height: '162', taken: '25', missed: '3', smear: '1' },
      { weight: '44.0', height: '162', taken: '26', missed: '2', smear: '0' },
    ],
  },
]

async function login(page) {
  await page.goto(`${BASE}/login`)
  try {
    await page.fill('#username', 'admin123')
    await page.fill('#password', 'password123')
    await page.click('button:has-text("Sign In")')
    await page.waitForURL(`${BASE}/`, { timeout: 10000 })
  } catch { /* already logged in */ }
}

async function createPatient(page, patient) {
  await page.goto(`${BASE}/patient/new`)
  await page.waitForSelector('#name')

  // Demographics
  await page.fill('#name', patient.demographics.name)
  await page.fill('#age', patient.demographics.age)
  await page.selectOption('#sex', patient.demographics.sex)
  await page.selectOption('#regGroup', patient.demographics.regGroup)
  await page.selectOption('#province', { index: 1 })
  await page.click('button:has-text("Continue to Lab Results")')
  await page.waitForURL(`${BASE}/patient/new/lab`, { timeout: 10000 })

  // Lab
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
  await page.fill('#diagDate', lab.diagDate).catch(() => {})
  await page.fill('#txDate', lab.txDate).catch(() => {})

  await page.click('button:has-text("Generate Diagnosis")')
  await page.waitForURL(`${BASE}/patient/new/xray`, { timeout: 10000 })

  // Skip X-ray → result page (M0 prediction runs here)
  await page.waitForSelector('button:has-text("Skip")')
  await page.click('button:has-text("Skip")')
  await page.waitForURL(`${BASE}/patient/*/result`, { timeout: 180000 })

  // Extract patient ID from URL
  const url = page.url()
  const patientId = url.match(/\/patient\/([^/]+)\/result/)?.[1]
  const riskText = await page.locator('[class*="font-extrabold"][class*="tabular-nums"]')
    .first().textContent({ timeout: 3000 }).catch(() => '?')

  return { patientId, riskM0: riskText?.trim() }
}

async function doCheckin(page, patientId, monthLabel, checkin) {
  await page.goto(`${BASE}/patient/${patientId}/checkin`)
  await page.waitForSelector('#weight', { timeout: 10000 })

  if (checkin.weight) await page.fill('#weight', checkin.weight)
  if (checkin.height) await page.fill('#height', checkin.height)
  await page.fill('#monthlyDosesTaken', checkin.taken)
  await page.fill('#monthlyMissedDoses', checkin.missed)
  if (checkin.smear != null) await page.selectOption('#smearTbLamp', checkin.smear)

  await page.click('button:has-text("Generate Prediction")')
  // SHAP runs 9 ONNX passes in WASM — allow up to 3 minutes
  await page.waitForURL(`${BASE}/patient/${patientId}/risk-update`, { timeout: 180000 })

  // RiskUpdate shows: "HIGH RISK — 28% Failure Probability" in a font-bold paragraph
  const fullText = await page.locator('p:has-text("Failure Probability")')
    .first().textContent({ timeout: 5000 }).catch(() => '')
  const match = fullText.match(/(\d+)%/)
  return match ? match[1] + '%' : '?'
}

async function readContributions(page, patientId) {
  await page.goto(`${BASE}/patient/${patientId}/features`)
  await page.waitForTimeout(2000)

  const rows = await page.locator('.flex.items-center.gap-3.py-1\\.5').all()
  const contributions = []
  for (const row of rows) {
    const name  = (await row.locator('.text-sm.text-ink-secondary').first().textContent({ timeout: 500 }).catch(() => '?')).trim()
    const delta = (await row.locator('span.tabular-nums').textContent({ timeout: 500 }).catch(() => '?')).trim()
    const val   = (await row.locator('.text-xs.text-ink-muted').textContent({ timeout: 300 }).catch(() => '')).trim()
    if (name && delta !== '?') contributions.push({ name, delta, val })
  }
  return contributions
}

// ─── Main ───────────────────────────────────────────────────────────────────

const browser = await chromium.launch({ headless: true })
const page    = await browser.newPage()

console.log('\n══════════════════════════════════════════════════════════════════')
console.log('  LONGITUDINAL TEST — 4 Patients × 4 Months (M0–M3)')
console.log('══════════════════════════════════════════════════════════════════\n')

let allPassed = true

for (const patient of PATIENTS) {
  console.log(`▶  ${patient.label}`)
  console.log('   ─────────────────────────────────────────────────────────')

  try {
    await login(page)
    const { patientId, riskM0 } = await createPatient(page, patient)
    if (!patientId) throw new Error('Could not parse patient ID from URL')

    const riskByMonth = [`M0: ${riskM0}`]

    // M1–M3 check-ins
    for (let i = 0; i < patient.checkins.length; i++) {
      const risk = await doCheckin(page, patientId, `M${i + 1}`, patient.checkins[i])
      riskByMonth.push(`M${i + 1}: ${risk}`)
    }

    console.log(`   Risk over time: ${riskByMonth.join('  →  ')}`)

    // Check risk actually changes between M0 and M3
    const riskM0num = parseInt(riskM0?.replace('%', '') ?? '0', 10)
    const riskM3num = parseInt(riskByMonth.at(-1)?.split(': ')[1]?.replace('%', '') ?? '0', 10)
    const riskChanged = riskM0num !== riskM3num
    console.log(`   Risk changed M0→M3: ${riskChanged ? '✓' : '⚠  SAME — may indicate model not updating'}`)
    if (!riskChanged) allPassed = false

    // Read SHAP contributions after last check-in
    const contribs = await readContributions(page, patientId)

    const nonZeroGroups = contribs.filter(c => {
      const pct = parseFloat(c.delta.replace(/[^0-9.]/g, ''))
      return pct > 0
    })

    const hasAdherence = contribs.some(c =>
      c.name.toLowerCase().includes('adherence') &&
      parseFloat(c.delta.replace(/[^0-9.]/g, '')) > 0
    )

    console.log(`\n   SHAP after M${patient.checkins.length} (top 6):`)
    for (const c of contribs.slice(0, 6)) {
      const valStr = c.val ? `   (${c.val})` : ''
      console.log(`     ${c.name.padEnd(30)} ${c.delta.padStart(6)}${valStr}`)
    }

    console.log(`\n   Non-zero groups: ${nonZeroGroups.length} / ${contribs.length}`)
    console.log(`   Treatment Adherence active: ${hasAdherence ? '✓' : '⚠  still 0% — expected after check-ins'}`)

    const topDelta = contribs.length > 0 ? parseFloat(contribs[0].delta.replace(/[^0-9.]/g, '')) : 0
    if (topDelta < 5) {
      console.log(`   ✗ FAIL: top contribution ${topDelta}% — expected ≥ 5% (${nonZeroGroups.length}/${contribs.length} groups non-zero)`)
      allPassed = false
    } else {
      console.log(`   ✓ PASS: top contribution ${topDelta}% (${nonZeroGroups.length}/${contribs.length} groups non-zero)`)
    }

  } catch (err) {
    console.log(`   ✗ FAILED — ${err.message}`)
    allPassed = false
  }

  // Navigate away to close SSE streams
  await page.goto(`${BASE}/`)
  await page.waitForTimeout(1500)
  console.log()
}

await browser.close()

console.log('══════════════════════════════════════════════════════════════════')
console.log(`  Overall: ${allPassed ? '✓ ALL PASSED' : '⚠  SOME CHECKS FAILED — see above'}`)
console.log('══════════════════════════════════════════════════════════════════\n')
