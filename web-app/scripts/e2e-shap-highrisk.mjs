/**
 * High-risk patient SHAP test — verifies ablation produces meaningful non-zero
 * deltas when the model's baseline failure probability is NOT near zero.
 *
 * Uses: Relapse patient, RR-TB, poor vitals (high HR, low O2), elderly.
 *
 * Usage (requires dev server on port 5299 + backend on port 8000):
 *   node scripts/e2e-shap-highrisk.mjs
 */

import { chromium } from 'playwright'

const BASE = `http://localhost:${process.env.PORT ?? 5299}`

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage()

const probes = []
page.on('console', msg => {
  const t = msg.text()
  if (t.includes('[SHAP-PROBE]')) probes.push(t)
})

try {
  // Login
  process.stdout.write('1. Login... ')
  await page.goto(`${BASE}/login`)
  await page.fill('#username', 'admin123')
  await page.fill('#password', 'password123')
  await page.click('button:has-text("Sign In")')
  await page.waitForURL(`${BASE}/`, { timeout: 10000 })
  console.log('OK')

  // Step 1: Demographics — high-risk profile
  process.stdout.write('2. Demographics (high-risk: elderly, relapse)... ')
  await page.goto(`${BASE}/patient/new`)
  await page.waitForSelector('#name')
  await page.fill('#name', 'E2E High Risk SHAP Test')
  await page.fill('#age', '72')           // elderly
  await page.selectOption('#sex', 'M')
  await page.selectOption('#regGroup', 'RELAPSE')  // relapse = high risk
  await page.selectOption('#province', { index: 1 })
  await page.click('button:has-text("Continue to Lab Results")')
  await page.waitForURL(`${BASE}/patient/new/lab`, { timeout: 10000 })
  console.log('OK')

  // Step 2: Lab — poor vitals, drug-resistant
  process.stdout.write('3. Lab data (poor vitals, drug-resistant)... ')
  await page.waitForSelector('#bactStatus')
  await page.selectOption('#bactStatus', 'Bacteriologically-confirmed TB')
  await page.selectOption('#microResult', '3+')   // high bacillary load
  await page.selectOption('#anatSite', 'P')
  await page.selectOption('#source', 'COMMUNITY')
  await page.selectOption('#caseType', 'DRTB')    // drug-resistant TB type
  // Poor vitals
  await page.fill('#heartRate', '115')   // tachycardia
  await page.fill('#o2Sat', '88')        // hypoxic
  await page.fill('#baseWeight', '42')   // underweight
  await page.fill('#baseHeight', '165')
  await page.selectOption('#drugResistance', 'RR-TB')  // rifampicin-resistant
  await page.click('button:has-text("Generate Diagnosis")')
  await page.waitForURL(`${BASE}/patient/new/xray`, { timeout: 10000 })
  console.log('OK')

  // Skip X-ray, commit patient
  process.stdout.write('4. Committing patient (running M0 prediction)... ')
  await page.waitForSelector('button:has-text("Skip")')
  await page.click('button:has-text("Skip")')
  await page.waitForURL(`${BASE}/patient/*/result`, { timeout: 60000 })
  console.log('OK')

  // Navigate to features
  process.stdout.write('5. Opening Feature Contributions... ')
  await page.click('button:has-text("View feature contributions")')
  await page.waitForURL(`${BASE}/patient/*/features`, { timeout: 10000 })
  await page.waitForTimeout(2500)
  console.log('OK')

  // Read feature bars
  console.log('\n── HIGH-RISK PATIENT FEATURE CONTRIBUTIONS ───────────────────')
  const rows = await page.locator('.flex.items-center.gap-3.py-1\\.5').all()
  if (rows.length === 0) {
    const html = await page.content()
    console.log('(no rows matched; page snippet):', html.slice(0, 400))
  } else {
    for (const row of rows) {
      const name = (await row.locator('.text-sm.text-ink-secondary').first().textContent({ timeout: 1000 }).catch(() => '?')).trim()
      const latestVal = (await row.locator('.text-xs.text-ink-muted').textContent({ timeout: 500 }).catch(() => '')).trim()
      const deltaText = (await row.locator('span.tabular-nums').textContent({ timeout: 500 }).catch(() => '?')).trim()
      console.log(`  ${name.padEnd(30)} ${deltaText.padStart(8)}${latestVal ? '   (' + latestVal + ')' : ''}`)
    }
  }

  if (probes.length > 0) {
    console.log('\n── SHAP-PROBE LOGS ────────────────────────────────────────────')
    probes.forEach(p => console.log('  ' + p))
  }

} catch (err) {
  console.error('\nTest failed:', err.message)
  console.error('URL:', page.url())
  if (probes.length > 0) {
    console.log('Partial probes:')
    probes.forEach(p => console.log('  ' + p))
  }
} finally {
  await browser.close()
}
