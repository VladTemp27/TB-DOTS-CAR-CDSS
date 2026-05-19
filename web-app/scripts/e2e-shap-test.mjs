/**
 * End-to-end SHAP ablation diagnostic test.
 *
 * Creates a fresh patient through the full intake flow, triggers a prediction,
 * navigates to Feature Contributions, and prints every bar's name + delta value.
 * Also captures [SHAP-PROBE] console logs for raw delta values.
 *
 * Usage:
 *   npx playwright install chromium          # first run only
 *   node scripts/e2e-shap-test.mjs
 *
 * Requires: dev server running on localhost:5173
 */

import { chromium } from 'playwright'

const BASE = `http://localhost:${process.env.PORT ?? 5299}`

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage()

// Collect console output
const probes = []
const errors = []
page.on('console', msg => {
  const t = msg.text()
  if (t.includes('[SHAP-PROBE]')) probes.push(t)
})
page.on('pageerror', e => errors.push(e.message))

try {
  // ── Step 1: Login ──────────────────────────────────────────────────────────
  process.stdout.write('1. Logging in... ')
  await page.goto(`${BASE}/login`)
  await page.fill('#username', 'admin123')
  await page.fill('#password', 'password123')
  await page.click('button:has-text("Sign In")')
  await page.waitForURL(`${BASE}/`, { timeout: 10000 })
  console.log('OK')

  // ── Step 2: Intake Step 1 — demographics ──────────────────────────────────
  process.stdout.write('2. Filling patient demographics... ')
  await page.goto(`${BASE}/patient/new`)
  await page.waitForSelector('#name')
  await page.fill('#name', 'E2E SHAP Test Patient')
  await page.fill('#age', '45')
  await page.selectOption('#sex', 'M')
  await page.selectOption('#regGroup', 'NEW')
  // Pick first non-empty province
  await page.selectOption('#province', { index: 1 })
  await page.click('button:has-text("Continue to Lab Results")')
  await page.waitForURL(`${BASE}/patient/new/lab`, { timeout: 10000 })
  console.log('OK')

  // ── Step 3: Intake Step 2 — required + several optional fields ─────────────
  process.stdout.write('3. Filling lab results & clinical data... ')
  await page.waitForSelector('#bactStatus')
  await page.selectOption('#bactStatus', 'Bacteriologically-confirmed TB')
  await page.selectOption('#microResult', '1+')
  await page.selectOption('#anatSite', 'P')
  await page.selectOption('#source', 'COMMUNITY')
  await page.selectOption('#caseType', 'DSTB')
  // Optional — fill these to exercise more ablation groups
  await page.fill('#heartRate', '80')
  await page.fill('#baseWeight', '60')
  await page.fill('#baseHeight', '165')
  await page.selectOption('#drugResistance', 'DS-TB')
  // Click Generate Diagnosis
  await page.click('button:has-text("Generate Diagnosis")')
  await page.waitForURL(`${BASE}/patient/new/xray`, { timeout: 10000 })
  console.log('OK')

  // ── Step 4: Skip X-ray upload — triggers commitIntakePatient ──────────────
  process.stdout.write('4. Committing patient (running M0 prediction)... ')
  await page.waitForSelector('button:has-text("Skip")')
  await page.click('button:has-text("Skip")')
  // Wait for prediction + backend save (may take ~5-10s due to 15 ablation passes)
  await page.waitForURL(`${BASE}/patient/*/result`, { timeout: 60000 })
  console.log('OK')

  // ── Step 5: Navigate to Feature Contributions ─────────────────────────────
  process.stdout.write('5. Opening Feature Contributions page... ')
  await page.click('button:has-text("View feature contributions")')
  await page.waitForURL(`${BASE}/patient/*/features`, { timeout: 10000 })
  // Wait for patient data to load from API
  await page.waitForTimeout(2500)
  console.log('OK')

  // ── Step 6: Read every FeatureBar row ─────────────────────────────────────
  console.log('\n── FEATURE CONTRIBUTIONS ──────────────────────────────────────')
  const rows = await page.locator('.flex.items-center.gap-3.py-1\\.5').all()
  if (rows.length === 0) {
    // Fallback: grab all text content from the page for debugging
    const text = await page.locator('[class*="Feature"]').allTextContents()
    console.log('(no rows matched; raw text sample):', text.slice(0, 5))
  } else {
    for (const row of rows) {
      const name = (await row.locator('.text-sm.text-ink-secondary').first().textContent({ timeout: 1000 }).catch(() => '?')).trim()
      const latestVal = (await row.locator('.text-xs.text-ink-muted').textContent({ timeout: 500 }).catch(() => '')).trim()
      const deltaText = (await row.locator('span.tabular-nums').textContent({ timeout: 500 }).catch(() => '?')).trim()
      console.log(`  ${name.padEnd(30)} ${deltaText.padStart(8)}${latestVal ? '   (' + latestVal + ')' : ''}`)
    }
  }

  // ── Step 7: Print SHAP-PROBE logs ─────────────────────────────────────────
  if (probes.length > 0) {
    console.log('\n── SHAP-PROBE CONSOLE LOGS ────────────────────────────────────')
    probes.forEach(p => console.log('  ' + p))
  } else {
    console.log('\n── SHAP-PROBE: no logs captured (probe tags may not be present) ──')
  }

  if (errors.length > 0) {
    console.log('\n── BROWSER ERRORS ─────────────────────────────────────────────')
    errors.forEach(e => console.log('  ERROR:', e))
  }

} catch (err) {
  console.error('\nTest failed:', err.message)
  // Dump page content for debugging
  const url = page.url()
  const title = await page.title().catch(() => '?')
  console.error('Current URL:', url, '| Title:', title)
  if (probes.length > 0) {
    console.log('\nPartial probe logs:')
    probes.forEach(p => console.log('  ' + p))
  }
} finally {
  await browser.close()
}
