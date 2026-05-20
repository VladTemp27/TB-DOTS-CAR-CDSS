/**
 * SHAP ablation diagnostic probe.
 *
 * Opens the app in a headed browser and captures every [SHAP-PROBE] console log
 * emitted during a prediction run. Does NOT automate login — you drive the browser.
 *
 * Usage:
 *   npx playwright install chromium          # first run only
 *   node scripts/probe-shap.mjs             # then open a patient and trigger a check-in
 *
 * Output is saved to scripts/probe-shap.log (gitignored).
 */

import { chromium } from 'playwright'
import { writeFileSync } from 'node:fs'

const PORT = process.env.PORT ?? 5173
const browser = await chromium.launch({ headless: false })
const context = await browser.newContext()
const page = await context.newPage()

const logs = []
page.on('console', msg => {
  const text = msg.text()
  if (text.includes('[SHAP-PROBE]')) {
    logs.push(text)
    process.stdout.write('  ' + text + '\n')
  }
})

await page.goto(`http://localhost:${PORT}`)
console.log(`\nBrowser open at localhost:${PORT}`)
console.log('Steps:')
console.log('  1. Log in')
console.log('  2. Open any patient')
console.log('  3. Start a monthly check-in (this fires the ablation)')
console.log('  4. Close the browser window when done\n')
console.log('[SHAP-PROBE] logs will stream here and be saved to scripts/probe-shap.log\n')

await page.waitForEvent('close', { timeout: 0 }).catch(() => {})
await browser.close()

writeFileSync(new URL('probe-shap.log', import.meta.url), logs.join('\n') + '\n')
console.log(`\nSaved ${logs.length} probe entries to scripts/probe-shap.log`)
