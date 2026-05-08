// B1 — webgpu-dna validation harness liveness via Playwright.
//
// Spawns the vite dev server, drives the actual webgpu-dna harness in
// headless Chromium with WebGPU enabled, clicks "Run validation", and
// waits for the first energy result to appear in the table. This proves
// the full pipeline is functional end-to-end — vite + harness HTML +
// main.ts + ensurePipelines + Phase A WGSL dispatch + table render.
//
// Why B1 not E#: still a liveness check, not a falsifiable physics
// claim. Future GPU experiments (E11 GPU vs IRT, multi-energy E5/E7/E8)
// will reference this as the prerequisite — if B1 fails, those can't
// run.
//
// Pass condition: first table row in #tb populated within timeoutMs
// (default 90s) AND its CSDA value (column 2) is plausible (> 100 nm
// for the lowest energy 100 eV, < 50000 nm).

import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { startDevServer } from '../lib/dev-server.mjs';
import { captureEnv } from '../lib/env.mjs';

const REPO_ROOT = join(import.meta.dirname, '..', '..');

// Plausibility bounds — the harness's lowest energy is 100 eV which
// has a CSDA of ~5-15 nm per ESTAR; 1 keV is ~50-100 nm; 10 keV is
// ~2700 nm. The "first row" is by ESTAR order and is usually 100 eV.
const CSDA_MIN_NM = 1;
const CSDA_MAX_NM = 50000;

const TIMEOUT_MS = 90_000;
const N_PRIMARIES = 1024; // minimum allowed by HTML input; fast first energy

async function readPlaywright() {
  const { chromium } = await import('playwright');
  return chromium;
}

async function runHarnessAndCaptureFirstRow(serverUrl, timeoutMs) {
  const chromium = await readPlaywright();
  const browser = await chromium.launch({
    headless: false,
    args: [
      '--headless=new',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--no-sandbox',
    ],
  });

  try {
    const context = await browser.newContext();
    const page = await context.newPage();

    // Capture console output for diagnostics.
    const consoleLines = [];
    page.on('console', (msg) => {
      consoleLines.push(`[${msg.type()}] ${msg.text()}`);
    });
    const pageErrors = [];
    page.on('pageerror', (err) => {
      pageErrors.push(err.message);
    });

    const t0 = Date.now();
    await page.goto(serverUrl, { waitUntil: 'domcontentloaded' });

    // Reduce N to fastest run.
    await page.fill('#np', String(N_PRIMARIES));

    // Click the run button.
    await page.click('#run');

    // Wait for the first row to appear in the result table.
    // Per index.html <thead>, columns are:
    //   1: Energy   2: Therm  3: Esc   4: CSDA Total (nm)   5: ESTAR CSDA  ...
    // We wait on column 4 (CSDA) since it's the canonical liveness value.
    const firstRowSelector = '#tb tr:first-child td:nth-child(4)';
    await page.waitForFunction(
      (sel) => {
        const cell = document.querySelector(sel);
        return cell && cell.textContent && cell.textContent.trim().length > 0;
      },
      firstRowSelector,
      { timeout: timeoutMs },
    );
    const elapsedSec = (Date.now() - t0) / 1000;

    // Capture the first row's full content + tail of the log.
    const captured = await page.evaluate(() => {
      const tr = document.querySelector('#tb tr:first-child');
      const cells = tr ? Array.from(tr.querySelectorAll('td')).map((c) => c.textContent?.trim() ?? '') : [];
      const log = document.getElementById('log');
      const logText = log?.textContent ?? '';
      return {
        firstRowCells: cells,
        logTail: logText.slice(-2000),
        nRows: document.querySelectorAll('#tb tr').length,
      };
    });

    return {
      elapsedSec,
      firstRowCells: captured.firstRowCells,
      logTail: captured.logTail,
      nRows: captured.nRows,
      consoleLineCount: consoleLines.length,
      consoleTail: consoleLines.slice(-10),
      pageErrors,
    };
  } finally {
    await browser.close();
  }
}

export async function runB1() {
  const t0 = Date.now();
  let serverInfo = null;
  let result = null;
  let error = null;

  try {
    serverInfo = await startDevServer();
    result = await runHarnessAndCaptureFirstRow(serverInfo.url, TIMEOUT_MS);
  } catch (err) {
    error = err;
  } finally {
    if (serverInfo) {
      try {
        await new Promise((res) => {
          serverInfo.process.once('exit', res);
          serverInfo.stop();
          // Backstop: don't hang here forever.
          setTimeout(res, 3000);
        });
      } catch { /* nothing */ }
    }
  }

  const totalSec = (Date.now() - t0) / 1000;

  if (error || !result) {
    return {
      meta: makeMeta(),
      env: captureEnv(),
      status: 'fail',
      diagnosis: error ? error.message.slice(0, 800) : 'no result captured',
      summary: { totalSec, headline: 'pipeline error before first row' },
      rows: [],
    };
  }

  // Parse first row cells. Per index.html <thead> the columns are:
  //   [0:Energy, 1:Therm, 2:Esc, 3:CSDA Total (nm), 4:ESTAR CSDA, 5:Ratio, ...]
  const firstRowCells = result.firstRowCells;
  const energyText = firstRowCells[0] ?? '';
  const csdaText = firstRowCells[3] ?? '';
  const csdaNm = parseFloat(csdaText.replace(/[^0-9.+\-eE]/g, ''));
  const energyEv = parseFloat(energyText.replace(/[^0-9.+\-eE]/g, ''));

  const failures = [];
  if (firstRowCells.length === 0) failures.push('no cells in first row after wait');
  if (!Number.isFinite(csdaNm)) failures.push(`CSDA cell did not parse as a number: "${csdaText}"`);
  else if (csdaNm < CSDA_MIN_NM || csdaNm > CSDA_MAX_NM) {
    failures.push(`CSDA ${csdaNm} ∉ [${CSDA_MIN_NM}, ${CSDA_MAX_NM}] nm — implausible`);
  }
  if (result.pageErrors.length > 0) {
    failures.push(`page errors: ${result.pageErrors.slice(0, 2).join(' / ')}`);
  }

  const status = failures.length === 0 ? 'pass' : 'fail';
  const diagnosis = failures.length === 0 ? null : failures.join('; ');

  return {
    meta: makeMeta(),
    env: captureEnv(),
    status,
    diagnosis,
    summary: {
      totalSec,
      firstRowSec: result.elapsedSec,
      nRowsAtCapture: result.nRows,
      firstRowCells: result.firstRowCells,
      energyEv,
      csdaNm,
      consoleLineCount: result.consoleLineCount,
      pageErrorCount: result.pageErrors.length,
      headline: `first row (E=${Number.isFinite(energyEv) ? energyEv : '?'} eV) in ${result.elapsedSec.toFixed(1)}s; CSDA=${Number.isFinite(csdaNm) ? csdaNm.toFixed(1) + ' nm' : '?'} (${result.nRows} rows captured)`,
    },
    rows: [
      {
        firstRowCells: result.firstRowCells,
        logTail: result.logTail,
        consoleTail: result.consoleTail,
      },
    ],
  };
}

function makeMeta() {
  return {
    protocol: 'B1-harness-liveness',
    hypothesis:
      'Spawning the vite dev server and driving the webgpu-dna harness in headless Chromium produces the first energy result in #tb within ~90 seconds at N=1024 primaries / E=100 eV (lowest ESTAR energy). The CSDA value in that row is plausible (> 1 nm, < 50000 nm).',
    passBar:
      'first row appears within 90s AND CSDA cell parses to a number in [1, 50000] nm AND no page-level errors during the run.',
    seed: 'n/a (harness uses its own RNG; this experiment only checks liveness)',
    warmup: 0,
    trials: 1,
    sources: {
      harness: 'src/app.ts → runValidation()',
      driver: 'experiments/lib/dev-server.mjs (vite) + experiments/lib/browser.mjs (Playwright)',
      nPrimaries: N_PRIMARIES,
    },
  };
}
