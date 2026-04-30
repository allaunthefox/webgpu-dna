/**
 * Entry point — wires the "Run validation" button to `runValidation()` and
 * binds the Reset button. Mirrors the final block of public/geant4dna.html.
 */

import { runValidation } from './app';
import { exportSnapshotAt10keV } from './splat/export';
import { createLogger } from './ui/log';

const lg = createLogger('log');

const $ = (id: string): HTMLElement | null =>
  typeof document !== 'undefined' ? document.getElementById(id) : null;

function readNumber(id: string, fallback: number): number {
  const el = $(id) as HTMLInputElement | null;
  const v = el ? parseFloat(el.value) : NaN;
  return Number.isFinite(v) ? v : fallback;
}

function main(): void {
  if (typeof document === 'undefined') return;

  const runBtn = $('run') as HTMLButtonElement | null;
  const resetBtn = $('reset') as HTMLButtonElement | null;

  if (runBtn) {
    runBtn.onclick = async () => {
      runBtn.disabled = true;
      if (resetBtn) resetBtn.disabled = true;
      try {
        await runValidation({
          np: Math.round(readNumber('np', 4096)),
          boxNm: readNumber('box', 15000),
          ceEV: readNumber('cut', 7.4),
          log: lg,
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        lg(`Error: ${msg}`, 'err');
        console.error(e);
      } finally {
        runBtn.disabled = false;
        if (resetBtn) resetBtn.disabled = false;
      }
    };
  }

  if (resetBtn) {
    resetBtn.onclick = () => {
      const tb = $('tb');
      const log = $('log');
      if (tb) tb.innerHTML = '';
      if (log) log.innerHTML = '';
    };
  }

  const exportBtn = $('export-4d') as HTMLButtonElement | null;
  if (exportBtn) {
    exportBtn.onclick = async () => {
      exportBtn.disabled = true;
      if (runBtn) runBtn.disabled = true;
      if (resetBtn) resetBtn.disabled = true;
      try {
        await exportSnapshotAt10keV({
          np: Math.round(readNumber('np', 4096)),
          boxNm: readNumber('box', 15000),
          ceEV: readNumber('cut', 7.4),
          log: lg,
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        lg(`Export error: ${msg}`, 'err');
        console.error(e);
      } finally {
        exportBtn.disabled = false;
        if (runBtn) runBtn.disabled = false;
        if (resetBtn) resetBtn.disabled = false;
      }
    };
  }

  lg('Geant4-DNA validation harness (TypeScript modular build)');
  lg('Click "Run validation" to measure CSDA range and stopping power vs NIST ESTAR.');
}

main();
