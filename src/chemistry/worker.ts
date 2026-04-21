/**
 * Thin wrapper around /irt-worker.js (kept as plain JS for portability).
 *
 * The worker runs the full Karamitros 2011 nine-reaction IRT chemistry
 * scheme per primary, then postbacks a `{ type: 'result', timeline, … }`
 * message. Progress messages arrive as `{ type: 'progress', msg }`.
 */

import type { ChemResult, LogFn } from '../physics/types';

interface WorkerProgressMsg {
  type: 'progress';
  msg: string;
}

interface WorkerReactionInfo {
  label: string;
  count: number;
  sigma: number;
  rc: number;
}

interface WorkerResultMsg {
  type: 'result';
  chem_n: number;
  t_wall: number;
  timeline: ChemResult['timeline'];
  n_reacted: number;
  rxn_info?: WorkerReactionInfo[];
}

export function runChemistryWorker(
  rad_buf_f32: Float32Array,
  rad_n: number,
  n_therm: number,
  E_eV: number,
  log?: LogFn,
  maxN?: number,
): Promise<ChemResult> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(`/irt-worker.js?v=${Date.now()}`);
    worker.onmessage = (e: MessageEvent<WorkerProgressMsg | WorkerResultMsg>): void => {
      const data = e.data;
      if (data.type === 'progress') {
        log?.(`  [worker] ${data.msg}`, 'data');
      } else if (data.type === 'result') {
        if (data.timeline) {
          for (const cp of data.timeline) {
            log?.(
              `  ${cp.label} ${cp.G_OH.toFixed(3)} ${cp.G_eaq.toFixed(3)} ${cp.G_H.toFixed(3)} ${cp.G_H2O2.toFixed(3)} ${cp.G_H2.toFixed(3)}`,
              'data',
            );
          }
        }
        log?.(
          `  IRT worker: ${data.n_reacted} reactions in ${(data.t_wall / 1000).toFixed(1)}s (${data.chem_n} radicals)`,
          'data',
        );
        if (data.rxn_info) {
          for (const rx of data.rxn_info) {
            log?.(`    rxn ${rx.label}: ${rx.count} (σ=${rx.sigma}nm rc=${rx.rc}nm)`, 'data');
          }
        }
        worker.terminate();
        resolve({
          chem_n: data.chem_n,
          t_wall: data.t_wall,
          timeline: data.timeline,
          chem_pos_final: null,
          chem_alive_final: null,
        });
      }
    };
    worker.onerror = (err): void => {
      log?.(`  IRT worker error: ${err.message}`, 'err');
      worker.terminate();
      reject(err);
    };

    // Transfer a detached copy of the buffer so the worker gets its own heap.
    const buf = rad_buf_f32.buffer.slice(0);
    worker.postMessage(
      { rad_buf: new Float32Array(buf), rad_n, n_therm, E_eV, max_N: maxN },
      [buf],
    );
  });
}
