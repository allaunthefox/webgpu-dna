// Wrapper around tools/run_irt.cjs — invokes the IRT chemistry worker on a
// rad_E*_N4096.bin radical-buffer dump and returns the parsed timeline.
//
// The .cjs runner writes its structured output to stdout as a single JSON
// line (`{"type":"result","timeline":[...],...}`), and progress / timing
// to stderr. We capture stdout, scan for the result line, and parse.
//
// Optional persistent cache: if `cachePath` is provided and the cache
// exists with a `binMtimeMs` matching the current .bin file's mtime, we
// skip the expensive run and return the cached result. After a fresh run
// completes, we write the cache. The cache makes E10 re-runs cheap and
// is used to recover from killed-but-completed runs (e.g. the 20 keV
// case where IRT finished at 610s but our 600s timeout fired).
//
// This is the Node-side equivalent of running the chemistry phase in the
// browser. The .bin inputs are produced by the browser harness via the
// `dump_server.cjs` POST endpoint and live (gitignored) under dumps/.

import { execFileSync } from 'node:child_process';
import { existsSync, statSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

export function runIrtWorker(binPath, nTherm, eEv, opts = {}) {
  if (!existsSync(binPath)) {
    throw new Error(
      `IRT input not found: ${binPath}. ` +
        `Generate via the browser harness (npm run dev → click "Dump radicals"), ` +
        `or skip this energy.`,
    );
  }
  const sizeMb = statSync(binPath).size / 1e6;
  const binMtimeMs = Math.floor(statSync(binPath).mtimeMs);

  // Check cache (if requested and valid).
  if (opts.cachePath && existsSync(opts.cachePath)) {
    try {
      const cached = JSON.parse(readFileSync(opts.cachePath, 'utf8'));
      if (cached.binMtimeMs === binMtimeMs && cached.nTherm === nTherm && cached.eEv === eEv) {
        return {
          binPath,
          sizeMb,
          nTherm,
          eEv,
          elapsedSec: 0,
          fromCache: true,
          cachePath: opts.cachePath,
          timeline: cached.irtResult.timeline,
          nReacted: cached.irtResult.n_reacted,
          chemN: cached.irtResult.chem_n,
          rxnInfo: cached.irtResult.rxn_info,
        };
      }
    } catch {
      // fall through to fresh run
    }
  }

  const t0 = Date.now();
  const stdout = execFileSync(
    'node',
    ['tools/run_irt.cjs', binPath, String(nTherm), String(eEv)],
    {
      encoding: 'utf8',
      maxBuffer: 32 * 1024 * 1024,
      timeout: opts.timeoutMs ?? 1_200_000,
      // run_irt.cjs writes progress to stderr; the JSON result is on stdout.
      stdio: ['ignore', 'pipe', opts.captureStderr ? 'pipe' : 'ignore'],
    },
  );
  const elapsedSec = (Date.now() - t0) / 1000;

  // The result line is the only line starting with `{"type":"result"`.
  const resultLine = stdout
    .split('\n')
    .find((line) => line.trim().startsWith('{"type":"result"'));
  if (!resultLine) {
    throw new Error(
      `IRT runner did not emit a {"type":"result"} JSON line for ${binPath}. ` +
        `stdout (truncated): ${stdout.slice(0, 500)}`,
    );
  }
  const result = JSON.parse(resultLine);

  // Write cache if requested.
  if (opts.cachePath) {
    try {
      mkdirSync(dirname(opts.cachePath), { recursive: true });
      writeFileSync(
        opts.cachePath,
        JSON.stringify(
          {
            source: 'fresh run via runIrtWorker',
            binPath,
            binMtimeMs,
            binSizeBytes: statSync(binPath).size,
            nTherm,
            eEv,
            elapsedSec,
            irtResult: result,
          },
          null,
          2,
        ),
      );
    } catch {
      // cache write failures are non-fatal
    }
  }

  return {
    binPath,
    sizeMb,
    nTherm,
    eEv,
    elapsedSec,
    fromCache: false,
    timeline: result.timeline,
    nReacted: result.n_reacted,
    chemN: result.chem_n,
    rxnInfo: result.rxn_info,
  };
}
