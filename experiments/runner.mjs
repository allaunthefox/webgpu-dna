// CLI dispatcher: `node experiments/runner.mjs <experiment-id>`
//
// Looks up the experiment by ID, runs it, writes the artifact JSON to
// experiments/results/<today-utc>/level-N/<id>.json, prints a one-line
// summary to stdout, and exits 0 on pass / 1 on fail.

import { join } from 'node:path';
import { writeArtifact, todayUtcDate } from './lib/artifact.mjs';
import { runE1 } from './level-1-cross-sections/E1-ion-xs-match.mjs';
import { runE2 } from './level-1-cross-sections/E2-exc-xs-match.mjs';
import { runE3 } from './level-1-cross-sections/E3-elastic-xs-match.mjs';

const REGISTRY = {
  E1: { run: runE1, level: 'level-1', id: 'E1-ion-xs-match' },
  E2: { run: runE2, level: 'level-1', id: 'E2-exc-xs-match' },
  E3: { run: runE3, level: 'level-1', id: 'E3-elastic-xs-match' },
};

const REPO_ROOT = join(import.meta.dirname, '..');

async function main() {
  const id = process.argv[2];
  if (!id || !REGISTRY[id]) {
    const known = Object.keys(REGISTRY).join(', ');
    console.error(`usage: node experiments/runner.mjs <id>   (known: ${known})`);
    process.exit(2);
  }

  const entry = REGISTRY[id];
  const result = await entry.run();
  const date = todayUtcDate();
  const outPath = join(REPO_ROOT, 'experiments', 'results', date, entry.level, `${entry.id}.json`);

  writeArtifact(outPath, {
    meta: result.meta,
    env: result.env,
    status: result.status,
    diagnosis: result.diagnosis,
    rows: result.rows,
  });

  // Print summary line — same shape webgpu-q's run-all uses.
  const s = result.summary ?? {};
  const tag = result.status === 'pass' ? '✓ PASS' : result.status === 'noisy' ? '⚠ NOISY' : '✗ FAIL';
  console.log(
    `[${id}] ${tag}  rows=${s.nRows ?? result.rows.length}  ` +
      `peak_ratio=${s.peakRatio?.toFixed(4) ?? '—'}  ` +
      `median=${s.medianRelErr?.toExponential(2) ?? '—'}  ` +
      `p90=${s.p90RelErr?.toExponential(2) ?? '—'}  ` +
      `max=${s.maxRelErrMeaningful?.toExponential(2) ?? '—'}  ` +
      `→ ${outPath.replace(REPO_ROOT + '/', '')}`,
  );
  if (result.diagnosis) console.log(`  diagnosis: ${result.diagnosis}`);

  process.exit(result.status === 'pass' ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
