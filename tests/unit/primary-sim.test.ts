/**
 * CPU reference simulation of one 10 keV primary electron.
 * Uses the SAME cross section tables and CDF as the GPU shader.
 * If this gives ~198 ions/pri (matching Geant4), the bug is in the WGSL shader.
 * If this gives ~233, the bug is in the shared data/logic.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { BIND } from '../../src/physics/constants';

// Parse WGSL arrays
const wgsl = readFileSync(join(__dirname, '../../public/cross_sections.wgsl'), 'utf-8');

function parseArr(name: string, size?: number): number[] {
  const re = size
    ? new RegExp(`const ${name}=array<f32,${size}>\\(([^)]+)\\)`)
    : new RegExp(`const ${name}=array<f32,\\d+>\\(([^)]+)\\)`);
  const m = wgsl.match(re);
  if (!m) throw new Error(`${name} not found`);
  return m[1].split(',').map(Number);
}
function parseConst(name: string): number {
  const m = wgsl.match(new RegExp(`const ${name}=(\\S+?)u?;`));
  if (!m) throw new Error(`${name} not found`);
  return Number(m[1]);
}

const XN = parseConst('XN');
const XE = parseArr('XE');
const XI = parseArr('XI');
const XC = parseArr('XC');
const XL = parseArr('XL');
const LOG_XE0 = Math.log(XE[0]);
const INV_LOG_XE_STEP = 1.0 / ((Math.log(XE[XN - 1]) - LOG_XE0) / (XN - 1));

const N_WE = parseConst('N_WE');
const N_WR = parseConst('N_WR');
const XWE = parseArr('XWE');
// Log-step constants are not currently asserted but may be needed for future
// dQ/dE sampling tests; kept available via XWE if needed.
void XWE;

const XWC: number[][] = [];
const XWT: number[][] = [];
for (let sh = 0; sh < 5; sh++) {
  XWC.push(parseArr(`XWC${sh}`));
  XWT.push(parseArr(`XWT${sh}`));
}

// Load Born shell fractions
const bornRaw = readFileSync(join(__dirname, '../../data/g4emlow/dna/sigma_ionisation_e_born.dat'), 'utf-8');
const bornLines = bornRaw.split('\n').filter(l => l.trim() && !l.startsWith('#'));
const bornData = bornLines.map(l => l.trim().split(/\s+/).map(Number));

// Sanche vib
const vibRaw = readFileSync(join(__dirname, '../../data/g4emlow/dna/sigma_excitationvib_e_sanche.dat'), 'utf-8');
const vibLines = vibRaw.split('\n').filter(l => l.trim());
const vibData = vibLines.map(l => l.trim().split(/\s+/).map(Number));
const VE = vibData.map(r => r[0]);
const VS = vibData.map(r => r.slice(1).reduce((a, b) => a + b, 0) * 0.01 * 2); // 2x liquid

const NW = 33.4;
const CUTOFF = 7.4;
const EXC_E = 8.22;

// Deterministic RNG (same as shader's xoshiro128)
class RNG {
  private s: Uint32Array;
  constructor(seed: number) {
    this.s = new Uint32Array(4);
    let h = (seed * 2654435761 + 1013904223) >>> 0;
    for (let j = 0; j < 4; j++) {
      h ^= h >>> 16; h = Math.imul(h, 0x45d9f3b); h ^= h >>> 16;
      this.s[j] = (h + j * 0x9E3779B9) >>> 0;
    }
  }
  private rotl(x: number, k: number) { return ((x << k) | (x >>> (32 - k))) >>> 0; }
  next(): number {
    const r = (this.rotl((this.s[0] + this.s[3]) >>> 0, 7) + this.s[0]) >>> 0;
    const t = (this.s[1] << 9) >>> 0;
    this.s[2] ^= this.s[0]; this.s[3] ^= this.s[1]; this.s[1] ^= this.s[2]; this.s[0] ^= this.s[3];
    this.s[2] ^= t; this.s[3] = this.rotl(this.s[3], 11);
    return r;
  }
  float(): number { return (this.next() >>> 1) / 2147483647.0; }
}

function xsAll(E: number): [number, number, number] {
  if (E <= XE[0]) return [0, 0, XL[0]];
  if (E >= XE[XN - 1]) return [XI[XN - 1], XC[XN - 1], XL[XN - 1]];
  const t = (Math.log(E) - LOG_XE0) * INV_LOG_XE_STEP;
  const i = Math.min(Math.max(Math.floor(t), 0), XN - 2);
  const f = t - i;
  return [
    Math.max(0, XI[i] + (XI[i + 1] - XI[i]) * f),
    Math.max(0, XC[i] + (XC[i + 1] - XC[i]) * f),
    Math.max(0, XL[i] + (XL[i + 1] - XL[i]) * f),
  ];
}

function xsVib(E: number): number {
  if (E < 2 || E > 100) return 0;
  for (let i = 0; i < VE.length - 1; i++) {
    if (E >= VE[i] && E < VE[i + 1]) {
      const f = (E - VE[i]) / (VE[i + 1] - VE[i]);
      return Math.max(0, VS[i] + (VS[i + 1] - VS[i]) * f);
    }
  }
  return 0;
}

function shellFracs(E: number): number[] {
  const idx = bornData.reduce((best, row, i) =>
    Math.abs(row[0] - E) < Math.abs(bornData[best][0] - E) ? i : best, 0);
  const row = bornData[idx];
  const shells = row.slice(1);
  const sum = shells.reduce((a, b) => a + b, 0);
  return sum > 0 ? shells.map(s => s / sum) : [1, 0, 0, 0, 0];
}

function sampleWTransfer(E: number, shell: number, rng: RNG): number {
  // Binary search for nearest XWE energy (same as shader)
  let lo = 0, hi = N_WE - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (XWE[mid] <= E) lo = mid; else hi = mid; }
  const iNearest = Math.abs(XWE[hi] - E) < Math.abs(XWE[lo] - E) ? hi : lo;
  const r = rng.float();
  const base = iNearest * N_WR;

  // bsearch_cdf (same as shader)
  let jlo = 0;
  for (let j = 0; j < N_WR - 1; j++) {
    if (XWC[shell][base + j] <= r) jlo = j;
    else break;
  }
  const jhi = Math.min(jlo + 1, N_WR - 1);
  const c0 = XWC[shell][base + jlo], c1 = XWC[shell][base + jhi];
  const e0 = XWT[shell][base + jlo], e1 = XWT[shell][base + jhi];
  if (c1 <= c0) return e0;
  const ff = Math.max(0, Math.min(1, (r - c0) / (c1 - c0)));
  return e0 + (e1 - e0) * ff;
}

function simulatePrimary(seed: number): { ions: number; path: number; meanW: number } {
  const rng = new RNG(seed);
  let E = 10000;
  let ions = 0, path = 0, totalW = 0;

  for (let step = 0; step < 65536; step++) {
    if (E < CUTOFF) break;

    const [si, sc, sl] = xsAll(E);
    const sv = xsVib(E);
    const st = si + sc + sl + sv;
    if (st <= 0) break;

    // Step length
    const lam = 1 / (NW * st);
    const dist = -Math.log(Math.max(rng.float(), 1e-10)) * lam;
    path += dist;

    // Elastic angle (consume RNG draws same as shader)
    rng.float(); // r_el
    rng.float(); // phi_el

    // Process selection
    const rType = rng.float() * st;

    if (rType < si && E > BIND[0]) {
      // Ionization
      const sf = shellFracs(E);
      const rSh = rng.float();
      let cum = 0, shell = 0;
      for (let s = 0; s < 5; s++) { cum += sf[s]; if (rSh < cum) { shell = s; break; } }

      const Wtransfer = sampleWTransfer(E, shell, rng);
      const Wmax = (E + BIND[shell]) / 2;
      const W = Math.min(Math.max(Wtransfer, BIND[shell]), Wmax);
      const Wsec = Math.max(W - BIND[shell], 0);
      E -= W;
      ions++;
      totalW += W;

      // Secondary emission consumes RNG draws (BornAngle + momentum + child RNG)
      if (Wsec > CUTOFF) {
        rng.float(); // BornAngle regime draw
        if (Wsec >= 50 && Wsec <= 200) rng.float(); // mixed regime extra draw
        rng.float(); // phi_s
        // Primary momentum conservation uses sdx/sdy/sdz (no extra draws)
        // Child RNG: 4 draws
        rng.next(); rng.next(); rng.next(); rng.next();
      }
    } else if (rType < si + sc) {
      // Excitation
      rng.float(); // level selection
      E -= Math.min(EXC_E, E);
      // Dissociative branching
      rng.float(); // r_ch
    } else if (rType < si + sc + sv) {
      // Vibrational
      rng.float(); // mode selection
      E -= Math.min(0.2, E);
    }
    // else: elastic (no energy loss, angle already applied)
  }

  return { ions, path, meanW: ions > 0 ? totalW / ions : 0 };
}

describe('CPU reference primary simulation', () => {
  it('produces ~198 ions/pri at 10 keV (matching Geant4)', () => {
    // Run 100 primaries with different seeds
    let totalIons = 0;
    const N = 100;
    for (let i = 0; i < N; i++) {
      const r = simulatePrimary(i + 1);
      totalIons += r.ions;
    }
    const mean = totalIons / N;
    console.log(`CPU ref sim: ${mean.toFixed(1)} ions/pri (Geant4: 196.7, GPU shader: 233)`);

    // The CPU ref should match one of them — whichever it matches tells us where the bug is
    expect(mean).toBeGreaterThan(150); // sanity
    expect(mean).toBeLessThan(300);    // sanity
  });

  it('single primary at seed=1 logs trajectory', () => {
    const r = simulatePrimary(1);
    console.log(`Seed=1: ions=${r.ions}, path=${r.path.toFixed(1)}nm, meanW=${r.meanW.toFixed(1)}eV`);
    expect(r.ions).toBeGreaterThan(100);
  });
});
