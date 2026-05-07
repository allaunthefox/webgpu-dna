// E6 — Mean free path per energy bin vs Geant4 ntuple.
//
// Important methodological subtlety: in a multi-process Poisson tracker,
// the step length conditional on a particular process firing is the
// same as the unconditional step length — both equal 1/(n × σ_total).
// What changes per process is the *probability* of that process being
// the one that fires (= σ_proc / σ_total), not the step length when it
// does fire. So Geant4's g4_mfp.csv shows essentially the same MFP for
// elastic / excitation / ionisation in each bin (5-10 keV: 13.09 / 13.44
// / 13.11 nm — < 3% spread, attributable to bin-sampling noise).
//
// Hypothesis:
//   For each energy_bin ∈ {[100,300), [300,500), [500,1000), [1000,3000),
//   [3000,5000), [5000,10000)} eV, WebGPU MFP_total = 1 / (n_water ×
//   (σ_ion + σ_exc + σ_el)(E_mid)) agrees with Geant4 ntuple's per-step
//   MFP within 25% (averaged across the three "process" rows for the
//   same bin, since they all measure the same MFP_total).
//
//   The 25% bar accommodates:
//   (1) Geant4 bins by event ekin within the bin; we evaluate σ at the
//       geometric midpoint. For log-log smooth σ this is < 5% per bin.
//   (2) Webgpu uses Emfietzoglou excitation (data-driven, 2.4× larger
//       than Born); Geant4 dnaphysics-Opt2 uses Born excitation. The
//       resulting σ_total_wgsl is slightly higher → slightly shorter
//       MFP_wgsl. README claims "MFP within 2-14%"; 25% is the
//       comfortable upper bound for scale-factor regression catching.
//
// Why this matters: MFP is the most direct test of the kinetics of
// secondary electron transport. CSDA (E5) catches integrated path;
// MFP (E6) catches the differential rate. A scale-factor regression
// in any XS table (peak_ratio bars in E1-E4 catch this in isolation)
// would shift MFP correspondingly here.

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { SEEDS } from '../lib/seeds.mjs';
import { captureEnv } from '../lib/env.mjs';
import { parseWgslArray, logLogInterp } from '../lib/xs-bitmatch.mjs';

const REPO_ROOT = join(import.meta.dirname, '..', '..');

// Liquid water number density at STP:
//   ρ = 1.000 g/cm³,  N_A = 6.022e23 / mol,  M(H₂O) = 18.015 g/mol
//   n = ρ N_A / M = 3.343e22 / cm³
//   1 cm³ = 1e21 nm³ → n = 33.43 / nm³
const N_WATER_PER_NM3 = 33.43;

// All three primary-electron processes contribute to σ_total above 100 eV.
// Vibrational (XVS) is below 100 eV, out of E6's bin range.
const PRIMARY_XS_ARRAYS = ['XI', 'XC', 'XL'];

function readG4Mfp(path) {
  const text = readFileSync(path, 'utf8');
  const lines = text.split(/\r\n|\r|\n/).map((l) => l.trim()).filter(Boolean);
  const [header, ...data] = lines;
  const cols = header.split(',');
  return data.map((line) => {
    const fields = line.split(',');
    const row = {};
    cols.forEach((c, i) => {
      row[c] = isNaN(Number(fields[i])) ? fields[i] : Number(fields[i]);
    });
    return row;
  });
}

export async function runE6() {
  const wgslText = readFileSync(join(REPO_ROOT, 'public', 'cross_sections.wgsl'), 'utf8');
  const xe = parseWgslArray(wgslText, 'XE');
  const sigmaTables = {
    XI: parseWgslArray(wgslText, 'XI'),
    XC: parseWgslArray(wgslText, 'XC'),
    XL: parseWgslArray(wgslText, 'XL'),
  };

  const g4Rows = readG4Mfp(join(REPO_ROOT, 'validation', 'g4_mfp.csv'));

  // Group g4 rows by (e_lo, e_hi); each bin has 3 process entries, all
  // measuring the same MFP_total. We average the three Geant4 entries
  // per bin to get one MFP_total reference per bin.
  const bins = new Map();
  for (const r of g4Rows) {
    const key = `${r.e_lo}-${r.e_hi}`;
    if (!bins.has(key)) {
      bins.set(key, { eLoEv: r.e_lo, eHiEv: r.e_hi, perProcess: [] });
    }
    bins.get(key).perProcess.push(r);
  }

  const rows = [];
  for (const bin of bins.values()) {
    const eMidEv = Math.sqrt(bin.eLoEv * bin.eHiEv);
    let sigmaTotal = 0;
    const perProcessSigma = {};
    for (const arr of PRIMARY_XS_ARRAYS) {
      const s = logLogInterp(xe, sigmaTables[arr], eMidEv);
      perProcessSigma[arr] = s;
      sigmaTotal += s;
    }
    const mfpWgsl = sigmaTotal > 0 ? 1 / (N_WATER_PER_NM3 * sigmaTotal) : Infinity;

    // Geant4 MFP_total reference: mean of the three per-process rows
    // (all measuring the same MFP_total, < 3% spread per bin).
    const g4MfpMean =
      bin.perProcess.reduce((a, b) => a + b.mfp_nm, 0) / bin.perProcess.length;
    const g4MfpSpread =
      Math.max(...bin.perProcess.map((p) => p.mfp_nm)) -
      Math.min(...bin.perProcess.map((p) => p.mfp_nm));

    const ratio = mfpWgsl / g4MfpMean;
    const inBand = Math.abs(ratio - 1) < 0.25;

    rows.push({
      eLoEv: bin.eLoEv,
      eHiEv: bin.eHiEv,
      eMidEv,
      sigmaXi: perProcessSigma.XI,
      sigmaXc: perProcessSigma.XC,
      sigmaXl: perProcessSigma.XL,
      sigmaTotal,
      mfpWgslNm: mfpWgsl,
      mfpG4MeanNm: g4MfpMean,
      mfpG4SpreadNm: g4MfpSpread,
      g4PerProcess: Object.fromEntries(bin.perProcess.map((p) => [p.process, p.mfp_nm])),
      ratio,
      pctErr: (ratio - 1) * 100,
      status: inBand ? 'pass' : 'fail',
    });
  }

  if (rows.length === 0) {
    throw new Error('No comparable rows — check g4_mfp.csv processes match XI/XC/XL.');
  }

  // ---- aggregates ----
  const ratios = rows.map((r) => r.ratio).sort((a, b) => a - b);
  const median = ratios[Math.floor(ratios.length / 2)];
  const minRatio = ratios[0];
  const maxRatio = ratios[ratios.length - 1];
  const failed = rows.filter((r) => r.status === 'fail');

  const status = failed.length === 0 ? 'pass' : 'fail';
  const diagnosis =
    failed.length === 0
      ? null
      : failed
          .slice(0, 4)
          .map((r) => `[${r.eLoEv},${r.eHiEv}) ratio=${r.ratio.toFixed(3)} (${r.pctErr.toFixed(1)}%)`)
          .join('; ');

  return {
    meta: {
      protocol: 'E6-mfp-vs-g4-ntuple',
      hypothesis:
        'WebGPU MFP_total(E_mid) = 1 / (n_water × Σ σ_proc(E_mid)) agrees with Geant4 ntuple MFP_total (mean of the three per-process rows in each energy bin) within 25%. n_water = 33.43 nm⁻³ (liquid water STP). E_mid = geometric midpoint of each bin.',
      passBar: '|MFP_wgsl / MFP_g4_mean − 1| < 0.25 for every energy bin.',
      seed: `E6_MFP=0x${SEEDS.E6_MFP.toString(16).toUpperCase()}`,
      warmup: 0,
      trials: 1,
      sources: {
        wgsl: 'public/cross_sections.wgsl (XI / XC / XL arrays)',
        g4Mfp: 'validation/g4_mfp.csv (Geant4 11.4.1 ntuple, primary-electron stats)',
      },
      constants: {
        nWaterPerNm3: N_WATER_PER_NM3,
      },
    },
    env: captureEnv(),
    status,
    diagnosis,
    summary: {
      nRows: rows.length,
      nFailed: failed.length,
      medianRatio: median,
      minRatio,
      maxRatio,
      headline: `${rows.length} (process, bin) cells; ratio range [${minRatio.toFixed(3)}, ${maxRatio.toFixed(3)}], median ${median.toFixed(3)}`,
    },
    rows,
  };
}
