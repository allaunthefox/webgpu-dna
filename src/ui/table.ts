/**
 * Results-table renderer. One `<tr>` per energy point with classification
 * bands — `ok` (0.9–1.1×), `meh` (0.5–2.0×), `bad` (outside). Identical
 * markup/cell order to the monolithic geant4dna.html row template.
 */

import type { EnergyResult } from '../physics/types';
import type { EstarEntry } from '../scoring/estar';

export interface DamageRow {
  /** Event-level direct SSBs. Zero → rendered as "—". */
  ssb_dir: number;
  /** Indirect SSBs (OH backbone reactions). */
  ssb_ind: number;
  /** Integer DSB count from greedy ±10 bp clustering. */
  dsb: number;
  /** DSB yield per Gy per Gbp. */
  dsb_per_gy_gbp: number;
}

export function classify(ratio: number): 'ok' | 'meh' | 'bad' {
  if (ratio >= 0.9 && ratio <= 1.1) return 'ok';
  if (ratio >= 0.5 && ratio <= 2.0) return 'meh';
  return 'bad';
}

function fmtEnergy(E: number): string {
  return E >= 1000 ? `${(E / 1000).toFixed(0)} keV` : `${E} eV`;
}

/** Format a number, or "—" if zero/falsy. */
function fmtOrDash(n: number, digits = 0): string {
  if (!n) return '—';
  return digits > 0 ? n.toFixed(digits) : String(n);
}

/** Append one row to the results tbody. */
export function appendResultRow(
  tbody: HTMLElement,
  r: EnergyResult,
  estar: EstarEntry,
  damage: DamageRow | null,
): void {
  const csdaRatio = r.mean_total / estar.csda;
  const spRatio = r.mean_sp / estar.sp;
  const dmg = damage ?? { ssb_dir: 0, ssb_ind: 0, dsb: 0, dsb_per_gy_gbp: 0 };

  const row = `
    <tr>
      <td>${fmtEnergy(r.E)}</td>
      <td>${r.n_therm}</td>
      <td>${r.n_esc}</td>
      <td>${r.mean_total.toFixed(1)}</td>
      <td>${estar.csda}</td>
      <td class="${classify(csdaRatio)}">${csdaRatio.toFixed(2)}×</td>
      <td>${r.mean_prod.toFixed(1)}</td>
      <td>${r.mean_sp.toFixed(2)}</td>
      <td>${estar.sp}</td>
      <td class="${classify(spRatio)}">${spRatio.toFixed(2)}×</td>
      <td>${r.mean_ions.toFixed(1)}</td>
      <td>${r.sec_per_pri.toFixed(1)}${r.sec_dropped > 0 ? ' ⚠' + r.sec_dropped : ''}</td>
      <td class="${classify(r.cons_ratio)}">${(r.cons_ratio * 100).toFixed(1)}%</td>
      <td>${r.G_OH.toFixed(2)}</td>
      <td>${r.G_eaq.toFixed(2)}</td>
      <td>${r.G_H.toFixed(2)}</td>
      <td>${fmtOrDash(dmg.ssb_dir, 1)}</td>
      <td>${fmtOrDash(dmg.ssb_ind)}</td>
      <td>${fmtOrDash(dmg.dsb)}</td>
      <td>${fmtOrDash(dmg.dsb_per_gy_gbp, 2)}</td>
      <td>${r.dt.toFixed(0)}</td>
    </tr>`;
  tbody.insertAdjacentHTML('beforeend', row);
}

/** Clear the results tbody. */
export function clearResults(tbody: HTMLElement): void {
  tbody.innerHTML = '';
}
