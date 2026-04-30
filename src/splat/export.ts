/**
 * 4D snapshot exporter.
 *
 * Runs ONE 10 keV validation pass through the GPU chemistry backend with
 * `dump_snapshots: true`, packages the resulting per-checkpoint radical
 * positions + DNA fiber geometry + initial radical buffer into a single
 * binary blob, and triggers a browser download. The output drives the
 * `/splat` 4D viewer.
 *
 * The export uses the GPU chemistry path (not the IRT worker) because only
 * the GPU path produces per-step `chem_pos` snapshots. G(t) values from this
 * path are slightly less accurate than the IRT worker's — that's a known
 * limitation documented in CLAUDE.md and is acceptable here because the
 * viewer is qualitative, not a primary validation artifact.
 */
import { ensurePipelines } from '../app';
import { runAtEnergy } from '../gpu/dispatch';
import { buildDNATarget } from '../physics/dna-geometry';
import { runChemistry as runChemistryGPU } from '../chemistry/schedule';
import { DNA_LENGTH_NM, DNA_GRID_N, DNA_SPACING_NM } from '../physics/constants';
import type { ChemResult, ChemSnapshot, DNATarget, LogFn } from '../physics/types';

export interface ExportConfig {
  np: number;
  boxNm: number;
  ceEV: number;
  log: LogFn;
  /** Number of radicals to subsample per checkpoint. Default 50_000. */
  snapshot_n?: number;
}

const MAGIC = new Uint8Array([0x57, 0x47, 0x44, 0x4e, 0x41, 0x34, 0x44, 0x00]); // "WGDNA4D\0"
const FORMAT_VERSION = 1;

/**
 * Pack snapshots + DNA + initial radical buffer into a binary blob.
 *
 * Layout:
 *   header  (32 B)        magic + version + counts + energy
 *   dna     (variable)    fy[], fz[], rise, spacing, L, x0
 *   rad0    (variable)    initial rad_buf (xyz + species_packed.w) × rad_n
 *   snaps   (variable)    [(t_ns, label, pos[snap_n*4], alive[snap_n])] × num_snaps
 *
 * All multi-byte values are little-endian (browser-native).
 */
export function packSnapshotBlob(
  snapshots: ChemSnapshot[],
  dna: DNATarget,
  rad_buf_initial: Float32Array,
  rad_n_initial: number,
  energy_eV: number,
): Blob {
  if (snapshots.length === 0) throw new Error('no snapshots to pack');
  const snap_n = snapshots[0].n;
  const num_snaps = snapshots.length;

  // Pre-encode labels so we know their byte sizes.
  const enc = new TextEncoder();
  const labelBytes: Uint8Array[] = snapshots.map((s) => enc.encode(s.label));

  // Compute total size.
  let total = 0;
  total += 32; // header
  total += 4 + dna.fy.byteLength + dna.fz.byteLength + 4 * 4; // n_fibers + fy + fz + rise/spacing/L/x0
  total += 4 + rad_n_initial * 16; // rad_n_initial + rad_buf
  for (let i = 0; i < num_snaps; i++) {
    total += 4 + 4 + labelBytes[i].byteLength; // t_ns + label_len + label
    total += snap_n * 16 + snap_n * 4;          // pos + alive
  }

  const buf = new ArrayBuffer(total);
  const dv = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let off = 0;

  // --- Header ---
  u8.set(MAGIC, off); off += 8;
  dv.setUint32(off, FORMAT_VERSION, true); off += 4;
  dv.setUint32(off, num_snaps, true); off += 4;
  dv.setUint32(off, snap_n, true); off += 4;
  dv.setUint32(off, dna.n_fibers, true); off += 4;
  dv.setUint32(off, dna.n_bp_per, true); off += 4;
  dv.setFloat32(off, energy_eV, true); off += 4;

  // --- DNA geometry ---
  dv.setUint32(off, dna.n_fibers, true); off += 4;
  u8.set(new Uint8Array(dna.fy.buffer, dna.fy.byteOffset, dna.fy.byteLength), off); off += dna.fy.byteLength;
  u8.set(new Uint8Array(dna.fz.buffer, dna.fz.byteOffset, dna.fz.byteLength), off); off += dna.fz.byteLength;
  dv.setFloat32(off, dna.rise, true); off += 4;
  dv.setFloat32(off, dna.spacing_nm, true); off += 4;
  dv.setFloat32(off, dna.L_nm, true); off += 4;
  dv.setFloat32(off, dna.x0, true); off += 4;

  // --- Initial rad_buf (subsampled to first rad_n_initial entries) ---
  dv.setUint32(off, rad_n_initial, true); off += 4;
  const initBytes = rad_n_initial * 16;
  u8.set(new Uint8Array(rad_buf_initial.buffer, rad_buf_initial.byteOffset, initBytes), off);
  off += initBytes;

  // --- Snapshots ---
  for (let i = 0; i < num_snaps; i++) {
    const s = snapshots[i];
    dv.setFloat32(off, s.t_ns, true); off += 4;
    dv.setUint32(off, labelBytes[i].byteLength, true); off += 4;
    u8.set(labelBytes[i], off); off += labelBytes[i].byteLength;
    u8.set(new Uint8Array(s.pos.buffer, s.pos.byteOffset, s.pos.byteLength), off); off += s.pos.byteLength;
    u8.set(new Uint8Array(s.alive.buffer, s.alive.byteOffset, s.alive.byteLength), off); off += s.alive.byteLength;
  }

  if (off !== total) throw new Error(`blob size mismatch: wrote ${off}, expected ${total}`);

  return new Blob([buf], { type: 'application/octet-stream' });
}

/** Trigger a browser download of `blob` as `filename`. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

export interface SnapshotResult {
  blob: Blob;
  filename: string;
  sizeMB: string;
  numCheckpoints: number;
}

/**
 * Run one 10 keV pass with snapshot capture and return the blob (no download).
 * Caller decides what to do with it: download, hand off to viewer, etc.
 */
export async function generateSnapshotAt10keV(cfg: ExportConfig): Promise<SnapshotResult | null> {
  const { np, boxNm, ceEV, log } = cfg;
  const snapshot_n = cfg.snapshot_n ?? 50_000;

  const cc = await ensurePipelines(np, log);
  if (!cc) {
    log('GPU init failed, cannot generate snapshot.', 'err');
    return null;
  }
  const { device, buffers, pipelines } = cc;

  log(`Snapshot capture: ${np} primaries @ 10 keV, snap_n=${snapshot_n}`, 'data');

  const dna = buildDNATarget(DNA_LENGTH_NM, DNA_GRID_N, DNA_SPACING_NM);

  const chemCallback = async (
    _radBuf: Float32Array,
    radN: number,
    nTherm: number,
    E_eV: number,
  ): Promise<ChemResult | null> =>
    runChemistryGPU(device, buffers, pipelines, radN, E_eV, nTherm, {
      dump_snapshots: true,
      snapshot_n,
    });

  const r = await runAtEnergy(device, buffers, pipelines, 10_000, np, boxNm, ceEV, dna, chemCallback);

  if (!r.chem_result?.snapshots || !r.rad_buf_final) {
    log('Snapshot capture: no chemistry result returned.', 'err');
    return null;
  }

  const initialN = Math.min(r.rad_n_stored, snapshot_n);
  const blob = packSnapshotBlob(
    r.chem_result.snapshots,
    dna,
    r.rad_buf_final,
    initialN,
    10_000,
  );

  const sizeMB = (blob.size / 1024 / 1024).toFixed(2);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const filename = `wgdna4d-${np}p-${snapshot_n}n-${stamp}.bin`;

  return { blob, filename, sizeMB, numCheckpoints: r.chem_result.snapshots.length };
}

/**
 * Open the splat viewer in a new tab and hand off the blob via postMessage.
 * The viewer signals readiness with a "splat-ready" message; we reply with
 * { type: 'snapshot', blob }. If the popup is blocked, falls back to
 * downloading the file so the user can open it manually.
 */
export function openInViewer(blob: Blob, log: LogFn): void {
  const win = window.open('/splat.html#handoff=1', '_blank');
  if (!win) {
    log('Popup blocked. Falling back to download.', 'err');
    downloadBlob(blob, 'wgdna4d-handoff.bin');
    return;
  }
  const handler = (e: MessageEvent): void => {
    if (e.source !== win) return;
    if (e.data === 'splat-ready') {
      win.postMessage({ type: 'wgdna-snapshot', blob }, '*');
      window.removeEventListener('message', handler);
    }
  };
  window.addEventListener('message', handler);
  // Safety: stop listening after 30s in case the viewer never came up.
  setTimeout(() => window.removeEventListener('message', handler), 30_000);
}

/**
 * Backwards-compat: run + download in one call. Used by the "Save .bin"
 * button if/when callers prefer the offline file path.
 */
export async function exportSnapshotAt10keV(cfg: ExportConfig): Promise<void> {
  const r = await generateSnapshotAt10keV(cfg);
  if (!r) return;
  downloadBlob(r.blob, r.filename);
  cfg.log(
    `Snapshot export: wrote ${r.filename} (${r.sizeMB} MB, ${r.numCheckpoints} checkpoints)`,
    'ok',
  );
}
