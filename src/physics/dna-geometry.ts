import type { DNATarget } from './types';

/**
 * Build a plaquette of parallel straight B-DNA fibers.
 *
 * Layout (matches public/geant4dna.html buildDNATarget):
 *  - `grid_N × grid_N` fibers along X on a regular (y,z) lattice, spacing
 *    `spacing_nm`.
 *  - Each fiber has `n_bp_per = floor(L_nm / 0.34)` base pairs at rise 0.34 nm.
 *  - Two backbone strands helically wound at radius `r_bb = 1.0` nm, pitch
 *    10.5 bp/turn. Strand offsets pre-computed into `rbb0`/`rbb1` (length
 *    2 × n_bp_per = (y₀,z₀,y₁,z₁,…)), independent of fiber.
 *
 * Default: 21×21 × 3000 nm × 150 nm = 3.89 Mbp.
 */
export function buildDNATarget(
  L_nm: number = 3000,
  grid_N: number = 21,
  spacing_nm: number = 150,
): DNATarget {
  const rise = 0.34;
  const bp_per_turn = 10.5;
  const r_bb = 1.0;
  const n_bp_per = Math.floor(L_nm / rise);
  const x0 = -(n_bp_per - 1) * rise * 0.5;
  const d_phase = (2 * Math.PI) / bp_per_turn;
  const n_fibers = grid_N * grid_N;
  const n_bp = n_fibers * n_bp_per;

  // Per-fiber (y, z) offset on the grid.
  const fy = new Float32Array(n_fibers);
  const fz = new Float32Array(n_fibers);
  const off = -((grid_N - 1) * spacing_nm) * 0.5;
  for (let fi = 0; fi < grid_N; fi++) {
    for (let fj = 0; fj < grid_N; fj++) {
      const idx = fi * grid_N + fj;
      fy[idx] = off + fi * spacing_nm;
      fz[idx] = off + fj * spacing_nm;
    }
  }

  // Two backbone strands: precompute fiber-independent (y,z) offsets per bp.
  const rbb0 = new Float32Array(n_bp_per * 2);
  const rbb1 = new Float32Array(n_bp_per * 2);
  for (let i = 0; i < n_bp_per; i++) {
    const phi = i * d_phase;
    rbb0[i * 2 + 0] = r_bb * Math.cos(phi);
    rbb0[i * 2 + 1] = r_bb * Math.sin(phi);
    rbb1[i * 2 + 0] = r_bb * Math.cos(phi + Math.PI);
    rbb1[i * 2 + 1] = r_bb * Math.sin(phi + Math.PI);
  }

  return {
    rise,
    r_bb,
    n_bp_per,
    n_fibers,
    n_bp,
    grid_N,
    spacing_nm,
    x0,
    L_nm,
    fy,
    fz,
    rbb0,
    rbb1,
  };
}
