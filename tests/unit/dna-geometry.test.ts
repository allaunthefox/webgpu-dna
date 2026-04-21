import { describe, it, expect } from 'vitest';
import { buildDNATarget } from '../../src/physics/dna-geometry';

describe('DNA target geometry', () => {
  const dna = buildDNATarget(3000, 21, 150);

  it('has 21x21 = 441 fibers', () => {
    expect(dna.n_fibers).toBe(441);
    expect(dna.grid_N).toBe(21);
  });

  it('has correct bp count (~8823 per fiber)', () => {
    const bpPer = Math.floor(3000 / 0.34);
    expect(dna.n_bp_per).toBe(bpPer);
    expect(dna.n_bp).toBe(441 * bpPer);
    expect(dna.n_bp).toBeGreaterThan(3_000_000); // ~3.89 Mbp
  });

  it('fiber spacing is 150 nm', () => {
    expect(dna.spacing_nm).toBe(150);
  });

  it('backbone radius is 1.0 nm', () => {
    expect(dna.r_bb).toBe(1.0);
  });

  it('rise is 0.34 nm/bp', () => {
    expect(dna.rise).toBeCloseTo(0.34, 2);
  });

  it('strand offset arrays have length 2 × n_bp_per', () => {
    expect(dna.rbb0.length).toBe(2 * dna.n_bp_per);
    expect(dna.rbb1.length).toBe(2 * dna.n_bp_per);
  });

  it('axial centering: x0 = -(n_bp_per-1) × rise / 2', () => {
    const expected = -(dna.n_bp_per - 1) * dna.rise * 0.5;
    expect(dna.x0).toBeCloseTo(expected, 6);
  });
});
