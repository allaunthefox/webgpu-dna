import { describe, it, expect } from 'vitest';
import { IRT_REACTIONS, DIFFUSION } from '../../src/physics/constants';
import type { ReactionTuple } from '../../src/physics/types';

describe('Reaction table', () => {
  function findReaction(a: number, b: number): ReactionTuple | undefined {
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    return IRT_REACTIONS.find(r => r[0] === lo && r[1] === hi);
  }

  it('OH+eaq has pc=0.98 (near-saturated)', () => {
    const rxn = findReaction(0, 1);
    expect(rxn).toBeDefined();
    expect(rxn![3]).toBeCloseTo(0.98, 2);
  });

  it('all reaction radii are positive', () => {
    for (const rxn of IRT_REACTIONS) {
      expect(rxn[2]).toBeGreaterThan(0);
    }
  });

  it('all contact probabilities are in (0, 1]', () => {
    for (const rxn of IRT_REACTIONS) {
      expect(rxn[3]).toBeGreaterThan(0);
      expect(rxn[3]).toBeLessThanOrEqual(1);
    }
  });

  it('product codes are valid (0-3)', () => {
    for (const rxn of IRT_REACTIONS) {
      expect(rxn[4]).toBeGreaterThanOrEqual(0);
      expect(rxn[4]).toBeLessThanOrEqual(3);
    }
  });

  it('eaq+H3O+ Onsager-corrected σ is ~0.47 nm', () => {
    const rxn = findReaction(1, 3);
    expect(rxn).toBeDefined();
    expect(rxn![2]).toBeCloseTo(0.47, 2);
  });

  it('Smoluchowski rate k = 4πRD matches Geant4 rate constants', () => {
    // OH+OH: k=4.4e9 L/mol/s, R=0.44, D=2.2+2.2=4.4
    const rxn = findReaction(0, 0)!;
    const D = DIFFUSION.OH + DIFFUSION.OH;
    const k_smol = 4 * Math.PI * rxn[2] * D;
    const k_obs = k_smol * rxn[3];
    const k_SI = k_obs * 6.022e23 * 1e-24 * 1e9;
    expect(k_SI / 4.4e9).toBeGreaterThan(0.5);
    expect(k_SI / 4.4e9).toBeLessThan(2.0);
  });
});
