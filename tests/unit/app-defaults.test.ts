/**
 * Pins the default chemistry backend. Changing this from 'worker' to 'gpu'
 * silently collapses G-values to ~0.001 at np>1 until the GPU chemistry path
 * is rewritten to handle the full `pid*8 + species` encoding plus all the
 * product-tracking that irt-worker.js does. Until then, 'worker' is the only
 * backend that reproduces Karamitros 2011 G-values, matching
 * public/geant4dna.html line 2264.
 */
import { describe, it, expect } from 'vitest';
import { DEFAULT_CHEM_BACKEND } from '../../src/chemistry/backend';

describe('App defaults', () => {
  it('uses the IRT worker chemistry backend by default', () => {
    expect(DEFAULT_CHEM_BACKEND).toBe('worker');
  });
});
