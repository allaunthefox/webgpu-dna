/**
 * Chemistry backend selection.
 *
 * The validation harness can drive radiolysis chemistry via three paths:
 *   - 'worker': off-main-thread IRT (public/irt-worker.js). Matches Karamitros
 *     2011 G-values. This is what public/geant4dna.html line 2264 uses.
 *   - 'gpu':    WebGPU grid-hash chemistry (src/shaders/chemistry.wgsl).
 *     Faster but undercounts long-time reactions vs IRT, and its product
 *     tracking (H2O2/H2/OH-) is less complete than the worker's.
 *   - 'none':   skip chemistry (Phase A+B only). Useful for CSDA-only testing.
 *
 * Kept in its own module — free of WebGPU imports — so unit tests can pin
 * the default without dragging `GPUBufferUsage` through the import graph.
 */

export type ChemBackend = 'gpu' | 'worker' | 'none';

export const DEFAULT_CHEM_BACKEND: ChemBackend = 'worker';
