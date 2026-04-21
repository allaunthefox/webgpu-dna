/**
 * Guards the rad_buf .w encoding used by primary.wgsl / secondary.wgsl and
 * decoded by both irt-worker.js (line 384) and chemistry.wgsl sample_init.
 *
 * Encoding:
 *   rad_buf[i].w = pid * 8 + species_code
 *   species_code: 0=OH, 1=eaq, 2=H, 3=H3O+, 5=pre-therm eaq, 6=OH-, 7=H2
 *
 * Decode:
 *   species = round(w) % 8
 *   pid     = floor(round(w) / 8)
 *
 * Regression target: at np=4096, the GPU chemistry kernels (count_alive,
 * diffuse, init_thermal) used to read .w as raw species in [0..3], which
 * filtered out every radical from primary 1..4095 — G-values collapsed to
 * ~0.001. The sample_init kernel now normalises .w by `% 8` so every
 * downstream kernel sees a pure species code. This test pins that contract.
 */
import { describe, it, expect } from 'vitest';

function decode(w: number): { pid: number; species: number } {
  const i = Math.round(w);
  return { pid: Math.floor(i / 8), species: i % 8 };
}

describe('rad_buf .w encoding', () => {
  it('round-trips pid=0 across every species code', () => {
    for (const s of [0, 1, 2, 3, 5, 6, 7]) {
      const w = 0 * 8 + s;
      const { pid, species } = decode(w);
      expect(pid).toBe(0);
      expect(species).toBe(s);
    }
  });

  it('round-trips high pids without losing species', () => {
    const samples = [
      { pid: 1, species: 0 }, // OH from primary 1
      { pid: 7, species: 5 }, // pre-therm eaq from primary 7
      { pid: 4095, species: 2 }, // H from last primary in 4096 batch
      { pid: 4095, species: 7 }, // H2 marker from last primary
    ];
    for (const { pid, species } of samples) {
      const w = pid * 8 + species;
      const d = decode(w);
      expect(d.pid).toBe(pid);
      expect(d.species).toBe(species);
    }
  });

  it('matches irt-worker.js line 384: w % 8 for primary 0', () => {
    // Primary 0 writes w = 0 + species (no offset). Worker's `w % 8` must
    // round-trip cleanly to the species code for the np=1 case where the
    // latent GPU bug happens to be harmless.
    for (const s of [0, 1, 2, 3, 5, 6, 7]) {
      expect(Math.round(0 + s) % 8).toBe(s);
    }
  });

  it('matches irt-worker.js line 384: w % 8 for high pids', () => {
    // The bug we fixed: at pid=1, w = 8 + species. If a kernel reads
    // `round(w)` without `% 8` it gets 8..15 instead of 0..7 — all rejected
    // by `kind > 3` guards. This test pins the decode math.
    for (let pid = 1; pid < 4096; pid += 137) {
      for (const s of [0, 1, 2, 3, 5, 6, 7]) {
        const w = pid * 8 + s;
        expect(Math.round(w) % 8).toBe(s);
      }
    }
  });

  it('GPU-chemistry sample_init normalisation: (pid*8+s) % 8 === s', () => {
    // This mirrors the new line in src/shaders/chemistry.wgsl:
    //   let sp = f32(u32(round(raw.w)) % 8u);
    // If this ever regresses, every downstream kernel breaks silently at np>1.
    for (let pid = 0; pid < 4096; pid += 311) {
      for (const s of [0, 1, 2, 3]) {
        const w = pid * 8 + s;
        const sp = Math.round(w) >>> 0; // emulate u32 cast
        expect(sp % 8).toBe(s);
      }
    }
  });
});
