# WebGPU Geant4-DNA

A WebGPU port of [Geant4-DNA](https://geant4-dna.in2p3.fr/) — CERN's Monte Carlo track-structure code for radiobiology — running entirely in the browser.

One GPU thread per primary electron, full particle history in a single fused compute dispatch, Karamitros 2011 Independent-Reaction-Time chemistry in a Web Worker, and SSB/DSB scoring on a 21×21 B-DNA fiber grid at 10 keV.

## Results (N = 4096 primaries @ 10 keV)

| Metric                     | This build | Reference                   | Ratio      |
| -------------------------- | ---------- | --------------------------- | ---------- |
| CSDA range (nm)            | 2714.4     | 2756.5 (Geant4-DNA direct)  | **0.985×** |
| Energy conservation        | 100.0 %    | 100.0 %                     | 1.000×     |
| Ions per primary (full)    | ≈ 509      | 509.1 (Geant4 direct)       | 1.00×      |
| G(OH) at 1 μs              | 1.55       | 2.50 (Karamitros 2011)      | 0.62×¹     |
| G(e⁻aq) at 1 μs            | 1.41       | 2.50                        | 0.56×¹     |
| G(H) at 1 μs               | 0.71       | 0.57                        | 1.24×      |
| G(H₂O₂) at 1 μs            | 0.60       | 0.73                        | 0.83×      |
| G(H₂) at 1 μs              | 0.47       | 0.42                        | 1.11×      |

¹ G(OH) / G(e⁻aq) at 10 keV LET are inherently below the Karamitros 2011 low-LET (~1 MeV) reference — track-core density drives higher radical recombination.

46 / 46 unit + integration tests pass. See `validation/compare.py` for the full side-by-side against a Geant4-DNA ntuple.

## Quick start

```bash
npm install
npm run dev            # http://localhost:8765
npm run test           # 46 tests, ~200 ms
npm run lint
npm run build          # dist/
```

Requires a WebGPU-capable browser (Chrome 113+, Safari TP, Edge, or Firefox with `dom.webgpu.enabled`).

## Project layout

```
src/
├── shaders/       WGSL compute shaders (helpers, primary, secondary, chemistry)
├── physics/       Constants, types, DNA geometry, cross-section loader
├── gpu/           Device init, buffers, pipelines, Phase A/B/C dispatch
├── chemistry/     IRT worker wiring, GPU chemistry schedule, reactions
├── scoring/       SSB/DSB scoring, ESTAR reference, dose projections
├── ui/            Results table, canvas dose projections
├── app.ts         runValidation orchestrator
└── main.ts        entry point

tests/             Vitest unit + integration (46 tests)
public/            Generated cross_sections.wgsl, irt-worker.js, monolithic reference HTML
tools/             Python + Node helpers (G4EMLOW converter, IRT driver)
validation/        Geant4-DNA comparison harness (compare.py, analyze_g4.py)
```

Deep-dive: [`ARCHITECTURE.md`](./ARCHITECTURE.md). Physics provenance and validation history: [`CLAUDE.md`](./CLAUDE.md).

## Regenerating cross sections

The committed `public/cross_sections.wgsl` (1.3 MB) is generated from the G4EMLOW reference data (245 MB, not committed). To rebuild:

```bash
# Download G4EMLOW from https://geant4-data.web.cern.ch/datasets/
# (e.g. G4EMLOW8.6.1.tar.gz) and extract so that data/g4emlow/dna/ exists.
npm run convert
```

## What's implemented

- **Physics:** Born ionization (5 shells, data-driven CDF sampling), Emfietzoglou excitation (5 levels, dissociative branching 0.65 / 0.55 / 0.80), Champion tabulated elastic angular CDF (< 200 eV), screened-Rutherford elastic (> 200 eV), Sanche 9-mode vibrational (2–100 eV), full primary-momentum conservation.
- **Chemistry:** Karamitros 2011 9-reaction IRT table (Type 0 Smoluchowski), 2.0 nm mother displacement, species-specific product displacement, e⁻aq thermalization at 1.7 eV, H₂O₂ / OH⁻ tracking with re-pairing.
- **DNA scoring:** Event-level direct SSB from `rad_buf` ionization sites, indirect SSB from diffused OH at 1 μs, greedy ±10 bp DSB clustering, kernel-level backbone hit counter as a cross-check.
- **Grid target:** 21×21 parallel B-DNA fibers × 3 μm × 150 nm spacing = 3.89 Mbp.

## Known gaps

- GPU-resident chemistry path (`chemBackend: 'gpu'`) undercounts long-time reactions vs IRT because the spatial hash search radius is narrower than the diffusion σ at 30 ns timesteps. Default backend is therefore the IRT worker.
- `data/g4emlow/` is not committed — download from CERN (link above) to rebuild cross sections.

## License

MIT for the simulation code. The Geant4-DNA cross-section data is distributed under the [Geant4 Software License](https://geant4.web.cern.ch/license/LICENSE.html) (BSD-like).
