# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/) starting
from `0.1.0`.

## [0.1.0] — 2026-05-04

First public release. The repo has been on GitHub for a while via
preview-only deploys; this is the line under which versioned releases start.

### Added

- **Physics engine** (`src/physics/` + `src/shaders/`) — Born ionization
  (5 shells, data-driven CDF sampling), Emfietzoglou excitation (5 levels,
  dissociative branching 0.65 / 0.55 / 0.80), Champion tabulated elastic
  angular CDF (< 200 eV), screened-Rutherford elastic (> 200 eV), Sanche
  9-mode vibrational (2–100 eV), full primary-momentum conservation. One
  GPU thread per primary electron, full particle history in a single fused
  compute dispatch.
- **Chemistry — Karamitros 2011 IRT** in a Web Worker (`src/chemistry/`):
  9-reaction Smoluchowski TDC + Onsager-screened PDC for charged pairs
  (G4EmDNAChemistry_option1). 2.0 nm mother displacement, species-specific
  product displacement, e⁻aq thermalization at 1.7 eV, H₂O₂ / OH⁻ tracked
  as reactive products with full re-pairing.
- **DNA scoring** (`src/scoring/`) — event-level direct SSB from `rad_buf`
  ionization sites, indirect SSB from diffused OH at 1 μs, greedy ±10 bp
  DSB clustering, kernel-level backbone hit counter as cross-check.
  Target: 21×21 parallel B-DNA fibers × 3 μm × 150 nm spacing = 3.89 Mbp.
- **Validation harness** (`validation/compare.py`) — side-by-side run
  against a Geant4-DNA ntuple (4096 primaries @ 10 keV).
- **G4EMLOW converter** (`tools/convert_g4data.py`) — Python pipeline that
  emits `public/cross_sections.wgsl` (1.3 MB committed) from the 245 MB
  CERN G4EMLOW reference data.
- **WGDNA-4D viewer** (`src/splat/`) — Gaussian-splat 4D visualisation of
  the simulation, with a one-click handoff from the landing page and a
  `/see` share view. Mobile-friendly with touch + pinch and a responsive
  control panel; perf-aware defaults for low-end devices.
- **Landing page** with verified browser-support claims, comparison table
  vs Geant4 direct, full SEO + social + PWA asset pass, and links to the
  companion projects ([kernelfusion.dev](https://kernelfusion.dev),
  [gpubench.dev](https://gpubench.dev),
  [zerotvm.com](https://zerotvm.com)).
- **Live deployment** at https://webgpudna.com.

### Validated against (N = 4096 primaries @ 10 keV)

| Metric                  | This build | Reference                  | Ratio       |
| ----------------------- | ---------- | -------------------------- | ----------- |
| CSDA range (nm)         | 2714.4     | 2756.5 (Geant4-DNA direct) | **0.985×**  |
| Energy conservation     | 100.0 %    | 100.0 %                    | 1.000×      |
| Ions per primary (full) | ≈ 509      | 509.1 (Geant4 direct)      | 1.00×       |
| G(OH) at 1 μs           | 1.55       | 2.50 (Karamitros 2011)     | 0.62×¹      |
| G(e⁻aq) at 1 μs         | 1.41       | 2.50                       | 0.56×¹      |
| G(H) at 1 μs            | 0.71       | 0.57                       | 1.24×       |
| G(H₂O₂) at 1 μs         | 0.60       | 0.73                       | 0.83×       |
| G(H₂) at 1 μs           | 0.47       | 0.42                       | 1.11×       |

¹ G(OH) / G(e⁻aq) at 10 keV LET are inherently below the Karamitros 2011
low-LET (~1 MeV) reference — track-core density drives higher radical
recombination.

### Test surface

- 46 unit tests across 7 files (Vitest).
- Geant4-DNA reference numbers shipped as JSON fixtures under
  `tests/fixtures/`.

### Known gaps

- GPU-resident chemistry path (`chemBackend: 'gpu'`) undercounts long-time
  reactions vs IRT because the spatial-hash search radius is narrower than
  the diffusion σ at 30 ns timesteps. Default backend is therefore the IRT
  worker.
- `data/g4emlow/` is not committed — download from CERN to rebuild
  cross sections via `npm run convert`.

[0.1.0]: https://github.com/abgnydn/webgpu-dna/releases/tag/v0.1.0
