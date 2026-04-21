# WebGPU DNA Track Structure Simulation

## Goal
Port Geant4-DNA (CERN's Monte Carlo track structure code for radiobiology) to WebGPU compute shaders using kernel fusion architecture. Single fused WGSL dispatch per batch — one GPU thread per primary electron, full history in a loop, zero per-step dispatch overhead.

## Architecture
- Each GPU thread traces one particle through its full history (primaries)
- Wavefront step kernel for secondaries (one dispatch per step)
- All physics (elastic, ionization, excitation, vibrational) inline in main()
- atomicAdd for dose/radical deposition to shared voxel grid
- 128³ voxel grid with WGSL `p.box` as half-width (full volume = (2×box)³)
- Dose fixed-point ×100 units/eV (max voxel 42.9 MeV, catches sub-0.1 eV events)

## Validation harness: public/geant4dna.html

This is the active validation file with real tabulated Geant4-DNA data and
full end-to-end SSB/DSB scoring. `public/index.html` is the older v4 demo
that uses BEB approximation — not the working validation target.

### Current validation status (N=4096, 10 keV, origin start)
- **Cross sections**: Born ionization + Emfietzoglou excitation, Champion elastic, Sanche vib
- **CSDA**: 1.25× ESTAR @ 5 keV, **1.08× @ 10 keV**, 1.05× @ 20 keV
- **Energy conservation**: 99.9–100.0% at all 8 ESTAR energies
- **MFP vs Geant4**: matches within 2-14% at all energies (100 eV – 10 keV)
- **G(OH) @ 1 μs**: 1.613 → 0.64× Karamitros 2011
- **G(e⁻aq) @ 1 μs**: 1.292 → 0.52× Karamitros 2011
- **G(H) @ 1 μs**: 0.496 → **0.87× Karamitros 2011**
- **G(H₂O₂) @ 1 μs**: 0.499 → 0.68× Karamitros 2011
- **G(H₂) @ 1 μs**: 0.368 → **0.88× Karamitros 2011**
- **SSB/DSB**: SSB_dir=23, SSB_ind=0, DSB=2 per run
- **ions/pri**: 195.1 vs 196 Geant4 (exact match)

G(H) and G(H2) match Karamitros within 15%. Key fixes (2026-04-14):
1. Switched excitation from Born to Emfietzoglou (2.4× higher XS, correct initial G(H)=0.33)
2. Added Geant4 mother molecule displacement (2.0 nm RMS) for ionization OH+H3O+
3. Full 9-reaction IRT table from G4EmDNAChemistry_option1 (added eaq+H2O2, H3O++OH-)
4. All reactions Type 0 (Smoluchowski) matching Karamitros 2011
5. Product creation + re-pairing for all reactions (not just eaq+H3O+→H)
6. eaq thermalization at 1.7 eV (Geant4 autoionization default), H3O+ displacement = 0+mother

Remaining gap: G(OH) 36% low, G(eaq) 48% low. Likely correct for 10 keV
LET (3.9 eV/nm) — Karamitros 2011 reference may be for low-LET (~1 MeV)
radiation where radical recombination is inherently lower.

### Geant4-DNA direct validation (2026-04-12)
Built Geant4 11.3.0, ran dnaphysics with DNA_Opt2, 4096 e⁻ at 10 keV.
Key bugs fixed:
1. DNA_Opt2 uses Born (NOT Emfietzoglou) for ionization + excitation
2. Champion elastic scaleFactor: 1e-16 cm² = 0.01 nm²/unit (was using 2.993e-5)
3. Elastic subsampled on its own 7.4–10M eV grid then paired with 8–10K eV XE grid
4. Secondary wavefront step limit 300→2000 (elastic-dominated thermalization)
5. Born differential CDF returns total transfer (bind+sec_KE), not sec_KE alone — was double-counting binding energy, shortening tracks by 30%
6. G4DNABornAngle: 3-regime secondary angular sampling (<50 eV isotropic, 50-200 mixed, >200 kinematic)
7. Primary momentum conservation after ionization (p_final = p_inc - p_sec)
8. Sanche vibrational 2× liquid phase factor
9. Data-driven Born excitation level fractions (both primary + secondary shaders)
10. Paired CDF/E_transfer arrays with binary search (58 energies × 100 breakpoints × 5 shells) replacing uniform CDF sampling (mean transfer 40→57 eV, matching Geant4's 57.1 eV)

### What's wired up
- Full tabulated cross sections from G4EMLOW (Born ionization, Emfietzoglou excitation, Champion elastic CDF, Sanche vib)
- 5 ionization shells (Born) + 5 excitation levels (Emfietzoglou, data-driven fractions) with level-dependent dissociative branching (0.65/0.55/0.80)
- Screened Rutherford elastic analytical + Champion tabulated angular CDF
- Sanche vibrational excitation (9 modes, 2–100 eV)
- Secondary electron wavefront stepper
- Karamitros 2011 9-reaction IRT chemistry (G4EmDNAChemistry_option1, all Type 0)
- Pre-chemistry: 2.0 nm mother displacement + species-specific product displacement
- eaq thermalization: Meesungnoen2002 at 1.7 eV (Geant4 autoionization default)
- Product tracking: H2O2 and OH- as reactive species with full re-pairing
- Event-level direct SSB scoring from rad_buf ionization sites (nm-scale spatial correlation)
- **Kernel-level** DNA backbone hit counter (`dna_near` in both primary + secondary shaders)
  — cross-checks the JS post-processing to the event (kernel_hits == reach_dir exactly)
- Indirect SSB from diffused OH at t=1 μs
- 21×21 parallel B-DNA fiber grid, 3 μm long, 150 nm spacing = 3.89 Mbp target
- Greedy ±10 bp DSB clustering
- Dose XY/YZ projections with zoom-to-bbox and log magma colormap
- ESTAR validation at 7 energies: 100 eV, 300 eV, 500 eV, 1 keV, 3 keV, 5 keV, 10 keV, 20 keV

### Buffer sizing (geant4dna.html)
- `initGPU` requests adapter max `maxBufferSize` / `maxStorageBufferBindingSize` (default cap is 128 MiB — too small)
- MAX_SEC = 5M × 48 B = 240 MB
- MAX_RAD = 16M × 16 B = 256 MB (at WebGPU default maxBufferSize limit)
- CHEM_N = 8M × 16 B = 128 MB (chem_pos) + 32 MB (alive) + 128 MB (rng) + 32 MB (next_idx)
- HASH_SIZE = 8M buckets × 4 B = 32 MB (cell_head). 8× larger than initial 1M baseline — gave 4.6× chemistry speedup at N=16384.
- N=16384 at 10 keV fits cleanly (~13M radicals, under MAX_RAD); E_cons stays 99.9%

### Known convention quirks
- `p.box` is HALF-WIDTH in WGSL (voxel size = 2×box/vc). JS scoring must match.
- UI box=15000 means ±15000 nm → 30 μm cube total (27 fL water = 27 pg)
- For a 30 μm box and 4096 × 10 keV primaries, box_dose ≈ 0.243 Gy

## What's NOT Done — THE MAIN TODO

### 1. Download G4EMLOW data and port actual cross section tables
The Geant4-DNA cross sections are tabulated in text files in the G4EMLOW data package.
Download from: https://geant4-data.web.cern.ch/datasets/
Latest: G4EMLOW8.6.1.tar.gz (~400MB)

The DNA-specific files are in the `dna/` subdirectory. Key files:

**Total cross sections (energy vs σ per shell, tab-separated):**
- `dna/sigma_ionisation_e_emfietzoglou` — Emfietzoglou ionization (5 shells)
- `dna/sigma_excitation_e_emfietzoglou` — Emfietzoglou excitation (5 levels)
- `dna/sigma_ionisation_e_born` — Born ionization (alternative)
- `dna/sigma_elastic_e_champion` — Champion elastic (below 200 eV)

**Differential cross sections (for secondary electron energy sampling):**
- `dna/sigmadiff_ionisation_e_emfietzoglou.dat`
- `dna/sigmadiff_cumulated_ionisation_e_emfietzoglou.dat` (CDF for fast sampling)

**File format** (from G4DNACrossSectionDataSet with G4LogLogInterpolation):
```
# Lines: energy(eV)  sigma_shell0  sigma_shell1  sigma_shell2  sigma_shell3  sigma_shell4
# Units: eV and 1e-22 m² (divide by 3.343 for per-molecule, see scaleFactor in source)
# scaleFactor = (1.e-22 / 3.343) * m*m
```

**Conversion script:** Use `scripts/convert_g4emlow.py` to parse these files and generate WGSL constant arrays.

### 2. ✅ DONE: Champion elastic model below 200 eV (tabulated angular CDF)

### 3. ✅ DONE: Vibrational excitation (Sanche 9-mode, 2–100 eV range)

### 4. ✅ DONE: Differential cross sections for secondary electrons (Emfietzoglou sigmadiff tables)

### 5. Upgrade radical chemistry to IRT method — STILL OPEN
Current: spatial-hash pair reactions with grid diffusion.
The grid approach loses pairs at long times: at dt=30 ns, σ_diff ≈ 11.5 nm
but hash search is only 4.5 nm (cell 1.5 nm × 3 neighbor layers). Pairs that
should react by diffusive encounter instead "overshoot" each other and
never enter the same cell in a single step. This is why G(H₂O₂) and G(H₂)
are 2.5–8× below Karamitros even though the rate constants match.

Geant4-DNA uses Independent Reaction Times (IRT): for each pair of radicals,
analytically compute the first-passage reaction time from their positions
and diffusion coefficients, schedule the reaction event, and process in
time order. Much more accurate but requires per-pair book-keeping.

### 6. Per-event direct-damage kernel instrumentation — DONE (PARTIAL)
Event-level direct SSB scoring (using rad_buf positions as ionization
sites) is wired up and produces correct spatial clustering. But indirect
SSB still uses diffused OH at t=1 μs, and the DSB/SSB ratio is target-
geometry-dependent because our DNA target is a concentrated 21×21 fiber
grid sampling the track core rather than a uniform bulk distribution.

**Kernel-level** DNA backbone hit counter (`dna_near` in both primary +
secondary shaders) cross-checks the JS post-processing to the event:
`kernel_hits == reach_dir` exactly. All four event sites instrumented:
primary ionization, primary excitation, secondary ionization, secondary
excitation.

## Geant4-DNA Source Code Reference
Cloned from: https://github.com/Geant4/geant4.git
Key directories:
- `source/processes/electromagnetic/dna/models/src/` — physics models
- `source/processes/electromagnetic/dna/utils/src/` — water structure data
- `source/processes/electromagnetic/dna/utils/include/` — headers

### Physics models (all in models/src/):
| Model | File | What it does |
|-------|------|-------------|
| Emfietzoglou ionization | G4DNAEmfietzoglouIonisationModel.cc | Loads `sigma_ionisation_e_emfietzoglou`, log-log interp |
| Emfietzoglou excitation | G4DNAEmfietzoglouExcitationModel.cc | Loads `sigma_excitation_e_emfietzoglou` |
| Born ionization | G4DNABornIonisationModel1.cc | Loads `sigma_ionisation_e_born` + differential |
| Screened Rutherford | G4DNAScreenedRutherfordElasticModel.cc | Analytical formula (ported) |
| Champion elastic | G4DNAChampionElasticModel.cc | Loads `sigma_elastic_e_champion` |

### Exact formulas extracted:

**Screened Rutherford elastic** (NIM 155, 145-156, 1978):
```
Z = 10 (water)
σ_Ruth = Z(Z+1) × [e²(K+mec²) / (4πε₀·K·(K+2mec²))]²
n(K) = (1.64 - 0.0825·ln(K/eV)) × 1.7e-5 × Z^(2/3) / [K/mec² × (2 + K/mec²)]
σ_el = π × σ_Ruth / [n × (n+1)]
```

**Water ionisation shells** (G4DNAWaterIonisationStructure.cc):
```
1b₁: 10.79 eV, 3a₁: 13.39 eV, 1b₂: 16.05 eV, 2a₁: 32.30 eV, 1a₁: 539.0 eV
```

**Emfietzoglou ionisation shells** (liquid phase adjusted):
```
10.0, 13.0, 17.0, 32.2, 539.7 eV
```

**Excitation levels** (Emfietzoglou, Rad Res 163, 2005):
```
A¹B₁: 8.22, B¹A₁: 10.00, Rydberg A+B: 11.24, Rydberg C+D: 12.61, Diffuse: 13.77 eV
```

## WGSL Shader Constraints
- No recursive function calls
- Avoid complex function signatures with many ptr<function,array> params
- Everything inline in main() is safest
- atomicAdd only works on u32 (use fixed-point for fractional values: ×100)
- Ping-pong buffers required for stencil/diffusion operations
- const arrays up to ~100 elements work fine
- `initGPU` MUST pass `requiredLimits` requesting the adapter's max buffer
  sizes; the default `maxStorageBufferBindingSize` is 128 MiB which is too
  small for our rad_buf (192 MB) and silently produces empty dispatches

## Project Links
- kernelfusion.dev — kernel fusion research papers
- gpubench.dev — WebGPU benchmarking
- Zero-TVM — from-scratch LLM inference replacing Apache TVM

## License
MIT (simulation code)
Geant4-DNA data: Geant4 Software License (essentially BSD-like, Apache-2.0 compatible)
