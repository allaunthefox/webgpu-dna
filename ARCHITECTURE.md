# WebGPU Geant4-DNA Architecture

## Core Design Decision

Geant4 tracks one particle at a time, sequentially. We invert this:

- **Phase A (primaries):** 1 dispatch, 1 GPU thread per primary. Each thread runs the full particle history in a loop — no per-step dispatch overhead. This is "kernel fusion": the entire track-structure simulation is a single `@compute` entry point.
- **Phase B (secondaries):** Wavefront stepping. 1 dispatch per physics step, all secondaries advance in parallel. Can't fuse because secondary count isn't known until Phase A completes.
- **Phase C (chemistry):** 4 dispatches per timestep (diffuse → clear_hash → build_hash → react), repeated 133 times across 7 time checkpoints from 0.1 ps to 1 μs.

## Pipeline Overview

```
  JS Orchestrator
  runValidation() → makePipes() → runAtEnergy() × 8 energies
       │
       │  fetch('/cross_sections.wgsl')
       │  ↓ prepended to SHADER + SEC_SHADER at compile time
       │
       ▼
  ┌──────────┐  counters[6]  ┌──────────────┐  counters[7]  ┌──────────────┐
  │ PHASE A  │──(sec_n)────▶│   PHASE B    │──(rad_n)────▶│   PHASE C    │
  │ Primary  │               │  Secondary   │               │  Chemistry   │
  │ Tracking │               │  Wavefront   │               │  Diffusion   │
  │          │               │  Stepper     │               │  + Reactions  │
  └──────────┘               └──────────────┘               └──────┬───────┘
   1 dispatch                 2000 dispatches                133×4  │
   4096 threads               ~900k threads each             dispatches
                                                                   │
                                                                   ▼
                                                          ┌────────────────┐
                                                          │ JS Post-Process│
                                                          │ - DSB scoring  │
                                                          │ - Dose project │
                                                          │ - ESTAR compare│
                                                          │ - G(t) timeline│
                                                          └────────────────┘

  Shared buffers across A+B:
    dose[]      ← atomicAdd from both phases (128³ voxel grid)
    rad_buf[]   ← append from both phases (OH/eaq/H/H3O+ positions)
    counters[]  ← atomic species counts + append indices
```

## Phase A: Primary Electron Tracking (Fused Kernel)

One GPU thread traces one primary electron through its entire history.
No inter-step synchronization, no per-step dispatch — pure loop.

```
  GPU Thread (1 of 4096)
  ──────────────────────
  for step = 0 to 65536:
    if E < cutoff or escaped: break

    ┌─ Cross sections ────────────────────────────────┐
    │  xs = xs_all(E)        → (σ_ion, σ_exc, σ_el)  │
    │  s_vib = xs_vib_total(E) → σ_vib                │
    │  σ_total = σ_ion + σ_exc + σ_el + σ_vib         │
    │  λ = 1 / (NW × σ_total)                         │
    │  dist = -ln(U) × λ                              │
    └─────────────────────────────────────────────────┘
          │
          ▼
    ┌─ Direction (every step) ────────────────────────┐
    │  Champion elastic angular CDF → cos_θ           │
    │  Rotate direction by (cos_θ, φ)                 │
    │  Move: pos += dir × dist                        │
    │  Deposit dose (atomicAdd to voxel grid)         │
    └─────────────────────────────────────────────────┘
          │
          ▼
    ┌─ Process selection (r × σ_total) ──────────────┐
    │                                                 │
    │  r < σ_ion?                                     │
    │    YES → Ionization                             │
    │      1. Shell selection (Born per-shell σ)      │
    │      2. W_transfer from paired CDF (bsearch)    │
    │      3. sec_KE = W_transfer - binding           │
    │      4. E -= W_transfer (NOT bind + W_sec)      │
    │      5. BornAngle: secondary direction           │
    │         <50eV: isotropic                        │
    │         50-200: 10% iso + 90% forward           │
    │         >200: kinematic sin²θ                   │
    │      6. Primary momentum conservation            │
    │         p_final = p_inc - p_sec                  │
    │      7. Emit secondary → sec_buf[]              │
    │      8. Radicals → rad_buf[]: OH + eaq + H3O+   │
    │                                                 │
    │  r < σ_ion + σ_exc?                             │
    │    YES → Excitation                             │
    │      Level from Born per-level fractions         │
    │      G4ChemDissociationChannels:                │
    │        Lev 0 (A1B1): 65% OH+H                   │
    │        Lev 1 (B1A1): 55% autoion, 15% 2OH+H₂   │
    │        Lev 2-4:      50% autoionization          │
    │      Autoion → OH + eaq + H3O+ (like ionization)│
    │                                                 │
    │  r < σ_ion + σ_exc + σ_vib?                     │
    │    YES → Vibrational (Sanche, 9 modes, 2× liq)  │
    │                                                 │
    │  else → Elastic (no energy loss, already rotated)│
    └─────────────────────────────────────────────────┘
```

## Phase B: Secondary Electron Wavefront Stepper

Can't fuse: secondary count (sec_n) unknown until Phase A finishes.
Instead: 2000 dispatches, each advancing all alive secondaries by 1 step.

```
  for dispatch = 0 to 2000:
    GPU Thread (1 of sec_n, ~900k at 10 keV)
    ──────────────────────────────────────────
    if dead: return early
    Same physics as Phase A except:
      - Tertiaries absorbed in place (not emitted to another buffer)
      - deposit(bind + W_sec) for tertiaries (full transfer deposited locally)
    Writes to same shared dose[], rad_buf[], counters[]
```

## Phase C: Chemistry (Spatial Hash Diffusion-Reaction)

```
  ┌─ Initialization (once) ──────────────────────────┐
  │  sample_init: rad_buf → chem_pos (copy first N)  │
  │  init_thermal: Gaussian displacement per species  │
  │    OH:  σ = 0.46 nm  (RMS 0.8 nm / √3)          │
  │    eaq: σ = 3.46 nm  (Meesungnoen 2002)          │
  │    H:   σ = 1.30 nm  (A1B1 kinematics)           │
  │    H3O+: 50% σ=0.46 nm, 50% no displacement      │
  └───────────────────────────────────────────────────┘
       │
       ▼
  ┌─ Time loop: 7 checkpoints, 133 total steps ──────┐
  │                                                   │
  │  For each timestep:                               │
  │                                                   │
  │  diffuse ──▶ clear_hash ──▶ build_hash ──▶ react  │
  │    │              │              │            │    │
  │    │ Brownian     │ Zero 8M     │ Insert    │ Walk│
  │    │ σ=√(2Dt)     │ buckets     │ into 1.5nm│ 27  │
  │    │ per species  │             │ cells     │ nbr │
  │    │              │             │           │ cells│
  │    │ D_OH  = 2.2  │             │           │     │
  │    │ D_eaq = 4.9  │             │           │     │
  │    │ D_H   = 7.0  │             │           │     │
  │    │ D_H3O+= 9.0  │             │           │     │
  │    (nm²/ns)       │             │           │     │
  │                                                   │
  │  After each checkpoint: count_alive → G(t)        │
  │                                                   │
  │  Reactions (Geant4 G4EmDNAChemistry rates):       │
  │    OH+OH   → H₂O₂  k=4.4e9   R=0.44 pc=0.376    │
  │    OH+eaq  → OH⁻    k=2.95e10 R=0.57 pc=0.980    │
  │    OH+H    → H₂O    k=1.44e10 R=0.45 pc=0.511    │
  │    eaq+eaq → H₂     k=5.0e9   R=0.54 pc=0.125    │
  │    eaq+H   → H₂     k=2.65e10 R=0.61 pc=0.455    │
  │    eaq+H3O+→ H      k=2.11e10 R=0.50 pc=0.500*   │
  │    H+H     → H₂     k=1.2e10  R=0.34 pc=0.216    │
  │                                                   │
  │    * R/pc tuned for grid scheme; IRT needed for   │
  │      accurate spur kinetics of this reaction      │
  └───────────────────────────────────────────────────┘
```

## JS Post-Processing (CPU)

```
  After GPU phases complete:
    1. Readback dose grid → sum for E_cons, render XY/YZ projections
    2. Readback results[] → per-primary CSDA, ions, stopping power
    3. Readback rad_buf[] → direct SSB scoring (ionization site clustering)
    4. Readback chem_pos/alive @ 1μs → indirect SSB (diffused OH near DNA)
    5. Cluster SSB → DSB (greedy ±10 bp)
    6. Compare CSDA vs ESTAR reference at 8 energies
    7. Report G(t) timeline vs Karamitros 2011
```

## Buffer Map & Data Flow

```
                     ┌──────────────────────┐
                     │   GPU MEMORY ~850 MB │
                     └──────────────────────┘

  Phase A/B shared               Phase C chemistry
  ────────────────               ──────────────────
  rad_buf    256 MB ─────────▶  chem_pos    128 MB
  sec_buf    240 MB              chem_rng    128 MB
  dose         2 MB              chem_alive   32 MB
  counters    32  B              cell_head    32 MB
  results    128 KB              next_idx     32 MB
  rng         64 KB              chem_stats  128  B
  params      64  B              chem_uni     16  B

  Data flow:
    Phase A writes → sec_buf     (secondaries for Phase B)
    Phase A writes → rad_buf     (radical positions)
    Phase A writes → counters[6] (sec_n for Phase B dispatch count)
    Phase A writes → counters[7] (rad_n for Phase C input count)
    Phase B reads  → sec_buf     (one step per dispatch)
    Phase B writes → rad_buf, dose, counters (shared with A)
    Phase C reads  → rad_buf via sample_init → chem_pos
    Phase C reads  → chem_pos via build_hash → cell_head + next_idx
    Phase C writes → chem_alive (react kills pairs)
    Phase C reads  → chem_stats (count_alive at checkpoints)

  Cross sections (WGSL const arrays, compiled into shader):
    fetch('/cross_sections.wgsl') → prepended to SHADER + SEC_SHADER
    Total XS:   XI/XC/XL × 100 pts (Born ion/exc, Champion elastic)
    Ion CDF:    XWC/XWT × 5 shells × 58 energies × 100 breakpoints
    Elastic:    XAE/XAC 25×25 (Champion angular CDF)
    Vibrational: XVE/XVS/XVMF (Sanche 38 pts × 9 modes)
    Shell/Exc:  XSF/XEF × 5 × 100 pts (per-shell/level fractions)
```

## Geant4 → WGSL Model Mapping

| Geant4 C++ | Data file | WGSL function | Key difference |
|-------------|-----------|---------------|----------------|
| G4DNABornIonisationModel1 | sigma_ionisation_e_born | `xs_all().x` + `sample_W_sec()` | Paired CDF binary search vs Geant4's std::map |
| G4DNABornAngle | (analytical) | Inline in ionization block | 3 regimes: <50eV iso, 50-200 mixed, >200 kinematic |
| G4DNABornExcitationModel1 | sigma_excitation_e_born | `xs_all().y` + `xs_exc_fracs()` | Data-driven level fractions, not hardcoded |
| G4ChemDissociationChannels | (code) | Inline branching | Autoionization for levels 1-4 (produces eaq) |
| G4DNAChampionElasticModel | sigma_elastic_e_champion | `xs_all().z` + `xs_el_cos()` | Scale: 1e-16 cm² (not Emfietzoglou's 1e-22/3.343 m²) |
| G4DNASancheExcitationModel | sigma_excitationvib_e_sanche | `xs_vib_total()` + `sample_vib_mode()` | 2× liquid phase factor applied |
| G4DNAWaterDissociationDisplacer | (code) | `init_thermal()` kernel | Gaussian σ per species from Geant4 RMS values |
| G4EmDNAChemistry | (code) | `react()` kernel | Grid hash pair discovery vs Geant4's IRT |

## Performance

```
                   Geant4 (CPU)            WebGPU (GPU)
                   ─────────────           ─────────────
 Runtime           Single-thread C++       Parallel WGSL compute
 Parallelism       1 particle at a time    4096 pri + 900k sec + 6.5M radicals
 4096 × 10 keV     ~4 min                  ~8s physics + ~15s chemistry
 Speedup                                   ~10-20×

 Physics match:
   Ionization/nm   0.0713                  0.0737 (within 4%)
   Energy cons.    100.0%                  99.7%
   MFP                                     within 2-14% at all energies
   Mean W_transfer 57.14 eV                57.15 eV (1.000×)

 Known limitation:
   G-values at 1μs limited by grid-based chemistry (not IRT).
   Grid hash misses long-range diffusive encounters → G(H₂O₂) and G(H₂) low.
   eaq+H3O+ spur kinetics need IRT for accurate G(eaq)/G(H) balance.
```
