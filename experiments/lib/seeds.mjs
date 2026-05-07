// Named deterministic seeds for webgpu-dna experiments.
//
// Seeds are u32 values; experiments use them to drive RNGs in MC paths.
// To add a seed: pick a unique u32 (avoid 0 — xorshift identity), prefix
// with the experiment ID. Once committed, never reuse a seed for a new
// purpose; future runs with that seed must reproduce the same numbers.

export const SEEDS = Object.freeze({
  // Level 1 — cross-section bit-match. Most experiments are deterministic
  // (no MC); seeds are reserved for any CDF-inversion validation we add.
  E1_ION_XS:         0x4D6E5301,
  E2_EXC_XS:         0x4D6E5302,
  E3_ELASTIC_XS:     0x4D6E5303,
  E4_VIB_XS:         0x4D6E5304,

  // Level 2 — track structure (primary-history MC).
  E5_CSDA:           0x5452_4B01,
  E6_MFP:            0x5452_4B02,
  E7_IONS_PER_PRI:   0x5452_4B03,
  E8_E_SPECTRUM:     0x5452_4B04,

  // Level 3 — pre-chemistry (initial G-values at ~1 ps).
  E9_PRECHEM_G_INIT: 0x50524501,

  // Level 4 — chemistry @ 1 μs.
  E10_IRT_G_VALUES:  0x43484D01,
  E11_GPU_VS_IRT:    0x43484D02,

  // Level 5 — DNA damage.
  E12_DIRECT_SSB:    0x444E4101,
  E13_INDIRECT_SSB:  0x444E4102,
  E14_DSB_CLUSTER:   0x444E4103,

  // Level 6 — performance.
  E15_DISPATCH:      0x50455201,
  E16_FUSED_VS_NAIVE:0x50455202,
});
