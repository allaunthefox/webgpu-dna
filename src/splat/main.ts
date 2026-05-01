/**
 * WGDNA-4D viewer — minimal WebGPU renderer for snapshot blobs produced by
 * `exportSnapshotAt10keV`. Each snapshot's radical positions are rendered as
 * instanced billboarded sprites coloured by species; a time slider scrubs
 * across the 8 checkpoints.
 *
 * v0 scope: load → parse → render radicals only → orbit camera → checkpoint
 * stepper. DNA fibers, dose grid overlay, temporal interpolation all come
 * later (tasks 5/6 in the build plan).
 */

interface Snapshot {
  label: string;
  t_ns: number;
  pos: Float32Array;   // length = 4*n
  alive: Uint32Array;  // length = n
}

interface Blob4D {
  version: number;
  num_snaps: number;
  snap_n: number;
  energy_eV: number;
  dna: {
    n_fibers: number;
    n_bp_per: number;
    fy: Float32Array;
    fz: Float32Array;
    rise: number;
    spacing_nm: number;
    L_nm: number;
    x0: number;
  };
  rad0: { rad_n_initial: number; rad_buf: Float32Array };
  snapshots: Snapshot[];
}

const MAGIC = 'WGDNA4D\0';

function parseBlob(buf: ArrayBuffer): Blob4D {
  const dv = new DataView(buf);
  const dec = new TextDecoder();
  let off = 0;

  // Header
  const magicBytes = new Uint8Array(buf, 0, 8);
  const magic = dec.decode(magicBytes);
  if (magic !== MAGIC) throw new Error(`bad magic: ${JSON.stringify(magic)}`);
  off += 8;
  const version = dv.getUint32(off, true); off += 4;
  if (version !== 1) throw new Error(`unsupported version ${version}`);
  const num_snaps = dv.getUint32(off, true); off += 4;
  const snap_n = dv.getUint32(off, true); off += 4;
  const n_fibers = dv.getUint32(off, true); off += 4;
  const n_bp_per = dv.getUint32(off, true); off += 4;
  const energy_eV = dv.getFloat32(off, true); off += 4;

  // DNA geometry
  const n_fibers2 = dv.getUint32(off, true); off += 4;
  if (n_fibers2 !== n_fibers) throw new Error('dna.n_fibers mismatch');
  const fy = new Float32Array(buf, off, n_fibers); off += n_fibers * 4;
  const fz = new Float32Array(buf, off, n_fibers); off += n_fibers * 4;
  const rise = dv.getFloat32(off, true); off += 4;
  const spacing_nm = dv.getFloat32(off, true); off += 4;
  const L_nm = dv.getFloat32(off, true); off += 4;
  const x0 = dv.getFloat32(off, true); off += 4;

  // Initial rad_buf
  const rad_n_initial = dv.getUint32(off, true); off += 4;
  const rad_buf = new Float32Array(buf, off, rad_n_initial * 4); off += rad_n_initial * 16;

  // Snapshots — labels have variable length, so pos/alive arrays may start at
  // a non-4-aligned offset within the file. Float32Array/Uint32Array views
  // require multiple-of-4 byte offsets into the source ArrayBuffer, so we
  // slice() each typed-array region into a fresh ArrayBuffer (guaranteed
  // aligned at offset 0). Costs a copy per checkpoint but is robust.
  const snapshots: Snapshot[] = [];
  for (let i = 0; i < num_snaps; i++) {
    const t_ns = dv.getFloat32(off, true); off += 4;
    const label_len = dv.getUint32(off, true); off += 4;
    const labelBytes = new Uint8Array(buf, off, label_len);
    const label = dec.decode(labelBytes); off += label_len;
    const pos = new Float32Array(buf.slice(off, off + snap_n * 16));
    off += snap_n * 16;
    const alive = new Uint32Array(buf.slice(off, off + snap_n * 4));
    off += snap_n * 4;
    snapshots.push({ label, t_ns, pos, alive });
  }

  return {
    version, num_snaps, snap_n, energy_eV,
    dna: { n_fibers, n_bp_per, fy, fz, rise, spacing_nm, L_nm, x0 },
    rad0: { rad_n_initial, rad_buf },
    snapshots,
  };
}

// =============================================================================
// WGSL — instanced billboarded sprites coloured by species
// =============================================================================

const RENDER_SHADER = /* wgsl */ `
  struct Uniforms {
    view_proj: mat4x4<f32>,    // 64 B
    species_scale: vec4<f32>,  // 16 B  per-species point size scale
    // 16 B  scalar pack:
    t_frac: f32,               // 0..1 interpolation factor between snapshot a and b
    species_mask: u32,         // bit i = species i visible (0..7)
    dna_x0: f32,               // DNA fiber start in nm
    dna_L: f32,                // DNA fiber length in nm
    extras: vec4<f32>,         // x=sprite_intensity, y=hit_intensity, z=slice_axis(int 0..3), w=slice_pos(0..1)
    bbox_min: vec4<f32>,       // 16 B
    bbox_span: vec4<f32>,      // 16 B (bbox_max - bbox_min)
  }

  @group(0) @binding(0) var<uniform> u: Uniforms;
  @group(0) @binding(1) var<storage, read> pos_a: array<vec4<f32>>;
  @group(0) @binding(2) var<storage, read> alive_a: array<u32>;
  @group(0) @binding(3) var<storage, read> pos_b: array<vec4<f32>>;
  @group(0) @binding(4) var<storage, read> alive_b: array<u32>;

  struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) species: f32,
    @location(1) uv: vec2<f32>,
    @location(2) alive_factor: f32,
  }

  const QUAD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
  );

  @vertex
  fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let pa = pos_a[ii];
    let pb = pos_b[ii];
    let aa = alive_a[ii];
    let ab = alive_b[ii];
    let s  = u32(pa.w) % 8u;
    let q  = QUAD[vi];

    // Linearly interpolate position between snapshots. Real radicals follow
    // a random walk so this isn't physically exact, but it reads as smooth
    // diffusion at the eye's framerate.
    let xyz = mix(pa.xyz, pb.xyz, u.t_frac);

    // Alive logic — three regimes:
    //   aa=0          → already reacted before snapshot a (never visible)
    //   aa=1, ab=0    → reacts during this interval, fade alpha linearly
    //   aa=1, ab=1    → alive across the whole interval
    var alive_factor = 0.0;
    if (aa == 1u) {
      alive_factor = select(1.0 - u.t_frac, 1.0, ab == 1u);
    }

    // Species visibility mask — bit i must be set for species i to render.
    if ((u.species_mask & (1u << s)) == 0u) {
      alive_factor = 0.0;
    }

    // Slice cutaway. extras: x=sprite intensity, y=hit (ignored here),
    // z=slice_axis (cast u32, 3=off), w=slice_pos in [0,1].
    let ax = u32(u.extras.z);
    if (ax < 3u) {
      let comp = select(select(xyz.z, xyz.y, ax == 1u), xyz.x, ax == 0u);
      let bmin = select(select(u.bbox_min.z, u.bbox_min.y, ax == 1u), u.bbox_min.x, ax == 0u);
      let bspan = select(select(u.bbox_span.z, u.bbox_span.y, ax == 1u), u.bbox_span.x, ax == 0u);
      let nrm  = (comp - bmin) / max(0.0001, bspan);
      if (nrm > u.extras.w) { alive_factor = 0.0; }
    }

    let center = u.view_proj * vec4<f32>(xyz, 1.0);
    let size = 0.012 * u.species_scale[min(s, 3u)];

    var out: VOut;
    out.clip = vec4<f32>(center.xy + q * size * center.w, center.z, center.w);
    out.species = f32(s);
    out.uv = q;
    out.alive_factor = alive_factor;
    return out;
  }

  @fragment
  fn fs(in: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) { discard; }
    if (in.alive_factor < 0.01) { discard; }

    let s = i32(in.species);
    var col = vec3<f32>(1.0, 1.0, 1.0);
    switch s {
      case 0: { col = vec3<f32>(0.30, 0.85, 1.00); }
      case 1: { col = vec3<f32>(1.00, 0.30, 0.90); }
      case 2: { col = vec3<f32>(1.00, 0.90, 0.30); }
      case 3: { col = vec3<f32>(1.00, 0.55, 0.20); }
      case 5: { col = vec3<f32>(0.65, 0.35, 1.00); }
      case 6: { col = vec3<f32>(0.30, 1.00, 0.55); }
      default: { col = vec3<f32>(0.85, 0.85, 0.95); }
    }

    // Implicit-sphere shading: treat the disc as the projection of a sphere
    // and reconstruct its surface normal. r=0 is the front-facing pole;
    // r=1 is the silhouette. Combines a soft lambert fill with a rim glow.
    let z = sqrt(max(0.0, 1.0 - r2));
    let lam = 0.32 + 0.68 * z;            // never fully dark — particles are emissive
    let rim = pow(1.0 - z, 4.0) * 1.4;    // bright halo at the silhouette
    let core = exp(-r2 * 1.6);            // hot core that punches into bloom
    let energy = (lam + rim + core) * 0.45;
    let alpha = energy * in.alive_factor * u.extras.x;
    return vec4<f32>(col * alpha, alpha);
  }
`;

// =============================================================================
// WGSL — DNA fiber lines (one line-list segment per fiber along X axis)
// =============================================================================

const LINE_SHADER = /* wgsl */ `
  struct Uniforms {
    view_proj: mat4x4<f32>,
    species_scale: vec4<f32>,
    t_frac: f32,
    species_mask: u32,
    dna_x0: f32,
    dna_L: f32,
    extras: vec4<f32>,         // x=sprite intensity (unused here), y=hit_intensity_global
    bbox_min: vec4<f32>,
    bbox_span: vec4<f32>,
  }

  @group(0) @binding(0) var<uniform> u: Uniforms;
  @group(0) @binding(1) var<storage, read> fy: array<f32>;
  @group(0) @binding(2) var<storage, read> fz: array<f32>;
  @group(0) @binding(3) var<storage, read> hit_a: array<f32>;
  @group(0) @binding(4) var<storage, read> hit_b: array<f32>;

  struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) along: f32,
    @location(1) hit: f32,
  }

  @vertex
  fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let along = select(0.0, 1.0, vi == 1u);
    let pos = vec3<f32>(u.dna_x0 + along * u.dna_L, fy[ii], fz[ii]);
    var out: VOut;
    out.clip = u.view_proj * vec4<f32>(pos, 1.0);
    out.along = along;
    out.hit = mix(hit_a[ii], hit_b[ii], u.t_frac) * u.extras.y;
    return out;
  }

  @fragment
  fn fs(in: VOut) -> @location(0) vec4<f32> {
    // Soft taper toward fiber ends so they don't visually clip into nothing.
    let edge = 1.0 - smoothstep(0.85, 1.0, abs(in.along * 2.0 - 1.0));
    // Base steel-blue + cyan flare proportional to local hit intensity.
    let baseAlpha = 0.18 * edge;
    let baseCol = vec3<f32>(0.45, 0.65, 0.85);
    let hot = clamp(in.hit, 0.0, 1.0);
    let hitCol = vec3<f32>(0.30, 1.10, 1.40);
    let alpha = baseAlpha * (1.0 + hot * 5.0);
    let col = mix(baseCol, hitCol, hot);
    return vec4<f32>(col * alpha, alpha);
  }
`;

// =============================================================================
// WGSL — Track footprint: render rad0 (initial radical positions) as bright dots.
// These mark where the electron actually ionized water along its path; not the
// path itself, but a 1-to-1 footprint of where chemistry began.
// =============================================================================

const TRACK_SHADER = /* wgsl */ `
  struct U {
    view_proj: mat4x4<f32>,
  }

  @group(0) @binding(0) var<uniform> u: U;
  @group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;

  struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
  }

  const QUAD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
  );

  @vertex
  fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let p = pos[ii].xyz;
    let q = QUAD[vi];
    let center = u.view_proj * vec4<f32>(p, 1.0);
    let size = 0.0035;       // small bright dots
    var out: VOut;
    out.clip = vec4<f32>(center.xy + q * size * center.w, center.z, center.w);
    out.uv = q;
    return out;
  }

  @fragment
  fn fs(in: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) { discard; }
    let core = pow(1.0 - r2, 2.0);
    let alpha = core * 0.85;
    // Warm white — reads as energy-deposition spark, distinct from radical species.
    let col = vec3<f32>(1.00, 0.92, 0.62);
    return vec4<f32>(col * alpha, alpha);
  }
`;

// =============================================================================
// WGSL — Reinhard tonemap pass: read HDR float texture, compress, write swapchain
// =============================================================================

// Bloom horizontal: reads full-res HDR with downsampled tap pattern (2× source
// stride per tap), extracts bright-pixel residual via soft knee, blurs along x.
const BLOOM_H_SHADER = /* wgsl */ `
  @group(0) @binding(0) var hdr: texture_2d<f32>;

  @vertex
  fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
      vec2<f32>(-1.0, -1.0), vec2<f32>( 3.0, -1.0), vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pts[vi], 0.0, 1.0);
  }

  @fragment
  fn fs(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Dst is half-res. Src pixel = dst pixel × 2.
    let basePix = vec2<i32>(i32(coord.x * 2.0), i32(coord.y * 2.0));
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    var sum = textureLoad(hdr, basePix, 0).rgb * weights[0];
    for (var i: i32 = 1; i < 5; i = i + 1) {
      let r = i * 4; // wider stride for chunkier bloom radius
      sum = sum + textureLoad(hdr, basePix + vec2<i32>(r, 0), 0).rgb * weights[i];
      sum = sum + textureLoad(hdr, basePix - vec2<i32>(r, 0), 0).rgb * weights[i];
    }
    // Soft knee: only the bright part above ~mid-grey contributes.
    let bright = max(sum - vec3<f32>(0.55), vec3<f32>(0.0));
    return vec4<f32>(bright, 1.0);
  }
`;

const BLOOM_V_SHADER = /* wgsl */ `
  @group(0) @binding(0) var src: texture_2d<f32>;

  @vertex
  fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
      vec2<f32>(-1.0, -1.0), vec2<f32>( 3.0, -1.0), vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pts[vi], 0.0, 1.0);
  }

  @fragment
  fn fs(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(i32(coord.x), i32(coord.y));
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    var sum = textureLoad(src, pix, 0).rgb * weights[0];
    for (var i: i32 = 1; i < 5; i = i + 1) {
      let r = i * 2;
      sum = sum + textureLoad(src, pix + vec2<i32>(0, r), 0).rgb * weights[i];
      sum = sum + textureLoad(src, pix - vec2<i32>(0, r), 0).rgb * weights[i];
    }
    return vec4<f32>(sum, 1.0);
  }
`;

const TONEMAP_SHADER = /* wgsl */ `
  struct TU { vp: vec4<f32>, fx: vec4<f32> }
  // vp: x=W, y=H, z=bloom_intensity, w=time_seconds
  // fx: x=vignette (0..1), y=grain (0..1), z=cinematic_mode_flag, w=_

  @group(0) @binding(0) var hdr: texture_2d<f32>;
  @group(0) @binding(1) var bloom: texture_2d<f32>;
  @group(0) @binding(2) var bloomSamp: sampler;
  @group(0) @binding(3) var<uniform> u: TU;

  @vertex
  fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
      vec2<f32>(-1.0, -1.0), vec2<f32>( 3.0, -1.0), vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pts[vi], 0.0, 1.0);
  }

  // ACES Filmic curve (Krzysztof Narkowicz's fitted approximation).
  // Drop-in replacement for Reinhard with much richer highlight rolloff.
  fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = vec3<f32>(2.51);
    let b = vec3<f32>(0.03);
    let c = vec3<f32>(2.43);
    let d = vec3<f32>(0.59);
    let e = vec3<f32>(0.14);
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
  }

  // 1D hash for grain (no texture sampling needed).
  fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }

  @fragment
  fn fs(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(i32(coord.x), i32(coord.y));
    let raw = textureLoad(hdr, pix, 0).rgb;
    let uv = coord.xy / u.vp.xy;
    let bl = textureSample(bloom, bloomSamp, uv).rgb;
    var col = raw + bl * u.vp.z;

    // Slight pre-exposure boost — ACES handles highlights well, so we can push.
    col = col * 1.1;

    // ACES filmic tonemap. Output is in [0,1] gamma-2.2-ish space already.
    col = aces(col);

    // Vignette: smooth radial falloff. Stronger in cinematic mode.
    let r = distance(uv, vec2<f32>(0.5)) * 1.4;
    let vignette_strength = u.fx.x * select(1.0, 1.6, u.fx.z > 0.5);
    let vignette = mix(1.0, smoothstep(1.05, 0.35, r), clamp(vignette_strength, 0.0, 1.0));
    col = col * vignette;

    // Film grain: per-pixel hash-based noise, animated by time.
    let g = hash12(coord.xy + vec2<f32>(u.vp.w * 60.0, u.vp.w * 91.0)) - 0.5;
    col = col + vec3<f32>(g * u.fx.y);

    // Final gamma lift so deep shadows aren't crushed.
    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 1.05));

    return vec4<f32>(col, 1.0);
  }
`;

// =============================================================================
// WGSL — Volume mode: clear voxels, splat radicals, ray-march density field
// =============================================================================

const DEFAULT_VOXEL_DIM = 128;
const MAX_VOXEL_DIM = 256;

const CLEAR_SHADER = /* wgsl */ `
  // total = number of u32 entries to clear; stride_x = grid X dim × 64 (since
  // workgroup_size is 64), used to recover linear index from a 2D dispatch.
  struct ClearU { total: u32, stride_x: u32, _p1: u32, _p2: u32 }
  @group(0) @binding(0) var<storage, read_write> voxels: array<atomic<u32>>;
  @group(0) @binding(1) var<uniform> u: ClearU;

  @compute @workgroup_size(64)
  fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * u.stride_x;
    if (i >= u.total) { return; }
    atomicStore(&voxels[i], 0u);
  }
`;

const SPLAT_SHADER = /* wgsl */ `
  struct SplatU {
    bbox_min: vec4<f32>,
    bbox_max: vec4<f32>,
    ipack:    vec4<u32>,   // n, dim, species_mask, stride_x
    fpack:    vec4<f32>,   // t_frac, _, _, _
  }

  @group(0) @binding(0) var<uniform> u: SplatU;
  @group(0) @binding(1) var<storage, read>       pos_a:   array<vec4<f32>>;
  @group(0) @binding(2) var<storage, read>       alive_a: array<u32>;
  @group(0) @binding(3) var<storage, read>       pos_b:   array<vec4<f32>>;
  @group(0) @binding(4) var<storage, read>       alive_b: array<u32>;
  @group(0) @binding(5) var<storage, read_write> voxels:  array<atomic<u32>>;

  // Each radical contributes a TOTAL weight of 256 (chosen so trilinear
  // fractions round to nicely-distributed u32 atomic increments) spread
  // across the 8 surrounding voxels via trilinear splat. Result is a
  // smooth density field rather than 1-voxel speckle. The raymarch divides
  // out the 256 scale to recover "radicals per voxel" units.
  @compute @workgroup_size(64)
  fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * u.ipack.w;
    let n = u.ipack.x;
    if (i >= n) { return; }
    let aa = alive_a[i];
    if (aa == 0u) { return; }
    let ab = alive_b[i];

    let pa = pos_a[i];
    let pb = pos_b[i];
    let s  = u32(pa.w) % 8u;
    let mask = u.ipack.z;
    if ((mask & (1u << s)) == 0u) { return; }

    let xyz = mix(pa.xyz, pb.xyz, u.fpack.x);
    let span = u.bbox_max.xyz - u.bbox_min.xyz;
    let nrm = (xyz - u.bbox_min.xyz) / span;
    if (nrm.x < 0.0 || nrm.x > 1.0 || nrm.y < 0.0 || nrm.y > 1.0 || nrm.z < 0.0 || nrm.z > 1.0) {
      return;
    }

    let dim = u.ipack.y;
    let dimI = i32(dim);
    let center = nrm * f32(dim) - vec3<f32>(0.5);
    let base = vec3<i32>(floor(center));
    let f = clamp(center - vec3<f32>(base), vec3<f32>(0.0), vec3<f32>(1.0));

    // Slight emphasis on alive-throughout vs. reacts-mid: full = 256, partial = 192.
    let total: f32 = select(192.0, 256.0, ab == 1u);

    for (var dz: i32 = 0; dz < 2; dz = dz + 1) {
      for (var dy: i32 = 0; dy < 2; dy = dy + 1) {
        for (var dx: i32 = 0; dx < 2; dx = dx + 1) {
          let cx = base.x + dx;
          let cy = base.y + dy;
          let cz = base.z + dz;
          if (cx < 0 || cy < 0 || cz < 0 || cx >= dimI || cy >= dimI || cz >= dimI) { continue; }
          let w = mix(1.0 - f.x, f.x, f32(dx)) *
                  mix(1.0 - f.y, f.y, f32(dy)) *
                  mix(1.0 - f.z, f.z, f32(dz));
          let amt = u32(w * total + 0.5);
          if (amt == 0u) { continue; }
          let idx = u32(cx) + u32(cy) * dim + u32(cz) * dim * dim;
          atomicAdd(&voxels[idx], amt);
        }
      }
    }
  }
`;

// Species-splat: 4 atomic channels per voxel (OH / eaq / H / H3O+ at indices 0..3).
// Other species (pre-eaq, OH-, H2) are dropped — they're noise relative to the
// primary 4 in the published Karamitros 2011 spur chemistry.
const SPLAT_SP_SHADER = /* wgsl */ `
  struct SplatU {
    bbox_min: vec4<f32>,
    bbox_max: vec4<f32>,
    ipack:    vec4<u32>,
    fpack:    vec4<f32>,
  }

  @group(0) @binding(0) var<uniform> u: SplatU;
  @group(0) @binding(1) var<storage, read>       pos_a:   array<vec4<f32>>;
  @group(0) @binding(2) var<storage, read>       alive_a: array<u32>;
  @group(0) @binding(3) var<storage, read>       pos_b:   array<vec4<f32>>;
  @group(0) @binding(4) var<storage, read>       alive_b: array<u32>;
  @group(0) @binding(5) var<storage, read_write> voxels:  array<atomic<u32>>;

  @compute @workgroup_size(64)
  fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * u.ipack.w;
    let n = u.ipack.x;
    if (i >= n) { return; }
    let aa = alive_a[i];
    if (aa == 0u) { return; }
    let ab = alive_b[i];

    let pa = pos_a[i];
    let pb = pos_b[i];
    let s  = u32(pa.w) % 8u;
    if (s >= 4u) { return; }
    let mask = u.ipack.z;
    if ((mask & (1u << s)) == 0u) { return; }

    let xyz = mix(pa.xyz, pb.xyz, u.fpack.x);
    let span = u.bbox_max.xyz - u.bbox_min.xyz;
    let nrm = (xyz - u.bbox_min.xyz) / span;
    if (nrm.x < 0.0 || nrm.x > 1.0 || nrm.y < 0.0 || nrm.y > 1.0 || nrm.z < 0.0 || nrm.z > 1.0) { return; }

    let dim = u.ipack.y;
    let dimI = i32(dim);
    let center = nrm * f32(dim) - vec3<f32>(0.5);
    let base = vec3<i32>(floor(center));
    let f = clamp(center - vec3<f32>(base), vec3<f32>(0.0), vec3<f32>(1.0));
    let total: f32 = select(192.0, 256.0, ab == 1u);

    for (var dz: i32 = 0; dz < 2; dz = dz + 1) {
      for (var dy: i32 = 0; dy < 2; dy = dy + 1) {
        for (var dx: i32 = 0; dx < 2; dx = dx + 1) {
          let cx = base.x + dx;
          let cy = base.y + dy;
          let cz = base.z + dz;
          if (cx < 0 || cy < 0 || cz < 0 || cx >= dimI || cy >= dimI || cz >= dimI) { continue; }
          let w = mix(1.0 - f.x, f.x, f32(dx)) *
                  mix(1.0 - f.y, f.y, f32(dy)) *
                  mix(1.0 - f.z, f.z, f32(dz));
          let amt = u32(w * total + 0.5);
          if (amt == 0u) { continue; }
          let voxIdx = u32(cx) + u32(cy) * dim + u32(cz) * dim * dim;
          atomicAdd(&voxels[voxIdx * 4u + s], amt);
        }
      }
    }
  }
`;

const VOLUME_SP_SHADER = /* wgsl */ `
  struct VolU {
    inv_vp:   mat4x4<f32>,
    cam_eye:  vec4<f32>,
    bbox_min: vec4<f32>,
    bbox_max: vec4<f32>,
    fpack:    vec4<f32>,
    ipack:    vec4<u32>,
    spack:    vec4<f32>,        // matched layout with VOLUME_SHADER
  }

  @group(0) @binding(0) var<uniform> u: VolU;
  @group(0) @binding(1) var<storage, read> voxels: array<u32>;

  @vertex
  fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
      vec2<f32>(-1.0, -1.0), vec2<f32>( 3.0, -1.0), vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pts[vi], 0.0, 1.0);
  }

  fn slice_cull_sp(p: vec3<f32>) -> bool {
    let ax = u.ipack.z;
    if (ax >= 3u) { return false; }
    let comp = select(select(p.z, p.y, ax == 1u), p.x, ax == 0u);
    let bmin = select(select(u.bbox_min.z, u.bbox_min.y, ax == 1u), u.bbox_min.x, ax == 0u);
    let bspn = select(select(u.bbox_max.z - u.bbox_min.z, u.bbox_max.y - u.bbox_min.y, ax == 1u),
                      u.bbox_max.x - u.bbox_min.x, ax == 0u);
    let nrm = (comp - bmin) / max(0.0001, bspn);
    return nrm > u.spack.x;
  }

  fn sample_4ch(p: vec3<f32>) -> vec4<f32> {
    let span = u.bbox_max.xyz - u.bbox_min.xyz;
    let nrm = (p - u.bbox_min.xyz) / span;
    if (nrm.x < 0.0 || nrm.x > 1.0 || nrm.y < 0.0 || nrm.y > 1.0 || nrm.z < 0.0 || nrm.z > 1.0) {
      return vec4<f32>(0.0);
    }
    if (slice_cull_sp(p)) { return vec4<f32>(0.0); }
    let dim = u.ipack.x;
    let scaled = nrm * f32(dim) - vec3<f32>(0.5);
    let base = clamp(vec3<i32>(floor(scaled)), vec3<i32>(0), vec3<i32>(i32(dim) - 2));
    let f = clamp(scaled - vec3<f32>(base), vec3<f32>(0.0), vec3<f32>(1.0));

    var acc = vec4<f32>(0.0);
    for (var dz: i32 = 0; dz < 2; dz = dz + 1) {
      for (var dy: i32 = 0; dy < 2; dy = dy + 1) {
        for (var dx: i32 = 0; dx < 2; dx = dx + 1) {
          let c = vec3<u32>(u32(base.x + dx), u32(base.y + dy), u32(base.z + dz));
          let voxIdx = c.x + c.y * dim + c.z * dim * dim;
          let i0 = voxIdx * 4u;
          let w = mix(1.0 - f.x, f.x, f32(dx)) *
                  mix(1.0 - f.y, f.y, f32(dy)) *
                  mix(1.0 - f.z, f.z, f32(dz));
          acc.x = acc.x + f32(voxels[i0 + 0u]) * w;
          acc.y = acc.y + f32(voxels[i0 + 1u]) * w;
          acc.z = acc.z + f32(voxels[i0 + 2u]) * w;
          acc.w = acc.w + f32(voxels[i0 + 3u]) * w;
        }
      }
    }
    return acc;
  }

  @fragment
  fn fs(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let viewport = u.fpack.xy;
    let ndc = vec2<f32>(
      (coord.x / viewport.x) * 2.0 - 1.0,
      1.0 - (coord.y / viewport.y) * 2.0,
    );
    let near_h = u.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let far_h  = u.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    let near_w = near_h.xyz / near_h.w;
    let far_w  = far_h.xyz  / far_h.w;
    let dir = normalize(far_w - near_w);
    let origin = u.cam_eye.xyz;

    let inv_dir = vec3<f32>(1.0) / dir;
    let t1 = (u.bbox_min.xyz - origin) * inv_dir;
    let t2 = (u.bbox_max.xyz - origin) * inv_dir;
    let tmin3 = min(t1, t2);
    let tmax3 = max(t1, t2);
    let t_near = max(0.0, max(max(tmin3.x, tmin3.y), tmin3.z));
    let t_far  = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_near >= t_far) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }

    let steps: i32 = 128;
    let dt = (t_far - t_near) / f32(steps);
    var t = t_near + dt * 0.5;
    var emission = vec3<f32>(0.0);
    var transmittance: f32 = 1.0;
    let density_scale = u.fpack.z;
    let exposure = u.fpack.w;

    let cOH  = vec3<f32>(0.30, 0.85, 1.00);
    let cEAQ = vec3<f32>(1.00, 0.30, 0.90);
    let cH   = vec3<f32>(1.00, 0.90, 0.30);
    let cH3O = vec3<f32>(1.00, 0.55, 0.20);

    for (var s: i32 = 0; s < steps; s = s + 1) {
      let p = origin + dir * t;
      let raw4 = sample_4ch(p) / 256.0;
      let total_d = raw4.x + raw4.y + raw4.z + raw4.w;
      if (total_d > 0.0) {
        let density = total_d * density_scale;
        let absorption = 1.0 - exp(-density * dt);
        let color = (cOH * raw4.x + cEAQ * raw4.y + cH * raw4.z + cH3O * raw4.w) / max(total_d, 1e-6);
        emission = emission + transmittance * absorption * color * exposure;
        transmittance = transmittance * (1.0 - absorption);
        if (transmittance < 0.005) { break; }
      }
      t = t + dt;
    }
    return vec4<f32>(emission, 1.0 - transmittance);
  }
`;

const VOLUME_SHADER = /* wgsl */ `
  struct VolU {
    inv_vp:   mat4x4<f32>,
    cam_eye:  vec4<f32>,
    bbox_min: vec4<f32>,
    bbox_max: vec4<f32>,
    fpack:    vec4<f32>,   // viewport.x, viewport.y, density_scale, exposure
    ipack:    vec4<u32>,   // dim, mode (0=density, 1=iso), slice_axis(0..3), _
    spack:    vec4<f32>,   // slice_pos (0..1), _, _, _
  }

  @group(0) @binding(0) var<uniform> u: VolU;
  @group(0) @binding(1) var<storage, read> voxels: array<u32>;

  @vertex
  fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
      vec2<f32>(-1.0, -1.0), vec2<f32>( 3.0, -1.0), vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pts[vi], 0.0, 1.0);
  }

  // Slice cutaway test: returns true if this point should be skipped.
  fn slice_cull(p: vec3<f32>) -> bool {
    let ax = u.ipack.z;
    if (ax >= 3u) { return false; }
    let comp = select(select(p.z, p.y, ax == 1u), p.x, ax == 0u);
    let bmin = select(select(u.bbox_min.z, u.bbox_min.y, ax == 1u), u.bbox_min.x, ax == 0u);
    let bspn = select(select(u.bbox_max.z - u.bbox_min.z, u.bbox_max.y - u.bbox_min.y, ax == 1u),
                      u.bbox_max.x - u.bbox_min.x, ax == 0u);
    let nrm = (comp - bmin) / max(0.0001, bspn);
    return nrm > u.spack.x;
  }

  fn sample_density(p: vec3<f32>) -> f32 {
    let span = u.bbox_max.xyz - u.bbox_min.xyz;
    let nrm = (p - u.bbox_min.xyz) / span;
    if (nrm.x < 0.0 || nrm.x > 1.0 || nrm.y < 0.0 || nrm.y > 1.0 || nrm.z < 0.0 || nrm.z > 1.0) {
      return 0.0;
    }
    if (slice_cull(p)) { return 0.0; }
    let dim = u.ipack.x;
    let scaled = nrm * f32(dim) - vec3<f32>(0.5);
    let base = clamp(vec3<i32>(floor(scaled)), vec3<i32>(0), vec3<i32>(i32(dim) - 2));
    let f = clamp(scaled - vec3<f32>(base), vec3<f32>(0.0), vec3<f32>(1.0));

    // Trilinear interpolation across 8 voxel corners.
    var acc: f32 = 0.0;
    for (var dz: i32 = 0; dz < 2; dz = dz + 1) {
      for (var dy: i32 = 0; dy < 2; dy = dy + 1) {
        for (var dx: i32 = 0; dx < 2; dx = dx + 1) {
          let c = vec3<u32>(u32(base.x + dx), u32(base.y + dy), u32(base.z + dz));
          let idx = c.x + c.y * dim + c.z * dim * dim;
          let w = mix(1.0 - f.x, f.x, f32(dx)) *
                  mix(1.0 - f.y, f.y, f32(dy)) *
                  mix(1.0 - f.z, f.z, f32(dz));
          acc += f32(voxels[idx]) * w;
        }
      }
    }
    return acc;
  }

  @fragment
  fn fs(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let viewport = u.fpack.xy;
    let ndc = vec2<f32>(
      (coord.x / viewport.x) * 2.0 - 1.0,
      1.0 - (coord.y / viewport.y) * 2.0,
    );
    // Unproject near and far points.
    let near_h = u.inv_vp * vec4<f32>(ndc, 0.0, 1.0);
    let far_h  = u.inv_vp * vec4<f32>(ndc, 1.0, 1.0);
    let near_w = near_h.xyz / near_h.w;
    let far_w  = far_h.xyz  / far_h.w;
    let dir = normalize(far_w - near_w);
    let origin = u.cam_eye.xyz;

    // Slab AABB intersection.
    let inv_dir = vec3<f32>(1.0) / dir;
    let t1 = (u.bbox_min.xyz - origin) * inv_dir;
    let t2 = (u.bbox_max.xyz - origin) * inv_dir;
    let tmin3 = min(t1, t2);
    let tmax3 = max(t1, t2);
    let t_near = max(0.0, max(max(tmin3.x, tmin3.y), tmin3.z));
    let t_far  = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_near >= t_far) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }

    let steps: i32 = 128;
    let dt = (t_far - t_near) / f32(steps);
    var t = t_near + dt * 0.5;
    var emission = vec3<f32>(0.0);
    var transmittance: f32 = 1.0;
    let density_scale = u.fpack.z;
    let exposure = u.fpack.w;
    let mode = u.ipack.y;

    // Iso thresholds (raw radicals/voxel, before scale): low / mid / hot shells.
    let iso0 = 0.6;
    let iso1 = 1.6;
    let iso2 = 4.0;
    var prev_raw: f32 = 0.0;

    for (var s: i32 = 0; s < steps; s = s + 1) {
      let p = origin + dir * t;
      let raw = sample_density(p) / 256.0;   // splat scale → radicals/voxel
      if (mode == 1u) {
        // Iso-surface mode: emit at threshold crossings.
        if (raw > 0.0) {
          var shell_em = vec3<f32>(0.0);
          var shell_a = 0.0;
          if ((prev_raw < iso0 && raw >= iso0) || (prev_raw >= iso0 && raw < iso0)) {
            shell_em += vec3<f32>(0.30, 0.55, 1.10) * exposure * 0.7;
            shell_a += 0.18;
          }
          if ((prev_raw < iso1 && raw >= iso1) || (prev_raw >= iso1 && raw < iso1)) {
            shell_em += vec3<f32>(0.85, 0.80, 1.00) * exposure * 0.95;
            shell_a += 0.30;
          }
          if ((prev_raw < iso2 && raw >= iso2) || (prev_raw >= iso2 && raw < iso2)) {
            shell_em += vec3<f32>(1.30, 0.80, 0.45) * exposure * 1.2;
            shell_a += 0.50;
          }
          if (shell_a > 0.0) {
            emission += transmittance * shell_em;
            transmittance *= 1.0 - shell_a;
          }
        }
        prev_raw = raw;
      } else if (raw > 0.0) {
        let density = raw * density_scale;
        let absorption = 1.0 - exp(-density * dt);
        // Cool-to-hot ramp on effective radical count per voxel.
        let ramp = clamp(log(raw * 4.0 + 1.0) * 0.45, 0.0, 1.0);
        let cool = vec3<f32>(0.30, 0.55, 1.10);
        let mid  = vec3<f32>(0.70, 0.80, 1.00);
        let hot  = vec3<f32>(1.20, 0.85, 0.55);
        var color: vec3<f32>;
        if (ramp < 0.5) {
          color = mix(cool, mid, ramp * 2.0);
        } else {
          color = mix(mid, hot, (ramp - 0.5) * 2.0);
        }
        emission += transmittance * absorption * color * exposure;
        transmittance *= 1.0 - absorption;
        if (transmittance < 0.005) { break; }
      }
      t += dt;
    }
    return vec4<f32>(emission, 1.0 - transmittance);
  }
`;

// =============================================================================
// WGSL — TRACK_LINE: a fading polyline through the initial radical positions,
// sorted by x so the line traces the primary electron's general direction
// across the box. A `progress` uniform makes it draw on during the cinematic
// intro (0→1), then sit at full alpha but low opacity afterward.
// =============================================================================

const TRACK_LINE_SHADER = /* wgsl */ `
  struct U {
    view_proj: mat4x4<f32>,
    pack:      vec4<f32>,    // x = progress (0..1), y = base_alpha
  }

  @group(0) @binding(0) var<uniform> u: U;
  @group(0) @binding(1) var<storage, read> pts: array<vec4<f32>>;

  struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) frac: f32,        // 0..1 along the polyline
    @location(1) revealed: f32,    // 1 if the draw-on cursor has reached this vertex
  }

  @vertex
  fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let n = arrayLength(&pts);
    let total_segments = max(1u, n - 1u);
    let idx = ii + vi;
    let frac = f32(idx) / f32(total_segments);
    let p = pts[min(idx, n - 1u)].xyz;
    var out: VOut;
    out.clip = u.view_proj * vec4<f32>(p, 1.0);
    out.frac = frac;
    out.revealed = select(0.0, 1.0, frac <= u.pack.x);
    return out;
  }

  @fragment
  fn fs(in: VOut) -> @location(0) vec4<f32> {
    if (in.revealed < 0.5) { discard; }
    // Bright leading edge, fading tail behind the cursor.
    let dist_from_head = max(0.0, u.pack.x - in.frac);
    let head_glow = exp(-dist_from_head * 8.0);
    let alpha = u.pack.y * (0.35 + 0.65 * head_glow);
    let col = vec3<f32>(1.10, 0.92, 0.55);    // warm ionization-track white
    return vec4<f32>(col * alpha, alpha);
  }
`;

// =============================================================================
// WGSL — FADE: copy previous-frame HDR into current target with multiplicative
// fade, so moving particles leave brief trails. One uniform: fade factor.
// =============================================================================

const FADE_SHADER = /* wgsl */ `
  struct U { fade: vec4<f32> }   // fade.x = multiplier in [0,1]
  @group(0) @binding(0) var src: texture_2d<f32>;
  @group(0) @binding(1) var<uniform> u: U;

  @vertex
  fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pts = array<vec2<f32>, 3>(
      vec2<f32>(-1.0, -1.0), vec2<f32>( 3.0, -1.0), vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pts[vi], 0.0, 1.0);
  }

  @fragment
  fn fs(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(coord.xy), 0);
    return vec4<f32>(c.rgb * u.fade.x, c.a * u.fade.x);
  }
`;

// =============================================================================
// WGSL — SPARK: reaction events. Each spark is a position + birth time;
// renders as an expanding cyan ring that fades over one checkpoint interval.
// =============================================================================

const SPARK_SHADER = /* wgsl */ `
  struct U {
    view_proj: mat4x4<f32>,
    pack:      vec4<f32>,    // x = current state.t, y = lifetime (in t-units), z = base_size_nm, w = unused
    slice:     vec4<f32>,    // x = axis (cast to int), y = pos (0..1), z = bbox_span_axis, w = bbox_min_axis
  }

  @group(0) @binding(0) var<uniform> u: U;
  @group(0) @binding(1) var<storage, read> sparks: array<vec4<f32>>;   // xyz + birth_t

  struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) age: f32,
  }

  const QUAD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
  );

  @vertex
  fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let s   = sparks[ii];
    let pos = s.xyz;
    let bt  = s.w;
    let age = (u.pack.x - bt) / max(0.0001, u.pack.y);

    var out: VOut;
    if (age < 0.0 || age > 1.0) {
      // Outside lifetime — collapse vertex to clip-space null so it discards.
      out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
      out.uv = vec2<f32>(0.0, 0.0);
      out.age = 1.0;
      return out;
    }

    // Optional slice cutaway.
    let ax = i32(u.slice.x);
    if (ax >= 0 && ax <= 2) {
      let comp = select(select(pos.z, pos.y, ax == 1), pos.x, ax == 0);
      let nrm  = (comp - u.slice.w) / max(0.0001, u.slice.z);
      if (nrm > u.slice.y) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = vec2<f32>(0.0, 0.0);
        out.age = 1.0;
        return out;
      }
    }

    let q = QUAD[vi];
    let center = u.view_proj * vec4<f32>(pos, 1.0);
    // Ring grows over its lifetime — sonar-ping look.
    let size = u.pack.z * (0.4 + age * 1.6);
    out.clip = vec4<f32>(center.xy + q * size * center.w, center.z, center.w);
    out.uv = q;
    out.age = age;
    return out;
  }

  @fragment
  fn fs(in: VOut) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    // Discard outside the disc AND the inner empty hole — saves ~50 % of fragments.
    if (r > 1.0 || r < 0.42) { discard; }
    // Hollow ring: peak at r ≈ 0.72, falls off both sides.
    let ring = exp(-pow((r - 0.72) * 4.5, 2.0));
    let lifeFade = 1.0 - in.age;
    let alpha = ring * lifeFade * 0.85;
    let col = vec3<f32>(0.45, 0.95, 1.20);
    return vec4<f32>(col * alpha, alpha);
  }
`;

// =============================================================================
// Camera math (right-handed, looking down -Z, simple perspective)
// =============================================================================

interface Camera {
  yaw: number; pitch: number; radius: number;
  target: [number, number, number];
}

function viewProj(cam: Camera, aspect: number): Float32Array {
  // Camera position: orbit around target.
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const eye: [number, number, number] = [
    cam.target[0] + cam.radius * cp * sy,
    cam.target[1] + cam.radius * sp,
    cam.target[2] + cam.radius * cp * cy,
  ];
  const f = normalize(sub(cam.target, eye));
  const up: [number, number, number] = [0, 1, 0];
  const r = normalize(cross(f, up));
  const u = cross(r, f);

  // View matrix (column-major).
  const tx = -dot(r, eye), ty = -dot(u, eye), tz = dot(f, eye);
  const view = new Float32Array([
    r[0], u[0], -f[0], 0,
    r[1], u[1], -f[1], 0,
    r[2], u[2], -f[2], 0,
    tx,   ty,    tz,   1,
  ]);

  // Perspective (Vulkan-style depth 0..1).
  const fovy = Math.PI / 4;
  const near = 1.0, far = 1.0e6;
  const t = 1.0 / Math.tan(fovy / 2);
  const proj = new Float32Array([
    t / aspect, 0, 0, 0,
    0, t, 0, 0,
    0, 0, far / (near - far), -1,
    0, 0, (near * far) / (near - far), 0,
  ]);

  return mul(proj, view);
}

const sub = (a: [number,number,number], b: [number,number,number]): [number,number,number] => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const dot = (a: [number,number,number], b: [number,number,number]): number => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a: [number,number,number], b: [number,number,number]): [number,number,number] =>
  [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]];
const normalize = (v: [number,number,number]): [number,number,number] => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/l, v[1]/l, v[2]/l];
};
function mul(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
      out[i * 4 + j] = s;
    }
  }
  return out;
}

function cameraEye(cam: Camera): [number, number, number] {
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  return [
    cam.target[0] + cam.radius * cp * sy,
    cam.target[1] + cam.radius * sp,
    cam.target[2] + cam.radius * cp * cy,
  ];
}

// Standard 4×4 column-major inverse (gl-matrix algorithm). Used by the volume
// ray-march to unproject pixel → world ray each frame.
function inverse4(m: Float32Array): Float32Array {
  const a00 = m[0],  a01 = m[1],  a02 = m[2],  a03 = m[3];
  const a10 = m[4],  a11 = m[5],  a12 = m[6],  a13 = m[7];
  const a20 = m[8],  a21 = m[9],  a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return new Float32Array(m);
  const inv = 1.0 / det;
  const out = new Float32Array(16);
  out[0]  = ( a11 * b11 - a12 * b10 + a13 * b09) * inv;
  out[1]  = (-a01 * b11 + a02 * b10 - a03 * b09) * inv;
  out[2]  = ( a31 * b05 - a32 * b04 + a33 * b03) * inv;
  out[3]  = (-a21 * b05 + a22 * b04 - a23 * b03) * inv;
  out[4]  = (-a10 * b11 + a12 * b08 - a13 * b07) * inv;
  out[5]  = ( a00 * b11 - a02 * b08 + a03 * b07) * inv;
  out[6]  = (-a30 * b05 + a32 * b02 - a33 * b01) * inv;
  out[7]  = ( a20 * b05 - a22 * b02 + a23 * b01) * inv;
  out[8]  = ( a10 * b10 - a11 * b08 + a13 * b06) * inv;
  out[9]  = (-a00 * b10 + a01 * b08 - a03 * b06) * inv;
  out[10] = ( a30 * b04 - a31 * b02 + a33 * b00) * inv;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * inv;
  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * inv;
  out[13] = ( a00 * b09 - a01 * b07 + a02 * b06) * inv;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * inv;
  out[15] = ( a20 * b03 - a21 * b01 + a22 * b00) * inv;
  return out;
}

// =============================================================================
// State + bootstrapping
// =============================================================================

const canvas = document.getElementById('c') as HTMLCanvasElement;
const fileInput = document.getElementById('file') as HTMLInputElement;
const slider = document.getElementById('t-slider') as HTMLInputElement;
const tLabel = document.getElementById('t-label') as HTMLSpanElement;
const hudTimeEl = document.getElementById('hud-time') as HTMLDivElement | null;
const hudCapEl = document.getElementById('hud-cap') as HTMLDivElement | null;
const hudFpsEl = document.getElementById('hud-fps') as HTMLDivElement | null;

// FPS accumulator — averages over a 1-second window, updates the badge once
// per second. Color-codes green / yellow / red so users can see in real time
// when an effect is too costly for their GPU.
let fpsAccumMs = 0;
let fpsFrames = 0;
function updateFpsHud(dtMs: number): void {
  if (!hudFpsEl) return;
  fpsAccumMs += dtMs;
  fpsFrames++;
  if (fpsAccumMs < 1000) return;
  const fps = (fpsFrames * 1000) / fpsAccumMs;
  fpsAccumMs = 0;
  fpsFrames = 0;
  hudFpsEl.textContent = `${fps.toFixed(0)} fps`;
  hudFpsEl.className = 'hud-fps ' + (fps >= 50 ? 'ok' : fps >= 28 ? 'warn' : 'bad');
}
const metaEl = document.getElementById('meta') as HTMLDivElement;
const statusEl = document.getElementById('status') as HTMLDivElement;
const cpsEl = document.getElementById('checkpoints') as HTMLDivElement;

const ALL_SPECIES_MASK = 0xff;

type ViewMode = 'sprites' | 'volume';

const state: {
  blob: Blob4D | null;
  /** Continuous time index in [0, num_snaps - 1]. Integer part = current
   *  pair-bind-group; fractional part = interpolation factor sent to vertex
   *  shader as t_frac. */
  t: number;
  /** Bitmask: bit i set = species i visible. */
  speciesMask: number;
  cam: Camera;
  drag: { active: boolean; lastX: number; lastY: number };
  /** Auto-play state. */
  playing: boolean;
  playSpeed: number; // checkpoints per second
  /** Auto-orbit when idle. */
  lastInteractionMs: number;
  /** Initial camera radius computed at blob load (used by Reset View). */
  initialRadius: number;
  /** Render style: instanced sprites or ray-marched density volume. */
  viewMode: ViewMode;
  /** Scene bbox (nm), used by volume splat + raymarch. Computed at load. */
  bboxMin: [number, number, number];
  bboxMax: [number, number, number];
  /** Volume voxel grid resolution (cube edge). Reallocates buffer on change. */
  voxelDim: number;
  /** Volume color mode: single density ramp, per-species RGB blend, or iso shells. */
  volColor: 'density' | 'species' | 'iso';
  /** Optional post-process: HDR bloom (downsample → blur → add). */
  bloom: boolean;
  /** Optional overlay: render rad0 (initial ionization sites) as bright dots. */
  showTracks: boolean;
  /** Verification overlay: in volume mode, also draw sprites at low alpha so
   *  misalignments between the two render paths are visually obvious. */
  compareOverlay: boolean;
  /** Frame-feedback motion-blur trails. When on, each frame's HDR target
   *  starts as the previous frame's contents multiplied by trailFade. */
  trails: boolean;
  /** Render reaction-event sparks (cyan rings where radicals reacted). */
  showSparks: boolean;
  /** Render DNA fiber hit pulses (fibers brighten when radicals are near). */
  showHits: boolean;
  /** Slice-plane cutaway. axis: 0=X, 1=Y, 2=Z, 3=off. pos in [0,1] (fraction of bbox span). */
  sliceAxis: 0 | 1 | 2 | 3;
  slicePos: number;
  /** Cinematic-mode toggle: hides the panel, ramps up bloom/trails/sparks/hits. */
  cinematic: boolean;
  /** Snapshot of pre-cinematic state, so toggling C key restores prior settings. */
  cinematicStash: null | {
    bloom: boolean;
    trails: boolean;
    showSparks: boolean;
    showHits: boolean;
    playing: boolean;
    playSpeed: number;
  };
  /** Cinematic intro: slow dolly from far → home over `duration` ms after blob load. */
  intro: null | { startMs: number; duration: number; fromYaw: number; fromPitch: number; fromRadius: number; toYaw: number; toPitch: number; toRadius: number };
} = {
  blob: null,
  t: 0,
  speciesMask: ALL_SPECIES_MASK,
  cam: { yaw: 0.6, pitch: 0.3, radius: 5000, target: [0, 0, 0] },
  drag: { active: false, lastX: 0, lastY: 0 },
  playing: false,
  playSpeed: 1.0,
  lastInteractionMs: performance.now(),
  initialRadius: 5000,
  viewMode: 'sprites',
  bboxMin: [-1, -1, -1],
  bboxMax: [1, 1, 1],
  voxelDim: DEFAULT_VOXEL_DIM,
  volColor: 'density',
  bloom: false,
  showTracks: false,
  compareOverlay: false,
  trails: false,
  showSparks: false,
  showHits: true,
  sliceAxis: 3,
  slicePos: 1.0,
  cinematic: false,
  cinematicStash: null,
  intro: null,
};

/** Wall-clock origin used by tonemap-shader grain. Set once at module init. */
const bootTimeMs = performance.now();

const markInteraction = (): void => {
  state.lastInteractionMs = performance.now();
  // First user input cancels the cinematic intro so they're not fighting an animation.
  if (state.intro) state.intro = null;
};

/** Toggle cinematic mode: hide UI, bloom/trails/sparks/hits on, slow auto-play.
 *  Stash prior values on enter, restore on exit. */
function setCinematic(on: boolean): void {
  if (on === state.cinematic) return;
  state.cinematic = on;
  if (on) {
    state.cinematicStash = {
      bloom: state.bloom,
      trails: state.trails,
      showSparks: state.showSparks,
      showHits: state.showHits,
      playing: state.playing,
      playSpeed: state.playSpeed,
    };
    state.bloom = true;
    state.trails = true;
    state.showSparks = true;
    state.showHits = true;
    state.playing = true;
    state.playSpeed = 0.4;
    // Reflect on UI controls so they show the truth.
    if (bloomCb) bloomCb.checked = true;
    if (trailsCb) trailsCb.checked = true;
    if (sparksCb) sparksCb.checked = true;
    if (hitsCb) hitsCb.checked = true;
    if (playBtn) playBtn.textContent = '⏸ pause';
    if (speedInput) speedInput.value = '0.4';
    document.body.classList.add('cinematic');
    setStatus('Cinematic mode — press C again to exit.');
  } else {
    const s = state.cinematicStash;
    if (s) {
      state.bloom = s.bloom;
      state.trails = s.trails;
      state.showSparks = s.showSparks;
      state.showHits = s.showHits;
      state.playing = s.playing;
      state.playSpeed = s.playSpeed;
      if (bloomCb) bloomCb.checked = s.bloom;
      if (trailsCb) trailsCb.checked = s.trails;
      if (sparksCb) sparksCb.checked = s.showSparks;
      if (hitsCb) hitsCb.checked = s.showHits;
      if (playBtn) playBtn.textContent = s.playing ? '⏸ pause' : '▶ play';
      if (speedInput) speedInput.value = String(s.playSpeed);
    }
    state.cinematicStash = null;
    document.body.classList.remove('cinematic');
    setStatus('Exited cinematic mode.');
  }
}

/** Start a 3s cinematic intro: dolly in from far + offset yaw to the home pose. */
function startIntro(): void {
  state.intro = {
    startMs: performance.now(),
    duration: 3000,
    fromYaw: state.cam.yaw - 1.0,
    fromPitch: state.cam.pitch + 0.18,
    fromRadius: state.cam.radius * 2.4,
    toYaw: state.cam.yaw,
    toPitch: state.cam.pitch,
    toRadius: state.cam.radius,
  };
}

const setStatus = (msg: string, err = false): void => {
  statusEl.textContent = msg;
  statusEl.style.color = err ? '#ff6090' : '#5a8af0';
};

let device: GPUDevice | null = null;
let context: GPUCanvasContext | null = null;
let format: GPUTextureFormat | null = null;
let pipeline: GPURenderPipeline | null = null;
let linePipe: GPURenderPipeline | null = null;
let tonemapPipe: GPURenderPipeline | null = null;
let uniformBuf: GPUBuffer | null = null;
let snapPosBufs: GPUBuffer[] = [];
let snapAliveBufs: GPUBuffer[] = [];
let bindGroups: GPUBindGroup[] = [];

// DNA fiber buffers (allocated on blob load).
let fyBuf: GPUBuffer | null = null;
let fzBuf: GPUBuffer | null = null;
let lineBindGroup: GPUBindGroup | null = null;
let nFibers = 0;

// HDR offscreen target — recreated on canvas resize.
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
let hdrTex: GPUTexture | null = null;
let hdrView: GPUTextureView | null = null;
let tonemapBG: GPUBindGroup | null = null;
let hdrSize: { w: number; h: number } = { w: 0, h: 0 };

// Bloom (half-res ping-pong textures + linear sampler).
let bloomTexA: GPUTexture | null = null;
let bloomTexB: GPUTexture | null = null;
let bloomViewA: GPUTextureView | null = null;
let bloomViewB: GPUTextureView | null = null;
let bloomSampler: GPUSampler | null = null;
let bloomHPipe: GPURenderPipeline | null = null;
let bloomVPipe: GPURenderPipeline | null = null;
let bloomHBG: GPUBindGroup | null = null;
let bloomVBG: GPUBindGroup | null = null;
let tonemapUniBuf: GPUBuffer | null = null;

// Track footprint overlay.
let trackPipe: GPURenderPipeline | null = null;
let trackPosBuf: GPUBuffer | null = null;
let trackBG: GPUBindGroup | null = null;
let nTracks = 0;

// Volume mode (compute splat + ray-march).
let clearPipe: GPUComputePipeline | null = null;
let splatPipe: GPUComputePipeline | null = null;
let volPipe: GPURenderPipeline | null = null;
let voxelBuf: GPUBuffer | null = null;
let clearUniBuf: GPUBuffer | null = null;
let splatUniBuf: GPUBuffer | null = null;
let volUniBuf: GPUBuffer | null = null;
let clearBG: GPUBindGroup | null = null;
let splatBGs: GPUBindGroup[] = [];   // one per snapshot pair
let volBG: GPUBindGroup | null = null;

// Volume species mode (4-channel atomic counts per voxel; capped at 128³ to keep memory bounded).
const SPECIES_DIM = 128;
let splatSpPipe: GPUComputePipeline | null = null;
let volSpPipe: GPURenderPipeline | null = null;
let voxelSpBuf: GPUBuffer | null = null;
let clearSpBG: GPUBindGroup | null = null;
let splatSpBGs: GPUBindGroup[] = [];
let volSpBG: GPUBindGroup | null = null;

// Trails — frame-feedback HDR texture + fade pipeline.
let hdrFeedbackTex: GPUTexture | null = null;
let hdrFeedbackView: GPUTextureView | null = null;
let fadePipe: GPURenderPipeline | null = null;
let fadeUniBuf: GPUBuffer | null = null;
let fadeBG: GPUBindGroup | null = null;
let prevFrameValid = false;

// Reaction-event sparks (computed on snapshot load).
let sparkPipe: GPURenderPipeline | null = null;
let sparkBuf: GPUBuffer | null = null;
let sparkUniBuf: GPUBuffer | null = null;
let sparkBG: GPUBindGroup | null = null;
let nSparks = 0;
const SPARK_LIFETIME = 0.9;     // in state.t units (one checkpoint interval ≈ 1 unit)

// DNA per-fiber hit intensities (computed on snapshot load).
let hitBufs: GPUBuffer[] = [];
let lineBindGroups: GPUBindGroup[] = [];   // one per snapshot pair

// Animated electron-track line — initial rad_buf positions sorted by x.
let trackLinePipe: GPURenderPipeline | null = null;
let trackLineUniBuf: GPUBuffer | null = null;
let trackLinePosBuf: GPUBuffer | null = null;
let trackLineBG: GPUBindGroup | null = null;
let nTrackLinePts = 0;

function ensureHdrTarget(w: number, h: number): void {
  if (!device || !tonemapPipe || !bloomHPipe || !bloomVPipe || !bloomSampler || !tonemapUniBuf || !fadePipe || !fadeUniBuf) return;
  if (hdrTex && hdrSize.w === w && hdrSize.h === h) return;
  hdrTex?.destroy();
  hdrFeedbackTex?.destroy();
  bloomTexA?.destroy();
  bloomTexB?.destroy();
  prevFrameValid = false;

  hdrTex = device.createTexture({
    size: { width: w, height: h, depthOrArrayLayers: 1 },
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  hdrView = hdrTex.createView();

  // Frame-feedback texture — destination of copy at end of frame, source for FADE_SHADER at start.
  hdrFeedbackTex = device.createTexture({
    size: { width: w, height: h, depthOrArrayLayers: 1 },
    format: HDR_FORMAT,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  hdrFeedbackView = hdrFeedbackTex.createView();
  fadeBG = device.createBindGroup({
    layout: fadePipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: hdrFeedbackView },
      { binding: 1, resource: { buffer: fadeUniBuf } },
    ],
  });

  // Half-resolution bloom buffers (clamped to >= 1).
  const bw = Math.max(1, Math.floor(w / 2));
  const bh = Math.max(1, Math.floor(h / 2));
  bloomTexA = device.createTexture({
    size: { width: bw, height: bh, depthOrArrayLayers: 1 },
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  bloomTexB = device.createTexture({
    size: { width: bw, height: bh, depthOrArrayLayers: 1 },
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  bloomViewA = bloomTexA.createView();
  bloomViewB = bloomTexB.createView();

  bloomHBG = device.createBindGroup({
    layout: bloomHPipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: hdrView }],
  });
  bloomVBG = device.createBindGroup({
    layout: bloomVPipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: bloomViewA }],
  });

  tonemapBG = device.createBindGroup({
    layout: tonemapPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: hdrView },
      { binding: 1, resource: bloomViewB },
      { binding: 2, resource: bloomSampler },
      { binding: 3, resource: { buffer: tonemapUniBuf } },
    ],
  });
  hdrSize = { w, h };
}

async function initGPU(): Promise<boolean> {
  if (!navigator.gpu) { setStatus('WebGPU not available in this browser.', true); return false; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { setStatus('No GPU adapter.', true); return false; }
  device = await adapter.requestDevice();
  context = canvas.getContext('webgpu');
  if (!context) { setStatus('Failed to get WebGPU canvas context.', true); return false; }
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  const module = device.createShaderModule({ code: RENDER_SHADER });
  pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: {
      module, entryPoint: 'fs',
      targets: [{
        format: HDR_FORMAT,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const tonemapModule = device.createShaderModule({ code: TONEMAP_SHADER });
  tonemapPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: tonemapModule, entryPoint: 'vs' },
    fragment: { module: tonemapModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });

  const bloomHModule = device.createShaderModule({ code: BLOOM_H_SHADER });
  bloomHPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: bloomHModule, entryPoint: 'vs' },
    fragment: { module: bloomHModule, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list' },
  });
  const bloomVModule = device.createShaderModule({ code: BLOOM_V_SHADER });
  bloomVPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: bloomVModule, entryPoint: 'vs' },
    fragment: { module: bloomVModule, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list' },
  });
  bloomSampler = device.createSampler({
    magFilter: 'linear', minFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
  });
  tonemapUniBuf = device.createBuffer({
    size: 32,                                         // 2 × vec4<f32> (vp + fx)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Trails — fade-from-feedback fullscreen pass.
  const fadeMod = device.createShaderModule({ code: FADE_SHADER });
  fadePipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: fadeMod, entryPoint: 'vs' },
    fragment: { module: fadeMod, entryPoint: 'fs', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list' },
  });
  fadeUniBuf = device.createBuffer({
    size: 16,                                         // vec4<f32>
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Animated electron-track line.
  const trackLineMod = device.createShaderModule({ code: TRACK_LINE_SHADER });
  trackLinePipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: trackLineMod, entryPoint: 'vs' },
    fragment: {
      module: trackLineMod, entryPoint: 'fs',
      targets: [{
        format: HDR_FORMAT,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'line-list' },
  });
  trackLineUniBuf = device.createBuffer({
    size: 64 + 16,                                    // mat4 + pack
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Reaction sparks — instanced ring sprites.
  const sparkMod = device.createShaderModule({ code: SPARK_SHADER });
  sparkPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: sparkMod, entryPoint: 'vs' },
    fragment: {
      module: sparkMod, entryPoint: 'fs',
      targets: [{
        format: HDR_FORMAT,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  });
  sparkUniBuf = device.createBuffer({
    size: 64 + 16 + 16,                               // mat4 + pack + slice
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const lineModule = device.createShaderModule({ code: LINE_SHADER });
  linePipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: lineModule, entryPoint: 'vs' },
    fragment: {
      module: lineModule, entryPoint: 'fs',
      targets: [{
        format: HDR_FORMAT,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'line-list' },
  });

  const trackModule = device.createShaderModule({ code: TRACK_SHADER });
  trackPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: trackModule, entryPoint: 'vs' },
    fragment: {
      module: trackModule, entryPoint: 'fs',
      targets: [{
        format: HDR_FORMAT,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  });

  uniformBuf = device.createBuffer({
    size: 64 + 16 + 16 + 16 + 16 + 16, // mat4 + species_scale + pack + extras + bbox_min + bbox_span
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ---- Volume mode setup -----------------------------------------------
  // Allocate at MAX dim so quality switching is just a dispatch-count change.
  const maxTotal = MAX_VOXEL_DIM * MAX_VOXEL_DIM * MAX_VOXEL_DIM;
  voxelBuf = device.createBuffer({
    size: maxTotal * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  clearUniBuf = device.createBuffer({
    size: 16,                                           // vec4<u32>
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Initialized at frame time based on current state.voxelDim.

  splatUniBuf = device.createBuffer({
    size: 64,                                           // 4 vec4
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  volUniBuf = device.createBuffer({
    size: 64 + 16 + 16 + 16 + 16 + 16 + 16,             // mat4 + 6 vec4 = 160
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const clearMod = device.createShaderModule({ code: CLEAR_SHADER });
  clearPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: clearMod, entryPoint: 'cs' },
  });
  clearBG = device.createBindGroup({
    layout: clearPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: voxelBuf } },
      { binding: 1, resource: { buffer: clearUniBuf } },
    ],
  });

  const splatMod = device.createShaderModule({ code: SPLAT_SHADER });
  splatPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: splatMod, entryPoint: 'cs' },
  });

  const volMod = device.createShaderModule({ code: VOLUME_SHADER });
  volPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: volMod, entryPoint: 'vs' },
    fragment: {
      module: volMod, entryPoint: 'fs',
      targets: [{
        format: HDR_FORMAT,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  });
  volBG = device.createBindGroup({
    layout: volPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: volUniBuf } },
      { binding: 1, resource: { buffer: voxelBuf } },
    ],
  });

  // ---- Species mode: separate 128³ × 4-channel buffer + own pipelines ------
  const spTotal = SPECIES_DIM * SPECIES_DIM * SPECIES_DIM * 4;   // 4 channels
  voxelSpBuf = device.createBuffer({
    size: spTotal * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  clearSpBG = device.createBindGroup({
    layout: clearPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: voxelSpBuf } },
      { binding: 1, resource: { buffer: clearUniBuf } },
    ],
  });
  const splatSpMod = device.createShaderModule({ code: SPLAT_SP_SHADER });
  splatSpPipe = device.createComputePipeline({
    layout: 'auto',
    compute: { module: splatSpMod, entryPoint: 'cs' },
  });
  const volSpMod = device.createShaderModule({ code: VOLUME_SP_SHADER });
  volSpPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: volSpMod, entryPoint: 'vs' },
    fragment: {
      module: volSpMod, entryPoint: 'fs',
      targets: [{
        format: HDR_FORMAT,
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        },
      }],
    },
    primitive: { topology: 'triangle-list' },
  });
  volSpBG = device.createBindGroup({
    layout: volSpPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: volUniBuf } },
      { binding: 1, resource: { buffer: voxelSpBuf } },
    ],
  });

  setStatus('GPU ready. Load a snapshot .bin to begin.');
  return true;
}

/** Per-fiber hit intensity at each snapshot. fiber-hit count, normalized to ~[0,1].
 *  Cheap O(snap_n × neighbour-fibers) thanks to the regular 21×21 fiber grid:
 *  for each radical, look up only the 4 fibers nearest in (Y,Z), check distance to
 *  the segment, brute-force is fine. ~20 ms total at snap_n = 50K, n_fibers = 441. */
function computeFiberHits(blob: Blob4D): Float32Array[] {
  const fy = blob.dna.fy, fz = blob.dna.fz;
  const nF = blob.dna.n_fibers;
  const x0 = blob.dna.x0, xL = blob.dna.x0 + blob.dna.L_nm;
  const R = 5.0;             // hit radius nm — generous enough to register OH-fiber proximity
  const R2 = R * R;
  const hits: Float32Array[] = [];

  // 2D bin index over (fy, fz) with bin width = ~ fiber spacing, so we look up 4 nearest fibers per radical.
  let yMin = Infinity, yMax = -Infinity, zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < nF; i++) {
    if (fy[i] < yMin) yMin = fy[i]; if (fy[i] > yMax) yMax = fy[i];
    if (fz[i] < zMin) zMin = fz[i]; if (fz[i] > zMax) zMax = fz[i];
  }
  const binN = 24;
  const yStep = (yMax - yMin) / binN || 1;
  const zStep = (zMax - zMin) / binN || 1;
  const bins: number[][] = new Array(binN * binN).fill(0).map(() => []);
  for (let i = 0; i < nF; i++) {
    const yi = Math.max(0, Math.min(binN - 1, Math.floor((fy[i] - yMin) / yStep)));
    const zi = Math.max(0, Math.min(binN - 1, Math.floor((fz[i] - zMin) / zStep)));
    bins[yi * binN + zi].push(i);
  }

  for (const s of blob.snapshots) {
    const h = new Float32Array(nF);
    const pos = s.pos, alive = s.alive;
    for (let i = 0; i < alive.length; i++) {
      if (!alive[i]) continue;
      const x = pos[i * 4 + 0], y = pos[i * 4 + 1], z = pos[i * 4 + 2];
      if (x < x0 || x > xL) continue;
      const yi = Math.max(0, Math.min(binN - 1, Math.floor((y - yMin) / yStep)));
      const zi = Math.max(0, Math.min(binN - 1, Math.floor((z - zMin) / zStep)));
      // check this bin + 8 neighbours (covers 5 nm reach with bin width ≈ 130 nm overshoot)
      for (let dy = -1; dy <= 1; dy++) {
        const yy = yi + dy; if (yy < 0 || yy >= binN) continue;
        for (let dz = -1; dz <= 1; dz++) {
          const zz = zi + dz; if (zz < 0 || zz >= binN) continue;
          for (const f of bins[yy * binN + zz]) {
            const ddy = y - fy[f], ddz = z - fz[f];
            if (ddy * ddy + ddz * ddz < R2) { h[f] += 1; }
          }
        }
      }
    }
    // Normalize: log-scale so a few hits already register, big clusters don't blow out.
    for (let f = 0; f < nF; f++) h[f] = Math.min(1, Math.log(1 + h[f]) / 3.5);
    hits.push(h);
  }
  return hits;
}

/** Build reaction-event spark list. For each transition (k, k+1), find every
 *  radical that flipped alive→dead and emit a spark at its last-known position
 *  with birth_t = k + 0.5. Subsamples down to MAX_SPARKS to keep the buffer bounded
 *  AND to avoid blowing fillrate on integrated GPUs. */
function computeSparks(blob: Blob4D): Float32Array {
  const MAX_SPARKS = 6000;
  const events: number[] = [];     // flat (x,y,z,birth_t)*N
  for (let k = 0; k < blob.snapshots.length - 1; k++) {
    const a = blob.snapshots[k], b = blob.snapshots[k + 1];
    const aliveA = a.alive, aliveB = b.alive;
    const posA = a.pos;
    const birth = k + 0.5;
    for (let i = 0; i < aliveA.length; i++) {
      if (aliveA[i] === 1 && aliveB[i] === 0) {
        events.push(posA[i * 4], posA[i * 4 + 1], posA[i * 4 + 2], birth);
      }
    }
  }
  const total = events.length / 4;
  if (total <= MAX_SPARKS) return new Float32Array(events);
  // Reservoir-style subsample.
  const stride = Math.ceil(total / MAX_SPARKS);
  const out = new Float32Array(MAX_SPARKS * 4);
  let w = 0;
  for (let r = 0; r < total; r += stride) {
    if (w >= MAX_SPARKS) break;
    out[w * 4 + 0] = events[r * 4 + 0];
    out[w * 4 + 1] = events[r * 4 + 1];
    out[w * 4 + 2] = events[r * 4 + 2];
    out[w * 4 + 3] = events[r * 4 + 3];
    w++;
  }
  return out.subarray(0, w * 4);
}

/** Sort the initial radical buffer by x and upload as a polyline so the track-
 *  line shader can connect them. Subsamples down to ≤4096 points to keep the
 *  line readable (full 50K is a noisy mess). */
function uploadTrackLine(blob: Blob4D): void {
  if (!device || !trackLinePipe || !trackLineUniBuf) return;
  trackLinePosBuf?.destroy();
  const src = blob.rad0.rad_buf;
  const n = blob.rad0.rad_n_initial;
  if (n === 0) { nTrackLinePts = 0; trackLineBG = null; return; }
  // Build (x,i) array, sort by x, take stride.
  const idx = new Int32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => src[a * 4] - src[b * 4]);
  const TARGET = Math.min(4096, n);
  const stride = Math.max(1, Math.floor(n / TARGET));
  const out = new Float32Array(Math.ceil(n / stride) * 4);
  let w = 0;
  for (let r = 0; r < n; r += stride) {
    const i = idx[r];
    out[w * 4 + 0] = src[i * 4 + 0];
    out[w * 4 + 1] = src[i * 4 + 1];
    out[w * 4 + 2] = src[i * 4 + 2];
    out[w * 4 + 3] = 0;
    w++;
  }
  const pts = out.subarray(0, w * 4);
  nTrackLinePts = w;
  trackLinePosBuf = device.createBuffer({
    size: pts.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(trackLinePosBuf, 0, pts);
  trackLineBG = device.createBindGroup({
    layout: trackLinePipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: trackLineUniBuf } },
      { binding: 1, resource: { buffer: trackLinePosBuf } },
    ],
  });
}

function uploadSparks(blob: Blob4D): void {
  if (!device || !sparkPipe || !sparkUniBuf) return;
  sparkBuf?.destroy();
  const sparks = computeSparks(blob);
  nSparks = sparks.length / 4;
  if (nSparks === 0) {
    sparkBG = null;
    return;
  }
  sparkBuf = device.createBuffer({
    size: sparks.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(sparkBuf, 0, sparks);
  sparkBG = device.createBindGroup({
    layout: sparkPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sparkUniBuf } },
      { binding: 1, resource: { buffer: sparkBuf } },
    ],
  });
}

function uploadTracks(blob: Blob4D): void {
  if (!device || !trackPipe || !uniformBuf) return;
  trackPosBuf?.destroy();
  trackPosBuf = device.createBuffer({
    size: blob.rad0.rad_buf.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(trackPosBuf, 0, blob.rad0.rad_buf);
  trackBG = device.createBindGroup({
    layout: trackPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: trackPosBuf } },
    ],
  });
  nTracks = blob.rad0.rad_n_initial;
}

function uploadDNA(blob: Blob4D): void {
  if (!device || !linePipe || !uniformBuf) return;
  fyBuf?.destroy();
  fzBuf?.destroy();
  hitBufs.forEach((b) => b.destroy());
  hitBufs = [];
  lineBindGroups = [];
  fyBuf = device.createBuffer({
    size: blob.dna.fy.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  fzBuf = device.createBuffer({
    size: blob.dna.fz.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(fyBuf, 0, blob.dna.fy);
  device.queue.writeBuffer(fzBuf, 0, blob.dna.fz);
  nFibers = blob.dna.n_fibers;

  // Per-snapshot hit intensities. Allocate one storage buffer per snapshot,
  // then a line bind group per (k, k+1) pair so the shader can lerp between them.
  const hitMatrix = computeFiberHits(blob);
  for (const h of hitMatrix) {
    const buf = device.createBuffer({
      size: h.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, h);
    hitBufs.push(buf);
  }
  const N = hitBufs.length;
  for (let k = 0; k < N; k++) {
    const a = k, b = Math.min(k + 1, N - 1);
    lineBindGroups.push(device.createBindGroup({
      layout: linePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: fyBuf } },
        { binding: 2, resource: { buffer: fzBuf } },
        { binding: 3, resource: { buffer: hitBufs[a] } },
        { binding: 4, resource: { buffer: hitBufs[b] } },
      ],
    }));
  }
  // Keep lineBindGroup as the first pair so any stale references render the t=0 view.
  lineBindGroup = lineBindGroups[0] ?? null;
}

function uploadSnapshots(blob: Blob4D): void {
  if (!device) return;

  // Free any previous buffers.
  snapPosBufs.forEach((b) => b.destroy());
  snapAliveBufs.forEach((b) => b.destroy());
  snapPosBufs = [];
  snapAliveBufs = [];
  bindGroups = [];

  // Per-snapshot pos + alive buffers.
  for (const s of blob.snapshots) {
    const posBuf = device.createBuffer({
      size: s.pos.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(posBuf, 0, s.pos);

    const aliveBuf = device.createBuffer({
      size: s.alive.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(aliveBuf, 0, s.alive);

    snapPosBufs.push(posBuf);
    snapAliveBufs.push(aliveBuf);
  }

  // Pair bind groups — one per (k, k+1) interval. Last entry repeats the
  // final snapshot so t = N-1 still has a valid bind group.
  splatBGs = [];
  splatSpBGs = [];
  const N = blob.snapshots.length;
  for (let k = 0; k < N; k++) {
    const a = k;
    const b = Math.min(k + 1, N - 1);
    const bg = device.createBindGroup({
      layout: pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuf! } },
        { binding: 1, resource: { buffer: snapPosBufs[a] } },
        { binding: 2, resource: { buffer: snapAliveBufs[a] } },
        { binding: 3, resource: { buffer: snapPosBufs[b] } },
        { binding: 4, resource: { buffer: snapAliveBufs[b] } },
      ],
    });
    bindGroups.push(bg);

    if (splatPipe && splatUniBuf && voxelBuf) {
      splatBGs.push(device.createBindGroup({
        layout: splatPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: splatUniBuf } },
          { binding: 1, resource: { buffer: snapPosBufs[a] } },
          { binding: 2, resource: { buffer: snapAliveBufs[a] } },
          { binding: 3, resource: { buffer: snapPosBufs[b] } },
          { binding: 4, resource: { buffer: snapAliveBufs[b] } },
          { binding: 5, resource: { buffer: voxelBuf } },
        ],
      }));
    }
    if (splatSpPipe && splatUniBuf && voxelSpBuf) {
      splatSpBGs.push(device.createBindGroup({
        layout: splatSpPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: splatUniBuf } },
          { binding: 1, resource: { buffer: snapPosBufs[a] } },
          { binding: 2, resource: { buffer: snapAliveBufs[a] } },
          { binding: 3, resource: { buffer: snapPosBufs[b] } },
          { binding: 4, resource: { buffer: snapAliveBufs[b] } },
          { binding: 5, resource: { buffer: voxelSpBuf } },
        ],
      }));
    }
  }
}

let lastFrameMs = performance.now();

function frame(): void {
  if (!device || !context || !pipeline || !uniformBuf || !state.blob) {
    requestAnimationFrame(frame);
    return;
  }

  const now = performance.now();
  const dtMs = now - lastFrameMs;
  const dt = Math.min(0.1, dtMs / 1000);
  lastFrameMs = now;
  updateFpsHud(dtMs);

  // --- Auto-play: advance state.t at playSpeed checkpoints/sec, loop at end.
  if (state.playing) {
    const N = state.blob.snapshots.length;
    state.t += state.playSpeed * dt;
    if (state.t >= N - 1) state.t = 0;
    setT(state.t);
  }

  // --- HUD opacity: bright while playing or in cinematic, dim when paused.
  if (hudTimeEl) hudTimeEl.classList.toggle('show', state.playing || state.cinematic);
  if (hudCapEl) hudCapEl.classList.toggle('show', state.playing || state.cinematic);

  // --- Cinematic intro: dolly cam from far→home with smoothstep ease.
  if (state.intro) {
    const u = Math.min(1, (now - state.intro.startMs) / state.intro.duration);
    const e = u * u * (3 - 2 * u);                      // smoothstep
    state.cam.yaw    = state.intro.fromYaw    + (state.intro.toYaw    - state.intro.fromYaw)    * e;
    state.cam.pitch  = state.intro.fromPitch  + (state.intro.toPitch  - state.intro.fromPitch)  * e;
    state.cam.radius = state.intro.fromRadius + (state.intro.toRadius - state.intro.fromRadius) * e;
    if (u >= 1) state.intro = null;
    state.lastInteractionMs = now;                       // suppress auto-orbit during intro
  }

  // --- Auto-orbit: gentle yaw drift if idle for >3s and no drag.
  if (!state.drag.active && !state.intro && now - state.lastInteractionMs > 3000) {
    // Slower drift in cinematic mode for a more deliberate camera feel.
    const rate = state.cinematic ? 0.05 : 0.08;
    state.cam.yaw += rate * dt;
  }

  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  const aspect = w / h;
  const vp = viewProj(state.cam, aspect);
  const uniData = new Float32Array(36);
  uniData.set(vp, 0);
  // species_scale: OH/eaq/H/H3O+ — eaq diffuses fastest so render slightly larger.
  uniData[16] = 1.0; uniData[17] = 1.4; uniData[18] = 1.2; uniData[19] = 1.0;

  // Resolve current pair index k and interpolation factor u ∈ [0,1].
  const N = state.blob.snapshots.length;
  const tClamped = Math.max(0, Math.min(N - 1, state.t));
  const k = Math.min(Math.floor(tClamped), N - 2);
  const u = tClamped - k;
  uniData[20] = u;                              // t_frac
  // species_mask is a u32 — write bit pattern via Uint32 view of same buffer.
  const uniU32 = new Uint32Array(uniData.buffer);
  uniU32[21] = state.speciesMask;               // species_mask
  uniData[22] = state.blob.dna.x0;              // dna_x0
  uniData[23] = state.blob.dna.L_nm;            // dna_L
  // extras: x=sprite intensity, y=hit gain (global), z=slice axis (cast int), w=slice pos.
  uniData[24] = (state.viewMode === 'sprites') ? 1.0 : (state.compareOverlay ? 0.4 : 0.0);
  uniData[25] = state.showHits ? 1.0 : 0.0;
  uniData[26] = state.sliceAxis;
  uniData[27] = state.slicePos;
  // bbox_min (vec4) + bbox_span (vec4) — for slice computations.
  uniData[28] = state.bboxMin[0]; uniData[29] = state.bboxMin[1]; uniData[30] = state.bboxMin[2]; uniData[31] = 0;
  uniData[32] = state.bboxMax[0] - state.bboxMin[0];
  uniData[33] = state.bboxMax[1] - state.bboxMin[1];
  uniData[34] = state.bboxMax[2] - state.bboxMin[2];
  uniData[35] = 0;
  device.queue.writeBuffer(uniformBuf, 0, uniData);

  ensureHdrTarget(w, h);
  if (!hdrView || !tonemapBG || !tonemapPipe) {
    requestAnimationFrame(frame);
    return;
  }

  const enc = device.createCommandEncoder();

  // ---- Volume mode: compute passes (clear voxels, then splat into them) ----
  if (state.viewMode === 'volume' && clearPipe && splatPipe && voxelBuf
      && splatUniBuf && volUniBuf && volPipe && volBG && clearBG
      && splatBGs.length > 0) {
    const speciesMode = state.volColor === 'species';
    // Species mode forces dim=128 (memory cap on the 4-channel buffer).
    const dim = speciesMode ? SPECIES_DIM : state.voxelDim;
    const channels = speciesMode ? 4 : 1;
    const totalVoxelEntries = dim * dim * dim * channels;

    // 2D dispatch helpers — WebGPU caps each axis at 65535 workgroups, so big
    // grids need a (X, Y) shape. Stride in X (in linear-thread units) is
    // X_workgroups × workgroup_size; the shader recovers `i` from gid.
    const dispatch2D = (totalThreads: number): { x: number; y: number; strideX: number } => {
      const wg = Math.ceil(totalThreads / 64);
      if (wg <= 65535) return { x: wg, y: 1, strideX: wg * 64 };
      const x = 65535;
      const y = Math.ceil(wg / x);
      return { x, y, strideX: x * 64 };
    };

    const clearDisp = dispatch2D(totalVoxelEntries);
    const splatDisp = dispatch2D(state.blob.snap_n);

    // Refresh clear-uniform with current total + X-stride.
    if (clearUniBuf) {
      device.queue.writeBuffer(clearUniBuf, 0, new Uint32Array([totalVoxelEntries, clearDisp.strideX, 0, 0]));
    }

    // Splat uniforms (bbox + ipack + fpack).
    const sUni = new ArrayBuffer(64);
    const sF = new Float32Array(sUni);
    const sU = new Uint32Array(sUni);
    sF[0] = state.bboxMin[0]; sF[1] = state.bboxMin[1]; sF[2] = state.bboxMin[2]; sF[3] = 0;
    sF[4] = state.bboxMax[0]; sF[5] = state.bboxMax[1]; sF[6] = state.bboxMax[2]; sF[7] = 0;
    sU[8]  = state.blob.snap_n;
    sU[9]  = dim;
    sU[10] = state.speciesMask;
    sU[11] = splatDisp.strideX;
    sF[12] = u; sF[13] = 0; sF[14] = 0; sF[15] = 0;
    device.queue.writeBuffer(splatUniBuf, 0, sUni);

    const cp = enc.beginComputePass();
    cp.setPipeline(clearPipe);
    cp.setBindGroup(0, speciesMode && clearSpBG ? clearSpBG : clearBG);
    cp.dispatchWorkgroups(clearDisp.x, clearDisp.y);
    if (speciesMode && splatSpPipe && splatSpBGs.length > 0) {
      cp.setPipeline(splatSpPipe);
      cp.setBindGroup(0, splatSpBGs[k]);
    } else {
      cp.setPipeline(splatPipe);
      cp.setBindGroup(0, splatBGs[k]);
    }
    cp.dispatchWorkgroups(splatDisp.x, splatDisp.y);
    cp.end();

    // Volume render uniforms (inv_vp + cam_eye + bbox + fpack + ipack + spack).
    const inv_vp = inverse4(vp);
    const eye = cameraEye(state.cam);
    const vUni = new ArrayBuffer(160);
    const vF = new Float32Array(vUni);
    const vU = new Uint32Array(vUni);
    vF.set(inv_vp, 0);
    vF[16] = eye[0]; vF[17] = eye[1]; vF[18] = eye[2]; vF[19] = 0;
    vF[20] = state.bboxMin[0]; vF[21] = state.bboxMin[1]; vF[22] = state.bboxMin[2]; vF[23] = 0;
    vF[24] = state.bboxMax[0]; vF[25] = state.bboxMax[1]; vF[26] = state.bboxMax[2]; vF[27] = 0;
    vF[28] = w; vF[29] = h;
    // density_scale governs how opaque each voxel-count is per nm of march.
    const dimScale = dim / DEFAULT_VOXEL_DIM;
    vF[30] = 0.06 * dimScale * dimScale;
    vF[31] = 1.1;     // exposure
    vU[32] = dim;
    vU[33] = state.volColor === 'iso' ? 1 : 0;          // ipack.y mode
    vU[34] = state.sliceAxis;                            // ipack.z slice_axis
    vU[35] = 0;
    vF[36] = state.slicePos;                             // spack.x slice_pos
    vF[37] = 0; vF[38] = 0; vF[39] = 0;
    device.queue.writeBuffer(volUniBuf, 0, vUni);
  }

  // ---- Trails: fade-and-copy previous frame into hdrTex (separate pass so we
  // can sample hdrFeedbackView; you can't sample what you're rendering to). ----
  const trailsActive = state.trails && fadePipe && fadeBG && fadeUniBuf && prevFrameValid;
  if (trailsActive) {
    device.queue.writeBuffer(fadeUniBuf!, 0, new Float32Array([0.86, 0, 0, 0]));
    const fp = enc.beginRenderPass({
      colorAttachments: [{
        view: hdrView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    fp.setPipeline(fadePipe!);
    fp.setBindGroup(0, fadeBG!);
    fp.draw(3, 1, 0, 0);
    fp.end();
  }

  // ---- HDR pass: DNA grid + (sprites OR volume) ----
  const hdrPass = enc.beginRenderPass({
    colorAttachments: [{
      view: hdrView,
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: trailsActive ? 'load' : 'clear', storeOp: 'store',
    }],
  });
  // DNA fiber lines first — they're a static reference grid.
  // Pick the line bind group matching the current snapshot pair so hit pulses lerp correctly.
  const linePairBG = lineBindGroups[Math.min(k, lineBindGroups.length - 1)] ?? lineBindGroup;
  if (linePipe && linePairBG && nFibers > 0) {
    hdrPass.setPipeline(linePipe);
    hdrPass.setBindGroup(0, linePairBG);
    hdrPass.draw(2, nFibers, 0, 0);
  }

  // Animated electron-track line — draws on during cinematic intro, then sits
  // at low alpha behind the cloud so the path is always visible.
  if (trackLinePipe && trackLineBG && trackLineUniBuf && nTrackLinePts > 1) {
    let progress = 1.0;
    let baseAlpha = 0.18;
    if (state.intro) {
      const u = Math.min(1, (now - state.intro.startMs) / state.intro.duration);
      // Track line draws on during the first 80% of the intro so it lands before settling.
      progress = Math.min(1, u / 0.8);
      baseAlpha = 0.55;     // brighter while drawing on
    }
    const tlUni = new ArrayBuffer(80);
    const tlF = new Float32Array(tlUni);
    tlF.set(vp, 0);
    tlF[16] = progress;
    tlF[17] = baseAlpha;
    tlF[18] = 0; tlF[19] = 0;
    device.queue.writeBuffer(trackLineUniBuf, 0, tlUni);
    hdrPass.setPipeline(trackLinePipe);
    hdrPass.setBindGroup(0, trackLineBG);
    // line-list: each segment = 2 vertices, instanced once per segment.
    const segments = nTrackLinePts - 1;
    hdrPass.draw(2, segments, 0, 0);
  }

  // Track footprint overlay — render before main view so cloud/sprites can
  // sit on top of the track structure.
  if (state.showTracks && trackPipe && trackBG && nTracks > 0) {
    hdrPass.setPipeline(trackPipe);
    hdrPass.setBindGroup(0, trackBG);
    hdrPass.draw(6, nTracks, 0, 0);
  }

  if (state.viewMode === 'volume' && volPipe && volBG) {
    if (state.volColor === 'species' && volSpPipe && volSpBG) {
      hdrPass.setPipeline(volSpPipe);
      hdrPass.setBindGroup(0, volSpBG);
    } else {
      hdrPass.setPipeline(volPipe);
      hdrPass.setBindGroup(0, volBG);
    }
    hdrPass.draw(3, 1, 0, 0);

    // Compare overlay — draw dimmed sprites on top of volume so misalignment
    // between the two render paths is visually obvious.
    if (state.compareOverlay) {
      hdrPass.setPipeline(pipeline);
      hdrPass.setBindGroup(0, bindGroups[k]);
      hdrPass.draw(6, state.blob.snap_n, 0, 0);
    }
  } else {
    hdrPass.setPipeline(pipeline);
    hdrPass.setBindGroup(0, bindGroups[k]);
    hdrPass.draw(6, state.blob.snap_n, 0, 0);
  }

  // Reaction-event sparks — drawn on top of sprites/volume.
  if (state.showSparks && sparkPipe && sparkBG && sparkUniBuf && nSparks > 0) {
    // Tune base ring size to scene scale: ~3% of bbox span feels right.
    const span = Math.max(
      state.bboxMax[0] - state.bboxMin[0],
      state.bboxMax[1] - state.bboxMin[1],
      state.bboxMax[2] - state.bboxMin[2],
    );
    const baseSize = Math.max(0.5, span * 0.012);
    const ax = state.sliceAxis;
    const bmin = ax === 0 ? state.bboxMin[0] : ax === 1 ? state.bboxMin[1] : ax === 2 ? state.bboxMin[2] : 0;
    const bspan = ax === 0 ? state.bboxMax[0] - state.bboxMin[0]
                : ax === 1 ? state.bboxMax[1] - state.bboxMin[1]
                : ax === 2 ? state.bboxMax[2] - state.bboxMin[2] : 1;
    const sUni = new ArrayBuffer(96);
    const sF = new Float32Array(sUni);
    sF.set(vp, 0);
    sF[16] = state.t;
    sF[17] = SPARK_LIFETIME;
    sF[18] = baseSize;
    sF[19] = 0;
    sF[20] = ax; sF[21] = state.slicePos; sF[22] = bspan; sF[23] = bmin;
    device.queue.writeBuffer(sparkUniBuf, 0, sUni);

    hdrPass.setPipeline(sparkPipe);
    hdrPass.setBindGroup(0, sparkBG);
    hdrPass.draw(6, nSparks, 0, 0);
  }

  hdrPass.end();

  // Pass 2a — Bloom (only when enabled). Two-pass separable Gaussian on a
  // half-res buffer. When disabled we skip the work; the tonemap shader still
  // samples bloomViewB but multiplies by intensity=0 so it contributes nothing.
  if (state.bloom && bloomHPipe && bloomVPipe && bloomHBG && bloomVBG && bloomViewA && bloomViewB) {
    const bw = Math.max(1, Math.floor(w / 2));
    const bh = Math.max(1, Math.floor(h / 2));
    const hPass = enc.beginRenderPass({
      colorAttachments: [{
        view: bloomViewA, clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    hPass.setPipeline(bloomHPipe);
    hPass.setBindGroup(0, bloomHBG);
    hPass.setViewport(0, 0, bw, bh, 0, 1);
    hPass.draw(3, 1, 0, 0);
    hPass.end();

    const vPass = enc.beginRenderPass({
      colorAttachments: [{
        view: bloomViewB, clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    vPass.setPipeline(bloomVPipe);
    vPass.setBindGroup(0, bloomVBG);
    vPass.setViewport(0, 0, bw, bh, 0, 1);
    vPass.draw(3, 1, 0, 0);
    vPass.end();
  }

  // Tonemap uniform: vp (W, H, bloom_intensity, time_s) + fx (vignette, grain, cinematic_flag, _).
  if (tonemapUniBuf) {
    const t_s = (now - bootTimeMs) / 1000;
    const tu = new Float32Array([
      w, h, state.bloom ? 1.0 : 0.0, t_s,
      state.cinematic ? 0.55 : 0.30,           // vignette strength
      state.cinematic ? 0.045 : 0.025,          // grain
      state.cinematic ? 1.0 : 0.0,              // cinematic flag
      0,
    ]);
    device.queue.writeBuffer(tonemapUniBuf, 0, tu);
  }

  // Pass 2 — Reinhard tonemap HDR (+ bloom) → swapchain.
  const swapView = context.getCurrentTexture().createView();
  const tmPass = enc.beginRenderPass({
    colorAttachments: [{
      view: swapView,
      clearValue: { r: 0.018, g: 0.020, b: 0.028, a: 1 },
      loadOp: 'clear', storeOp: 'store',
    }],
  });
  tmPass.setPipeline(tonemapPipe);
  tmPass.setBindGroup(0, tonemapBG);
  tmPass.draw(3, 1, 0, 0);
  tmPass.end();

  // Copy current HDR into feedback texture so next frame's fade pass can sample it.
  if (state.trails && hdrTex && hdrFeedbackTex) {
    enc.copyTextureToTexture(
      { texture: hdrTex },
      { texture: hdrFeedbackTex },
      { width: hdrSize.w, height: hdrSize.h, depthOrArrayLayers: 1 },
    );
    prevFrameValid = true;
  } else {
    prevFrameValid = false;
  }

  device.queue.submit([enc.finish()]);

  requestAnimationFrame(frame);
}

// -----------------------------------------------------------------------------
// UI handlers
// -----------------------------------------------------------------------------

function fmtTime(t_ns: number): string {
  if (t_ns === 0) return '0';
  if (t_ns < 0.001) return `${(t_ns * 1000).toFixed(2)} ps`;
  if (t_ns < 1) return `${t_ns.toFixed(3)} ns`;
  if (t_ns < 1000) return `${t_ns.toFixed(1)} ns`;
  return `${(t_ns / 1000).toFixed(2)} μs`;
}

// Cached per-snapshot alive counts + validation report.
let aliveCounts: number[] = [];
let validationStatus: string = '';
let validationDetails: string[] = [];

interface ValidationReport {
  counts: number[];
  status: string;       // 'OK' or first issue summary
  details: string[];    // bullet list of all checks (passed + failed)
}

function computeAliveCounts(blob: Blob4D): ValidationReport {
  const counts: number[] = [];
  for (const s of blob.snapshots) {
    let n = 0;
    for (let i = 0; i < s.alive.length; i++) n += s.alive[i];
    counts.push(n);
  }
  const issues: string[] = [];
  const details: string[] = [];

  // Check 1: t=0 alive should equal snap_n (alive[] initialized to 1 for all).
  if (counts[0] === blob.snap_n) {
    details.push(`✓ t=0 alive = snap_n = ${blob.snap_n.toLocaleString()}`);
  } else {
    const msg = `t=0 alive=${counts[0]} ≠ snap_n=${blob.snap_n}`;
    issues.push(msg); details.push(`✗ ${msg}`);
  }

  // Check 2: counts monotonically non-increasing (chemistry only consumes).
  let monoOK = true;
  for (let i = 1; i < counts.length; i++) {
    if (counts[i] > counts[i - 1]) {
      const msg = `${blob.snapshots[i].label} alive ↑ (${counts[i - 1]}→${counts[i]})`;
      issues.push(msg); details.push(`✗ ${msg}`); monoOK = false;
    }
  }
  if (monoOK) {
    details.push(`✓ alive monotonically non-increasing across ${counts.length} checkpoints`);
  }

  // Check 3: species partition at t=0 sums to snap_n. Each radical's species
  // is encoded in pos.w; valid species are 0..7.
  const sp = new Array<number>(8).fill(0);
  const pos0 = blob.snapshots[0].pos;
  const alive0 = blob.snapshots[0].alive;
  for (let i = 0; i < alive0.length; i++) {
    if (!alive0[i]) continue;
    const s = (pos0[i * 4 + 3] | 0) & 7;
    sp[s]++;
  }
  const spSum = sp.reduce((a, b) => a + b, 0);
  if (spSum === counts[0]) {
    const breakdown = ['OH', 'eaq', 'H', 'H₃O⁺', 's4', 'pre-eaq', 'OH⁻', 'H₂']
      .map((lbl, i) => sp[i] ? `${lbl}=${sp[i]}` : '').filter(Boolean).join(', ');
    details.push(`✓ species partition at t=0 sums to ${spSum.toLocaleString()} (${breakdown})`);
  } else {
    const msg = `species partition Σ=${spSum} ≠ alive[0]=${counts[0]}`;
    issues.push(msg); details.push(`✗ ${msg}`);
  }

  // Check 4: at last snapshot, every alive radical has a finite position
  // (not NaN/Inf — would indicate a corrupt buffer). Bbox containment is
  // tautological since bbox is computed FROM this snapshot, so we skip it.
  const last = blob.snapshots[blob.snapshots.length - 1];
  let nonFinite = 0;
  for (let i = 0; i < last.alive.length; i++) {
    if (!last.alive[i]) continue;
    const x = last.pos[i * 4], y = last.pos[i * 4 + 1], z = last.pos[i * 4 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) nonFinite++;
  }
  if (nonFinite === 0) {
    details.push(`✓ all positions finite at ${last.label} (${counts[counts.length - 1].toLocaleString()} radicals)`);
  } else {
    const msg = `${nonFinite} non-finite positions at ${last.label}`;
    issues.push(msg); details.push(`✗ ${msg}`);
  }

  return {
    counts,
    status: issues.length === 0 ? 'OK' : issues[0],
    details,
  };
}

function renderCascadeTable(blob: Blob4D): string {
  const total = blob.snap_n;
  const rows = blob.snapshots.map((s, i) => {
    const n = aliveCounts[i];
    const pct = ((n / total) * 100).toFixed(1);
    return `<tr><td>${s.label}</td><td style="text-align:right">${n.toLocaleString()}</td><td style="text-align:right;color:#6a8af0">${pct}%</td></tr>`;
  });
  return `<table style="font-size:10px;border-collapse:collapse;margin-top:6px;width:100%">
    <thead><tr style="color:#6a6a7a"><th style="text-align:left">checkpoint</th><th style="text-align:right">alive</th><th style="text-align:right">survival</th></tr></thead>
    <tbody>${rows.join('')}</tbody>
  </table>`;
}

/** Linear-in-log time interpolation between checkpoint k and k+1. */
function interpolatedTimeNs(t: number, snaps: Snapshot[]): number {
  const N = snaps.length;
  const tc = Math.max(0, Math.min(N - 1, t));
  const k = Math.min(Math.floor(tc), N - 2);
  const u = tc - k;
  const ta = snaps[k].t_ns;
  const tb = snaps[k + 1].t_ns;
  if (ta <= 0) return u * tb; // first interval starts at t=0; linear lerp
  return Math.exp((1 - u) * Math.log(ta) + u * Math.log(tb));
}

function setT(t: number): void {
  if (!state.blob) return;
  const N = state.blob.snapshots.length;
  state.t = Math.max(0, Math.min(N - 1, t));
  slider.value = String(state.t);

  const t_ns = interpolatedTimeNs(state.t, state.blob.snapshots);
  // Approximate alive count — linear lerp of integer counts.
  const k = Math.min(Math.floor(state.t), N - 2);
  const u = state.t - k;
  const alive_count = Math.round(aliveCounts[k] * (1 - u) + aliveCounts[k + 1] * u);

  tLabel.textContent = fmtTime(t_ns);
  if (hudTimeEl) hudTimeEl.textContent = fmtTime(t_ns);
  const okColor = validationStatus === 'OK' ? '#3acf6a' : '#ff6090';
  const detailsHTML = validationDetails.length
    ? `<details style="margin-top:4px;border-top:none;padding-top:0"><summary style="font-size:9px;letter-spacing:0.08em">round-trip detail</summary><div style="margin-top:4px;font-size:10px;line-height:1.55;color:#8a8a9a">${validationDetails.map(d => `<div>${d}</div>`).join('')}</div></details>`
    : '';
  metaEl.innerHTML =
    `<b>${state.blob.snap_n.toLocaleString()}</b> radicals subsampled · ` +
    `<b>${alive_count.toLocaleString()}</b> alive @ ${fmtTime(t_ns)}.<br>` +
    `Energy: <b>${(state.blob.energy_eV / 1000).toFixed(0)} keV</b> · ` +
    `DNA: <b>${state.blob.dna.n_fibers}</b> fibers × <b>${state.blob.dna.n_bp_per}</b> bp.<br>` +
    `Round-trip: <b style="color:${okColor}">${validationStatus}</b>` +
    detailsHTML +
    renderCascadeTable(state.blob);

  // Highlight nearest checkpoint button.
  const nearestIdx = Math.round(state.t);
  for (const btn of cpsEl.querySelectorAll('button')) {
    btn.classList.toggle('active', btn.getAttribute('data-i') === String(nearestIdx));
  }
}

function rebuildCheckpointButtons(): void {
  cpsEl.innerHTML = '';
  if (!state.blob) return;
  state.blob.snapshots.forEach((s, i) => {
    const btn = document.createElement('button');
    btn.textContent = s.label;
    btn.setAttribute('data-i', String(i));
    btn.onclick = () => setT(i);
    cpsEl.appendChild(btn);
  });
}

fileInput.addEventListener('change', async () => {
  const f = fileInput.files?.[0];
  if (f) await loadFile(f);
});

slider.addEventListener('input', () => { setT(parseFloat(slider.value)); markInteraction(); });

// Mouse orbit + scroll zoom.
canvas.addEventListener('mousedown', (e) => {
  state.drag.active = true;
  state.drag.lastX = e.clientX;
  state.drag.lastY = e.clientY;
  markInteraction();
});
window.addEventListener('mouseup', () => { state.drag.active = false; markInteraction(); });
window.addEventListener('mousemove', (e) => {
  if (!state.drag.active) return;
  const dx = e.clientX - state.drag.lastX;
  const dy = e.clientY - state.drag.lastY;
  state.drag.lastX = e.clientX;
  state.drag.lastY = e.clientY;
  state.cam.yaw -= dx * 0.005;
  state.cam.pitch = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1, state.cam.pitch + dy * 0.005));
  markInteraction();
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = Math.exp(e.deltaY * 0.001);
  state.cam.radius = Math.max(50, Math.min(1e6, state.cam.radius * factor));
  markInteraction();
}, { passive: false });

// -----------------------------------------------------------------------------
// Polish: drag-drop, play/pause, species toggles, reset cam, screenshot
// -----------------------------------------------------------------------------

/** Most-recently-loaded raw blob — fed to the in-viewer "Save .bin" button. */
let lastLoadedBlob: Blob | null = null;
let lastLoadedFilename: string = '';

async function loadFile(f: File): Promise<void> {
  setStatus(`Loading ${f.name} (${(f.size / 1024 / 1024).toFixed(2)} MB)…`);
  lastLoadedBlob = f;
  lastLoadedFilename = f.name;
  try {
    const buf = await f.arrayBuffer();
    const blob = parseBlob(buf);
    state.blob = blob;
    const report = computeAliveCounts(blob);
    aliveCounts = report.counts;
    validationStatus = report.status;
    validationDetails = report.details;
    uploadSnapshots(blob);
    uploadDNA(blob);
    uploadTracks(blob);
    uploadSparks(blob);
    uploadTrackLine(blob);

    // Compute scene bbox using the most-diffused snapshot (radicals at their
    // widest spread). Volume mode + camera framing both key off this.
    const last = blob.snapshots[blob.snapshots.length - 1];
    let mn: [number, number, number] = [Infinity, Infinity, Infinity];
    let mx: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < last.alive.length; i++) {
      if (!last.alive[i]) continue;
      const x = last.pos[i * 4], y = last.pos[i * 4 + 1], z = last.pos[i * 4 + 2];
      if (x < mn[0]) mn[0] = x; if (y < mn[1]) mn[1] = y; if (z < mn[2]) mn[2] = z;
      if (x > mx[0]) mx[0] = x; if (y > mx[1]) mx[1] = y; if (z > mx[2]) mx[2] = z;
    }
    if (!isFinite(mn[0])) { mn = [-100, -100, -100]; mx = [100, 100, 100]; }
    // Pad by 8% on each side and inflate to >= unit cube to avoid zero-extent axes.
    const minSpan = 1.0;
    for (let a = 0; a < 3; a++) {
      let span = mx[a] - mn[a];
      if (span < minSpan) {
        const c = (mx[a] + mn[a]) * 0.5;
        mn[a] = c - minSpan * 0.5; mx[a] = c + minSpan * 0.5;
        span = minSpan;
      }
      const pad = span * 0.08;
      mn[a] -= pad; mx[a] += pad;
    }
    state.bboxMin = mn;
    state.bboxMax = mx;

    // Frame the camera around the bbox. Target = bbox center, radius scaled
    // by bounding sphere so the whole cloud fits even at extreme yaw/pitch.
    const center: [number, number, number] = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
    const halfSpan: [number, number, number] = [(mx[0] - mn[0]) * 0.5, (mx[1] - mn[1]) * 0.5, (mx[2] - mn[2]) * 0.5];
    const boundingRadius = Math.hypot(halfSpan[0], halfSpan[1], halfSpan[2]);
    const dnaExtent = Math.max(Math.abs(blob.dna.x0), Math.abs(blob.dna.x0 + blob.dna.L_nm));
    state.cam.target = center;
    state.cam.radius = Math.max(boundingRadius * 2.2, dnaExtent * 1.6, 100);
    state.initialRadius = state.cam.radius;
    state.cam.yaw = 0.6;
    state.cam.pitch = 0.3;
    slider.disabled = false;
    slider.min = '0';
    slider.max = String(blob.snapshots.length - 1);
    slider.step = '0.02';
    rebuildCheckpointButtons();
    setT(0);
    startIntro();
    setStatus(`Loaded ${blob.num_snaps} checkpoints, ${blob.snap_n.toLocaleString()} radicals each.`);
  } catch (e) {
    setStatus(`Parse error: ${e instanceof Error ? e.message : String(e)}`, true);
  }
}

// Drag-and-drop anywhere on the page.
const dropOverlay = document.createElement('div');
dropOverlay.className = 'drop-overlay';
dropOverlay.textContent = 'Drop snapshot .bin to load';
document.body.appendChild(dropOverlay);
let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragDepth++;
  dropOverlay.classList.add('active');
});
window.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropOverlay.classList.remove('active');
});
window.addEventListener('dragover', (e) => { e.preventDefault(); });
window.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropOverlay.classList.remove('active');
  const f = e.dataTransfer?.files?.[0];
  if (f) await loadFile(f);
});

// Play / pause.
const playBtn = document.getElementById('play-btn') as HTMLButtonElement | null;
const speedInput = document.getElementById('speed') as HTMLInputElement | null;
function setPlaying(on: boolean): void {
  state.playing = on;
  if (playBtn) playBtn.textContent = on ? '⏸ pause' : '▶ play';
  markInteraction();
}
playBtn?.addEventListener('click', () => setPlaying(!state.playing));
speedInput?.addEventListener('input', () => {
  state.playSpeed = parseFloat(speedInput.value);
});

// Reset view.
const resetBtn = document.getElementById('reset-cam') as HTMLButtonElement | null;
function resetCamera(): void {
  state.cam.yaw = 0.6;
  state.cam.pitch = 0.3;
  state.cam.radius = state.initialRadius;
  state.cam.target = [0, 0, 0];
  markInteraction();
}
resetBtn?.addEventListener('click', resetCamera);

// Species toggles.
function buildSpeciesToggles(): void {
  const container = document.getElementById('species-toggles');
  if (!container) return;
  const items = [
    { idx: 0, label: 'OH',     swatch: '#4cd9ff' },
    { idx: 1, label: 'e⁻aq',   swatch: '#ff4ce6' },
    { idx: 2, label: 'H',      swatch: '#ffe64c' },
    { idx: 3, label: 'H₃O⁺',   swatch: '#ff8c33' },
    { idx: 5, label: 'pre-eaq', swatch: '#a659ff' },
    { idx: 6, label: 'OH⁻',    swatch: '#4cff8c' },
    { idx: 7, label: 'H₂',     swatch: '#d9d9f2' },
  ];
  container.innerHTML = items.map((it) =>
    `<label><input type="checkbox" data-species="${it.idx}" checked />` +
    `<i style="background:${it.swatch}"></i>${it.label}</label>`
  ).join('');
  container.querySelectorAll('input[type=checkbox]').forEach((cb) => {
    cb.addEventListener('change', () => {
      let mask = 0;
      container.querySelectorAll('input[type=checkbox]').forEach((c) => {
        const inp = c as HTMLInputElement;
        if (inp.checked) mask |= 1 << parseInt(inp.dataset.species ?? '0', 10);
      });
      state.speciesMask = mask;
    });
  });
}

// View mode + volume sub-options.
const viewModeSel = document.getElementById('view-mode') as HTMLSelectElement | null;
const volQualityRow = document.getElementById('vol-row-quality') as HTMLDivElement | null;
const volColorRow = document.getElementById('vol-row-color') as HTMLDivElement | null;
const volQualitySel = document.getElementById('vol-quality') as HTMLSelectElement | null;
const volColorSel = document.getElementById('vol-color') as HTMLSelectElement | null;
const bloomCb = document.getElementById('bloom') as HTMLInputElement | null;
const tracksCb = document.getElementById('tracks') as HTMLInputElement | null;

function syncVolumeRowVisibility(): void {
  const show = state.viewMode === 'volume' ? '' : 'none';
  if (volQualityRow) volQualityRow.style.display = show;
  if (volColorRow) volColorRow.style.display = show;
}
syncVolumeRowVisibility();

viewModeSel?.addEventListener('change', () => {
  const v = viewModeSel.value;
  state.viewMode = (v === 'volume') ? 'volume' : 'sprites';
  syncVolumeRowVisibility();
  markInteraction();
});

volQualitySel?.addEventListener('change', () => {
  const dim = parseInt(volQualitySel.value, 10);
  if (Number.isFinite(dim) && dim >= 64 && dim <= MAX_VOXEL_DIM) {
    state.voxelDim = dim;
    markInteraction();
  }
});

volColorSel?.addEventListener('change', () => {
  const v = volColorSel.value;
  state.volColor = v === 'species' ? 'species' : v === 'iso' ? 'iso' : 'density';
  markInteraction();
});

bloomCb?.addEventListener('change', () => {
  state.bloom = !!bloomCb.checked;
  markInteraction();
});

tracksCb?.addEventListener('change', () => {
  state.showTracks = !!tracksCb.checked;
  markInteraction();
});

const compareCb = document.getElementById('compare') as HTMLInputElement | null;
compareCb?.addEventListener('change', () => {
  state.compareOverlay = !!compareCb.checked;
  markInteraction();
});

const trailsCb = document.getElementById('trails') as HTMLInputElement | null;
trailsCb?.addEventListener('change', () => {
  state.trails = !!trailsCb.checked;
  prevFrameValid = false;
  markInteraction();
});

const sparksCb = document.getElementById('sparks') as HTMLInputElement | null;
sparksCb?.addEventListener('change', () => {
  state.showSparks = !!sparksCb.checked;
  markInteraction();
});

const hitsCb = document.getElementById('hits') as HTMLInputElement | null;
hitsCb?.addEventListener('change', () => {
  state.showHits = !!hitsCb.checked;
  markInteraction();
});

const sliceAxisSel = document.getElementById('slice-axis') as HTMLSelectElement | null;
const slicePosInput = document.getElementById('slice-pos') as HTMLInputElement | null;
const sliceLabel = document.getElementById('slice-label') as HTMLSpanElement | null;
sliceAxisSel?.addEventListener('change', () => {
  const v = parseInt(sliceAxisSel.value, 10);
  state.sliceAxis = (v >= 0 && v <= 3 ? v : 3) as 0 | 1 | 2 | 3;
  markInteraction();
});
slicePosInput?.addEventListener('input', () => {
  const v = parseFloat(slicePosInput.value);
  state.slicePos = Math.max(0, Math.min(1, v / 100));
  if (sliceLabel) sliceLabel.textContent = `${Math.round(v)}%`;
  markInteraction();
});

// Screenshot.
const shotBtn = document.getElementById('screenshot') as HTMLButtonElement | null;
shotBtn?.addEventListener('click', () => {
  canvas.toBlob((b) => {
    if (!b) return;
    const url = URL.createObjectURL(b);
    const a = document.createElement('a');
    a.href = url;
    a.download = `wgdna4d-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
  }, 'image/png');
});

// Keyboard shortcuts.
window.addEventListener('keydown', (e) => {
  // Ignore when typing in inputs.
  const tag = (e.target as HTMLElement | null)?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  switch (e.key) {
    case ' ':
      e.preventDefault();
      setPlaying(!state.playing);
      break;
    case 'ArrowRight':
      if (state.blob) setT(Math.min(state.blob.snapshots.length - 1, Math.floor(state.t) + 1));
      break;
    case 'ArrowLeft':
      if (state.blob) setT(Math.max(0, Math.ceil(state.t) - 1));
      break;
    case 'r': case 'R':
      resetCamera();
      break;
    case 's': case 'S':
      shotBtn?.click();
      break;
    case 'f': case 'F':
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
      break;
    case 'c': case 'C':
      setCinematic(!state.cinematic);
      break;
  }
});

buildSpeciesToggles();

// Save-bin button — exports whatever blob was loaded (handoff, fetch, or pick).
const saveBinBtn = document.getElementById('save-bin') as HTMLButtonElement | null;
saveBinBtn?.addEventListener('click', () => {
  if (!lastLoadedBlob) return;
  const url = URL.createObjectURL(lastLoadedBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = lastLoadedFilename || 'wgdna4d-snapshot.bin';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
});

/** Fetch the bundled demo snapshot and load it. Returns false on 404/offline.
 *  Served same-origin from `/wgdna-default.bin` — the file is .gitignored
 *  but pulled into `public/` at build time by `tools/fetch-demo.mjs`
 *  (wired as the `prebuild` npm script). For local dev, run `npm run
 *  fetch-demo` once to populate `public/wgdna-default.bin`. */
async function loadBundledDemo(): Promise<boolean> {
  try {
    const resp = await fetch('/wgdna-default.bin', { cache: 'force-cache' });
    if (!resp.ok) return false;
    const buf = await resp.arrayBuffer();
    const f = new File([buf], 'wgdna-default.bin');
    setStatus('Loading bundled demo snapshot…');
    await loadFile(f);
    return true;
  } catch {
    return false;
  }
}

// "Use ready data" — explicit one-click load of the shipped demo.
const loadReadyBtn = document.getElementById('load-ready') as HTMLButtonElement | null;
loadReadyBtn?.addEventListener('click', async () => {
  loadReadyBtn.disabled = true;
  setStatus('Fetching bundled demo snapshot…');
  const ok = await loadBundledDemo();
  if (!ok) setStatus('Demo snapshot unavailable (offline or 404).', true);
  loadReadyBtn.disabled = false;
});

/**
 * Try to auto-load a snapshot, in order:
 *   1) `#handoff=1` in URL → wait for window.opener.postMessage with the blob
 *   2) `#empty` in URL → skip auto-load (user wants the empty viewer)
 *   3) `/wgdna-default.bin` shipped with the build (for cold landings)
 *   4) Nothing — show the file picker / drop overlay as before
 */
async function autoLoadSnapshot(): Promise<void> {
  if (location.hash.includes('empty')) return;
  if (location.hash.includes('handoff=1') && window.opener) {
    setStatus('Waiting for snapshot from harness…');
    const got = await new Promise<File | null>((resolve) => {
      const t = setTimeout(() => {
        window.removeEventListener('message', handler);
        resolve(null);
      }, 15_000);
      const handler = (e: MessageEvent): void => {
        if (e.data?.type !== 'wgdna-snapshot' || !(e.data.blob instanceof Blob)) return;
        clearTimeout(t);
        window.removeEventListener('message', handler);
        resolve(new File([e.data.blob], 'harness-handoff.bin'));
      };
      window.addEventListener('message', handler);
      try { window.opener.postMessage('splat-ready', '*'); } catch { /* opener gone */ }
    });
    if (got) { await loadFile(got); return; }
    setStatus('No handoff received. Trying default snapshot…');
  }

  // Cold landing: try the shipped default. 404 is silently OK.
  await loadBundledDemo();
}

// Bootstrap.
initGPU().then((ok) => { if (ok) { frame(); autoLoadSnapshot(); } });
