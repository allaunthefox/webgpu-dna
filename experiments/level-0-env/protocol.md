# Level 0 — Environment / infrastructure sanity

## Status: in progress
B0 (browser-runner sanity check) implemented and passing. This level
isn't a "research" level in the falsifiable-physics sense — it's a
reproducibility floor. Future GPU experiments will reference its
adapter capture to verify they ran on a compatible machine.

## Thesis fragment
> The browser-runner infrastructure (Playwright + headless Chromium
> with --headless=new + --enable-unsafe-webgpu) exposes a working
> WebGPU adapter with limits sufficient for webgpu-dna's resource
> needs (rad_buf = 256 MB, etc).

## Experiments

### B0 — Browser env capture
- **Status:** **Implemented; passing.** First artifact:
  `experiments/results/<date>/level-0/B0-browser-env.json`.
- **Hypothesis:** Headless Chromium with the configured flags exposes
  navigator.gpu in a secure-context page; the adapter's
  maxBufferSize and maxStorageBufferBindingSize are both ≥ 256 MB
  (webgpu-dna's rad_buf allocation).
- **Method:** `experiments/lib/browser.mjs` launches Chromium, navigates
  to `https://webgpureport.org/` for a secure context, evaluates
  `await navigator.gpu.requestAdapter()` and serializes the info /
  limits / userAgent / platform.
- **Pass bar:** WebGPU available AND maxBufferSize ≥ 256 MB AND
  maxStorageBufferBindingSize ≥ 256 MB.
- **Why webgpureport.org:** WebGPU requires a secure context (HTTPS
  or localhost). `setContent` and `about:blank` don't qualify and
  navigator.gpu is missing there. webgpureport.org is a public HTTPS
  page; we only call `navigator.gpu` — none of webgpureport's own JS
  is touched. Future experiments that drive webgpu-dna's harness will
  navigate to a localhost vite dev server instead.

### B1 — webgpu-dna harness liveness
- **Status:** **Implemented; passing.** First artifact:
  `experiments/results/2026-05-08/level-0/B1-harness-liveness.json`.
- **Headline:** first row (E=100 eV) appears in 4.4s; CSDA=15.7 nm.
- **Hypothesis:** Spawning the vite dev server and driving the
  webgpu-dna harness in headless Chromium produces the first energy
  result in `#tb` within 90 seconds at N=1024 primaries (the harness
  HTML's `min` bound). The CSDA in that row is plausible (1-50000 nm).
- **Method:** `experiments/lib/dev-server.mjs` spawns `npm run dev`
  and waits for the "ready" log line. Playwright navigates to
  `http://localhost:8765`, sets `#np` to 1024, clicks `#run`, and
  waits via `page.waitForFunction` for `#tb tr:first-child td:nth-child(4)`
  (the CSDA cell per the harness's table layout) to be populated.
- **Pass bar:** first row appears within 90s AND CSDA cell parses to
  a number in [1, 50000] nm AND no page-level errors during the run.
- **Why this matters:** prerequisite for E11 (GPU vs IRT chemistry)
  and any other browser-driven physics experiment. The 15.7 nm CSDA
  at E=100 eV matches expected ESTAR low-energy CSDA — the harness
  ran real WebGPU physics, not just a stub.

### Future B2 — full 8-energy validation snapshot
- **Status:** deferred.
- **Hypothesis:** All 8 ESTAR energies' results appear within 5 minutes
  at N=4096 and the captured numbers match `validation/webgpu-results.json`
  (the static reference) within MC noise. Promotes the manual
  "paste-from-browser-into-compare.py" workflow to a fully-automated
  research-grade run.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-0/B<k>-<slug>.json`. Same
shape as research-grade levels — `meta / env / status / diagnosis /
summary / rows` — even though the content is environmental rather
than physics-falsifiable.
