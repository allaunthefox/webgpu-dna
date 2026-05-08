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

### Future B1 — webgpu-dna harness liveness
- **Status:** deferred.
- **Hypothesis:** A vite dev server hosting webgpu-dna's `runValidation()`
  reaches the "validation done" state within 60 seconds at N=4096
  primaries / E=10 keV when driven by Playwright.
- **Why this matters:** prerequisite for E11 (GPU vs IRT chemistry)
  and any other browser-driven physics experiment. Catches harness
  startup regressions before they break downstream.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-0/B<k>-<slug>.json`. Same
shape as research-grade levels — `meta / env / status / diagnosis /
summary / rows` — even though the content is environmental rather
than physics-falsifiable.
