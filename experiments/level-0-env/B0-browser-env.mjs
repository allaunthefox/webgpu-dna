// B0 — Browser env capture (browser-runner sanity check).
//
// First experiment to actually use the browser-runner infrastructure
// (experiments/lib/browser.mjs). Captures the test machine's WebGPU
// adapter info + limits to a JSON artifact, matching the env-block
// shape webgpu-q uses (vendor / architecture / device / description /
// maxBufferSize / etc).
//
// Why this is a "B0" not a research-grade pass/fail: this is a
// per-machine snapshot, not a falsifiable physics claim. It serves as
// (a) sanity proof that the browser-runner pipeline is wired
// end-to-end, and (b) reference adapter info that future GPU
// experiments will compare their own captured env against (catches
// "this experiment was run on a different GPU than the protocol
// assumes" cases).
//
// Pass condition: WebGPU is available AND maxBufferSize is large
// enough for webgpu-dna's rad_buf (256 MB).

import { SEEDS } from '../lib/seeds.mjs';
import { captureBrowserEnv } from '../lib/env-browser.mjs';

const RAD_BUF_BYTES = 256 * 1024 * 1024; // webgpu-dna's MAX_RAD allocation

export async function runB0() {
  const env = await captureBrowserEnv();

  const failures = [];
  if (!env.adapter) {
    failures.push('no WebGPU adapter (browser-runner infrastructure not functional)');
  } else {
    if ((env.limits?.maxBufferSize ?? 0) < RAD_BUF_BYTES) {
      failures.push(
        `maxBufferSize ${env.limits?.maxBufferSize ?? 0} < ${RAD_BUF_BYTES} (rad_buf would not fit)`,
      );
    }
    if ((env.limits?.maxStorageBufferBindingSize ?? 0) < RAD_BUF_BYTES) {
      failures.push(
        `maxStorageBufferBindingSize ${env.limits?.maxStorageBufferBindingSize ?? 0} < ${RAD_BUF_BYTES}`,
      );
    }
  }

  const status = failures.length === 0 ? 'pass' : 'fail';
  const diagnosis = failures.length === 0 ? null : failures.join('; ');

  return {
    meta: {
      protocol: 'B0-browser-env',
      hypothesis:
        'Headless Chromium with --headless=new + --enable-unsafe-webgpu exposes navigator.gpu when the page is in a secure context (webgpureport.org), and the adapter has limits sufficient for webgpu-dna (rad_buf = 256 MB).',
      passBar: 'navigator.gpu available AND maxBufferSize ≥ 256 MB AND maxStorageBufferBindingSize ≥ 256 MB.',
      seed: 'n/a (deterministic env query, no MC)',
      warmup: 0,
      trials: 1,
      sources: {
        browserLauncher: 'experiments/lib/browser.mjs (Playwright + Chromium)',
        captureUrl: 'https://webgpureport.org/ (default; secure-context HTTPS page used for navigator.gpu access)',
      },
    },
    env,
    status,
    diagnosis,
    summary: {
      adapterVendor: env.adapter?.vendor ?? null,
      adapterArchitecture: env.adapter?.architecture ?? null,
      maxBufferBytes: env.limits?.maxBufferSize ?? null,
      maxStorageBufferBindingBytes: env.limits?.maxStorageBufferBindingSize ?? null,
      maxComputeWorkgroupsPerDimension: env.limits?.maxComputeWorkgroupsPerDimension ?? null,
      maxComputeInvocationsPerWorkgroup: env.limits?.maxComputeInvocationsPerWorkgroup ?? null,
      headline: env.adapter
        ? `${env.adapter.vendor || 'unknown'}/${env.adapter.architecture || '?'} | maxBuf=${((env.limits?.maxBufferSize ?? 0) / 1024 / 1024).toFixed(0)}MB`
        : 'no adapter',
    },
    rows: [],
  };
}
