#!/usr/bin/env node
// Quick validation tool for the browser-runner infrastructure.
//
// Launches headless Chromium with WebGPU flags, requests an adapter,
// and prints adapter info + key limits. Used to verify the Playwright
// + WebGPU pipeline is functional before running real experiments.
//
// Run:  node experiments/tools/check-browser.mjs
//
// Exits 0 if WebGPU is available, 1 otherwise.

import { captureAdapter } from '../lib/browser.mjs';

async function main() {
  console.log('[check-browser] launching headless Chromium with --enable-unsafe-webgpu ...');
  const t0 = Date.now();
  const adapter = await captureAdapter();
  const dt = ((Date.now() - t0) / 1000).toFixed(2);

  if (!adapter || !adapter.available) {
    console.error(`[check-browser] ✗ WebGPU NOT available: ${adapter?.reason ?? 'unknown'}`);
    console.error(`[check-browser] (took ${dt}s)`);
    process.exit(1);
  }

  console.log(`[check-browser] ✓ WebGPU available (took ${dt}s)`);
  console.log('');
  console.log('Adapter info:');
  for (const [k, v] of Object.entries(adapter.info)) {
    console.log(`  ${k.padEnd(13)} ${v || '(empty)'}`);
  }
  console.log('');
  console.log(`User agent: ${adapter.userAgent}`);
  console.log('');
  console.log('Key WebGPU limits (subset):');
  const keyLimits = [
    'maxBufferSize',
    'maxStorageBufferBindingSize',
    'maxComputeWorkgroupsPerDimension',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupStorageSize',
  ];
  for (const k of keyLimits) {
    const v = adapter.limits[k];
    console.log(`  ${k.padEnd(36)} ${v?.toLocaleString() ?? '(missing)'}`);
  }
}

main().catch((err) => {
  console.error('[check-browser] ✗ unexpected error:', err.message);
  process.exit(2);
});
