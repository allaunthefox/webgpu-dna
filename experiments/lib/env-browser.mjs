// Browser-runner env capture. Extends the Node-side captureEnv() with
// a `gpu` block from a real headless Chromium WebGPU adapter, matching
// the shape webgpu-q's experiment artifacts use:
//
//   env: {
//     gitSha, timestamp, runner: 'playwright-chromium',
//     userAgent, platform,
//     adapter: { vendor, architecture, device, description },
//     limits: { maxBufferSize, ... },
//   }
//
// The Node-side captureEnv (env.mjs) is the base; this just merges the
// browser GPU block on top.

import { captureEnv } from './env.mjs';
import { captureAdapter } from './browser.mjs';

export async function captureBrowserEnv() {
  const node = captureEnv();
  const adapter = await captureAdapter();
  if (!adapter || !adapter.available) {
    return {
      ...node,
      runner: 'playwright-chromium',
      gpu: { available: false, reason: adapter?.reason ?? 'unknown' },
    };
  }
  return {
    ...node,
    runner: 'playwright-chromium',
    userAgent: adapter.userAgent,
    platform: adapter.platform,
    adapter: adapter.info,
    limits: adapter.limits,
  };
}
