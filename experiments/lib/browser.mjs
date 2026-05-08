// Playwright wrapper for browser-runner experiments.
//
// WebGPU + headless Chromium gotcha: navigator.gpu is only exposed in a
// **secure context** (HTTPS or localhost). Pages opened via `setContent`
// or `goto('about:blank')` are NOT secure contexts, so navigator.gpu is
// missing there even with --enable-unsafe-webgpu.
//
// The working configuration on macOS Apple Silicon (Playwright 1.59):
//   - bundled Chromium (Playwright's default; no `channel: 'chrome'`)
//   - `headless: false` paired with `--headless=new` arg (Chrome's new
//     headless mode; legacy `headless: true` doesn't expose WebGPU)
//   - `--enable-unsafe-webgpu`
//   - `--enable-features=Vulkan`
//   - navigate to an HTTPS URL or localhost (do NOT use setContent for
//     pages that need WebGPU — their about:blank context fails the
//     secure-context check)
//
// Usage:
//   const result = await runInBrowser({
//     baseUrl: 'http://localhost:8765',  // or HTTPS / localhost
//     evaluate: async () => {
//       const adapter = await navigator.gpu.requestAdapter();
//       return { vendor: adapter.info.vendor, ... };
//     },
//   });

import { chromium } from 'playwright';

const DEFAULT_ARGS = [
  '--headless=new',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--no-sandbox',
];

// Public page that successfully exposes navigator.gpu. Used by
// captureAdapter() when no baseUrl is provided. The evaluate function
// only reads from `navigator.gpu` — webgpureport.org's own JS isn't
// touched.
const ADAPTER_CAPTURE_DEFAULT_URL = 'https://webgpureport.org/';

export async function runInBrowser({ baseUrl, html, evaluate, args }) {
  if (!baseUrl && !html) {
    throw new Error('runInBrowser: provide baseUrl (preferred for WebGPU) or html');
  }
  const browser = await chromium.launch({
    headless: false,
    args: args ?? DEFAULT_ARGS,
  });
  try {
    const context = await browser.newContext();
    const page = await context.newPage();

    if (baseUrl) {
      await page.goto(baseUrl);
    } else {
      // Note: setContent creates an about:blank-class context which is
      // NOT a secure context — WebGPU will be unavailable here. Use
      // baseUrl with HTTPS or localhost instead.
      await page.setContent(html);
    }

    const result = await page.evaluate(evaluate);
    return result;
  } finally {
    await browser.close();
  }
}

// Convenience: capture WebGPU adapter info + limits in the format used
// by experiment artifact env blocks. By default navigates to
// webgpureport.org (a stable HTTPS page) just so we have a secure context;
// only `navigator.gpu` is touched, the page's own JS is irrelevant.
// Pass `baseUrl` to point at a different secure-context page (e.g. your
// own localhost dev server).
export async function captureAdapter({ baseUrl = ADAPTER_CAPTURE_DEFAULT_URL, args } = {}) {
  return runInBrowser({
    baseUrl,
    args,
    evaluate: async () => {
      if (!navigator.gpu) return { available: false, reason: 'navigator.gpu missing (secure-context issue?)' };
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { available: false, reason: 'requestAdapter returned null' };
      const info = adapter.info ?? {};
      const limits = {};
      for (const key in adapter.limits) {
        const v = adapter.limits[key];
        if (typeof v === 'number') limits[key] = v;
      }
      return {
        available: true,
        info: {
          vendor: info.vendor ?? '',
          architecture: info.architecture ?? '',
          device: info.device ?? '',
          description: info.description ?? '',
        },
        limits,
        userAgent: navigator.userAgent,
        platform: navigator.platform,
      };
    },
  });
}
