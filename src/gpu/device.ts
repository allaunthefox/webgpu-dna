/**
 * WebGPU device initialization.
 *
 * WebGPU default caps `maxStorageBufferBindingSize` at 128 MiB which is
 * too small for our 192–256 MB radical/chem_pos buffers. We explicitly
 * request the adapter's max via `requiredLimits`.
 */

import type { LogFn } from '../physics/types';

export async function initGPU(log?: LogFn): Promise<GPUDevice | null> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    log?.('WebGPU not available in this browser.', 'err');
    return null;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    log?.('No GPU adapter found.', 'err');
    return null;
  }

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });

  const info = (adapter as unknown as { info?: { description?: string } }).info;
  log?.(`GPU initialized: ${info?.description ?? 'default adapter'}`, 'ok');
  return device;
}
