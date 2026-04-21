/**
 * Minimal DOM logger factory. Returns an `lg(msg, cls?)` function that writes
 * timestamped lines into `#log` and scrolls to the bottom.
 *
 * If no element is found, falls back to console.log so tests / headless
 * environments still work.
 */

import type { LogFn } from '../physics/types';

export function createLogger(elementId: string = 'log'): LogFn {
  return (msg: string, cls: string = '') => {
    const d = typeof document !== 'undefined' ? document.getElementById(elementId) : null;
    if (!d) {
      console.log(`[${cls || 'info'}] ${msg}`);
      return;
    }
    const p = document.createElement('div');
    if (cls) p.className = cls;
    const t = new Date().toTimeString().slice(0, 8);
    p.textContent = `[${t}] ${msg}`;
    d.appendChild(p);
    d.scrollTop = d.scrollHeight;
  };
}
