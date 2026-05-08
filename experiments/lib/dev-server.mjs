// Vite dev-server lifecycle helper for browser-runner experiments.
//
// Spawns `npm run dev` (vite on http://localhost:8765 per the project's
// vite.config.ts), waits for the "ready" log line, and returns a
// disposer that kills the process group on cleanup.
//
// Usage:
//   const server = await startDevServer();
//   try {
//     // navigate Playwright to server.url, run experiment
//   } finally {
//     await server.stop();
//   }

import { spawn } from 'node:child_process';
import { join } from 'node:path';

const REPO_ROOT = join(import.meta.dirname, '..', '..');

// Vite typically picks 8765 per vite.config.ts; if taken, it'll bump.
// We capture the actual URL from the ready log line.
const VITE_READY_REGEX = /Local:\s+(https?:\/\/\S+?)\/?\s*$/im;
const VITE_READY_FALLBACK_REGEX = /ready in/i;

export async function startDevServer({ readyTimeoutMs = 60_000 } = {}) {
  return new Promise((resolve, reject) => {
    const proc = spawn('npm', ['run', 'dev'], {
      cwd: REPO_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, BROWSER: 'none', CI: '1' },
      detached: true, // own process group so we can kill children too
    });

    let stdout = '';
    let stderr = '';
    let resolved = false;
    let resolvedUrl = null;

    const timer = setTimeout(() => {
      if (resolved) return;
      resolved = true;
      stop();
      reject(new Error(`vite dev server did not signal ready within ${readyTimeoutMs}ms.\n--- stdout ---\n${stdout}\n--- stderr ---\n${stderr}`));
    }, readyTimeoutMs);

    function maybeReady(text) {
      if (resolved) return;
      const m = text.match(VITE_READY_REGEX);
      if (m) {
        resolvedUrl = m[1];
        finish();
        return;
      }
      // Fallback: "ready in Xms" appears slightly before the Local: URL on some vite versions.
      // Wait a beat, then assume default.
      if (VITE_READY_FALLBACK_REGEX.test(text) && !resolvedUrl) {
        setTimeout(() => {
          if (!resolved) {
            resolvedUrl = 'http://localhost:8765';
            finish();
          }
        }, 300);
      }
    }

    function finish() {
      resolved = true;
      clearTimeout(timer);
      resolve({
        url: resolvedUrl,
        process: proc,
        stdout: () => stdout,
        stderr: () => stderr,
        stop,
      });
    }

    function stop() {
      try {
        // Kill the whole process group (-pid) — vite spawns child workers.
        if (proc.pid) process.kill(-proc.pid, 'SIGTERM');
      } catch {
        try { proc.kill('SIGTERM'); } catch { /* nothing */ }
      }
    }

    proc.stdout.on('data', (b) => {
      const s = b.toString();
      stdout += s;
      maybeReady(s);
    });
    proc.stderr.on('data', (b) => {
      const s = b.toString();
      stderr += s;
      maybeReady(s);
    });
    proc.on('error', (err) => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timer);
      reject(err);
    });
    proc.on('exit', (code) => {
      if (resolved) return;
      resolved = true;
      clearTimeout(timer);
      reject(new Error(`vite dev server exited prematurely (code=${code}).\n--- stdout ---\n${stdout}\n--- stderr ---\n${stderr}`));
    });
  });
}
