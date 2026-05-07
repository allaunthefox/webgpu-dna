// Environment capture for experiment artifacts. Mirrors webgpu-q's `env`
// block: git SHA, timestamp, platform, plus Node-runtime metadata. GPU
// experiments (Level 2+) extend this with adapter info from
// `navigator.gpu.requestAdapter()` when run in a browser.

import { execSync } from 'node:child_process';
import { hostname, platform, arch, cpus, totalmem } from 'node:os';

function safeGitSha() {
  try {
    return execSync('git rev-parse HEAD', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return 'dev-unknown';
  }
}

function safeGitDirty() {
  try {
    const out = execSync('git status --porcelain', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
    return out.trim().length > 0;
  } catch {
    return null;
  }
}

export function captureEnv() {
  return {
    gitSha: safeGitSha(),
    gitDirty: safeGitDirty(),
    timestamp: new Date().toISOString(),
    runner: 'node',
    nodeVersion: process.version,
    platform: platform(),
    arch: arch(),
    hostname: hostname(),
    cpuCount: cpus().length,
    totalMemBytes: totalmem(),
  };
}
