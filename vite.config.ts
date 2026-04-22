import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    target: 'esnext',
  },
  assetsInclude: ['**/*.wgsl'],
  server: {
    port: 8765,
    host: '127.0.0.1',
  },
});
