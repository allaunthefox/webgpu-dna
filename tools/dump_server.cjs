#!/usr/bin/env node
/**
 * Static file server for public/ + POST /dump/<name> to save binary blobs.
 * Used to dump rad_buf from the browser to disk for offline IRT chemistry.
 *
 * Usage: node tools/dump_server.js [port] [public_dir] [dump_dir]
 *   port:       default 8765
 *   public_dir: default ./public
 *   dump_dir:   default ./dumps
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT       = parseInt(process.argv[2] || '8765', 10);
const PUBLIC_DIR = path.resolve(process.argv[3] || 'public');
const DUMP_DIR   = path.resolve(process.argv[4] || 'dumps');

fs.mkdirSync(DUMP_DIR, { recursive: true });

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.ico':  'image/x-icon',
  '.bin':  'application/octet-stream',
};

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);

  // POST /dump/<name>  → save body to dumps/<name>
  if (req.method === 'POST' && url.pathname.startsWith('/dump/')) {
    const name = url.pathname.slice('/dump/'.length).replace(/[^a-zA-Z0-9._-]/g, '_');
    if (!name) { res.writeHead(400); return res.end('bad name'); }
    const fp = path.join(DUMP_DIR, name);
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => {
      const buf = Buffer.concat(chunks);
      fs.writeFileSync(fp, buf);
      console.log(`[dump] ${name}: ${buf.length} bytes (${(buf.length/1e6).toFixed(2)} MB)`);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, path: fp, bytes: buf.length }));
    });
    req.on('error', e => { res.writeHead(500); res.end(String(e)); });
    return;
  }

  // GET /list → list dumps
  if (req.method === 'GET' && url.pathname === '/list') {
    const files = fs.readdirSync(DUMP_DIR).map(f => {
      const st = fs.statSync(path.join(DUMP_DIR, f));
      return { name: f, bytes: st.size, mtime: st.mtime };
    });
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(files));
  }

  // GET static
  let p = url.pathname === '/' ? '/geant4dna.html' : url.pathname;
  const fp = path.join(PUBLIC_DIR, p);
  if (!fp.startsWith(PUBLIC_DIR)) { res.writeHead(403); return res.end('nope'); }
  fs.stat(fp, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); return res.end('not found'); }
    const ext = path.extname(fp).toLowerCase();
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': 'no-store',
      'Access-Control-Allow-Origin': '*',
    });
    fs.createReadStream(fp).pipe(res);
  });
});

server.listen(PORT, () => {
  console.log(`[server] http://localhost:${PORT}  static=${PUBLIC_DIR}  dumps=${DUMP_DIR}`);
});
