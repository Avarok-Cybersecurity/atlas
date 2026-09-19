// =============================================================================
// serve.mjs — a static file server over the built site, for the recorders.
// -----------------------------------------------------------------------------
// The site is prerendered with adapter-static, so a route is a file: /pricing
// is build/pricing.html, and /platform is build/platform.html even though a
// build/platform/ directory of child pages sits beside it. This resolver tries
// the path, then path.html, then path/index.html, which is what Cloudflare
// Pages does in production. Node builtins only; no third-party deps.
// =============================================================================

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.txt': 'text/plain; charset=utf-8',
  '.xml': 'application/xml',
  '.webmanifest': 'application/manifest+json',
  '.wasm': 'application/wasm',
  '.gz': 'application/gzip'
};

async function isFile(p) {
  try {
    return (await stat(p)).isFile();
  } catch {
    return false;
  }
}

/**
 * Serve `root` on a free loopback port.
 * @param {string} root absolute path of the build directory
 * @returns {Promise<{ port: number, origin: string, close: () => Promise<void> }>}
 */
export function serve(root) {
  const base = resolve(root);
  const server = createServer(async (req, res) => {
    const url = new URL(req.url ?? '/', 'http://localhost');
    let pathname;
    try {
      pathname = decodeURIComponent(url.pathname);
    } catch {
      res.writeHead(400).end();
      return;
    }
    const target = resolve(join(base, normalize(pathname)));
    if (target !== base && !target.startsWith(base + sep)) {
      res.writeHead(403).end();
      return;
    }
    const candidates = [target, `${target}.html`, join(target, 'index.html')];
    for (const c of candidates) {
      if (await isFile(c)) {
        const body = await readFile(c);
        res.writeHead(200, { 'content-type': MIME[extname(c)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
        res.end(body);
        return;
      }
    }
    const notFound = join(base, '404.html');
    if (await isFile(notFound)) {
      res.writeHead(404, { 'content-type': MIME['.html'] });
      res.end(await readFile(notFound));
      return;
    }
    res.writeHead(404).end();
  });
  return new Promise((ok, fail) => {
    server.once('error', fail);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      ok({ port, origin: `http://127.0.0.1:${port}`, close: () => new Promise((r) => server.close(() => r())) });
    });
  });
}
