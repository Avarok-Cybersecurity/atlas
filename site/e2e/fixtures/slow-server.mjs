// =============================================================================
// slow-server.mjs — a local HTTP server that streams the fixture corpus .gz in
// small delayed chunks. Playwright's route.fulfill delivers a body atomically,
// which makes the 'downloading' phase invisible for a small fixture; tests
// that need a *visible, interruptible* download 302-redirect the corpus URL
// here instead. Sends CORS headers (the page fetches cross-origin).
// =============================================================================

import { createServer } from 'node:http';

/**
 * @param {Buffer} body bytes to stream
 * @param {{chunkSize?: number, delayMs?: number}} opts
 * @returns {Promise<{url: string, close: () => Promise<void>}>}
 */
export function startSlowServer(body, { chunkSize = 512, delayMs = 60 } = {}) {
  const server = createServer((req, res) => {
    res.writeHead(200, {
      'content-type': 'application/gzip',
      'content-length': String(body.byteLength),
      'access-control-allow-origin': '*'
    });
    let offset = 0;
    let closed = false;
    res.on('close', () => { closed = true; });
    const tick = () => {
      if (closed) return;
      if (offset >= body.byteLength) { res.end(); return; }
      res.write(body.subarray(offset, offset + chunkSize));
      offset += chunkSize;
      setTimeout(tick, delayMs);
    };
    tick();
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({
        url: `http://127.0.0.1:${port}/corpus.jsonl.gz`,
        close: () => new Promise((r) => { server.closeAllConnections?.(); server.close(() => r()); })
      });
    });
  });
}
