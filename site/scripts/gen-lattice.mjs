#!/usr/bin/env node
// =============================================================================
// gen-lattice.mjs — vendor the LatticeDB wasm build into static/lattice/
// -----------------------------------------------------------------------------
// SSOT: the pinned v0.3.3 release of Avarok-Cybersecurity/lattice-db.
//   The loader (lattice_server.js) references exactly one sibling asset,
//   `lattice_server_bg.wasm` (verified against the release tarball — there is
//   no wasm-bindgen snippets/ dir), so those two files are the complete set.
//
// Regenerate with:   node site/scripts/gen-lattice.mjs
//
// Every file is verified against a hardcoded sha256 pin: a release asset that
// changes under a tag is a supply-chain event and MUST fail the build loudly.
// Downloads are cached in scripts/.cache/lattice/ so repeat builds are
// offline-capable; no cache + no network is a hard failure (this generator is
// in vite.config.js's hard-fail group).
// =============================================================================

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const VERSION = 'v0.3.3';
const BASE_URL = `https://github.com/Avarok-Cybersecurity/lattice-db/releases/download/${VERSION}`;

// sha256 pins computed from the v0.3.3 release assets (identical bytes ship
// both as standalone assets and inside lattice-db-wasm.tar.gz).
const ASSETS = [
  {
    name: 'lattice_server.js',
    sha256: 'a9f45e938d5483f7a0bf70044aa1610eddfaa0d9da5d8836e219ce1ebfaa1f9a',
    bytes: 31783
  },
  {
    name: 'lattice_server_bg.wasm',
    sha256: '75a7fc9a6e010cd17111be8bbe5d4d173cfffbdff9a72053d3128cfb0d21cd3b',
    bytes: 763172
  }
];

const here = dirname(fileURLToPath(import.meta.url));
const CACHE_DIR = resolve(here, '.cache', 'lattice');
const OUT_DIR = resolve(here, '..', 'static', 'lattice');

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

async function fetchAsset(name) {
  const url = `${BASE_URL}/${name}`;
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  return Buffer.from(await res.arrayBuffer());
}

function verify(buf, asset, origin) {
  const digest = sha256(buf);
  if (digest !== asset.sha256) {
    console.error(
      `gen-lattice: sha256 MISMATCH for ${asset.name} (${origin})\n` +
        `  expected ${asset.sha256}\n` +
        `  actual   ${digest}\n` +
        `A changed release asset under the pinned tag ${VERSION} is a supply-chain ` +
        `event — refusing to vendor it.`
    );
    process.exit(1);
  }
  if (buf.length !== asset.bytes) {
    console.error(
      `gen-lattice: size mismatch for ${asset.name} (${origin}): ` +
        `expected ${asset.bytes} B, got ${buf.length} B`
    );
    process.exit(1);
  }
}

mkdirSync(CACHE_DIR, { recursive: true });
mkdirSync(OUT_DIR, { recursive: true });

let downloaded = 0;
let fromCache = 0;

for (const asset of ASSETS) {
  const cachePath = resolve(CACHE_DIR, asset.name);
  const outPath = resolve(OUT_DIR, asset.name);

  let buf = null;
  if (existsSync(cachePath)) {
    const cached = readFileSync(cachePath);
    if (sha256(cached) === asset.sha256) {
      buf = cached;
      fromCache++;
    } else {
      // Corrupt/stale cache entry: treat as absent and re-download.
      console.error(`gen-lattice: cached ${asset.name} failed its pin — re-downloading`);
    }
  }

  if (buf === null) {
    try {
      buf = await fetchAsset(asset.name);
    } catch (err) {
      console.error(
        `gen-lattice: cannot vendor ${asset.name}: no valid cache and download failed ` +
          `(${err && err.message ? err.message : err}). SSOT: ${BASE_URL}/${asset.name}`
      );
      process.exit(1);
    }
    verify(buf, asset, 'download');
    writeFileSync(cachePath, buf);
    downloaded++;
  } else {
    verify(buf, asset, 'cache');
  }

  writeFileSync(outPath, buf);
}

console.log(
  `Wrote ${OUT_DIR}\n  ${ASSETS.length} LatticeDB ${VERSION} assets ` +
    `(${fromCache} from cache, ${downloaded} downloaded), all sha256-pinned`
);
for (const a of ASSETS) console.log(`  - ${a.name} (${a.bytes} B) ${a.sha256.slice(0, 12)}…`);
