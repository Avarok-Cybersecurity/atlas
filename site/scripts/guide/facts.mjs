// SPDX-License-Identifier: AGPL-3.0-only
//
// What the site guide tracks, in one place, so the generator and the test that
// keeps it honest cannot disagree about it.
//
// A TRACKED FACT is a value that appears on the site and will one day change
// without anyone touching the design: a name, an address, a link, a licence
// line. Each gets a revision and a date in guide/ledger.json. Change the value
// and the unit suite fails until `bun run guide` records the change, which is
// how "we changed Eric's address in three places and missed the fourth" stops
// happening: there is one place, and the ledger says when it last moved.
//
// A TRACKED ASSET is a file a visitor is served that came from somewhere: a
// logo, a portrait, a clip, a font, a brand master. Its hash is tracked the same
// way, so a swapped file is a recorded event, not a surprise.

import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
export const SITE_DIR = resolve(here, '..', '..');
export const REPO_DIR = resolve(SITE_DIR, '..');
export const LEDGER = join(SITE_DIR, 'guide', 'ledger.json');

const posix = (p) => p.split(sep).join('/');

/** Everything brand.js says that is a name, an address, a link or a licence line. */
export async function collectFacts(brandFile = join(SITE_DIR, 'src', 'lib', 'content', 'brand.js')) {
  const brand = await import(pathToFileURL(brandFile).href + `?t=${statSync(brandFile).mtimeMs}`);
  const facts = {};
  facts['site.origin'] = brand.SITE;
  for (const [k, v] of Object.entries(brand.company)) facts[`company.${k}`] = v;
  for (const [k, v] of Object.entries(brand.links)) facts[`links.${k}`] = v;
  for (const [k, v] of Object.entries(brand.contacts)) facts[`contacts.${k}`] = v;
  facts['footer.slogan'] = brand.footer.slogan;
  facts['footer.legal'] = brand.footer.legal;
  facts['footer.license'] = brand.footer.license;
  facts['footer.mlperf'] = brand.footer.mlperf;
  facts['form.endpoint'] = brand.formEndpoint || '(none: forms draft an email)';
  return facts;
}

/** Where tracked assets live, relative to the repository root. */
export const ASSET_ROOTS = ['site/static/logos', 'site/static/team', 'site/static/media', 'site/static/fonts', 'site/static/brand', 'assets/brand'];
/** Single files worth tracking that sit directly in static/. */
export const ASSET_FILES = ['site/static/favicon.svg', 'site/static/favicon.ico', 'site/static/og-image.png', 'site/static/apple-touch-icon.png', 'site/static/nvidia-inception.webp', 'blog/static/og-image.png'];

/**
 * The bytes that are hashed. A text asset is hashed with Unix line endings, whatever
 * the checkout has: git on Windows hands out CRLF, CI on Linux gets LF, and the
 * same SVG must not have two hashes depending on who ran the generator.
 */
const TEXT = /\.(svg|css|txt|html|xml|js|mjs)$/i;
export const canonical = (file, buf) => (TEXT.test(file) ? Buffer.from(buf.toString('utf8').replace(/\r\n/g, '\n'), 'utf8') : buf);

const walk = (dir) => (existsSync(dir) ? readdirSync(dir).flatMap((f) => (statSync(join(dir, f)).isDirectory() ? walk(join(dir, f)) : [join(dir, f)])) : []);

/** path -> { sha, bytes } for every tracked file. The sha is the first 12 hex of sha256. */
export function collectAssets() {
  const files = [...ASSET_ROOTS.flatMap((r) => walk(join(REPO_DIR, r))), ...ASSET_FILES.map((f) => join(REPO_DIR, f)).filter(existsSync)];
  const out = {};
  for (const file of files.sort()) {
    if (/\.(md|json|txt)$/i.test(file) && !/LICENSE/i.test(file)) continue; // notes about assets are not assets
    const buf = canonical(file, readFileSync(file));
    out[posix(relative(REPO_DIR, file))] = { sha: createHash('sha256').update(buf).digest('hex').slice(0, 12), bytes: buf.length };
  }
  return out;
}

export const readLedger = () => (existsSync(LEDGER) ? JSON.parse(readFileSync(LEDGER, 'utf8')) : { started: null, changes: [], facts: {}, assets: {} });

/** What differs between the ledger and the tree. Empty arrays mean the ledger is current. */
export function diffLedger(ledger, facts, assets) {
  const live = (o) => Object.fromEntries(Object.entries(o).filter(([, v]) => !v.removed));
  const changedFacts = Object.keys(facts).filter((k) => ledger.facts[k]?.removed || ledger.facts[k]?.value !== facts[k]);
  const removedFacts = Object.keys(live(ledger.facts)).filter((k) => !(k in facts));
  const changedAssets = Object.keys(assets).filter((k) => ledger.assets[k]?.removed || ledger.assets[k]?.sha !== assets[k].sha);
  const removedAssets = Object.keys(live(ledger.assets)).filter((k) => !(k in assets));
  return { changedFacts, removedFacts, changedAssets, removedAssets };
}
