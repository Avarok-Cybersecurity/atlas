// SPDX-License-Identifier: AGPL-3.0-only
//
// static/_headers is Cloudflare Pages' cache policy, and it names document
// routes one by one. Its own header explains why: Pages concatenates a header
// that two matching rules both set, so a blanket rule is not available. That
// makes it a file a new page can silently miss. The facelift added thirty
// pages and not one had a rule until this test existed.
//
// Three things are held here: every page has a cache rule, no path is matched
// by two Cache-Control rules, and the file stays inside Pages' limit.
import { expect, test } from 'bun:test';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { pages } from './index.js';

const SITE_DIR = join(import.meta.dir, '..', '..', '..');
const STATIC_DIR = join(SITE_DIR, 'static');

/** [{ path, headers: { name: value } }] in file order. */
function parse(text) {
  const rules = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith('#')) continue;
    if (!/^\s/.test(line)) rules.push({ path: line.trim(), headers: {} });
    else if (rules.length) {
      const i = line.indexOf(':');
      rules.at(-1).headers[line.slice(0, i).trim().toLowerCase()] = line.slice(i + 1).trim();
    }
  }
  return rules;
}

const rules = parse(readFileSync(join(STATIC_DIR, '_headers'), 'utf8'));
const cacheRules = rules.filter((r) => 'cache-control' in r.headers);
const escape = (s) => s.replace(/[.+?^${}()|[\]\\]/g, '\\$&');
const matcher = (rule) => new RegExp('^' + rule.split('*').map(escape).join('.*') + '$');
const matching = (path) => cacheRules.filter((r) => matcher(r.path).test(path));

const walk = (dir) =>
  readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });

// Both spellings: Pages serves pricing.html at /pricing, and a link can ask for either.
const documents = [...pages.map((p) => p.path), '/engine', '/control', '/diligence'].flatMap((p) => (p === '/' ? ['/', '/index.html'] : [p, `${p}.html`]));

test('every page has exactly one cache rule, in both spellings', () => {
  // The 404 document is served for paths that match nothing, so it is never requested by its own name.
  const problems = documents.filter((d) => !d.startsWith('/404') && matching(d).length !== 1).map((d) => `${d}: ${matching(d).length} rules`);
  expect(problems).toEqual([]);
});

test('no shipped file is matched by two Cache-Control rules, which Pages would concatenate', () => {
  const files = walk(STATIC_DIR)
    .map((f) => '/' + relative(STATIC_DIR, f).split(/[\\/]/).join('/'))
    .filter((f) => !f.startsWith('/lattice/') || f.endsWith('.wasm'));
  const doubled = files.filter((f) => matching(f).length > 1).map((f) => `${f}: ${matching(f).map((r) => r.path).join(' and ')}`);
  expect(doubled).toEqual([]);
});

test('the media and the fonts are cached', () => {
  for (const f of ['/media/console-ask.mp4', '/media/console-ask.webm', '/media/console-ask.webp', '/media/art/art-finance.webp', '/fonts/ibm-plex-sans-latin-400-normal.woff2', '/fonts/plex.css']) {
    expect(matching(f).length, f).toBe(1);
  }
});

test('the file stays inside the 100 rule limit Cloudflare Pages enforces', () => {
  expect(rules.length).toBeLessThanOrEqual(100);
  // Room for the next few pages. If this trips, reach for a splat before a name.
  expect(rules.length).toBeLessThan(90);
});
