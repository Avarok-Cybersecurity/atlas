// SPDX-License-Identifier: AGPL-3.0-only
//
// The marketing site is data: a page registry, a nav tree, and copy modules
// that pages render. That makes the ways it breaks checkable without a
// browser. A link to a page that was renamed, a page with no registry entry, a
// video slot whose file is a placeholder, a sentence in the wrong voice: each
// of these shipped silently at least once while this site was being built.
import { expect, test } from 'bun:test';
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { pages, routes, industries, solutionHref } from './index.js';
import { nav, footer } from './brand.js';
import { allClips, media } from './media.js';
import * as home from './home.js';
import * as why from './why.js';
import * as platform from './platform.js';
import * as solutions from './solutions.js';
import * as pricing from './pricing.js';
import * as company from './company.js';
import * as resources from './resources.js';
import * as faq from './faq.js';

const SITE_DIR = join(import.meta.dir, '..', '..', '..');
const ROUTES_DIR = join(SITE_DIR, 'src', 'routes');
const STATIC_DIR = join(SITE_DIR, 'static');

// The developer pages keep their own heads and are not in the registry.
const ENGINE_ROUTES = ['/engine', '/control', '/diligence'];
const registry = new Set(pages.map((p) => p.path));

const walk = (dir) =>
  readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });

/** '/platform/engine' for src/routes/(marketing)/platform/engine/+page.svelte */
function routePath(file) {
  const parts = file
    .slice(ROUTES_DIR.length)
    .split(/[\\/]/)
    .filter((s) => s && !s.startsWith('(') && !s.startsWith('+'));
  return '/' + parts.join('/');
}

const pageFiles = walk(ROUTES_DIR).filter((f) => f.endsWith('+page.svelte'));
const staticRoutes = pageFiles.map(routePath).filter((p) => !p.includes('['));

// --- the registry and the routes agree ---------------------------------------

test('every page in the registry has a route', () => {
  const missing = pages
    .map((p) => p.path)
    .filter((path) => !staticRoutes.includes(path) && !(path.startsWith('/solutions/') && industries.some((i) => solutionHref(i.slug) === path)));
  expect(missing).toEqual([]);
});

test('every route outside the developer pages is in the registry', () => {
  const unregistered = staticRoutes.filter((p) => !registry.has(p) && !ENGINE_ROUTES.includes(p));
  expect(unregistered).toEqual([]);
});

test('every industry has a registered solution page', () => {
  expect(industries.map((i) => solutionHref(i.slug)).filter((p) => !registry.has(p))).toEqual([]);
});

test('every named route points at a page that exists', () => {
  const dead = Object.entries(routes)
    .map(([name, href]) => [name, href.split('#')[0]])
    .filter(([, path]) => !registry.has(path) && !ENGINE_ROUTES.includes(path))
    .map(([name, path]) => `${name} -> ${path}`);
  expect(dead).toEqual([]);
});

// A route may name a place on a page, like the booking form at /demo#book. A
// link to an anchor that does not exist does not fail, it lands at the top of
// the page, which reads as a button that did nothing.
test('every named route with an anchor points at an id on that page', () => {
  const lost = Object.entries(routes)
    .filter(([, href]) => href.includes('#'))
    .filter(([, href]) => {
      const [path, id] = href.split('#');
      const file = join(ROUTES_DIR, '(marketing)', ...path.split('/').filter(Boolean), '+page.svelte');
      return !existsSync(file) || !readFileSync(file, 'utf8').includes(`id="${id}"`);
    })
    .map(([name, href]) => `${name} -> ${href}`);
  expect(lost).toEqual([]);
});

test('titles and descriptions are unique and sized for a results page', () => {
  const titles = pages.map((p) => p.title);
  const descriptions = pages.map((p) => p.description);
  expect(new Set(titles).size).toBe(titles.length);
  expect(new Set(descriptions).size).toBe(descriptions.length);
  for (const p of pages) {
    expect(p.title.length, `${p.path} title`).toBeLessThanOrEqual(70);
    expect(p.description.length, `${p.path} description is ${p.description.length}`).toBeGreaterThanOrEqual(50);
    expect(p.description.length, `${p.path} description is ${p.description.length}`).toBeLessThanOrEqual(200);
  }
});

// --- every internal link resolves ---------------------------------------------

/** Every string in a module tree, with the key path it was found at. */
function strings(value, at = '', out = []) {
  if (typeof value === 'string') out.push([at, value]);
  else if (Array.isArray(value)) value.forEach((v, i) => strings(v, `${at}[${i}]`, out));
  else if (value && typeof value === 'object') for (const [k, v] of Object.entries(value)) strings(v, at ? `${at}.${k}` : k, out);
  return out;
}

const modules = { home, why, platform, solutions, pricing, company, resources, faq, nav: { nav }, footer: { footer } };
const allStrings = Object.entries(modules).flatMap(([name, mod]) => strings(mod, name));
const isInternalLink = (s) => /^\/[^\s]*$/.test(s) && !s.startsWith('//');

test('every internal link in the copy, the nav and the footer resolves', () => {
  const dead = [];
  for (const [at, s] of allStrings) {
    if (!isInternalLink(s)) continue;
    const path = s.split('#')[0].split('?')[0].replace(/\/$/, '') || '/';
    if (registry.has(path) || ENGINE_ROUTES.includes(path)) continue;
    if (existsSync(join(STATIC_DIR, path))) continue; // a shipped file: /llms.txt, /logos/x.svg
    if (path === '/sitemap.xml') continue; // an endpoint, src/routes/sitemap.xml
    dead.push(`${at}: ${s}`);
  }
  expect(dead).toEqual([]);
});

test('an anchor that points into the front page or the pricing page exists there', () => {
  const anchors = allStrings.map(([, s]) => s).filter((s) => isInternalLink(s) && s.includes('#'));
  const source = walk(join(SITE_DIR, 'src')).filter((f) => f.endsWith('.svelte')).map((f) => readFileSync(f, 'utf8')).join('\n');
  const missing = [...new Set(anchors)].filter((a) => {
    const id = a.split('#')[1];
    return !ENGINE_ROUTES.includes(a.split('#')[0]) && !new RegExp(`id=["'{\`]${id}["'}\`]`).test(source);
  });
  expect(missing).toEqual([]);
});

// --- the voice ----------------------------------------------------------------
// The copy is written without em dashes, semicolons or exclamation marks. It
// is a small rule that keeps sentences short and claims plain. Strings that
// are not prose are left alone: links, addresses, code, placeholders.

const isProse = (s) => s.includes(' ') && !/^https?:|^mailto:|^\/|@[a-z0-9-]+\./i.test(s);
const isCode = (at) => /(\.|^)(cmd|command|code|install|snippet|curl|sh)(\.|\[|$)/i.test(at);

test('the copy keeps its voice: no em dashes, semicolons or exclamation marks', () => {
  const offenders = allStrings
    .filter(([at, s]) => isProse(s) && !isCode(at))
    .filter(([, s]) => /[—;!]/.test(s))
    .map(([at, s]) => `${at}: ${s.slice(0, 90)}`);
  expect(offenders).toEqual([]);
});

test('no placeholder is misspelled: every {name} in the copy is one live.js fills', async () => {
  const { live } = await import('./live.js');
  const unknown = allStrings.flatMap(([at, s]) => [...s.matchAll(/\{(\w+)\}/g)].filter((m) => !(m[1] in live)).map((m) => `${at}: {${m[1]}}`));
  expect(unknown).toEqual([]);
});

// --- the media is real -----------------------------------------------------------

test('every clip in the manifest ships an mp4, a webm and a poster, and none is a placeholder', () => {
  const problems = [];
  for (const clip of new Map(allClips.map((c) => [c.name, c])).values()) {
    for (const [kind, url, min, max] of [
      // Canvas recordings land well under 1.6 MB. Photographic b-roll from the
      // prompt pack is heavier; 4.5 MB is 10 s of dark 720p at the encoder's crf.
      ['mp4', clip.mp4, 40_000, clip.maxBytes ?? 4_500_000],
      ['webm', clip.webm, 40_000, clip.maxBytes ?? 4_500_000],
      ['poster', clip.poster, 4_000, 160_000]
    ]) {
      const file = join(STATIC_DIR, url);
      if (!existsSync(file)) problems.push(`${clip.name}: ${kind} is missing`);
      else {
        const size = statSync(file).size;
        if (size < min) problems.push(`${clip.name}: ${kind} is ${size} bytes, that is a placeholder`);
        if (size > max) problems.push(`${clip.name}: ${kind} is ${size} bytes, over budget`);
      }
    }
    if (clip.alt.length < 20) problems.push(`${clip.name}: alt text is too short to describe a video`);
  }
  expect(problems).toEqual([]);
});

test('the hero poster is small enough to be the largest paint on the front page', () => {
  expect(statSync(join(STATIC_DIR, media.hero.poster)).size).toBeLessThan(100_000);
});

test('every installed still exists, has alt text, and sits on pages that exist', () => {
  const problems = [];
  for (const art of Object.values(media.art)) {
    if (!existsSync(join(STATIC_DIR, art.src))) problems.push(`${art.slot}: ${art.src} is missing`);
    if (!art.alt || art.alt.length < 20) problems.push(`${art.slot}: needs alt text`);
    for (const p of art.pages) if (!registry.has(p)) problems.push(`${art.slot}: ${p} is not a page`);
  }
  expect(problems).toEqual([]);
});

test('the logo wall only shows files that ship', () => {
  // `file` and `fileDark` are basenames of an SVG under static/logos, `emblem` of a WebP.
  const logo = (name, ext) => existsSync(join(STATIC_DIR, 'logos', `${name}.${ext}`));
  const missing = [];
  for (const i of home.logoWall.items) {
    if (i.file && !logo(i.file, 'svg')) missing.push(`${i.name}: ${i.file}.svg`);
    if (i.emblem && !logo(i.emblem, 'webp')) missing.push(`${i.name}: ${i.emblem}.webp`);
  }
  for (const p of home.logoWall.programs) {
    for (const f of [p.file, p.fileDark]) if (f && !logo(f, 'svg')) missing.push(`${p.name}: ${f}.svg`);
    if (p.src && !existsSync(join(STATIC_DIR, p.src))) missing.push(`${p.name}: ${p.src}`);
  }
  expect(missing).toEqual([]);
});

test('every logo that ships has its source and terms written down', () => {
  const readme = readFileSync(join(STATIC_DIR, 'logos', 'README.md'), 'utf8');
  const undocumented = readdirSync(join(STATIC_DIR, 'logos')).filter((f) => f !== 'README.md' && !readme.includes(`\`${f}\``));
  expect(undocumented).toEqual([]);
});

// An emblem entry must still make sense as plain type, because turning
// `logoWall.emblems` off is the documented way back.
test('every emblem on the wall has a plain text form', () => {
  expect(typeof home.logoWall.emblems).toBe('boolean');
  expect(home.logoWall.items.filter((i) => i.emblem && !(i.short || i.name))).toEqual([]);
  expect(home.logoWall.note).toContain('does not imply or constitute DoD endorsement');
});
