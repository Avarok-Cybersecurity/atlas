#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
// =============================================================================
// gen-site-guide.mjs — write SITE-GUIDE.md, the map of the whole site
// -----------------------------------------------------------------------------
// The guide is the first thing a person or an agent opens before changing the
// site. It answers, without reading any code: what pages exist, which file draws
// each and which file holds its words, every button and link and where it goes,
// every outside link and address and every page that uses it, every logo, clip
// and portrait with its hash, and when each of those last changed.
//
// It is GENERATED, from the built site and the content modules, so it cannot
// drift from what ships. Two files are kept by it and committed:
//
//   guide/ledger.json   every tracked fact and asset with a revision and a date,
//                       and a short change log. The only file with memory.
//   SITE-GUIDE.md       the map, for reading.          } rebuilt from scratch
//   guide/site-guide.json  the same map, for programs. } on every run
//
//   bun x --bun vite build                 the guide reads build/
//   bun run guide -- --note "what changed" [--pr 1136]
//   bun run guide:check                    exits 1 if anything is out of date
//
// A change to a tracked fact (an address, a link, a name) fails the unit suite
// until this has been run, and `guide:check` runs with the browser suite. So the
// guide is current on every green build, by construction rather than by memory.
// =============================================================================

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { LEDGER, REPO_DIR, SITE_DIR, collectAssets, collectFacts, diffLedger, readLedger } from './guide/facts.mjs';

const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const opt = (name) => (args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : undefined);
const CHECK = flag('check');
const now = new Date();
const TODAY = opt('date') ?? `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;

const BUILD = join(SITE_DIR, 'build');
const OUT_MD = join(SITE_DIR, 'SITE-GUIDE.md');
const OUT_JSON = join(SITE_DIR, 'guide', 'site-guide.json');
const posix = (p) => p.split(sep).join('/');
const rel = (p) => posix(relative(SITE_DIR, p));

if (!existsSync(join(BUILD, 'index.html'))) {
  console.error(`guide: no build at ${BUILD}. Run \`bun x --bun vite build\` in site/ first.`);
  process.exit(2);
}

// ---- the content modules ------------------------------------------------------
const load = (file) => import(pathToFileURL(join(SITE_DIR, 'src', 'lib', 'content', file)).href);
const brand = await load('brand.js');
const { pages: registry } = await load('index.js');

// ---- 1. the ledger: facts and assets, each with a revision and a date ------------
const facts = await collectFacts();
const assets = collectAssets();
const ledger = readLedger();
const diff = diffLedger(ledger, facts, assets);
const dirty = diff.changedFacts.length + diff.removedFacts.length + diff.changedAssets.length + diff.removedAssets.length > 0;

if (CHECK && dirty) {
  console.error('guide: the ledger is behind the site.');
  for (const [label, keys] of Object.entries(diff)) if (keys.length) console.error(`  ${label}: ${keys.slice(0, 12).join(', ')}${keys.length > 12 ? ` and ${keys.length - 12} more` : ''}`);
  console.error('  Run `bun run guide -- --note "what changed"` and commit guide/ and SITE-GUIDE.md.');
  process.exit(1);
}

if (!CHECK) {
  const bump = (bucket, key, next, keep) => {
    const prev = bucket[key];
    if (!prev) return (bucket[key] = { ...next, rev: 1, date: TODAY });
    const was = [...(prev.was ?? []), { ...keep(prev), until: TODAY }].slice(-5);
    bucket[key] = { ...next, rev: prev.rev + 1, date: TODAY, was };
  };
  for (const k of diff.changedFacts) bump(ledger.facts, k, { value: facts[k] }, (p) => ({ value: p.value ?? '(removed)' }));
  for (const k of diff.changedAssets) bump(ledger.assets, k, assets[k], (p) => ({ sha: p.sha ?? '(removed)' }));
  for (const k of diff.removedFacts) ledger.facts[k] = { ...ledger.facts[k], removed: TODAY };
  for (const k of diff.removedAssets) ledger.assets[k] = { ...ledger.assets[k], removed: TODAY };
  const note = opt('note');
  if (dirty || note) {
    let base = '';
    try {
      base = execFileSync('git', ['rev-parse', '--short=9', 'HEAD'], { cwd: REPO_DIR, encoding: 'utf8' }).trim();
    } catch {
      base = '';
    }
    const touched = [...diff.changedFacts, ...diff.removedFacts, ...diff.changedAssets.map((a) => a.split('/').pop()), ...diff.removedAssets.map((a) => a.split('/').pop())];
    ledger.started ??= TODAY;
    ledger.changes.push({
      rev: ledger.changes.length + 1,
      date: TODAY,
      ...(opt('pr') ? { pr: Number(opt('pr')) } : {}),
      ...(base ? { base } : {}),
      note: note ?? 'Tracked values changed.',
      touched: touched.slice(0, 40)
    });
  }
  const sorted = (o) => Object.fromEntries(Object.keys(o).sort().map((k) => [k, o[k]]));
  mkdirSync(dirname(LEDGER), { recursive: true });
  writeFileSync(LEDGER, JSON.stringify({ started: ledger.started, changes: ledger.changes, facts: sorted(ledger.facts), assets: sorted(ledger.assets) }, null, 2) + '\n');
}

// ---- 2. the pages, read from the build --------------------------------------------
const htmlFiles = [];
const walk = (dir) => {
  for (const f of readdirSync(dir)) {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) walk(p);
    else if (f.endsWith('.html')) htmlFiles.push(p);
  }
};
walk(BUILD);
const toRoute = (file) => {
  const r = '/' + posix(relative(BUILD, file)).replace(/index\.html$/, '').replace(/\.html$/, '');
  return r.length > 1 ? r.replace(/\/$/, '') : '/';
};

// The visible words of a fragment of our own built HTML. Entities are decoded in ONE
// pass, from a table: decoding `&amp;` first and the rest after it would turn
// `&amp;quot;` into a quote mark, which is a different string from the one on the page.
const ENTITY = { amp: '&', quot: '"', apos: "'", '#39': "'", nbsp: ' ', lt: '<', gt: '>' };
const text = (html) =>
  html
    .replace(/<(script|style|svg)\b[\s\S]*?<\/\1\s*>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&(amp|quot|apos|#39|nbsp|lt|gt);/g, (_, name) => ENTITY[name])
    .replace(/[→↗↓↑]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
const attr = (tag, name) => (tag.match(new RegExp(`\\s${name}="([^"]*)"`)) || [])[1];

const routesDir = join(SITE_DIR, 'src', 'routes');
const routeFiles = [];
const walkRoutes = (dir) => {
  for (const f of readdirSync(dir)) {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) walkRoutes(p);
    else if (f === '+page.svelte') routeFiles.push(p);
  }
};
walkRoutes(routesDir);
/** The +page.svelte that draws a path: exact folder first, then a [param] folder. */
function routeFileFor(path) {
  const segs = path.split('/').filter(Boolean);
  let best = null;
  for (const file of routeFiles) {
    const parts = posix(relative(routesDir, dirname(file))).split('/').filter((s) => s && !/^\(.*\)$/.test(s));
    if (parts.length !== segs.length) continue;
    const exact = parts.every((p, i) => p === segs[i]);
    const dynamic = parts.every((p, i) => p === segs[i] || /^\[.*\]$/.test(p));
    if (exact) return file;
    if (dynamic) best = file;
  }
  return best;
}

/** Follow a file's imports through components to the content modules that hold its words. */
const importCache = new Map();
function contentSources(file, seen = new Set()) {
  if (seen.has(file) || !existsSync(file)) return new Map();
  seen.add(file);
  if (!importCache.has(file)) {
    const src = readFileSync(file, 'utf8');
    importCache.set(file, [...src.matchAll(/import\s+(?:([\w*\s{},$]+?)\s+from\s+)?['"]([^'"]+)['"]/g)].map((m) => ({ names: m[1] ?? '', from: m[2] })));
  }
  const found = new Map();
  for (const { names, from } of importCache.get(file)) {
    let target = null;
    if (from.startsWith('$lib/')) target = join(SITE_DIR, 'src', 'lib', from.slice(5));
    else if (from.startsWith('.')) target = join(dirname(file), from);
    if (!target) continue;
    const t = posix(relative(SITE_DIR, target));
    if (/^src\/lib\/(content\/[\w.-]+\.js|data\.js)$/.test(t)) {
      const picked = (names.match(/\{([^}]*)\}/) || [, ''])[1].split(',').map((n) => n.trim().split(/\s+as\s+/)[0]).filter(Boolean);
      const set = found.get(t) ?? new Set();
      picked.forEach((n) => set.add(n));
      found.set(t, set);
    } else if (/\.svelte$/.test(target)) {
      for (const [k, v] of contentSources(target, seen)) {
        const set = found.get(k) ?? new Set();
        v.forEach((n) => set.add(n));
        found.set(k, set);
      }
    }
  }
  return found;
}

// Pages whose links are drawn from DATA, not from copy: the contributor list comes
// from GitHub, the model catalogue from another repository, the release notes
// from the changelog. Their links change when that data does, on somebody else's
// commit, and a guide that went stale every time a contributor arrived would be
// a gate nobody could keep green. So they are mapped, and their links are not
// inventoried. Everything else on the site is copy, and copy is deterministic.
const DATA_DRIVEN = ['/engine', '/control', '/diligence', '/benchmarks', '/platform/hardware', '/resources/contributors', '/resources/updates'];

const kindOf = (href) => (/^mailto:/.test(href) ? 'mail' : /^https?:/.test(href) ? 'out' : href.startsWith('#') ? 'anchor' : 'page');
const pages = [];
for (const file of htmlFiles.sort()) {
  const path = toRoute(file);
  if (path === '/404' || path === '/broll') continue;
  const html = readFileSync(file, 'utf8');
  const main = (html.match(/<main\b[\s\S]*?<\/main>/) || [html])[0];
  const links = [];
  for (const m of main.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/g)) {
    const href = attr(`<a${m[1]}>`, 'href');
    if (!href) continue;
    const label = text(m[2]) || attr(`<a${m[1]}>`, 'aria-label') || '(image)';
    links.push({ text: label.slice(0, 90), href: href.replace(/&amp;/g, '&'), kind: kindOf(href), button: /\bav-btn\b/.test(m[1]) });
  }
  const generated = DATA_DRIVEN.includes(path);
  const seen = new Set();
  const unique = generated ? [] : links.filter((l) => !seen.has(l.text + '|' + l.href) && seen.add(l.text + '|' + l.href));
  const controls = generated ? [] : [...new Set([...main.matchAll(/<button\b([^>]*)>([\s\S]*?)<\/button>/g)].map((m) => text(m[2]) || attr(`<button${m[1]}>`, 'aria-label') || '').filter(Boolean))];
  const forms = [...main.matchAll(/<form\b[^>]*aria-labelledby="([^"]+)-form-title"/g)].map((m) => m[1]);
  const route = routeFileFor(path);
  const sources = route ? [...contentSources(route)].map(([f, names]) => ({ file: f, exports: [...names].sort() })).sort((a, b) => a.file.localeCompare(b.file)) : [];
  const reg = registry.find((p) => p.path === path);
  pages.push({
    path,
    title: reg?.title ?? text((html.match(/<title>([\s\S]*?)<\/title>/) || [, ''])[1]),
    h1: text((main.match(/<h1\b[^>]*>([\s\S]*?)<\/h1>/) || [, ''])[1]),
    route: route ? rel(route) : null,
    copy: sources,
    generated,
    links: unique,
    controls,
    forms,
    html
  });
}

// ---- 3. where each fact and asset is used -------------------------------------------
const usedOn = (needle) => (needle && needle.length > 3 ? pages.filter((p) => p.html.includes(needle) || p.html.includes(needle.replace(/&/g, '&amp;'))).map((p) => p.path) : []);
const current = readLedger();
const book = CHECK ? current : JSON.parse(readFileSync(LEDGER, 'utf8'));
const factRows = Object.entries(book.facts).filter(([, v]) => !v.removed).map(([key, v]) => ({ key, ...v, used: usedOn(String(v.value)) }));
const assetRows = Object.entries(book.assets).filter(([, v]) => !v.removed).map(([path, v]) => {
  const served = path.startsWith('site/static/') ? path.slice('site/static'.length) : null;
  return { path, ...v, used: served ? usedOn(served) : [] };
});

const outbound = new Map();
const mail = new Map();
for (const p of pages) {
  for (const l of p.links) {
    if (l.kind === 'out') (outbound.get(l.href) ?? outbound.set(l.href, new Set()).get(l.href)).add(p.path);
    if (l.kind === 'mail') {
      const address = decodeURIComponent(l.href.slice(7).split('?')[0]);
      (mail.get(address) ?? mail.set(address, new Set()).get(address)).add(p.path);
    }
  }
}
const navLinks = brand.nav.groups.flatMap((g) => (g.columns ? g.columns.flatMap((c) => c.items.map((i) => ({ menu: g.label, text: i.text, href: i.href }))) : [{ menu: '(top level)', text: g.label, href: g.href }]));
const footerLinks = brand.footer.cols.flatMap((c) => c.links.map((l) => ({ column: c.heading, text: l.text, href: l.href })));
for (const l of [...navLinks, ...footerLinks]) if (/^https?:/.test(l.href)) (outbound.get(l.href) ?? outbound.set(l.href, new Set()).get(l.href)).add('(every page: header or footer)');

// ---- 4. write ----------------------------------------------------------------------------
const last = book.changes.at(-1);
const few = (list, n = 4) => (list.length <= n ? list.join(', ') : `${list.slice(0, n).join(', ')} and ${list.length - n} more`);
// A Markdown table cell: the backslash first, or escaping the pipe would leave a
// stray backslash in the input free to unescape it again.
const cell = (s) => String(s).replace(/\\/g, '\\\\').replace(/\|/g, '\\|').replace(/\n/g, ' ');
// Cut at a whole word. A label cut in the middle of one reads badly, and the
// repository's spelling check rightly calls the stump a typo.
const clip = (s, n = 96) => {
  const str = String(s);
  if (str.length <= n) return str;
  const cut = str.slice(0, n - 1);
  const space = cut.lastIndexOf(' ');
  return (space > n * 0.5 ? cut.slice(0, space) : cut).replace(/[\s,.;:]+$/, '') + ' …';
};
const L = [];
L.push('# Avarok site guide', '');
L.push(`Revision ${last?.rev ?? 0}, ${last?.date ?? TODAY}${last?.pr ? `, pull request #${last.pr}` : ''}${last?.base ? `, on top of \`${last.base}\`` : ''}. ${pages.length} pages, ${factRows.length} tracked facts, ${assetRows.length} tracked assets.`, '');
L.push('> Generated by `bun run guide` from the built site and `src/lib/content/`. Do not edit it by hand: change the', '> source, run the command, commit the result. `AGENTS.md` beside this file says how to work on the site.', '');
L.push('## How to use this', '');
L.push('- **Changing words?** Find the page under [Pages](#pages). Its `copy` line names the file and the exports that hold its words.');
L.push('- **Changing a name, an address or a link?** Find it under [Tracked facts](#tracked-facts). It is defined once, in `src/lib/content/brand.js`, and the `used on` column is every page that will change with it.');
L.push('- **Swapping a logo, a portrait or a clip?** Find it under [Tracked assets](#tracked-assets). Replace the file, keep the name.');
L.push('- **Wondering where a button goes?** Every page lists its buttons and links with their targets.');
L.push('- **Then** run `bun run guide -- --note "what you changed"`. The unit suite fails until you do, which is the point.', '');
L.push('## Change log', '', '| rev | date | pointer | what changed |', '| --- | --- | --- | --- |');
for (const c of [...book.changes].reverse().slice(0, 25)) L.push(`| ${c.rev} | ${c.date} | ${[c.pr ? `#${c.pr}` : '', c.base ? `after \`${c.base}\`` : ''].filter(Boolean).join(' ')} | ${cell(c.note)}${c.touched?.length ? ` <br>Touched: ${cell(few(c.touched, 8))}` : ''} |`);
L.push('', '## Tracked facts', '', 'Names, addresses, links and licence lines. Each is defined once in `src/lib/content/brand.js`. `rev` counts how many times the value has changed since the ledger began, `date` is the last change.', '', '| key | value | rev | date | used on |', '| --- | --- | --- | --- | --- |');
for (const f of factRows) L.push(`| \`${f.key}\` | ${cell(clip(f.value))} | ${f.rev} | ${f.date} | ${f.used.length ? `${f.used.length}: ${cell(few(f.used, 3))}` : 'not rendered as text'} |`);
L.push('', '## Tracked assets', '', 'Every file a visitor is served that came from somewhere, and the brand masters. `static/logos/README.md` and `media-brief/` record sources and terms.', '', '| file | bytes | sha256 | rev | date | used on |', '| --- | --- | --- | --- | --- | --- |');
for (const a of assetRows) L.push(`| \`${a.path}\` | ${a.bytes.toLocaleString('en-US')} | \`${a.sha}\` | ${a.rev} | ${a.date} | ${a.used.length ? `${a.used.length}: ${cell(few(a.used, 3))}` : ''} |`);
L.push('', '## Header menu', '', 'From `nav` in `src/lib/content/brand.js`. The same on every page.', '', '| menu | item | goes to |', '| --- | --- | --- |');
for (const l of navLinks) L.push(`| ${cell(l.menu)} | ${cell(l.text)} | \`${l.href}\` |`);
L.push(`| (button) | ${cell(brand.nav.cta.text)} | \`${brand.nav.cta.href}\` |`);
L.push('', '## Footer', '', 'From `footer` in `src/lib/content/brand.js`. The same on every page.', '', '| column | item | goes to |', '| --- | --- | --- |');
for (const l of footerLinks) L.push(`| ${cell(l.column)} | ${cell(l.text)} | \`${l.href}\` |`);
L.push('', '## Pages', '');
for (const p of pages) {
  L.push(`### \`${p.path}\``, '', `**${cell(p.title)}**${p.h1 ? `. Headline: ${cell(clip(p.h1, 120))}` : ''}`, '');
  L.push(`- drawn by: ${p.route ? `\`${p.route}\`` : 'unknown'}`);
  if (p.copy.length) L.push(`- copy: ${p.copy.map((c) => `\`${c.file}\`${c.exports.length ? ` (${c.exports.join(', ')})` : ''}`).join(', ')}`);
  if (p.forms.length) L.push(`- forms: ${p.forms.map((f) => `\`${f}\``).join(', ')} (one component, \`src/lib/components/avarok/DemoForm.svelte\`)`);
  if (p.controls.length) L.push(`- controls: ${p.controls.map((c) => `"${cell(clip(c, 40))}"`).slice(0, 14).join(', ')}${p.controls.length > 14 ? ` and ${p.controls.length - 14} more` : ''}`);
  if (p.generated) L.push('- buttons and links: drawn from data (contributors, recipes, records or the changelog), so not listed here. They change with the data, not with the copy.');
  if (p.links.length) {
    L.push('- buttons and links:');
    for (const l of p.links) L.push(`  - ${l.button ? '[button] ' : ''}"${cell(clip(l.text, 70))}" goes to \`${clip(l.href, 110)}\``);
  }
  L.push('');
}
L.push('## Outbound links', '', 'Every address on another site, and the pages that link to it. Change one in `links` in `brand.js` and every use follows.', '', '| link | pages |', '| --- | --- |');
for (const [href, set] of [...outbound].sort()) L.push(`| ${cell(clip(href, 110))} | ${cell(few([...set], 4))} |`);
L.push('', '## Mail links', '', '| address | pages |', '| --- | --- |');
for (const [address, set] of [...mail].sort()) L.push(`| ${address} | ${cell(few([...set], 6))} |`);
L.push('');
const md = L.join('\n');
const json = JSON.stringify(
  {
    revision: last?.rev ?? 0,
    date: last?.date ?? TODAY,
    changes: book.changes,
    facts: factRows,
    assets: assetRows,
    nav: navLinks,
    footer: footerLinks,
    pages: pages.map(({ html, ...p }) => p),
    outbound: Object.fromEntries([...outbound].sort().map(([k, v]) => [k, [...v]])),
    mail: Object.fromEntries([...mail].sort().map(([k, v]) => [k, [...v]]))
  },
  null,
  1
) + '\n';

if (CHECK) {
  const stale = [];
  if (!existsSync(OUT_MD) || readFileSync(OUT_MD, 'utf8').replace(/\r\n/g, '\n') !== md) stale.push(rel(OUT_MD));
  if (!existsSync(OUT_JSON) || readFileSync(OUT_JSON, 'utf8').replace(/\r\n/g, '\n') !== json) stale.push(rel(OUT_JSON));
  if (stale.length) {
    console.error(`guide: out of date: ${stale.join(', ')}. A page, a button or a link changed.`);
    console.error('  Run `bun run guide -- --note "what changed"` after a build, and commit the result.');
    process.exit(1);
  }
  console.log(`guide: current. Revision ${last?.rev ?? 0}, ${pages.length} pages, ${factRows.length} facts, ${assetRows.length} assets.`);
} else {
  writeFileSync(OUT_MD, md);
  writeFileSync(OUT_JSON, json);
  console.log(`guide: wrote ${rel(OUT_MD)} and ${rel(OUT_JSON)}. Revision ${last?.rev ?? 0}, ${pages.length} pages, ${factRows.length} facts, ${assetRows.length} assets.`);
  if (dirty) console.log(`guide: recorded ${diff.changedFacts.length} fact and ${diff.changedAssets.length} asset changes.`);
}
