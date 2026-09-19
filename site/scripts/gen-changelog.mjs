#!/usr/bin/env node
// =============================================================================
// gen-changelog.mjs — src/lib/changelog.generated.json from ../CHANGELOG.md
// -----------------------------------------------------------------------------
// The product updates page renders the repository changelog rather than a
// second copy of it, so "what shipped" cannot drift from what shipped. Each
// release becomes { version, date, sections: [{ kind, items }] }, where an
// item is one bullet with its inline markdown reduced to plain text and its
// sub-bullets folded in.
//
// Keep a Changelog structure: `## [Version] - date` headers, `### Added`,
// `### Changed`, `### Fixed` ... sections, `- ` bullets. The top N releases
// are kept; the page links to GitHub for the rest.
// =============================================================================
import { readFileSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '..');
const src = resolve(site, '..', 'CHANGELOG.md');
const out = resolve(site, 'src/lib/changelog.generated.json');
const KEEP = 6;
const MAX_ITEMS = 12;

const md = readFileSync(src, 'utf8');
const plain = (s) =>
  s
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();

const releases = [];
let rel = null;
let section = null;
let item = null;
const flushItem = () => {
  if (item && section) section.items.push(plain(item));
  item = null;
};
for (const line of md.split('\n')) {
  const h2 = line.match(/^## \[?([^\]]+?)\]?(?:\s*-\s*(\d{4}-\d{2}-\d{2}))?\s*$/);
  if (h2) {
    flushItem();
    rel = { version: h2[1].trim(), date: h2[2] ?? '', sections: [] };
    releases.push(rel);
    section = null;
    continue;
  }
  const h3 = line.match(/^### (\w+)/);
  if (h3 && rel) {
    flushItem();
    const kind = h3[1];
    section = rel.sections.find((s) => s.kind === kind);
    if (!section) {
      section = { kind, items: [] };
      rel.sections.push(section);
    }
    continue;
  }
  if (!section) continue;
  if (/^- /.test(line)) {
    flushItem();
    item = line.slice(2);
  } else if (item !== null && /^\s+\S/.test(line)) {
    item += ' ' + line.trim().replace(/^- /, '');
  } else if (line.trim() === '') {
    flushItem();
  }
}
flushItem();

const kept = releases
  .filter((r) => r.sections.some((s) => s.items.length))
  .slice(0, KEEP)
  .map((r) => ({
    ...r,
    sections: r.sections.map((s) => ({ kind: s.kind, items: s.items.slice(0, MAX_ITEMS), more: Math.max(0, s.items.length - MAX_ITEMS) }))
  }));
if (kept.length === 0) throw new Error('gen-changelog: no releases parsed from CHANGELOG.md');

let sha = 'unknown';
try {
  sha = execSync('git rev-parse --short=10 HEAD', { cwd: site, encoding: 'utf8' }).trim();
} catch {}
const json = { generated_sha: sha, generated_date: new Date().toISOString().slice(0, 10), source: 'CHANGELOG.md', releases: kept };
const prev = (() => {
  try {
    return readFileSync(out, 'utf8');
  } catch {
    return '';
  }
})();
// Stable output: only rewrite when the releases changed, so a build does not
// churn the generated file for a date stamp alone.
const prevReleases = prev ? JSON.stringify(JSON.parse(prev).releases) : '';
if (prevReleases !== JSON.stringify(kept)) {
  writeFileSync(out, JSON.stringify(json, null, 2) + '\n');
  console.log(`gen-changelog: wrote ${out} (${kept.length} releases)`);
} else {
  console.log(`gen-changelog: ${out} unchanged (${kept.length} releases)`);
}
