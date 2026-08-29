#!/usr/bin/env bun
/**
 * Every fragment the blog points at on atlasinference.io must exist there.
 *
 *   bun blog/e2e/check-crosslinks.mjs [site-build-dir] [blog-build-dir]
 *
 * The blog's footer deep-links into the marketing site's sections. Those ids
 * live in another app, in another build, and nothing connects the two — so a
 * renamed section leaves a link that still resolves (the page loads) and
 * silently lands the reader at the top of it. Two of these shipped in the first
 * draft, `#benchmarks` and `#get-running`, both invented rather than looked up.
 *
 * Checked against the BUILT html rather than the source, because the id has to
 * survive the build to be worth anything.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const siteDir = process.argv[2] ?? 'site/build';
const blogDir = process.argv[3] ?? 'blog/build';
const MAIN = 'https://atlasinference.io';

const walk = (dir) =>
  readdirSync(dir).flatMap((e) => {
    const p = join(dir, e);
    return statSync(p).isDirectory() ? walk(p) : p.endsWith('.html') ? [p] : [];
  });

const siteIds = new Set();
for (const f of walk(siteDir)) {
  for (const m of readFileSync(f, 'utf8').matchAll(/\sid="([^"]+)"/g)) siteIds.add(m[1]);
}
if (siteIds.size === 0) {
  console.error(`No ids found under ${siteDir} — wrong directory, or the site did not build.`);
  process.exit(1);
}

let bad = 0, checked = 0;
for (const f of walk(blogDir)) {
  const html = readFileSync(f, 'utf8');
  for (const m of html.matchAll(new RegExp(`href="${MAIN}/?#([^"]+)"`, 'g'))) {
    checked++;
    if (!siteIds.has(m[1])) {
      console.error(`  MISSING  ${f}: links to ${MAIN}/#${m[1]}, which is not an id on the marketing site`);
      bad++;
    } else {
      console.log(`  ok       #${m[1]}`);
    }
  }
}

if (checked === 0) {
  // Not a pass. If the footer stops emitting cross-links this check silently
  // stops checking, which is how a guard becomes decorative.
  console.error(`No ${MAIN}/#fragment links found in ${blogDir}. Expected at least one.`);
  process.exit(1);
}
if (bad) {
  console.error(`\n${bad} of ${checked} cross-property fragment link(s) point at nothing.`);
  process.exit(1);
}
console.log(`\nall ${checked} cross-property fragment links resolve`);
