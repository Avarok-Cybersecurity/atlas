#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
//
// scripts-inject-meta.mjs — the per-page half of the docs' social metadata.
//
// theme/head.hbs carries everything handlebars can express. It cannot express
// the rest: mdBook's template has no string helpers, so `{{ path }}`
// ("getting-started/quickstart.md") cannot be turned into a canonical URL, and
// there is no per-page description in the template context at all — mdBook's
// `{{ description }}` is the book-level one, stamped identically on every page.
//
// So this walks the built HTML and adds what the template could not. The split
// is strict: no tag is emitted by both, so nothing is duplicated. It also
// REPLACES two things mdBook writes itself — the book-level description, and a
// hardcoded white `theme-color` that would otherwise sit beside the brand one
// (browsers disagree about which of two wins, so there must be exactly one).
//
// Written in plain node with no dependencies, like scripts-gen-llms.mjs beside
// it, and it is its own assertion: any page that comes out without a complete
// card set exits non-zero.

import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

const ORIGIN = 'https://docs.atlasinference.io';
const BOOK = 'The Atlas Book';
const root = process.argv[2];
if (!root) {
  console.error('usage: node scripts-inject-meta.mjs <book output dir>');
  process.exit(2);
}

const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]
  );

const strip = (html) =>
  html
    .replace(/<(script|style)[\s\S]*?<\/\1>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&[a-z]+;/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim();

/** Trim to a length a card will actually show, on a word boundary. */
const clamp = (s, n = 200) => {
  if (s.length <= n) return s;
  const cut = s.slice(0, n);
  return `${cut.slice(0, cut.lastIndexOf(' '))}…`;
};

const esc = (s) =>
  s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const pages = walk(root).filter((f) => f.endsWith('.html'));
const problems = [];
let done = 0;

for (const file of pages) {
  const rel = relative(root, file).split(sep).join('/');
  let html = readFileSync(file, 'utf8');

  // rustdoc brings its own head; the print page is already noindex; redirect
  // stubs are not pages anyone should land on or share.
  if (rel.startsWith('api/') || rel === 'print.html' || rel === '404.html') continue;
  if (/http-equiv=["']refresh["']/i.test(html)) continue;

  // mdBook copies the first chapter to index.html. Both URLs exist, so give
  // the copy the same canonical as the original rather than letting the two
  // compete as duplicate content.
  const url = rel === 'index.html' ? `${ORIGIN}/` : `${ORIGIN}/${rel}`;
  const canonical = rel === 'introduction.html' ? `${ORIGIN}/` : url;

  const title = strip(/<title>([\s\S]*?)<\/title>/i.exec(html)?.[1] ?? BOOK);
  const main = /<main[^>]*>([\s\S]*?)<\/main>/i.exec(html)?.[1] ?? '';
  const firstPara = [...main.matchAll(/<p[^>]*>([\s\S]*?)<\/p>/gi)]
    .map((m) => strip(m[1]))
    .find((t) => t.length > 40);
  const description = clamp(firstPara || `${title} — part of ${BOOK}.`);

  // Exactly one theme-color, and it is the brand's.
  html = html.replace(/\s*<meta name="theme-color" content="#ffffff">/i, '');
  // Exactly one description, and it is this page's.
  html = html.replace(
    /<meta name="description" content="[^"]*">/i,
    `<meta name="description" content="${esc(description)}">`
  );

  const ld = JSON.stringify({
    '@context': 'https://schema.org',
    '@type': rel === 'index.html' ? 'WebSite' : 'TechArticle',
    headline: title,
    description,
    url: canonical,
    isPartOf: { '@type': 'WebSite', name: BOOK, url: `${ORIGIN}/` },
    publisher: { '@type': 'Organization', name: 'Atlas Inference', url: 'https://atlasinference.io/' }
    // JSON.stringify does not escape "<", so a closing script tag anywhere in a
    // chapter's prose would end the block early and spill markup into the page.
  }).replace(/</g, '\\u003c');

  // Idempotent: a previous run's block is removed before a new one is written.
  // Without this a second invocation — a re-run of the CI step, or someone
  // checking output locally — silently appends a SECOND og:url and canonical,
  // and duplicate metadata is worse than none because scrapers pick
  // arbitrarily. The self-check below caught exactly that.
  html = html.replace(/\n?<!-- atlas:meta -->[\s\S]*?<!-- \/atlas:meta -->/g, '');

  const injected = [
    `<link rel="canonical" href="${canonical}">`,
    `<meta property="og:url" content="${canonical}">`,
    `<meta property="og:type" content="${rel === 'index.html' ? 'website' : 'article'}">`,
    `<meta property="og:description" content="${esc(description)}">`,
    `<meta name="twitter:description" content="${esc(description)}">`,
    `<script type="application/ld+json">${ld}</script>`
  ];
  const block = `<!-- atlas:meta -->\n${injected.join('\n')}\n<!-- /atlas:meta -->`;

  html = html.replace('</head>', `${block}\n</head>`);
  writeFileSync(file, html);
  done += 1;

  for (const [needle, what] of [
    ['rel="canonical"', 'canonical'],
    ['property="og:title"', 'og:title'],
    ['property="og:url"', 'og:url'],
    ['property="og:image"', 'og:image'],
    ['name="twitter:card"', 'twitter:card']
  ]) {
    const n = html.split(needle).length - 1;
    if (n !== 1) problems.push(`${rel}: ${what} appears ${n} times, expected exactly 1`);
  }
  const themeColors = html.split('name="theme-color"').length - 1;
  if (themeColors !== 1) problems.push(`${rel}: ${themeColors} theme-color tags, expected exactly 1`);
}

console.log(`social meta: ${done} page(s)`);
if (problems.length) {
  for (const p of problems) console.error(`::error::${p}`);
  process.exit(1);
}
