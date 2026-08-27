#!/usr/bin/env node
// =============================================================================
// gen-sitemap.mjs — generate static/sitemap.xml from the routes the site ships
// -----------------------------------------------------------------------------
// The previous sitemap named one URL dated 2026-06-26. That is a GEO hole:
//   /control, /blog, and every article were invisible to crawlers even though
//   they prerender. This file is generated from the same blog SSOT the pages
//   render, plus the two public static routes that are not posts.
//
// Diligence is omitted on purpose. That page is noindex.
//
// Hard-fails if blog.js exports no posts, because a sitemap that silently
// dropped the writing URLs still looks complete (PCND).
//
// Regenerate with:   node site/scripts/gen-sitemap.mjs
// =============================================================================

import { writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '..');
const blog = await import(pathToFileURL(resolve(site, 'src/lib/blog.js')).href);
const { posts, sitemapPaths, SITE_ORIGIN, writing } = blog;

if (!posts?.length) throw new Error('gen-sitemap: blog.js exported no posts');

const newest = posts.map((p) => p.date).sort().at(-1);

const lastmodFor = (path) => {
  if (path.startsWith('/blog/') && path.endsWith('.html')) {
    const slug = path.slice('/blog/'.length, -'.html'.length);
    const post = posts.find((p) => p.slug === slug);
    return post?.date ?? newest;
  }
  return newest;
};

const locFor = (path) => (path === '/' ? `${SITE_ORIGIN}/` : `${SITE_ORIGIN}${path}`);

const urls = sitemapPaths().map((path) => ({ loc: locFor(path), lastmod: lastmodFor(path) }));

const body = [
  '<?xml version="1.0" encoding="UTF-8"?>',
  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
  ...urls.flatMap((u) => [
    '  <url>',
    `    <loc>${u.loc}</loc>`,
    `    <lastmod>${u.lastmod}</lastmod>`,
    '  </url>'
  ]),
  '</urlset>',
  ''
].join('\n');

const out = resolve(site, 'static/sitemap.xml');
writeFileSync(out, body);
console.log(
  `gen-sitemap: wrote ${out} (${urls.length} urls, writing index ${writing.indexHref}, newest ${newest})`
);
