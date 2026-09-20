// SPDX-License-Identifier: AGPL-3.0-only
//
// The sitemap, generated from the page registry rather than hand edited: a
// page that exists is in it, a page that is noindex or excluded is not, and
// the developer routes that live outside the registry are named once here.
import { pages, SITE } from '$lib/content/index.js';

export const prerender = true;

const DEVELOPER = [
  { path: '/engine', priority: 0.8 },
  { path: '/control', priority: 0.5 }
];

export function GET() {
  const today = new Date().toISOString().slice(0, 10);
  const urls = [
    ...pages.filter((p) => !p.noindex && p.sitemap !== false).map((p) => ({ path: p.path, priority: p.priority ?? 0.5 })),
    ...DEVELOPER
  ];
  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls
  .map(
    (u) => `  <url>
    <loc>${SITE}${u.path === '/' ? '/' : u.path}</loc>
    <lastmod>${today}</lastmod>
    <priority>${u.priority.toFixed(1)}</priority>
  </url>`
  )
  .join('\n')}
</urlset>
`;
  return new Response(body, { headers: { 'content-type': 'application/xml; charset=utf-8' } });
}
