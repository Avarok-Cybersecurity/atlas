import { posts } from '$lib/posts.js';
import { SITE, tags, authors } from '$lib/content.js';

export const prerender = true;

export function GET() {
  const urls = [
    { loc: `${SITE}/`, lastmod: posts[0]?.date },
    ...posts.map((p) => ({ loc: `${SITE}${p.href}`, lastmod: p.date })),
    ...Object.keys(tags).map((t) => ({ loc: `${SITE}/tags/${t}` })),
    ...Object.keys(authors).map((a) => ({ loc: `${SITE}/authors/${a}` }))
  ];
  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls
  .map(
    (u) =>
      `  <url><loc>${u.loc}</loc>${u.lastmod ? `<lastmod>${u.lastmod.slice(0, 10)}</lastmod>` : ''}</url>`
  )
  .join('\n')}
</urlset>
`;
  return new Response(body, { headers: { 'content-type': 'application/xml; charset=utf-8' } });
}
