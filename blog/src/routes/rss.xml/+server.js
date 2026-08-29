import { posts } from '$lib/posts.js';
import { SITE, blog, authors } from '$lib/content.js';

export const prerender = true;

/* Five characters, and every one of them will eventually appear in a title:
   an unescaped `&` alone makes the whole document unparseable, and most feed
   readers report that as "no items" rather than as an error. */
const xml = (s) =>
  String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');

export function GET() {
  const items = posts
    .map(
      (p) => `    <item>
      <title>${xml(p.title)}</title>
      <link>${SITE}${p.href}</link>
      <guid isPermaLink="true">${SITE}${p.href}</guid>
      <pubDate>${new Date(p.date).toUTCString()}</pubDate>
      <category>${xml(p.tag)}</category>
      <dc:creator>${xml(authors[p.author].name)}</dc:creator>
      <description>${xml(p.dek)}</description>
    </item>`
    )
    .join('\n');

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>${xml(blog.name)}</title>
    <link>${SITE}/</link>
    <atom:link href="${SITE}/rss.xml" rel="self" type="application/rss+xml"/>
    <description>${xml(blog.description)}</description>
    <language>en</language>
${items}
  </channel>
</rss>
`;
  return new Response(body, { headers: { 'content-type': 'application/rss+xml; charset=utf-8' } });
}
