import { posts, formatDate } from '$lib/posts.js';
import { SITE, MAIN_SITE, DOCS_SITE, blog, tags, authors, githubUrl, discordUrl, xUrl } from '$lib/content.js';

export const prerender = true;

/**
 * llms.txt — https://llmstxt.org
 *
 * Generated from the post index rather than written by hand, for the same
 * reason the sitemap and the feed are: three hand-maintained lists of the same
 * posts diverge the first time someone adds one. Publishing a stale index to
 * the agents that read this file is worse than publishing none, because they
 * have no way to tell.
 */
export function GET() {
  const line = (p) =>
    `- [${p.title}](${SITE}${p.href}): ${p.dek} (${tags[p.tag].name}, ${formatDate(p.date)}, ${p.readingMinutes} min)`;

  const byTag = Object.entries(tags)
    .map(([slug, t]) => {
      const items = posts.filter((p) => p.tag === slug);
      if (!items.length) return null;
      return [`### ${t.name}`, '', t.blurb, '', ...items.map(line), ''].join('\n');
    })
    .filter(Boolean);

  const body = `# ${blog.name}

> ${blog.description}

${blog.lede}

Posts are engineering notes with the measurement attached: what was measured, on
what hardware, with the harness and the commit. Where a number is quoted from
another source it is attributed as such.

## Posts

${posts.map(line).join('\n')}

## By category

${byTag.join('\n')}
## Authors

${Object.entries(authors).map(([slug, a]) => `- [${a.name}](${SITE}/authors/${slug}): ${a.role}. ${a.bio}`).join('\n')}

## Machine-readable

- [RSS feed](${SITE}/rss.xml): every post, newest first
- [Sitemap](${SITE}/sitemap.xml): every URL on this site

## Optional

- [Atlas Inference](${MAIN_SITE}): the engine this blog is about — also at ${MAIN_SITE}/llms.txt
- [Documentation](${DOCS_SITE}): the full book — also at ${DOCS_SITE}/llms.txt
- [Source](${githubUrl}): pure Rust and CUDA, AGPL-3.0-only
- [Discord](${discordUrl})
- [X](${xUrl})
`;

  return new Response(body, { headers: { 'content-type': 'text/plain; charset=utf-8' } });
}
