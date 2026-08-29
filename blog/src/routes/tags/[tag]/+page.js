import { error } from '@sveltejs/kit';
import { tags, cleanSlug } from '$lib/content.js';
import { byTag } from '$lib/posts.js';

// Every declared category gets a page, including one with no posts yet — the
// header links all four, and a linked 404 is worse than an empty list.
export const entries = () => Object.keys(tags).map((tag) => ({ tag }));

export function load({ params }) {
  const slug = cleanSlug(params.tag);
  const tag = tags[slug];
  if (!tag) error(404, `No category named "${slug}"`);
  return { slug, tag, items: byTag(slug) };
}
