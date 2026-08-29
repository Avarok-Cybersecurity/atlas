import { error } from '@sveltejs/kit';
import { tags } from '$lib/content.js';
import { byTag } from '$lib/posts.js';

// Every declared category gets a page, including one with no posts yet — the
// header links all four, and a linked 404 is worse than an empty list.
export const entries = () => Object.keys(tags).map((tag) => ({ tag }));

export function load({ params }) {
  const tag = tags[params.tag];
  if (!tag) error(404, `No category named "${params.tag}"`);
  return { slug: params.tag, tag, items: byTag(params.tag) };
}
