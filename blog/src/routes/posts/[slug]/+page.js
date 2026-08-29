import { error } from '@sveltejs/kit';
import { cleanSlug } from '$lib/content.js';
import { posts, bySlug, neighbours } from '$lib/posts.js';

// Declared rather than left to the crawler. The index links every post today,
// but a draft or an unlinked post would silently stop being prerendered — and
// with `fallback: undefined` that means a 404 in production, not a slow page.
export const entries = () => posts.map((p) => ({ slug: p.slug }));

export function load({ params }) {
  const slug = cleanSlug(params.slug);
  const post = bySlug(slug);
  if (!post) error(404, `No post named "${slug}"`);
  return { post, ...neighbours(slug) };
}
