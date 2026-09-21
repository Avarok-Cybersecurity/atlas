import { error } from '@sveltejs/kit';
import { authors, cleanSlug } from '$lib/content.js';
import { byAuthor } from '$lib/posts.js';

export const entries = () => Object.keys(authors).map((slug) => ({ slug }));

export function load({ params }) {
  const slug = cleanSlug(params.slug);
  const author = authors[slug];
  if (!author) error(404, `No author named "${slug}"`);
  return { author, items: byAuthor(slug) };
}
