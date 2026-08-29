import { error } from '@sveltejs/kit';
import { authors } from '$lib/content.js';
import { byAuthor } from '$lib/posts.js';

export const entries = () => Object.keys(authors).map((slug) => ({ slug }));

export function load({ params }) {
  const author = authors[params.slug];
  if (!author) error(404, `No author named "${params.slug}"`);
  return { author, items: byAuthor(params.slug) };
}
