/**
 * The post index.
 *
 * A post is EITHER a Svelte component or a markdown file; the glob below
 * takes both. The Svelte shape exists because these posts
 * carry measured tables, annotated code blocks, callouts and inline figures,
 * none of which markdown expresses without a plugin stack — and that stack
 * (mdsvex, a markdown parser, a highlighter) would outweigh the entire WebGL
 * background by more than an order of magnitude. A post is a component that
 * exports its own front matter from `<script module>`.
 *
 * Adding a post: drop `src/lib/posts/<slug>.svelte` in place, exporting `meta`.
 * It appears on the index, on its tag page, on its author page, in the RSS
 * feed, in the sitemap and in the prev/next chain. Nothing else to register.
 *
 * The rules live in postindex.js so they can be tested without a bundler.
 */
import { tags, authors } from './content.js';
import { indexPosts, neighboursOf } from './postindex.js';

export const posts = indexPosts(
  import.meta.glob('./posts/*.{svelte,md}', { eager: true }),
  { tags, authors },
  // Drafts stay out of production entirely; a CI build sets the flag so the
  // example post is still exercised by the bundle and Lighthouse gates.
  { includeDrafts: import.meta.env?.VITE_DRAFTS === '1' }
);

export const bySlug = (slug) => posts.find((p) => p.slug === slug);
// Membership, not equality: a markdown post may sit in more than one category.
export const byTag = (tag) => posts.filter((p) => p.categories.includes(tag));
export const byAuthor = (author) => posts.filter((p) => p.author === author);
export const neighbours = (slug) => neighboursOf(posts, slug);

const FMT = new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });
export const formatDate = (iso) => FMT.format(new Date(iso));
