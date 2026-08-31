/**
 * The post index, as pure logic.
 *
 * Kept apart from `posts.js` on the SBIO line: discovering the post modules is
 * I/O (Vite's `import.meta.glob`, which only exists inside a Vite build), and
 * validating and ordering them is not. Splitting them is what lets the rules
 * below be tested directly instead of through a bundler.
 */

/**
 * @param {Record<string, {meta?: object, default?: unknown}>} modules  path -> module
 * @param {{tags: object, authors: object}} vocab  the declared categories and authors
 * @param {{includeDrafts?: boolean}} [opts]
 * @returns {Array<object>} newest first
 */
export function indexPosts(modules, { tags, authors }, { includeDrafts = false } = {}) {
  const out = [];
  for (const [path, mod] of Object.entries(modules)) {
    // Two accepted formats, one index. A markdown post has already been
    // through the compiler's own validation by the time it gets here; the
    // checks below still run on it, because they are the ones a unit test can
    // reach without a bundler.
    const slug = path.slice(path.lastIndexOf('/') + 1).replace(/\.(svelte|md)$/, '');
    const m = mod.meta;
    // Every one of these is a mistake that would otherwise render: a missing
    // tag shows an uncoloured dot and drops the post out of its category page,
    // a bad date sorts it to the top of the index forever. Fail the build.
    if (!m) throw new Error(`${path}: no \`meta\` export. Add a <script module> block.`);
    for (const k of ['title', 'dek', 'date', 'tag', 'author', 'readingMinutes']) {
      if (m[k] === undefined || m[k] === '') throw new Error(`${path}: meta.${k} is required`);
    }
    if (!tags[m.tag]) throw new Error(`${path}: meta.tag "${m.tag}" is not declared in content.js`);
    if (!authors[m.author]) throw new Error(`${path}: meta.author "${m.author}" is not declared in content.js`);
    if (Number.isNaN(Date.parse(m.date))) throw new Error(`${path}: meta.date "${m.date}" is not a date`);
    // A markdown post may rename its URL without renaming its file; the
    // override goes through the same duplicate check as a filename slug.
    const finalSlug = m.slug || slug;
    for (const c of m.categories ?? []) {
      if (!tags[c]) throw new Error(`${path}: category "${c}" is not declared in content.js`);
    }
    // Drafts are dropped HERE rather than at each consumer, so the index, the
    // feed, the sitemap, llms.txt and the prev/next chain all agree without
    // any of them knowing drafts exist.
    if (m.draft === true && !includeDrafts) continue;
    if (out.some((p) => p.slug === finalSlug)) throw new Error(`${path}: duplicate slug "${finalSlug}"`);
    out.push({
      // A Svelte post declares one `tag`; a markdown post declares
      // `categories`. Normalise to both shapes so every consumer downstream
      // keeps working unchanged whichever format wrote the post.
      categories: m.categories ?? [m.tag],
      keywords: m.keywords ?? [],
      ogImage: m.ogImage ?? '',
      hasMath: m.hasMath === true,
      updated: m.updated ?? null,
      canonical: m.canonical ?? null,
      format: m.format ?? 'svelte',
      ...m,
      slug: finalSlug,
      href: `/posts/${finalSlug}`,
      Component: mod.default
    });
  }
  // Newest first. Ties break on slug so the order is a property of the content
  // rather than of whatever order the glob happened to return.
  out.sort((a, b) => Date.parse(b.date) - Date.parse(a.date) || a.slug.localeCompare(b.slug));
  return out;
}

/** Adjacent posts in reading order. `newer` is the one above it on the index. */
export function neighboursOf(posts, slug) {
  const i = posts.findIndex((p) => p.slug === slug);
  if (i < 0) return { newer: null, older: null };
  return { newer: i > 0 ? posts[i - 1] : null, older: i < posts.length - 1 ? posts[i + 1] : null };
}
