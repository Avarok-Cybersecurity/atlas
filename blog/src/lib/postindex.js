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
 * @returns {Array<object>} newest first
 */
export function indexPosts(modules, { tags, authors }) {
  const out = [];
  for (const [path, mod] of Object.entries(modules)) {
    const slug = path.slice(path.lastIndexOf('/') + 1).replace(/\.svelte$/, '');
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
    if (out.some((p) => p.slug === slug)) throw new Error(`${path}: duplicate slug "${slug}"`);
    out.push({ ...m, slug, href: `/posts/${slug}`, Component: mod.default });
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
