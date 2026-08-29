import { test, expect } from 'bun:test';
import { indexPosts, neighboursOf } from './postindex.js';
import { tags, authors } from './content.js';

const VOCAB = { tags, authors };
const TAG = Object.keys(tags)[0];
const AUTHOR = Object.keys(authors)[0];

const post = (over = {}) => ({
  meta: { title: 't', dek: 'd', date: '2026-01-01', tag: TAG, author: AUTHOR, readingMinutes: 3, ...over },
  default: () => {}
});

/* --- ordering ------------------------------------------------------------ */

test('orders newest first regardless of the order the glob returns', () => {
  // Deliberately given oldest-first and in a different key order than the
  // result, so a pass cannot come from the input already being sorted.
  const mods = {
    './posts/old.svelte': post({ date: '2024-05-01' }),
    './posts/new.svelte': post({ date: '2026-08-01' }),
    './posts/mid.svelte': post({ date: '2025-02-14' })
  };
  expect(indexPosts(mods, VOCAB).map((p) => p.slug)).toEqual(['new', 'mid', 'old']);
});

test('same-date posts order by slug, so the build is reproducible', () => {
  const a = indexPosts({ './posts/beta.svelte': post(), './posts/alpha.svelte': post() }, VOCAB);
  const b = indexPosts({ './posts/alpha.svelte': post(), './posts/beta.svelte': post() }, VOCAB);
  expect(a.map((p) => p.slug)).toEqual(['alpha', 'beta']);
  expect(b.map((p) => p.slug)).toEqual(a.map((p) => p.slug));
});

test('derives slug and href from the filename', () => {
  const [p] = indexPosts({ './posts/what-it-costs.svelte': post() }, VOCAB);
  expect(p.slug).toBe('what-it-costs');
  expect(p.href).toBe('/posts/what-it-costs');
});

/* --- the validations, each proven to actually fire ------------------------ */

test('rejects a post with no meta export', () => {
  expect(() => indexPosts({ './posts/x.svelte': { default: () => {} } }, VOCAB)).toThrow(/no `meta` export/);
});

test.each(['title', 'dek', 'date', 'tag', 'author', 'readingMinutes'])(
  'rejects a post missing meta.%s',
  (field) => {
    const bad = post();
    delete bad.meta[field];
    expect(() => indexPosts({ './posts/x.svelte': bad }, VOCAB)).toThrow(new RegExp(`meta\\.${field} is required`));
  }
);

test('rejects a tag that is not declared in content.js', () => {
  // The failure this prevents is silent: an undeclared tag renders an
  // uncoloured dot and the post is absent from every category page.
  expect(() => indexPosts({ './posts/x.svelte': post({ tag: 'kernels' }) }, VOCAB))
    .toThrow(/meta.tag "kernels" is not declared/);
});

test('rejects an author that is not declared in content.js', () => {
  expect(() => indexPosts({ './posts/x.svelte': post({ author: 'ghost' }) }, VOCAB))
    .toThrow(/meta.author "ghost" is not declared/);
});

test('rejects an unparseable date, which would otherwise sort unpredictably', () => {
  expect(() => indexPosts({ './posts/x.svelte': post({ date: 'last tuesday' }) }, VOCAB))
    .toThrow(/is not a date/);
});

test('accepts the vocabulary it declares — the validations are not blanket-rejecting', () => {
  // Without this, every rejection test above would still pass if indexPosts
  // simply threw on everything.
  for (const tag of Object.keys(tags)) {
    for (const author of Object.keys(authors)) {
      expect(indexPosts({ './posts/x.svelte': post({ tag, author }) }, VOCAB)).toHaveLength(1);
    }
  }
});

/* --- neighbours ----------------------------------------------------------- */

test('neighbours walks the index in reading order', () => {
  const posts = indexPosts(
    {
      './posts/a.svelte': post({ date: '2026-03-01' }),
      './posts/b.svelte': post({ date: '2026-02-01' }),
      './posts/c.svelte': post({ date: '2026-01-01' })
    },
    VOCAB
  );
  expect(neighboursOf(posts, 'b')).toMatchObject({ newer: { slug: 'a' }, older: { slug: 'c' } });
});

test('the ends of the index have no neighbour past them', () => {
  const posts = indexPosts(
    { './posts/a.svelte': post({ date: '2026-03-01' }), './posts/b.svelte': post({ date: '2026-01-01' }) },
    VOCAB
  );
  expect(neighboursOf(posts, 'a').newer).toBeNull();
  expect(neighboursOf(posts, 'b').older).toBeNull();
});

test('an unknown slug yields no neighbours rather than the last post', () => {
  // findIndex returns -1; without the guard, posts[-1+1] would hand back the
  // FIRST post as "older" and the prev/next rail would link a stranger.
  const posts = indexPosts({ './posts/a.svelte': post() }, VOCAB);
  expect(neighboursOf(posts, 'nope')).toEqual({ newer: null, older: null });
});

/* --- the real content passes its own rules -------------------------------- */

test('every shipped post satisfies the schema', async () => {
  // Not a duplicate of the unit tests above: those use fixtures, this asserts
  // the actual posts directory is well-formed, which is what breaks the build.
  const { Glob } = await import('bun');
  const files = [...new Glob('src/lib/posts/*.svelte').scanSync('.')];
  expect(files.length).toBeGreaterThan(0);
  for (const f of files) {
    const src = await Bun.file(f).text();
    expect(src, `${f} must export meta from a module script`).toMatch(/<script module>[\s\S]*export const meta\s*=/);
  }
});
