import { test, expect } from 'bun:test';
import { readFileSync, existsSync } from 'node:fs';

/**
 * All three Atlas properties publish llms.txt (https://llmstxt.org), and each
 * points at the other two. That cross-linking is the whole value: an agent that
 * finds one has no other way to discover the rest — none of the three hosts is
 * derivable from the others.
 *
 * The blog's is generated at build time from the post index, so it is checked
 * against the BUILT file when one exists rather than against the source.
 */
const root = new URL('../../../', import.meta.url);
const read = (rel) => readFileSync(new URL(rel, root), 'utf8');

const HOSTS = {
  site: 'https://atlasinference.io',
  blog: 'https://blog.atlasinference.io',
  docs: 'https://docs.atlasinference.io'
};

/** The generated marketing-site file, and the generated book file. */
const FILES = {
  site: 'site/static/llms.txt',
  docs: 'book/src/llms.txt'
};

test('every property has an llms.txt', () => {
  for (const [name, rel] of Object.entries(FILES)) {
    expect(existsSync(new URL(rel, root)), `${name}: ${rel} is missing`).toBe(true);
    expect(read(rel).length, `${name}: ${rel} is empty`).toBeGreaterThan(200);
  }
  // The blog's is a prerendered route, so its source is the endpoint.
  expect(existsSync(new URL('blog/src/routes/llms.txt/+server.js', root))).toBe(true);
});

test.each(Object.entries(FILES))('%s llms.txt has the required shape', (name, rel) => {
  const t = read(rel);
  // llmstxt.org: an H1 name, then a blockquote summary.
  expect(t.startsWith('# '), `${name}: must open with an H1`).toBe(true);
  expect(t, `${name}: needs a > blockquote summary`).toMatch(/\n>\s+\S/);
  expect(t, `${name}: needs at least one H2 section`).toMatch(/\n## \S/);
});

test.each(Object.entries(FILES))('%s llms.txt links to the other two properties', (name, rel) => {
  const t = read(rel);
  for (const [other, host] of Object.entries(HOSTS)) {
    if (other === name) continue;
    expect(t, `${name} does not link to ${other} (${host})`).toContain(host);
  }
});

test('the blog endpoint emits links to the other two', () => {
  // Its output is generated, so assert on what it is built to emit.
  const src = read('blog/src/routes/llms.txt/+server.js');
  expect(src).toContain('MAIN_SITE');
  expect(src).toContain('DOCS_SITE');
  expect(src).toContain('llmstxt.org');
});

test('the built blog llms.txt lists every shipped post', async () => {
  const built = new URL('blog/build/llms.txt', root);
  if (!existsSync(built)) return; // only meaningful after a build
  const t = readFileSync(built, 'utf8');
  const { readdirSync } = await import('node:fs');
  const slugs = readdirSync(new URL('blog/src/lib/posts/', root))
    .filter((f) => /\.(svelte|md)$/.test(f))
    .map((f) => f.replace(/\.(svelte|md)$/, ''));
  expect(slugs.length).toBeGreaterThan(0);
  for (const s of slugs) {
    expect(t, `built llms.txt omits post "${s}"`).toContain(`/posts/${s})`);
  }
});
