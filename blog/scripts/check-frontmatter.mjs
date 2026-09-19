#!/usr/bin/env bun
// SPDX-License-Identifier: AGPL-3.0-only
//
// check-frontmatter.mjs — validate every markdown post, and say where.
//
// The rules live in src/lib/postmd.js, which is pure and unit-tested; this is
// only the I/O around them: find the posts, read them, and turn each finding
// into an annotation GitHub will pin to the offending line in the PR's
// Files-changed view.
//
// It validates ALL posts, not just the ones a PR touched. Checking only the
// diff would miss the case that actually bites — an author or a category
// removed from content.js, which invalidates posts nobody edited. Reading
// every post costs milliseconds.

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { authors, tags } from '../src/lib/content.js';
import { analysePost, splitFrontmatter } from '../src/lib/postmd.js';

const here = dirname(fileURLToPath(import.meta.url));
const postsDir = join(here, '..', 'src', 'lib', 'posts');
const staticDir = join(here, '..', 'static');

/** Line of the first occurrence of a key, so the annotation lands usefully. */
const lineOf = (src, needle) => {
  const i = src.indexOf(needle);
  return i < 0 ? 1 : src.slice(0, i).split('\n').length;
};

const files = readdirSync(postsDir).filter((f) => f.endsWith('.md'));
let failures = 0;
const rows = [];

for (const file of files) {
  const rel = `blog/src/lib/posts/${file}`;
  const src = readFileSync(join(postsDir, file), 'utf8');
  const problems = [];

  let analysed = null;
  try {
    analysed = analysePost(src, { tags, authors }, file);
  } catch (e) {
    for (const line of String(e.message).split('\n  - ').slice(1)) problems.push(line);
  }

  // Referenced files must exist. The pure validator cannot check this — it
  // never touches a filesystem — and a missing image is invisible in review
  // and obvious in production, which is the worst possible ordering.
  if (analysed) {
    const refs = analysed.images.map((i) => i.src);
    const card = analysed.meta['og-image'];
    if (card) refs.push(card);
    for (const m of splitFrontmatter(src).body.matchAll(/poster="([^"]+)"/g)) refs.push(m[1]);
    for (const ref of refs) {
      if (!existsSync(join(staticDir, ref.replace(/^\//, '')))) {
        problems.push(`referenced file does not exist: ${ref}`);
      }
    }
  }

  for (const p of problems) {
    const key = /`([a-z-]+)`/.exec(p)?.[1];
    console.log(`::error file=${rel},line=${key ? lineOf(src, `${key}:`) : 1}::${p}`);
    failures += 1;
  }
  rows.push(`| \`${file}\` | ${problems.length ? `❌ ${problems.length}` : '✅'} |`);
}

const summary = [`### Blog post frontmatter`, '', '| post | status |', '| --- | --- |', ...rows].join('\n');
console.log(summary);
if (process.env.GITHUB_STEP_SUMMARY) {
  const { appendFileSync } = await import('node:fs');
  appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${summary}\n`);
}

console.log(failures ? `\n${failures} problem(s) across ${files.length} post(s)` : `\n${files.length} post(s) valid`);
process.exit(failures ? 1 : 0);
