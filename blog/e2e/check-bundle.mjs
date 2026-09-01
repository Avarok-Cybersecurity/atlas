#!/usr/bin/env bun
// SPDX-License-Identifier: AGPL-3.0-only
//
// check-bundle.mjs — prove that the markdown toolchain never reaches a reader.
//
// `posts.js` records why posts were Svelte components rather than markdown: a
// parser, a highlighter and a maths renderer would outweigh everything else the
// blog ships. Markdown is admitted only because all three now run in the BUILD.
// That is a claim, and claims rot — the first time someone imports
// `md/compile.js` from a component, the whole trade quietly inverts and nothing
// else in CI would notice.
//
// ── The distinction this gate has to get right ──────────────────────────────
// The rendered OUTPUT of those tools legitimately ships: a post's markup
// contains `<pre class="shiki">` spans and `<span class="katex">` trees, and
// the post route contains the string "katex.min.css". Grepping for "shiki" or
// "katex" therefore reports a leak on a perfectly clean build. What must be
// absent is the LIBRARY — its entry points and internals. So the scanner looks
// for API identifiers, and a positive control asserts the output IS present,
// because a scanner that finds nothing anywhere passes every build including a
// broken one.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const BUILD = new URL('../build/', import.meta.url).pathname;

/** Library entry points and internals. None of these may ship. */
const FORBIDDEN = [
  ['createHighlighter', 'shiki'],
  ['getSingletonHighlighter', 'shiki'],
  ['loadWasm', 'shiki/oniguruma'],
  ['renderToString', 'katex'],
  ['ParseError', 'katex'],
  ['__defineSymbol', 'katex internals'],
  ['new Marked', 'marked'],
  ['class Lexer', 'marked'],
  ['blockTokens', 'marked internals'],
  ['inlineTokens', 'marked internals'],
  ['compileMarkdown', 'the blog’s own compiler'],
  ['atlasMarkdown', 'the blog’s own preprocessor']
];

/** Total client JavaScript. A leak that survives minification still shows here. */
const JS_BUDGET_KB = 260;

const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]
  );

const fail = [];
const ok = (msg) => console.log(`  ok    ${msg}`);
const bad = (msg) => {
  fail.push(msg);
  console.log(`  FAIL  ${msg}`);
};

// ── 1. no library code in the client bundle ────────────────────────────────
const js = walk(join(BUILD, '_app', 'immutable')).filter((f) => f.endsWith('.js'));
if (js.length === 0) bad('no client JavaScript found — did the build run?');
let leaks = 0;
for (const file of js) {
  const text = readFileSync(file, 'utf8');
  for (const [needle, owner] of FORBIDDEN) {
    if (text.includes(needle)) {
      bad(`${file.replace(BUILD, '')} contains \`${needle}\` (${owner}) — a build tool reached the client`);
      leaks += 1;
    }
  }
}
if (leaks === 0) ok(`${js.length} client chunks carry no parser, highlighter or maths engine`);

const bytes = js.reduce((a, f) => a + statSync(f).size, 0);
const kb = bytes / 1024;
if (kb > JS_BUDGET_KB) bad(`client JavaScript is ${kb.toFixed(1)} KB, over the ${JS_BUDGET_KB} KB budget`);
else ok(`client JavaScript is ${kb.toFixed(1)} KB (budget ${JS_BUDGET_KB} KB)`);

// ── 2. positive control: the OUTPUT is there ───────────────────────────────
// Without this, every assertion above would also pass on a build that rendered
// nothing at all — which is the failure mode a scanner is least likely to
// notice and most likely to be trusted through.
// ★ SCOPE NOTE (2026-08-31). This used to render `blog-post-example.md`, a
// `draft: true` fixture that exercised EVERY block the pipeline can emit. That
// fixture was removed with the other pre-launch posts, so the maths (KaTeX /
// MathML) and code-highlighting (Shiki) assertions below have no subject any
// more and are NOT checked here. The pipeline still ships and still runs for
// any future post that uses those blocks — it is simply unguarded until a post
// or a fixture exercises it again. Tracked, so this does not rot silently.
const sample = join(BUILD, 'posts', 'seven-tenets-powering-atlas-inference.html');
let html = '';
try {
  html = readFileSync(sample, 'utf8');
} catch {
  bad(`${sample} is missing — the blog rendered no posts at all`);
}
if (html) {
  const must = [
    ['property="og:image"', 'a social card'],
    ['name="twitter:card"', 'a large-image card'],
    ['application/ld+json', 'structured data'],
    ['loading="eager"', 'the first image is not lazy (it is the LCP candidate)']
  ];
  for (const [needle, why] of must) {
    if (html.includes(needle)) ok(`sample post: ${why}`);
    else bad(`sample post is missing ${needle} — ${why}`);
  }
}

// ── 3. a post without maths must not pay for the stylesheet ────────────────
const plain = join(BUILD, 'posts', 'seven-tenets-powering-atlas-inference.html');
try {
  if (readFileSync(plain, 'utf8').includes('katex.min.css')) {
    bad('a post with no maths links the KaTeX stylesheet — it should be conditional');
  } else ok('a post with no maths does not link the KaTeX stylesheet');
} catch {
  bad(`${plain} is missing`);
}

// ── 4. the blog still has no runtime dependencies ──────────────────────────
const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
if (pkg.dependencies && Object.keys(pkg.dependencies).length) {
  bad(`package.json grew runtime dependencies: ${Object.keys(pkg.dependencies).join(', ')}`);
} else ok('no runtime dependencies — the markdown toolchain is all devDependencies');

console.log(fail.length ? `\n${fail.length} failure(s)` : '\nbundle check passed');
process.exit(fail.length ? 1 : 0);
