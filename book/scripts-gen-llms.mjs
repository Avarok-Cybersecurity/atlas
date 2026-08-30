#!/usr/bin/env node
/**
 * Generate book/src/llms.txt from SUMMARY.md — https://llmstxt.org
 *
 * From SUMMARY.md rather than by hand, because SUMMARY.md is already the book's
 * table of contents and a second one written by hand goes stale the first time
 * a chapter is added. mdBook copies non-markdown files in `src/` straight to the
 * build output, so this lands at https://docs.atlasinference.io/llms.txt.
 *
 *   node book/scripts-gen-llms.mjs        # writes book/src/llms.txt
 *   node book/scripts-gen-llms.mjs --check # fails if the file is out of date
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const SITE = 'https://docs.atlasinference.io';
const summary = readFileSync(resolve(here, 'src/SUMMARY.md'), 'utf8');

// `# Heading` starts a part; `- [Title](path)` (at any indent) is a chapter.
// Indented entries are sub-chapters and are kept out of the index: llms.txt is
// meant to be a map, and 200 leaf links is not a map.
const parts = [];
let cur = null;
for (const raw of summary.split('\n')) {
  const part = /^#{1,2}\s+(.+?)\s*$/.exec(raw);
  if (part) { cur = { name: part[1], items: [] }; parts.push(cur); continue; }
  const item = /^(\s*)[-*]\s+\[([^\]]+)\]\(([^)]+)\)/.exec(raw);
  if (item && item[1].length === 0 && cur) {
    const url = SITE + '/' + item[3].replace(/^\.\//, '').replace(/\.md$/, '.html');
    cur.items.push(`- [${item[2]}](${url})`);
  }
}
const withItems = parts.filter((p) => p.items.length);
if (!withItems.length) {
  console.error('gen-llms(book): parsed no chapters from SUMMARY.md — refusing to write an empty index');
  process.exit(1);
}

const body = `# The Atlas Book

> Documentation for Atlas — an open source, pure Rust and CUDA LLM inference
> engine. One ~75 MB binary, no Python, no PyTorch.

This is the reference documentation. It covers installing and running the
engine, the architecture of the kernel dispatch pipeline, per-crate references,
and operational guides. The rustdoc API reference is generated from the source
and published alongside it.

${withItems.map((p) => `## ${p.name}\n\n${p.items.join('\n')}`).join('\n\n')}

## Machine-readable

- [Full API reference](${SITE}/api/): rustdoc for every public crate
- [Print view](${SITE}/print.html): the entire book as one HTML document

## Optional

- [Atlas Inference](https://atlasinference.io): the project site — also at https://atlasinference.io/llms.txt
- [Engineering blog](https://blog.atlasinference.io): measured notes — also at https://blog.atlasinference.io/llms.txt
- [Source](https://github.com/Avarok-Cybersecurity/atlas): pure Rust and CUDA, AGPL-3.0-only
- [Recipes](https://github.com/Avarok-Cybersecurity/atlas-recipes): the model SSOT
- [Discord](https://discord.gg/RQcGakU2jW)
`;

const out = resolve(here, 'src/llms.txt');
if (process.argv.includes('--check')) {
  let curr = '';
  try { curr = readFileSync(out, 'utf8'); } catch {}
  if (curr !== body) {
    console.error(`gen-llms(book): ${out} is out of date. Run: node book/scripts-gen-llms.mjs`);
    process.exit(1);
  }
  console.log(`gen-llms(book): up to date (${withItems.length} parts)`);
} else {
  writeFileSync(out, body);
  console.log(`gen-llms(book): wrote ${out} (${withItems.length} parts, ${withItems.reduce((n, p) => n + p.items.length, 0)} chapters)`);
}
