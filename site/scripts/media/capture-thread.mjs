#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
// =============================================================================
// capture-thread.mjs — "The exchange" on the company page, from the real thread
// -----------------------------------------------------------------------------
// The company's story starts with four public comments on a pull request. The
// page shows them as GitHub drew them, not retyped, so nobody has to take our
// word for what was said. This opens the public thread in headless Chromium,
// in both colour schemes, and writes four WebP files:
//
//   static/media/about/exchange-{light,dark}.webp          the question and "Your point?"
//   static/media/about/exchange-before-{light,dark}.webp   the two comments before them
//
// Rendered with the clock set to UTC, so the dates in the picture are the dates
// the timeline beside it cites (the thread spans midnight UTC). The comments'
// words are also the images' alt text, in `exchange` in src/lib/content/company.js.
// Needs ffmpeg on the path, like the rest of scripts/media.
//
//   node scripts/media/capture-thread.mjs
// =============================================================================

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const site = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const out = join(site, 'static', 'media', 'about');
const tmp = join(site, '.media-raw', 'thread');
mkdirSync(out, { recursive: true });
mkdirSync(tmp, { recursive: true });

const THREAD = 'https://github.com/ggml-org/llama.cpp/pull/18680';
// The maintainer's close, the author's reply, the question, and the two words.
const COMMENTS = ['issuecomment-3721439645', 'issuecomment-3721446381', 'issuecomment-3721568646', 'issuecomment-3721611676'];
const SCALE = 2;

const browser = await chromium.launch();
for (const scheme of ['light', 'dark']) {
  const ctx = await browser.newContext({ viewport: { width: 1012, height: 1400 }, deviceScaleFactor: SCALE, colorScheme: scheme, timezoneId: 'UTC', locale: 'en-US' });
  const page = await ctx.newPage();
  await page.goto(THREAD, { waitUntil: 'domcontentloaded', timeout: 90_000 });
  await page.waitForTimeout(4000);
  const present = await page.evaluate((ids) => ids.every((id) => document.getElementById(id)), COMMENTS);
  if (!present) throw new Error(`capture-thread: the comments were not found on ${THREAD}. GitHub may have changed its markup or paginated the thread.`);
  await page.addStyleTag({ content: '.gh-header-sticky, .js-sticky, header[role="banner"] { visibility: hidden !important; } .js-notification-shelf, .flash { display: none !important; }' });
  const box = await page.evaluate((ids) => {
    const items = ids.map((id) => document.getElementById(id).closest('.TimelineItem, .js-timeline-item') || document.getElementById(id));
    const avatars = items.flatMap((e) => [...e.querySelectorAll('.TimelineItem-avatar, img.avatar')]);
    const rects = [...items, ...avatars].map((e) => e.getBoundingClientRect()).filter((r) => r.width > 0);
    const top = Math.min(...rects.map((r) => r.top)) + scrollY - 10;
    const bottom = Math.max(...rects.map((r) => r.bottom)) + scrollY + 10;
    const left = Math.max(0, Math.min(...rects.map((r) => r.left)) - 8);
    const right = Math.max(...rects.map((r) => r.right)) + 8;
    const split = items[2].getBoundingClientRect().top + scrollY - top - 16; // just above the question
    return { x: left, y: top, width: right - left, height: bottom - top, split };
  }, COMMENTS);
  await page.evaluate((y) => scrollTo(0, y - 80), box.y);
  await page.waitForTimeout(1200);
  const png = join(tmp, `thread-${scheme}.png`);
  await page.screenshot({ path: png, fullPage: true, clip: { x: box.x, y: box.y, width: box.width, height: box.height } });
  const w = Math.round(box.width * SCALE);
  const h = Math.round(box.height * SCALE);
  const cut = Math.round(box.split * SCALE);
  const webp = (name, crop, q) => execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-i', png, '-vf', `crop=${crop}`, '-c:v', 'libwebp', '-q:v', String(q), '-compression_level', '6', join(out, name)]);
  webp(`exchange-${scheme}.webp`, `${w}:${h - cut}:0:${cut}`, 86);
  webp(`exchange-before-${scheme}.webp`, `${w}:${cut}:0:0`, 84);
  console.log(`capture-thread: ${scheme}: exchange ${w}x${h - cut}, before ${w}x${cut}`);
  await ctx.close();
}
await browser.close();
rmSync(tmp, { recursive: true, force: true });
console.log('capture-thread: if the sizes changed, update `exchange.image` and `exchange.before.image` in src/lib/content/company.js, then `bun run guide`.');
