#!/usr/bin/env node
// =============================================================================
// og.mjs — render static/og-image.png, the social card, from the brand masters
// -----------------------------------------------------------------------------
// The card is the full Avarok lockup on the dark ground with the front page's
// headline under it, set in the site's own IBM Plex, so a shared link looks
// like the page it opens. It is rendered by the same headless chromium the
// recorders use, from the built site, so the fonts and the SVG are the shipped
// bytes and nothing is re-typed here.
//
//   bun x --bun vite build && node scripts/media/og.mjs
// =============================================================================

import { chromium } from '@playwright/test';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from './serve.mjs';
import { SITE } from '../../src/lib/content/brand.js';
import { hero } from '../../src/lib/content/home.js';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '..', '..');
const repo = resolve(site, '..');
const BUILD = resolve(site, 'build');
const OUT = resolve(site, 'static', 'og-image.png');

if (!existsSync(resolve(BUILD, 'index.html'))) {
  console.error(`og: no build at ${BUILD}. Run \`bun x --bun vite build\` in site/ first.`);
  process.exit(1);
}

const lockup = readFileSync(resolve(repo, 'assets/brand/logo-full-ondark.svg'), 'utf8').replace(/<\?xml[^>]*>\s*/, '');
const lines = Array.isArray(hero.title) ? hero.title : String(hero.title).split('\n');
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');

const html = `<!doctype html>
<html><head><meta charset="utf-8">
<style>
  @font-face { font-family: 'IBM Plex Sans'; font-weight: 600; src: url('/fonts/ibm-plex-sans-latin-600-normal.woff2') format('woff2'); }
  @font-face { font-family: 'IBM Plex Sans'; font-weight: 400; src: url('/fonts/ibm-plex-sans-latin-400-normal.woff2') format('woff2'); }
  @font-face { font-family: 'IBM Plex Mono'; font-weight: 400; src: url('/fonts/ibm-plex-mono-latin-400-normal.woff2') format('woff2'); }
  html, body { margin: 0; width: 1200px; height: 630px; background: #0F1216; overflow: hidden; }
  body { position: relative; font-family: 'IBM Plex Sans', system-ui, sans-serif; color: #E4E7EC; }
  .glow { position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.55; }
  .a { width: 620px; height: 620px; left: -160px; top: -240px; background: #BE9DF8; opacity: 0.22; }
  .b { width: 560px; height: 560px; right: -180px; bottom: -260px; background: #49C3DB; opacity: 0.16; }
  .card { position: absolute; inset: 0; display: grid; grid-template-rows: 1fr auto; padding: 72px 84px 56px; box-sizing: border-box; }
  .lockup { display: flex; align-items: center; }
  .lockup svg { width: 560px; height: auto; }
  h1 { margin: 0; font-size: 46px; line-height: 1.12; font-weight: 600; letter-spacing: -0.02em; }
  h1 span { display: block; }
  h1 .dim { color: #82868F; }
  .foot { position: absolute; right: 84px; bottom: 60px; font-family: 'IBM Plex Mono', monospace; font-size: 18px; letter-spacing: 0.08em; color: #82868F; text-transform: uppercase; }
</style></head>
<body>
  <div class="glow a"></div><div class="glow b"></div>
  <div class="card">
    <div class="lockup">${lockup}</div>
    <h1>${lines.map((l, i) => `<span class="${i === lines.length - 1 && lines.length > 1 ? 'dim' : ''}">${esc(l)}</span>`).join('')}</h1>
  </div>
  <div class="foot">${esc(new URL(SITE).host)}</div>
</body></html>`;

const server = await serve(BUILD);
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
  await page.goto(`${server.origin}/`, { waitUntil: 'load' });
  await page.setContent(html, { waitUntil: 'load' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(150);
  await page.screenshot({ path: OUT, type: 'png' });
  console.log(`og: wrote ${OUT}`);
} finally {
  await browser.close();
  await server.close();
}
