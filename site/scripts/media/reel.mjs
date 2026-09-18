#!/usr/bin/env node
// =============================================================================
// reel.mjs — assemble the product film from the cut in media-brief/reel.json
// -----------------------------------------------------------------------------
// The film is built, not edited by hand, so it can be rebuilt whenever a clip
// or a line of copy changes:
//
//   1. every caption is checked against the site's own copy. A sentence the
//      pages do not say fails the build, so the film cannot drift from the site
//   2. headless chromium renders one overlay per segment, in the site's fonts,
//      from the vector lockup: a lower third for ambient footage, a framed
//      window for product footage, a lockup card, an end card
//   3. ffmpeg cuts each segment (trim, speed, fit, overlay) and cross fades
//      them into one file, then encodes what the site ships:
//        static/media/reel.mp4, reel.webm and the poster reel.webp
//
// Sources, best first: the raw generated take in media-brief/takes (full
// quality, not committed), else the encoded slot in static/media. Product
// footage comes from the raw recording in .media-raw when it is there, else
// from the encoded slot. So the film builds from a plain clone, just softer.
//
// Product footage is shown in a window rather than full bleed so its "Demo
// data" chip stays in frame and its caption never covers the interface.
//
//   bun x --bun vite build && node scripts/media/reel.mjs
// =============================================================================

import { chromium } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { serve } from './serve.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '..', '..');
const repo = resolve(site, '..');
const BUILD = resolve(site, 'build');
const WORK = resolve(site, '.media-raw', 'reel');
const OUT = resolve(site, 'static', 'media');
const FFMPEG = process.env.FFMPEG ?? 'ffmpeg';
const load = (p) => import(pathToFileURL(resolve(site, p)).href);
const die = (m) => {
  console.error(`reel: ${m}`);
  process.exit(1);
};

if (!existsSync(resolve(BUILD, 'index.html'))) die(`no build at ${BUILD}. Run \`bun x --bun vite build\` in site/ first.`);
mkdirSync(WORK, { recursive: true });

const cut = JSON.parse(readFileSync(resolve(site, 'media-brief', 'reel.json'), 'utf8'));
const [W, H] = cut.size;
const FPS = cut.fps;
const XF = cut.crossfade;
const win = cut.window;

// --- 1. the captions are the site's own words --------------------------------
const modules = await Promise.all(['home', 'why', 'platform', 'pricing', 'company'].map((m) => load(`src/lib/content/${m}.js`)));
const copy = [];
const collect = (v) => {
  if (typeof v === 'string') copy.push(v);
  else if (Array.isArray(v)) v.forEach(collect);
  else if (v && typeof v === 'object') Object.values(v).forEach(collect);
};
modules.forEach(collect);
for (const seg of cut.segments) {
  const sentences = seg.caption.split(/(?<=[.?])\s+/).map((s) => s.replace(/[.?]$/, '').trim()).filter(Boolean);
  const stray = sentences.filter((s) => !copy.some((c) => c.includes(s)));
  if (stray.length) die(`segment "${seg.id}" says "${stray.join('", "')}", which is not in src/lib/content. Use a line the site already says.`);
}
const { routes, SITE } = await load('src/lib/content/brand.js');
const home = modules[0];

// --- 2. overlays ----------------------------------------------------------------
const svg = (name) => readFileSync(resolve(repo, 'assets', 'brand', name), 'utf8').replace(/<\?xml[^>]*>\s*/, '');
const lockupFull = svg('logo-full-ondark.svg');
const lockupH = svg('logo-horizontal-ondark.svg');
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const css = `
  @font-face { font-family: 'IBM Plex Sans'; font-weight: 600; src: url('/fonts/ibm-plex-sans-latin-600-normal.woff2') format('woff2'); }
  @font-face { font-family: 'IBM Plex Sans'; font-weight: 400; src: url('/fonts/ibm-plex-sans-latin-400-normal.woff2') format('woff2'); }
  @font-face { font-family: 'IBM Plex Mono'; font-weight: 600; src: url('/fonts/ibm-plex-mono-latin-600-normal.woff2') format('woff2'); }
  html, body { margin: 0; width: ${W}px; height: ${H}px; overflow: hidden; background: transparent; }
  body { position: relative; font-family: 'IBM Plex Sans', system-ui, sans-serif; color: #E4E7EC; }
  .label { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 15px; letter-spacing: 0.14em; text-transform: uppercase; color: #BE9DF8; display: flex; align-items: center; gap: 12px; }
  .label::before { content: ''; width: 22px; height: 2px; background: currentColor; border-radius: 2px; }
  .cap { font-weight: 600; letter-spacing: -0.02em; line-height: 1.12; margin: 0; }
  /* The masters carry their own width and height. Without this the lockup
     renders at 1365 px and fills the frame. */
  .mark svg { display: block; width: 100%; height: auto; }
  /* ambient: a lower third over full bleed footage */
  .scrim { position: absolute; inset: 0; background: linear-gradient(to top, rgba(15,18,22,0.94) 0%, rgba(15,18,22,0.55) 26%, rgba(15,18,22,0) 52%), linear-gradient(to bottom, rgba(15,18,22,0.6) 0%, rgba(15,18,22,0) 22%); }
  .lower { position: absolute; left: 72px; right: 72px; bottom: 62px; display: grid; gap: 14px; }
  .lower .cap { font-size: 46px; max-width: 980px; }
  .corner { position: absolute; left: 72px; top: 44px; width: 150px; }
  /* product: an opaque stage with a window cut out of it */
  .stage { position: absolute; inset: 0; background: radial-gradient(900px 520px at 12% 0%, rgba(190,157,248,0.20), transparent 62%), radial-gradient(820px 520px at 100% 100%, rgba(73,195,219,0.16), transparent 60%), #0F1216;
    clip-path: path(evenodd, 'M0 0H${W}V${H}H0Z M${win.x + win.radius} ${win.y}H${win.x + win.w - win.radius}A${win.radius} ${win.radius} 0 0 1 ${win.x + win.w} ${win.y + win.radius}V${win.y + win.h - win.radius}A${win.radius} ${win.radius} 0 0 1 ${win.x + win.w - win.radius} ${win.y + win.h}H${win.x + win.radius}A${win.radius} ${win.radius} 0 0 1 ${win.x} ${win.y + win.h - win.radius}V${win.y + win.radius}A${win.radius} ${win.radius} 0 0 1 ${win.x + win.radius} ${win.y}Z'); }
  .ring { position: absolute; left: ${win.x - 1}px; top: ${win.y - 1}px; width: ${win.w}px; height: ${win.h}px; border-radius: ${win.radius + 1}px; border: 1px solid rgba(190,157,248,0.35); box-shadow: 0 0 0 1px rgba(15,18,22,0.9), 0 30px 80px -20px rgba(0,0,0,0.8); }
  .under { position: absolute; left: ${win.x}px; right: ${win.x}px; top: ${win.y + win.h + 22}px; display: flex; align-items: baseline; gap: 22px; }
  .under .cap { font-size: 32px; }
  .under .label { flex-shrink: 0; }
  /* cards */
  .veil { position: absolute; inset: 0; background: rgba(15,18,22,0.7); }
  .center { position: absolute; inset: 0; display: grid; place-content: center; justify-items: center; gap: 34px; text-align: center; }
  .center .mark { width: 520px; }
  .center .cap { font-size: 40px; }
  .end { position: absolute; inset: 0; background: radial-gradient(900px 560px at 10% 0%, rgba(190,157,248,0.22), transparent 62%), radial-gradient(820px 520px at 100% 100%, rgba(73,195,219,0.16), transparent 60%), #0F1216; display: grid; place-content: center; justify-items: center; gap: 30px; text-align: center; padding: 0 140px; }
  .end .mark { width: 360px; }
  .end .cap { font-size: 40px; max-width: 900px; }
  .pill { display: inline-flex; padding: 16px 30px; border-radius: 999px; background: #662AF4; color: #fff; font-weight: 600; font-size: 24px; }
  .url { font-family: 'IBM Plex Mono', monospace; font-size: 18px; letter-spacing: 0.1em; text-transform: uppercase; color: #82868F; }
`;
function overlay(seg) {
  const cap = `<p class="cap">${esc(seg.caption)}</p>`;
  const label = seg.label ? `<span class="label">${esc(seg.label)}</span>` : '';
  if (seg.kind === 'ambient') return `<div class="scrim"></div><div class="corner mark">${lockupH}</div><div class="lower">${label}${cap}</div>`;
  if (seg.kind === 'product') return `<div class="stage"></div><div class="ring"></div><div class="under">${label}${cap}</div>`;
  if (seg.kind === 'lockup') return `<div class="veil"></div><div class="center"><div class="mark">${lockupFull}</div>${cap}</div>`;
  return `<div class="end"><div class="mark">${lockupFull}</div>${cap}<span class="pill">${esc(home.cta.primary.text)}</span><span class="url">${esc(new URL(SITE).host + routes.demo)}</span></div>`;
}

const server = await serve(BUILD);
const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });
  await page.goto(`${server.origin}/`, { waitUntil: 'load' });
  for (const seg of cut.segments) {
    await page.setContent(`<!doctype html><html><head><meta charset="utf-8"><style>${css}</style></head><body>${overlay(seg)}</body></html>`, { waitUntil: 'load' });
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(120);
    await page.screenshot({ path: resolve(WORK, `${seg.id}.png`), type: 'png', omitBackground: true });
  }
} finally {
  await browser.close();
  await server.close();
}
console.log(`reel: ${cut.segments.length} overlays rendered`);

// --- 3. segments ------------------------------------------------------------------
const run = (args) => execFileSync(FFMPEG, ['-y', '-hide_banner', '-loglevel', 'error', ...args], { stdio: ['ignore', 'inherit', 'inherit'] });
const rawManifestPath = resolve(site, '.media-raw', 'manifest.json');
const rawManifest = existsSync(rawManifestPath) ? JSON.parse(readFileSync(rawManifestPath, 'utf8')) : {};

/** The best available source for a segment, and how far in its scene starts. */
function source(seg) {
  if (seg.take && existsSync(resolve(site, 'media-brief', 'takes', seg.take))) return { file: resolve(site, 'media-brief', 'takes', seg.take), lead: 0, from: 'take' };
  const raw = resolve(site, '.media-raw', `${seg.slot}.webm`);
  if (seg.kind === 'product' && existsSync(raw) && rawManifest[seg.slot]) return { file: raw, lead: rawManifest[seg.slot].trim, from: 'raw recording' };
  const slot = resolve(OUT, `${seg.slot}.mp4`);
  if (!existsSync(slot)) die(`segment "${seg.id}" has no source: no take, no raw recording and no ${seg.slot}.mp4`);
  return { file: slot, lead: 0, from: 'encoded slot' };
}

const files = [];
const lengths = [];
for (const seg of cut.segments) {
  const png = resolve(WORK, `${seg.id}.png`);
  const out = resolve(WORK, `${seg.id}.mp4`);
  const enc = ['-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '14', '-pix_fmt', 'yuv420p', '-r', String(FPS), out];
  if (seg.kind === 'end') {
    run(['-loop', '1', '-t', String(seg.seconds), '-i', png, '-vf', `scale=${W}:${H},fps=${FPS},format=yuv420p`, ...enc]);
    lengths.push(seg.seconds);
  } else {
    const src = source(seg);
    const speed = seg.speed ?? 1;
    const d = (seg.out - seg.in) / speed;
    const input = ['-ss', (src.lead + seg.in).toFixed(3), '-t', (seg.out - seg.in).toFixed(3), '-i', src.file];
    const speedUp = `setpts=PTS/${speed}`;
    let graph;
    if (seg.kind === 'product') {
      // the footage sits in the window, the opaque stage goes over it
      graph = `color=c=0x0F1216:s=${W}x${H}:r=${FPS}:d=${d.toFixed(3)}[bg];[0:v]${speedUp},scale=${win.w}:${win.h}:flags=lanczos,fps=${FPS}[v];[bg][v]overlay=${win.x}:${win.y}:shortest=1[b];[b][1:v]overlay=0:0,format=yuv420p[o]`;
    } else {
      // full bleed, with the caption fading in and out over it
      const fadeOut = Math.max(0.6, d - 0.75).toFixed(3);
      graph = `[0:v]${speedUp},scale=${W}:${H}:force_original_aspect_ratio=increase:flags=lanczos,crop=${W}:${H},fps=${FPS}[v];[1:v]format=rgba,fade=in:st=0.2:d=0.5:alpha=1,fade=out:st=${fadeOut}:d=0.5:alpha=1[c];[v][c]overlay=0:0:shortest=1,format=yuv420p[o]`;
    }
    run([...input, '-loop', '1', '-t', d.toFixed(3), '-i', png, '-filter_complex', graph, '-map', '[o]', '-t', d.toFixed(3), ...enc]);
    lengths.push(d);
    console.log(`  ${seg.id.padEnd(11)} ${d.toFixed(1).padStart(5)}s  from the ${src.from}${speed !== 1 ? `, ${speed}x` : ''}`);
  }
  files.push(out);
}

// --- 4. cross fade into one film -----------------------------------------------------
const inputs = files.flatMap((f) => ['-i', f]);
let chain = '';
let last = '[0:v]';
let t = 0;
for (let i = 1; i < files.length; i++) {
  t += lengths[i - 1] - XF;
  const label = i === files.length - 1 ? '[film]' : `[x${i}]`;
  chain += `${last}[${i}:v]xfade=transition=fade:duration=${XF}:offset=${t.toFixed(3)}${label};`;
  last = label;
}
const total = lengths.reduce((a, b) => a + b, 0) - XF * (files.length - 1);
const master = resolve(WORK, 'reel-master.mp4');
run([...inputs, '-filter_complex', chain.replace(/;$/, ''), '-map', '[film]', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '15', '-pix_fmt', 'yuv420p', '-r', String(FPS), master]);

// --- 5. what the site ships -----------------------------------------------------------
const mp4 = resolve(OUT, 'reel.mp4');
const webm = resolve(OUT, 'reel.webm');
const webp = resolve(OUT, 'reel.webp');
run(['-i', master, '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '25', '-profile:v', 'high', '-level', '4.0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4]);
run(['-i', master, '-an', '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '35', '-row-mt', '1', '-deadline', 'good', '-cpu-used', '2', webm]);
// The poster is the lockup card: it says whose film this is before it plays.
const lockupAt = lengths.slice(0, 2).reduce((a, b) => a + b, 0) - XF * 2 + lengths[2] * 0.6;
run(['-ss', lockupAt.toFixed(3), '-i', master, '-frames:v', '1', '-c:v', 'libwebp', '-quality', '84', '-compression_level', '6', webp]);

const provPath = resolve(OUT, 'provenance.json');
const prov = existsSync(provPath) ? JSON.parse(readFileSync(provPath, 'utf8')) : {};
prov.reel = { kind: 'assembled', from: 'media-brief/reel.json', at: new Date().toISOString().slice(0, 10) };
writeFileSync(provPath, `${JSON.stringify(Object.fromEntries(Object.entries(prov).sort(([a], [b]) => a.localeCompare(b))), null, 2)}${String.fromCharCode(10)}`);

const mb = (p) => (statSync(p).size / 1024 / 1024).toFixed(2);
console.log(`reel: ${total.toFixed(1)}s  mp4 ${mb(mp4)} MB  webm ${mb(webm)} MB  poster ${mb(webp)} MB`);
