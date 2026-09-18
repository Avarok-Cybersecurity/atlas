#!/usr/bin/env node
// =============================================================================
// record.mjs — record the clips in the media manifest
// -----------------------------------------------------------------------------
// One recording per clip in src/lib/content/media.js. A clip's name says what
// to open and where:
//
//   broll-<scene>     /broll?scene=<scene>     a procedural loop, served from
//                                              this repository's own build
//                                              (src/lib/broll/scenes.js)
//   console-<scene>   /console?scene=<scene>   the product mockup running a
//                                              scripted scene. The mockup is a
//                                              PRIVATE project and is not in
//                                              this repository. Start its
//                                              preview server and pass its
//                                              origin with --console-origin.
//
// Without --console-origin the console clips are skipped, not failed: the
// encoded files committed under static/media stay as they are, and encode.mjs
// leaves them alone. See media-brief/README.md for where the mockup lives.
//
// The page stamps window.__sceneStart on its first frame and sets
// window.__sceneDone when the scene ends; both are read here and written to
// .media-raw/manifest.json beside the raw recording, so encode.mjs can trim
// the page load off the front and cut each clip to its true length (a b-roll
// loop to exactly one period, which is what makes it seamless).
//
// Needs a build in ../../build (bun x --bun vite build) and the Playwright
// chromium the e2e suite already installs. Nothing leaves the machine.
//
//   node scripts/media/record.mjs
//   node scripts/media/record.mjs --console-origin http://127.0.0.1:4174
//   node scripts/media/record.mjs --only console-ask,broll-field --console-origin http://127.0.0.1:4174
// =============================================================================

import { chromium } from '@playwright/test';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serve } from './serve.mjs';
import { allClips } from '../../src/lib/content/media.js';
import { scenes as procedural } from '../../src/lib/broll/scenes.js';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '..', '..');
const BUILD = resolve(site, 'build');
const RAW = resolve(site, '.media-raw');
const MANIFEST = resolve(RAW, 'manifest.json');

function arg(name) {
  const eq = process.argv.find((a) => a.startsWith(`--${name}=`));
  if (eq) return eq.slice(name.length + 3);
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : '';
}
const only = arg('only');
const wanted = only ? new Set(only.split(',').map((s) => s.trim())) : null;
const consoleOrigin = (arg('console-origin') || process.env.AVAROK_CONSOLE_ORIGIN || '').replace(/\/$/, '');

if (!existsSync(resolve(BUILD, 'index.html'))) {
  console.error(`record: no build at ${BUILD}. Run \`bun x --bun vite build\` in site/ first.`);
  process.exit(1);
}
mkdirSync(RAW, { recursive: true });

// Dedupe: the manifest reuses console-ask for the hero and the first tour tab.
const clips = [...new Map(allClips.map((c) => [c.name, c])).values()].filter((c) => !wanted || wanted.has(c.name));
if (clips.length === 0) {
  console.error('record: nothing to record (check --only)');
  process.exit(1);
}

const manifest = existsSync(MANIFEST) ? JSON.parse(readFileSync(MANIFEST, 'utf8')) : {};
const server = await serve(BUILD);

/** Where a clip is recorded from, or null when its source is not available. */
function target(name) {
  if (name.startsWith('broll-')) {
    // Only the loops drawn by src/lib/broll/scenes.js have a page to record.
    // The other broll slots hold generated footage and are filled by install.mjs.
    const scene = name.slice('broll-'.length);
    return procedural[scene] ? `${server.origin}/broll?scene=${scene}` : undefined;
  }
  if (name.startsWith('console-')) return consoleOrigin ? `${consoleOrigin}/console?scene=${name.slice('console-'.length)}` : null;
  // Anything else is not recorded from a page: the film is assembled by
  // reel.mjs, and a slot filled by install.mjs has no page behind it.
  return undefined;
}

const browser = await chromium.launch();
let recorded = 0;
const skipped = [];
try {
  for (const clip of clips) {
    const url = target(clip.name);
    if (url === undefined) continue;
    if (!url) {
      skipped.push(clip.name);
      continue;
    }
    const context = await browser.newContext({
      viewport: { width: clip.width, height: clip.height },
      deviceScaleFactor: 1,
      colorScheme: 'dark',
      reducedMotion: 'no-preference',
      recordVideo: { dir: RAW, size: { width: clip.width, height: clip.height } }
    });
    const pageT0 = Date.now();
    const page = await context.newPage();
    page.on('pageerror', (e) => console.error(`  page error on ${url}: ${e.message}`));
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__sceneDone === true, null, { timeout: 90_000 });
    const sceneStart = await page.evaluate(() => window.__sceneStart ?? null);
    const sceneEnd = Date.now();
    // a beat after the end so the cut never lands on the last state change
    await page.waitForTimeout(700);
    const video = page.video();
    await context.close();
    await video.saveAs(resolve(RAW, `${clip.name}.webm`));
    await video.delete();
    const trim = sceneStart ? Math.max(0, (sceneStart - pageT0) / 1000) : 0.4;
    const seconds = sceneStart ? (sceneEnd - sceneStart) / 1000 : (sceneEnd - pageT0) / 1000 - trim;
    manifest[clip.name] = {
      // The mockup's address is local and private; only the path is kept.
      path: new URL(url).pathname + new URL(url).search,
      width: clip.width,
      height: clip.height,
      trim: Math.round(trim * 1000) / 1000,
      seconds: Math.round(seconds * 1000) / 1000,
      recordedAt: new Date().toISOString()
    };
    writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + '\n');
    recorded++;
    console.log(`recorded ${clip.name.padEnd(20)} ${seconds.toFixed(1)}s after a ${trim.toFixed(2)}s load`);
  }
} finally {
  await browser.close();
  await server.close();
}
if (skipped.length) {
  console.log(`record: skipped ${skipped.join(', ')}. The console mockup is private and not in this repository;`);
  console.log('        start its preview server and pass --console-origin to record these. The committed encodes are untouched.');
}
console.log(`record: ${recorded} raw recording(s) in ${RAW}. Next: node scripts/media/encode.mjs`);
