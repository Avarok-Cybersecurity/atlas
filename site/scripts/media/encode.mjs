#!/usr/bin/env node
// =============================================================================
// encode.mjs — turn the raw recordings into the files the site ships
// -----------------------------------------------------------------------------
// For every clip in the media manifest that has a raw recording in
// .media-raw/<name>.webm:
//
//   static/media/<name>.mp4    H.264 high profile, yuv420p, faststart, 30 fps
//   static/media/<name>.webm   VP9, same frames
//   static/media/<name>.webp   the poster, one frame, the LCP image on the
//                              front page when the clip is the hero
//
// The trim and the length come from .media-raw/manifest.json, written by
// record.mjs: the page load is cut off the front, a console clip runs for its
// scene, a b-roll clip runs for exactly one loop so it repeats without a seam.
//
// A clip with no raw recording keeps its committed encode. That is the normal
// case for the console clips on a machine without the private mockup. A clip
// with neither a recording nor a committed encode is an error.
//
// A slot that install.mjs filled by hand is never overwritten here. Generated
// footage from the prompt pack replaces the procedural loops in the broll-*
// slots, and the procedural recordings are still sitting in .media-raw; without
// this guard the next `bun run media` would quietly put the placeholders back.
// static/media/provenance.json records which is which. --force overrides it.
//
// Budget: the front page loads four posters and, once they scroll near, four
// videos. Every file is reported; anything over 1.5 MB is flagged so it gets
// looked at before it ships. Needs ffmpeg with libx264, libvpx-vp9 and libwebp
// on PATH (or FFMPEG=/path/to/ffmpeg).
//
//   node scripts/media/encode.mjs
//   node scripts/media/encode.mjs --only broll-field
// =============================================================================

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { LOOP } from '../../src/lib/broll/scenes.js';
import { allClips } from '../../src/lib/content/media.js';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '..', '..');
const RAW = resolve(site, '.media-raw');
const OUT = resolve(site, 'static', 'media');
const FFMPEG = process.env.FFMPEG ?? 'ffmpeg';
const BUDGET = 1.5 * 1024 * 1024;

function arg(name) {
  const eq = process.argv.find((a) => a.startsWith(`--${name}=`));
  if (eq) return eq.slice(name.length + 3);
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : '';
}
const only = arg('only');
const wanted = only ? new Set(only.split(',').map((s) => s.trim())) : null;
const force = process.argv.includes('--force');

const manifestPath = resolve(RAW, 'manifest.json');
const manifest = existsSync(manifestPath) ? JSON.parse(readFileSync(manifestPath, 'utf8')) : {};
mkdirSync(OUT, { recursive: true });
const provPath = resolve(OUT, 'provenance.json');
const prov = existsSync(provPath) ? JSON.parse(readFileSync(provPath, 'utf8')) : {};

const run = (args) => execFileSync(FFMPEG, ['-y', '-hide_banner', '-loglevel', 'error', ...args], { stdio: ['ignore', 'inherit', 'inherit'] });
const mb = (p) => (statSync(p).size / 1024 / 1024).toFixed(2);
const flag = (p) => (statSync(p).size > BUDGET ? '  <-- over budget' : '');

const clips = [...new Map(allClips.map((c) => [c.name, c])).values()].filter((c) => !wanted || wanted.has(c.name));
let failed = 0;
let encoded = 0;
for (const clip of clips) {
  const raw = resolve(RAW, `${clip.name}.webm`);
  const m = manifest[clip.name];
  const mp4 = resolve(OUT, `${clip.name}.mp4`);
  const webm = resolve(OUT, `${clip.name}.webm`);
  const webp = resolve(OUT, `${clip.name}.webp`);
  if (prov[clip.name]?.kind === 'installed' && !force) {
    console.log(`${clip.name.padEnd(20)} installed by hand from ${prov[clip.name].from}, left alone (use --force to replace it)`);
    continue;
  }
  if (!existsSync(raw) || !m) {
    if ([mp4, webm, webp].every(existsSync)) {
      console.log(`${clip.name.padEnd(20)} no new recording, kept the committed encode`);
    } else {
      console.error(`encode: ${clip.name} has no recording and no committed encode. Run record.mjs --only ${clip.name}`);
      failed++;
    }
    continue;
  }
  const isLoop = clip.name.startsWith('broll-');
  const ss = m.trim.toFixed(3);
  // A loop is cut to its period. A console clip runs to the end of its scene.
  const length = isLoop ? LOOP : m.seconds;
  const t = length.toFixed(3);
  const vf = `scale=${clip.width}:${clip.height}:flags=lanczos,format=yuv420p`;

  run(['-ss', ss, '-i', raw, '-t', t, '-an', '-vf', vf, '-r', '30', '-c:v', 'libx264', '-preset', 'slow', '-crf', '27', '-profile:v', 'high', '-level', '4.0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4]);
  run(['-ss', ss, '-i', raw, '-t', t, '-an', '-vf', vf, '-r', '30', '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '36', '-row-mt', '1', '-deadline', 'good', '-cpu-used', '2', webm]);
  // The poster. A loop looks the same everywhere. A console scene is most
  // itself at the end, once the answer and its receipt, the finished rollout
  // or the ledger is on screen; the clip's `poster` fraction says where.
  const at = (m.trim + length * (clip.posterAt ?? (isLoop ? 0.3 : 0.94))).toFixed(3);
  run(['-ss', at, '-i', raw, '-frames:v', '1', '-vf', `scale=${clip.width}:${clip.height}:flags=lanczos`, '-c:v', 'libwebp', '-quality', '82', '-compression_level', '6', webp]);

  prov[clip.name] = { kind: 'recorded', from: m.path, at: new Date().toISOString().slice(0, 10) };
  encoded++;
  console.log(`${clip.name.padEnd(20)} ${t.padStart(7)}s  mp4 ${mb(mp4)} MB${flag(mp4)}  webm ${mb(webm)} MB${flag(webm)}  poster ${mb(webp)} MB`);
}
if (encoded) {
  const sorted = Object.fromEntries(Object.entries(prov).sort(([a], [b]) => a.localeCompare(b)));
  writeFileSync(provPath, `${JSON.stringify(sorted, null, 2)}${String.fromCharCode(10)}`);
}
if (failed) process.exit(1);
console.log(`encode: ${encoded} clip(s) written to ${OUT}, ${clips.length - encoded} kept`);
