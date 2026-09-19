#!/usr/bin/env node
// =============================================================================
// install.mjs — put an approved file into a media slot
// -----------------------------------------------------------------------------
// The one command that swaps media on the site. It takes a file from anywhere
// (a generated clip from the prompt pack, a recording of the real product, a
// photograph) and writes the slot's files in the formats the site ships.
//
//   a video   ->  static/media/<slot>.mp4, .webm and the poster .webp
//   an image  ->  static/media/art/<slot>.webp, registered in
//                 src/lib/content/art.json with its alt text and its pages
//
// A video slot must already exist in src/lib/content/media.js, because a page
// has to reference it. An image slot is registered here and shows up on the
// pages listed for it in media-brief/shots.json, or the ones given with --pages.
//
// Generated video rarely loops. --crossfade 1 blends the last second into the
// first, which costs that second of length and removes the jump.
//
//   node scripts/media/install.mjs --from ~/Downloads/aisle.mp4 --as broll-rack --crossfade 1
//   node scripts/media/install.mjs --from ~/Downloads/vault.png --as art-finance
//   node scripts/media/install.mjs --from take3.mp4 --as console-ask --start 1.5 --length 16 --poster-at 0.9
//
// Look at the result before committing it. media-brief/README.md has the
// acceptance checks: no logos, no readable text, no product UI that was not
// recorded from the product.
// =============================================================================

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { allClips } from '../../src/lib/content/media.js';

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, '..', '..');
const FFMPEG = process.env.FFMPEG ?? 'ffmpeg';
const FFPROBE = process.env.FFPROBE ?? 'ffprobe';

function arg(name) {
  const eq = process.argv.find((a) => a.startsWith(`--${name}=`));
  if (eq) return eq.slice(name.length + 3);
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : '';
}
const die = (msg) => {
  console.error(`install: ${msg}`);
  process.exit(1);
};

const from = arg('from');
const slot = arg('as');
if (!from || !slot) die('usage: install.mjs --from <file> --as <slot> [--start s] [--length s] [--crossfade s] [--poster-at 0..1] [--alt "..."] [--pages /a,/b]');
const src = resolve(process.cwd(), from.replace(/^~(?=$|\/|\\)/, process.env.HOME ?? process.env.USERPROFILE ?? '~'));
if (!existsSync(src)) die(`no such file: ${src}`);
if (!/^[a-z0-9-]+$/.test(slot)) die(`a slot name is lowercase letters, digits and dashes, got "${slot}"`);

const run = (args) => execFileSync(FFMPEG, ['-y', '-hide_banner', '-loglevel', 'error', ...args], { stdio: ['ignore', 'inherit', 'inherit'] });
const probe = (args) => execFileSync(FFPROBE, ['-v', 'error', ...args], { encoding: 'utf8' }).trim();
const mb = (p) => (statSync(p).size / 1024 / 1024).toFixed(2);
const isImage = ['.png', '.jpg', '.jpeg', '.webp', '.avif', '.tif', '.tiff'].includes(extname(src).toLowerCase());

if (isImage) {
  // --- a still ---------------------------------------------------------------
  const outDir = resolve(site, 'static', 'media', 'art');
  mkdirSync(outDir, { recursive: true });
  const out = resolve(outDir, `${slot}.webp`);
  // 1600 wide is twice the widest the hero frame renders at. Never upscale.
  run(['-i', src, '-frames:v', '1', '-vf', "scale='min(1600,iw)':-2:flags=lanczos", '-c:v', 'libwebp', '-quality', '80', '-compression_level', '6', out]);
  const [width, height] = probe(['-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=x', out]).split('x').map(Number);

  const shots = JSON.parse(readFileSync(resolve(site, 'media-brief', 'shots.json'), 'utf8')).shots;
  const shot = shots.find((s) => s.slot === slot && s.pages) ?? shots.find((s) => s.slot === slot);
  const alt = arg('alt') || shot?.alt;
  if (!alt) die(`no alt text for ${slot}: pass --alt "..." or add the slot to media-brief/shots.json`);
  const pages = (arg('pages') ? arg('pages').split(',') : (shot?.pages ?? []).filter((p) => p.startsWith('/')))
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => {
      // Git Bash on Windows rewrites an argument that starts with a slash into
      // a path under its own install directory. Catch it rather than register it.
      if (/^[A-Za-z]:[\\/]/.test(p)) die(`--pages got "${p}": the shell rewrote it. Write it without the leading slash (platform/deployment) or set MSYS_NO_PATHCONV=1`);
      return p.startsWith('/') ? p : `/${p}`;
    });

  const manifestPath = resolve(site, 'src', 'lib', 'content', 'art.json');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  manifest[slot] = { alt, width, height, pages };
  writeFileSync(manifestPath, JSON.stringify(Object.fromEntries(Object.entries(manifest).sort(([a], [b]) => a.localeCompare(b))), null, 2) + '\n');
  console.log(`install: ${slot} -> static/media/art/${slot}.webp (${width}x${height}, ${mb(out)} MB)`);
  console.log(`         registered in src/lib/content/art.json for ${pages.length ? pages.join(', ') : 'no page yet (pass --pages)'}`);
} else {
  // --- a clip ----------------------------------------------------------------
  const clip = allClips.find((c) => c.name === slot);
  if (!clip) die(`${slot} is not a clip in src/lib/content/media.js. Add the slot there first, so a page can reference it. Known: ${[...new Set(allClips.map((c) => c.name))].join(', ')}`);
  const duration = Number(probe(['-show_entries', 'format=duration', '-of', 'csv=p=0', src]));
  if (!Number.isFinite(duration) || duration <= 0) die(`could not read a duration from ${src}`);
  const start = Number(arg('start') || 0);
  const fade = Number(arg('crossfade') || 0);
  const length = Math.min(Number(arg('length') || duration - start), duration - start);
  if (length <= fade * 2 + 0.5) die(`a ${length.toFixed(1)}s clip is too short for a ${fade}s crossfade`);
  const posterAt = Number(arg('poster-at') || clip.posterAt || 0.3);

  const outDir = resolve(site, 'static', 'media');
  const mp4 = resolve(outDir, `${slot}.mp4`);
  const webm = resolve(outDir, `${slot}.webm`);
  const webp = resolve(outDir, `${slot}.webp`);
  // Cover the slot's frame: scale up to fill, crop the overflow, so a 16:9
  // slot takes a 21:9 or a 4:3 source without letterboxing.
  const fit = `scale=${clip.width}:${clip.height}:force_original_aspect_ratio=increase:flags=lanczos,crop=${clip.width}:${clip.height},fps=30`;
  // With a crossfade the clip opens `fade` seconds in and ends by dissolving
  // into its own opening, so the last frame is the frame the loop restarts on.
  // trim drops the stream's frame rate and xfade refuses a stream without one,
  // so each branch states it again.
  const graph = fade
    ? `[0:v]${fit},split[a][b];[a]trim=0:${fade},setpts=PTS-STARTPTS,fps=30,settb=AVTB[head];[b]trim=${fade}:${length},setpts=PTS-STARTPTS,fps=30,settb=AVTB[body];[body][head]xfade=transition=fade:duration=${fade}:offset=${(length - 2 * fade).toFixed(3)},format=yuv420p[v]`
    : `[0:v]${fit},format=yuv420p[v]`;
  const input = ['-ss', start.toFixed(3), '-t', length.toFixed(3), '-i', src];
  run([...input, '-filter_complex', graph, '-map', '[v]', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '27', '-profile:v', 'high', '-level', '4.0', '-movflags', '+faststart', mp4]);
  run([...input, '-filter_complex', graph, '-map', '[v]', '-an', '-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '36', '-row-mt', '1', '-deadline', 'good', '-cpu-used', '2', webm]);
  const shipped = length - fade;
  run(['-ss', (shipped * posterAt).toFixed(3), '-i', mp4, '-frames:v', '1', '-c:v', 'libwebp', '-quality', '82', '-compression_level', '6', webp]);
  // Remember that this slot was filled by hand. encode.mjs reads this and will
  // not overwrite an installed clip with a procedural recording unless forced.
  const provPath = resolve(outDir, 'provenance.json');
  const prov = existsSync(provPath) ? JSON.parse(readFileSync(provPath, 'utf8')) : {};
  prov[slot] = { kind: 'installed', from: basename(src), at: new Date().toISOString().slice(0, 10) };
  const sorted = Object.fromEntries(Object.entries(prov).sort(([a], [b]) => a.localeCompare(b)));
  writeFileSync(provPath, `${JSON.stringify(sorted, null, 2)}${String.fromCharCode(10)}`);
  console.log(`install: ${slot} -> ${shipped.toFixed(1)}s  mp4 ${mb(mp4)} MB  webm ${mb(webm)} MB  poster ${mb(webp)} MB${fade ? `  (crossfaded ${fade}s)` : ''}`);
  if (statSync(mp4).size > 1.5 * 1024 * 1024) console.log('         the mp4 is over the 1.5 MB budget: shorten it with --length, or accept it knowingly');
}
