// SPDX-License-Identifier: AGPL-3.0-only

/**
 * Rasterises static/icons/icon.svg into the PNG set the web app manifest
 * references. The SVG is the single source of truth for the mark — nothing
 * here redraws it, so the installed icon can never drift from the favicon.
 *
 * Run via `npm run icons` (build does this automatically).
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const HERE = dirname(fileURLToPath(import.meta.url));
const ICON_DIR = join(HERE, '..', 'static', 'icons');
const SOURCE = join(ICON_DIR, 'icon.svg');

/** Manifest background_color — must match static/manifest.webmanifest. */
const BACKGROUND = { r: 0x0c, g: 0x0d, b: 0x10, alpha: 1 };

/**
 * Maskable icons are cropped to an arbitrary shape by the launcher. The spec's
 * safe zone is the centre 80% circle, so the glyph is inset well inside it.
 */
const MASKABLE_SAFE_FRACTION = 0.86;

/**
 * The launcher draws its own plate for a maskable icon, so ours is removed
 * rather than shrunk — a rounded rect floating inside a squircle reads as a
 * mistake. Strips the `<g id="plate">` block from the shared source.
 */
function stripPlate(svg) {
  // Replaced rather than deleted: a fully transparent rect keeps the 512x512
  // canvas, so the glyph stays centred instead of being cropped to its bbox.
  const stripped = svg
    .toString('utf8')
    .replace(
      /<g id="plate">[\s\S]*?<\/g>/,
      '<rect width="512" height="512" fill="none" stroke="none"/>'
    );
  if (stripped.includes('id="plate"')) {
    throw new Error('gen-icons: could not strip the plate group from icon.svg');
  }
  return Buffer.from(stripped, 'utf8');
}

/** @type {{ file: string, size: number, maskable?: boolean }[]} */
const TARGETS = [
  { file: 'icon-192.png', size: 192 },
  { file: 'icon-512.png', size: 512 },
  { file: 'apple-touch-icon.png', size: 180 },
  { file: 'icon-maskable-512.png', size: 512, maskable: true }
];

async function render(svg, target) {
  if (!target.maskable) {
    return sharp(svg, { density: 384 }).resize(target.size, target.size).png({ compressionLevel: 9 }).toBuffer();
  }

  const inner = Math.round(target.size * MASKABLE_SAFE_FRACTION);
  const glyph = await sharp(stripPlate(svg), { density: 384 })
    .resize(inner, inner)
    .png()
    .toBuffer();
  const offset = Math.round((target.size - inner) / 2);

  return sharp({
    create: {
      width: target.size,
      height: target.size,
      channels: 4,
      background: BACKGROUND
    }
  })
    .composite([{ input: glyph, top: offset, left: offset }])
    .png({ compressionLevel: 9 })
    .toBuffer();
}

const svg = await readFile(SOURCE);
await mkdir(ICON_DIR, { recursive: true });

for (const target of TARGETS) {
  const png = await render(svg, target);
  await writeFile(join(ICON_DIR, target.file), png);
  console.log(`icons: ${target.file} (${target.size}x${target.size}, ${png.length} B)`);
}
