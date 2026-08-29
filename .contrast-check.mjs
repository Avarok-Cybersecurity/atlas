/**
 * Chevron-field contrast gate.
 *
 * The field brightens the page ground behind live text. This proves that with
 * the field at its shipped density, every text token still clears WCAG AA on
 * the ground it is painted on — for EVERY pixel and EVERY moment, not for a
 * sample of them.
 *
 * It differs from FIELD-NOTES.md's method deliberately. That measured 14 time
 * samples over the viewport and reported the worst it saw. Sampling cannot see
 * the case where all three depth layers happen to overlap on one pixel at one
 * instant, which is rare but not impossible — and it is exactly the case that
 * would put a line of metadata gray below AA. So this computes the ANALYTIC
 * BOUND instead: the most luminance the shader is capable of adding to any
 * pixel, whether or not that configuration was ever sampled.
 *
 *   max added colour = (sum of the three layer dims) x amt_max x unit-luma hue
 *
 * The bound is strictly stronger than the sampled figure, and it is why the
 * shipped density is 0.38 rather than 1.0.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p) => readFileSync(resolve(here, p), 'utf8');

/* ---------- inputs, each from its single source of truth ---------- */

const tokensCss = read('web-shared/atlas-tokens.css');
const shader = read('web-shared/gl/chevron-field.glsl');
const runtime = read('web-shared/gl/chevron-field.js');

const token = (name) => {
  const m = tokensCss.match(new RegExp(`--${name}:\\s*(#[0-9a-fA-F]{6})`));
  if (!m) throw new Error(`token --${name} not found in web-shared/atlas-tokens.css`);
  return m[1];
};

const density = (() => {
  const m = runtime.match(/density:\s*([0-9.]+)/);
  if (!m) throw new Error('density default not found in chevron-field.js');
  return Number(m[1]);
})();

/* The shader is the SSOT for the field's maths. This check hardcodes the three
   constants the bound is derived from, so it MUST fail if the shader's copy of
   them changes — otherwise the gate would keep passing against a field it no
   longer describes. */
const SHADER_ANCHORS = [
  ['amplitude',  'float amt = u_density * mask * (0.030 + sweep * 0.020);'],
  ['luma cap',   'col /= max(1.0, lum);'],
  ['layer count','for (int i = 0; i < 3; i++)'],
  ['unit luma',  'hue /= max(dot(hue, vec3(0.2126, 0.7152, 0.0722)), 1e-3);'],
];
for (const [what, anchor] of SHADER_ANCHORS) {
  if (!shader.includes(anchor)) {
    console.error(`FAIL: the shader no longer contains the ${what} line this check is derived from:`);
    console.error(`      ${anchor}`);
    console.error('      Re-derive the bound below before changing it.');
    process.exit(1);
  }
}

const AMT_MAX = 0.030 + 0.020;   // mask = 1 and the sweep at its peak
// The shader clamps accumulated luma to one layer's worth, so this is the
// ceiling on the colour vector's luma no matter how many layers overlap — see
// the note above `col /= max(1.0, lum)` in the shader.
const LUMA_CAP = 1.0;

/* ---------- colour maths ---------- */

const srgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const relLum = (v) => 0.2126 * lin(v[0]) + 0.7152 * lin(v[1]) + 0.0722 * lin(v[2]);
const contrast = (a, b) => {
  const [hi, lo] = [relLum(a), relLum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};
// The shader normalises each hue by its luma in sRGB VALUE space, before
// linearisation — that is what `dot(hue, vec3(.2126,.7152,.0722))` does to a
// non-linear value. Reproduced exactly, or the bound is wrong.
const unitLuma = (v) => {
  const y = Math.max(0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2], 1e-3);
  return v.map((c) => c / y);
};

const ground = srgb(token('bg'));
const hues = {
  violet: srgb(token('ch-violet')),
  cyan: srgb(token('ch-cyan')),
  green: srgb(token('ch-green')),
  gold: srgb(token('ch-gold')),
};
const texts = {
  't1 headings': token('t1'),
  't2 body': token('t2'),
  't3 metadata': token('t3'),
};

/* Worst ground the field can produce, per hue. */
const worstGround = {};
for (const [name, h] of Object.entries(hues)) {
  const add = unitLuma(h).map((c) => c * LUMA_CAP * AMT_MAX * density);
  worstGround[name] = ground.map((c, i) => Math.min(1, c + add[i]));
}

/* ---------- report ---------- */

const AA = 4.5;
let worstOverall = Infinity;
let failed = 0;

console.log(`ground ${token('bg')}   density ${density}   ` +
            `max added value = ${(LUMA_CAP * AMT_MAX * density).toFixed(4)} x unit-luma hue`);
console.log('');
const head = ['text token', 'bare ground', ...Object.keys(hues)];
const rows = [];
for (const [label, hex] of Object.entries(texts)) {
  const t = srgb(hex);
  const bare = contrast(t, ground);
  const cells = Object.keys(hues).map((k) => contrast(t, worstGround[k]));
  const worst = Math.min(...cells);
  worstOverall = Math.min(worstOverall, worst);
  if (worst < AA) failed++;
  rows.push([`${label} ${hex}`, bare.toFixed(2), ...cells.map((c) => c.toFixed(2))]);
}
const w = head.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i].length)));
const line = (cells) => cells.map((c, i) => c.padEnd(w[i])).join('  ');
console.log(line(head));
console.log(w.map((n) => '-'.repeat(n)).join('  '));
for (const r of rows) console.log(line(r));
console.log('');

if (failed) {
  console.error(`FAIL: ${failed} token(s) fall below AA ${AA}:1 under the worst ground the field can paint.`);
  console.error('      Lower `density` in web-shared/gl/chevron-field.js, or darken the token.');
  process.exit(1);
}
console.log(`PASS: every token clears AA ${AA}:1; tightest is ${worstOverall.toFixed(2)}:1.`);
