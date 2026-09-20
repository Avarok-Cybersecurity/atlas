// SPDX-License-Identifier: AGPL-3.0-only
//
// Procedural b-roll. Three ambient loops drawn on a canvas from the brand
// palette: no stock footage, no generated imagery, nothing to license.
//
// Every motion is a function of phase, phase = 2π·t/LOOP, with integer
// harmonics only, so the frame at phase 0 and the frame at phase 2π are the
// same frame and the encoded clip loops without a seam. broll.test.js proves
// that by drawing both frames into a recording context and comparing them.
//
// A scene is { alt, setup(rng) -> layout, draw(ctx, layout, phase) }. setup
// runs once with a seeded generator, so the layout is the same on every
// machine and every run; draw is pure in (layout, phase).

export const LOOP = 10; // seconds per loop
export const W = 1280; // logical frame, scaled to cover the canvas
export const H = 720;
export const TAU = Math.PI * 2;

export const ink = {
  bg: '#0F1216',
  bg2: '#141821',
  card: '#191E27',
  card2: '#1F252F',
  sunk: '#0A0C0F',
  border: '#24262B',
  borderStrong: '#303338',
  t3: '#82868F',
  violet: '#BE9DF8',
  cyan: '#49C3DB',
  green: '#12B981',
  gold: '#EFB338'
};
const HUES = [ink.violet, ink.cyan, ink.green, ink.gold];

// mulberry32: small, fast, and identical everywhere.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// FNV-1a, so a scene's seed is its name.
export function seedFrom(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

// Colour with alpha, alpha rounded so two frames that should match, match.
export function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  const al = Math.max(0, Math.min(1, Math.round(a * 1000) / 1000));
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${al})`;
}

const mod = (x, m) => ((x % m) + m) % m;

function glow(ctx, x, y, r, hue, a) {
  const g = ctx.createRadialGradient(x, y, 0, x, y, r);
  g.addColorStop(0, rgba(hue, a));
  g.addColorStop(1, rgba(hue, 0));
  ctx.fillStyle = g;
  ctx.fillRect(x - r, y - r, r * 2, r * 2);
}

function vignette(ctx, strength = 0.55) {
  const g = ctx.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, W * 0.72);
  g.addColorStop(0, rgba(ink.bg, 0));
  g.addColorStop(1, rgba(ink.bg, strength));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}

// One chevron of the mark: arm 320 wide by 280 tall, so a chevron `size` tall
// is size * 320 / 560 wide. Round caps and joins, like the artwork.
function chevron(ctx, x, y, size) {
  const h = size / 2;
  const w = (size * 320) / 560;
  ctx.beginPath();
  ctx.moveTo(x - w / 2, y - h);
  ctx.lineTo(x + w / 2, y);
  ctx.lineTo(x - w / 2, y + h);
  ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

// --- field: the chevron field, drifting --------------------------------------
// Three depth layers. Each layer tiles horizontally with its own period and
// drifts exactly one period per loop, so the far layers move slowly and the
// near layer quickly, and all three land back on frame 0 together.
const field = {
  alt: 'The Avarok chevron field, drifting.',
  setup(rng) {
    const layers = [
      { period: 520, count: 44, size: 24, width: 3.5, alpha: 0.14 },
      { period: 900, count: 28, size: 44, width: 6.5, alpha: 0.26 },
      { period: 1480, count: 15, size: 84, width: 12, alpha: 0.5 }
    ].map((l) => ({
      ...l,
      items: Array.from({ length: l.count }, () => ({
        x: rng() * l.period,
        y: rng() * H,
        hue: HUES[Math.floor(rng() * HUES.length)],
        bob: 4 + rng() * 10,
        m: 1 + Math.floor(rng() * 2),
        off: rng() * TAU,
        breathe: rng() * TAU
      }))
    }));
    return { layers };
  },
  draw(ctx, { layers }, phase) {
    ctx.fillStyle = ink.bg;
    ctx.fillRect(0, 0, W, H);
    glow(ctx, W * 0.32, H * 0.5, 540, ink.violet, 0.11 + 0.04 * Math.sin(phase));
    glow(ctx, W * 0.78, H * 0.62, 470, ink.cyan, 0.07 + 0.03 * Math.sin(phase + 2.1));
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const l of layers) {
      ctx.lineWidth = l.width;
      const shift = (phase / TAU) * l.period;
      const tiles = Math.ceil(W / l.period) + 1;
      for (const it of l.items) {
        const base = mod(it.x + shift, l.period);
        const y = it.y + it.bob * Math.sin(it.m * phase + it.off);
        ctx.strokeStyle = rgba(it.hue, l.alpha * (0.75 + 0.25 * Math.sin(phase + it.breathe)));
        for (let j = -1; j < tiles; j++) chevron(ctx, base + j * l.period, y, l.size);
      }
    }
    vignette(ctx, 0.5);
  }
};

// --- tokens: a river of tokens through a grid of GPUs -------------------------
// Every GPU carries a utilisation bar that breathes on an integer harmonic.
// Tokens run along the lanes between the rows, one or two laps per loop.
// The dashed links between neighbours carry a padlock: the connections in a
// mesh are encrypted end to end, and the picture says so without a caption.
const tokens = {
  alt: 'A river of tokens moving through a grid of GPUs.',
  setup(rng) {
    const cols = 8;
    const rows = 3;
    const cw = 118;
    const ch = 72;
    const gx = 34;
    const pitch = 152;
    const gridW = cols * cw + (cols - 1) * gx;
    const x0 = (W - gridW) / 2;
    const y0 = 128;
    const gpus = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const u = rng();
        gpus.push({
          x: x0 + c * (cw + gx),
          y: y0 + r * pitch,
          w: cw,
          h: ch,
          base: 0.5 + rng() * 0.2,
          amp: 0.12 + rng() * 0.2,
          n: 1 + Math.floor(rng() * 3),
          off: rng() * TAU,
          hue: u < 0.1 ? ink.violet : u < 0.24 ? ink.gold : u < 0.4 ? ink.cyan : ink.green
        });
      }
    }
    const lanes = [];
    for (let r = 0; r <= rows; r++) {
      const span = W + 120;
      lanes.push({
        y: y0 - 40 + r * pitch,
        k: 1 + (r % 2),
        span,
        items: Array.from({ length: 15 }, () => ({
          x: rng() * span,
          hue: [ink.cyan, ink.violet, ink.green][Math.floor(rng() * 3)],
          r: 2 + rng() * 2,
          wob: rng() * TAU
        }))
      });
    }
    return { gpus, lanes, cols, rows, cw, ch, gx };
  },
  draw(ctx, L, phase) {
    ctx.fillStyle = ink.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = rgba(ink.t3, 0.1);
    for (let x = 20; x < W; x += 40) for (let y = 20; y < H; y += 40) ctx.fillRect(x, y, 1.5, 1.5);
    glow(ctx, W * 0.5, H * 0.48, 620, ink.violet, 0.06 + 0.02 * Math.sin(phase));

    // links, dashed, dashes travelling three periods per loop
    ctx.setLineDash([6, 10]);
    ctx.lineDashOffset = -mod((phase / TAU) * 16 * 3, 16); // wrapped to the dash period, so the last frame is the first
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = rgba(ink.cyan, 0.32);
    for (const g of L.gpus) {
      const right = L.gpus.find((o) => o.y === g.y && o.x === g.x + g.w + L.gx);
      if (!right) continue;
      const y = g.y + g.h / 2;
      ctx.beginPath();
      ctx.moveTo(g.x + g.w, y);
      ctx.lineTo(right.x, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.lineDashOffset = 0;
    // padlocks at the midpoints
    for (const g of L.gpus) {
      const right = L.gpus.find((o) => o.y === g.y && o.x === g.x + g.w + L.gx);
      if (!right) continue;
      const mx = g.x + g.w + L.gx / 2;
      const my = g.y + g.h / 2;
      ctx.fillStyle = ink.bg;
      ctx.fillRect(mx - 7, my - 8, 14, 16);
      ctx.strokeStyle = rgba(ink.cyan, 0.7);
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(mx, my - 2, 3, Math.PI, 0);
      ctx.stroke();
      ctx.fillStyle = rgba(ink.cyan, 0.7);
      ctx.fillRect(mx - 4.5, my - 2, 9, 7);
    }

    // GPUs
    for (const g of L.gpus) {
      const util = Math.max(0.12, Math.min(0.98, g.base + g.amp * Math.sin(g.n * phase + g.off)));
      ctx.fillStyle = rgba(g.hue, 0.1 * util);
      roundRect(ctx, g.x - 6, g.y - 6, g.w + 12, g.h + 12, 12);
      ctx.fill();
      ctx.fillStyle = ink.card;
      ctx.strokeStyle = ink.borderStrong;
      ctx.lineWidth = 1;
      roundRect(ctx, g.x, g.y, g.w, g.h, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = rgba(ink.t3, 0.55);
      ctx.fillRect(g.x + 12, g.y + 13, 28, 4);
      ctx.fillRect(g.x + 12, g.y + 22, 18, 4);
      ctx.fillStyle = rgba(g.hue, 0.9);
      ctx.beginPath();
      ctx.arc(g.x + g.w - 14, g.y + 15, 3, 0, TAU);
      ctx.fill();
      ctx.fillStyle = rgba(ink.t3, 0.18);
      ctx.fillRect(g.x + 12, g.y + g.h - 20, g.w - 24, 6);
      ctx.fillStyle = rgba(g.hue, 0.45 + 0.5 * util);
      ctx.fillRect(g.x + 12, g.y + g.h - 20, (g.w - 24) * util, 6);
    }

    // tokens
    for (const lane of L.lanes) {
      const shift = (phase / TAU) * lane.span * lane.k;
      for (const it of lane.items) {
        const x = mod(it.x + shift, lane.span) - 60;
        const y = lane.y + 3 * Math.sin(2 * phase + it.wob);
        const g = ctx.createLinearGradient(x - 30, y, x, y);
        g.addColorStop(0, rgba(it.hue, 0));
        g.addColorStop(1, rgba(it.hue, 0.7));
        ctx.strokeStyle = g;
        ctx.lineWidth = it.r;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(x - 30, y);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.fillStyle = rgba(it.hue, 0.22);
        ctx.beginPath();
        ctx.arc(x, y, it.r * 3, 0, TAU);
        ctx.fill();
        ctx.fillStyle = rgba(it.hue, 0.95);
        ctx.beginPath();
        ctx.arc(x, y, it.r, 0, TAU);
        ctx.fill();
      }
    }
    vignette(ctx, 0.45);
  }
};

// --- rack: a datacenter aisle, pulsing with load -----------------------------
// Two walls of identical bays recede to a vanishing point; the camera dollies
// forward one bay per loop, so the far bay that enters is the near bay that
// left. LED brightness depends on screen depth rather than on which bay it is,
// which is what makes the dolly seamless.
const ZMAX = 14;
const ZC = 1.15; // camera distance to the z = 0 plane, world units
const F = 760; // focal length, px
const rack = {
  alt: 'Rack lights in a dark datacenter aisle, pulsing with load.',
  setup(rng) {
    const rowsPerBay = 13;
    const ledsPerRow = 5;
    const leds = [];
    for (let r = 0; r < rowsPerBay; r++) {
      for (let c = 0; c < ledsPerRow; c++) {
        const u = rng();
        leds.push({
          r,
          c,
          hue: u < 0.7 ? ink.green : u < 0.88 ? ink.cyan : u < 0.95 ? ink.gold : ink.violet,
          n: 1 + Math.floor(rng() * 4),
          off: rng() * TAU,
          zk: 0.5 + rng() * 0.9
        });
      }
    }
    const pulses = Array.from({ length: 6 }, () => ({ z0: rng() * ZMAX, hue: rng() < 0.55 ? ink.cyan : ink.violet, len: 0.5 + rng() * 0.7 }));
    return { leds, pulses, rowsPerBay, ledsPerRow };
  },
  draw(ctx, L, phase) {
    const vpx = W / 2;
    const vpy = H * 0.47;
    const P = (x, y, z) => {
      const s = F / (z + ZC);
      return [vpx + x * s, vpy - y * s, s];
    };
    ctx.fillStyle = ink.bg;
    ctx.fillRect(0, 0, W, H);
    // floor and ceiling
    let g = ctx.createLinearGradient(0, vpy, 0, H);
    g.addColorStop(0, ink.bg);
    g.addColorStop(1, ink.bg2);
    ctx.fillStyle = g;
    ctx.fillRect(0, vpy, W, H - vpy);
    g = ctx.createLinearGradient(0, 0, 0, vpy);
    g.addColorStop(0, ink.sunk);
    g.addColorStop(1, ink.bg);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, vpy);
    glow(ctx, vpx, vpy, 260, ink.cyan, 0.16 + 0.03 * Math.sin(phase));

    const frac = phase / TAU;
    const wallX = 1.0;
    const top = 0.58;
    const bottom = -0.58;
    for (let i = ZMAX + 1; i >= 0; i--) {
      const z = i - frac;
      if (z < -0.85) continue;
      const fog = Math.max(0, 1 - z / ZMAX);
      if (fog <= 0) continue;
      for (const side of [-1, 1]) {
        const x = side * wallX;
        const [ax, ay] = P(x, top, z);
        const [bx, by] = P(x, top, z + 0.86);
        const [cx, cy] = P(x, bottom, z + 0.86);
        const [dx, dy] = P(x, bottom, z);
        ctx.fillStyle = rgba(ink.card, 0.15 + 0.85 * fog);
        ctx.strokeStyle = rgba(ink.borderStrong, fog);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.lineTo(cx, cy);
        ctx.lineTo(dx, dy);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        // rails
        ctx.strokeStyle = rgba(ink.border, fog * 0.9);
        for (let r = 1; r < L.rowsPerBay; r++) {
          const y = top - (r / L.rowsPerBay) * (top - bottom);
          const [rx1, ry1] = P(x, y, z);
          const [rx2, ry2] = P(x, y, z + 0.86);
          ctx.beginPath();
          ctx.moveTo(rx1, ry1);
          ctx.lineTo(rx2, ry2);
          ctx.stroke();
        }
        // LEDs
        for (const led of L.leds) {
          const y = top - ((led.r + 0.5) / L.rowsPerBay) * (top - bottom);
          const lz = z + 0.1 + led.c * 0.16;
          const [lx, ly, s] = P(x, y, lz);
          const b = 0.5 + 0.5 * Math.sin(led.n * phase + led.zk * z + led.off);
          const size = Math.max(1.2, 0.022 * s);
          ctx.fillStyle = rgba(led.hue, fog * (0.08 + 0.2 * b));
          ctx.fillRect(lx - size * 1.6, ly - size * 1.6, size * 3.2, size * 3.2);
          ctx.fillStyle = rgba(led.hue, fog * (0.25 + 0.75 * b));
          ctx.fillRect(lx - size / 2, ly - size / 2, size, size);
        }
      }
    }
    // the aisle strip and its pulses
    const [sx1, sy1] = P(0, bottom, -0.6);
    const [sx2, sy2] = P(0, bottom, ZMAX);
    ctx.strokeStyle = rgba(ink.cyan, 0.16);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx1, sy1);
    ctx.lineTo(sx2, sy2);
    ctx.stroke();
    for (const p of L.pulses) {
      const z = mod(p.z0 - frac * ZMAX, ZMAX);
      const [px1, py1, s] = P(0, bottom, z);
      const [px2, py2] = P(0, bottom, z + p.len);
      const a = Math.max(0, 1 - z / ZMAX);
      ctx.strokeStyle = rgba(p.hue, 0.15 * a);
      ctx.lineWidth = Math.max(2, 0.05 * s);
      ctx.beginPath();
      ctx.moveTo(px1, py1);
      ctx.lineTo(px2, py2);
      ctx.stroke();
      ctx.strokeStyle = rgba(p.hue, 0.85 * a);
      ctx.lineWidth = Math.max(1, 0.012 * s);
      ctx.beginPath();
      ctx.moveTo(px1, py1);
      ctx.lineTo(px2, py2);
      ctx.stroke();
    }
    vignette(ctx, 0.6);
  }
};

export const scenes = { field, tokens, rack };
