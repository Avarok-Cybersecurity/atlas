// SPDX-License-Identifier: AGPL-3.0-only
//
// The b-roll loops are only worth shipping if they loop. A clip that jumps at
// the seam reads as a bug on every page it sits on, and nobody finds it until
// the video has been recorded, encoded and deployed. So the property is tested
// where it is cheap: draw the frame at phase 0 and the frame at phase 2π into a
// context that records every call, and require the two recordings to match.
import { expect, test } from 'bun:test';
import { H, LOOP, TAU, W, mulberry32, scenes, seedFrom } from './scenes.js';
import { allClips } from '../content/media.js';

const round = (v) => (typeof v === 'number' ? Math.round(v * 1e4) / 1e4 : v);

/** A stand-in 2D context. Every call and every property set lands in `log`. */
function recorder() {
  const log = [];
  const gradient = (kind, args) => {
    const stops = [];
    return {
      addColorStop: (o, c) => stops.push(`${round(o)}:${c}`),
      toString: () => `${kind}(${args.map(round).join(',')})[${stops.join('|')}]`
    };
  };
  const ctx = new Proxy(
    {},
    {
      get(_, key) {
        if (key === 'createRadialGradient' || key === 'createLinearGradient') return (...a) => gradient(key, a);
        return (...a) => {
          log.push(`${String(key)}(${a.map((v) => (Array.isArray(v) ? `[${v.map(round)}]` : round(v))).join(',')})`);
        };
      },
      set(_, key, value) {
        log.push(`${String(key)}=${typeof value === 'object' ? String(value) : round(value)}`);
        return true;
      }
    }
  );
  return { ctx, log };
}

function frame(name, phase) {
  const def = scenes[name];
  const layout = def.setup(mulberry32(seedFrom(name)));
  const { ctx, log } = recorder();
  def.draw(ctx, layout, phase);
  return log;
}

for (const name of Object.keys(scenes)) {
  test(`${name}: the last frame of the loop is the first frame`, () => {
    const first = frame(name, 0);
    const last = frame(name, TAU);
    expect(first.length).toBeGreaterThan(50);
    expect(last.length).toBe(first.length);
    const firstDiff = first.findIndex((line, i) => line !== last[i]);
    expect(firstDiff === -1 ? 'seamless' : `call ${firstDiff}: ${first[firstDiff]} vs ${last[firstDiff]}`).toBe('seamless');
  });

  test(`${name}: it moves, so the match above is not a still image`, () => {
    expect(frame(name, Math.PI)).not.toEqual(frame(name, 0));
  });

  test(`${name}: the layout is the same on every run`, () => {
    const a = scenes[name].setup(mulberry32(seedFrom(name)));
    const b = scenes[name].setup(mulberry32(seedFrom(name)));
    expect(a).toEqual(b);
  });

  test(`${name}: has alternative text for the clip it becomes`, () => {
    expect(scenes[name].alt.length).toBeGreaterThan(20);
  });
}

test('every procedural scene has a matching b-roll slot at the logical frame size', () => {
  for (const name of Object.keys(scenes)) {
    const clip = allClips.find((c) => c.name === `broll-${name}`);
    expect(clip, `missing slot for scene ${name}`).toBeTruthy();
    expect([clip.width, clip.height]).toEqual([W, H]);
  }
});

test('generated-only b-roll slots are 1280 by 720 and have alt text', () => {
  const broll = allClips.filter((c) => c.name.startsWith('broll-'));
  for (const clip of broll) {
    expect([clip.width, clip.height]).toEqual([W, H]);
    expect(clip.alt.length).toBeGreaterThan(20);
  }
});

test('the loop is short enough to ship and long enough not to strobe', () => {
  expect(LOOP).toBeGreaterThanOrEqual(6);
  expect(LOOP).toBeLessThanOrEqual(14);
});
