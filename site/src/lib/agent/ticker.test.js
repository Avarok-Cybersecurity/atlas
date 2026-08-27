// SPDX-License-Identifier: AGPL-3.0-only

// One timer for sixty cards, none for a page with no cards, and no way to drive
// the count negative — because a negative count means the next consumer starts
// no timer and every clock-driven badge on the page silently freezes, which is
// the bug the clock was added to fix, reintroduced by its own bookkeeping.

import { test, expect } from 'bun:test';
import { makeTicker } from './ticker.js';

/** Fake timers, so a test never waits a real second. */
function fakes() {
  let next = 1;
  const live = new Map();
  return {
    api: {
      setInterval: (fn) => {
        const id = next++;
        live.set(id, fn);
        return id;
      },
      clearInterval: (id) => live.delete(id)
    },
    count: () => live.size,
    fire: () => live.forEach((fn) => fn())
  };
}

test('idle until someone holds it', () => {
  const f = fakes();
  const t = makeTicker(() => {}, 1000, f.api);
  expect(t.running()).toBe(false);
  expect(f.count()).toBe(0);
});

test('one consumer starts exactly one timer, releasing stops it', () => {
  const f = fakes();
  const t = makeTicker(() => {}, 1000, f.api);
  const off = t.acquire();
  expect(t.running()).toBe(true);
  expect(f.count()).toBe(1);
  off();
  expect(t.running()).toBe(false);
  expect(f.count()).toBe(0);
});

test('sixty consumers still cost one timer', () => {
  const f = fakes();
  const t = makeTicker(() => {}, 1000, f.api);
  const offs = Array.from({ length: 60 }, () => t.acquire());
  expect(f.count()).toBe(1);
  expect(t.users()).toBe(60);
  offs.slice(1).forEach((o) => o());
  expect(t.running()).toBe(true);
  offs[0]();
  expect(t.running()).toBe(false);
});

test('releasing twice is harmless', () => {
  const f = fakes();
  const t = makeTicker(() => {}, 1000, f.api);
  const a = t.acquire();
  const b = t.acquire();
  a();
  a();
  a();
  expect(t.users()).toBe(1);
  expect(t.running()).toBe(true);
  b();
  expect(t.running()).toBe(false);
  // And the next consumer still gets a working clock.
  const c = t.acquire();
  expect(t.running()).toBe(true);
  c();
});

test('it actually ticks while held', () => {
  const f = fakes();
  let n = 0;
  const t = makeTicker(() => (n += 1), 1000, f.api);
  const off = t.acquire();
  f.fire();
  f.fire();
  expect(n).toBe(2);
  off();
  f.fire();
  expect(n).toBe(2);
});
