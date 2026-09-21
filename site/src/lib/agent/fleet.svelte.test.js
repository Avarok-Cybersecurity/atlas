// SPDX-License-Identifier: AGPL-3.0-only

// The first tests for a rune module in this repo.
//
// `.svelte.js` files use `$state`, which is a compiler construct rather than a
// runtime function, so `bun test` could not import them at all — six modules
// with nine exported items, none of them reachable from a test. That is not a
// small gap: a latching-state regression in this very file reached main and had
// to be found by reading the call graph, because nothing here could be driven.
//
// `test-runes.js` (loaded via `--preload`) compiles the runes and resolves
// SvelteKit's `$lib` alias, which vite supplies during a build and bun does not.

import { test, expect } from 'bun:test';
import { preferredAddress, linkWarns, isStale, FleetSession } from '$lib/agent/fleet.svelte.js';

const addr = (cls, speedMbps = null, iface = cls) => ({ iface, addr: '10.0.0.1', class: cls, speedMbps });

test('preferredAddress never returns a virtual or loopback interface', () => {
  // These are reachable only from the machine itself, so offering one as the
  // address a PEER should dial is offering an address that cannot work.
  expect(preferredAddress({ addresses: [addr('loopback'), addr('virtual')] })).toBeNull();
  expect(preferredAddress({ addresses: [] })).toBeNull();
  const picked = preferredAddress({ addresses: [addr('virtual'), addr('ethernet')] });
  expect(picked.class).toBe('ethernet');
});

test('preferredAddress ranks fabrics above ethernet, and speed only breaks ties', () => {
  // A 1 Gb RoCE link still beats a 100 Gb ethernet one: the class is the
  // decision, speed is the tiebreak. Sorting on speed first would hand a
  // DGX pair the wrong interface.
  const picked = preferredAddress({
    addresses: [addr('ethernet', 100_000), addr('roce', 1_000)],
  });
  expect(picked.class).toBe('roce');

  const tie = preferredAddress({
    addresses: [addr('ethernet', 1_000, 'slow'), addr('ethernet', 10_000, 'fast')],
  });
  expect(tie.iface).toBe('fast');
});

test('an unverified link ranks below every known class but is still offered', () => {
  // Unverified is missing information, not a bad link — it must lose to
  // anything known, and still be returned when it is all there is.
  const beaten = preferredAddress({ addresses: [addr('unverified'), addr('wireless')] });
  expect(beaten.class).toBe('wireless');
  const only = preferredAddress({ addresses: [addr('unverified')] });
  expect(only.class).toBe('unverified');
});

test('linkWarns stays silent for fabrics and for unverified', () => {
  // Warning about `unverified` would be inventing a problem out of an absence
  // of information, which is the comment's own reasoning.
  expect(linkWarns('roce')).toBe(false);
  expect(linkWarns('infini_band')).toBe(false);
  expect(linkWarns('unverified')).toBe(false);
  expect(linkWarns('ethernet')).toBe(true);
  expect(linkWarns('wireless')).toBe(true);
});

test('isStale is measured against the sample, not the clock reading', () => {
  const now = 1_000_000;
  expect(isStale({ lastSeen: now }, now)).toBe(false);
  expect(isStale({ lastSeen: now - 1000 }, now)).toBe(false);
  // Far enough back that no plausible threshold calls it fresh.
  expect(isStale({ lastSeen: now - 60 * 60 * 1000 }, now)).toBe(true);
});

// --- the probe loop's state machine -------------------------------------
//
// This is the class of bug the harness exists for. A change to the unpaired
// branch stopped it rescheduling, and because every OTHER re-arm site gates on
// `mode === 'no_agent'`, the page latched in 'browser_unpaired' until a full
// reload. It reached main and was found by reading the call graph.
//
// No fake timers are needed: capture `setTimeout`, then invoke the callback it
// was handed. That drives one tick of the loop deterministically.

// ASYNC and awaited. A first version was `try { return fn(...) } finally
// { restore }` with an async `fn`, so the finally ran the instant the promise
// was created -- setTimeout was restored before the body under test ever
// called it, and the assertion saw zero scheduled timers. The failure looked
// like the code under test, which is the worst kind of test bug.
async function withCapturedTimers(fn) {
  const real = globalThis.setTimeout;
  const scheduled = [];
  globalThis.setTimeout = (cb, ms) => {
    scheduled.push({ cb, ms });
    return scheduled.length; // a token clearTimeout can accept
  };
  const realClear = globalThis.clearTimeout;
  globalThis.clearTimeout = () => {};
  try {
    return await fn(scheduled);
  } finally {
    globalThis.setTimeout = real;
    globalThis.clearTimeout = realClear;
  }
}

const fakeAgent = (phase) => ({
  phase,
  message: phase === 'unpaired' ? 'paste a token' : 'nothing answered',
  async connect() {
    return false;
  },
  async watchFleet() {},
});

test('an unpaired probe keeps probing, so a token pasted elsewhere is noticed', async () => {
  await withCapturedTimers(async (scheduled) => {
    const f = new FleetSession();
    f.agent = fakeAgent('unavailable');

    // `start`, not `retry`: the probe callback returns immediately unless the
    // session has been started, so a test that only calls `retry` schedules a
    // timer whose body never runs — and would pass or fail for the wrong
    // reason. Going through the real entry point is the point.
    await f.start({ watch: true });
    expect(f.mode).toBe('no_agent');
    expect(scheduled.length).toBe(1);

    // The agent comes up, but this browser has never paired with it.
    f.agent = fakeAgent('unpaired');
    await scheduled[0].cb();

    expect(f.mode).toBe('browser_unpaired');
    // THE POINT: it must have re-armed. `connect()` re-reads the stored token
    // every call, so this loop is exactly what notices one pasted on another
    // page. Returning here latched the UI.
    expect(scheduled.length).toBeGreaterThan(1);
  });
});

test('a successful probe clears the detail a failure left behind', async () => {
  // `#connect` has always cleared `detail` on success; the probe loop did not,
  // so a message written while the agent was refusing survived into 'live'.
  // Invisible today because the only consumer renders in a non-live branch —
  // which is exactly the kind of latent wrong-text that surfaces the moment
  // someone adds a second consumer.
  await withCapturedTimers(async (scheduled) => {
    const f = new FleetSession();
    f.agent = fakeAgent('unavailable');
    await f.start({ watch: true });
    expect(f.detail).not.toBe('');

    // The agent comes back.
    // A success path agent needs the watch surface too: `#openWatch` subscribes
    // and asks for the fleet, so a fake missing either throws inside the probe
    // rather than exercising it.
    f.agent = {
      phase: 'ready',
      message: '',
      async connect() {
        return true;
      },
      onEvent() {},
      async listNodes() {
        return { ok: true, nodes: [] };
      },
      async watchFleet() {
        return { ok: true };
      },
    };
    await scheduled[0].cb();

    expect(f.mode).toBe('live');
    expect(f.detail).toBe('');
  });
});
