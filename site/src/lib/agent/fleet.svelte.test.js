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
import { preferredAddress, linkWarns, isStale } from '$lib/agent/fleet.svelte.js';

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
