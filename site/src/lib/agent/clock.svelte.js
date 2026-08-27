// SPDX-License-Identifier: AGPL-3.0-only

// A reactive clock, for UI that goes stale by the passage of time rather than
// by an event.
//
// `$derived(isStale(node))` reads `Date.now()` inside a derived that depends
// only on `node`. Svelte re-evaluates it when `node` changes — and a node that
// has stopped reporting never changes, which is precisely the case the stale
// badge exists to show. So the badge appeared only if something else happened
// to touch the node, and "last seen Ns ago" froze at whatever it read when the
// last update arrived.
//
// The counting lives in `ticker.js`, which is plain JS and therefore testable;
// this file is the one line of rune that cannot be.

import { makeTicker } from './ticker.js';

const TICK_MS = 1000;

let now = $state(Date.now());

const ticker = makeTicker(() => {
  now = Date.now();
}, TICK_MS);

/**
 * The current time, as a reactive read. Anything derived from it re-evaluates
 * once a second.
 *
 * @returns {number} milliseconds since the epoch
 */
export function nowMs() {
  return now;
}

/**
 * Hold the clock for the lifetime of an effect:
 *
 * ```js
 * $effect(() => useClock());
 * ```
 *
 * @returns {() => void} release
 */
export function useClock() {
  return ticker.acquire();
}
