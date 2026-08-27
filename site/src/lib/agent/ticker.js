// SPDX-License-Identifier: AGPL-3.0-only

// A reference-counted interval.
//
// Pure and separate from `clock.svelte.js` for the reason the rest of this
// directory is split that way: a file holding runes cannot be imported by a
// test runner, so anything living in it is untestable by construction. The
// interesting behaviour here is the counting — one timer for sixty cards, none
// for a page with no cards, and no way to drive the count negative — and that
// is all expressible without a single rune.

/**
 * Make a ticker that runs `onTick` every `ms` while at least one consumer holds
 * it.
 *
 * @param {() => void} onTick
 * @param {number} ms
 * @param {{setInterval: Function, clearInterval: Function}} [timers] injected
 *   for tests, which must not wait a real second to observe a tick
 */
export function makeTicker(
  onTick,
  ms,
  // Wrapped, not referenced: `{ setInterval }` calls the global as a method
  // of this object, and a browser's setInterval is this-sensitive — Chromium
  // throws "Illegal invocation" mid effect-flush, which took the reactive
  // clock (and every stale badge behind it) down on the live page while every
  // bun test kept passing, because bun's globals do not care who `this` is.
  timers = {
    setInterval: (fn, t) => setInterval(fn, t),
    clearInterval: (h) => clearInterval(h)
  }
) {
  let handle = null;
  let users = 0;

  return {
    /**
     * Register a consumer, starting the interval if it was idle.
     *
     * @returns {() => void} release; safe to call more than once
     */
    acquire() {
      users += 1;
      if (handle === null) handle = timers.setInterval(onTick, ms);
      let released = false;
      return () => {
        // Guarded because a component can unmount twice, and an effect's
        // cleanup can run again on re-run. Without this the count goes
        // negative, the next consumer finds `users > 0` untrue only after
        // several acquires, and every clock-driven badge on the page silently
        // stops updating.
        if (released) return;
        released = true;
        users -= 1;
        if (users <= 0) {
          users = 0;
          if (handle !== null) {
            timers.clearInterval(handle);
            handle = null;
          }
        }
      };
    },
    /** Whether the interval is running. */
    running() {
      return handle !== null;
    },
    /** How many consumers hold it. */
    users() {
      return users;
    }
  };
}
