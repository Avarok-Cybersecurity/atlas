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
 * The real timers, wrapped so they are called on the global object.
 *
 * NOT `{ setInterval, clearInterval }`. That object's properties hold the
 * global functions, but calling `timers.setInterval(...)` invokes them with
 * `this === timers` — and a browser's `setInterval` is a method of `Window`
 * that WebIDL requires to be called on one. Chromium throws
 * `TypeError: Illegal invocation` on the first acquire, which took out every
 * clock-driven badge on the control page.
 *
 * Node and Bun's timers are plain functions and are not `this`-sensitive, so
 * the unit tests passed against a default that could never work in the browser
 * the code exists to run in. The arrow wrappers below call them on the global.
 */
function browserTimers() {
  return {
    setInterval: (fn, ms) => setInterval(fn, ms),
    clearInterval: (h) => clearInterval(h)
  };
}

/**
 * Make a ticker that runs `onTick` every `ms` while at least one consumer holds
 * it.
 *
 * @param {() => void} onTick
 * @param {number} ms
 * @param {{setInterval: Function, clearInterval: Function}} [timers] injected
 *   for tests, which must not wait a real second to observe a tick
 */
export function makeTicker(onTick, ms, timers = browserTimers()) {
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
