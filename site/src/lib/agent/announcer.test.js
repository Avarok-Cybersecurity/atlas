// SPDX-License-Identifier: AGPL-3.0-only

import { test, expect } from 'bun:test';
import { makeAnnouncer, ANNOUNCE_DEBOUNCE_MS } from './announce.js';

/** A controllable clock, so no test waits 1.5 real seconds. */
function fakeTimers() {
  let next = 1;
  const pending = new Map();
  return {
    api: {
      setTimeout: (fn, ms) => { const id = next++; pending.set(id, { fn, ms }); return id; },
      clearTimeout: (id) => { pending.delete(id); }
    },
    fire() { for (const { fn } of [...pending.values()]) fn(); pending.clear(); },
    count: () => pending.size
  };
}

const critical = [{ node: 'a', kind: 'thermal', severity: 'critical', detail: 'hot' }];

test('an announcement survives repeated updates inside the debounce window', () => {
  // THE regression. `fleet.alerts` derives from `fleet.nodes`, which is rebuilt
  // on every ~1 Hz vitals event, so the component re-renders many times inside
  // the 1500 ms window. When the timer lived in the effect's cleanup, each
  // re-run cancelled the pending announcement and nothing re-armed it — the
  // live region stayed empty forever on any wire carrying telemetry.
  const timers = fakeTimers();
  let said = '';
  const a = makeAnnouncer((t) => (said = t), timers.api);

  a.update(critical);
  for (let i = 0; i < 10; i += 1) a.update(critical); // vitals ticks, no change
  timers.fire();

  expect(said).not.toBe('');
  expect(said.toLowerCase()).toContain('critical');
});

test('a storm that escalates inside the window is read once, at its worst', () => {
  const timers = fakeTimers();
  const said = [];
  const a = makeAnnouncer((t) => said.push(t), timers.api);

  a.update([{ node: 'a', kind: 'k', severity: 'warning', detail: 'd' }]);
  a.update(critical);
  expect(timers.count()).toBe(1); // superseded, not queued twice
  timers.fire();

  expect(said.length).toBe(1);
  expect(said[0].toLowerCase()).toContain('critical');
});

test('an unchanged severity arms nothing at all', () => {
  const timers = fakeTimers();
  const a = makeAnnouncer(() => {}, timers.api);
  a.update(critical);
  timers.fire();
  a.update(critical);
  expect(timers.count()).toBe(0);
});

test('dispose drops a pending announcement', () => {
  const timers = fakeTimers();
  let said = '';
  const a = makeAnnouncer((t) => (said = t), timers.api);
  a.update(critical);
  a.dispose();
  timers.fire();
  expect(said).toBe('');
});

test('the debounce is the documented window', () => {
  expect(ANNOUNCE_DEBOUNCE_MS).toBe(1500);
});
