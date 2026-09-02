// SPDX-License-Identifier: AGPL-3.0-only
//
// A component must not open a loopback connection while mounting.
//
// `new WebSocket('ws://127.0.0.1:34333/ws')` from https://atlasinference.io is a
// public->loopback request, which is Chrome's Local Network Access gate. When
// that runs in a mount-time `$effect`, the browser asks "Access other apps and
// services on this device" on first paint, before the visitor has touched
// anything. FleetPill did exactly that, guarded on `storedToken()` — a guard that
// holds for a first-time visitor and fails for every operator, because the token
// is permanent localStorage while the permission is a separate per-origin grant
// that is re-asked on a fresh profile or after a permissions reset.
//
// SCOPE, deliberately: `src/lib/components/` only. A *route* may still connect on
// arrival — /control exists to drive the agent, and it carries a button and copy
// that pre-explain the prompt. A shared component has no such context: it renders
// wherever it is placed, including on pages whose visitors have never heard of
// the agent. That asymmetry is the rule, so the test encodes it rather than
// banning connections outright.
//
// The harness has no DOM (see test-runes.js), so components cannot be mounted and
// this asserts over source. That is weaker than observing a socket, and the limit
// is written here rather than implied: it catches a connect *lexically inside* an
// `$effect`, not one reached through a helper the effect calls.
import { test, expect } from 'bun:test';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const DIR = new URL('./components/', import.meta.url).pathname;

/** Source of every `$effect(...)` body in a component, brace-matched. */
function effectBodies(src) {
  const bodies = [];
  for (let i = src.indexOf('$effect'); i !== -1; i = src.indexOf('$effect', i + 1)) {
    const open = src.indexOf('{', i);
    if (open === -1) continue;
    let depth = 0;
    for (let j = open; j < src.length; j++) {
      if (src[j] === '{') depth++;
      else if (src[j] === '}' && --depth === 0) {
        bodies.push(src.slice(open, j + 1));
        break;
      }
    }
  }
  return bodies;
}

const CONNECTORS = [/\bfleet\s*\.\s*start\s*\(/, /\bagent\s*\.\s*connect\s*\(/, /new\s+WebSocket\s*\(/];

test('no shared component opens a loopback connection while mounting', () => {
  const offenders = [];
  for (const f of readdirSync(DIR).filter((n) => n.endsWith('.svelte'))) {
    const src = readFileSync(join(DIR, f), 'utf8');
    for (const body of effectBodies(src)) {
      for (const re of CONNECTORS) {
        if (re.test(body)) offenders.push(`${f}: $effect calls ${re.source}`);
      }
    }
  }
  expect(
    offenders,
    `A mount-time connect makes the browser prompt for local-network access on ` +
      `first paint, before any user gesture. Move it behind the click that needs ` +
      `it.\n  ${offenders.join('\n  ')}`
  ).toEqual([]);
});

test('the guard can actually fail', () => {
  // Negative control: the matcher must fire on the exact shape that shipped.
  const shipped = `$effect(() => { if (asked) return; asked = true; fleet.start({ watch: false }); });`;
  const hit = effectBodies(shipped).some((b) => CONNECTORS.some((re) => re.test(b)));
  expect(hit, 'the detector no longer recognises the original FleetPill bug').toBe(true);
});
