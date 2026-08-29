// SPDX-License-Identifier: AGPL-3.0-only

import { test, expect } from 'bun:test';
import { versionAdvice } from './protocol.js';
import { installCommandFor } from '../install/platform.js';
import { installerUrl, powershellInstallerUrl } from '../data.js';

// The remedy is now the CALLER's to supply, because only the caller knows
// which machine the visitor is on. These tests pass the unix line except
// where they are specifically about a Windows visitor.
const UNIX = 'curl -fsSL https://atlasinference.io/install.sh | sh';

test('a version the agent supports is not a problem', () => {
  expect(versionAdvice(4, 4, 4, UNIX)).toEqual({ ok: true });
  expect(versionAdvice(3, 1, 4, UNIX)).toEqual({ ok: true });
  expect(versionAdvice(1, 1, 4, UNIX)).toEqual({ ok: true });
});

test('an out-of-date agent is named as the agent, with the line that fixes it', () => {
  // The common case the day protocol 4 shipped: every existing agent speaks
  // something older. "Update whichever is older" made the operator work out
  // which side was behind from two numbers they have no reason to care about.
  const a = versionAdvice(4, 1, 1, UNIX);
  expect(a.ok).toBe(false);
  expect(a.side).toBe('agent');
  expect(a.message).toContain('Your agent is out of date');
  expect(a.message).toContain('install.sh');
  // It has to say WHERE to run it — every hop of this product is a different
  // machine, and the agent is not the one showing this page.
  expect(a.message).toContain('on that machine');
});

// The whole point of the parameter: a Windows operator with a stale agent used
// to be handed `curl … | sh`, which PowerShell cannot parse. The advice is about
// the LOOPBACK agent, so the visitor's own OS is the right one to ask about.
test('a Windows visitor is told to run the Windows line, not curl', () => {
  const win = installCommandFor('windows', {
    shellUrl: installerUrl,
    powershellUrl: powershellInstallerUrl
  }).command;
  const a = versionAdvice(4, 1, 1, win);
  expect(a.ok).toBe(false);
  expect(a.message).toContain('install.ps1');
  expect(a.message).not.toContain('curl');
  // and the unix visitor is unchanged
  expect(versionAdvice(4, 1, 1, UNIX).message).toContain('install.sh');
});

test('a stale page is named as the page, because updating the agent cannot fix it', () => {
  // A browser holding a cached bundle against a newer agent. Telling this
  // operator to reinstall the agent sends them to the wrong machine to fix a
  // problem that lives in their own tab.
  const a = versionAdvice(3, 4, 4, UNIX);
  expect(a.ok).toBe(false);
  expect(a.side).toBe('page');
  expect(a.message).toContain('This page is out of date');
  expect(a.message).toMatch(/reload/i);
  expect(a.message).not.toContain('install.sh');
});

test('a single supported version reads as one number, not a range', () => {
  expect(versionAdvice(4, 2, 2, UNIX).message).toContain('protocol 2,');
  expect(versionAdvice(4, 1, 3, UNIX).message).toContain('1–3');
});

test('a welcome that carries no version is not guessed at', () => {
  // Deciding which side is behind from a non-number would name a remedy at
  // random, and half the time send the operator to the wrong machine.
  for (const bad of [undefined, null, 'four', NaN, 1.5]) {
    const a = versionAdvice(4, bad, bad, UNIX);
    expect(a.ok).toBe(false);
    expect(a.message).toContain('did not say');
  }
});
