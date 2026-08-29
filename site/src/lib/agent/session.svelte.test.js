// SPDX-License-Identifier: AGPL-3.0-only

// `probe()` is the background poll the launch dialog runs while it shows the
// "no agent yet" guide — and, since #803, while it shows a failure whose
// remedy sends the operator to a terminal.
//
// That second case only works because the poll is SILENT: it may move the
// dialog forward when an agent answers, and must not repaint what the operator
// is reading when it does not. Nothing pinned that, so #803's fix rested on an
// invariant a one-line edit could have removed without any test noticing.

import { test, expect } from 'bun:test';
import { LaunchSession } from '$lib/agent/session.svelte.js';

const refusing = (phase, message) => ({
  phase,
  message,
  async connect() {
    return false;
  },
});

test('a silent probe that fails changes nothing the operator is reading', async () => {
  const s = new LaunchSession();
  s.phase = 'failed';
  s.detail = 'Your agent is out of date — update it on that machine: curl … | sh';
  s.agent = refusing('error', 'a different, newer complaint');

  await s.probe();

  expect(s.phase).toBe('failed');
  expect(s.detail).toBe('Your agent is out of date — update it on that machine: curl … | sh');
  expect(s.busy).toBe(false);
});

test('a silent probe still moves FORWARD when an agent answers but wants pairing', async () => {
  // "Silent" suppresses the visible effects of a FAILED attempt, not of
  // progress: an agent that answered is news, and the dialog must act on it.
  const s = new LaunchSession();
  s.phase = 'guide';
  s.agent = refusing('unpaired', 'paste a token');

  await s.probe();

  expect(s.phase).toBe('pairing');
  expect(s.detail).toBe('paste a token');
});
