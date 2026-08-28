// SPDX-License-Identifier: AGPL-3.0-only

// describeError renders whatever the agent sent over the socket, and it is the
// function the UI reaches for precisely when something has already gone wrong.
// Throwing there replaces a legible failure with a broken page, and returning
// an empty string leaves the operator staring at a screen that says nothing
// happened. Both were reachable from a malformed frame.

import { expect, test } from 'bun:test';
import { describeError, looksLikeToken, versionAdvice, PROTOCOL_VERSION } from './protocol.js';

test('a token check answers "no" for junk instead of throwing', () => {
  for (const junk of [undefined, null, 42, {}, [], true]) {
    expect(looksLikeToken(junk)).toBe(false);
  }
  expect(looksLikeToken('a'.repeat(63))).toBe(false);
  expect(looksLikeToken('A'.repeat(64))).toBe(false); // hex is lowercase
  expect(looksLikeToken(`  ${'a'.repeat(64)}  `)).toBe(true);
});

test('a malformed bad_settings frame explains itself instead of throwing', () => {
  for (const errors of ['oops', { a: 1 }, 42, null, undefined]) {
    const msg = describeError({ code: 'bad_settings', errors });
    expect(typeof msg).toBe('string');
    expect(msg.length).toBeGreaterThan(0);
  }
});

test('bad_settings names the keys it was given, and says so when it has none', () => {
  expect(describeError({ code: 'bad_settings', errors: [{ key: 'gpu_util' }, { key: 'ctx' }] }))
    .toBe('The agent rejected these settings: gpu_util, ctx');
  // A member with no usable key still counts as a rejected setting.
  expect(describeError({ code: 'bad_settings', errors: [{}, { key: 7 }] }))
    .toBe('The agent rejected these settings: setting, setting');
  expect(describeError({ code: 'bad_settings', errors: [] }))
    .toBe('The agent rejected the settings but did not say which.');
});

test('an unnameable code never reaches the UI as an object', () => {
  const generic = 'The agent reported an unknown problem.';
  expect(describeError({ code: { n: 1 } })).toBe(generic);
  expect(describeError({ code: 42 })).toBe(generic);
  expect(describeError({ code: '' })).toBe(generic);
  expect(describeError({})).toBe(generic);
  expect(describeError(null)).toBe(generic);
  expect(describeError('a string')).toBe(generic);
  // An unrecognised but well-formed code is still worth showing verbatim.
  expect(describeError({ code: 'some_new_code' })).toBe('some_new_code');
});

test('the codes the agent actually sends still read as sentences', () => {
  expect(describeError({ code: 'not_paired' })).toContain('atlasctl agent token');
  expect(describeError({ code: 'unknown_recipe', recipe: 'qwen' })).toContain('qwen');
  expect(describeError({ code: 'docker_unavailable', detail: 'no socket' })).toContain('no socket');
  expect(describeError({ code: 'already_running' })).toContain('already running');
});

// versionAdvice is the first thing an operator sees when a fleet of older
// agents meets a newer page, so its two branches must name different remedies.
test('a version mismatch names the side that is behind', () => {
  expect(versionAdvice(PROTOCOL_VERSION, 1, PROTOCOL_VERSION).ok).toBe(true);
  const agentOld = versionAdvice(4, 1, 2);
  expect(agentOld.side).toBe('agent');
  expect(agentOld.message).toContain('install.sh');
  const pageOld = versionAdvice(2, 3, 4);
  expect(pageOld.side).toBe('page');
  expect(pageOld.message).toContain('Shift-R');
  // A welcome with no usable versions must not guess a side at random.
  expect(versionAdvice(4, undefined, null).ok).toBe(false);
});
