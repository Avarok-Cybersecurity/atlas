// SPDX-License-Identifier: AGPL-3.0-only

// describeError renders whatever the agent sent over the socket, and it is the
// function the UI reaches for precisely when something has already gone wrong.
// Throwing there replaces a legible failure with a broken page, and returning
// an empty string leaves the operator staring at a screen that says nothing
// happened. Both were reachable from a malformed frame.

import { expect, test } from 'bun:test';
import {
  describeError,
  looksLikeToken,
  normaliseToken,
  versionAdvice,
  PROTOCOL_VERSION,
} from './protocol.js';

test('a token check answers "no" for junk instead of throwing', () => {
  for (const junk of [undefined, null, 42, {}, [], true]) {
    expect(looksLikeToken(junk)).toBe(false);
  }
  expect(looksLikeToken('a'.repeat(63))).toBe(false);
  expect(looksLikeToken(`  ${'a'.repeat(64)}  `)).toBe(true);
});

test('a token that wrapped in the terminal is still the token', () => {
  // The agent prints it on a labelled line, so a narrow window wraps it and the
  // copy carries a newline through the middle of 64 perfect hex characters.
  // That was refused, while the operator looked straight at what they pasted.
  const t = 'a'.repeat(64);
  const wrapped = `${t.slice(0, 40)}\n${t.slice(40)}`;
  expect(normaliseToken(wrapped)).toBe(t);
  expect(normaliseToken(`${t.slice(0, 20)} ${t.slice(20)}`)).toBe(t);
});

test('an uppercased paste is normalised, not refused', () => {
  // The agent only ever emits lowercase, so uppercase means the paste came
  // through something that changed it. The BYTES are the same, and the agent
  // compares the string exactly — so the fix is to fold the case before
  // sending, not to teach the operator about hex.
  expect(normaliseToken('A'.repeat(64))).toBe('a'.repeat(64));
  expect(looksLikeToken('A'.repeat(64))).toBe(true);
});

test('removing whitespace cannot turn a wrong paste into a right one', () => {
  // The laxness has to stop somewhere: what survives must still be exactly 64
  // hex characters, so a short token padded with spaces stays refused.
  expect(normaliseToken(`  ${'a'.repeat(63)}  `)).toBe(null);
  expect(normaliseToken(`${'a'.repeat(64)} deadbeef`)).toBe(null);
  expect(normaliseToken('g'.repeat(64))).toBe(null);
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

// An agent that omits a detail field is not hypothetical — an older build
// simply does not send the newer ones. Interpolating a missing field put the
// literal "undefined" in front of an operator who is, by definition, already
// looking at something that went wrong.
test('a message never shows the word undefined for a field the agent omitted', () => {
  for (const error of [
    { code: 'unsupported_protocol' },
    { code: 'unsupported_protocol', min: 1 },
    { code: 'unsupported_protocol', min: null, max: null },
    { code: 'unknown_recipe' },
    { code: 'not_launchable' },
    { code: 'docker_unavailable' },
    { code: 'launch_failed' },
    { code: 'unknown_recipe', recipe: '' },
    { code: 'launch_failed', detail: '   ' }
  ]) {
    const msg = describeError(error);
    expect(typeof msg).toBe('string');
    expect(msg.length).toBeGreaterThan(0);
    expect(msg).not.toContain('undefined');
    expect(msg).not.toContain('null');
  }
});

test('a detail the agent DID send is still shown', () => {
  expect(describeError({ code: 'unsupported_protocol', min: 1, max: 2 })).toContain('1–2');
  expect(describeError({ code: 'unknown_recipe', recipe: 'qwen' })).toContain('qwen');
  expect(describeError({ code: 'not_launchable', reason: 'no gpu' })).toContain('no gpu');
  expect(describeError({ code: 'docker_unavailable', detail: 'no socket' })).toContain('no socket');
  expect(describeError({ code: 'launch_failed', detail: 'oom' })).toContain('oom');
});

// data.js declares `installerUrl` the one authority and says so: "a second copy
// is how the two drift". joincommand.js builds its one-liner from it; these two
// remediation messages carried their own copies of the value. They are what an
// operator is told to run when their agent is too old — a stale URL there is a
// dead end at exactly the wrong moment.
test('the reinstall instructions are built from the declared installer URL', async () => {
  const { installerUrl } = await import('../data.js');
  const agentOld = versionAdvice(4, 1, 2);
  expect(agentOld.side).toBe('agent');
  expect(agentOld.message).toContain(installerUrl);

  const noVersion = versionAdvice(4, undefined, null);
  expect(noVersion.message).toContain(installerUrl);
});
