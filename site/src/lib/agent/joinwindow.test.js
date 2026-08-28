// SPDX-License-Identifier: AGPL-3.0-only

// The join guide's rules. The cases that matter most are the lying ones: a
// throttled tab must not make the countdown claim time that is gone, and the
// carry note must never show half a credential.

import { expect, test } from 'bun:test';
import {
  JOIN_PORT,
  STALL_AFTER_MS,
  WARN_UNDER_S,
  deadlineMs,
  dialHost,
  groupedCode,
  normalizeJoinReply,
  offerKind,
  remaining,
  shortForm,
  stalled
} from './joinwindow.js';
import { joinCommand, joinCommandPowerShell } from './joincommand.js';

// --- normalizeJoinReply ------------------------------------------------------

test('a good reply maps to the join shape', () => {
  const j = normalizeJoinReply({ code: '84315907', addresses: ['10.10.10.1'], expires_in_s: 600 });
  expect(j).toEqual({ code: '84315907', addresses: ['10.10.10.1'], expiresInS: 600 });
});

test('a reply with no code is null, not a half-filled join', () => {
  expect(normalizeJoinReply({ addresses: ['10.10.10.1'], expires_in_s: 600 })).toBeNull();
  expect(normalizeJoinReply({ code: '', addresses: ['10.10.10.1'] })).toBeNull();
  expect(normalizeJoinReply({ code: '   ' })).toBeNull();
  expect(normalizeJoinReply(null)).toBeNull();
  expect(normalizeJoinReply(undefined)).toBeNull();
});

test('malformed fields degrade to safe values rather than propagating junk', () => {
  const j = normalizeJoinReply({ code: ' 84315907 ', addresses: 'not-a-list', expires_in_s: 'soon' });
  expect(j).toEqual({ code: '84315907', addresses: [], expiresInS: 0 });
});

// --- offerKind ---------------------------------------------------------------

test('no join at all means the by-hand path', () => {
  expect(offerKind(null)).toBe('manual');
  expect(offerKind(undefined)).toBe('manual');
  expect(offerKind({ addresses: ['10.0.0.4'] })).toBe('manual');
});

test('a code with only loopback addresses is no_address, exactly when joinCommand is empty', () => {
  const join = { code: '84315907', addresses: ['127.0.0.1', '::1'] };
  expect(joinCommand(join)).toBe('');
  expect(offerKind(join)).toBe('no_address');
  expect(offerKind({ code: '84315907', addresses: [] })).toBe('no_address');
});

test('a dialable join offers the command', () => {
  expect(offerKind({ code: '84315907', addresses: ['10.10.10.1'] })).toBe('command');
});

// --- deadline + remaining ----------------------------------------------------

const T0 = 1_700_000_000_000;

test('the deadline is mint time plus the window', () => {
  expect(deadlineMs(T0, 600)).toBe(T0 + 600_000);
});

test('remaining at the boundaries: 61, 60, 1, 0, -1 seconds', () => {
  const dl = T0 + 600_000;
  expect(remaining(dl, dl - 61_000)).toEqual({ seconds: 61, label: '1:01', warning: false, expired: false });
  // Exactly WARN_UNDER_S left is not yet a warning; one ms less is.
  expect(remaining(dl, dl - 60_000)).toEqual({ seconds: 60, label: '1:00', warning: false, expired: false });
  expect(remaining(dl, dl - 59_999).warning).toBe(true);
  expect(remaining(dl, dl - 1_000)).toEqual({ seconds: 1, label: '0:01', warning: true, expired: false });
  expect(remaining(dl, dl)).toEqual({ seconds: 0, label: '0:00', warning: true, expired: true });
  // Past the deadline the label clamps at 0:00 — never a negative label.
  expect(remaining(dl, dl + 1_000)).toEqual({ seconds: 0, label: '0:00', warning: true, expired: true });
});

test('a throttled tab cannot make the countdown lie', () => {
  // The tab was frozen for nine minutes of a ten-minute window; the first tick
  // after it wakes must report the truth of the wall clock, not one decrement.
  const dl = deadlineMs(T0, 600);
  const afterFreeze = remaining(dl, T0 + 9 * 60_000);
  expect(afterFreeze.seconds).toBe(60);
  const afterLongerFreeze = remaining(dl, T0 + 11 * 60_000);
  expect(afterLongerFreeze.expired).toBe(true);
  expect(afterLongerFreeze.label).toBe('0:00');
});

test('the label reads 0:01 until the deadline has actually passed', () => {
  const dl = T0;
  expect(remaining(dl, dl - 1)).toEqual({ seconds: 1, label: '0:01', warning: true, expired: false });
  expect(remaining(dl, dl - 999)).toEqual({ seconds: 1, label: '0:01', warning: true, expired: false });
});

test('a window shorter than the warning threshold warns from the first tick', () => {
  const dl = deadlineMs(T0, 30);
  const r = remaining(dl, T0);
  expect(r.warning).toBe(true);
  expect(r.expired).toBe(false);
  expect(r.label).toBe('0:30');
});

test('minutes format as m:ss with zero-padded seconds', () => {
  const dl = T0 + 600_000;
  expect(remaining(dl, T0).label).toBe('10:00');
  expect(remaining(dl, T0 + 28_000).label).toBe('9:32');
  expect(remaining(dl, T0 + 558_000).label).toBe('0:42');
});

// --- stalled -----------------------------------------------------------------

test('the stall escalates exactly at the threshold, not before', () => {
  expect(stalled(T0, T0 + STALL_AFTER_MS - 1)).toBe(false);
  expect(stalled(T0, T0 + STALL_AFTER_MS)).toBe(true);
  expect(stalled(T0, T0 + STALL_AFTER_MS + 1)).toBe(true);
});

test('the stall threshold is the designed two minutes', () => {
  expect(STALL_AFTER_MS).toBe(120_000);
  expect(WARN_UNDER_S).toBe(60);
  expect(JOIN_PORT).toBe(34334);
});

// --- groupedCode -------------------------------------------------------------

test('the code groups in fours for reading aloud', () => {
  expect(groupedCode('84315907')).toBe('8431 5907');
});

test('odd lengths group without a trailing space', () => {
  expect(groupedCode('843159071')).toBe('8431 5907 1');
  expect(groupedCode('84315')).toBe('8431 5');
  expect(groupedCode('8431')).toBe('8431');
  expect(groupedCode('84')).toBe('84');
  expect(groupedCode('')).toBe('');
  expect(groupedCode(null)).toBe('');
});

// --- shortForm ---------------------------------------------------------------

test('the carry tail matches the command it came from', () => {
  const join = { code: '84315907', addresses: ['10.10.10.1'] };
  expect(shortForm(join)).toBe('84315907@10.10.10.1');
  // The invariant is that the tail on screen IS the operand in the command --
  // not that it is the command's last characters. `endsWith` stopped being able
  // to say that when the sh operand gained its quotes; both now come from
  // joinOperand, so assert the relationship that actually matters.
  expect(joinCommand(join)).toContain(`'${shortForm(join)}'`);
  expect(joinCommandPowerShell(join)).toContain(`'${shortForm(join)}'`);
  // and the tail itself is never quoted: it is prose on screen, not a paste.
  expect(shortForm(join)).not.toContain("'");
});

test('shortForm is empty exactly when joinCommand is — never half a credential', () => {
  for (const join of [
    null,
    undefined,
    { code: '', addresses: ['10.10.10.1'] },
    { code: '84315907', addresses: [] },
    { code: '84315907', addresses: ['127.0.0.1', '::1', '127.9.9.9'] },
    { addresses: ['10.10.10.1'] }
  ]) {
    expect(joinCommand(join)).toBe('');
    expect(shortForm(join)).toBe('');
  }
});

// --- dialHost ----------------------------------------------------------------

test('dialHost names the same machine as the command', () => {
  const join = { code: '84315907', addresses: ['10.10.10.1', '192.168.1.5'] };
  expect(dialHost(join)).toBe('10.10.10.1');
  expect(joinCommand(join)).toContain(`@${dialHost(join)}`);
  expect(dialHost({ addresses: ['127.0.0.1'] })).toBeNull();
  expect(dialHost(null)).toBeNull();
});
