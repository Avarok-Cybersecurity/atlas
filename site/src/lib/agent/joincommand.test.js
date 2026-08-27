// SPDX-License-Identifier: AGPL-3.0-only

// This string carries a credential to another machine. The cases below are the
// ones where rendering *something* would be worse than rendering nothing.

import { expect, test } from 'bun:test';
import { bestAddress, joinCommand } from './joincommand.js';

test('an ordinary invitation renders one pasteable line', () => {
  const cmd = joinCommand({ code: '12345678', addresses: ['10.10.10.1'] });
  expect(cmd).toContain('--join 12345678@10.10.10.1');
  expect(cmd).toContain('install.sh');
  expect(cmd.split('\n')).toHaveLength(1);
});

test('the first address wins, because the agent already ordered them', () => {
  const cmd = joinCommand({ code: '12345678', addresses: ['10.10.10.1', '192.168.1.5'] });
  expect(cmd).toContain('@10.10.10.1');
});

// A command naming loopback installs cleanly and then fails to pair, on the far
// machine, which is the most confusing failure available here.
test('loopback is never offered to another machine', () => {
  expect(bestAddress(['127.0.0.1', '10.0.0.4'])).toBe('10.0.0.4');
  expect(bestAddress(['::1', '10.0.0.4'])).toBe('10.0.0.4');
  expect(bestAddress(['127.0.1.1'])).toBeNull();
});

test('no usable address renders nothing rather than half a command', () => {
  expect(joinCommand({ code: '12345678', addresses: [] })).toBe('');
  expect(joinCommand({ code: '12345678', addresses: ['127.0.0.1'] })).toBe('');
});

test('no code renders nothing', () => {
  expect(joinCommand({ code: '', addresses: ['10.0.0.1'] })).toBe('');
  expect(joinCommand(null)).toBe('');
  expect(joinCommand({ addresses: ['10.0.0.1'] })).toBe('');
});

test('junk in the address list is skipped rather than rendered', () => {
  expect(bestAddress([null, undefined, '  ', '10.0.0.9'])).toBe('10.0.0.9');
  expect(bestAddress(null)).toBeNull();
});

test('the control grant is a visible flag, not an implication of joining', () => {
  const join = { code: '12345678', addresses: ['10.10.10.1'] };
  expect(joinCommand(join)).not.toContain('--grant-control');
  expect(joinCommand(join, true)).toContain('--grant-control');
  // It must sit at the end of the same one-liner, so the operator reads it on
  // the line they are pasting rather than discovering it afterwards.
  expect(joinCommand(join, true).endsWith('--grant-control')).toBe(true);
});

test('no address still means no command, grant or not', () => {
  // Half an invitation with a privilege flag on the end is worse than none.
  expect(joinCommand({ code: '12345678', addresses: [] }, true)).toBe('');
  expect(joinCommand({ code: '', addresses: ['10.0.0.1'] }, true)).toBe('');
});
