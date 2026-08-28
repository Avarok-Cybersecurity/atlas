// SPDX-License-Identifier: AGPL-3.0-only

import { test, expect } from 'bun:test';
import { checkTarget, subnetOf, classLabel, describeAddress, networksOf } from './network.js';

test('a subnet is masked, not truncated', () => {
  expect(subnetOf('192.168.68.67', 24)).toBe('192.168.68.0/24');
  expect(subnetOf('10.10.10.2', 24)).toBe('10.10.10.0/24');
  // The DGX Spark's point-to-point RoCE links are /30s, documented in the
  // repo. A /24 assumption puts four of them on one network.
  expect(subnetOf('10.10.10.9', 30)).toBe('10.10.10.8/30');
  expect(subnetOf('10.10.10.13', 30)).toBe('10.10.10.12/30');
  expect(subnetOf('172.16.5.130', 22)).toBe('172.16.4.0/22');
});

test('a mask wider than 24 bits is not built with a signed shift', () => {
  // `0xffffffff << 31` is negative in JS. Without the >>> the octets come out
  // as garbage, and only the wide prefixes show it.
  expect(subnetOf('192.168.68.67', 1)).toBe('128.0.0.0/1');
  expect(subnetOf('192.168.68.67', 8)).toBe('192.0.0.0/8');
  expect(subnetOf('255.255.255.255', 32)).toBe('255.255.255.255/32');
});

test('anything that cannot be computed honestly is null, never a guess', () => {
  expect(subnetOf('192.168.68.67', 0)).toBeNull(); // agent did not report one
  expect(subnetOf('fe80::1', 64)).toBeNull(); // IPv6 is not masked here
  expect(subnetOf('192.168.68', 24)).toBeNull();
  expect(subnetOf('192.168.68.999', 24)).toBeNull();
  expect(subnetOf('192.168.68.a', 24)).toBeNull();
  expect(subnetOf('', 24)).toBeNull();
  expect(subnetOf('192.168.68.67', 33)).toBeNull();
  expect(subnetOf('192.168.68.67', -1)).toBeNull();
  expect(subnetOf('192.168.68.67', 24.5)).toBeNull();
});

test('a link class reads as a person would say it', () => {
  expect(classLabel('roce')).toBe('RoCE');
  expect(classLabel('wireless')).toBe('Wi-Fi');
  // An unknown class keeps its own name rather than being relabelled into one
  // of the known ones — inventing "Ethernet" for something unrecognised is how
  // someone gets told their fabric is slow.
  expect(classLabel('something_new')).toBe('something_new');
});

test('an address is described with what is known and nothing else', () => {
  const d = describeAddress({
    addr: '10.10.10.9',
    iface: 'enp1s0f0np0',
    class: 'roce',
    prefixLen: 30,
    speedMbps: 200000,
    rdma: true
  });
  expect(d.subnet).toBe('10.10.10.8/30');
  expect(d.detail).toBe('RoCE · RDMA · 200 Gb');

  // No speed reported means no speed shown. A zero would read as a real
  // measurement of nothing.
  const q = describeAddress({ addr: '1.2.3.4', class: 'wireless', prefixLen: 24, speedMbps: null });
  expect(q.detail).toBe('Wi-Fi');
});

test('two addresses on one LAN are one network; distinct LANs stay distinct', () => {
  const node = {
    addresses: [
      { addr: '10.10.10.9', iface: 'ib0', class: 'roce', prefixLen: 24, rdma: true },
      { addr: '10.10.10.13', iface: 'ib1', class: 'roce', prefixLen: 24, rdma: true },
      { addr: '192.168.68.68', iface: 'en0', class: 'ethernet', prefixLen: 24 }
    ]
  };
  const nets = networksOf(node);
  expect(nets.map((n) => n.subnet)).toEqual(['10.10.10.0/24', '192.168.68.0/24']);
  // Order is the agent's ranking: the fabric first.
  expect(nets[0].detail).toContain('RoCE');
});

test('unnamed networks do not collapse into each other', () => {
  // Two IPv6 addresses both yield a null subnet. Keying on that would merge
  // two genuinely different addresses into one row.
  const nets = networksOf({
    addresses: [
      { addr: '2001:db8::1', iface: 'en0', class: 'ethernet', prefixLen: 64 },
      { addr: '2001:db8:1::1', iface: 'en1', class: 'ethernet', prefixLen: 64 }
    ]
  });
  expect(nets.length).toBe(2);
});

// ---- what the operator types ---------------------------------------------

test('a plausible address is accepted in every form the agent resolves', () => {
  for (const t of [
    '10.10.10.2',
    '10.10.10.2:34334',
    '192.168.68.68',
    'spark-43fa.local',
    'spark-43fa.local:34334',
    '[fe80::1]',
    '[fe80::1]:34334',
    'fe80::1' // bare v6: the agent resolves it, so this field must not refuse it
  ]) {
    expect(checkTarget(t).ok).toBe(true);
  }
});

test('the mistakes people actually make are caught before a round trip', () => {
  // Each of these would come back from the agent as "does not resolve", which
  // is true and unhelpful.
  expect(checkTarget('').why).toContain('Enter the address');
  expect(checkTarget('   ').why).toContain('Enter the address');
  expect(checkTarget('http://10.10.10.2').why).toContain('no http://');
  expect(checkTarget('https://10.10.10.2:34334').why).toContain('no http://');
  expect(checkTarget('curl -fsSL https://x | sh').ok).toBe(false);
  expect(checkTarget('10.10.10.2:abc').why).toContain('port');
  expect(checkTarget('10.10.10.2:0').why).toContain('port');
  expect(checkTarget('10.10.10.2:70000').why).toContain('port');
  expect(checkTarget(':34334').why).toContain('no host');
});

test('a bare IPv6 literal is not mistaken for a host and port', () => {
  // '2001:db8::1' has many colons; splitting on the last one would produce the
  // nonsense host '2001:db8:' and port ':1'.
  expect(checkTarget('2001:db8::1').ok).toBe(true);
  expect(checkTarget('[2001:db8::1]:34334').ok).toBe(true);
});

test('a /0 is refused rather than masked, so the mask never shifts by 32', () => {
  // `0xffffffff << 32` is a shift by 32 % 32 = 0 in JS, which leaves the mask
  // all-ones — a /0 would come back as the address unchanged, claiming the
  // whole internet is one subnet. The guard makes that unreachable; this pins
  // it, because the mask arithmetic below now assumes prefixLen >= 1.
  expect(subnetOf('192.168.68.67', 0)).toBeNull();
  expect(subnetOf('0.0.0.0', 0)).toBeNull();
  // And the narrowest real prefix still works, which is what the guard protects.
  expect(subnetOf('192.168.68.67', 1)).toBe('128.0.0.0/1');
  expect(subnetOf('0.0.0.0', 32)).toBe('0.0.0.0/32');
});

// The manual-entry box exists to catch a paste that is not an address. These
// four are not typos — each is a real string sitting one panel away on this
// same page, and each used to be accepted whole and then fail later as an
// unresolvable hostname, which is the confusing half of the failure.
test('the pastes this box exists to catch are refused, with a reason', () => {
  const cases = [
    ['10.0.0.5,10.0.0.6', 'list'],       // what joinCommand renders
    ['ABC123@10.0.0.5', '@'],            // the join line's code@host
    ['example.com/path', 'path'],        // a URL minus its scheme
    ['[]', 'bracket'],                   // an empty bracketed host
    ['[::1', 'bracket']                  // a bracket that never closes
  ];
  for (const [input] of cases) {
    const r = checkTarget(input);
    expect(r.ok).toBe(false);
    expect(typeof r.why).toBe('string');
    expect(r.why.length).toBeGreaterThan(0);
  }
});

test('and the forms an operator legitimately types still pass', () => {
  for (const ok of ['10.0.0.5', 'spark-256a', 'host:9000', '::1', '[fe80::1]:34334']) {
    expect(checkTarget(ok).ok).toBe(true);
  }
});
