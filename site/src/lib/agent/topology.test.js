// SPDX-License-Identifier: AGPL-3.0-only

// A picture is allowed to be wrong in ways a number is not, so the properties
// worth asserting are the structural ones: it does not move when it should not,
// nodes do not overlap, and a line never claims a better link than it has.

import { describe as suite, expect, test } from 'bun:test';
import * as T from './topology.js';

const node = (id, extra = {}) => ({
  id,
  name: id,
  isLocal: false,
  pairing: 'paired',
  addresses: [{ iface: 'eth0', addr: '10.0.0.1', class: 'ethernet', speedMbps: 1000, rdma: false }],
  ...extra
});

const roce = (addr, speed = 200000) => ({
  iface: 'enp1s0',
  addr,
  class: 'roce',
  speedMbps: speed,
  rdma: true
});

suite('order does not depend on when a machine was discovered', () => {
  test('the same fleet in any order draws the same picture', () => {
    const fleet = [node('cccc'), node('aaaa', { isLocal: true }), node('bbbb')];
    const a = T.points(fleet).map((p) => [p.node.id, Math.round(p.x), Math.round(p.y)]);
    const b = T.points([...fleet].reverse()).map((p) => [p.node.id, Math.round(p.x), Math.round(p.y)]);
    expect(a).toEqual(b);
  });

  test('this machine comes first, then trusted, then strangers', () => {
    const got = T.ordered([
      node('zzz', { pairing: 'discovered' }),
      node('mmm'),
      node('aaa', { isLocal: true })
    ]);
    expect(got.map((n) => n.id)).toEqual(['aaa', 'mmm', 'zzz']);
  });

  test('a node without an id is not drawn rather than drawn wrong', () => {
    expect(T.ordered([{ name: 'x' }, node('ok')]).map((n) => n.id)).toEqual(['ok']);
    expect(T.points(null)).toEqual([]);
  });
});

suite('nodes do not sit on top of each other', () => {
  // The old single-row layout tiled circles against each other by six.
  for (const n of [2, 3, 4, 5, 6, 8]) {
    test(`${n} machines stay at least a diameter apart`, () => {
      const pts = T.points(Array.from({ length: n }, (_, i) => node(`n${i}`)));
      expect(pts).toHaveLength(n);
      for (let i = 0; i < pts.length; i += 1) {
        for (let j = i + 1; j < pts.length; j += 1) {
          const d = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
          expect(d).toBeGreaterThanOrEqual(2 * T.R);
        }
      }
    });
  }

  for (const n of [1, 2, 4, 8]) {
    test(`${n} machines stay inside the canvas`, () => {
      for (const p of T.points(Array.from({ length: n }, (_, i) => node(`n${i}`)))) {
        expect(p.x).toBeGreaterThanOrEqual(T.R);
        expect(p.x).toBeLessThanOrEqual(T.W - T.R);
        expect(p.y).toBeGreaterThanOrEqual(T.R);
        expect(p.y).toBeLessThanOrEqual(T.H - T.R);
      }
    });
  }
});

suite('a line never claims a better link than it has', () => {
  test('the worse endpoint decides the class', () => {
    const fast = node('aaaa', { isLocal: true, addresses: [roce('10.10.10.1')] });
    const slow = node('bbbb');
    const [e] = T.edges(T.points([fast, slow]));
    expect(e.cls).toBe('ethernet');
    expect(e.warn).toBe(true);
  });

  test('two RDMA machines are not warned about', () => {
    const a = node('aaaa', { isLocal: true, addresses: [roce('10.10.10.1')] });
    const b = node('bbbb', { addresses: [roce('10.10.10.2')] });
    const [e] = T.edges(T.points([a, b]));
    expect(e.cls).toBe('roce');
    expect(e.warn).toBe(false);
    expect(e.speed).toBe(200000);
  });

  test('an endpoint with no usable link makes the edge unknown, not ethernet', () => {
    const a = node('aaaa', { isLocal: true });
    const b = node('bbbb', { addresses: [{ addr: '127.0.0.1', class: 'loopback' }] });
    const [e] = T.edges(T.points([a, b]));
    expect(e.cls).toBe('none');
  });

  test('unverified is missing information, not a fault to warn about', () => {
    const a = node('aaaa', { isLocal: true, addresses: [{ addr: '1.1.1.1', class: 'unverified' }] });
    const b = node('bbbb', { addresses: [{ addr: '2.2.2.2', class: 'unverified' }] });
    const [e] = T.edges(T.points([a, b]));
    expect(e.warn).toBe(false);
  });

  test('an untrusted machine has no relationship to draw', () => {
    const pts = T.points([node('aaaa', { isLocal: true }), node('zzzz', { pairing: 'discovered' })]);
    expect(T.edges(pts)).toHaveLength(0);
  });

  test('speed is the slower end, and absent when either end is silent', () => {
    const a = node('aaaa', { isLocal: true, addresses: [roce('10.0.0.1', 200000)] });
    const b = node('bbbb', { addresses: [roce('10.0.0.2', 100000)] });
    expect(T.edges(T.points([a, b]))[0].speed).toBe(100000);

    const c = node('cccc', { addresses: [{ addr: '9.9.9.9', class: 'roce' }] });
    expect(T.edges(T.points([a, c]))[0].speed).toBeNull();
  });
});

suite('labels identify the machine, not its hostname', () => {
  test('the fingerprint is used, because Sparks ship with colliding names', () => {
    expect(T.label(node('a1b2c3d4', { name: 'spark-256a' }))).toBe('a1b2');
  });

  test('a machine with no id still renders something rather than blank', () => {
    expect(T.label({})).toBe('????');
  });
});
