// SPDX-License-Identifier: AGPL-3.0-only

import { test, expect } from 'bun:test';
import { tiers, reach, provenance, ROOT } from './hops.js';

const node = (id, over = {}) => ({ id: id.repeat(64).slice(0, 64), name: id, ...over });
const LOCAL = node('a', { isLocal: true, name: 'macbook' });
const DGX1 = node('b', { name: 'dgx1' });
const DGX2 = node('c', { name: 'dgx2', reachedVia: node('b').id });
const DGX3 = node('d', { name: 'dgx3', reachedVia: node('b').id });

test('the browser is always the root, even with no agent', () => {
  const t = tiers([]);
  expect(t[0].nodes[0]).toBe(ROOT);
  expect(reach([])).toEqual([]);
});

test('the operator topology is hops from the browser, not a mesh', () => {
  const t = tiers([LOCAL, DGX1, DGX2, DGX3]);
  expect(t.map((x) => x.tier)).toEqual([0, 1, 2, 3]);
  expect(t[1].nodes.map((n) => n.name)).toEqual(['macbook']);
  expect(t[2].nodes.map((n) => n.name)).toEqual(['dgx1']);
  // dgx2 and dgx3 are only reachable through dgx1, so they are a hop further
  // out — drawing them beside dgx1 would promise a path that does not exist.
  expect(t[3].nodes.map((n) => n.name)).toEqual(['dgx2', 'dgx3']);
});

test('each machine is reached exactly one way', () => {
  const e = reach([LOCAL, DGX1, DGX2, DGX3]);
  expect(e.length).toBe(4); // you->macbook, macbook->dgx1, dgx1->dgx2, dgx1->dgx3
  expect(e.filter((x) => x.kind === 'vouched').map((x) => x.to)).toEqual([DGX2.id, DGX3.id]);
  expect(e.every((x) => x.from !== x.to)).toBe(true);
  // One edge per machine: a node must not appear as a destination twice, or the
  // picture claims two paths where the protocol provides one.
  expect(new Set(e.map((x) => x.to)).size).toBe(e.length);
});

test('a machine vouched for by something absent is still drawn', () => {
  // Its voucher dropped off the fleet. The machine exists and the operator
  // paired it; dropping it would be a worse lie than drawing it one tier in.
  const orphan = node('e', { name: 'ghost', reachedVia: 'f'.repeat(64) });
  const t = tiers([LOCAL, orphan]);
  expect(t[2].nodes.map((n) => n.name)).toEqual(['ghost']);
  expect(reach([LOCAL, orphan]).find((x) => x.to === orphan.id).kind).toBe('direct');
});

test('a node cannot be vouched for by the local agent', () => {
  // `via` pointing at the machine we are already connected to is the same thing
  // as direct: there is no intermediate hop. Treating it as vouched would draw
  // a self-referential extra tier.
  const odd = node('g', { name: 'odd', reachedVia: LOCAL.id });
  const t = tiers([LOCAL, odd]);
  expect(t.length).toBe(3);
  expect(t[2].nodes.map((n) => n.name)).toEqual(['odd']);
  expect(reach([LOCAL, odd]).find((x) => x.to === odd.id).kind).toBe('direct');
});

test('provenance says how a machine is known, in words', () => {
  expect(provenance(LOCAL, [LOCAL])).toContain('This machine');
  expect(provenance(DGX1, [LOCAL, DGX1])).toContain('directly');
  const p = provenance(DGX2, [LOCAL, DGX1, DGX2]);
  expect(p).toContain('dgx1');
  expect(p).toContain('carried by that machine');
});

test('a machine behind an unreachable voucher says so, and is not removed', () => {
  // Rule 5. It must stop implying the machine is reachable without pretending
  // it left the fleet — a member that is off is still yours.
  const dead = { ...DGX1, pairing: 'unreachable' };
  const p = provenance(DGX2, [LOCAL, dead, DGX2]);
  expect(p).toContain('unreachable');
  expect(p).toContain('has not been removed');
  expect(tiers([LOCAL, dead, DGX2])[3].nodes.map((n) => n.name)).toEqual(['dgx2']);
});

test('vouched-but-directly-reachable is described as unverified, not as relayed', () => {
  // `vouchedBy` without `reachedVia`: somebody said it exists and we can reach
  // it ourselves. Claiming control goes "through" the voucher would be false.
  const n = { ...node('h'), name: 'hearsay', pairing: 'vouched', vouchedBy: DGX1.id };
  const p = provenance(n, [LOCAL, DGX1, n]);
  expect(p).toContain('vouched for it');
  expect(p).not.toContain('carried by');
});

test('a silent voucher is named as the reason there is no route', () => {
  // The agent keeps the row with reached_via cleared when a voucher goes quiet,
  // so this is the shape the page actually receives. Saying only "nothing has
  // been verified first-hand" would be true and would hide the actionable part:
  // the machine to go and check is the voucher.
  const quiet = { ...DGX1, pairing: 'unreachable' };
  const behind = { ...node('i'), name: 'behind', pairing: 'vouched', vouchedBy: DGX1.id };
  const p = provenance(behind, [LOCAL, quiet, behind]);
  expect(p).toContain('dgx1');
  expect(p).toContain('not answering');
  expect(p).toContain('has not been removed');
});
