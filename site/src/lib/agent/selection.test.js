// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import * as Sel from './selection.js';

const ID = (n) => String(n).repeat(64).slice(0, 64);
const NODES = [
  { id: ID(1), isLocal: true },
  { id: ID(2) },
  { id: ID(3) }
];

describe('number keys jump to roster rows', () => {
  test('1 selects the first row, 3 the third', () => {
    expect(Sel.selectByKey(NODES, '1')).toBe(ID(1));
    expect(Sel.selectByKey(NODES, '3')).toBe(ID(3));
  });

  test('a key past the roster selects nothing rather than clamping', () => {
    expect(Sel.selectByKey(NODES, '4')).toBeNull();
    expect(Sel.selectByKey(NODES, '8')).toBeNull();
  });

  test('keys outside 1-8 are not selection keys at all', () => {
    for (const k of ['0', '9', 'a', '', null, undefined, 1]) {
      expect(Sel.selectByKey(NODES, k)).toBeNull();
    }
  });

  test('rows past the eighth have no hotkey', () => {
    const many = Array.from({ length: 10 }, (_, i) => ({ id: `${'0'.repeat(63)}${i}` }));
    const vm = Sel.rosterVm(many, null);
    expect(vm[7].key).toBe('8');
    expect(vm[8].key).toBeNull();
    expect(Sel.selectByKey(many, '8')).toBe(many[7].id);
  });
});

describe('arrow roving', () => {
  test('steps down and up, one row at a time', () => {
    expect(Sel.move(NODES, ID(1), 1)).toBe(ID(2));
    expect(Sel.move(NODES, ID(2), -1)).toBe(ID(1));
  });

  test('clamps at the ends instead of wrapping to a different machine', () => {
    expect(Sel.move(NODES, ID(3), 1)).toBe(ID(3));
    expect(Sel.move(NODES, ID(1), -1)).toBe(ID(1));
  });

  test('no selection roves to the first row; an empty roster roves nowhere', () => {
    expect(Sel.move(NODES, null, 1)).toBe(ID(1));
    expect(Sel.move([], null, 1)).toBeNull();
  });

  test('a stride other than one row is a caller bug', () => {
    expect(() => Sel.move(NODES, ID(1), 2)).toThrow(TypeError);
  });
});

describe('the selection survives the fleet changing under it', () => {
  test('a still-present selection is kept', () => {
    expect(Sel.reselect(NODES, ID(2))).toBe(ID(2));
  });

  test('a vanished node falls back to the local machine', () => {
    expect(Sel.reselect(NODES, ID(9))).toBe(ID(1));
  });

  test('no local machine falls back to the first row, and an empty fleet to nothing', () => {
    expect(Sel.reselect([{ id: ID(2) }, { id: ID(3) }], ID(9))).toBe(ID(2));
    expect(Sel.reselect([], ID(9))).toBeNull();
  });
});

describe('aria-selected bookkeeping', () => {
  test('exactly the selected row is marked', () => {
    const vm = Sel.rosterVm(NODES, ID(2));
    expect(vm.map((r) => r.selected)).toEqual([false, true, false]);
  });
});

describe('hash persistence', () => {
  const REAL = 'ab'.repeat(32);

  test('a selection round-trips through the hash by id, never by index', () => {
    const nodes = [{ id: REAL }];
    const hash = Sel.toHash(REAL);
    expect(hash).toBe(`#node=${REAL}`);
    expect(Sel.fromHash(hash, nodes)).toBe(REAL);
  });

  test('only a validated node id may enter the URL bar', () => {
    // The hash round-trips through history and copy-paste, so it is wire
    // input on the way back in; anything else must not be emitted.
    for (const bad of [null, '', 'dgx1', REAL.toUpperCase(), REAL.slice(1)]) {
      expect(Sel.toHash(bad)).toBe('');
    }
  });

  test('a hash naming a machine not in the fleet selects nothing', () => {
    expect(Sel.fromHash(`#node=${'cd'.repeat(32)}`, [{ id: REAL }])).toBeNull();
  });

  test('a malformed hash fails closed', () => {
    for (const bad of [undefined, '', '#node=', '#node=zzz', '#other=1', `#node=${REAL}x`]) {
      expect(Sel.fromHash(bad, [{ id: REAL }])).toBeNull();
    }
  });
});
