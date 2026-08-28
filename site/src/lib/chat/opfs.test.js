// SPDX-License-Identifier: AGPL-3.0-only

// `pruneStale` keeps one corpus and deletes the rest, so the consequence of a
// bad argument is not a wrong file but ALL of them: `latticeFileName(undefined)`
// is `lattice-db-undefined.jsonl`, which matches nothing cached, so every real
// entry takes the delete branch. Both callers pass a validated sha today; this
// pins the precondition so that stays true of a future third caller.

import { expect, test, afterEach } from 'bun:test';
import { pruneStale } from './opfs.js';

const realNavigator = globalThis.navigator;
afterEach(() => {
  if (realNavigator === undefined) delete globalThis.navigator;
  else globalThis.navigator = realNavigator;
});

/** A directory holding the named corpora, recording what gets removed. */
function fakeOpfs(names) {
  const removed = [];
  const dir = {
    async *[Symbol.asyncIterator]() {
      for (const n of names) {
        yield [n, { kind: 'file', getFile: async () => ({ size: 1, lastModified: 1 }) }];
      }
    },
    removeEntry: async (n) => {
      removed.push(n);
    }
  };
  globalThis.navigator = { storage: { getDirectory: async () => dir } };
  return removed;
}

test('a bad sha deletes nothing at all, rather than everything', async () => {
  for (const bad of [undefined, null, '', 0, {}, []]) {
    const removed = fakeOpfs(['lattice-db-aaa.jsonl', 'lattice-db-bbb.jsonl']);
    await pruneStale(bad);
    expect(removed).toEqual([]);
  }
});

test('a real sha keeps its own corpus and drops the others', async () => {
  const removed = fakeOpfs([
    'lattice-db-keep.jsonl',
    'lattice-db-old1.jsonl',
    'lattice-db-old2.jsonl',
    'unrelated.txt'
  ]);
  await pruneStale('keep');
  expect(removed.sort()).toEqual(['lattice-db-old1.jsonl', 'lattice-db-old2.jsonl']);
});
