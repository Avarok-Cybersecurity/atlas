// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import * as L from './launch.js';

const TWO = { id: 'ep2', nodes: 2 };
const ONE = { id: 'solo', nodes: 1 };

/** A flow with a recipe chosen and `ids` selected, head = first. */
function chosen(recipe, ids) {
  let s = L.setRecipe(L.initial(), recipe.id);
  for (const id of ids) s = L.toggleNode(s, id, recipe);
  return s;
}

describe('how many machines a recipe needs', () => {
  test('a missing or nonsense count means one machine, never zero', () => {
    for (const r of [null, undefined, {}, { nodes: 0 }, { nodes: -3 }, { nodes: 'two' }]) {
      expect(L.required(r)).toBe(1);
    }
    expect(L.required(TWO)).toBe(2);
  });

  // A recipe pinned to two nodes launched across three would silently run a
  // different topology from the one its numbers were measured on.
  test('the count must be exact, not merely sufficient', () => {
    const s = chosen(TWO, ['a', 'b']);
    expect(L.ready(s, TWO)).toBe(true);
    expect(L.ready(chosen(TWO, ['a']), TWO)).toBe(false);
  });
});

describe('selection', () => {
  test('selecting past the count is refused rather than silently dropping one', () => {
    const s = chosen(TWO, ['a', 'b']);
    const after = L.toggleNode(s, 'c', TWO);
    expect(after.selected).toEqual(['a', 'b']);
  });

  test('the first machine chosen becomes the head', () => {
    expect(chosen(TWO, ['a', 'b']).head).toBe('a');
  });

  test('deselecting the head hands rank 0 to another selected machine', () => {
    let s = chosen(TWO, ['a', 'b']);
    s = L.toggleNode(s, 'a', TWO);
    expect(s.selected).toEqual(['b']);
    expect(s.head).toBe('b');
  });

  test('deselecting everything leaves no head to be stale', () => {
    let s = chosen(ONE, ['a']);
    s = L.toggleNode(s, 'a', ONE);
    expect(s.selected).toEqual([]);
    expect(s.head).toBeNull();
  });

  test('the head must be one of the chosen machines', () => {
    const s = chosen(TWO, ['a', 'b']);
    expect(L.setHead(s, 'zzz').head).toBe('a');
    expect(L.setHead(s, 'b').head).toBe('b');
  });

  test('changing recipe resets a selection sized for a different one', () => {
    const s = chosen(TWO, ['a', 'b']);
    const after = L.setRecipe(s, 'solo');
    expect(after.selected).toEqual([]);
    expect(after.head).toBeNull();
    expect(after.recipe).toBe('solo');
  });
});

describe('what the operator is told to do next', () => {
  test('each blocker names an action, not a fact', () => {
    expect(L.blocker(L.initial(), null, 2)).toBe('Choose a recipe.');
    expect(L.blocker(chosen(TWO, ['a']), TWO, 3)).toBe('Select 1 more machine.');
    expect(L.blocker(chosen(TWO, ['a', 'b']), TWO, 3)).toBeNull();
  });

  test('too few machines in the fleet says how many more to pair', () => {
    const msg = L.blocker(chosen(TWO, ['a']), TWO, 1);
    expect(msg).toContain('needs 2 machines');
    expect(msg).toContain('Pair 1 more');
  });

  test('a headless selection asks which machine serves the API', () => {
    const s = { ...chosen(TWO, ['a', 'b']), head: null };
    expect(L.blocker(s, TWO, 2)).toBe('Choose which machine serves the API.');
  });
});

describe('results never outlive the plan that produced them', () => {
  // Leaving a preview on screen after the plan changed would show commands for
  // machines the operator has just deselected.
  test('changing the selection clears a preview', () => {
    let s = L.previewed(chosen(TWO, ['a', 'b']), {
      ranks: [{ rank: 0, command: 'docker run x' }],
      link_warning: 'ethernet only',
    });
    expect(s.ranks).toHaveLength(1);

    s = L.toggleNode(s, 'b', TWO);
    expect(s.ranks).toEqual([]);
    expect(s.linkWarning).toBeNull();
    expect(s.phase).toBe('choosing');
  });

  // While a prepare is held, real machines are holding reservations. Editing
  // the plan underneath them would strand those reservations with nobody left
  // holding the epoch that releases them, so the operator has to abandon first.
  test('a held prepare cannot be edited; it has to be abandoned', () => {
    const held = L.prepared(L.previewed(chosen(TWO, ['a', 'b']), { ranks: [{ node: 'a', rank: 0, command: 'x' }] }), {
      epoch: 'e1',
      ranks: [{ node: 'a', prepared: true }],
      may_commit: true,
    });
    expect(L.setHead(held, 'b')).toBe(held);
    expect(L.toggleNode(held, 'c', TWO)).toBe(held);
    expect(L.setRecipe(held, 'solo')).toBe(held);

    // Abandoning releases the epoch and hands the plan back, editable.
    const free = L.abandoned(held);
    expect(free.epoch).toBeNull();
    expect(free.answers).toEqual([]);
    expect(L.setHead(free, 'b').head).toBe('b');
  });
});

describe('the commit gate', () => {
  const twoRanks = (a, b) => ({
    epoch: 'e1',
    ranks: [
      { node: 'a', rank: 0, prepared: a },
      { node: 'b', rank: 1, prepared: b },
    ],
    may_commit: a && b,
  });

  test('every rank must have accepted', () => {
    const base = L.previewed(chosen(TWO, ['a', 'b']), { ranks: [{ node: 'a', rank: 0, command: 'x' }] });
    expect(L.mayCommit(L.prepared(base, twoRanks(true, true)))).toBe(true);
    expect(L.mayCommit(L.prepared(base, twoRanks(true, false)))).toBe(false);
    expect(L.mayCommit(L.prepared(base, twoRanks(false, false)))).toBe(false);
  });

  // A malformed reply must not be able to enable the button that starts a
  // cluster. The gate reads the answers, not the flag.
  test('a reply claiming may_commit with a refusing rank does not open the gate', () => {
    const base = L.previewed(chosen(TWO, ['a', 'b']), { ranks: [{ node: 'a', rank: 0, command: 'x' }] });
    const lying = L.prepared(base, {
      epoch: 'e1',
      ranks: [
        { node: 'a', prepared: true },
        { node: 'b', prepared: false },
      ],
      may_commit: true,
    });
    expect(L.mayCommit(lying)).toBe(false);
  });

  // An answer from nobody is not an answer. Treating it as a prepare left the
  // operator with a dead button and no reason.
  test('an empty answer list fails loudly rather than sitting silent', () => {
    const base = L.previewed(chosen(TWO, ['a', 'b']), {
      ranks: [{ node: 'a', rank: 0, command: 'x' }],
    });
    const s = L.prepared(base, { epoch: 'e1', ranks: [], may_commit: true });
    expect(s.phase).toBe('failed');
    expect(s.reason).toContain('without naming any machine');
    expect(L.mayCommit(s)).toBe(false);
  });

  test('a prepare without an epoch cannot be committed', () => {
    const base = L.previewed(chosen(TWO, ['a', 'b']), { ranks: [{ node: 'a', rank: 0, command: 'x' }] });
    const s = L.prepared(base, { ranks: [{ node: 'a', prepared: true }], may_commit: true });
    expect(L.mayCommit(s)).toBe(false);
  });

  test('a refusal is an answer, not a crash: the phase still shows the reasons', () => {
    const base = L.previewed(chosen(TWO, ['a', 'b']), { ranks: [{ node: 'a', rank: 0, command: 'x' }] });
    const s = L.prepared(base, {
      epoch: 'e1',
      ranks: [{ node: 'b', rank: 1, prepared: false, reason: 'no disk' }],
      may_commit: false,
    });
    expect(s.phase).toBe('prepared');
    expect(s.answers[0].reason).toBe('no disk');
  });
});

describe('a reply that says nothing', () => {
  // This is the shape a wiring bug produces: the call resolved, so there was no
  // error, but the frame carried no plan. Rendering it as a preview gave a
  // screen with no commands, no error and no button.
  test('a preview with no ranks is a failure, not an empty preview', () => {
    const s = L.previewed(chosen(TWO, ['a', 'b']), { ranks: [] });
    expect(s.phase).toBe('failed');
    expect(s.reason).toContain('without a plan');
    expect(s.ranks).toEqual([]);
  });

  test('a malformed preview is treated the same as an empty one', () => {
    for (const reply of [undefined, null, {}, { ranks: 'nope' }]) {
      expect(L.previewed(chosen(TWO, ['a', 'b']), reply).phase).toBe('failed');
    }
  });
});

describe('failure', () => {
  // Whatever went wrong, the agent has already rolled the prepare back; leaving
  // an epoch on screen would offer a commit against a reservation nobody holds.
  test('a failure clears the epoch so no stale commit can be offered', () => {
    const base = L.previewed(chosen(TWO, ['a', 'b']), { ranks: [{ node: 'a', rank: 0, command: 'x' }] });
    const s = L.failed(
      L.prepared(base, { epoch: 'e1', ranks: [{ prepared: true }], may_commit: true }),
      { reason: 'spark-43fa could not start' },
    );
    expect(s.epoch).toBeNull();
    expect(L.mayCommit(s)).toBe(false);
    expect(s.reason).toBe('spark-43fa could not start');
  });

  test('a failed flow can be edited again without being reset', () => {
    const s = L.failed(chosen(TWO, ['a', 'b']), 'nope');
    expect(L.toggleNode(s, 'b', TWO).selected).toEqual(['a']);
  });

  test('every shape of error becomes one readable line', () => {
    expect(L.describe('plain')).toBe('plain');
    expect(L.describe({ reason: 'r' })).toBe('r');
    expect(L.describe({ detail: 'd' })).toBe('d');
    expect(L.describe(new Error('boom'))).toBe('boom');
    expect(L.describe(null)).toBe('Something went wrong.');
    expect(L.describe({ code: 'not_launchable' })).toBe('Something went wrong.');
  });

  test('an empty string is not a reason worth showing', () => {
    expect(L.describe({ reason: '', detail: '', message: '' })).toBe('Something went wrong.');
  });
});

describe('phases in flight are not editable', () => {
  // A prepare that half succeeded left reservations on real machines; letting
  // the operator edit the plan underneath it would make the rollback ambiguous.
  test('a selection cannot change while the fleet is being asked', () => {
    for (const phase of ['previewing', 'preparing', 'committing', 'running']) {
      const s = { ...chosen(TWO, ['a', 'b']), phase };
      expect(L.toggleNode(s, 'c', TWO)).toBe(s);
      expect(L.setHead(s, 'b')).toBe(s);
      expect(L.setRecipe(s, 'solo')).toBe(s);
    }
  });
});

describe('stopping a running cluster', () => {
  function running() {
    let s = L.previewed(chosen(TWO, ['a', 'b']), {
      ranks: [
        { node: 'a', rank: 0, command: 'x' },
        { node: 'b', rank: 1, command: 'y' },
      ],
    });
    s = L.prepared(s, {
      epoch: 'e1',
      ranks: [
        { node: 'a', rank: 0, prepared: true },
        { node: 'b', rank: 1, prepared: true },
      ],
      may_commit: true,
    });
    return L.started(s, {
      ranks: [
        { node: 'a', rank: 0, container: 'c0', endpoint: 'http://10.0.0.1:8888' },
        { node: 'b', rank: 1, container: 'c1', endpoint: null },
      ],
    });
  }

  test('a running cluster is not editable', () => {
    const s = running();
    expect(s.phase).toBe('running');
    expect(L.toggleNode(s, 'c', TWO)).toBe(s);
  });

  // Most often an operator stops a cluster to change one setting and start it
  // again; throwing the plan away would make them rebuild it from nothing.
  test('stopping keeps the plan and drops the containers', () => {
    const s = L.stopped(running());
    expect(s.phase).toBe('previewed');
    expect(s.started).toEqual([]);
    expect(s.epoch).toBeNull();
    expect(s.ranks).toHaveLength(2);
    expect(L.setHead(s, 'b').head).toBe('b');
  });

  test('a failed stop leaves the containers on screen to try again', () => {
    const s = L.failed(running(), 'the container runtime is not answering');
    expect(s.started).toHaveLength(2);
    expect(s.reason).toContain('not answering');
  });

  test('BUSY names every phase in which the fleet is mid-question', () => {
    expect(L.BUSY).toContain('stopping');
    for (const phase of L.BUSY) {
      expect(L.toggleNode({ ...running(), phase }, 'c', TWO).phase).toBe(phase);
    }
  });
});

test('a commit reply naming no machine is a failure, not a running launch', () => {
  // previewed() and prepared() both guard this and say why; started() did not.
  // phase 'running' with an empty started list renders neither panel, so the
  // operator sees an empty screen with no error immediately after the step
  // that actually spends machines.
  const held = L.prepared(
    L.previewed(chosen(TWO, ['a', 'b']), { ranks: [{ node: 'a', rank: 0, command: 'x' }] }),
    { epoch: 'e1', ranks: [{ node: 'a', prepared: true }], may_commit: true }
  );
  const out = L.started(L.beginCommit(held), { ranks: [] });
  expect(out.phase).toBe('failed');
  expect(out.reason).toMatch(/named no machine/);
  expect(out.epoch).toBeNull();
});

test('a commit reply with ranks still starts', () => {
  const held = L.prepared(
    L.previewed(chosen(TWO, ['a', 'b']), { ranks: [{ node: 'a', rank: 0, command: 'x' }] }),
    { epoch: 'e1', ranks: [{ node: 'a', prepared: true }], may_commit: true }
  );
  const out = L.started(L.beginCommit(held), { ranks: [{ node: 'a' }] });
  expect(out.phase).toBe('running');
  expect(out.started.length).toBe(1);
});
