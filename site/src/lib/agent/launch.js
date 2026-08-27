// SPDX-License-Identifier: AGPL-3.0-only

// The cluster-launch flow, as a pure state machine.
//
// Deliberately free of runes, sockets and DOM, because the parts worth testing
// are the rules — how many machines a recipe needs, which of them may hold a
// rank, which one is the head, and what an operator is allowed to do at each
// step. A flow that reaches the network to answer those questions can only be
// tested against a network.
//
// The phases are linear and each one names what it is waiting for:
//
//   choosing -> previewing -> previewed -> preparing -> prepared
//                                                    -> committing -> running
//
// Anything can fall to `failed`, which always carries a reason, and `reset()`
// is the only way back. There is no implicit retry: a prepare that half
// succeeded left reservations on real machines, and quietly trying again would
// leave the operator with no idea which attempt the fleet is holding.

/** Phases in which the selection may still be edited. */
const EDITABLE = new Set(['choosing', 'previewed', 'failed']);

/** Phases in which the fleet is being asked something and must not be disturbed. */
export const BUSY = ['previewing', 'preparing', 'committing', 'stopping'];

/**
 * A fresh flow.
 *
 * @returns {object}
 */
export function initial() {
  return {
    phase: 'choosing',
    /** Recipe id, or null. */
    recipe: null,
    /** Selected node fingerprints, in selection order. */
    selected: [],
    /** Fingerprint of the node that serves rank 0. */
    head: null,
    /** Per-rank previews, once previewed. */
    ranks: [],
    /** Warning about the fabric, when the plan would not run on RDMA. */
    linkWarning: null,
    /** Epoch pinning a commit to its prepare. */
    epoch: null,
    /** Per-rank prepare answers. */
    answers: [],
    /** Per-rank containers, once running. */
    started: [],
    /** Why the flow failed. */
    reason: null,
  };
}

/**
 * How many machines a recipe needs. Absent or nonsense means one.
 *
 * @param {object|null|undefined} recipe
 * @returns {number}
 */
export function required(recipe) {
  const n = Number(recipe?.nodes);
  return Number.isInteger(n) && n >= 1 ? n : 1;
}

/**
 * Whether the selection is complete enough to ask the fleet anything.
 *
 * The count must be exact, not merely sufficient. A recipe pinned to two nodes
 * launched across three would silently run a different topology from the one
 * its numbers were measured on.
 *
 * @param {object} state
 * @param {object|null} recipe
 * @returns {boolean}
 */
export function ready(state, recipe) {
  return (
    state.recipe != null &&
    recipe != null &&
    state.selected.length === required(recipe) &&
    state.head != null &&
    state.selected.includes(state.head)
  );
}

/**
 * Why the flow is not ready, phrased for an operator rather than a developer.
 *
 * Returns null when it is ready. Never returns a bare count: "select 1 more
 * machine" is actionable, "2 required" is a fact the operator has to translate.
 *
 * @param {object} state
 * @param {object|null} recipe
 * @param {number} available how many machines could hold a rank
 * @returns {string|null}
 */
export function blocker(state, recipe, available) {
  if (state.recipe == null || recipe == null) return 'Choose a recipe.';
  const need = required(recipe);
  if (available < need) {
    const missing = need - available;
    return `This recipe needs ${need} machines and ${available} ${
      available === 1 ? 'is' : 'are'
    } available. Pair ${missing} more.`;
  }
  const short = need - state.selected.length;
  if (short > 0) return `Select ${short} more machine${short === 1 ? '' : 's'}.`;
  if (short < 0) {
    return `This recipe runs on exactly ${need} machines. Deselect ${-short}.`;
  }
  if (state.head == null || !state.selected.includes(state.head)) {
    return 'Choose which machine serves the API.';
  }
  return null;
}

/**
 * Add or remove a machine from the selection.
 *
 * Selecting past the recipe's count is refused rather than silently dropping
 * the oldest choice: an operator who clicks a fourth machine and sees the first
 * one quietly deselect has been lied to about what will run.
 *
 * @param {object} state
 * @param {string} id
 * @param {object|null} recipe
 * @returns {object}
 */
export function toggleNode(state, id, recipe) {
  if (!EDITABLE.has(state.phase)) return state;
  const has = state.selected.includes(id);
  if (!has && state.selected.length >= required(recipe)) return state;

  const selected = has ? state.selected.filter((x) => x !== id) : [...state.selected, id];
  // The head must be one of the chosen machines. Dropping it silently would
  // leave a plan whose rank 0 is not in the cluster.
  let head = state.head;
  if (head != null && !selected.includes(head)) head = null;
  if (head == null && selected.length > 0) head = selected[0];
  return { ...clearResults(state), selected, head };
}

/**
 * Name the machine that serves rank 0.
 *
 * @param {object} state
 * @param {string} id
 * @returns {object}
 */
export function setHead(state, id) {
  if (!EDITABLE.has(state.phase) || !state.selected.includes(id)) return state;
  return { ...clearResults(state), head: id };
}

/**
 * Choose the recipe, which resets a selection sized for a different one.
 *
 * @param {object} state
 * @param {string|null} id
 * @returns {object}
 */
export function setRecipe(state, id) {
  if (!EDITABLE.has(state.phase)) return state;
  if (id === state.recipe) return state;
  return { ...initial(), recipe: id };
}

/**
 * Drop anything computed from a selection that has just changed.
 *
 * A preview or a prepare describes one exact plan. Leaving either on screen
 * after the plan changed would show an operator commands for machines they have
 * just deselected.
 *
 * @param {object} state
 * @returns {object}
 */
function clearResults(state) {
  return {
    ...state,
    phase: 'choosing',
    ranks: [],
    linkWarning: null,
    epoch: null,
    answers: [],
    started: [],
    reason: null,
  };
}

/**
 * A setting changed, so anything computed from the old one is stale.
 *
 * Refused while a prepare is held or a cluster is running, for the same reason
 * the selection is: real machines are holding reservations or containers, and
 * the flow has to be abandoned or stopped explicitly first.
 *
 * @param {object} state
 * @returns {object}
 */
export function settingsChanged(state) {
  if (!EDITABLE.has(state.phase)) return state;
  return clearResults(state);
}

/** Enter a waiting phase. */
export function beginPreview(state) {
  return { ...state, phase: 'previewing', reason: null };
}

/** Enter the prepare wait. */
export function beginPrepare(state) {
  return { ...state, phase: 'preparing', reason: null };
}

/** Enter the commit wait. */
export function beginCommit(state) {
  return { ...state, phase: 'committing', reason: null };
}

/**
 * Record a preview.
 *
 * @param {object} state
 * @param {{ranks: object[], link_warning?: string|null}} reply
 * @returns {object}
 */
export function previewed(state, reply) {
  const ranks = Array.isArray(reply?.ranks) ? reply.ranks : [];
  // A preview with no ranks is not a preview. Rendering it as one produced a
  // screen with no commands, no error and no button — the operator had no way
  // to tell a broken reply from a slow one.
  if (ranks.length === 0) {
    return failed(state, 'The agent replied without a plan. Nothing was reserved.');
  }
  return {
    ...state,
    phase: 'previewed',
    ranks,
    linkWarning: reply?.link_warning ?? null,
  };
}

/**
 * Record a prepare.
 *
 * A prepare in which any rank refused is not a failure of the flow — it is an
 * answer, and the operator needs to see which machine refused and why. So the
 * phase still becomes 'prepared'; it is `mayCommit` that decides what they can
 * do next.
 *
 * @param {object} state
 * @param {{epoch: string, ranks: object[], may_commit: boolean}} reply
 * @returns {object}
 */
export function prepared(state, reply) {
  const ranks = Array.isArray(reply?.ranks) ? reply.ranks : [];
  // Same reasoning as `previewed`: an answer from nobody is not an answer, and
  // `mayCommit` refusing quietly would leave the operator with a dead button.
  if (ranks.length === 0) {
    return failed(state, 'The agent answered without naming any machine. Nothing was reserved.');
  }
  return {
    ...state,
    phase: 'prepared',
    epoch: typeof reply?.epoch === 'string' ? reply.epoch : null,
    answers: ranks,
  };
}

/**
 * Whether every rank accepted and a commit may therefore proceed.
 *
 * Derived from the answers rather than trusted from the reply's own flag, so a
 * malformed reply cannot enable a button. An empty answer list is not consent.
 *
 * @param {object} state
 * @returns {boolean}
 */
export function mayCommit(state) {
  return (
    state.phase === 'prepared' &&
    state.epoch != null &&
    state.answers.length > 0 &&
    state.answers.every((r) => r.prepared === true)
  );
}

/**
 * Record a commit.
 *
 * @param {object} state
 * @param {{ranks: object[]}} reply
 * @returns {object}
 */
export function started(state, reply) {
  const ranks = Array.isArray(reply?.ranks) ? reply.ranks : [];
  // The same guard `previewed` and `prepared` carry, and it was missing here.
  // `phase: 'running'` with nothing started renders neither the running panel
  // (which needs `started.length > 0`) nor the local panel (which needs a
  // recipe running here) — so the operator gets a screen with no commands, no
  // error and no button, after the one action that actually spends machines.
  if (ranks.length === 0) {
    return failed(
      state,
      'The agent reported the launch started but named no machine. Nothing is known to be running.'
    );
  }
  return {
    ...state,
    phase: 'running',
    started: ranks,
  };
}

/**
 * Record a failure.
 *
 * The epoch is cleared: whatever went wrong, the agent has already rolled the
 * prepare back, and leaving an epoch on screen would offer a commit against a
 * reservation nobody is holding.
 *
 * @param {object} state
 * @param {unknown} reason
 * @returns {object}
 */
export function failed(state, reason) {
  return {
    ...state,
    phase: 'failed',
    epoch: null,
    answers: [],
    reason: describe(reason),
  };
}

/** Enter the stop wait. */
export function beginStop(state) {
  return { ...state, phase: 'stopping', reason: null };
}

/**
 * Record that every rank stopped.
 *
 * Back to 'previewed' rather than 'choosing': the operator most often stops a
 * cluster to change one setting and start it again, and throwing the plan away
 * would make them rebuild it from nothing.
 *
 * @param {object} state
 * @returns {object}
 */
export function stopped(state) {
  return { ...state, phase: 'previewed', started: [], epoch: null, answers: [] };
}

/** Release a prepare without starting anything. */
export function abandoned(state) {
  return { ...state, phase: 'previewed', epoch: null, answers: [] };
}

/**
 * Turn whatever the agent or the socket produced into one readable line.
 *
 * @param {unknown} e
 * @returns {string}
 */
export function describe(e) {
  if (e == null) return 'Something went wrong.';
  if (typeof e === 'string') return e;
  if (typeof e?.reason === 'string' && e.reason.length > 0) return e.reason;
  if (typeof e?.detail === 'string' && e.detail.length > 0) return e.detail;
  if (typeof e?.message === 'string' && e.message.length > 0) return e.message;
  return 'Something went wrong.';
}
