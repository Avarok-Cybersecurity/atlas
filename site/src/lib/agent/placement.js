// SPDX-License-Identifier: AGPL-3.0-only

// Deciding where a recipe should run, before asking how.
//
// The launch dialog used to go straight from "an agent answered" to a settings
// form for *this* machine. That is right exactly when this machine is the only
// candidate, and wrong the rest of the time — most sharply on a laptop that
// cannot run models at all, where the form was for a machine that was never
// going to run anything.
//
// Pure, because the interesting part is the decision rather than the markup:
// which machines could hold this recipe, whether the operator should be asked
// at all, and what they are told when the answer is "none of them". The
// `.svelte` file is a surface over this, the same split `launch.js` and
// `ClusterLaunch.svelte` already use.

/**
 * How many machines a recipe needs. Absent or nonsense means one.
 *
 * Mirrors `launch.js:required` deliberately rather than importing it: this
 * module is about *where*, that one is about the cluster flow, and a shared
 * helper would tie a chooser to a state machine it does not otherwise touch.
 *
 * @param {object|null|undefined} recipe
 * @returns {number}
 */
export function required(recipe) {
  const n = Number(recipe?.nodes);
  return Number.isFinite(n) && n >= 1 ? Math.floor(n) : 1;
}

/**
 * The machines that could hold a rank for this recipe, this machine first.
 *
 * Only paired peers and this machine: a discovered stranger is a grey dot on a
 * graph, not somewhere to send a workload.
 *
 * @param {object[]} nodes
 * @returns {object[]}
 */
export function candidates(nodes) {
  const usable = (Array.isArray(nodes) ? nodes : []).filter(
    (n) => n?.canLaunch && (n.isLocal || n.pairing === 'paired')
  );
  // Local first, then by name, so the list does not reorder as vitals arrive.
  return usable.slice().sort((a, b) => {
    if (a.isLocal !== b.isLocal) return a.isLocal ? -1 : 1;
    return String(a.name).localeCompare(String(b.name));
  });
}

/**
 * What the launch dialog should do about placement.
 *
 * Returns one of:
 *   - `{ kind: 'ask', options }`     — more than one real choice
 *   - `{ kind: 'only', target }`     — exactly one; skip the step
 *   - `{ kind: 'none', reason, canOnboard }` — nowhere to run it
 *
 * A chooser with one option is a nag, so it is skipped. That is the whole
 * reason this returns a decision rather than a list: the caller should not
 * have to re-derive "was that worth asking about".
 *
 * @param {object[]} nodes
 * @param {object|null} recipe
 * @param {boolean|null} localCanLaunch tri-state; null = the agent has not said
 * @returns {object}
 */
export function decide(nodes, recipe, localCanLaunch = null) {
  const need = required(recipe);
  const options = candidates(nodes);

  if (options.length === 0) {
    return {
      kind: 'none',
      reason:
        localCanLaunch === false
          ? 'This machine cannot run models, and no machine that can is paired yet.'
          : 'No machine here can run this yet.',
      canOnboard: true
    };
  }

  // A multi-node recipe is not a placement question — it is a cluster plan,
  // and the control page owns that. Saying so is better than offering a
  // chooser that cannot express "both of them".
  if (need > 1) {
    return options.length >= need
      ? { kind: 'cluster', need, options }
      : {
          kind: 'none',
          reason: `This recipe needs ${need} machines and ${options.length} ${
            options.length === 1 ? 'is' : 'are'
          } available.`,
          canOnboard: true
        };
  }

  if (options.length === 1) return { kind: 'only', target: options[0] };
  return { kind: 'ask', options };
}

/**
 * A one-line description of a machine, for the chooser.
 *
 * Says what distinguishes it — the OS and accelerator an operator picks by —
 * and omits anything the agent has not actually reported rather than printing
 * a blank field.
 *
 * @param {object} node
 * @returns {string}
 */
export function describe(node) {
  const bits = [];
  if (node?.isLocal) bits.push('this machine');
  if (node?.os) bits.push(node.os);
  if (node?.accelerator) bits.push(node.accelerator);
  const addr = node?.addresses?.find((a) => a.class !== 'loopback' && a.class !== 'virtual');
  if (!node?.isLocal && addr?.addr) bits.push(addr.addr);
  return bits.join(' · ');
}
