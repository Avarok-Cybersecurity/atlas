// SPDX-License-Identifier: AGPL-3.0-only

// The fleet as the operator reaches it, rather than as a set of machines.
//
// The existing topology map draws every trusted pair joined to every other,
// which is the right picture of a CLUSTER — for a collective, what matters is
// that all ranks can talk to all ranks. It is the wrong picture of REACHABILITY,
// and reachability is the question someone asks when a machine is missing:
// "what is this connected through?"
//
// So this arranges the fleet in hops away from the browser:
//
//   you → the agent on this machine → machines it has paired → machines those
//   vouch for
//
// A node's `via` names the peer that told us about it. `null` means this agent
// reached it itself. That distinction is the whole point: a vouched machine is
// known second-hand, it is controlled THROUGH its voucher, and if the voucher
// goes away it goes with it. Drawing it as though it were directly attached
// would promise the operator something the network does not do.

/** The browser itself. Always present, always the root. */
export const ROOT = { id: '__you__', kind: 'browser', label: 'You' };

/**
 * Group the fleet into hops from the browser.
 *
 * Returns tiers, nearest first. Tier 0 is always the browser. Tier 1 is the
 * agent this page is connected to. Tier 2 is everything that agent reached
 * itself, and tier 3 is everything vouched for by something in tier 2.
 *
 * A node whose `via` names a machine that is not present is placed in tier 2
 * rather than dropped: it exists, we simply cannot draw what it hangs off, and
 * silently discarding a machine the operator has paired would be worse than
 * drawing it in the wrong column.
 *
 * @param {object[]} nodes
 * @returns {{tier: number, nodes: object[]}[]}
 */
export function tiers(nodes) {
  const list = (Array.isArray(nodes) ? nodes : []).filter(
    (n) => n && typeof n.id === 'string' && n.id.length > 0
  );
  const local = list.filter((n) => n.isLocal);
  const remote = list.filter((n) => !n.isLocal);
  const present = new Set(local.concat(remote).map((n) => n.id));

  const direct = remote.filter((n) => !n.via || !present.has(n.via) || n.via === local[0]?.id);
  const vouched = remote.filter((n) => n.via && present.has(n.via) && n.via !== local[0]?.id);

  const byId = (a, b) => a.id.localeCompare(b.id);
  const out = [{ tier: 0, nodes: [ROOT] }];
  if (local.length) out.push({ tier: 1, nodes: local.slice().sort(byId) });
  if (direct.length) out.push({ tier: 2, nodes: direct.slice().sort(byId) });
  if (vouched.length) out.push({ tier: 3, nodes: vouched.slice().sort(byId) });
  return out;
}

/**
 * The reachability edges: what each machine is reached through.
 *
 * Exactly one edge per machine, because a machine is reached one way. This is
 * deliberately not the all-pairs graph the cluster view draws — that answers
 * "can these talk to each other", and this answers "how do I get to this".
 *
 * @param {object[]} nodes
 * @returns {{from: string, to: string, kind: 'browser'|'direct'|'vouched'}[]}
 */
export function reach(nodes) {
  const list = (Array.isArray(nodes) ? nodes : []).filter(
    (n) => n && typeof n.id === 'string' && n.id.length > 0
  );
  const local = list.find((n) => n.isLocal);
  const present = new Set(list.map((n) => n.id));
  const out = [];
  if (local) out.push({ from: ROOT.id, to: local.id, kind: 'browser' });
  for (const n of list) {
    if (n.isLocal) continue;
    const viaPresent = n.via && present.has(n.via) && n.via !== local?.id;
    if (viaPresent) {
      out.push({ from: n.via, to: n.id, kind: 'vouched' });
    } else if (local) {
      out.push({ from: local.id, to: n.id, kind: 'direct' });
    }
  }
  return out;
}

/**
 * How this machine is known, in words the operator can act on.
 *
 * @param {object} node
 * @param {object[]} all
 * @returns {string}
 */
export function provenance(node, all) {
  if (node?.isLocal) return 'This machine — the agent this page is connected to.';
  const via = node?.via ? (all ?? []).find((n) => n.id === node.via) : null;
  if (via) {
    return `Reached through ${via.name || via.id.slice(0, 8)}. It is not directly reachable from here, so control of it is carried by that machine.`;
  }
  return 'Reached directly from this machine.';
}
