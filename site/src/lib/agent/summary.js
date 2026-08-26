// SPDX-License-Identifier: AGPL-3.0-only

// One line describing a fleet, for the topbar.
//
// **Most visitors to this site are not customers.** They must see a clean page,
// not a broken-looking widget reporting that it cannot reach something they
// have never installed. So the summary's first job is deciding whether there is
// anything to say at all, and the honest answer is usually no.
//
// Pure, because the interesting part is that decision rather than the markup.

/**
 * A peer must be missing this many consecutive reads before it leaves the
 * count.
 *
 * Discovery is multicast and lossy; a node that blinks out of one read and back
 * into the next is not a fleet change, and a count that flickers between 2 and
 * 3 teaches an operator to ignore it.
 */
export const MISSES_BEFORE_GONE = 2;

/**
 * What the topbar should show.
 *
 * @param {object} fleet a FleetSession-shaped object
 * @returns {{show: boolean, tone: string, label: string, detail: string}}
 */
export function summarize(fleet) {
  // Anything other than a working local agent is silence. Not an error, not a
  // prompt to install something — silence. The one place that pitches the
  // agent is the page that exists to explain it.
  if (fleet?.mode !== 'live') {
    return { show: false, tone: 'idle', label: '', detail: '' };
  }

  const nodes = Array.isArray(fleet.nodes) ? fleet.nodes : [];
  const reachable = nodes.filter((n) => n.isLocal || n.pairing === 'paired');
  const serving = reachable.filter((n) => n.running).length;
  const worst = worstSeverity(nodes);

  return {
    show: true,
    tone: worst ?? (serving > 0 ? 'serving' : 'ok'),
    label: `${reachable.length} ${reachable.length === 1 ? 'node' : 'nodes'}`,
    detail: serving === 0 ? 'idle' : `${serving} serving`,
  };
}

/**
 * The worst alert severity anywhere in the fleet, or null.
 *
 * @param {object[]} nodes
 * @returns {string|null}
 */
export function worstSeverity(nodes) {
  const rank = { critical: 0, warning: 1 };
  let worst = null;
  for (const n of nodes ?? []) {
    for (const a of n.alerts ?? []) {
      if (!(a.severity in rank)) continue;
      if (worst == null || rank[a.severity] < rank[worst]) worst = a.severity;
    }
  }
  return worst;
}

/**
 * Fold a fresh node list into a stable one, holding a briefly-missing node.
 *
 * Returns the list to display plus the new miss counts. A node that reappears
 * has its count cleared, so a flapping link does not slowly evict it.
 *
 * @param {object[]} previous what is currently shown
 * @param {object[]} fresh what the agent just reported
 * @param {Record<string, number>} misses
 * @returns {{nodes: object[], misses: Record<string, number>}}
 */
export function settle(previous, fresh, misses) {
  const freshIds = new Set(fresh.map((n) => n.id));
  const nextMisses = {};
  const out = [...fresh];

  for (const n of previous) {
    if (freshIds.has(n.id)) continue;
    const count = (misses?.[n.id] ?? 0) + 1;
    if (count < MISSES_BEFORE_GONE) {
      nextMisses[n.id] = count;
      out.push(n);
    }
    // At the threshold it is dropped, and its count goes with it.
  }
  return { nodes: out, misses: nextMisses };
}
