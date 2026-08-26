// SPDX-License-Identifier: AGPL-3.0-only

// Where each machine sits in the picture, and what the line between two of
// them means.
//
// Pure and separate from the SVG for the reason the rest of this directory is:
// the rules are worth testing and the markup is not. Layout is COMPUTED, never
// dragged — the same fleet draws the same pixels every time, so nothing shifts
// under a cursor and nothing depends on the order machines happened to be
// discovered in.
//
// The previous layout was a single horizontal row. That is right for two
// machines and unreadable by six: circles tile against each other and every
// edge label lands on the same midline. A row up to three, a ring beyond, both
// derived from a stable sort.

/** The canvas the caller draws into. Fixed, so node count never resizes it. */
export const W = 760;
export const H = 320;
/** Node radius. */
export const R = 34;

/** Link classes ranked; higher is better. Mirrors the agent's own ordering. */
const RANK = { infini_band: 5, roce: 4, ethernet: 3, wireless: 2, unverified: 1 };

/**
 * The address a collective would actually use, or null.
 *
 * Loopback and virtual links are excluded: reachable from one machine and no
 * other, so they say nothing about a link *between* two.
 *
 * @param {object} node
 * @returns {object|null}
 */
export function preferred(node) {
  const usable = (node?.addresses ?? []).filter(
    (a) => a.class !== 'virtual' && a.class !== 'loopback'
  );
  if (usable.length === 0) return null;
  return usable.slice().sort((a, b) => {
    const r = (RANK[b.class] ?? 0) - (RANK[a.class] ?? 0);
    return r !== 0 ? r : (b.speedMbps ?? 0) - (a.speedMbps ?? 0);
  })[0];
}

/**
 * Machines in drawing order: this one, then trusted, then the rest.
 *
 * Ordering cannot depend on arrival time, or a node appearing would renumber
 * every position and shove the whole picture sideways.
 *
 * @param {object[]} nodes
 * @returns {object[]}
 */
export function ordered(nodes) {
  const weight = (n) => (n?.isLocal ? 0 : n?.pairing === 'paired' ? 1 : 2);
  return (Array.isArray(nodes) ? nodes : [])
    .filter((n) => n && typeof n.id === 'string' && n.id.length > 0)
    .slice()
    .sort((a, b) => weight(a) - weight(b) || a.id.localeCompare(b.id));
}

/**
 * Positions for a fleet.
 *
 * Up to three machines sit in a row, which reads as a line of hardware. Beyond
 * that they sit on a ring, which is the only arrangement that keeps every pair
 * visibly connected without edges crossing the nodes themselves.
 *
 * @param {object[]} nodes
 * @returns {{node: object, x: number, y: number}[]}
 */
export function points(nodes) {
  const list = ordered(nodes);
  const n = list.length;
  if (n === 0) return [];
  if (n === 1) return [{ node: list[0], x: W / 2, y: H / 2 }];

  if (n <= 3) {
    const span = Math.min(W - 2 * (R + 40), n * 210);
    const step = span / (n - 1);
    return list.map((node, i) => ({
      node,
      x: W / 2 - span / 2 + i * step,
      y: H / 2
    }));
  }

  // A ring, starting at the left so the local machine keeps the position it
  // had in the row — moving it would make growing a fleet feel like a reset.
  const rx = Math.min(W / 2 - (R + 46), 300);
  const ry = Math.min(H / 2 - (R + 22), 110);
  return list.map((node, i) => {
    const t = Math.PI + (2 * Math.PI * i) / n;
    return {
      node,
      x: W / 2 + rx * Math.cos(t),
      y: H / 2 + ry * Math.sin(t)
    };
  });
}

/**
 * The lines between machines that trust each other.
 *
 * Only trusted pairs: an unpaired machine has no relationship to draw. The
 * class is the worse of the two endpoints, because a link is only as good as
 * its poorer half, and an unknown end makes the whole edge unknown rather than
 * optimistically ethernet.
 *
 * @param {{node: object, x: number, y: number}[]} pts
 * @returns {object[]}
 */
export function edges(pts) {
  const trusted = pts.filter((p) => p.node.isLocal || p.node.pairing === 'paired');
  const out = [];
  for (let i = 0; i < trusted.length; i += 1) {
    for (let j = i + 1; j < trusted.length; j += 1) {
      const a = trusted[i];
      const b = trusted[j];
      const pa = preferred(a.node);
      const pb = preferred(b.node);
      const cls = !pa || !pb ? 'none' : (RANK[pa.class] ?? 0) <= (RANK[pb.class] ?? 0) ? pa.class : pb.class;
      const speed =
        pa?.speedMbps && pb?.speedMbps ? Math.min(pa.speedMbps, pb.speedMbps) : null;
      out.push({
        a,
        b,
        cls,
        speed,
        // RDMA is what multi-node decode needs; anything else is worth saying
        // out loud, but "unverified" is missing information rather than a slow
        // link and must not be reported as a fault.
        warn: cls !== 'roce' && cls !== 'infini_band' && cls !== 'unverified'
      });
    }
  }
  return out;
}

/**
 * A short, stable label for a machine.
 *
 * The fingerprint, not the hostname. Sparks ship with colliding names, and the
 * previous label — the last four characters of the hostname — collided for
 * exactly the machines an operator most needs to tell apart.
 *
 * @param {object} node
 * @returns {string}
 */
export function label(node) {
  return String(node?.id ?? '').slice(0, 4) || '????';
}
