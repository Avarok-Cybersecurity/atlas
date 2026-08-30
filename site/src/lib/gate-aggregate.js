// SPDX-License-Identifier: AGPL-3.0-only

// gate-aggregate.js — local aggregation for the gate dashboard's time charts.
//
// A benchmark accumulates one record per certified commit, so a chart that
// draws every record turns into 77 overplotted dots. This module collapses
// ADJACENT records into a single plotted node, under rules chosen so that
// collapsing can never hide something the dashboard exists to show:
//
//   * The FIRST and LAST points are never aggregated. They are the two a
//     reader actually looks up — "where did we start" and "where are we now".
//   * A non-PASS record is never aggregated. There are 9 FAIL and 6 `info`
//     records in 578; a median over three would silently erase them, which is
//     the exact regression this dashboard is meant to catch.
//   * An aggregated node plots a REAL MEMBER RECORD (the lower median by
//     value), never a computed average. The plotted point therefore keeps a
//     true git_sha, a true timestamp and a true measurement. An interpolated
//     median would put a number on the chart that no commit ever produced.
//   * Nothing is ever dropped: the member count over all nodes always equals
//     the input length, and every member is reachable through the node.
//
// Pure and dependency-free so `bun test` can measure it directly.

/**
 * The most points one chart may draw ALONG ITS X-AXIS.
 *
 * Bounding the longest series rather than the sum of all of them, because
 * every series on a panel shares one time axis and is sampled at the same
 * commits: `median` and `p90` of the same model sit at identical x positions.
 * Summing them would count one column of points twice and shrink the group
 * size for a reason the reader cannot see — on the four-model TTFT panel that
 * forced groups of five, where two is enough to clear the crowding.
 */
export const MAX_VISIBLE_POINTS_PER_CHART = 48;

/**
 * Whether a point may be merged into a group at all.
 *
 * Verdict is read from the record rather than passed in, because "a failure is
 * never averaged away" is a property of this module's honesty guarantee, not a
 * caller's styling choice — a caller must not be able to switch it off.
 *
 * @param {{rec: {verdict?: string}}} pt
 */
export const isAggregatable = (pt) => pt?.rec?.verdict === 'PASS';

/**
 * Sizes of the buckets a series is cut into, in order, summing to `pts.length`.
 *
 * Indices that must stand alone (first, last, every non-PASS) emit a bucket of
 * 1. Each remaining contiguous run is split into near-equal buckets of at most
 * `g`. Near-equal rather than fixed-`g` chunking is what avoids a runt: 7
 * middles at g=3 become [3,2,2], not [3,3,1].
 *
 * @param {Array<{rec?: object}>} pts
 * @param {number} g maximum bucket size; g <= 1 disables aggregation
 * @returns {number[]}
 */
export function bucketSizes(pts, g) {
  const n = pts.length;
  if (g <= 1 || n <= 2) return Array(n).fill(1);

  const solo = pts.map((p, i) => i === 0 || i === n - 1 || !isAggregatable(p));
  const out = [];
  let run = 0;
  const flush = () => {
    if (run === 0) return;
    const k = Math.ceil(run / g);
    const base = Math.floor(run / k);
    const extra = run % k;
    for (let i = 0; i < k; i += 1) out.push(base + (i < extra ? 1 : 0));
    run = 0;
  };
  for (let i = 0; i < n; i += 1) {
    if (solo[i]) {
      flush();
      out.push(1);
    } else {
      run += 1;
    }
  }
  flush();
  return out;
}

/**
 * How many nodes a series yields at group size `g`.
 * @param {Array<object>} pts
 * @param {number} g
 */
export const nodeCountFor = (pts, g) => bucketSizes(pts, g).length;

/**
 * The smallest group size at which every series in the chart fits inside `cap`.
 *
 * Returns 1 when nothing needs aggregating, so a chart under the cap renders
 * exactly as it did before this module existed.
 *
 * One size is shared by every series on purpose: two series drawn on one axis
 * at different temporal resolutions would misrepresent their relative
 * volatility — the coarser one would look smoother because it was binned
 * harder, not because it moved less.
 *
 * @param {Array<Array<object>>} seriesPts every series in the chart
 * @param {number} cap
 * @returns {number} group size >= 1
 */
export function chartGroupSize(seriesPts, cap = MAX_VISIBLE_POINTS_PER_CHART) {
  const total = (g) => Math.max(0, ...seriesPts.map((pts) => nodeCountFor(pts, g)));
  if (total(1) <= cap) return 1;
  // Monotone non-increasing in g, and bounded: past the longest series every
  // aggregatable run is a single bucket, so growing g further changes nothing.
  const gMax = Math.max(2, ...seriesPts.map((p) => p.length));
  for (let g = 2; g < gMax; g += 1) if (total(g) <= cap) return g;
  // Unreachable with today's panels (<=2 metrics x <=4 models). If it ever is
  // reached, the forced singletons alone exceed the cap: keeping first, last
  // and every failure visible outranks the cap, so return the coarsest size
  // rather than starting to merge them.
  return gMax;
}

/**
 * The member that represents a bucket: the LOWER MEDIAN by value.
 *
 * Lower median rather than an averaged one so the result is an observation
 * that actually happened. For an even bucket the mean of the two middles would
 * be a number no run produced, and it could not be linked to a commit.
 *
 * @param {Array<{v: number}>} members
 */
export function medianMember(members) {
  const sorted = [...members].sort((a, b) => a.v - b.v);
  return sorted[Math.ceil(sorted.length / 2) - 1];
}

/**
 * @typedef {object} Node
 * @property {number} t      plotted time — the representative member's, not an average
 * @property {number} v      plotted value — the representative member's
 * @property {object} rec    the representative record
 * @property {boolean} aggregated true when the node stands for more than one run
 * @property {number} count
 * @property {Array<object>} members chronological, verbatim
 * @property {number} vMin
 * @property {number} vMax
 * @property {number} tMin
 * @property {number} tMax
 * @property {boolean} allPass false when any member failed
 * @property {string} id      stable across rebuilds, unique within a series
 */

/**
 * Collapse a chronologically sorted series into plotted nodes.
 * @param {Array<{t: number, v: number, rec: object}>} pts
 * @param {number} g
 * @returns {Node[]}
 */
export function aggregateSeries(pts, g) {
  let at = 0;
  return bucketSizes(pts, g).map((size) => {
    const members = pts.slice(at, at + size);
    at += size;
    const pick = medianMember(members);
    const vs = members.map((p) => p.v);
    return {
      t: pick.t,
      v: pick.v,
      rec: pick.rec,
      aggregated: size > 1,
      count: size,
      members,
      vMin: Math.min(...vs),
      vMax: Math.max(...vs),
      tMin: members[0].t,
      tMax: members[size - 1].t,
      allPass: members.every(isAggregatable),
      id: `${members[0].rec.git_sha}:${members[0].t}:${size}`
    };
  });
}

/**
 * Re-express record-level lineage edges as edges between nodes.
 *
 * Lineage is a property of commits, so it is computed on the raw records and
 * only then lifted here — never the other way round. An aggregated node has no
 * git_sha of its own, so asking `trendEdges` about nodes would mean inventing a
 * commit identity for a group, which is a lineage claim the records do not
 * support.
 *
 * A lifted edge asserts exactly this: AT LEAST ONE receipt in `b` has a
 * generator-proven `trend_predecessor` inside `a`. `support` counts how many.
 * It does not claim the two plotted medians are themselves parent and child;
 * the modal exposes the members so the claim stays auditable.
 *
 * Edges inside one bucket disappear — a bucket is a positional grouping, not a
 * lineage claim — and two buckets with no proven edge between them stay
 * disconnected, exactly as unproven singletons do today.
 *
 * @param {Node[]} nodes
 * @param {Array<Array<object>>} recordEdges output of `trendEdges` over the SAME pts
 * @returns {Array<{a: Node, b: Node, support: number}>}
 */
export function liftEdges(nodes, recordEdges) {
  const nodeOf = new Map();
  for (const n of nodes) for (const m of n.members) nodeOf.set(m, n);
  const out = new Map();
  for (const edge of recordEdges) {
    // trendEdges yields [predecessor, successor] pairs; the loop also copes
    // with a longer chain should that representation ever widen.
    for (let i = 1; i < edge.length; i += 1) {
      const a = nodeOf.get(edge[i - 1]);
      const b = nodeOf.get(edge[i]);
      if (!a || !b || a === b) continue;
      const k = `${a.id}>${b.id}`;
      const seen = out.get(k);
      if (seen) seen.support += 1;
      else out.set(k, { a, b, support: 1 });
    }
  }
  return [...out.values()];
}
