// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import {
  MAX_VISIBLE_POINTS_PER_CHART,
  aggregateSeries,
  bucketSizes,
  chartGroupSize,
  liftEdges,
  medianMember,
  nodeCountFor
} from './gate-aggregate.js';

/** A point as GateChart builds them, with only the fields this module reads. */
const pt = (t, v, verdict = 'PASS', sha = `sha${t}`) => ({
  t,
  v,
  rec: { git_sha: sha, verdict, recorded_at: t }
});
const run = (n, f = (i) => i) => Array.from({ length: n }, (_, i) => pt(i, f(i)));
const sizes = (nodes) => nodes.map((n) => n.count);

describe('the cap', () => {
  test('a chart at or under the cap is not aggregated at all', () => {
    // The boundary matters: 48 must render raw, 49 must not. A cap that were
    // exclusive, or off by one, changes exactly one of these.
    expect(chartGroupSize([run(41)])).toBe(1);
    expect(chartGroupSize([run(48)])).toBe(1);
    expect(chartGroupSize([run(49)])).not.toBe(1);
  });

  test('group size answers to the cap and to the series lengths, not a constant', () => {
    // Four different answers from one function: a hardcoded g cannot pass.
    expect(chartGroupSize([run(77)])).toBe(2);
    expect(chartGroupSize([run(77)], 10)).toBe(10);
    expect(chartGroupSize([run(41)])).toBe(1);
    expect(chartGroupSize([run(300)])).toBeGreaterThan(2);
  });

  test('the LONGEST series sets the size; adding a twin does not shrink it', () => {
    // Every series on a panel shares one x-axis and is sampled at the same
    // commits, so a second metric adds no horizontal crowding. Summing the
    // series would halve the group size for a reason invisible to the reader.
    expect(chartGroupSize([run(77), run(77)])).toBe(chartGroupSize([run(77)]));
    expect(chartGroupSize([run(41), run(77)])).toBe(chartGroupSize([run(77)]));
    // ...but a longer series does raise it
    expect(chartGroupSize([run(41), run(300)])).toBeGreaterThan(chartGroupSize([run(41), run(77)]));
  });

  test('every series is brought under the cap', () => {
    const many = [run(77), run(120), run(41)];
    const g = chartGroupSize(many);
    for (const pts of many) {
      expect(nodeCountFor(pts, g)).toBeLessThanOrEqual(MAX_VISIBLE_POINTS_PER_CHART);
    }
  });

  test('the exported cap is the one the derivation uses', () => {
    // Guards against the constant drifting away from the default parameter.
    expect(chartGroupSize([run(MAX_VISIBLE_POINTS_PER_CHART)])).toBe(1);
    expect(chartGroupSize([run(MAX_VISIBLE_POINTS_PER_CHART + 1)])).toBeGreaterThan(1);
  });
});

describe('bucketing', () => {
  test('the first and last points are never merged into a group', () => {
    for (const n of [49, 77]) {
      const pts = run(n);
      const s = bucketSizes(pts, chartGroupSize([pts]));
      expect(s[0]).toBe(1);
      expect(s[s.length - 1]).toBe(1);
    }
  });

  test('the endpoints keep their own value after aggregation', () => {
    const pts = run(77, (i) => i * 3);
    const nodes = aggregateSeries(pts, chartGroupSize([pts]));
    expect(nodes[0].v).toBe(pts[0].v);
    expect(nodes[nodes.length - 1].v).toBe(pts[76].v);
    expect(nodes[0].aggregated).toBe(false);
    expect(nodes[nodes.length - 1].aggregated).toBe(false);
  });

  test('middle buckets are balanced rather than leaving a runt', () => {
    // 9 points => 7 middles at g=3. Naive fixed chunking gives [3,3,1]; the
    // balanced split gives [3,2,2]. This is the case that distinguishes them.
    expect(bucketSizes(run(9), 3)).toEqual([1, 3, 2, 2, 1]);
  });

  test('no bucket ever exceeds the group size, and sizes differ by at most one', () => {
    for (const n of [7, 9, 20, 49, 77]) {
      for (const g of [2, 3, 4, 5]) {
        const s = bucketSizes(run(n), g);
        const middle = s.slice(1, -1);
        expect(Math.max(...middle, 1)).toBeLessThanOrEqual(g);
        if (middle.length) expect(Math.max(...middle) - Math.min(...middle)).toBeLessThanOrEqual(1);
      }
    }
  });

  test('degenerate series survive', () => {
    for (const n of [0, 1, 2, 3]) {
      expect(bucketSizes(run(n), 3).reduce((a, b) => a + b, 0)).toBe(n);
    }
    expect(bucketSizes(run(3), 3)).toEqual([1, 1, 1]);
    expect(aggregateSeries(run(0), 3)).toEqual([]);
  });

  test('nothing is dropped: members always re-add to the input length', () => {
    for (const n of [5, 41, 49, 77]) {
      const pts = run(n);
      const nodes = aggregateSeries(pts, 3);
      expect(nodes.reduce((a, x) => a + x.count, 0)).toBe(n);
      expect(nodes.flatMap((x) => x.members)).toEqual(pts);
    }
  });

  test('a group size of one leaves every point exactly where it was', () => {
    const pts = run(41, (i) => i * 7);
    const nodes = aggregateSeries(pts, 1);
    expect(nodes.map((n) => [n.t, n.v, n.aggregated])).toEqual(
      pts.map((p) => [p.t, p.v, false])
    );
  });
});

describe('a failure is never averaged away', () => {
  test('a non-PASS record stays its own point', () => {
    const pts = run(30);
    pts[13].rec.verdict = 'FAIL';
    pts[20].rec.verdict = 'info';
    const nodes = aggregateSeries(pts, 3);
    const fail = nodes.find((n) => n.members.some((m) => m.rec.verdict === 'FAIL'));
    const info = nodes.find((n) => n.members.some((m) => m.rec.verdict === 'info'));
    expect(fail.count).toBe(1);
    expect(fail.aggregated).toBe(false);
    expect(info.count).toBe(1);
  });

  test('NEGATIVE CONTROL: the same series all-PASS does aggregate there', () => {
    // Without this, the test above would pass on an implementation that simply
    // never aggregates anything.
    const nodes = aggregateSeries(run(30), 3);
    expect(Math.max(...sizes(nodes))).toBeGreaterThan(1);
  });

  test('allPass reports the group honestly in both directions', () => {
    const bad = run(4);
    bad[1].rec.verdict = 'FAIL';
    expect(aggregateSeries(bad, 4).some((n) => n.allPass === false)).toBe(true);
    expect(aggregateSeries(run(4), 4).every((n) => n.allPass)).toBe(true);
  });
});

describe('the plotted value', () => {
  test('is the lower median member, not the mean', () => {
    // mean([1,2,100]) is 34.33 and mean([1,2,3,100]) is 26.5; neither is a
    // value any run produced. Both assertions fail under a mean.
    expect(medianMember([pt(0, 1), pt(1, 2), pt(2, 100)]).v).toBe(2);
    expect(medianMember([pt(0, 1), pt(1, 2), pt(2, 3), pt(3, 100)]).v).toBe(2);
  });

  test('is an actual record, carrying its own sha and timestamp', () => {
    // Five points at g=3 bucket as [1,3,1]: the endpoints stay solo and the
    // middle three form the group under test.
    const pts = [
      pt(9, 1, 'PASS', 'zzz'), pt(10, 5, 'PASS', 'aaa'),
      pt(11, 900, 'PASS', 'bbb'), pt(12, 7, 'PASS', 'ccc'), pt(13, 2, 'PASS', 'yyy')
    ];
    const nodes = aggregateSeries(pts, 3);
    expect(sizes(nodes)).toEqual([1, 3, 1]);
    const group = nodes[1];
    // lower median of [5, 7, 900] is 7 -> the record 'ccc' actually produced it
    expect(group.v).toBe(7);
    expect(group.t).toBe(12);
    expect(group.rec.git_sha).toBe('ccc');
    // and the extreme it did not plot is still reachable
    expect(group.vMax).toBe(900);
  });

  test('spread reports the true extremes of the group', () => {
    const pts = [pt(9, 1), pt(10, 5), pt(11, 900), pt(12, 7), pt(13, 2)];
    const group = aggregateSeries(pts, 3)[1];
    expect([group.vMin, group.vMax]).toEqual([5, 900]);
    expect([group.tMin, group.tMax]).toEqual([10, 12]);
  });

  test('ids are stable across rebuilds and unique within a series', () => {
    const pts = run(20);
    const a = aggregateSeries(pts, 3).map((n) => n.id);
    const b = aggregateSeries(pts, 3).map((n) => n.id);
    expect(a).toEqual(b);
    expect(new Set(a).size).toBe(a.length);
  });
});

describe('lifting lineage onto nodes', () => {
  test('cross-bucket edges are kept and within-bucket edges disappear', () => {
    const pts = run(6);
    const nodes = aggregateSeries(pts, 2); // [1,2,2,1]
    expect(sizes(nodes)).toEqual([1, 2, 2, 1]);
    const chain = [
      [pts[0], pts[1]], [pts[1], pts[2]], [pts[2], pts[3]],
      [pts[3], pts[4]], [pts[4], pts[5]]
    ];
    const lifted = liftEdges(nodes, chain);
    expect(lifted.map((e) => [nodes.indexOf(e.a), nodes.indexOf(e.b)])).toEqual([
      [0, 1], [1, 2], [2, 3]
    ]);
  });

  test('NEGATIVE CONTROL: no proven edges means no drawn edges', () => {
    // Records that never went through the generator carry no predecessor, so
    // trendEdges returns nothing. Aggregation must not invent adjacency.
    const pts = run(20);
    const nodes = aggregateSeries(pts, 3);
    expect(nodes.length).toBeGreaterThan(1);
    expect(liftEdges(nodes, [])).toEqual([]);
  });

  test('two receipts crossing the same bucket boundary make one edge with support 2', () => {
    const pts = run(6);
    const nodes = aggregateSeries(pts, 2);
    const lifted = liftEdges(nodes, [[pts[0], pts[1]], [pts[0], pts[2]]]);
    expect(lifted).toHaveLength(1);
    expect(lifted[0].support).toBe(2);
  });
});
