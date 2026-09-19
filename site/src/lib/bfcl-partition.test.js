// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  aggregate,
  foldPartitions,
  shardOf,
  talliesOf,
  unionTallies
} from './bfcl-partition.js';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

/** Every committed BFCL record, as the generator would read them. */
function committed(bench) {
  const dir = join(REPO, '.benchmarks', bench);
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith('.json'))
    .map((f) => JSON.parse(readFileSync(join(dir, f), 'utf8')));
}

const BENCHES = ['bfcl-subset', 'bfcl-subset-echolp'];

// -- the conformance test --------------------------------------------------
// This is the one that matters. `aggregate()` is a hand port of
// `crates/avarok-plugin/src/benchmarks/bfcl/aggregate.rs`, which is itself a
// hand port of BFCL's `score.py`. A port can be self-consistently wrong, so it
// is never tested against numbers this file chose: it is fed each committed
// record's OWN per-subset tallies and must reproduce the `overall_accuracy`
// and `normalized_single_turn_score` the Rust scorer wrote beside them.
//
// Feeding one shard's tallies is legitimate precisely because the arithmetic
// is applied to whatever union it is handed -- which is the same property that
// makes aggregating six shards correct.
describe('conformance with the scorer that wrote the records', () => {
  for (const bench of BENCHES) {
    const recs = committed(bench);
    test(`${bench}: every committed record re-scores to its own published numbers (n=${recs.length})`, () => {
      expect(recs.length).toBeGreaterThan(0);
      let checked = 0;
      for (const rec of recs) {
        const t = talliesOf(rec);
        if (Object.keys(t).length === 0) continue; // pre-dates per-subset tallies
        const a = aggregate(t);
        expect({
          file: rec.git_sha,
          overall: a.overall_accuracy,
          norm: a.normalized_single_turn_score,
          samples: a.total_samples
        }).toEqual({
          file: rec.git_sha,
          overall: rec.metrics.overall_accuracy,
          norm: rec.metrics.normalized_single_turn_score,
          samples: rec.metrics.samples
        });
        checked += 1;
      }
      expect(checked).toBeGreaterThan(0);
    });
  }
});

// -- the negative control --------------------------------------------------
// A conformance test that passes because both sides are trivially equal proves
// nothing. Perturb one hit in one subset and the re-score MUST diverge; if it
// does not, the test above is reading a field it never computed.
test('the conformance check can fail: one flipped sample moves the score', () => {
  const rec = committed('bfcl-subset').find((r) => Object.keys(talliesOf(r)).length > 0);
  expect(rec).toBeDefined();
  const t = talliesOf(rec);
  const before = aggregate(t);
  const victim = Object.keys(t).find((k) => t[k].hits > 0);
  t[victim] = { hits: t[victim].hits - 1, n: t[victim].n };
  const after = aggregate(t);
  expect(after.overall_accuracy).not.toBe(before.overall_accuracy);
  expect(after.normalized_single_turn_score).not.toBe(before.normalized_single_turn_score);
});

// -- the weighting the naive mean gets wrong --------------------------------
describe('the union is not the mean of the parts', () => {
  test('hallucination is unweighted, so shard means and the union disagree', () => {
    // irrelevance is small and perfect; live_irrelevance is large and weaker.
    // Split them across two shards so each shard sees only one -- exactly the
    // stride a real 6-way split produces -- and the mean-of-shards differs.
    const a = { irrelevance: { hits: 12, n: 12 }, live_irrelevance: { hits: 30, n: 50 } };
    const b = { irrelevance: { hits: 10, n: 12 }, live_irrelevance: { hits: 20, n: 50 } };
    const union = aggregate(unionTallies([a, b]));
    const meanOfShards =
      (aggregate(a).category_scores.hallucination + aggregate(b).category_scores.hallucination) / 2;
    // union: irrelevance 22/24 = 91.67, live_irrelevance 50/100 = 50 -> 70.83
    expect(union.category_scores.hallucination).toBe(70.83);
    expect(meanOfShards).not.toBe(union.category_scores.hallucination);
  });

  test('the three simple_* subsets collapse to ONE term inside non_live', () => {
    // simple_python is 8x simple_javascript here. If non_live were a flat mean
    // over four subsets, a perfect javascript would lift it far more than the
    // hierarchy allows.
    const t = {
      simple_python: { hits: 160, n: 200 },
      simple_javascript: { hits: 25, n: 25 },
      multiple: { hits: 80, n: 100 },
      parallel: { hits: 70, n: 100 }
    };
    const a = aggregate(t);
    // simple term = mean(0.8, 1.0) = 0.9; non_live = mean(0.9, 0.8, 0.7) = 0.8
    expect(a.category_scores.non_live).toBe(80);
    const flat = ((0.8 + 1.0 + 0.8 + 0.7) / 4) * 100;
    expect(a.category_scores.non_live).not.toBe(Math.round(flat * 100) / 100);
  });

  test('live is sample-weighted, unlike hallucination', () => {
    const t = {
      live_simple: { hits: 90, n: 100 },
      live_parallel: { hits: 2, n: 10 }
    };
    // weighted: 92/110 = 83.64, not mean(90, 20) = 55
    expect(aggregate(t).category_scores.live).toBe(83.64);
  });

  test('an absent subset changes no divisor', () => {
    const both = aggregate({
      irrelevance: { hits: 10, n: 10 },
      live_irrelevance: { hits: 0, n: 10 }
    });
    const one = aggregate({ irrelevance: { hits: 10, n: 10 } });
    expect(both.category_scores.hallucination).toBe(50);
    expect(one.category_scores.hallucination).toBe(100);
  });
});

// -- shard coordinates ------------------------------------------------------
describe('shardOf reads the record, never the filename', () => {
  test('an unsharded record is unsharded', () => {
    expect(shardOf({ metrics: { overall_accuracy: 80 } })).toBeNull();
  });
  test('a params.shard string alone is not enough', () => {
    expect(shardOf({ params: { shard: '5/6' }, metrics: {} })).toBeNull();
  });
  test('nonsense coordinates are refused rather than guessed at', () => {
    expect(shardOf({ metrics: { 'shard.index': 6, 'shard.count': 6 } })).toBeNull();
    expect(shardOf({ metrics: { 'shard.index': -1, 'shard.count': 6 } })).toBeNull();
    expect(shardOf({ metrics: { 'shard.index': 1.5, 'shard.count': 6 } })).toBeNull();
  });
  test('real coordinates are read', () => {
    expect(shardOf({ metrics: { 'shard.index': 4, 'shard.count': 6 } })).toEqual({
      index: 4,
      count: 6
    });
  });
});

// -- the fold ---------------------------------------------------------------
const shard = (i, c, sha, when, tallies, extra = {}) => ({
  benchmark_id: 'bfcl-subset',
  git_sha: sha,
  recorded_at: when,
  verdict: 'info',
  params: { min_overall: '82.6', min_normalized: '82.56' },
  metrics: {
    'shard.index': i,
    'shard.count': c,
    ...Object.fromEntries(
      Object.entries(tallies).flatMap(([k, t]) => [
        [`subset.${k}.hits`, t.hits],
        [`subset.${k}.n`, t.n]
      ])
    ),
    ...extra
  }
});

test('recorded_at must be epoch seconds; an ISO string is refused, not coerced', () => {
  // The fixtures in this file originally used ISO strings, which no record in
  // .benchmarks has ever contained. Date.parse'ing the real integer yields
  // NaN, every record collapses to time 0, and "newest per index" silently
  // becomes "whichever the caller listed last" -- so the trap is pinned here.
  const bad = shard(0, 2, 'jjj', 1788000000, { irrelevance: { hits: 5, n: 6 } });
  bad.recorded_at = '2026-09-01T00:00:00Z';
  expect(() => foldPartitions([bad])).toThrow(/numeric recorded_at/);
});

describe('foldPartitions', () => {
  test('a complete partition becomes ONE point carrying the union', () => {
    const recs = [
      shard(0, 2, 'aaa', 1788000000, { irrelevance: { hits: 5, n: 6 } }),
      shard(1, 2, 'aaa', 1788003600, { irrelevance: { hits: 4, n: 6 } })
    ];
    const { records, partial } = foldPartitions(recs);
    expect(partial).toEqual([]);
    expect(records).toHaveLength(1);
    expect(records[0].metrics['subset.irrelevance.hits']).toBe(9);
    expect(records[0].metrics['subset.irrelevance.n']).toBe(12);
    expect(records[0].metrics.samples).toBe(12);
    expect(records[0].partition.count).toBe(2);
    expect(records[0].partition.members).toHaveLength(2);
    // The completion time, not the first shard's.
    expect(records[0].recorded_at).toBe(1788003600);
  });

  test('an incomplete partition is withheld, and SAID so -- never plotted, never dropped', () => {
    const recs = [shard(0, 6, 'bbb', 1788000000, { irrelevance: { hits: 5, n: 6 } })];
    const { records, partial } = foldPartitions(recs);
    expect(records).toEqual([]);
    expect(partial).toHaveLength(1);
    expect(partial[0]).toMatchObject({ count: 6, git_sha: 'bbb', held: [0] });
    expect(partial[0].records).toHaveLength(1);
  });

  test('a duplicated index is a re-run, not a double count', () => {
    // Two records of index 0 and none of index 1: the sample total still looks
    // plausible, which is exactly the silently-wrong green the gate's own
    // select_partition exists to prevent.
    const recs = [
      shard(0, 2, 'ccc', 1788000000, { irrelevance: { hits: 5, n: 6 } }),
      shard(0, 2, 'ccc', 1788007200, { irrelevance: { hits: 6, n: 6 } })
    ];
    const { records, partial } = foldPartitions(recs);
    expect(records).toEqual([]);
    expect(partial[0].held).toEqual([0]);
    // and the newest won, so a later completion would use it
    expect(partial[0].records[0].recorded_at).toBe(1788007200);
  });

  test('a partition is never assembled ACROSS commits', () => {
    const recs = [
      shard(0, 2, 'ddd', 1788000000, { irrelevance: { hits: 5, n: 6 } }),
      shard(1, 2, 'eee', 1788003600, { irrelevance: { hits: 4, n: 6 } })
    ];
    const { records, partial } = foldPartitions(recs);
    expect(records).toEqual([]);
    expect(partial).toHaveLength(2);
  });

  test('unsharded records pass through untouched and stay sorted with the folds', () => {
    const plain = {
      git_sha: 'fff',
      recorded_at: 1788086400,
      verdict: 'PASS',
      metrics: { overall_accuracy: 83 }
    };
    const { records } = foldPartitions([
      plain,
      shard(0, 2, 'aaa', 1788000000, { irrelevance: { hits: 5, n: 6 } }),
      shard(1, 2, 'aaa', 1788003600, { irrelevance: { hits: 4, n: 6 } })
    ]);
    expect(records.map((r) => r.git_sha)).toEqual(['aaa', 'fff']);
    expect(records[1]).toBe(plain);
  });

  test('the verdict is re-derived from the gate thresholds, not inherited from info', () => {
    const under = foldPartitions([
      shard(0, 2, 'ggg', 1788000000, { irrelevance: { hits: 1, n: 6 } }),
      shard(1, 2, 'ggg', 1788003600, { irrelevance: { hits: 1, n: 6 } })
    ]).records[0];
    expect(under.verdict).toBe('FAIL');
    const over = foldPartitions([
      shard(0, 2, 'hhh', 1788000000, { irrelevance: { hits: 6, n: 6 } }),
      shard(1, 2, 'hhh', 1788003600, { irrelevance: { hits: 6, n: 6 } })
    ]).records[0];
    expect(over.verdict).toBe('PASS');
  });

  test('with no threshold in params the fold claims no verdict', () => {
    const bare = (i) => {
      const r = shard(i, 2, 'iii', 1788000000 + i * 3600, { irrelevance: { hits: 6, n: 6 } });
      r.params = {};
      return r;
    };
    const rec = foldPartitions([bare(0), bare(1)]).records[0];
    expect(rec.verdict).toBe('info');
  });
});

// -- the whole committed set ------------------------------------------------
describe('on this repo`s own records', () => {
  for (const bench of BENCHES) {
    test(`${bench}: every shard is accounted for, either folded or named`, () => {
      const recs = committed(bench);
      const { records, partial } = foldPartitions(recs);
      const foldedShards = records.reduce((a, r) => a + (r.partition?.members.length ?? 0), 0);
      const plain = records.filter((r) => !r.partition).length;
      const held = partial.reduce((a, p) => a + p.records.length, 0);
      const sharded = recs.filter((r) => shardOf(r)).length;
      // Newest-per-index can legitimately shadow a re-run, so the identity is
      // an upper bound, not an equality -- but nothing may appear twice.
      expect(plain + sharded).toBe(recs.length);
      expect(foldedShards + held).toBeLessThanOrEqual(sharded);
      expect(records.length).toBeLessThan(recs.length);
    });
  }
});
