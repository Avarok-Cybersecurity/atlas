// SPDX-License-Identifier: AGPL-3.0-only

// bfcl-partition.js — turn BFCL SHARD records into the one number the gate judged.
//
// A sharded BFCL campaign commits N records per commit, one per shard, each
// with `verdict: "info"` because no single shard is a result. The dashboard
// used to plot all N, which is why the BFCL panels read "all over the place":
// six points per commit, each a quarter-view of a strided draw, each drawn
// with a fail-ring because `info` is not PASS.
//
// This module does what the gate itself does:
//
//   1. group the shard records by (shard.count, git_sha) and keep the NEWEST
//      record per index — a re-run is a replacement, never a double count
//      (`gate/group.rs::select_partition`);
//   2. accept only a COMPLETE partition, every index 0..count present;
//   3. sum the per-subset (hits, n) INTEGER tallies across the partition and
//      aggregate ONCE over the union, mirroring `bfcl/aggregate.rs`, which in
//      turn mirrors `score.py`.
//
// Aggregating once over the union is the whole point. `score.py` weights
// hierarchically — the three `simple_*` subsets collapse to ONE term inside
// `non_live`, `hallucination` is an UNWEIGHTED mean of two subsets of very
// different size, `live` is sample-weighted — so a mean of six shards'
// `normalized_single_turn_score` is not the whole-set value. On the records in
// this repo the two differ by more than a point on hallucination
// (mean-of-shards 86.24 vs the true 86.17 on bfcl-subset-echolp), which is
// larger than the margins these panels exist to show.
//
// Pure and dependency-free so `bun test` can measure it directly.

/** Non-live subsets, in `score.py`'s order. */
export const NON_LIVE = [
  'simple_python',
  'simple_java',
  'simple_javascript',
  'multiple',
  'parallel',
  'parallel_multiple'
];
/** Live subsets. */
export const LIVE = ['live_simple', 'live_multiple', 'live_parallel', 'live_parallel_multiple'];
/** Hallucination subsets. */
export const HALLUCINATION = ['irrelevance', 'live_irrelevance'];
/** The three `simple_*` subsets collapse to ONE term inside `non_live`. */
export const SIMPLE_AST = ['simple_python', 'simple_java', 'simple_javascript'];

const round2 = (x) => Math.round(x * 100) / 100;
const mean = (v) => (v.length === 0 ? 0 : v.reduce((a, b) => a + b, 0) / v.length);

/**
 * Per-subset `(hits, n)` counts carried by one shard record's metrics.
 *
 * Read from `metrics['subset.<name>.hits'] / ['subset.<name>.n']`, which the
 * benchmark writes for every subset it scored. A subset the shard did not see
 * is ABSENT, not zero: `score.py` builds each category from the subsets
 * present, so a zero would silently change a divisor.
 *
 * @param {{metrics?: Record<string, number>}} rec
 * @returns {Record<string, {hits: number, n: number}>}
 */
export function talliesOf(rec) {
  const out = {};
  const m = rec?.metrics ?? {};
  for (const key of Object.keys(m)) {
    const hit = /^subset\.(.+)\.hits$/.exec(key);
    if (!hit) continue;
    const name = hit[1];
    const n = m[`subset.${name}.n`];
    if (typeof n !== 'number' || typeof m[key] !== 'number') continue;
    out[name] = { hits: m[key], n };
  }
  return out;
}

/**
 * Union of several shards' tallies.
 * @param {Array<Record<string, {hits: number, n: number}>>} parts
 */
export function unionTallies(parts) {
  const out = {};
  for (const part of parts) {
    for (const [name, t] of Object.entries(part)) {
      const cur = out[name] ?? { hits: 0, n: 0 };
      out[name] = { hits: cur.hits + t.hits, n: cur.n + t.n };
    }
  }
  return out;
}

/**
 * Aggregate per-subset tallies exactly as `score.py` would over the same rows.
 *
 * Mirrors `crates/avarok-plugin/src/benchmarks/bfcl/aggregate.rs::aggregate`.
 * Given ONE shard's tallies it returns that shard's own score, which is what
 * makes the conformance test against the committed shard records possible.
 *
 * @param {Record<string, {hits: number, n: number}>} tallies
 */
export function aggregate(tallies) {
  /** @type {Record<string, number>} */
  const subsetMean = {};
  for (const [k, t] of Object.entries(tallies)) if (t.n > 0) subsetMean[k] = t.hits / t.n;

  /** @type {Record<string, number>} */
  const cat = {};

  // hallucination: UNWEIGHTED mean over the subsets present.
  const halluc = HALLUCINATION.filter((s) => s in subsetMean).map((s) => subsetMean[s]);
  if (halluc.length) cat.hallucination = mean(halluc);

  // live: sample-weighted, i.e. the flat mean over live samples.
  const live = LIVE.filter((s) => s in subsetMean);
  if (live.length) {
    const total = live.reduce((a, s) => a + tallies[s].n, 0);
    if (total > 0) cat.live = live.reduce((a, s) => a + subsetMean[s] * tallies[s].n, 0) / total;
  }

  // non_live: HIERARCHICAL — the three simple_* collapse to one term.
  const nonLivePresent = NON_LIVE.filter((s) => s in subsetMean);
  if (nonLivePresent.length) {
    const simple = SIMPLE_AST.filter((s) => s in subsetMean).map((s) => subsetMean[s]);
    const top = [];
    if (simple.length) top.push(mean(simple));
    for (const s of nonLivePresent) if (!SIMPLE_AST.includes(s)) top.push(subsetMean[s]);
    if (top.length) cat.non_live = mean(top);
  }

  const keys = Object.keys(cat).sort();
  const normalized = keys.length === 0 ? 0 : mean(keys.map((k) => cat[k]));
  const hits = Object.values(tallies).reduce((a, t) => a + t.hits, 0);
  const n = Object.values(tallies).reduce((a, t) => a + t.n, 0);

  /** @type {Record<string, number>} */
  const category_scores = {};
  for (const k of keys) category_scores[k] = round2(cat[k] * 100);
  return {
    overall_accuracy: round2((n > 0 ? hits / n : 0) * 100),
    normalized_single_turn_score: round2(normalized * 100),
    category_scores,
    total_samples: n
  };
}

/**
 * The shard coordinates a record declares, or null for an unsharded record.
 *
 * Read from `metrics['shard.index'] / ['shard.count']` — written by the driver
 * from the rows it was handed — rather than parsed out of the filename or the
 * `params.shard` string, so a record that says nothing is treated as unsharded
 * instead of being guessed at.
 *
 * @param {{metrics?: Record<string, number>}} rec
 */
export function shardOf(rec) {
  const i = rec?.metrics?.['shard.index'];
  const c = rec?.metrics?.['shard.count'];
  if (typeof i !== 'number' || typeof c !== 'number') return null;
  if (!Number.isInteger(i) || !Number.isInteger(c) || c < 1 || i < 0 || i >= c) return null;
  return { index: i, count: c };
}

/**
 * A record's measurement time.
 *
 * `recorded_at` is EPOCH SECONDS, an integer — every one of the 944 records in
 * this repo. It is read strictly rather than coerced because order is
 * load-bearing here twice over: it picks the surviving record when an index
 * was re-run, and it stamps the partition with its completion time. A silent
 * `|| 0` on a field this module misread would make every record equally old
 * and hand "newest per index" to whatever order the caller happened to pass —
 * the double-count `select_partition` exists to prevent, re-introduced in the
 * port of it.
 *
 * @param {{recorded_at?: unknown}} rec
 */
const at = (rec) => {
  const t = rec?.recorded_at;
  if (typeof t !== 'number' || !Number.isFinite(t)) {
    throw new TypeError(`record has no numeric recorded_at: ${JSON.stringify(t)}`);
  }
  return t;
};

/**
 * Replace every COMPLETE shard partition with the single aggregate record the
 * gate judged, and say what could not be assembled.
 *
 * Unsharded records pass through untouched and keep their identity — this is
 * not a filter, it is a fold of the records that were never individually a
 * result.
 *
 * Shards that do not complete a partition are NOT plotted and NOT silently
 * discarded: they come back in `partial`, so the caller can say how many
 * records are being withheld and why. A half-measured draw drawn as points
 * would be the same lie the fold exists to remove.
 *
 * @param {Array<object>} records for ONE benchmark
 * @returns {{records: Array<object>, partial: Array<{count: number, git_sha: string, held: number[], records: Array<object>}>}}
 */
export function foldPartitions(records) {
  const plain = [];
  /** @type {Map<string, {count: number, git_sha: string, byIndex: Map<number, object>}>} */
  const groups = new Map();

  for (const rec of records) {
    // Read the clock for EVERY record on the way in, sharded or not. Validating
    // only where a value is needed would let a malformed record through
    // whenever it happened to be the first of its index -- which is the case
    // this module is least able to notice and most damaged by.
    at(rec);
    const s = shardOf(rec);
    if (!s) {
      plain.push(rec);
      continue;
    }
    const key = `${s.count}@${rec.git_sha}`;
    let g = groups.get(key);
    if (!g) {
      g = { count: s.count, git_sha: rec.git_sha, byIndex: new Map() };
      groups.set(key, g);
    }
    // Newest per index: a re-run replaces, never double-counts.
    const prev = g.byIndex.get(s.index);
    if (!prev || at(rec) >= at(prev)) g.byIndex.set(s.index, rec);
  }

  const partial = [];
  for (const g of groups.values()) {
    const held = [...g.byIndex.keys()].sort((a, b) => a - b);
    const members = held.map((i) => g.byIndex.get(i));
    if (held.length !== g.count) {
      partial.push({ count: g.count, git_sha: g.git_sha, held, records: members });
      continue;
    }
    plain.push(aggregateRecord(members, g));
  }

  plain.sort((a, b) => at(a) - at(b));
  return { records: plain, partial };
}

/**
 * The synthetic record standing for one complete partition.
 *
 * Its `metrics` are RECOMPUTED from the union of the members' tallies, never
 * averaged from the members' own scores. Every provenance field is taken from
 * a member and is therefore true of every member: the partition is pinned to
 * one commit by construction, so `git_sha` is exact; `recorded_at` is the
 * partition's LAST shard, because that is when the measurement completed.
 *
 * The verdict is re-derived from the gate's own thresholds carried in
 * `params` (`min_overall`, `min_normalized`) rather than inherited from the
 * members, all of which read `info` precisely because a shard is not judged.
 * When a record carries no threshold the verdict stays `info` — the dashboard
 * must not invent a PASS the gate never issued.
 */
function aggregateRecord(members, g) {
  const agg = aggregate(unionTallies(members.map(talliesOf)));
  const newest = members.reduce((a, b) => (at(b) > at(a) ? b : a));
  const base = members[0];

  const metrics = {
    overall_accuracy: agg.overall_accuracy,
    normalized_single_turn_score: agg.normalized_single_turn_score,
    samples: agg.total_samples,
    transport_errors: members.reduce((a, m) => a + (m.metrics?.transport_errors ?? 0), 0)
  };
  for (const [k, v] of Object.entries(agg.category_scores)) metrics[`category.${k}`] = v;
  for (const [name, t] of Object.entries(unionTallies(members.map(talliesOf)))) {
    metrics[`subset.${name}.hits`] = t.hits;
    metrics[`subset.${name}.n`] = t.n;
  }

  const num = (x) => {
    const v = Number(x);
    return Number.isFinite(v) ? v : null;
  };
  const minOverall = num(base.params?.min_overall);
  const minNorm = num(base.params?.min_normalized);
  let verdict = 'info';
  let verdict_reason = `aggregate of ${g.count} shards; no threshold in params, so no verdict is claimed`;
  if (minOverall !== null || minNorm !== null) {
    const failed = [];
    if (minOverall !== null && agg.overall_accuracy < minOverall)
      failed.push(`overall ${agg.overall_accuracy} < ${minOverall}`);
    if (minNorm !== null && agg.normalized_single_turn_score < minNorm)
      failed.push(`normalized ${agg.normalized_single_turn_score} < ${minNorm}`);
    verdict = failed.length ? 'FAIL' : 'PASS';
    verdict_reason = failed.length
      ? failed.join('; ')
      : `aggregate of ${g.count} shards over ${agg.total_samples} samples`;
  }

  return {
    ...base,
    recorded_at: newest.recorded_at,
    metrics,
    verdict,
    verdict_reason,
    /** How this point came to exist, for the modal. */
    partition: {
      count: g.count,
      git_sha: g.git_sha,
      members,
      category_scores: agg.category_scores
    }
  };
}
