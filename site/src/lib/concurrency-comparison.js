// SPDX-License-Identifier: AGPL-3.0-only

// concurrency-comparison.js — what the comparison chart of a subject tab draws.
//
// The decision is made here, pure and exported, so the chart, the panel
// around it (header tiles, the bridging note over the gate sweep) and the
// tests cannot disagree about what is being drawn:
//
//   published  the frozen campaign PAIR from the subject's manifest (dense);
//   live       the newest passing gate record on main, Atlas on the gate's
//              instrument — plus any one-shot baseline whose instrument
//              fingerprint EQUALS the record's (ladder-baselines.js decides,
//              axis by axis; a mismatch is named, never drawn);
//   baseline   a one-shot vLLM baseline on the published instrument and no
//              Atlas leg at it yet (MoE) — the vLLM curve, dated, and an
//              honest "no Atlas run at this instrument" state;
//   none       nothing to draw yet.
//
// A ladder is only honoured for the subject whose checkpoint it measured:
// the generator refuses a mismatch, and this refuses it again rather than
// trusting a build product — a 27B curve must never appear on a 35B tab.

import { comparable, describeDiffers } from './ladder-baselines.js';

export const subjectSeriesOf = (ladder) => ladder.series.find((s) => s.role === 'subject') ?? null;
export const baselineSeriesOf = (ladder) => ladder.series.filter((s) => s.role === 'baseline');

/** The generated ladder of a subject, or null when it has none for THIS checkpoint. */
export function ladderFor(subject, ladders) {
  if (subject.published_manifest === null) return null;
  const ladder = ladders.subjects[subject.id] ?? null;
  return ladder && ladder.workload.checkpoint === subject.checkpoint ? ladder : null;
}

/** The published pair — a ladder with a subject series — or null. */
export function publishedFor(subject, ladders) {
  const ladder = ladderFor(subject, ladders);
  return ladder && subjectSeriesOf(ladder) ? ladder : null;
}

/** A ladder with baselines only, no subject series — or null. */
export function baselineOnlyFor(subject, ladders) {
  const ladder = ladderFor(subject, ladders);
  return ladder && !subjectSeriesOf(ladder) ? ladder : null;
}

/** Newest passing gate record on main — the one the live series is drawn from. */
export const liveRecordOf = (records) => records.filter((r) => r.verdict === 'PASS' && !r.branch).at(-1) ?? null;

export function comparisonStateOf(subject, records, ladders) {
  if (publishedFor(subject, ladders)) return 'published';
  if (liveRecordOf(records)) return 'live';
  if (baselineOnlyFor(subject, ladders)) return 'baseline';
  return 'none';
}

/**
 * What ladder-baselines.js fingerprints for one baseline series: the axes it
 * declared, plus the checkpoint the whole ladder measured. The checkpoint is
 * read from the ladder, not copied into each series (SSOT).
 */
export const fingerprintOf = (ladder, series) => ({
  checkpoint: ladder.workload.checkpoint,
  instrument: series.instrument
});

/**
 * Which of a ladder's baselines may be drawn against a gate record, and why
 * the others may not. `why` is the axis list as the page prints it.
 * @returns {{drawn: object[], refused: Array<{series: object, differs: object[], why: string}>}}
 */
export function pairWith(live, ladder) {
  const drawn = [];
  const refused = [];
  for (const b of baselineSeriesOf(ladder)) {
    const { ok, differs } = comparable(live, fingerprintOf(ladder, b));
    if (ok) drawn.push(b);
    else refused.push({ series: b, differs, why: describeDiffers(differs) });
  }
  return { drawn, refused };
}

/**
 * The header tile beside "vLLM baseline", decided by the same rules as the
 * chart so the two cannot disagree: a pair, a dated one-shot that IS drawn,
 * a one-shot that exists but is on another instrument, or none.
 */
export function baselineTileOf(subject, records, ladders) {
  const state = comparisonStateOf(subject, records, ladders);
  if (state === 'published') return 'published pair';
  const ladder = ladderFor(subject, ladders);
  if (!ladder) return 'none';
  const dated = (bs) => `one-shot · ${measuredRange(bs.flatMap((b) => b.rungs))}`;
  if (state === 'baseline') return dated(baselineSeriesOf(ladder));
  const { drawn, refused } = pairWith(liveRecordOf(records), ladder);
  if (drawn.length) return dated(drawn);
  return refused.length ? 'other instrument' : 'none';
}

/**
 * Why a rung has no baseline point, from the manifest: a rung a series lists
 * as unmeasured carries its recorded reason; any other absence is named as
 * exactly that — never a zero, never a guess.
 */
export function absentReasonOf(ladder, c) {
  const listed = baselineSeriesOf(ladder).find((b) => b.unmeasured?.rungs.includes(c));
  return listed
    ? `C=${c} · not measured — ${listed.unmeasured.reason}`
    : `C=${c} · not in this manifest: neither measured nor listed as unmeasured`;
}

/**
 * The gate's instrument, read off a record, for chart titles and captions.
 * ISL/OSL are `params` (the harness pins them); batch cap and KV dtype are
 * `serve_overrides`. Every field is printed only if the record carries it —
 * a title must not claim a setting the run did not record.
 */
export function instrumentLabel(record) {
  const p = record.params ?? {};
  const o = record.serve_overrides ?? {};
  const parts = [];
  if (p.isls !== undefined && p.osl !== undefined) parts.push(`ISL ${p.isls} / OSL ${p.osl}`);
  if (p.prompt_mode) parts.push(`${p.prompt_mode} fixture`);
  if (o.max_batch_size) parts.push(`batch cap ${o.max_batch_size}`);
  if (o.kv_cache_dtype) parts.push(`${o.kv_cache_dtype} KV`);
  return parts.join(' · ');
}

/** `2026-08-17 → 2026-08-18`, or one date when a series was measured in a day. */
export function measuredRange(rungs) {
  const days = rungs.map((r) => r.measured_utc.slice(0, 10)).sort();
  const [from, to] = [days[0], days[days.length - 1]];
  return from === to ? from : `${from} → ${to}`;
}

/** The batch cap a baseline's serve command pinned, if its CLI records one. */
export function batchCapOf(cli) {
  const m = /--(?:max-num-seqs|max-batch-size)\s+(\d+)/.exec(cli ?? '');
  return m ? m[1] : null;
}

/** The legend chip every one-shot baseline gets: dated from its rungs, never typed. */
export const oneShotChip = (b) => `${b.label} · one-shot · measured ${measuredRange(b.rungs)} · not re-run`;
