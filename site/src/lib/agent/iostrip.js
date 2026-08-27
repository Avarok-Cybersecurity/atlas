// SPDX-License-Identifier: AGPL-3.0-only

// Turning a LaunchReading into the serving I/O strip's tiles.
//
// Pure and plain `.js` for the house reason: these are the rules that decide
// whether an operator sees a number, a dash, or nothing, and a file holding
// runes cannot be imported by the test runner.
//
// **Absent is never zero, and absence comes in kinds.** A tile can be:
//   reading      the wire carried a value — 0 renders as 0 only here
//   pending      no reading has arrived yet — skeleton, "waiting for first sample"
//   absent       a reading arrived and this field was not in it — em-dash,
//                "the engine does not report this"
//   placeholder  the protocol itself has no field yet (ISL/OSL) — no value at
//                all, not even a dash; a dash would claim the engine was asked
// Collapsing any two of these teaches the operator that the dashboard guesses.

import { sanitize } from './ingest.js';
import * as S from './stats.js';

/**
 * A rate differenced across a window longer than this is a fabrication: it
 * averages over a gap — a backgrounded tab, an unreachable relay — and
 * presents the result as current throughput.
 */
export const MAX_RATE_WINDOW_S = 60;

/**
 * The two fields that are rates. TTFT percentiles and hit rates are window
 * statistics too, but they describe requests that really happened in the
 * window, however long it was; a tok/s number is the one that lies when the
 * window includes dead air.
 */
const RATE_FIELDS = ['decode_tokens_per_s', 'prompt_tokens_per_s'];

/**
 * Gate the rate fields of a reading against fabrication.
 *
 * Dropped — returned as absent, never zeroed — when:
 *  - this is the page's first poll: the agent differences against its previous
 *    scrape, which this page never scheduled and which may be hours old;
 *  - the reading does not say its window: without `window_s` the gap rule
 *    cannot be checked, so it fails closed;
 *  - the window exceeds `MAX_RATE_WINDOW_S`: that window spans a gap.
 *
 * @param {object|null} reading
 * @param {{firstPoll: boolean}} opts required — "don't know" must not pass the gate
 * @returns {object|null}
 */
export function gateRates(reading, opts) {
  if (opts?.firstPoll !== true && opts?.firstPoll !== false) {
    throw new TypeError('gateRates must be told whether this is the first poll');
  }
  if (reading == null) return null;
  const windowOk =
    Number.isFinite(reading.window_s) && reading.window_s <= MAX_RATE_WINDOW_S;
  if (!opts.firstPoll && windowOk) return reading;
  const out = { ...reading };
  for (const f of RATE_FIELDS) delete out[f];
  return out;
}

/**
 * What the strip as a whole shows.
 *
 * 'off'        nothing serving on this node — one sentence, never a wall of dashes
 * 'pending'    a launch is running and the first poll has not come back
 * 'unanswered' polls are failing — the launch exists but is not answering
 * 'quiet'      the engine answered with every field absent — loading, not idle
 * 'live'       there is at least one number to show
 *
 * `serving`, `reading` and `failure` are all required: a missing input here
 * is precisely the "don't know" this function exists to classify, and a
 * default would classify it silently.
 *
 * @param {{serving: boolean, reading: object|null, failure: string|null}} input
 * @returns {'off'|'pending'|'unanswered'|'quiet'|'live'}
 */
export function mode({ serving, reading, failure }) {
  if (serving !== true && serving !== false) {
    throw new TypeError('mode must be told whether a launch is running');
  }
  if (!serving) return 'off';
  if (reading == null) return failure ? 'unanswered' : 'pending';
  return S.hasAnything(reading) ? 'live' : 'quiet';
}

/** The tile set, in strip order. Fixed: telemetry arriving never reflows the strip. */
const TILES = [
  { id: 'decode', label: 'Decode', field: 'decode_tokens_per_s', fmt: S.tokens, unit: 'tok/s' },
  { id: 'prompt', label: 'Prompt', field: 'prompt_tokens_per_s', fmt: S.tokens, unit: 'tok/s' },
  { id: 'requests-total', label: 'Requests', field: 'requests_total', fmt: S.count, unit: '' },
  { id: 'requests-active', label: 'In flight', field: 'requests_active', fmt: S.count, unit: '' },
  { id: 'ttft-p50', label: 'TTFT median', field: 'ttft_p50_s', fmt: S.duration, unit: '' },
  { id: 'ttft-p90', label: 'TTFT p90', field: 'ttft_p90_s', fmt: S.duration, unit: '' },
  {
    id: 'accept',
    label: 'Draft accepted',
    field: 'accept_rate',
    fmt: S.percent,
    unit: '',
    // Absence has a specific, known meaning for this field.
    absentNote: 'not speculating'
  },
  { id: 'prefix', label: 'Prefix cache', field: 'prefix_hit_rate', fmt: S.percent, unit: '' }
];

/** ISL/OSL: same footprint as their live neighbours, no value slot at all. */
const PLACEHOLDER_TILES = [
  { id: 'isl', label: 'ISL' },
  { id: 'osl', label: 'OSL' }
];

/**
 * Tile view-models for one node's strip.
 *
 * The placeholder tiles carry NO `text` property — not '—', not ''. The
 * em-dash means "the engine was asked and does not report this", which is a
 * claim; for a field the protocol cannot carry yet, the claim would be false.
 *
 * @param {object|null} reading already gated by `gateRates`
 * @param {{paused: boolean}} opts
 * @returns {{id: string, label: string, kind: string}[]}
 */
export function tiles(reading, opts) {
  if (opts?.paused !== true && opts?.paused !== false) {
    throw new TypeError('tiles must be told whether polling is paused');
  }
  const out = TILES.map((t) => {
    if (reading == null) {
      return { id: t.id, label: t.label, kind: 'pending', paused: opts.paused };
    }
    const v = reading[t.field];
    if (!Number.isFinite(v)) {
      return {
        id: t.id,
        label: t.label,
        kind: 'absent',
        text: '—',
        note: t.absentNote ?? 'not reported',
        paused: opts.paused
      };
    }
    return {
      id: t.id,
      label: t.label,
      kind: 'reading',
      text: t.fmt(v),
      unit: t.unit,
      paused: opts.paused
    };
  });
  for (const p of PLACEHOLDER_TILES) {
    out.push({ id: p.id, label: p.label, kind: 'placeholder' });
  }
  return out;
}

/**
 * The strip caption: how fresh the numbers are, and who carried them.
 *
 * No `window_s` ⇒ no "measured over" claim — the page never invents a window
 * it was not told. `via` is a display name that crossed a peer channel, so it
 * is sanitised even though ingest usually got there first.
 *
 * @param {object|null} reading
 * @param {{via: string|null}} opts
 * @returns {string}
 */
export function caption(reading, opts) {
  const parts = [];
  if (Number.isFinite(reading?.window_s)) {
    parts.push(`measured over ${S.duration(reading.window_s)}`);
  }
  const via = opts?.via == null ? '' : sanitize(opts.via, 63);
  if (via) parts.push(`via ${via}`);
  return parts.join(' · ');
}
