// SPDX-License-Identifier: AGPL-3.0-only

// The fleet Σ, and why it is captioned an approximation.
//
// Pure and plain `.js` for the house reason: whether stale telemetry leaks
// into the headline number is testable logic, and a file holding runes cannot
// be imported by the test runner.
//
// Each node's latest reading covers ITS OWN window, set by its own poll — the
// selected node every second or two, the rest every ten. Summing them is
// therefore not a measurement anyone took; it is a sum of latest per-node
// readings over windows that differ. Useful, and honest only while the
// caption says exactly that — so the caption is built here, beside the sum,
// where the two cannot drift apart.

import { sanitize } from './ingest.js';

/**
 * A reading older than this is excluded from the Σ. At the slowest poll
 * cadence (10s) plus backoff headroom, 30s means three missed rounds — the
 * node is not "between polls", it has stopped answering, and its last number
 * would silently pad the fleet total for as long as it stayed wedged.
 */
export const STALE_AFTER_MS = 30_000;

/** The always-true part of the caption. */
export const CAPTION = 'Σ of latest per-node readings · windows differ · this session';

/**
 * Sum one optional field across readings. Null when NO reading carries it:
 * eight nodes that have not reported decode yet are not a fleet doing
 * 0 tok/s, and rendering the difference is the whole absent-is-never-zero
 * rule. A wire-carried 0 still sums as 0.
 *
 * @param {object[]} readings
 * @param {string} field
 * @returns {number|null}
 */
function sum(readings, field) {
  let total = null;
  for (const r of readings) {
    const v = r?.[field];
    if (Number.isFinite(v)) total = (total ?? 0) + v;
  }
  return total;
}

/**
 * The fleet aggregate: Σ decode tok/s, Σ requests active, and the caption
 * that keeps them honest.
 *
 * @param {{id: string, name: string, at: number, reading: object|null}[]} entries
 *   one per node with a running launch; `at` is when its reading arrived
 * @param {number} nowMs
 * @returns {{decode: number|null, active: number|null, included: number,
 *   excluded: {id: string, name: string}[], caption: string}}
 */
export function aggregate(entries, nowMs) {
  if (!Number.isFinite(nowMs)) {
    throw new TypeError('aggregate needs the clock passed in, not assumed');
  }
  const list = (Array.isArray(entries) ? entries : []).filter((e) => e && typeof e.id === 'string');

  const fresh = [];
  const excluded = [];
  for (const e of list) {
    // An entry that cannot prove when its reading arrived is treated as
    // stale: freshness is the one claim this sum makes, and "unknown" is not
    // fresh.
    if (Number.isFinite(e.at) && nowMs - e.at <= STALE_AFTER_MS) {
      fresh.push(e);
    } else {
      excluded.push({ id: e.id, name: sanitize(e.name, 63) || e.id.slice(0, 8) });
    }
  }

  const readings = fresh.map((e) => e.reading);
  let caption = CAPTION;
  if (excluded.length > 0) {
    // The dropped nodes are NAMED, not counted: "3 excluded" sends the
    // operator hunting, a name sends them to the machine.
    caption += ` · excluding ${excluded.map((e) => e.name).join(', ')} (stale)`;
  }

  return {
    decode: sum(readings, 'decode_tokens_per_s'),
    active: sum(readings, 'requests_active'),
    included: fresh.length,
    excluded,
    caption
  };
}
