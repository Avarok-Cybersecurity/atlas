// SPDX-License-Identifier: AGPL-3.0-only

// Write a generated file only when its DATA changed.
//
// The generators stamp their output with the commit and date that produced it.
// Those stamps move on every build, so `npm run build` rewrote four committed
// files whose contents were otherwise byte-identical, and left the tree dirty.
// That is not cosmetic: `git add site/src` then sweeps them into an unrelated
// commit, which happened twice in one session before this existed.
//
// Keeping the OLD stamp when the data is unchanged is also the more accurate
// record. The stamp answers "which commit produced this data", and if a later
// commit produces byte-identical data, the earlier answer is the true one.

import { existsSync, readFileSync, writeFileSync } from 'node:fs';

/** Deep equality over JSON-shaped values, ignoring key order. */
function sameJson(a, b) {
  return JSON.stringify(sortKeys(a)) === JSON.stringify(sortKeys(b));
}

function sortKeys(v) {
  if (Array.isArray(v)) return v.map(sortKeys);
  if (v && typeof v === 'object') {
    return Object.fromEntries(
      Object.keys(v)
        .sort()
        .map((k) => [k, sortKeys(v[k])])
    );
  }
  return v;
}

/**
 * Write `obj` to `path`, unless the only difference from what is already there
 * is one of `stampKeys`.
 *
 * @param {string} path
 * @param {object} obj              the freshly generated object
 * @param {string[]} stampKeys      top-level provenance keys to ignore when comparing
 * @param {(o: object) => string} serialize  each generator keeps its own formatting
 * @returns {boolean} whether the file was written
 */
export function writeStable(path, obj, stampKeys, serialize) {
  if (existsSync(path)) {
    try {
      const old = JSON.parse(readFileSync(path, 'utf8'));
      const strip = (o) => {
        const c = { ...o };
        for (const k of stampKeys) delete c[k];
        return c;
      };
      if (sameJson(strip(old), strip(obj))) {
        // Keep the file exactly as it is — including its original stamps.
        return false;
      }
    } catch {
      // Unreadable or not JSON: fall through and write a good one.
    }
  }
  writeFileSync(path, serialize(obj));
  return true;
}
