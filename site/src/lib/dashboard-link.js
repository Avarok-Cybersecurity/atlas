// SPDX-License-Identifier: AGPL-3.0-only
//
// dashboard-link.js — the benchmark dashboard's deep link.
//
//   /engine#bench=concurrency&subject=qwen36-35b-a3b&c=64
//
// A hash, not a query string: /engine is prerendered and touching
// `url.searchParams` during prerender is a build error, while the hash never
// reaches the prerenderer (the deck made the same call). This module is pure —
// the component reads `location` and writes with `$app/navigation` — and it
// never throws on the hash, because a pasted URL is untrusted input. What is
// valid (tab ids, subject ids, rungs) is passed in, so the resolution rules are
// testable without the generated data and a stale hash cannot pick a tab or
// subject that no longer exists.

export const RUNG_ALL = 'all';

const KEY = { tab: 'bench', subject: 'subject', rung: 'c' };
const KNOWN_LISTS = ['tabIds', 'subjectIds', 'rungs'];

/** An unknown or missing tab is "not a deep link" — the dashboard opens as it would from a click. */
export function resolveTab(raw, tabIds) {
  return tabIds.includes(raw) ? raw : null;
}

/** An unknown or missing subject lands on the first subject; `null` only when there are none. */
export function resolveSubject(raw, subjectIds) {
  if (subjectIds.includes(raw)) return raw;
  return subjectIds.length > 0 ? subjectIds[0] : null;
}

/**
 * `all` or one of the declared rungs, else `all`. The match is on the exact
 * decimal spelling so `064`, ` 64` and `64.0` do not sneak in as C=64.
 */
export function resolveRung(raw, rungs) {
  if (raw === RUNG_ALL) return RUNG_ALL;
  if (typeof raw !== 'string' || !/^[1-9]\d*$/.test(raw)) return RUNG_ALL;
  const c = Number(raw);
  return rungs.includes(c) ? c : RUNG_ALL;
}

function assertKnown(known) {
  for (const k of KNOWN_LISTS) {
    if (!Array.isArray(known?.[k])) throw new Error(`dashboard-link: known.${k} must be an array`);
  }
}

/**
 * @param {unknown} hash `location.hash`, with or without its leading `#`
 * @param {{tabIds:string[], subjectIds:string[], rungs:number[]}} known
 * @returns {{tab:string|null, subject:string|null, c:string|number}}
 */
export function parseDashboardHash(hash, known) {
  assertKnown(known);
  const params = new URLSearchParams(String(hash ?? '').replace(/^#/, ''));
  return {
    tab: resolveTab(params.get(KEY.tab), known.tabIds),
    subject: resolveSubject(params.get(KEY.subject), known.subjectIds),
    c: resolveRung(params.get(KEY.rung), known.rungs)
  };
}

export const isDeepLink = (link) => link.tab !== null;

/**
 * The hash for a dashboard state, without the `#`. No tab means no dashboard
 * to link to, so the result is empty; subject and rung are written only when
 * given, so a TTFT link does not carry a concurrency subject.
 */
export function formatDashboardHash({ tab, subject = null, c = null }) {
  if (tab === null || tab === undefined) return '';
  const params = new URLSearchParams();
  params.set(KEY.tab, tab);
  if (subject !== null && subject !== undefined) params.set(KEY.subject, subject);
  if (c !== null && c !== undefined) params.set(KEY.rung, String(c));
  return params.toString();
}
