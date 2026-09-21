// SPDX-License-Identifier: AGPL-3.0-only
//
// concurrency-subjects.js — the three subjects of the Concurrency tab.
//
// The list itself is concurrency-subjects.json BESIDE this file; this module
// only reads it. It sits under site/ deliberately: both of its readers -- this
// module and site/scripts/gen-ladder.mjs -- live here, and a repo-root path
// would make an otherwise web-only change draw the whole binary CI matrix,
// including a self-hosted Metal job that has nothing to do with a chart.
// Two subjects
// share a gate id (dense and MoE both run `concurrency-sweep`) and two share a
// checkpoint (dense and DFlash), so a record is assigned on BOTH fields — either
// alone would file a MoE run under the dense tab or a DFlash run under it.
import subjects from './concurrency-subjects.json';

const REQUIRED_STRINGS = ['id', 'label', 'checkpoint', 'gate', 'baselines_dir'];

/**
 * Refuse a malformed subject list at load time rather than rendering three
 * half-labelled tabs. Exported so the test can prove each rule bites.
 * @param {unknown} list
 * @returns {Array<{id:string,label:string,checkpoint:string,gate:string,published_manifest:string|null,baselines_dir:string}>}
 */
export function assertSubjects(list) {
  if (!Array.isArray(list) || list.length === 0) throw new Error('concurrency-subjects: expected a non-empty array');
  const seen = new Set();
  for (const s of list) {
    for (const k of REQUIRED_STRINGS) {
      if (typeof s?.[k] !== 'string' || s[k] === '')
        throw new Error(`concurrency-subjects: "${k}" must be a non-empty string (subject ${JSON.stringify(s?.id)})`);
    }
    if (s.published_manifest !== null && typeof s.published_manifest !== 'string')
      throw new Error(`concurrency-subjects: "published_manifest" must be a path or null (subject ${s.id})`);
    if (seen.has(s.id)) throw new Error(`concurrency-subjects: duplicate id ${s.id}`);
    seen.add(s.id);
  }
  return list;
}

export const SUBJECTS = assertSubjects(subjects);

/** The subject a gate record belongs to, or `null` when no subject claims it. */
export function subjectOf(record) {
  return SUBJECTS.find((s) => s.gate === record.benchmark_id && s.checkpoint === record.target_model) ?? null;
}

/**
 * A subject's records, in the order `recordsFor` yields them (chronological).
 * @param {{id:string,gate:string}} subject
 * @param {(benchId:string) => object[]} recordsFor
 */
export function recordsOf(subject, recordsFor) {
  return recordsFor(subject.gate).filter((r) => subjectOf(r)?.id === subject.id);
}

/**
 * `params.concurrencies` is the harness' own "1, 2, 4, 8, 16" string. A value
 * that is not a list of positive integers is a broken record, not an empty
 * rung set, so it is refused loudly and named by sha.
 */
export function parseConcurrencies(text, sha) {
  const parts = String(text ?? '').split(',').map((p) => p.trim());
  const rungs = parts.map(Number);
  if (parts.some((p) => p === '') || rungs.some((c) => !Number.isInteger(c) || c <= 0))
    throw new Error(`record ${sha}: params.concurrencies ${JSON.stringify(text)} is not a list of rungs`);
  return [...new Set(rungs)].sort((a, b) => a - b);
}

/**
 * The rungs the subject's gate currently declares — read from its NEWEST
 * record, because the dense gate widened from C<=16 to C<=128 on 2026-08-30
 * and an older record would still describe the narrow instrument. `[]` when
 * the subject has no records yet.
 */
export function rungsDeclared(subject, recordsFor) {
  const records = recordsOf(subject, recordsFor);
  if (records.length === 0) return [];
  const newest = records[records.length - 1];
  return parseConcurrencies(newest.params?.concurrencies, newest.git_sha);
}

/**
 * Records of the given benches that no subject claims, grouped for the
 * footer's `unassigned: <bench> · <model> (n)` line. Listed, never dropped:
 * a new checkpoint's runs must be visible somewhere until it gets a subject.
 * @returns {Array<{bench:string, model:string, n:number}>}
 */
export function unassignedRecords(benchIds, recordsFor) {
  const groups = new Map();
  for (const bench of benchIds) {
    for (const r of recordsFor(bench)) {
      if (subjectOf(r)) continue;
      const key = JSON.stringify([bench, r.target_model]);
      const entry = groups.get(key) ?? { bench, model: r.target_model, n: 0 };
      entry.n += 1;
      groups.set(key, entry);
    }
  }
  return [...groups.values()];
}

export const formatUnassigned = ({ bench, model, n }) => `${bench} · ${model} (${n})`;
