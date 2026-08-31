// =============================================================================
// gates.js — taxonomy + read helpers for the benchmark dashboard.
// Data comes exclusively from gates.generated.json (see scripts/gen-gates.mjs).
// This module is the SSOT for how benchmarks map to tabs, panels and colors —
// components render what these specs say and add nothing of their own.
// =============================================================================
import gates from '$lib/gates.generated.json';
import { splitByVariant } from './gate-variants.js';

export const gateData = gates;
export const GH_COMMIT = 'https://github.com/Avarok-Cybersecurity/atlas/commit/';

export { MODEL_COLORS, UNKNOWN_MODEL_COLOR, colorFor } from './series-colors.js';
export {
  dashFor,
  groupFor,
  groupRecords,
  groupedBenches,
  isLatestOfVariant,
  splitByVariant,
  variantLabel
} from './gate-variants.js';

export { shortModel } from './series-colors.js';

// ---- tab taxonomy -----------------------------------------------------------
// One tab per benchmark family; a family only earns a tab when it has records.
// ttft warm+cold share a tab (same metric, same model, two conditions).
// The two BFCL draws stay SEPARATE panels: different models AND different
// sample draws — overlaying them on one axis would let a 27B number read as a
// 35B one, or one draw's score read as comparable to another's.
const TAB_DEFS = [
  { id: 'agentic', label: 'Agentic', benches: ['agentic-webserver'] },
  { id: 'bfcl', label: 'BFCL', benches: ['bfcl-subset', 'bfcl-subset-echolp'] },
  { id: 'ttft', label: 'TTFT', benches: ['ttft-warm-gate', 'ttft-cold-gate'] },
  // Wired ahead of data: records for these land only after calibration on
  // the fixed instrument (2026-08-15 concurrency re-scope). Until then the
  // records-filter below keeps the tabs hidden and the ids show in the
  // footer's "gated, not yet published" line — nothing renders empty.
  { id: 'decode', label: 'Decode', benches: ['decode-floor'] },
  // Both concurrency gates share this tab AND one set of charts — see
  // gate-variants.js. They run the same fixture at the same rungs on the
  // same checkpoint and differ only in whether the engine speculates, so
  // two lines on one axis is the comparison; two panels is not.
  {
    id: 'concurrency',
    label: 'Concurrency',
    benches: ['concurrency-sweep', 'concurrency-sweep-dflash2']
  }
];
export const tabs = TAB_DEFS.filter((t) =>
  t.benches.some((b) => (gates.benchmarks[b]?.records ?? []).length > 0)
);

// Registered in the suite (descriptor SSOT) but with zero published records —
// named honestly in the footer instead of rendering empty tabs.
const withRecords = new Set(Object.keys(gates.benchmarks));
export const unpublished = (gates.registered ?? []).filter((id) => !withRecords.has(id));

export const models = [...new Set(Object.values(gates.benchmarks).flatMap((b) => b.records.map((r) => r.target_model)))].sort();

// ---- panel specs ------------------------------------------------------------
// floor/cap lines are read from the records themselves (params or the
// verdict_reason's "(floor N)" text) — never invented here.
const floorFromReason = (r) => {
  const m = /floor ([0-9.]+)/.exec(r.verdict_reason ?? '');
  return m ? +m[1] : null;
};

export function panelsFor(benchId, records) {
  if (records.length === 0) return [];
  const latest = records[records.length - 1];
  if (benchId === 'agentic-webserver') {
    return [
      {
        title: 'Σ wall time',
        unit: 's',
        metrics: [{ key: 'sum_wall_s', label: 'Σ wall (s)' }],
        caps: [...new Set(records.map((r) => +r.params?.wall_budget_s || 0).filter(Boolean))].map((v) => ({
          value: v,
          label: `budget ${v}s`
        }))
      },
      {
        title: 'webserver_ok per run',
        unit: `/ ${latest.metrics?.iterations ?? 10} iterations`,
        metrics: [{ key: 'webserver_ok', label: 'webserver_ok' }],
        domain: [0, latest.metrics?.iterations ?? 10]
      }
    ];
  }
  if (benchId.startsWith('bfcl')) {
    return [
      {
        title: 'overall accuracy',
        unit: 'score',
        metrics: [{ key: 'overall_accuracy', label: 'overall' }],
        caps: [],
        floors: [...new Set(records.map(floorFromReason).filter(Boolean))].map((v) => ({
          value: v,
          label: `floor ${v}`
        }))
      }
    ];
  }
  if (benchId.startsWith('ttft')) {
    return [
      {
        title: benchId === 'ttft-warm-gate' ? 'warm TTFT' : 'cold TTFT',
        unit: 'ms',
        metrics: [
          { key: 'median_ms', label: 'median' },
          { key: 'p90_ms', label: 'p90', dashed: true }
        ]
      }
    ];
  }
  if (benchId === 'decode-floor') {
    // Single-value trend. The key is read from the records rather than
    // pinned here: thresholds/records land after calibration, and wiring
    // that invents a name the producer never emits would render an empty
    // chart forever. Prefer a tok/s-shaped key, else the first numeric one.
    const keys = Object.keys(latest.metrics ?? {});
    const key = keys.find((k) => /tok_s/.test(k)) ?? keys.find((k) => k !== 'samples');
    return key ? [{ title: 'decode floor', unit: 'tok/s', metrics: [{ key, label: key }] }] : [];
  }
  if (benchId === 'concurrency-sweep' || benchId === 'concurrency-sweep-dflash2') {
    // Two panels: the ladder curve (throughput vs C, latest runs overlaid —
    // rendered by GateLadderChart via kind: 'ladder') and the peak's trend
    // over time. Keys come from the sweep's metrics map
    // (c{C}_aggregate_tok_s / peak_aggregate_tok_s); vacuous cells were
    // already excluded by the producer, so every point here is comparable.
    const panels = [];
    if (records.some((r) => Object.keys(r.metrics ?? {}).some((k) => LADDER_KEY.test(k)))) {
      panels.push({ kind: 'ladder', title: 'throughput vs concurrency', unit: 'tok/s' });
    }
    // One series per variant present, never one series across both: a line
    // that joined a DFlash2 peak to a no-drafter peak would read as a
    // regression and a recovery at every alternation. `variant` filters the
    // records inside GateChart; the dash is the only other difference,
    // because colour follows the model and both variants serve one
    // checkpoint.
    const peak = splitByVariant(records)
      .filter((v) => v.records.some((r) => Number.isFinite(r.metrics?.peak_aggregate_tok_s)))
      .map((v) => ({
        key: 'peak_aggregate_tok_s',
        label: v.label ? `peak (${v.label})` : 'peak',
        variant: v.bench,
        dashed: v.dash !== null
      }));
    if (peak.length > 0) {
      panels.push({ title: 'peak aggregate throughput', unit: 'tok/s', metrics: peak });
    }
    return panels;
  }
  // Unknown future benchmark: chart its first numeric metric so new suites
  // appear without a code change.
  const key = Object.keys(latest.metrics ?? {}).find((k) => k !== 'samples');
  return key ? [{ title: key, unit: '', metrics: [{ key, label: key }] }] : [];
}

// One rung of the concurrency ladder: c{C}_aggregate_tok_s from the sweep's
// metrics map. Shared with GateLadderChart so the panel test and the renderer
// cannot drift apart on the key shape.
export const LADDER_KEY = /^c(\d+)_aggregate_tok_s$/;
export const ladderPoints = (record) =>
  Object.entries(record?.metrics ?? {})
    .map(([k, v]) => {
      const m = LADDER_KEY.exec(k);
      return m ? { c: +m[1], v } : null;
    })
    .filter(Boolean)
    .sort((a, b) => a.c - b.c);

export const recordsFor = (benchId) => gates.benchmarks[benchId]?.records ?? [];
export const benchName = (benchId) => gates.benchmarks[benchId]?.name ?? benchId;
export const fmtDate = (unix) => new Date(unix * 1000).toISOString().slice(0, 10);
export const fmtDateTime = (unix) => new Date(unix * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
export const sampleCount = (r) => r.metrics?.samples ?? r.metrics?.iterations ?? null;
