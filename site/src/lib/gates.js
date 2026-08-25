// =============================================================================
// gates.js — taxonomy + read helpers for the benchmark dashboard.
// Data comes exclusively from gates.generated.json (see scripts/gen-gates.mjs).
// This module is the SSOT for how benchmarks map to tabs, panels and colors —
// components render what these specs say and add nothing of their own.
// =============================================================================
import gates from '$lib/gates.generated.json';

export const gateData = gates;
export const GH_COMMIT = 'https://github.com/Avarok-Cybersecurity/atlas/commit/';

// Series color follows the MODEL (the entity), never the tab or verdict.
//
// Re-derived 2026-08-25 when the canvas moved from paper to the deep-violet
// dark theme. The previous trio (copper #b5622f, steel #1f6a9e, teal #1c7a6b)
// was validated against #f4f0e8/#fbf9f3 and does not survive the inversion:
// steel falls to 2.87:1 on the card surface, under the >=3:1 floor this palette
// has always held to. The old comment asked for all three to be re-derived
// together if the palette were ever revisited, so they were.
//
// Same three hue families, lifted for a near-black canvas. Measured with
// CIEDE2000 under Vienot dichromat simulation, against #14111f and #201b30.
// The method reproduces the previous comment's figures exactly on the old
// surfaces (43.8 / 16.3 / 27.1, 26.1 / 27.5 / 20.3), so these numbers ARE
// continuous with the 2026-08-14 set and can be compared directly:
//   copper #ee6f2f  6.15:1 / 5.51:1
//   steel  #2f88ee  5.20:1 / 4.66:1
//   teal   #51cdb0  9.48:1 / 8.49:1
// pairwise, normal / protan / deutan:
//   copper vs steel  49.6 / 60.2 / 68.3
//   copper vs teal   55.3 / 25.0 / 26.4
//   steel  vs teal   39.9 / 42.4 / 32.6   <- the load-bearing comparison
// Worst case 25.0, against 16.3 before; the steel-teal pair, which is the whole
// point of the series (3.8 vs 3.6-27B: same architecture, same draw, read as a
// generation-over-generation delta), improves from 20.3 to 32.6.
//
// Lightness was capped during the search. An unconstrained optimum scored 31.4
// but put teal at 14.4:1 — a near-white cyan that no longer reads as teal, and
// glares on a dark canvas. Separation is not worth spending the hue identity on.
const MODEL_COLORS = {
  'Qwen/Qwen3.6-35B-A3B-FP8': '#ee6f2f',
  'unsloth/Qwen3.6-27B-NVFP4': '#2f88ee',
  'unsloth/Qwen3.8-27B-NVFP4': '#51cdb0'
};
export const colorFor = (model) => MODEL_COLORS[model] ?? '#6f6a8d';
export const shortModel = (model) => (model || '').split('/').pop() || model;

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
  { id: 'concurrency', label: 'Concurrency', benches: ['concurrency-sweep'] }
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
  if (benchId === 'concurrency-sweep') {
    // Two panels: the ladder curve (throughput vs C, latest runs overlaid —
    // rendered by GateLadderChart via kind: 'ladder') and the peak's trend
    // over time. Keys come from the sweep's metrics map
    // (c{C}_aggregate_tok_s / peak_aggregate_tok_s); vacuous cells were
    // already excluded by the producer, so every point here is comparable.
    const panels = [];
    if (records.some((r) => Object.keys(r.metrics ?? {}).some((k) => LADDER_KEY.test(k)))) {
      panels.push({ kind: 'ladder', title: 'throughput vs concurrency', unit: 'tok/s' });
    }
    if (records.some((r) => Number.isFinite(r.metrics?.peak_aggregate_tok_s))) {
      panels.push({
        title: 'peak aggregate throughput',
        unit: 'tok/s',
        metrics: [{ key: 'peak_aggregate_tok_s', label: 'peak' }]
      });
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
