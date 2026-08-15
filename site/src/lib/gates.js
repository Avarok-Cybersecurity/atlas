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
// Pair validated for CVD + contrast on the paper surfaces (#f4f0e8/#fbf9f3):
// copper #b5622f ↔ steel #1f6a9e, protan ΔE 17.6, normal ΔE 24.5, both ≥3:1.
//
// Qwen3.8-27B added 2026-08-14 as teal #1c7a6b. The pair above was a PAIR —
// a third entry has to hold against BOTH, and the load-bearing comparison is
// 3.8 vs 3.6-27B (steel): same architecture, same draw, so they overlay on one
// axis and the whole point is reading the generation-over-generation delta.
// Measured under CIEDE2000 with Viénot CVD simulation (NOT the method behind
// the 17.6/24.5 figures above, so these numbers are not continuous with them —
// re-derive all three together if the palette is ever revisited):
//   vs steel  #1f6a9e — normal 26.1, protan 27.5, deutan 20.3
//   vs copper #b5622f — normal 43.8, protan 16.3, deutan 27.1
//   contrast on both paper surfaces — 4.57:1 (≥3:1)
// Plum #7d3c6e was rejected: only ΔE 12.1 from steel under both protan and
// deutan, i.e. it blurred exactly the comparison this series exists to show.
// Maroon #8f2f3f scored marginally better on worst case (19.1) but reads as a
// failure color on a chart that already marks failed runs by verdict.
const MODEL_COLORS = {
  'Qwen/Qwen3.6-35B-A3B-FP8': '#b5622f',
  'unsloth/Qwen3.6-27B-NVFP4': '#1f6a9e',
  'unsloth/Qwen3.8-27B-NVFP4': '#1c7a6b'
};
export const colorFor = (model) => MODEL_COLORS[model] ?? '#625c51';
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
  { id: 'ttft', label: 'TTFT', benches: ['ttft-warm-gate', 'ttft-cold-gate'] }
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
  // Unknown future benchmark: chart its first numeric metric so new suites
  // appear without a code change.
  const key = Object.keys(latest.metrics ?? {}).find((k) => k !== 'samples');
  return key ? [{ title: key, unit: '', metrics: [{ key, label: key }] }] : [];
}

export const recordsFor = (benchId) => gates.benchmarks[benchId]?.records ?? [];
export const benchName = (benchId) => gates.benchmarks[benchId]?.name ?? benchId;
export const fmtDate = (unix) => new Date(unix * 1000).toISOString().slice(0, 10);
export const fmtDateTime = (unix) => new Date(unix * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
export const sampleCount = (r) => r.metrics?.samples ?? r.metrics?.iterations ?? null;
