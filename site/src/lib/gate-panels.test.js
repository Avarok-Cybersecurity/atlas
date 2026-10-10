// SPDX-License-Identifier: AGPL-3.0-only
//
// panelsFor — the gate tab's chart list. It had NO test before this file, and
// the decode-floor branch is the one that now decides whether `stability` gets
// drawn at all, so the guard needs a control on both sides: a record that
// carries the key must produce the panel, and one that does not must produce
// exactly the panel list it produced before.
import { describe, expect, test } from 'bun:test';
import { panelsFor } from './gates.js';
import { buildSeries } from './gate-series.js';

/** A decode-floor record carrying whatever metrics the test needs. */
const rec = (metrics) => ({
  benchmark_id: 'decode-floor',
  git_sha: 'abc1234567',
  recorded_at: 1789900000,
  verdict: 'PASS',
  target_model: 'unsloth/Qwen3.8-27B-NVFP4',
  params: {},
  serve_overrides: {},
  metrics
});

// The shape the September records actually carry, values from 2a822a73b1.
const MEASURED = { server_decode_tok_s: 26.19, arrival_gap_cv: 0.0527, stability: 0.0423 };

describe('panelsFor · decode-floor', () => {
  test('a record carrying stability gets a second panel for it', () => {
    const panels = panelsFor('decode-floor', [rec(MEASURED)]);
    expect(panels).toHaveLength(2);
    expect(panels[1].metrics).toEqual([{ key: 'stability', label: 'stability' }]);
  });

  // ★ ABSENT IS NOT ZERO. `stability` landed with the campaign-3 metrics work;
  // every record older than that has none, and those tabs must render exactly
  // as they did before rather than growing an empty chart.
  test('a record WITHOUT stability gets only the decode-floor panel', () => {
    const { server_decode_tok_s } = MEASURED;
    const panels = panelsFor('decode-floor', [rec({ server_decode_tok_s })]);
    expect(panels).toHaveLength(1);
    expect(panels[0].title).toBe('decode floor');
  });

  // ★ ITS OWN AXIS. stability is a tail spread (~0.04, lower = smoother) and
  // tok/s is ~26. On one axis the stability line is pinned to the bottom and
  // reads as "nothing happening" rather than as a different quantity.
  test('stability is a separate panel with its own unit, never merged into tok/s', () => {
    const [decode, stability] = panelsFor('decode-floor', [rec(MEASURED)]);
    expect(decode.unit).toBe('tok/s');
    expect(stability.unit).not.toBe(decode.unit);
    expect(decode.metrics.map((m) => m.key)).not.toContain('stability');
  });

  test('the newest record decides, so a tab does not lose the panel to old history', () => {
    const { server_decode_tok_s } = MEASURED;
    const panels = panelsFor('decode-floor', [rec({ server_decode_tok_s }), rec(MEASURED)]);
    expect(panels).toHaveLength(2);
  });
});

// ---- concurrency-sweep · ITL p90, client beside server (#1218) ---------------
// The client's SSE-derived ITL and the server's own count are independent
// measurements of one quantity, and the panel exists so a gap between them can
// be seen. Two curves that coincide are indistinguishable from one curve drawn
// twice, so the control is a fixture where they genuinely diverge: it must come
// out as two series with different values and a different stroke.
const sweep = (bench, metrics, at = 1789900000) => ({
  benchmark_id: bench,
  git_sha: 'f314604aca',
  recorded_at: at,
  verdict: 'PASS',
  target_model: 'unsloth/Qwen3.8-27B-NVFP4',
  params: {},
  serve_overrides: {},
  metrics
});

// Client values from f314604aca; the c128 server value is moved off the
// client's on purpose (the real records agree to five significant figures).
const ITL = {
  peak_aggregate_tok_s: 544.2,
  c16_tpot_p90_ms: 75.55,
  c16_server_tpot_p90_ms: 75.55,
  c128_tpot_p90_ms: 235.31,
  c128_server_tpot_p90_ms: 201.0
};

describe('panelsFor · concurrency-sweep ITL', () => {
  test('client and server ITL share one panel at the widest rung, and diverging values draw two lines', () => {
    const records = [sweep('concurrency-sweep', ITL), sweep('concurrency-sweep', ITL, 1789986400)];
    const itl = panelsFor('concurrency-sweep', records).find((p) => p.title.startsWith('ITL'));
    expect(itl.title).toBe('ITL p90 at c128 (no drafter)');
    expect(itl.unit).toBe('ms');
    const [client, server] = buildSeries(itl, records);
    expect(client.dashed).toBe(false);
    expect(server.dashed).toBe(true);
    expect(client.nodes.map((n) => n.v)).toEqual([235.31, 235.31]);
    expect(server.nodes.map((n) => n.v)).toEqual([201.0, 201.0]);
  });

  // ★ ABSENT IS NOT ZERO. Records older than the server-ITL work carry no
  // tpot keys at all; they must render exactly the panels they rendered before.
  test('records without tpot keys get no ITL panel', () => {
    const panels = panelsFor('concurrency-sweep', [sweep('concurrency-sweep', { peak_aggregate_tok_s: 544.2 })]);
    expect(panels.some((p) => p.title.startsWith('ITL'))).toBe(false);
  });

  // ★ splitByVariant applies, as for the peak panel: a DFlash2 point joined to
  // a no-drafter point reads as a regression and a recovery at every alternation.
  test('each variant gets its own ITL panel, scoped to its own records', () => {
    const records = [sweep('concurrency-sweep', ITL), sweep('concurrency-sweep-dflash2', ITL, 1789900001)];
    const itl = panelsFor('concurrency-sweep', records).filter((p) => p.title.startsWith('ITL'));
    expect(itl.map((p) => p.title)).toEqual(['ITL p90 at c128 (no drafter)', 'ITL p90 at c128 (DFlash2)']);
    expect(itl.map((p) => p.metrics.map((m) => m.variant))).toEqual([
      ['concurrency-sweep', 'concurrency-sweep'],
      ['concurrency-sweep-dflash2', 'concurrency-sweep-dflash2']
    ]);
  });
});
