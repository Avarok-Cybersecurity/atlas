// SPDX-License-Identifier: AGPL-3.0-only
//
// panelsFor — the gate tab's chart list. It had NO test before this file, and
// the decode-floor branch is the one that now decides whether `stability` gets
// drawn at all, so the guard needs a control on both sides: a record that
// carries the key must produce the panel, and one that does not must produce
// exactly the panel list it produced before.
import { describe, expect, test } from 'bun:test';
import { panelsFor } from './gates.js';

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
