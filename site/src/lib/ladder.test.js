// SPDX-License-Identifier: AGPL-3.0-only

import { test, expect } from 'bun:test';
import { headroom, signed } from './ladder.js';
import real from './ladder.generated.json';

const rungs = (...rows) => rows;

test('scales the top two rungs against the lower rung leader', () => {
  const h = headroom(
    rungs(
      { c: 64, avarok: 100, best_baseline_id: 'x', baselines: [{ id: 'x', label: 'X', tok_s: 100 }] },
      { c: 128, avarok: 150, best_baseline_id: 'x', baselines: [{ id: 'x', label: 'X', tok_s: 90 }] }
    )
  );
  expect(h.from).toBe(64);
  expect(h.to).toBe(128);
  expect(h.avarok).toBe(50);
  expect(h.baseline).toBe(-10);
});

test('follows the lower rung leader even when another config leads the top', () => {
  // The real shape: vLLM+MTP leads at C=64, then falls behind vLLM-no-spec at
  // C=128. Reporting "best at each rung" would compare two different
  // deployments and call it a scaling curve.
  const h = headroom(
    rungs(
      {
        c: 64,
        avarok: 100,
        best_baseline_id: 'mtp',
        baselines: [
          { id: 'mtp', label: 'vLLM + MTP', tok_s: 90 },
          { id: 'nospec', label: 'vLLM, no speculation', tok_s: 80 }
        ]
      },
      {
        c: 128,
        avarok: 150,
        best_baseline_id: 'nospec',
        baselines: [
          { id: 'mtp', label: 'vLLM + MTP', tok_s: 88 },
          { id: 'nospec', label: 'vLLM, no speculation', tok_s: 120 }
        ]
      }
    )
  );
  expect(h.label).toBe('vLLM + MTP');
  expect(h.baseline).toBeCloseTo(-2.2, 1);
});

test('refuses to report when the leader was not run at the top rung', () => {
  expect(
    headroom(
      rungs(
        { c: 64, avarok: 100, best_baseline_id: 'x', baselines: [{ id: 'x', label: 'X', tok_s: 100 }] },
        { c: 128, avarok: 150, best_baseline_id: 'y', baselines: [{ id: 'y', label: 'Y', tok_s: 90 }] }
      )
    )
  ).toBeNull();
});

test('needs two rungs', () => {
  expect(headroom([])).toBeNull();
  expect(headroom([{ c: 1, avarok: 1, best_baseline_id: 'x', baselines: [] }])).toBeNull();
  expect(headroom(null)).toBeNull();
});

test('signs percentages so the direction survives the sentence', () => {
  expect(signed(23.7)).toBe('+23.7%');
  expect(signed(-0.8)).toBe('-0.8%');
  expect(signed(0)).toBe('+0.0%');
});

test('the shipped ladder still supports the claim the copy makes', () => {
  // If a regenerated ladder ever stops showing Avarok climbing while the
  // baseline flattens, the prose in Verified.svelte becomes false. Fail here
  // rather than on the live page.
  const h = headroom(real.rows ?? []);
  expect(h).not.toBeNull();
  expect(h.avarok).toBeGreaterThan(0);
  expect(h.avarok).toBeGreaterThan(h.baseline);
});

test('C=128 is 1.333× against matched MTP, not 1.225× against unmatched nospec', () => {
  // Live /engine used to print 1.225× because the headline denominator was
  // "whichever vLLM is faster at this rung". At C=128 that is nospec (390.42),
  // a different fingerprint. The published claim is 478.11 / 358.57 = 1.333×.
  const row = (real.rows ?? []).find((r) => r.c === 128);
  expect(row).toBeDefined();
  expect(row.best_baseline_id).toBe('vllm-mtp');
  expect(row.avarok).toBe(478.11);
  expect(row.baselines.find((b) => b.id === 'vllm-mtp').tok_s).toBe(358.57);
  expect(row.ratio_vs_best).toBe(1.333);
  expect(real.summary.max_ratio).toBe(1.333);
  expect(row.ratio_vs_fastest).toBe(1.225);
});
