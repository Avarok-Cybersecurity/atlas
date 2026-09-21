// SPDX-License-Identifier: AGPL-3.0-only
//
// The rule behind the legend pills on the concurrency comparison: press to
// hide, press again to show, and the last drawn series cannot be hidden —
// that press is refused with a reason the page prints. The pills once did
// nothing at all; a rule that lives here is one the component cannot drift
// from.
import { describe, expect, test } from 'bun:test';
import { toggleSeries, visibleOf } from './series-visibility.js';

const SERIES = [
  { id: 'atlas', label: 'Atlas' },
  { id: 'vllm-mtp', label: 'vLLM + MTP' },
  { id: 'vllm-nospec', label: 'vLLM, no speculation' }
];

describe('toggleSeries', () => {
  test('a press hides, a second press shows, order of the rest untouched', () => {
    const off = toggleSeries(SERIES, [], 'vllm-mtp');
    expect(off).toEqual({ hidden: ['vllm-mtp'], refused: null });
    expect(visibleOf(SERIES, off.hidden).map((s) => s.id)).toEqual(['atlas', 'vllm-nospec']);
    const on = toggleSeries(SERIES, off.hidden, 'vllm-mtp');
    expect(on).toEqual({ hidden: [], refused: null });
  });

  test('the input array is never mutated', () => {
    const hidden = ['atlas'];
    toggleSeries(SERIES, hidden, 'vllm-mtp');
    toggleSeries(SERIES, hidden, 'atlas');
    expect(hidden).toEqual(['atlas']);
  });

  test('the subject can be hidden like any other series while another is drawn', () => {
    expect(toggleSeries(SERIES, [], 'atlas')).toEqual({ hidden: ['atlas'], refused: null });
  });

  test('hiding the last drawn series is refused, named, and leaves the set unchanged', () => {
    const hidden = ['atlas', 'vllm-mtp'];
    const r = toggleSeries(SERIES, hidden, 'vllm-nospec');
    expect(r.hidden).toBe(hidden);
    expect(r.refused).toBe(
      'vLLM, no speculation stays drawn — it is the only series left. Show another series before hiding it.'
    );
    expect(visibleOf(SERIES, r.hidden)).toHaveLength(1);
  });

  test('a refusal clears on the next accepted press', () => {
    const refused = toggleSeries(SERIES, ['atlas', 'vllm-mtp'], 'vllm-nospec');
    const shown = toggleSeries(SERIES, refused.hidden, 'atlas');
    expect(shown).toEqual({ hidden: ['vllm-mtp'], refused: null });
  });

  test('an unknown id is a wiring bug and throws', () => {
    expect(() => toggleSeries(SERIES, [], 'vllm')).toThrow('unknown series "vllm"');
  });
});
