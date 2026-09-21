// SPDX-License-Identifier: AGPL-3.0-only
//
// The comparison decisions, pure: which state a subject tab is in, which
// one-shot baselines may be drawn against a gate record, and what the tile
// beside "vLLM baseline" says. The rendered checks live in
// concurrency-tab.test.js; these prove the rules on inputs the call sites
// produce, including the refusal that keeps a ladder-instrument vLLM leg off
// a gate-instrument axis.
import { describe, expect, test } from 'bun:test';
import ladders from './ladders.generated.json';
import subjects from './concurrency-subjects.json';
import {
  absentReasonOf,
  baselineOnlyFor,
  baselineTileOf,
  comparisonStateOf,
  fingerprintOf,
  ladderFor,
  oneShotChip,
  pairWith,
  publishedFor
} from './concurrency-comparison.js';

const byId = (id) => subjects.find((s) => s.id === id);
const DENSE = byId('qwen38-27b');
const MOE = byId('qwen36-35b-a3b');
const DFLASH = byId('qwen38-27b-dflash');
const moe = ladders.subjects[MOE.id];
const vllm = moe.series.find((s) => s.id === 'vllm-mtp');

// A passing main-branch gate record on the gate instrument, string-valued as
// the harness records it.
const gate = (over = {}) => ({
  benchmark_id: MOE.gate,
  target_model: MOE.checkpoint,
  git_sha: 'feedfacede',
  recorded_at: 1_800_000_000,
  verdict: 'PASS',
  branch: '',
  params: { concurrencies: '1, 2, 4', isls: '512', osl: '320', prompt_mode: 'natural' },
  serve_overrides: { max_batch_size: '128', kv_cache_dtype: 'fp8', max_model_len: '4096' },
  metrics: { c1_aggregate_tok_s: 10, c2_aggregate_tok_s: 20, c4_aggregate_tok_s: 40 },
  ...over
});
const ON_GATE = { isl: 512, osl: 320, prompt_mode: 'natural', max_model_len: 4096, max_batch_size: 128, kv_cache_dtype: 'fp8' };
const withInstrument = (instrument) => ({ ...moe, series: [{ ...vllm, instrument }] });
const laddersWith = (ladder) => ({ subjects: { ...ladders.subjects, [MOE.id]: ladder } });

describe('which ladder a subject gets', () => {
  test('its own, only for its checkpoint; none when it declares no manifest', () => {
    expect(ladderFor(MOE, ladders)).toBe(moe);
    expect(ladderFor(DFLASH, ladders)).toBeNull();
    expect(ladderFor({ ...MOE, checkpoint: DENSE.checkpoint }, ladders)).toBeNull();
    expect(ladderFor({ ...MOE, published_manifest: null }, ladders)).toBeNull();
    expect(ladderFor({ ...MOE, id: 'other' }, ladders)).toBeNull();
  });

  test('a pair and a baseline-only ladder are told apart by the subject series, not by name', () => {
    expect(publishedFor(DENSE, ladders)).toBe(ladders.subjects[DENSE.id]);
    expect(baselineOnlyFor(DENSE, ladders)).toBeNull();
    expect(publishedFor(MOE, ladders)).toBeNull();
    expect(baselineOnlyFor(MOE, ladders)).toBe(moe);
    const promoted = { ...moe, series: [...moe.series, { id: 'atlas', role: 'subject', rungs: [] }] };
    expect(publishedFor(MOE, laddersWith(promoted))).toBe(promoted);
    expect(baselineOnlyFor(MOE, laddersWith(promoted))).toBeNull();
  });
});

describe('the state, in precedence order', () => {
  test('published > live > baseline > none', () => {
    expect(comparisonStateOf(DENSE, [gate({ target_model: DENSE.checkpoint })], ladders)).toBe('published');
    expect(comparisonStateOf(MOE, [], ladders)).toBe('baseline');
    expect(comparisonStateOf(MOE, [gate()], ladders)).toBe('live');
    expect(comparisonStateOf(MOE, [gate({ verdict: 'FAIL' })], ladders)).toBe('baseline');
    expect(comparisonStateOf(MOE, [gate({ branch: 'pr/x' })], ladders)).toBe('baseline');
    expect(comparisonStateOf(DFLASH, [], ladders)).toBe('none');
    expect(comparisonStateOf({ ...MOE, published_manifest: null }, [], ladders)).toBe('none');
  });
});

describe('pairing a one-shot with a gate record', () => {
  test('THE REFUSAL: the MoE one-shot is on the ladder instrument and is never drawn against a gate record', () => {
    const { drawn, refused } = pairWith(gate(), moe);
    expect(drawn).toEqual([]);
    expect(refused).toHaveLength(1);
    expect(refused[0].series).toBe(vllm);
    expect(refused[0].differs).toEqual([
      { axis: 'isl', a: '512', b: '128' },
      { axis: 'osl', a: '320', b: '1024' },
      { axis: 'prompt_mode', a: 'natural', b: 'essay' },
      { axis: 'max_model_len', a: '4096', b: '2048' },
      { axis: 'kv_cache_dtype', a: 'fp8', b: 'bf16' }
    ]);
    expect(refused[0].why).toBe(
      'isl 512 → 128, osl 320 → 1024, prompt_mode natural → essay, max_model_len 4096 → 2048, kv_cache_dtype fp8 → bf16'
    );
  });

  test('POSITIVE CONTROL: a one-shot fingerprinted on the gate axes is drawn, one-sided axes ignored', () => {
    const { drawn, refused } = pairWith(gate(), withInstrument(ON_GATE));
    expect(refused).toEqual([]);
    expect(drawn.map((b) => b.id)).toEqual(['vllm-mtp']);
    // A gate record axis the baseline never declared (ssm_cache_slots) is no mismatch.
    const withSlots = gate({ serve_overrides: { max_batch_size: '128', kv_cache_dtype: 'fp8', max_model_len: '4096', ssm_cache_slots: '32' } });
    expect(pairWith(withSlots, withInstrument(ON_GATE)).drawn).toHaveLength(1);
  });

  test('one axis off is a refusal that names that axis', () => {
    const { drawn, refused } = pairWith(gate(), withInstrument({ ...ON_GATE, kv_cache_dtype: 'bf16' }));
    expect(drawn).toEqual([]);
    expect(refused[0].why).toBe('kv_cache_dtype fp8 → bf16');
  });

  test('the checkpoint is part of the fingerprint, read from the ladder', () => {
    expect(fingerprintOf(moe, vllm)).toEqual({ checkpoint: MOE.checkpoint, instrument: vllm.instrument });
    const other = pairWith(gate({ target_model: DENSE.checkpoint }), withInstrument(ON_GATE));
    expect(other.drawn).toEqual([]);
    expect(other.refused[0].why).toBe(`checkpoint ${DENSE.checkpoint} → ${MOE.checkpoint}`);
  });
});

describe('the tile follows the chart', () => {
  test('pair, dated one-shot, other instrument, none', () => {
    expect(baselineTileOf(DENSE, [], ladders)).toBe('published pair');
    expect(baselineTileOf(MOE, [], ladders)).toBe('one-shot · 2026-09-19');
    expect(baselineTileOf(MOE, [gate()], ladders)).toBe('other instrument');
    expect(baselineTileOf(MOE, [gate()], laddersWith(withInstrument(ON_GATE)))).toBe('one-shot · 2026-09-19');
    expect(baselineTileOf(DFLASH, [], ladders)).toBe('none');
    expect(baselineTileOf(DFLASH, [gate({ benchmark_id: DFLASH.gate, target_model: DFLASH.checkpoint })], ladders)).toBe('none');
  });
});

describe('what the page says about a rung with no point', () => {
  test('the recorded reason for a listed rung; the plain fact for any other', () => {
    expect(absentReasonOf(moe, 64)).toBe(`C=64 · not measured — ${vllm.unmeasured.reason}`);
    expect(absentReasonOf(moe, 3)).toBe('C=3 · not in this manifest: neither measured nor listed as unmeasured');
    const noList = withInstrument(vllm.instrument);
    delete noList.series[0].unmeasured;
    expect(absentReasonOf(noList, 64)).toBe('C=64 · not in this manifest: neither measured nor listed as unmeasured');
  });

  test('the one-shot chip is dated from the rungs', () => {
    expect(oneShotChip(vllm)).toBe('vLLM + MTP · one-shot · measured 2026-09-19 · not re-run');
    const twoDays = { ...vllm, rungs: [{ measured_utc: '2026-09-19T22:25:43Z' }, { measured_utc: '2026-09-20T01:00:00Z' }] };
    expect(oneShotChip(twoDays)).toBe('vLLM + MTP · one-shot · measured 2026-09-19 → 2026-09-20 · not re-run');
  });
});
