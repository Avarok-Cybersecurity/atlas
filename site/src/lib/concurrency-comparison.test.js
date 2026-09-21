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
  test('live-when-it-pairs > published > live > baseline > none', () => {
    // DENSE has a published pair AND this record is on the RETIRED gate
    // instrument, so nothing pairs and the frozen pair is still the best
    // thing to draw. The case where it DOES pair is the describe below.
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

// ── the live Atlas leg outranks the frozen published pair ───────────────────
//
// Ask, 2026-09-21: "the first chart shows atlas vs vllm; we MUST make that
// graph show the latest values of Atlas ... the bottom graph needs to be the
// same test as the top one". Two changes make that true and BOTH are load-
// bearing, so both are controlled here:
//   * kernels/gb10/qwen3.8-27b/BENCH.toml re-points the gate to the published
//     ladder's instrument (isl 128 / osl 1024 / essay / ctx 2048), and
//     bench/ladder38/published.json declares the prompt_mode those vLLM legs
//     ran, without which a REQUIRED axis is undeclared and nothing can pair;
//   * comparisonStateOf prefers a live record that pairs over the snapshot.
// Remove either and these tests go red — proved by mutation, not assumed.
describe('a live record that pairs outranks the published pair', () => {
  const dense = ladders.subjects[DENSE.id];
  // The instrument BENCH.toml pins after the re-point, as a record carries it
  // (strings; threshold params present because a real record carries them and
  // THRESHOLD_PARAM must keep excluding them from the fingerprint).
  const repointed = (over = {}) => ({
    ...gate({ target_model: DENSE.checkpoint }),
    params: {
      concurrencies: '1, 2, 4, 8, 16, 32, 64, 128',
      isls: '128',
      osl: '1024',
      prompt_mode: 'essay',
      warmup: '1',
      min_c1: '0',
      min_peak: '0'
    },
    serve_overrides: {
      max_batch_size: '128',
      kv_cache_dtype: 'fp8',
      ssm_cache_slots: '32',
      max_model_len: '2048'
    },
    ...over
  });

  test('THE ASK: a record on the published instrument makes the dense tab live, drawn against vllm-mtp', () => {
    expect(comparisonStateOf(DENSE, [repointed()], ladders)).toBe('live');
    const { drawn, refused } = pairWith(repointed(), dense);
    expect(drawn.map((d) => d.id)).toEqual(['vllm-mtp']);
    // The no-speculation leg is NOT quietly folded in: it is a different
    // context and a different KV dtype, and it says so.
    expect(refused.map((r) => r.series.id)).toEqual(['vllm-nospec']);
    expect(refused[0].why).toBe('max_model_len 2048 → 4096, kv_cache_dtype fp8 → bf16');
    // and the tile stops saying "published pair" and dates the leg it draws
    expect(baselineTileOf(DENSE, [repointed()], ladders)).toBe('one-shot · 2026-08-17 → 2026-08-18');
    expect(baselineTileOf(DENSE, [gate({ target_model: DENSE.checkpoint })], ladders)).toBe('published pair');
  });

  test('the fallback is not a formality: each axis alone sends it back to the snapshot', () => {
    // One axis at a time, each the value the retired instrument had. Every one
    // of these is a real mutation of the BENCH.toml change, and every one must
    // cost the live series — otherwise "same test top and bottom" is unproved.
    const off = (params, serve) =>
      comparisonStateOf(DENSE, [repointed({ params: { ...repointed().params, ...params }, serve_overrides: { ...repointed().serve_overrides, ...serve } })], ladders);
    expect(off({ isls: '512' }, {})).toBe('published');
    expect(off({ osl: '320' }, {})).toBe('published');
    expect(off({ prompt_mode: 'natural' }, {})).toBe('published');
    expect(off({}, { max_model_len: '4096' })).toBe('published');
    expect(off({}, { max_batch_size: '32' })).toBe('published');
    expect(off({}, { kv_cache_dtype: 'bf16' })).toBe('published');
  });

  test('an UNDECLARED prompt_mode on the manifest is a difference, not a match', () => {
    // The control for the published.json half of the change: strip the axis
    // the manifest now declares and the pair is refused again, naming it.
    const { prompt_mode: _drop, ...noMode } = dense.series.find((b) => b.id === 'vllm-mtp').instrument;
    const stripped = {
      ...dense,
      series: dense.series.map((b) => (b.id === 'vllm-mtp' ? { ...b, instrument: noMode } : b))
    };
    const { drawn, refused } = pairWith(repointed(), stripped);
    expect(drawn).toEqual([]);
    expect(refused.find((r) => r.series.id === 'vllm-mtp').why).toContain('prompt_mode essay → undeclared');
    expect(comparisonStateOf(DENSE, [repointed()], { subjects: { ...ladders.subjects, [DENSE.id]: stripped } })).toBe('published');
  });

  test('a paired record that is not eligible to be live does not win — the live rules still apply first', () => {
    expect(comparisonStateOf(DENSE, [repointed({ verdict: 'FAIL' })], ladders)).toBe('published');
    expect(comparisonStateOf(DENSE, [repointed({ branch: 'pr/x' })], ladders)).toBe('published');
    // newest passing wins, so a retired-instrument record AFTER a re-pointed
    // one takes the tab back to the snapshot rather than drawing a stale pair
    expect(comparisonStateOf(DENSE, [repointed(), gate({ target_model: DENSE.checkpoint })], ladders)).toBe('published');
    expect(comparisonStateOf(DENSE, [gate({ target_model: DENSE.checkpoint }), repointed()], ladders)).toBe('live');
  });
});
