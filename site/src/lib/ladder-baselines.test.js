// SPDX-License-Identifier: AGPL-3.0-only

import { expect, test } from 'bun:test';
import { assignTrendPredecessors, trendEdges } from './gate-lineage.js';
import { REQUIRED_AXES, comparable, describeDiffers, instrumentKey, instrumentOf } from './ladder-baselines.js';

// ---- fixtures: the shape the call site produces ----------------------------
// Verbatim from gates.generated.json (`concurrency-sweep`, 2026-09-13,
// 3c8809557c) minus metrics the fingerprint never reads. Every params /
// serve_overrides value is a STRING because that is how the harness records
// them; a fixture with numbers here would test a shape no record has.
const GATE = Object.freeze({
  benchmark_id: 'concurrency-sweep',
  git_sha: '3c8809557c',
  recorded_at: 1789503584,
  target_model: 'unsloth/Qwen3.8-27B-NVFP4',
  served_by: 'qwen3.8/qwen3.8-27b-nvfp4-unsloth',
  machine_id: '7af66f30966a49b6886e00e2fce4b42f',
  perf_class: 'gb10@spark-43fa',
  verdict: 'PASS',
  branch: '',
  trend_predecessor: '',
  hardware: { gpu: 'NVIDIA GB10', driver: '580.126.09', sm_clock_mhz: 2509, gpu_count: 1, source: 'nvidia-smi' },
  params: {
    concurrencies: '1, 2, 4, 8, 16, 32, 64, 128',
    isls: '512',
    min_c1: '20.099999999999998',
    min_c128: '107.6',
    min_c16: '82.55',
    min_c2: '24.05',
    min_c32: '96.8',
    min_c4: '35.7',
    min_c64: '107.6',
    min_c8: '45.7',
    min_peak: '107.6',
    osl: '320',
    prompt_mode: 'natural',
    request_timeout_s: '600',
    warmup: '1'
  },
  serve_overrides: { kv_cache_dtype: 'fp8', max_batch_size: '128', max_model_len: '4096', ssm_cache_slots: '32' },
  metrics: { c64_aggregate_tok_s: 116.72922399679739 }
});

const clone = (x) => JSON.parse(JSON.stringify(x));
const gate = (mutate) => {
  const r = clone(GATE);
  mutate?.(r);
  return r;
};

// The ladder38 vLLM+MTP leg as plan §2.3 expresses it: instrument read off
// bench/ladder38/published.json (`--max-model-len 2048 --max-num-seqs 128
// --kv-cache-dtype fp8`) and the raw leg header (`isl: 128, osl: 1024`).
// The harness has no `natural` fixture, so prompt_mode is undeclared.
const LADDER38_VLLM_MTP = Object.freeze({
  engine: 'vLLM 0.27.1',
  speculation: "vLLM's own Qwen3_5MTP, K=4 (num_speculative_tokens=3)",
  instrument: { isl: 128, osl: 1024, max_model_len: 2048, max_batch_size: 128, kv_cache_dtype: 'fp8', reps: 3, warmup: 1, temperature: 0, seed: 42 },
  measured_to: '2026-08-18T15:24:39Z',
  rungs: [{ c: 64, tok_s: 478.1 }]
});

// A one-shot on the GATE instrument, shaped per plan §2.3: numbers where the
// manifest is typed by hand, plus axes the gate never records.
const ONE_SHOT = Object.freeze({
  engine: 'vLLM 0.27.1',
  speculation: 'MTP K=4',
  instrument: { isl: 512, osl: 320, prompt_mode: 'natural', max_model_len: 4096, max_batch_size: 128, kv_cache_dtype: 'fp8', warmup: 1, reps: 3, temperature: 0, seed: 0 },
  measured_from: '2026-10-02T01:00:00Z',
  measured_to: '2026-10-02T03:00:00Z',
  rungs: [{ c: 64, tok_s: 109.4 }]
});

// ---- instrumentOf -----------------------------------------------------------

test('a gate record fingerprints its workload axes and nothing about the box', () => {
  const { kind, axes } = instrumentOf(GATE);
  expect(kind).toBe('gate');
  expect(axes).toEqual({
    checkpoint: 'unsloth/Qwen3.8-27B-NVFP4',
    gpu: 'NVIDIA GB10',
    concurrencies: '1, 2, 4, 8, 16, 32, 64, 128',
    isl: '512',
    osl: '320',
    prompt_mode: 'natural',
    request_timeout_s: '600',
    warmup: '1',
    kv_cache_dtype: 'fp8',
    max_batch_size: '128',
    max_model_len: '4096',
    ssm_cache_slots: '32'
  });
  // Thresholds and box identity are absent, not merely equal.
  for (const k of Object.keys(axes)) expect(k).not.toMatch(/^min_/);
  expect(Object.values(axes)).not.toContain('7af66f30966a49b6886e00e2fce4b42f');
  expect(Object.values(axes)).not.toContain('580.126.09');
  expect(Object.values(axes)).not.toContain('gb10@spark-43fa');
});

test('a baseline fingerprints its instrument with the same axis names', () => {
  const { kind, axes } = instrumentOf(ONE_SHOT);
  expect(kind).toBe('baseline');
  expect(axes.isl).toBe('512');
  expect(axes.max_model_len).toBe('4096');
  expect(axes.reps).toBe('3');
  expect(axes.checkpoint).toBeUndefined();
});

test('refuses a value that is neither a gate record nor a baseline', () => {
  expect(() => instrumentOf({})).toThrow(/gate record .* or a baseline/);
  expect(() => instrumentOf(null)).toThrow(TypeError);
  expect(() => instrumentOf({ params: {}, instrument: {} })).toThrow(TypeError);
});

test('refuses a non-scalar or non-finite axis instead of stringifying it', () => {
  expect(() => instrumentOf(gate((r) => (r.serve_overrides.max_batch_size = { n: 128 })))).toThrow(/scalar/);
  expect(() => instrumentOf({ instrument: { isl: NaN } })).toThrow(/finite/);
});

test('instrumentKey ignores which box ran it', () => {
  const other = gate((r) => {
    r.machine_id = 'e8b2';
    r.perf_class = 'gb10@edgexpert-2640';
    r.hardware.driver = '580.173.02';
    r.hardware.sm_clock_mhz = 2405;
  });
  expect(instrumentKey(other)).toBe(instrumentKey(GATE));
  expect(instrumentKey(gate((r) => (r.params.osl = '200')))).not.toBe(instrumentKey(GATE));
});

// ---- comparable: the pin --------------------------------------------------

test('THE PIN: a ladder38 vLLM leg can never be drawn against a gate record', () => {
  const { ok, differs } = comparable(GATE, LADDER38_VLLM_MTP);
  expect(ok).toBe(false);
  expect(differs).toEqual([
    { axis: 'isl', a: '512', b: '128' },
    { axis: 'osl', a: '320', b: '1024' },
    { axis: 'prompt_mode', a: 'natural', b: null },
    { axis: 'max_model_len', a: '4096', b: '2048' }
  ]);
  expect(describeDiffers(differs)).toBe(
    'isl 512 → 128, osl 320 → 1024, prompt_mode natural → undeclared, max_model_len 4096 → 2048'
  );
});

test('a one-shot on the gate instrument is comparable despite one-sided axes', () => {
  // The manifest declares reps/temperature/seed the gate never records, and
  // the gate declares ssm_cache_slots/concurrencies vLLM has no notion of.
  const { ok, differs } = comparable(GATE, ONE_SHOT);
  expect(differs).toEqual([]);
  expect(ok).toBe(true);
  expect(comparable(ONE_SHOT, GATE).ok).toBe(true);
});

test('a required axis missing on either side is a difference, never a match', () => {
  for (const axis of REQUIRED_AXES) {
    const b = clone(ONE_SHOT);
    delete b.instrument[axis];
    const { ok, differs } = comparable(GATE, b);
    expect(ok).toBe(false);
    expect(differs).toEqual([{ axis, a: expect.any(String), b: null }]);
  }
});

test('a shared non-required axis that differs across kinds still counts', () => {
  const b = clone(ONE_SHOT);
  b.instrument.warmup = 0;
  expect(comparable(GATE, b)).toEqual({ ok: false, differs: [{ axis: 'warmup', a: '1', b: '0' }] });
});

test('gate vs gate: a floor change alone is the same instrument', () => {
  const later = gate((r) => {
    r.params.min_c1 = '17.2';
    r.params.min_c64 = '100';
    r.params.min_peak = '100';
  });
  expect(comparable(GATE, later)).toEqual({ ok: true, differs: [] });
});

test('gate vs gate: an override present on one side only is a different instrument', () => {
  const dflash = gate((r) => {
    r.serve_overrides.dflash = 'true';
  });
  expect(comparable(GATE, dflash)).toEqual({ ok: false, differs: [{ axis: 'dflash', a: null, b: 'true' }] });
});

test('gate vs gate: the 2026-08-30 widening is named by the axes that moved', () => {
  const narrow = gate((r) => {
    r.params.concurrencies = '1, 4, 8, 16';
    r.serve_overrides.max_batch_size = '32';
  });
  const { ok, differs } = comparable(narrow, GATE);
  expect(ok).toBe(false);
  expect(differs).toEqual([
    { axis: 'max_batch_size', a: '32', b: '128' },
    { axis: 'concurrencies', a: '1, 4, 8, 16', b: '1, 2, 4, 8, 16, 32, 64, 128' }
  ]);
});

test('values compare as trimmed strings so a typed 4096 equals a recorded "4096"', () => {
  const b = clone(ONE_SHOT);
  b.instrument.kv_cache_dtype = ' fp8 ';
  b.instrument.max_model_len = '4096';
  expect(comparable(GATE, b).ok).toBe(true);
});

// ---- consistency with gate-lineage --------------------------------------
// gate-lineage keys a trend edge on the same axes PLUS the box. So for every
// mutation of a record, lineage links the pair iff comparable() says ok —
// except on box axes and thresholds, where this module deliberately says ok
// and lineage deliberately does not. Any other disagreement means one of the
// two modules has a private notion of "same instrument".

const lineageLinks = (a, b) => {
  const records = [clone(a), clone(b)];
  records[1].git_sha = 'child000000';
  records[1].recorded_at = records[0].recorded_at + 60;
  assignTrendPredecessors(records, () => true);
  return trendEdges(records).length === 1;
};

const MUTATIONS = [
  ['target_model', (r) => (r.target_model = 'Qwen/Qwen3.6-35B-A3B-FP8'), 'workload'],
  ['served_by', (r) => (r.served_by = 'qwen3.8/qwen3.8-27b-nvfp4-dflash2'), 'lineage-only'],
  ['hardware.gpu', (r) => (r.hardware.gpu = 'NVIDIA H100'), 'workload'],
  ['params.isls', (r) => (r.params.isls = '128'), 'workload'],
  ['params.osl', (r) => (r.params.osl = '1024'), 'workload'],
  ['params.prompt_mode', (r) => (r.params.prompt_mode = 'synthetic'), 'workload'],
  ['params.concurrencies', (r) => (r.params.concurrencies = '1, 4, 8, 16'), 'workload'],
  ['params.warmup', (r) => (r.params.warmup = '0'), 'workload'],
  ['serve_overrides.max_model_len', (r) => (r.serve_overrides.max_model_len = '2048'), 'workload'],
  ['serve_overrides.max_batch_size', (r) => (r.serve_overrides.max_batch_size = '32'), 'workload'],
  ['serve_overrides.kv_cache_dtype', (r) => (r.serve_overrides.kv_cache_dtype = 'bf16'), 'workload'],
  ['serve_overrides.ssm_cache_slots', (r) => (r.serve_overrides.ssm_cache_slots = '8'), 'workload'],
  ['serve_overrides.+dflash', (r) => (r.serve_overrides.dflash = 'true'), 'workload'],
  ['machine_id', (r) => (r.machine_id = 'other-box'), 'box'],
  ['hardware.driver', (r) => (r.hardware.driver = '595.71.05'), 'box'],
  ['params.min_c64', (r) => (r.params.min_c64 = '100'), 'threshold'],
  ['params.min_peak', (r) => (r.params.min_peak = '100'), 'threshold']
];

for (const [name, mutate, category] of MUTATIONS) {
  test(`lineage consistency: ${name} (${category})`, () => {
    const mutated = gate(mutate);
    const ours = comparable(GATE, mutated).ok;
    const theirs = lineageLinks(GATE, mutated);
    if (category === 'workload') {
      expect(ours).toBe(false);
      expect(theirs).toBe(false);
    } else if (category === 'lineage-only') {
      // served_by names the recipe (engine configuration), which lineage
      // keys on and a workload fingerprint must not: a baseline is another
      // engine by definition. Documented disagreement.
      expect(theirs).toBe(false);
      expect(ours).toBe(true);
    } else {
      // box / threshold: the documented refinement of the lineage key.
      expect(theirs).toBe(false);
      expect(ours).toBe(true);
    }
  });
}

test('lineage consistency: an unmutated record links in both', () => {
  expect(comparable(GATE, gate()).ok).toBe(true);
  expect(lineageLinks(GATE, gate())).toBe(true);
});
