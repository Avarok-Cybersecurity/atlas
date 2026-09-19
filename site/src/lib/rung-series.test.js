// SPDX-License-Identifier: AGPL-3.0-only

import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { floorOf, rungMetricKey, rungPanel } from './rung-series.js';

// ---- fixtures: the shape the call site produces ----------------------------
// Three real dense records from gates.generated.json, chosen for the history
// they span: the narrow instrument (2026-08-16), the first widened run
// (2026-08-30, concurrencies 1…128, batch 128) and the latest (2026-09-13,
// floors raised). Every params / serve_overrides value is a STRING and
// recorded_at is an INTEGER, exactly as recorded.
const NARROW = Object.freeze({
  benchmark_id: 'concurrency-sweep',
  git_sha: '33975ee507',
  recorded_at: 1786878759,
  target_model: 'unsloth/Qwen3.8-27B-NVFP4',
  served_by: 'qwen3.8/qwen3.8-27b-nvfp4-unsloth',
  machine_id: '',
  perf_class: '',
  verdict: 'PASS',
  hardware: { gpu: 'NVIDIA GB10', driver: '580.126.09', sm_clock_mhz: 2463, source: 'nvidia-smi' },
  params: { concurrencies: '1, 4, 8, 16', isls: '512', min_c1: '16.2', min_c16: '72', min_c4: '33.5', min_c8: '50.5', min_peak: '72', osl: '320', prompt_mode: 'natural', request_timeout_s: '600', warmup: '1' },
  serve_overrides: { kv_cache_dtype: 'fp8', max_batch_size: '32', max_model_len: '4096', ssm_cache_slots: '32' },
  metrics: { c16_aggregate_tok_s: 82.39239770538258, c1_aggregate_tok_s: 18.54256628151848, c4_aggregate_tok_s: 41.59705997720475, c8_aggregate_tok_s: 56.78057748315703, peak_aggregate_tok_s: 82.39239770538258, vacuous_cells: 0 }
});

const WIDENED = Object.freeze({
  benchmark_id: 'concurrency-sweep',
  git_sha: 'f0f6e48845',
  recorded_at: 1788074082,
  target_model: 'unsloth/Qwen3.8-27B-NVFP4',
  served_by: 'qwen3.8/qwen3.8-27b-nvfp4-unsloth',
  machine_id: '7af66f30966a49b6886e00e2fce4b42f',
  perf_class: 'gb10@spark-256a',
  verdict: 'PASS',
  hardware: { gpu: 'NVIDIA GB10', driver: '580.126.09', sm_clock_mhz: 2457, source: 'nvidia-smi' },
  params: { concurrencies: '1, 2, 4, 8, 16, 32, 64, 128', isls: '512', min_c1: '17.2', min_c128: '107.6', min_c16: '82.55', min_c2: '24.05', min_c32: '96.8', min_c4: '35.7', min_c64: '107.6', min_c8: '45.7', min_peak: '107.6', osl: '320', prompt_mode: 'natural', request_timeout_s: '600', warmup: '1' },
  serve_overrides: { kv_cache_dtype: 'fp8', max_batch_size: '128', max_model_len: '4096', ssm_cache_slots: '8' },
  metrics: { c128_aggregate_tok_s: 114.90779918842817, c16_aggregate_tok_s: 87.75533899558741, c1_aggregate_tok_s: 18.19488616982209, c2_aggregate_tok_s: 28.000794999971678, c32_aggregate_tok_s: 103.88529337245855, c4_aggregate_tok_s: 45.35345473150269, c64_aggregate_tok_s: 115.38751912362589, c8_aggregate_tok_s: 56.89954503694784, peak_aggregate_tok_s: 115.38751912362589, vacuous_cells: 0 }
});

const LATEST = Object.freeze({
  benchmark_id: 'concurrency-sweep',
  git_sha: '3c8809557c',
  recorded_at: 1789503584,
  target_model: 'unsloth/Qwen3.8-27B-NVFP4',
  served_by: 'qwen3.8/qwen3.8-27b-nvfp4-unsloth',
  machine_id: '7af66f30966a49b6886e00e2fce4b42f',
  perf_class: 'gb10@spark-43fa',
  verdict: 'PASS',
  hardware: { gpu: 'NVIDIA GB10', driver: '580.126.09', sm_clock_mhz: 2509, gpu_count: 1, source: 'nvidia-smi' },
  params: { concurrencies: '1, 2, 4, 8, 16, 32, 64, 128', isls: '512', min_c1: '20.099999999999998', min_c128: '107.6', min_c16: '82.55', min_c2: '24.05', min_c32: '96.8', min_c4: '35.7', min_c64: '107.6', min_c8: '45.7', min_peak: '107.6', osl: '320', prompt_mode: 'natural', request_timeout_s: '600', warmup: '1' },
  serve_overrides: { kv_cache_dtype: 'fp8', max_batch_size: '128', max_model_len: '4096', ssm_cache_slots: '8' },
  metrics: { c128_aggregate_tok_s: 116.71027409231166, c16_aggregate_tok_s: 93.5483836334447, c1_aggregate_tok_s: 21.197417852866263, c2_aggregate_tok_s: 27.411469436285678, c32_aggregate_tok_s: 105.22135595769224, c4_aggregate_tok_s: 53.808212972834305, c64_aggregate_tok_s: 116.72922399679739, c8_aggregate_tok_s: 70.07572402227657, peak_aggregate_tok_s: 116.72922399679739, vacuous_cells: 0 }
});

// The DFlash2 gate stops at C=16 and declares "0" (the gate's OFF state) for
// the rungs it never runs.
const DFLASH = Object.freeze({
  benchmark_id: 'concurrency-sweep-dflash2',
  git_sha: '3c8809557c',
  recorded_at: 1789505290,
  target_model: 'unsloth/Qwen3.8-27B-NVFP4',
  served_by: 'qwen3.8/qwen3.8-27b-nvfp4-dflash2',
  machine_id: '7af66f30966a49b6886e00e2fce4b42f',
  perf_class: 'gb10@spark-43fa',
  verdict: 'PASS',
  hardware: { gpu: 'NVIDIA GB10', driver: '580.126.09', sm_clock_mhz: 2424, gpu_count: 1, source: 'nvidia-smi' },
  params: { concurrencies: '1, 2, 4, 8, 16', isls: '512', min_c1: '22.75', min_c128: '0', min_c16: '57', min_c2: '35.38', min_c32: '0', min_c4: '41.77', min_c64: '0', min_c8: '49.62', min_peak: '57', osl: '200', prompt_mode: 'natural', request_timeout_s: '600', warmup: '1' },
  serve_overrides: { dflash: 'true', dflash_gamma: '8', draft_model: 'incoai/Qwen3.8-27B-DFlash2', kv_cache_dtype: 'fp8', max_batch_size: '16', max_model_len: '4096', ssm_cache_slots: '32' },
  metrics: { c16_aggregate_tok_s: 65.20392122935594, c1_aggregate_tok_s: 25.01418522238198, c2_aggregate_tok_s: 48.83333368426188, c4_aggregate_tok_s: 57.34948089939581, c8_aggregate_tok_s: 69.11915101499761, peak_aggregate_tok_s: 69.11915101499761 }
});

const clone = (x) => JSON.parse(JSON.stringify(x));
/** A later run on LATEST's instrument, with the given sha, time offset and c64 value. */
const run = (sha, dt, c64, mutate) => {
  const r = clone(LATEST);
  r.git_sha = sha;
  r.recorded_at = LATEST.recorded_at + dt;
  r.metrics.c64_aggregate_tok_s = c64;
  mutate?.(r);
  return r;
};

const ONE_SHOT = Object.freeze({
  engine: 'vLLM 0.27.1',
  speculation: 'MTP K=4',
  instrument: { isl: 512, osl: 320, prompt_mode: 'natural', max_model_len: 4096, max_batch_size: 128, kv_cache_dtype: 'fp8', warmup: 1, reps: 3, temperature: 0, seed: 0 },
  measured_from: '2026-10-02T01:00:00Z',
  measured_to: '2026-10-02T03:00:00Z',
  rungs: [{ c: 64, tok_s: 109.4 }, { c: 128, tok_s: 112.0 }]
});

const LADDER38_VLLM_MTP = Object.freeze({
  engine: 'vLLM 0.27.1',
  instrument: { isl: 128, osl: 1024, max_model_len: 2048, max_batch_size: 128, kv_cache_dtype: 'fp8', reps: 3, warmup: 1, temperature: 0, seed: 42 },
  measured_to: '2026-08-18T15:24:39Z',
  rungs: [{ c: 64, tok_s: 478.1 }]
});

// ---- metric key --------------------------------------------------------------

test('the rung metric key is the shape gates.js#LADDER_KEY parses', () => {
  // gates.js cannot be imported under bun test ($lib json), so pin against
  // its source text: the regex it exports must accept and parse our key.
  const src = readFileSync(join(import.meta.dir, 'gates.js'), 'utf8');
  const m = /export const LADDER_KEY = \/(.+)\/;/.exec(src);
  expect(m).not.toBeNull();
  const ladderKey = new RegExp(m[1]);
  expect(ladderKey.exec(rungMetricKey(64))?.[1]).toBe('64');
  expect(ladderKey.test(rungMetricKey(1))).toBe(true);
  expect(rungPanel(64, [LATEST]).key).toBe(rungMetricKey(64));
});

// ---- time axis ---------------------------------------------------------------

test('x is recorded_at in epoch seconds and points are sorted by it whatever the input order', () => {
  const p = rungPanel(1, [LATEST, NARROW, WIDENED]);
  expect(p.points.map((pt) => pt.t)).toEqual([1786878759, 1788074082, 1789503584]);
  expect(p.points.map((pt) => pt.rec.git_sha)).toEqual(['33975ee507', 'f0f6e48845', '3c8809557c']);
  expect(p.latest.rec).toBe(p.points[2].rec);
  expect(p.runs).toBe(3);
});

test('a recorded_at that is not a finite number is refused, never coerced', () => {
  // The trap: Date.parse on an integer is NaN, and an ISO string here would
  // silently sort as NaN. Both must throw by sha rather than plot.
  expect(() => rungPanel(64, [run('iso', 0, 100, (r) => (r.recorded_at = '2026-09-13T00:00:00Z'))])).toThrow(/iso.*recorded_at/);
  expect(() => rungPanel(64, [run('none', 0, 100, (r) => delete r.recorded_at)])).toThrow(/none.*recorded_at/);
  expect(() => rungPanel(64, [run('nan', 0, 100, (r) => (r.recorded_at = NaN))])).toThrow(/recorded_at/);
});

test('the rung must be a positive integer', () => {
  expect(() => rungPanel(0, [LATEST])).toThrow(TypeError);
  expect(() => rungPanel('64', [LATEST])).toThrow(TypeError);
});

// ---- floors ------------------------------------------------------------------

test('floorOf reads the declared floor as a number and nothing else', () => {
  expect(floorOf(LATEST, 64)).toBe(107.6);
  expect(floorOf(LATEST, 1)).toBe(20.099999999999998);
  expect(floorOf(NARROW, 64)).toBeNull(); // never declared
  expect(floorOf(DFLASH, 32)).toBeNull(); // declared "0": the gate's OFF state
  expect(() => floorOf(run('bad', 0, 1, (r) => (r.params.min_c64 = 'n/a')), 64)).toThrow(/bad.*min_c64/);
  expect(() => floorOf(run('empty', 0, 1, (r) => (r.params.min_c64 = '')), 64)).toThrow(/empty.*min_c64/);
});

test('a record with no declared floor draws no floor line and gets no zero', () => {
  const p = rungPanel(64, [NARROW]);
  expect(p.measured).toBe(false); // NARROW never measured C=64
  const q = rungPanel(1, [run('nofloor', 0, 21, (r) => delete r.params.min_c1)]);
  expect(q.floors).toEqual([]);
  expect(q.points[0].floor).toBeNull();
  expect(q.points[0].belowFloor).toBe(false);
});

test('a declared "0" floor is the gate OFF state, not a line at zero', () => {
  const p = rungPanel(16, [run('zero', 0, 90, (r) => (r.params.min_c16 = '0'))]);
  expect(p.floors).toEqual([]);
  expect(p.points[0].floor).toBeNull();
});

test('floors are a step: distinct values in order of first use, each with its span', () => {
  const p = rungPanel(1, [NARROW, WIDENED, LATEST]);
  expect(p.floors).toEqual([
    { value: 16.2, label: 'gate floor 16.2', from: 1786878759, to: 1786878759 },
    { value: 17.2, label: 'gate floor 17.2', from: 1788074082, to: 1788074082 },
    { value: 20.099999999999998, label: 'gate floor 20.1', from: 1789503584, to: 1789503584 }
  ]);
  expect(p.points.map((pt) => pt.floor)).toEqual([16.2, 17.2, 20.099999999999998]);
});

test('a same-floor run extends the span rather than adding a second line', () => {
  const p = rungPanel(64, [LATEST, run('a', 100, 116), run('b', 200, 117)]);
  expect(p.floors).toEqual([{ value: 107.6, label: 'gate floor 107.6', from: LATEST.recorded_at, to: LATEST.recorded_at + 200 }]);
});

test('belowFloor is read against that record’s own floor', () => {
  const fail = run('fail', 100, 100.0, (r) => {
    r.verdict = 'FAIL';
  });
  const p = rungPanel(64, [LATEST, fail]);
  expect(p.points[1]).toMatchObject({ pass: false, verdict: 'FAIL', belowFloor: true, newLow: false });
  expect(p.points[0].belowFloor).toBe(false);
});

// ---- new low -----------------------------------------------------------------

test('a PASS below every earlier same-instrument PASS is a new low; the first never is', () => {
  // LATEST is 116.73: 116.0 is below it (low), 115.0 below that (low),
  // 115.5 is above the running low (not a low, though below LATEST).
  const p = rungPanel(64, [LATEST, run('a', 100, 116.0), run('b', 200, 115.0), run('c', 300, 115.5)]);
  expect(p.points.map((pt) => pt.newLow)).toEqual([false, true, true, false]);
});

test('an equal value is not a new low', () => {
  const p = rungPanel(64, [LATEST, run('a', 100, LATEST.metrics.c64_aggregate_tok_s)]);
  expect(p.points[1].newLow).toBe(false);
});

test('a FAIL is never a new low and does not lower the bar for later PASSes', () => {
  const fail = run('f', 100, 90, (r) => (r.verdict = 'FAIL'));
  const p = rungPanel(64, [LATEST, fail, run('a', 200, 116.0)]);
  expect(p.points.map((pt) => pt.newLow)).toEqual([false, false, true]);
});

test('a lower value on ANOTHER instrument is not a new low, and the earlier instrument still counts when it returns', () => {
  const slots32 = (r) => (r.serve_overrides.ssm_cache_slots = '32');
  const p = rungPanel(64, [
    LATEST, //            slots 8, 116.73
    run('a', 100, 110.0, slots32), // slots 32: first on its instrument → not a low
    run('b', 200, 116.0), //          slots 8: below 116.73 → new low on slots 8
    run('c', 300, 109.0, slots32) //  slots 32: below 110 → new low on slots 32
  ]);
  expect(p.points.map((pt) => pt.newLow)).toEqual([false, false, true, true]);
});

// ---- regimes -----------------------------------------------------------------

test('an instrument change marks the first point of the new regime with the axes that moved', () => {
  const p = rungPanel(1, [NARROW, WIDENED, LATEST]);
  expect(p.regimes).toHaveLength(1);
  expect(p.regimes[0]).toMatchObject({ index: 1, t: WIDENED.recorded_at, rec: p.points[1].rec });
  expect(p.regimes[0].differs).toEqual([
    { axis: 'max_batch_size', a: '32', b: '128' },
    { axis: 'concurrencies', a: '1, 4, 8, 16', b: '1, 2, 4, 8, 16, 32, 64, 128' },
    { axis: 'ssm_cache_slots', a: '32', b: '8' }
  ]);
  expect(p.regimes[0].label).toBe(
    'max_batch_size 32 → 128, concurrencies 1, 4, 8, 16 → 1, 2, 4, 8, 16, 32, 64, 128, ssm_cache_slots 32 → 8'
  );
});

test('a box change is not a regime change', () => {
  const otherBox = run('box', 100, 116, (r) => {
    r.machine_id = 'e8b2';
    r.perf_class = 'gb10@edgexpert-2640';
    r.hardware.driver = '580.173.02';
  });
  expect(rungPanel(64, [LATEST, otherBox]).regimes).toEqual([]);
});

test('a floor change is not a regime change', () => {
  const raised = run('floor', 100, 116, (r) => (r.params.min_c64 = '110'));
  const p = rungPanel(64, [LATEST, raised]);
  expect(p.regimes).toEqual([]);
  expect(p.floors).toHaveLength(2);
});

// ---- baselines ---------------------------------------------------------------

test('a comparable baseline becomes a dated reference line at this rung', () => {
  const p = rungPanel(64, [LATEST], { baselines: [ONE_SHOT] });
  expect(p.refs).toEqual([
    { kind: 'baseline', value: 109.4, label: 'vLLM 0.27.1 one-shot · 2026-10-02 · 109.4', engine: 'vLLM 0.27.1', date: '2026-10-02', baseline: ONE_SHOT }
  ]);
  expect(p.notes).toEqual([]);
});

test('THE PIN: a ladder38 vLLM leg never becomes a reference line on a gate-instrument rung', () => {
  const p = rungPanel(64, [LATEST], { baselines: [LADDER38_VLLM_MTP] });
  expect(p.refs).toEqual([]);
  expect(p.notes).toHaveLength(1);
  expect(p.notes[0]).toMatchObject({ engine: 'vLLM 0.27.1', kind: 'not-comparable' });
  expect(p.notes[0].differs.map((d) => d.axis)).toEqual(['isl', 'osl', 'prompt_mode', 'max_model_len']);
  expect(p.notes[0].label).toBe(
    'vLLM 0.27.1 — not comparable on this instrument (isl 512 → 128, osl 320 → 1024, prompt_mode natural → undeclared, max_model_len 4096 → 2048)'
  );
});

test('comparability is judged against the LATEST instrument, not an earlier regime', () => {
  // ONE_SHOT matches LATEST (batch 128); a panel whose latest run is narrow
  // (batch 32) must not draw it even though an earlier point would match.
  const narrowLater = run('narrow', 100, 100, (r) => {
    r.params.concurrencies = '1, 4, 8, 16, 64';
    r.serve_overrides.max_batch_size = '32';
  });
  const p = rungPanel(64, [LATEST, narrowLater], { baselines: [ONE_SHOT] });
  expect(p.refs).toEqual([]);
  // `concurrencies` is one-sided across kinds and therefore not a difference.
  expect(p.notes[0].differs).toEqual([{ axis: 'max_batch_size', a: '32', b: '128' }]);
});

test('a baseline that was not run at this rung is a note, not a line and not an error', () => {
  const p = rungPanel(32, [LATEST], { baselines: [ONE_SHOT] });
  expect(p.refs).toEqual([]);
  expect(p.notes).toEqual([{ engine: 'vLLM 0.27.1', kind: 'no-rung', label: 'vLLM 0.27.1 was not run at C=32' }]);
});

test('a baseline rung without a numeric tok_s is refused, never transcribed', () => {
  const typed = { ...ONE_SHOT, rungs: [{ c: 64, tok_s: '109.4' }] };
  expect(() => rungPanel(64, [LATEST], { baselines: [typed] })).toThrow(/tok_s/);
});

test('a baseline without a measured_to date cannot be labelled a one-shot and is refused', () => {
  const undated = { ...ONE_SHOT, measured_to: undefined };
  expect(() => rungPanel(64, [LATEST], { baselines: [undated] })).toThrow(/measured_to/);
  const epoch = { ...ONE_SHOT, measured_to: 1791000000 };
  expect(rungPanel(64, [LATEST], { baselines: [epoch] }).refs[0].date).toBe('2026-10-03');
});

test('no baselines is a state: no refs, no notes, nothing thrown', () => {
  const p = rungPanel(64, [LATEST]);
  expect(p.refs).toEqual([]);
  expect(p.notes).toEqual([]);
});

// ---- not measured ------------------------------------------------------------

test('a rung the subject never measured is an empty, honest panel', () => {
  const p = rungPanel(64, [DFLASH], { baselines: [ONE_SHOT] });
  expect(p).toMatchObject({ c: 64, title: 'C=64', unit: 'tok/s', measured: false, runs: 0, latest: null, points: [], floors: [], regimes: [], refs: [] });
  expect(p.notes).toEqual([{ engine: 'vLLM 0.27.1', kind: 'no-rung', label: 'nothing measured at C=64 to compare vLLM 0.27.1 with' }]);
  expect(p.metrics).toEqual([{ key: 'c64_aggregate_tok_s', label: 'aggregate tok/s' }]);
});

test('DFlash rungs it did run carry their own floors and instrument', () => {
  const p = rungPanel(16, [DFLASH]);
  expect(p.measured).toBe(true);
  expect(p.floors).toEqual([{ value: 57, label: 'gate floor 57.0', from: DFLASH.recorded_at, to: DFLASH.recorded_at }]);
  expect(p.points[0].v).toBeCloseTo(65.2039, 3);
});

// ---- the real record set -----------------------------------------------------
// Invariants only (the file is a build product and grows): nothing throws on
// any rung, time is strictly ascending, every regime index points at a real
// point, every floor is positive, and the 2026-08-30 widening — a fact of the
// history — is a regime rule on every narrow-era rung.

test('every rung of the real dense record set builds and holds its invariants', () => {
  const gates = JSON.parse(readFileSync(join(import.meta.dir, 'gates.generated.json'), 'utf8'));
  const records = gates.benchmarks['concurrency-sweep']?.records ?? [];
  expect(records.length).toBeGreaterThan(0);
  for (const c of [1, 2, 4, 8, 16, 32, 64, 128]) {
    const p = rungPanel(c, records);
    for (let i = 1; i < p.points.length; i += 1) expect(p.points[i].t).toBeGreaterThanOrEqual(p.points[i - 1].t);
    for (const r of p.regimes) expect(p.points[r.index].rec).toBe(r.rec);
    for (const f of p.floors) expect(f.value).toBeGreaterThan(0);
    expect(p.points.filter((pt) => pt.newLow).every((pt) => pt.pass)).toBe(true);
  }
  const c1 = rungPanel(1, records);
  expect(c1.regimes.some((r) => r.differs.some((d) => d.axis === 'concurrencies'))).toBe(true);
});
