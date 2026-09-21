// SPDX-License-Identifier: AGPL-3.0-only
//
// concurrency-instrument-parity.test.js — the two charts of the Concurrency
// tab must measure ONE workload, and this is the assertion that keeps them
// that way after everyone in this conversation has forgotten why.
//
// The tab draws Atlas against a measured vLLM bar on top and Atlas's own
// commit-to-commit movement below. Until 2026-09-21 the top was a frozen
// August campaign pair on ISL 128 / OSL 1024 and the bottom was the live gate
// on ISL 512 / OSL 320 — ~478 vs ~116 tok/s on the same checkpoint, which
// readers took for an inconsistent engine. `ladder-baselines.js` was written
// to make that impossible to draw; this makes it impossible to DRIFT, by
// checking the two declarations against each other in CI:
//
//   the TOP's bar   bench/ladder38/published.json, series vllm-mtp — the leg
//                   the published ratio is computed against (`parity:
//                   matched`), NOT the unmatched no-speculation leg;
//   the BOTTOM      kernels/gb10/qwen3.8-27b/BENCH.toml, gate
//                   concurrency-sweep — [benchmarks.param_overrides] and
//                   [benchmarks.serve_overrides], which `check_record`
//                   demands on every record, so the file IS the instrument.
//
// It is deliberately `comparable()` doing the deciding rather than a list of
// equalities written here: one rule, one place, and a REQUIRED axis added
// later is enforced here the day it is added instead of silently skipped.
import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseToml } from '../../scripts/lib/bench-toml.mjs';
import { REQUIRED_AXES, comparable, instrumentOf } from './ladder-baselines.js';

const repo = (p) => fileURLToPath(new URL(`../../../${p}`, import.meta.url));
const manifest = JSON.parse(readFileSync(repo('bench/ladder38/published.json'), 'utf8'));
const bench = parseToml(readFileSync(repo('kernels/gb10/qwen3.8-27b/BENCH.toml'), 'utf8'));

const entry = bench.benchmarks.find((b) => b.gate === 'concurrency-sweep');
const mtp = manifest.series.find((s) => s.id === 'vllm-mtp');

/** The gate's declaration, shaped as the record `check_record` will demand. */
const declared = (over = {}) => ({
  target_model: entry.checkpoint,
  params: { ...entry.param_overrides, ...(over.params ?? {}) },
  serve_overrides: { ...entry.serve_overrides, ...(over.serve_overrides ?? {}) }
});
const bar = (instrument = mtp.instrument) => ({ checkpoint: manifest.workload.checkpoint, instrument });

describe('the gate and the published bar declare one instrument', () => {
  test('THE INVARIANT: BENCH.toml concurrency-sweep is comparable to published.json vllm-mtp', () => {
    const { ok, differs } = comparable(declared(), bar());
    expect(differs).toEqual([]);
    expect(ok).toBe(true);
  });

  test('and every REQUIRED axis is actually declared on both sides, not merely equal', () => {
    // comparable() already refuses an undeclared axis, but a future refactor
    // could satisfy it with two absences; name them instead.
    const gateAxes = instrumentOf(declared()).axes;
    const barAxes = instrumentOf(bar()).axes;
    for (const axis of REQUIRED_AXES) {
      expect(`${axis}=${gateAxes[axis]}`).toBe(`${axis}=${barAxes[axis]}`);
      expect(gateAxes[axis]).toBeDefined();
    }
    // the values themselves, pinned, so a drift shows the reader WHAT moved
    expect(REQUIRED_AXES.map((a) => `${a} ${gateAxes[a]}`)).toEqual([
      'isl 128',
      'osl 1024',
      'prompt_mode essay',
      'max_model_len 2048',
      'max_batch_size 128',
      'kv_cache_dtype fp8'
    ]);
  });

  test('THE CONTROL: each axis moved alone breaks the pair, so the invariant above can fail', () => {
    // Every one of these is a plausible edit to BENCH.toml. If any left the
    // pair intact, the test above would be measuring nothing.
    const mutations = [
      [{ params: { isls: '512' } }, 'isl'],
      [{ params: { osl: '320' } }, 'osl'],
      [{ params: { prompt_mode: 'natural' } }, 'prompt_mode'],
      [{ serve_overrides: { max_model_len: '4096' } }, 'max_model_len'],
      [{ serve_overrides: { max_batch_size: '32' } }, 'max_batch_size'],
      [{ serve_overrides: { kv_cache_dtype: 'bf16' } }, 'kv_cache_dtype']
    ];
    for (const [over, axis] of mutations) {
      const { ok, differs } = comparable(declared(over), bar());
      expect(ok).toBe(false);
      expect(differs.map((d) => d.axis)).toContain(axis);
    }
    // and dropping the axis from the manifest is refused just as loudly
    const { prompt_mode: _gone, ...noMode } = mtp.instrument;
    expect(comparable(declared(), bar(noMode)).differs.map((d) => d.axis)).toContain('prompt_mode');
  });

  test('the bar is the MATCHED leg — the no-speculation leg stays unmatched on purpose', () => {
    expect(mtp.parity).toBe('matched');
    const nospec = manifest.series.find((s) => s.id === 'vllm-nospec');
    const { ok, differs } = comparable(declared(), bar(nospec.instrument));
    expect(ok).toBe(false);
    expect(differs.map((d) => d.axis).sort()).toEqual(['kv_cache_dtype', 'max_model_len']);
  });

  test('the rung floors are OFF, because they were cut on the retired instrument', () => {
    // Re-pointing moved every rung 1.15x-4.3x; a floor carried over would
    // report a guard that cannot fail. `RUNGS` in concurrency.rs documents
    // 0.0 as record-without-gating. Delete this test when they are re-cut
    // from records on THIS instrument — and not before.
    for (const key of ['c1', 'c2', 'c4', 'c8', 'c16', 'c32', 'c64', 'c128', 'peak']) {
      expect(`${key} ${entry.metrics[`${key}_aggregate_tok_s`].min}`).toBe(`${key} 0`);
    }
    // what is NOT off: the two bars the re-point does not invalidate.
    expect(entry.metrics.vacuous_cells.max).toBe(0);
    expect(entry.metrics.min_completion_tokens.min).toBe(260);
  });
});
