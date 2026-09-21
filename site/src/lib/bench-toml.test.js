// SPDX-License-Identifier: AGPL-3.0-only
//
// The BENCH.toml reader behind gen-gates.mjs#gate_limits. Two kinds of proof:
// hand-made files that exercise the traps those files actually contain (a
// table header quoted inside a `note`, backslash continuations, a metric row
// with `noise` but no bound), and a cross-check of the whole parse against
// Bun's own TOML parser on every real kernels/gb10/*/BENCH.toml — the reader
// runs under Node in the generator, where Bun's is not available.

import { describe, expect, test } from 'bun:test';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { declaredLimitsOf, mergeDeclaredLimits, parseToml } from '../../scripts/lib/bench-toml.mjs';

const KERNELS = fileURLToPath(new URL('../../../kernels/gb10/', import.meta.url));
const benchFiles = readdirSync(KERNELS)
  .map((d) => `${KERNELS}${d}/BENCH.toml`)
  .filter((p) => {
    try {
      readFileSync(p);
      return true;
    } catch {
      return false;
    }
  });

const SAMPLE = `
# a comment with [benchmarks.metrics.trap] in it
[[benchmarks]]
quant = "nvfp4"
checkpoint = "unsloth/Qwen3.8-27B-NVFP4"
gate = "concurrency-sweep"
default = true
note = """\\
prose that quotes [benchmarks.metrics.c1_aggregate_tok_s] and says \\
min = 999 as text, then a "quoted" word and a line that ends the string"""

[benchmarks.param_overrides]
concurrencies = "1,2,4"

[benchmarks.metrics.c1_aggregate_tok_s]
# ratcheted 2026-09-11
min = 20.4 # trailing comment
noise = 0.3

[benchmarks.metrics.c2_aggregate_tok_s]
noise = 0.95

[benchmarks.metrics.vacuous_cells]
max = 0.0

[[benchmarks]]
quant = "nvfp4"
checkpoint = "unsloth/Qwen3.8-27B-NVFP4"
gate = "ttft-warm-gate"
status = "unmeasured"
note = 'literal [benchmarks.metrics.median_ms] max = 1'
`;

describe('the reader on a hand-made file', () => {
  const parsed = parseToml(SAMPLE);

  test('a header quoted inside a note string is prose, not a table', () => {
    expect(parsed.benchmarks).toHaveLength(2);
    expect(parsed.benchmarks[0].note).toBe(
      'prose that quotes [benchmarks.metrics.c1_aggregate_tok_s] and says min = 999 as text, then a "quoted" word and a line that ends the string'
    );
    expect(parsed.benchmarks[0].metrics.c1_aggregate_tok_s).toEqual({ min: 20.4, noise: 0.3 });
    expect(parsed.benchmarks[0].param_overrides).toEqual({ concurrencies: '1,2,4' });
    expect(parsed.benchmarks[1].note).toBe('literal [benchmarks.metrics.median_ms] max = 1');
  });

  test('limits: only min/max rows count, an unmeasured entry contributes nothing', () => {
    expect(declaredLimitsOf(parsed, 'sample')).toEqual({
      'concurrency-sweep': {
        'unsloth/Qwen3.8-27B-NVFP4': { c1_aggregate_tok_s: { min: 20.4 }, vacuous_cells: { max: 0 } }
      }
    });
  });

  test('NEGATIVE CONTROL: a duplicate (gate, checkpoint) is refused, in one file and across files', () => {
    const dup = parseToml(`${SAMPLE}\n[[benchmarks]]\ncheckpoint = "unsloth/Qwen3.8-27B-NVFP4"\ngate = "concurrency-sweep"\n[benchmarks.metrics.c1_aggregate_tok_s]\nmin = 1\n`);
    expect(() => declaredLimitsOf(dup, 'dup')).toThrow(/declares limits for unsloth\/Qwen3.8-27B-NVFP4 twice/);
    const one = declaredLimitsOf(parsed, 'a');
    expect(() => mergeDeclaredLimits([['a', one], ['b', one]])).toThrow(/already declared elsewhere/);
  });

  test('NEGATIVE CONTROL: a bound that is not a number, or an entry without a gate, is refused', () => {
    const bad = parseToml('[[benchmarks]]\ncheckpoint = "x"\ngate = "g"\n[benchmarks.metrics.m]\nmin = "20"\n');
    expect(() => declaredLimitsOf(bad)).toThrow(/m\.min is not a finite number/);
    const nogate = parseToml('[[benchmarks]]\ncheckpoint = "x"\n[benchmarks.metrics.m]\nmin = 2\n');
    expect(() => declaredLimitsOf(nogate)).toThrow(/lacks a string gate or checkpoint/);
  });

  test('NEGATIVE CONTROL: syntax the reader does not cover is refused, never guessed past', () => {
    expect(() => parseToml('when = 2026-09-20T00:00:00Z\n')).toThrow(/dates are not read/);
    expect(() => parseToml('a = 1 b = 2\n')).toThrow(/expected end of line/);
    expect(() => parseToml('s = "open\n')).toThrow(/unterminated/);
    expect(() => parseToml('a = 1\na = 2\n')).toThrow(/defined twice/);
  });

  test('values: strings with escapes, numbers with underscores, booleans, arrays, inline tables, dotted keys', () => {
    const p = parseToml(
      's = "a\\tb\\u00e9"\nn = 1_000\nf = -2.5e3\nb = false\narr = [1, "two", [3]]\nt = { x = 1, y.z = "q" }\nd.e = 7\n'
    );
    expect(p).toEqual({
      s: 'a\tbé',
      n: 1000,
      f: -2500,
      b: false,
      arr: [1, 'two', [3]],
      t: { x: 1, y: { z: 'q' } },
      d: { e: 7 }
    });
  });
});

describe('the reader on every real BENCH.toml', () => {
  test('there are BENCH.toml files to check', () => {
    expect(benchFiles.length).toBeGreaterThan(0);
  });

  for (const file of benchFiles) {
    test(`${file.slice(KERNELS.length)} parses exactly as Bun.TOML does`, () => {
      const text = readFileSync(file, 'utf8');
      expect(parseToml(text)).toEqual(Bun.TOML.parse(text));
    });
  }

  test('every declared limit is a finite number under a string gate and checkpoint', () => {
    const merged = mergeDeclaredLimits(benchFiles.map((f) => [f, declaredLimitsOf(parseToml(readFileSync(f, 'utf8')), f)]));
    expect(Object.keys(merged).length).toBeGreaterThan(0);
    for (const [gate, byCheckpoint] of Object.entries(merged)) {
      expect(typeof gate).toBe('string');
      for (const limits of Object.values(byCheckpoint)) {
        for (const lim of Object.values(limits)) {
          expect(Object.keys(lim).every((k) => k === 'min' || k === 'max')).toBe(true);
          expect(Object.values(lim).every(Number.isFinite)).toBe(true);
        }
      }
    }
  });
});
