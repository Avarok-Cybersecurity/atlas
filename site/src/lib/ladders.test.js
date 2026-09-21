// SPDX-License-Identifier: AGPL-3.0-only
//
// The committed ladders against their receipts.
//
// Every number in ladders.generated.json must be recomputable from the raw
// harness file its rung names — by a SECOND implementation of the statistics,
// written here, not imported from the generator. A transcribed number is how
// a chart quietly stops matching its receipts; this is where it would show.
//
// The MoE block pins what was communicated for the 2026-09-19 vLLM one-shot:
// which rungs exist, which are absent and why, which harness copy ran, which
// image, and which kernel was forced. Those are the facts a reader comparing
// engines is owed, so they are pinned as facts rather than trusted as prose.
import { describe, expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ladders from './ladders.generated.json';
import ladder from './ladder.generated.json';
import subjects from './concurrency-subjects.json';
import { buildLadder } from '../../scripts/lib/ladder-build.mjs';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));
const hash = (algo, p) => createHash(algo).update(readFileSync(p)).digest('hex');
const manifestPathOf = (s) => resolve(REPO, s.published_manifest);
const rawReader = (s) => (file) => readJson(join(dirname(manifestPathOf(s)), file));
const withManifest = subjects.filter((s) => s.published_manifest !== null);
const strip = ({ generated_utc, ...rest }) => rest;

// Independent statistics.
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
const round2 = (v) => Math.round(v * 100) / 100;

describe('one generated ladder per subject with a manifest', () => {
  test('the keys of ladders.generated.json are exactly the subjects that declare a manifest', () => {
    expect(Object.keys(ladders.subjects).sort()).toEqual(withManifest.map((s) => s.id).sort());
    for (const s of withManifest) {
      expect(ladders.subjects[s.id].manifest).toBe(s.published_manifest);
      expect(ladders.subjects[s.id].workload.checkpoint).toBe(s.checkpoint);
    }
  });

  test('the dense entry deep-equals ladder.generated.json, the file every other consumer reads', () => {
    expect(ladders.subjects['qwen38-27b']).toEqual(strip(ladder));
  });

  test('regenerating from the committed manifests reproduces the committed ladders', () => {
    for (const s of withManifest) {
      const manifest = readJson(manifestPathOf(s));
      const rebuilt = buildLadder(manifest, {
        subject: s,
        rawOf: rawReader(s),
        harnessRepoSha256: hash('sha256', resolve(REPO, manifest.workload.harness))
      });
      expect(rebuilt).toEqual(ladders.subjects[s.id]);
    }
  });
});

describe('every published rung recomputes from the raw file it names', () => {
  const cases = withManifest.flatMap((s) =>
    ladders.subjects[s.id].series.flatMap((series) =>
      series.rungs.map((r) => [s.id, series.id, r.c, r.source, r])
    )
  );
  expect(cases.length).toBeGreaterThan(0);
  const manifests = Object.fromEntries(withManifest.map((s) => [s.id, readJson(manifestPathOf(s))]));

  test.each(cases)('%s / %s C=%i from %s', (subjectId, seriesId, c, source, r) => {
    const s = withManifest.find((x) => x.id === subjectId);
    const doc = rawReader(s)(source);
    const rung = doc.rungs.find((x) => x.concurrency === c);
    const tok = rung.reps.map((x) => x.tok_s);
    expect(r.reps).toBe(rung.reps.length);
    expect(r.tok_s).toBe(round2(mean(tok)));
    expect(r.tok_s_median).toBe(round2(median(tok)));
    expect(r.spread_pct).toBe(round2(((Math.max(...tok) - Math.min(...tok)) / mean(tok)) * 100));
    expect(r.ttft_p50_ms).toBe(round2(median(rung.reps.map((x) => x.ttft_p50_ms))));
    expect(r.tpot_p50_ms).toBe(round2(median(rung.reps.map((x) => x.tpot_p50_ms))));
    expect(rung.reps.reduce((a, x) => a + x.n_err, 0)).toBe(0);
    expect(r.measured_utc).toBe(doc.started_utc);
    expect(doc.driver_sha256.startsWith(r.harness_sha256)).toBe(true);
    expect(Object.keys(manifests[subjectId].harness_shas)).toContain(r.harness_sha256);
    // The raw header is the instrument the manifest claims.
    const w = manifests[subjectId].workload;
    expect([doc.model, doc.isl, doc.osl, doc.reps, doc.warmup, doc.seed]).toEqual([
      w.checkpoint, w.isl_tokens, w.osl_tokens, w.reps, w.warmup, w.seed
    ]);
  });
});

describe('the MoE one-shot of 2026-09-19', () => {
  const subject = subjects.find((s) => s.id === 'qwen36-35b-a3b');
  const moe = ladders.subjects['qwen36-35b-a3b'];
  const vllm = moe.series.find((s) => s.id === 'vllm-mtp');
  const raw = rawReader(subject)(vllm.rungs[0].source);
  const dense = ladder.series.find((s) => s.id === 'vllm-mtp');

  test('is a baseline-only ladder: no subject series, no rows, no summary, no zero anywhere', () => {
    expect(moe.series.map((s) => [s.id, s.role])).toEqual([['vllm-mtp', 'baseline']]);
    expect('rows' in moe).toBe(false);
    expect('summary' in moe).toBe(false);
    expect(moe.subject_note).toMatch(/no Atlas run exists at this instrument/i);
    expect(vllm.rungs.every((r) => r.tok_s > 0)).toBe(true);
  });

  test('exactly C=1..16 from one raw file; 32/64/128 are absent from the ladder AND the raw file, with the reason', () => {
    expect(moe.concurrencies).toEqual([1, 2, 4, 8, 16]);
    expect(new Set(vllm.rungs.map((r) => r.source)).size).toBe(1);
    expect(vllm.unmeasured.rungs).toEqual([32, 64, 128]);
    expect(vllm.unmeasured.reason).toMatch(/powercycle/);
    expect(raw.rungs.map((r) => r.concurrency)).toEqual([1, 2, 4, 8, 16]);
    for (const c of vllm.unmeasured.rungs) expect(vllm.rungs.find((r) => r.c === c)).toBeUndefined();
  });

  test('the numbers communicated on 2026-09-19, as the raw file yields them', () => {
    expect(vllm.rungs.map((r) => [r.c, r.tok_s, r.spread_pct, r.reps])).toEqual([
      [1, 52.11, 10.93, 3],
      [2, 93.16, 4.96, 3],
      [4, 145.94, 3.33, 3],
      [8, 222.08, 6.44, 3],
      [16, 329.93, 0.34, 3]
    ]);
    expect(raw.rungs.reduce((a, r) => a + r.errors_total, 0)).toBe(0);
    expect(raw.reps).toBe(3);
    expect(raw.warmup).toBe(1);
  });

  test('measured once: every rung carries the same 2026-09-19 stamp', () => {
    expect(new Set(vllm.rungs.map((r) => r.measured_utc))).toEqual(new Set([raw.started_utc]));
    expect(raw.started_utc.slice(0, 10)).toBe('2026-09-19');
    expect(raw.finished_utc.slice(0, 10)).toBe('2026-09-19');
  });

  test('the harness is the committed copy: raw driver sha == sha256 of the tree file == the generated repo hash', () => {
    const tree = resolve(REPO, moe.workload.harness);
    expect(raw.driver_sha256).toBe(hash('sha256', tree));
    expect(moe.harness_repo_sha256).toBe(raw.driver_sha256);
    expect(hash('md5', tree)).toBe('4aa694d8aabe404edc8f1e3eb2acbf31');
    expect(Object.keys(moe.harness_shas)).toEqual([raw.driver_sha256.slice(0, 10), 'equivalence']);
    expect(moe.harness_shas.equivalence).toMatch(/replaced/);
  });

  test('the instrument the fingerprint compares: read from the raw header, spelled as a gate record spells it', () => {
    expect(vllm.instrument).toEqual({
      isl: raw.isl, osl: raw.osl, reps: raw.reps, warmup: raw.warmup, temperature: raw.temperature, seed: raw.seed,
      prompt_mode: 'essay',
      max_model_len: 2048, max_batch_size: 128, kv_cache_dtype: 'bf16'
    });
    // Declared 2026-09-20 (owner decision). It is DERIVED, not recorded: the
    // raw file carries isl, osl, reps, warmup, temperature, seed and
    // chat_template_kwargs but NOT prompt_mode, so this asserts the value the
    // pinned harness defaults to (harness_w55_conc_ladder.py:75) given that
    // this series' recorded env does not set W55_PROMPT_MODE. It exists so a
    // future concurrency-sweep-moe record on this same instrument can pair
    // with this bar -- ladder-baselines.js counts an undeclared axis as a
    // DIFFERENCE, never a match. Re-check if the raw file is ever replaced.
    expect(vllm.instrument.prompt_mode).toBe('essay');
    expect(vllm.cli).toContain('--max-model-len 2048 --max-num-seqs 128 --gpu-memory-utilization 0.85');
    expect(vllm.cli).toContain('--dtype bfloat16 --kv-cache-dtype auto');
    expect(vllm.cli).toContain('--speculative-config \'{"method":"mtp","num_speculative_tokens":3}\'');
    expect(vllm.instrument_note).toMatch(/auto.*bfloat16/s);
    expect(raw.chat_template_kwargs).toEqual({ enable_thinking: false });
    expect(raw.temperature).toBe(0);
    expect(raw.seed).toBe(42);
  });

  test('the engine and image are recorded, and the forced kernel is provenance, not a footnote', () => {
    expect(vllm.engine).toBe('vLLM 0.27.1');
    expect(vllm.build).toBe(dense.build); // the same image digest the dense legs ran, read on the box
    expect(vllm.build_note).toMatch(/docker image inspect/);
    expect(vllm.env).toContain('VLLM_USE_DEEP_GEMM=0');
    expect(vllm.env).toContain('VLLM_TEST_FORCE_FP8_MARLIN=1');
    expect(vllm.kernel_override.env).toBe('VLLM_USE_DEEP_GEMM=0 VLLM_TEST_FORCE_FP8_MARLIN=1');
    expect(vllm.cli).toContain('-e VLLM_USE_DEEP_GEMM=0 -e VLLM_TEST_FORCE_FP8_MARLIN=1');
    expect(vllm.kernel_override.error).toContain('layout.hpp:60');
    expect(vllm.kernel_override.error).toContain('Unknown SF transformation');
    expect(vllm.kernel_override.what).toMatch(/NOT running its own default/);
    expect(moe.box.name).toBe('dgx2 (spark-43fa)');
    expect(moe.workload.checkpoint).toBe('Qwen/Qwen3.6-35B-A3B-FP8');
  });

  test('the raw file is the one the manifest names, byte for byte', () => {
    const file = join(dirname(manifestPathOf(subject)), vllm.rungs[0].source);
    expect(vllm.source_note).toContain(hash('sha256', file));
  });
});
