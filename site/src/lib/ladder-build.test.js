// SPDX-License-Identifier: AGPL-3.0-only
//
// Every guard in scripts/lib/ladder-build.mjs, broken on purpose.
//
// The fixtures are the COMMITTED manifests and raw files, deep-cloned and
// mutated one field at a time, so each case is an input the real generator
// could be handed. A guard that is never seen to go red is a guard nobody
// knows is wired; every case below is one that would otherwise let a typed
// number, a phantom rung, an unlisted harness revision or a mismatched
// instrument reach the chart.
import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import subjects from './concurrency-subjects.json';
import { buildLadder, cliFlag, r2, r3 } from '../../scripts/lib/ladder-build.mjs';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));
const clone = (x) => JSON.parse(JSON.stringify(x));

/** A subject's manifest and raw files, loaded once, cloned per case. */
function fixture(id) {
  const subject = subjects.find((s) => s.id === id);
  const dir = dirname(resolve(REPO, subject.published_manifest));
  const manifest = readJson(resolve(REPO, subject.published_manifest));
  const raws = {};
  for (const s of manifest.series) for (const f of Object.values(s.sources)) raws[f] ??= readJson(join(dir, f));
  return { subject, manifest, raws };
}
const MOE = fixture('qwen36-35b-a3b');
const DENSE = fixture('qwen38-27b');

/** Build from a mutated copy: `mut(manifest, raws, subject)` edits in place. */
function build(fix, mut) {
  const manifest = clone(fix.manifest);
  const raws = clone(fix.raws);
  const subject = clone(fix.subject);
  mut?.(manifest, raws, subject);
  return buildLadder(manifest, {
    subject,
    rawOf: (file) => {
      if (!(file in raws)) throw new Error(`cannot read raw ${file}`);
      return raws[file];
    },
    harnessRepoSha256: 'f'.repeat(64)
  });
}
const vllm = (m) => m.series.find((s) => s.id === 'vllm-mtp');
const RAW = 'vllm_moe_c1_16.json';

describe('the committed inputs build', () => {
  test('MoE: a baseline-only ladder with five rungs and no scored pair', () => {
    const l = build(MOE);
    expect(l.concurrencies).toEqual([1, 2, 4, 8, 16]);
    expect(l.rows).toBeUndefined();
    expect(l.summary).toBeUndefined();
    expect(l.harness_repo_sha256).toBe('f'.repeat(64));
  });

  test('dense: a scored pair, 8 rungs, the matched baseline as the denominator', () => {
    const l = build(DENSE);
    expect(l.summary.rungs).toBe(8);
    expect(l.rows.every((r) => r.best_baseline_id === 'vllm-mtp')).toBe(true);
  });
});

describe('typed numbers are refused', () => {
  test.each(['rungs', 'tok_s', 'rows', 'summary'])('a series carrying "%s"', (k) => {
    expect(() => build(MOE, (m) => (vllm(m)[k] = [{ c: 1, tok_s: 52.11 }]))).toThrow(new RegExp(`typed "${k}"`));
  });
  test.each(['rows', 'summary'])('a manifest carrying top-level "%s"', (k) => {
    expect(() => build(MOE, (m) => (m[k] = []))).toThrow(new RegExp(`typed "${k}"`));
  });
});

describe('the raw file must prove the manifest', () => {
  test('a source that does not exist', () => {
    expect(() => build(MOE, (m) => (vllm(m).sources['1'] = 'nope.json'))).toThrow(/cannot read raw nope.json/);
  });
  test('a source without the rung it is named for', () => {
    expect(() => build(MOE, (m) => (vllm(m).sources['32'] = RAW))).toThrow(/has no rung for C=32/);
  });
  test('a rung with no reps', () => {
    expect(() => build(MOE, (m, raws) => (raws[RAW].rungs[0].reps = []))).toThrow(/C=1 has no reps/);
  });
  test('a request error anywhere in a rung', () => {
    expect(() => build(MOE, (m, raws) => (raws[RAW].rungs[2].reps[1].n_err = 1))).toThrow(/C=4 recorded 1 request errors/);
  });
  test('a non-numeric tok_s', () => {
    expect(() => build(MOE, (m, raws) => (raws[RAW].rungs[0].reps[0].tok_s = '52.11'))).toThrow(/non-numeric tok_s/);
  });
  test.each([
    ['model', 'other/checkpoint', 'checkpoint'],
    ['isl', 512, 'isl_tokens'],
    ['osl', 320, 'osl_tokens'],
    ['reps', 5, 'reps'],
    ['warmup', 0, 'warmup'],
    ['temperature', 0.7, 'temperature'],
    ['seed', 0, 'seed']
  ])('a raw header whose %s disagrees with the workload', (k, v, wl) => {
    expect(() => build(MOE, (m, raws) => (raws[RAW][k] = v))).toThrow(new RegExp(`header ${k}=.* != workload\\.${wl}`));
  });
  test('a raw file without a date or a driver sha cannot be stamped', () => {
    expect(() => build(MOE, (m, raws) => delete raws[RAW].started_utc)).toThrow(/no started_utc/);
    expect(() => build(MOE, (m, raws) => delete raws[RAW].driver_sha256)).toThrow(/no driver_sha256/);
  });
  test('the subject and the manifest must name one checkpoint', () => {
    expect(() => build(MOE, (m, raws, subject) => (subject.checkpoint = 'unsloth/Qwen3.8-27B-NVFP4'))).toThrow(
      /is for Qwen\/Qwen3.6-35B-A3B-FP8, subject qwen36-35b-a3b is unsloth\/Qwen3.8-27B-NVFP4/
    );
  });
});

describe('harness_shas is exactly the set of revisions the raw files carry', () => {
  test('an unlisted revision — the dgx2 stale-copy failure this exists to catch', () => {
    expect(() => build(MOE, (m, raws) => (raws[RAW].driver_sha256 = 'deadbeef00' + 'a'.repeat(54)))).toThrow(
      /harness revision deadbeef00 produced vllm_moe_c1_16.json but is not listed/
    );
    expect(() => build(MOE, (m) => delete m.harness_shas['41e242c072'])).toThrow(/41e242c072 produced .* but is not listed/);
  });
  test('a listed revision no raw file carries is a phantom', () => {
    expect(() => build(MOE, (m) => (m.harness_shas['0000000000'] = 'imaginary'))).toThrow(/lists 0000000000 but no raw file/);
  });
  test('the dense manifest lists all three revisions its files carry, and no fourth', () => {
    expect(() => build(DENSE, (m) => delete m.harness_shas['1f10d4887b'])).toThrow(/1f10d4887b produced c2_atlas_dgx2_20260818.json, c2_vllm_mtp_dgx2_20260818.json/);
  });
});

describe('an unmeasured rung is absent everywhere', () => {
  test('listed as unmeasured but present in the raw file', () => {
    expect(() => build(MOE, (m) => {
      delete vllm(m).sources['16'];
      vllm(m).unmeasured.rungs = [16];
    })).toThrow(/C=16 as unmeasured but vllm_moe_c1_16.json contains it/);
  });
  test('listed as unmeasured but also sourced', () => {
    expect(() => build(MOE, (m) => vllm(m).unmeasured.rungs.push(1))).toThrow(/C=1 as unmeasured but sources names a file/);
  });
  test('without a reason, or with a non-rung', () => {
    expect(() => build(MOE, (m) => (vllm(m).unmeasured.reason = ' '))).toThrow(/unmeasured.reason is required/);
    expect(() => build(MOE, (m) => (vllm(m).unmeasured.rungs = [32, 0]))).toThrow(/unmeasured.rungs must be/);
  });
});

describe('a baseline declares the instrument the fingerprint compares', () => {
  test('no instrument at all', () => {
    expect(() => build(MOE, (m) => delete vllm(m).instrument)).toThrow(/declares no instrument/);
  });
  test.each([['isl', 512], ['osl', 320], ['reps', 1], ['warmup', 0], ['temperature', 1], ['seed', 7]])(
    'instrument.%s disagreeing with the workload',
    (k, v) => {
      expect(() => build(MOE, (m) => (vllm(m).instrument[k] = v))).toThrow(new RegExp(`instrument\\.${k}=${v} != workload`));
    }
  );
  test.each(['max_model_len', 'max_batch_size', 'kv_cache_dtype'])('instrument.%s undeclared', (k) => {
    expect(() => build(MOE, (m) => delete vllm(m).instrument[k])).toThrow(new RegExp(`instrument\\.${k} is undeclared`));
  });
  test('a batch cap or context the command line does not carry', () => {
    expect(() => build(MOE, (m) => (vllm(m).instrument.max_batch_size = 64))).toThrow(/max_batch_size=64 but its cli says 128/);
    expect(() => build(MOE, (m) => (vllm(m).instrument.max_model_len = 4096))).toThrow(/max_model_len=4096 but its cli says 2048/);
    expect(() => build(MOE, (m) => (vllm(m).cli = 'vllm serve'))).toThrow(/max_batch_size=128 but its cli says nothing/);
  });
  test('a KV dtype spelled differently from the command needs its note', () => {
    expect(() => build(MOE, (m) => delete vllm(m).instrument_note)).toThrow(/kv_cache_dtype=bf16 but its cli says auto; add instrument_note/);
    // With the command spelling the same value, no note is needed.
    expect(() => build(MOE, (m) => {
      delete vllm(m).instrument_note;
      vllm(m).cli = vllm(m).cli.replace('--kv-cache-dtype auto', '--kv-cache-dtype bf16');
    })).not.toThrow();
  });
});

describe('a pair is scored only when it is whole', () => {
  test('no matched-parity baseline', () => {
    expect(() => build(DENSE, (m) => (vllm(m).parity = 'unmatched'))).toThrow(/no matched-parity baseline/);
  });
  test('a baseline missing a subject rung', () => {
    expect(() => build(DENSE, (m) => delete vllm(m).sources['64'])).toThrow(/baseline vllm-mtp is missing rung C=64/);
  });
  test('a second subject, an unknown role, an empty series list', () => {
    expect(() => build(DENSE, (m) => (m.series[2].role = 'subject'))).toThrow(/more than one subject/);
    expect(() => build(MOE, (m) => (vllm(m).role = 'reference'))).toThrow(/unknown role "reference"/);
    expect(() => build(MOE, (m) => (m.series = []))).toThrow(/has no series/);
    expect(() => build(MOE, (m) => (vllm(m).role = 'variant'))).toThrow(/no baseline series/);
    expect(() => build(MOE, (m) => (m.schema = 2))).toThrow(/schema 2, expected 1/);
  });
});

describe('helpers', () => {
  test('cliFlag reads the first matching flag, whole-token only', () => {
    expect(cliFlag('x --max-num-seqs 128 --kv-cache-dtype auto', '--max-num-seqs')).toBe('128');
    expect(cliFlag('x --max-batch-size 16', '--max-num-seqs', '--max-batch-size')).toBe('16');
    expect(cliFlag('x --max-num-seqs-extra 9', '--max-num-seqs')).toBeNull();
    expect(cliFlag(undefined, '--x')).toBeNull();
  });
  test('rounding keeps the decimals the site prints', () => {
    expect(r2(52.11033488999998)).toBe(52.11);
    expect(r3(1.0044)).toBe(1.004);
  });
});
