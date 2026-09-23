// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import {
  SUBJECTS,
  assertSubjects,
  formatUnassigned,
  parseConcurrencies,
  recordsOf,
  rungsDeclared,
  subjectOf,
  unassignedRecords
} from './concurrency-subjects.js';

const DENSE = 'unsloth/Qwen3.8-27B-NVFP4';
const MOE = 'Qwen/Qwen3.6-35B-A3B-FP8';

const rec = (benchmark_id, target_model, git_sha, concurrencies) => ({
  benchmark_id,
  target_model,
  git_sha,
  params: concurrencies === undefined ? {} : { concurrencies }
});

/** A `recordsFor` over a fixed bench → records map, chronological as given. */
const store = (byBench) => (bench) => byBench[bench] ?? [];

const byId = (id) => SUBJECTS.find((s) => s.id === id);

describe('the subject list', () => {
  test('is read from the repo-root JSON, unchanged, in the owner\'s order', () => {
    const onDisk = JSON.parse(readFileSync(new URL('./concurrency-subjects.json', import.meta.url), 'utf8'));
    expect(SUBJECTS).toEqual(onDisk);
    expect(SUBJECTS.map((s) => s.id)).toEqual(['qwen38-27b', 'qwen36-35b-a3b', 'qwen38-27b-dflash']);
  });

  test('the MoE subject is the FP8 checkpoint the records carry, never an NVFP4 35B', () => {
    // BENCH.toml says `quant = "nvfp4"` for this entry; the checkpoint id is
    // what the record carries and what the tab must print.
    expect(byId('qwen36-35b-a3b').checkpoint).toBe(MOE);
    expect(SUBJECTS.some((s) => /35B.*NVFP4/i.test(s.checkpoint))).toBe(false);
  });

  test('the MoE has its own gate id; dense and DFlash share a checkpoint', () => {
    // A required gate has ONE declared subject per box class (`record_is_required_subject`
    // in crates/avarok-plugin/src/gate/check.rs), so the MoE ladder cannot be a second
    // subject of `concurrency-sweep`: it is `concurrency-sweep-moe`, the mechanism the
    // DFlash2 ladder already uses, and its records land in their own directory.
    expect(byId('qwen38-27b').gate).toBe('concurrency-sweep');
    expect(byId('qwen36-35b-a3b').gate).toBe('concurrency-sweep-moe');
    expect(byId('qwen38-27b').checkpoint).toBe(byId('qwen38-27b-dflash').checkpoint);
  });
});

describe('assertSubjects', () => {
  const good = () => structuredClone(SUBJECTS);

  test('accepts the shipped list and returns it', () => {
    expect(assertSubjects(good())).toEqual(SUBJECTS);
  });

  test('refuses a duplicate id', () => {
    const dup = good();
    dup[1].id = dup[0].id;
    expect(() => assertSubjects(dup)).toThrow(/duplicate id qwen38-27b/);
  });

  test.each(['id', 'label', 'checkpoint', 'gate', 'baselines_dir'])('refuses a missing or empty %s', (k) => {
    const missing = good();
    delete missing[2][k];
    expect(() => assertSubjects(missing)).toThrow(new RegExp(`"${k}" must be a non-empty string`));
    const empty = good();
    empty[2][k] = '';
    expect(() => assertSubjects(empty)).toThrow(new RegExp(`"${k}" must be a non-empty string`));
  });

  test('published_manifest is a path or an explicit null, never undefined', () => {
    const undef = good();
    delete undef[0].published_manifest;
    expect(() => assertSubjects(undef)).toThrow(/published_manifest/);
  });

  test('refuses an empty or non-array list', () => {
    expect(() => assertSubjects([])).toThrow(/non-empty array/);
    expect(() => assertSubjects({})).toThrow(/non-empty array/);
  });
});

describe('subjectOf', () => {
  test('assigns on gate AND checkpoint — the two shared fields cannot mislead it', () => {
    expect(subjectOf(rec('concurrency-sweep', DENSE, 'a'))?.id).toBe('qwen38-27b');
    // The MoE's own gate id with its checkpoint: MoE.
    expect(subjectOf(rec('concurrency-sweep-moe', MOE, 'b'))?.id).toBe('qwen36-35b-a3b');
    // Same checkpoint as dense, different gate: DFlash, not dense.
    expect(subjectOf(rec('concurrency-sweep-dflash2', DENSE, 'c'))?.id).toBe('qwen38-27b-dflash');
  });

  test('a record no subject claims is null, not the nearest match', () => {
    expect(subjectOf(rec('concurrency-sweep', 'nvidia/Qwen3.6-35B-A3B-NVFP4', 'd'))).toBeNull();
    expect(subjectOf(rec('concurrency-sweep-dflash2', MOE, 'e'))).toBeNull();
    expect(subjectOf(rec('decode-floor', DENSE, 'f'))).toBeNull();
    // A MoE run filed under the DENSE gate id is unassigned — listed in the
    // footer, never quietly folded into the MoE tab as if the gate were shared.
    expect(subjectOf(rec('concurrency-sweep', MOE, 'g'))).toBeNull();
    // And the dense checkpoint under the MoE gate id is not the dense subject.
    expect(subjectOf(rec('concurrency-sweep-moe', DENSE, 'h'))).toBeNull();
  });
});

describe('recordsOf', () => {
  const recordsFor = store({
    'concurrency-sweep': [rec('concurrency-sweep', DENSE, 'd1'), rec('concurrency-sweep', MOE, 'stray'), rec('concurrency-sweep', DENSE, 'd2')],
    'concurrency-sweep-moe': [rec('concurrency-sweep-moe', MOE, 'm1')],
    'concurrency-sweep-dflash2': [rec('concurrency-sweep-dflash2', DENSE, 'f1')]
  });

  test('keeps only the subject\'s own records, in the order given', () => {
    expect(recordsOf(byId('qwen38-27b'), recordsFor).map((r) => r.git_sha)).toEqual(['d1', 'd2']);
    expect(recordsOf(byId('qwen36-35b-a3b'), recordsFor).map((r) => r.git_sha)).toEqual(['m1']);
    expect(recordsOf(byId('qwen38-27b-dflash'), recordsFor).map((r) => r.git_sha)).toEqual(['f1']);
  });
});

describe('parseConcurrencies', () => {
  test('reads the harness string into sorted, de-duplicated rungs', () => {
    expect(parseConcurrencies('1, 2, 4, 8, 16', 'x')).toEqual([1, 2, 4, 8, 16]);
    expect(parseConcurrencies('16,1,4,4', 'x')).toEqual([1, 4, 16]);
  });

  test.each(['', '1, , 4', '1, 2, four', '0, 1', '-4', '1.5', undefined, null])(
    'refuses %p loudly, naming the record',
    (bad) => {
      expect(() => parseConcurrencies(bad, 'deadbeef')).toThrow(/record deadbeef: params\.concurrencies/);
    }
  );
});

describe('rungsDeclared', () => {
  test('comes from the NEWEST record — the widened instrument, not the narrow one', () => {
    const recordsFor = store({
      'concurrency-sweep': [
        rec('concurrency-sweep', DENSE, 'old', '1, 4, 8, 16'),
        rec('concurrency-sweep', DENSE, 'new', '1, 2, 4, 8, 16, 32, 64, 128')
      ]
    });
    expect(rungsDeclared(byId('qwen38-27b'), recordsFor)).toEqual([1, 2, 4, 8, 16, 32, 64, 128]);
  });

  test('ignores another subject\'s newer record on the same gate', () => {
    const recordsFor = store({
      'concurrency-sweep': [rec('concurrency-sweep', DENSE, 'd', '1, 4'), rec('concurrency-sweep', MOE, 'm', '1, 2, 4, 8')]
    });
    expect(rungsDeclared(byId('qwen38-27b'), recordsFor)).toEqual([1, 4]);
  });

  test('is empty, not an error, for a subject with no records', () => {
    expect(rungsDeclared(byId('qwen36-35b-a3b'), store({}))).toEqual([]);
  });

  test('a newest record without concurrencies is a broken record, not an empty rung set', () => {
    const recordsFor = store({ 'concurrency-sweep': [rec('concurrency-sweep', DENSE, 'bad')] });
    expect(() => rungsDeclared(byId('qwen38-27b'), recordsFor)).toThrow(/record bad/);
  });
});

describe('unassignedRecords', () => {
  test('lists every unclaimed bench · model with its count and nothing that is claimed', () => {
    const stray = 'nvidia/Qwen3.6-35B-A3B-NVFP4';
    const recordsFor = store({
      'concurrency-sweep': [rec('concurrency-sweep', DENSE, 'a'), rec('concurrency-sweep', stray, 'b'), rec('concurrency-sweep', stray, 'c')],
      'concurrency-sweep-dflash2': [rec('concurrency-sweep-dflash2', MOE, 'd')]
    });
    const out = unassignedRecords(['concurrency-sweep', 'concurrency-sweep-dflash2'], recordsFor);
    expect(out).toEqual([
      { bench: 'concurrency-sweep', model: stray, n: 2 },
      { bench: 'concurrency-sweep-dflash2', model: MOE, n: 1 }
    ]);
    expect(out.map(formatUnassigned)).toEqual([
      `concurrency-sweep · ${stray} (2)`,
      `concurrency-sweep-dflash2 · ${MOE} (1)`
    ]);
  });

  test('is empty when every record has a subject', () => {
    const recordsFor = store({
      'concurrency-sweep': [rec('concurrency-sweep', DENSE, 'a')],
      'concurrency-sweep-moe': [rec('concurrency-sweep-moe', MOE, 'b')]
    });
    expect(unassignedRecords(['concurrency-sweep', 'concurrency-sweep-moe'], recordsFor)).toEqual([]);
  });
});
