import { describe, expect, it } from 'bun:test';
import {
  dashFor,
  groupFor,
  groupRecords,
  groupedBenches,
  isLatestOfVariant,
  splitByVariant,
  variantLabel
} from './gate-variants.js';

const rec = (benchmark_id, recorded_at, extra = {}) => ({
  benchmark_id,
  recorded_at,
  target_model: 'unsloth/Qwen3.8-27B-NVFP4',
  metrics: {},
  ...extra
});

describe('the concurrency group', () => {
  it('claims both concurrency gates and nothing else', () => {
    expect(groupFor('concurrency-sweep')?.tab).toBe('concurrency');
    expect(groupFor('concurrency-sweep-dflash2')?.tab).toBe('concurrency');
    expect(groupFor('decode-floor')).toBeNull();
    expect(groupFor('bfcl-subset')).toBeNull();
    expect(groupedBenches.has('concurrency-sweep-dflash2')).toBe(true);
    expect(groupedBenches.has('decode-floor')).toBe(false);
  });

  it('distinguishes the two by dash and never by colour', () => {
    // The palette is keyed by MODEL and both members serve one checkpoint, so
    // a second hue would assert a second model. The dash carries the whole
    // difference — if this ever flips, series-contrast.test.js is not the
    // test that would catch it.
    expect(dashFor('concurrency-sweep')).toBeNull();
    expect(dashFor('concurrency-sweep-dflash2')).toBe('5 4');
    expect(variantLabel('concurrency-sweep-dflash2')).toBe('DFlash2');
    expect(variantLabel('decode-floor')).toBeNull();
  });
});

describe('splitByVariant', () => {
  it('never puts two variants in one bucket', () => {
    // The defect this prevents: interleaved records joined into one polyline
    // read as a regression and a recovery at every alternation.
    const records = [
      rec('concurrency-sweep', 10),
      rec('concurrency-sweep-dflash2', 20),
      rec('concurrency-sweep', 30),
      rec('concurrency-sweep-dflash2', 40)
    ];
    const buckets = splitByVariant(records);
    expect(buckets).toHaveLength(2);
    expect(buckets[0].bench).toBe('concurrency-sweep');
    expect(buckets[0].records.map((r) => r.recorded_at)).toEqual([10, 30]);
    expect(buckets[1].bench).toBe('concurrency-sweep-dflash2');
    expect(buckets[1].records.map((r) => r.recorded_at)).toEqual([20, 40]);
  });

  it('drops a variant that has no records rather than drawing an empty line', () => {
    const buckets = splitByVariant([rec('concurrency-sweep', 1)]);
    expect(buckets).toHaveLength(1);
    expect(buckets[0].dash).toBeNull();
  });

  it('leaves an ungrouped benchmark exactly as it was', () => {
    const buckets = splitByVariant([rec('decode-floor', 1), rec('decode-floor', 2)]);
    expect(buckets).toHaveLength(1);
    expect(buckets[0].bench).toBeNull();
    expect(buckets[0].dash).toBeNull();
    expect(buckets[0].records).toHaveLength(2);
  });
});

describe('isLatestOfVariant', () => {
  // The bug a global "last element" would cause: the newest DFlash2 run is
  // usually not the newest run overall, so one whole variant would render
  // permanently faded and read as stale.
  const records = [
    rec('concurrency-sweep', 10),
    rec('concurrency-sweep-dflash2', 20),
    rec('concurrency-sweep', 30)
  ];

  it('marks the newest of each variant, not the newest overall', () => {
    expect(isLatestOfVariant(records[2], records)).toBe(true);
    expect(isLatestOfVariant(records[1], records)).toBe(true);
    expect(isLatestOfVariant(records[0], records)).toBe(false);
  });
});

describe('groupRecords', () => {
  it('merges the members chronologically', () => {
    const byBench = {
      'concurrency-sweep': [rec('concurrency-sweep', 30), rec('concurrency-sweep', 10)],
      'concurrency-sweep-dflash2': [rec('concurrency-sweep-dflash2', 20)]
    };
    const merged = groupRecords(groupFor('concurrency-sweep'), (b) => byBench[b] ?? []);
    expect(merged.map((r) => r.recorded_at)).toEqual([10, 20, 30]);
  });

  it('survives a member with no records at all', () => {
    const merged = groupRecords(groupFor('concurrency-sweep'), (b) =>
      b === 'concurrency-sweep' ? [rec('concurrency-sweep', 1)] : []
    );
    expect(merged).toHaveLength(1);
  });
});
