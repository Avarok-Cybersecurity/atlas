// SPDX-License-Identifier: AGPL-3.0-only

import { describe, expect, test } from 'bun:test';
import * as IO from './iostrip.js';

const LIVE = { paused: false };

describe('absent is never zero, and the kinds of absence stay distinct', () => {
  test('a field missing from an arrived reading dashes, never renders 0', () => {
    const t = IO.tiles({ requests_total: 12, window_s: 2 }, LIVE);
    const decode = t.find((x) => x.id === 'decode');
    expect(decode.kind).toBe('absent');
    expect(decode.text).toBe('—');
    // The one field that WAS carried still shows.
    expect(t.find((x) => x.id === 'requests-total').text).toBe('12');
  });

  test('a wire-carried zero renders as zero', () => {
    const t = IO.tiles({ decode_tokens_per_s: 0, requests_active: 0, window_s: 2 }, LIVE);
    expect(t.find((x) => x.id === 'decode')).toMatchObject({ kind: 'reading', text: '0.00' });
    expect(t.find((x) => x.id === 'requests-active')).toMatchObject({ kind: 'reading', text: '0' });
  });

  test('no reading yet means pending, which is not the dash', () => {
    const t = IO.tiles(null, LIVE);
    for (const tile of t.filter((x) => x.kind !== 'placeholder')) {
      expect(tile.kind).toBe('pending');
      expect(tile.text).toBeUndefined();
    }
  });

  test('the strip no longer carries a placeholder tile', () => {
    // ISL and OSL were the only two, and the agent emits them now. If one comes
    // back as a promise, it must come back deliberately — with a value slot
    // that stays empty rather than an em-dash, which would be a claim.
    for (const reading of [null, { decode_tokens_per_s: 5, window_s: 2 }]) {
      expect(IO.tiles(reading, LIVE).filter((x) => x.kind === 'placeholder')).toEqual([]);
    }
  });

  test('ISL and OSL read from the agent, and absence means no request finished', () => {
    const live = IO.tiles(
      { isl_mean: 512, osl_mean: 128, window_s: 4 },
      LIVE
    );
    const isl = live.find((t) => t.id === 'isl');
    const osl = live.find((t) => t.id === 'osl');
    expect(isl.kind).toBe('reading');
    expect(isl.text).toContain('512');
    expect(osl.text).toContain('128');

    // Tokens can be flowing the whole window while nothing FINISHES — a long
    // request accrues them without completing — so the note must not say "no
    // traffic", which would be a different and wrong claim.
    const none = IO.tiles({ decode_tokens_per_s: 40, window_s: 4 }, LIVE);
    const absent = none.find((t) => t.id === 'isl');
    expect(absent.kind).toBe('absent');
    expect(absent.text).toBe('—');
    expect(absent.note).toBe('no request finished');
  });

  test('absence with a known meaning says it', () => {
    const t = IO.tiles({ decode_tokens_per_s: 5, window_s: 2 }, LIVE);
    expect(t.find((x) => x.id === 'accept').note).toBe('not speculating');
  });

  test('the tile set is fixed: telemetry arriving never reflows the strip', () => {
    const ids = (r) => IO.tiles(r, LIVE).map((x) => x.id);
    expect(ids(null)).toEqual(ids({ decode_tokens_per_s: 5, window_s: 2 }));
  });
});

describe('no rate on first poll, none across a gap', () => {
  const reading = {
    decode_tokens_per_s: 40,
    prompt_tokens_per_s: 900,
    requests_total: 7,
    ttft_p50_s: 1.4,
    window_s: 2
  };

  test('the first poll surrenders its rates — the window predates this page', () => {
    const g = IO.gateRates(reading, { firstPoll: true });
    expect(g.decode_tokens_per_s).toBeUndefined();
    expect(g.prompt_tokens_per_s).toBeUndefined();
    // Counters and percentiles are statements about real requests; they stay.
    expect(g.requests_total).toBe(7);
    expect(g.ttft_p50_s).toBe(1.4);
  });

  test('a window longer than the gap limit is a fabricated rate', () => {
    const g = IO.gateRates({ ...reading, window_s: IO.MAX_RATE_WINDOW_S + 1 }, { firstPoll: false });
    expect(g.decode_tokens_per_s).toBeUndefined();
    expect(g.requests_total).toBe(7);
  });

  test('a reading that will not say its window fails closed', () => {
    const { window_s: _, ...unwindowed } = reading;
    const g = IO.gateRates(unwindowed, { firstPoll: false });
    expect(g.decode_tokens_per_s).toBeUndefined();
  });

  test('a second poll inside the limit keeps its rates', () => {
    expect(IO.gateRates(reading, { firstPoll: false })).toEqual(reading);
  });

  test('the gate must be told, not left to assume', () => {
    expect(() => IO.gateRates(reading, {})).toThrow(TypeError);
    expect(() => IO.gateRates(reading)).toThrow(TypeError);
  });
});

describe('strip mode: nothing serving vs all fields absent vs not answering', () => {
  test('the five modes classify without overlap', () => {
    expect(IO.mode({ serving: false, reading: null, failure: null })).toBe('off');
    expect(IO.mode({ serving: true, reading: null, failure: null })).toBe('pending');
    expect(IO.mode({ serving: true, reading: null, failure: 'timed out' })).toBe('unanswered');
    expect(IO.mode({ serving: true, reading: {}, failure: null })).toBe('quiet');
    expect(IO.mode({ serving: true, reading: { requests_total: 0 }, failure: null })).toBe('live');
  });

  test('"don\'t know whether anything is serving" is refused, not defaulted', () => {
    expect(() => IO.mode({ reading: null, failure: null })).toThrow(TypeError);
  });
});

describe('caption composition', () => {
  test('window and provenance compose', () => {
    expect(IO.caption({ window_s: 2.5 }, { via: null })).toBe('measured over 2.50 s');
    expect(IO.caption({ window_s: 2.5 }, { via: 'dgx1' })).toBe('measured over 2.50 s · via dgx1');
    expect(IO.caption(null, { via: 'dgx1' })).toBe('via dgx1');
  });

  test('no window means no freshness claim', () => {
    expect(IO.caption({}, { via: null })).toBe('');
    expect(IO.caption(null, { via: null })).toBe('');
  });

  test('a relay name is sanitised before it reaches the strip', () => {
    // A bidi override in a relay name could reorder the caption around it.
    expect(IO.caption(null, { via: 'dgx\u202e1' })).toBe('via dgx1');
  });
});

describe('paused state rides every real tile', () => {
  test('tiles say paused when polling is paused, and placeholders do not care', () => {
    const t = IO.tiles({ decode_tokens_per_s: 5, window_s: 2 }, { paused: true });
    for (const tile of t.filter((x) => x.kind !== 'placeholder')) expect(tile.paused).toBe(true);
  });

  test('the flag is required', () => {
    expect(() => IO.tiles(null, {})).toThrow(TypeError);
  });
});
