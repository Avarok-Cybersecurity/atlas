// SPDX-License-Identifier: AGPL-3.0-only
//
// cost.js, measured directly. Every test here traces to one of the four rules
// the module exists to keep:
//
//   * the dollar arithmetic is the physics, checked against a hand-worked
//     fixture rather than against itself;
//   * an ABSENT key is "not measured" and can never become a number — a zero
//     joule count would read as FREE, which is the one claim this must not make;
//   * an under-sampled window is marked and left out of every tile, verdict
//     and trend, but stays visible;
//   * a losing rung is returned exactly like a winning one, and k = 0 flips
//     the headline to name the winner.
import { describe, expect, test } from 'bun:test';
import { instrumentKey } from './ladder-baselines.js';
import {
  CROSS_CHECK_TOLERANCE,
  DEFAULT_USD_PER_KWH,
  KEY,
  MIN_POWER_SAMPLES,
  RUN_KEY,
  baselineSnapshots,
  cellKey,
  costInstrumentKey,
  costLadder,
  costPerMillion,
  costTrend,
  emptyStateOf,
  energyOf,
  extremeRungs,
  fmtUsd,
  jPerToken,
  readEnergy,
  rungVerdicts,
  rungsOf,
  tokPerWh,
  trendMetricKey,
  verdictTile,
  wattsOf,
  MIN_SPREAD_RUNS,
  rungSpread,
  spreadVerdict
} from './cost.js';

// ---- fixtures ---------------------------------------------------------------

const SUBJECT = {
  id: 'qwen38-27b',
  label: 'Qwen 3.8 27B',
  checkpoint: 'unsloth/Qwen3.8-27B-NVFP4',
  gate: 'concurrency-sweep',
  published_manifest: 'bench/ladder38/published.json',
  baselines_dir: 'bench/baselines/qwen38-27b'
};

/**
 * One cell's keys. Defaults are the worked fixture: 80 W at 25 tok/s over a
 * 100 s window is 8000 J for 2500 tokens = 3.2 J/token.
 */
const cell = (c, o = {}) => {
  const { watts = 80, tokS = 25, windowS = 100, samples = 400, ...rest } = o;
  const energyJ = watts * windowS;
  const tokens = tokS * windowS;
  return {
    [`c${c}_aggregate_tok_s`]: tokS,
    [cellKey(c, KEY.energyJ)]: energyJ,
    [cellKey(c, KEY.tokens)]: tokens,
    [cellKey(c, KEY.windowS)]: windowS,
    [cellKey(c, KEY.samples)]: samples,
    [cellKey(c, KEY.meanW)]: watts,
    ...Object.fromEntries(Object.entries(rest).map(([k, v]) => [cellKey(c, KEY[k] ?? k), v]))
  };
};

const rec = (o = {}) => ({
  git_sha: 'abc1234567',
  recorded_at: 1789900000,
  verdict: 'PASS',
  branch: null,
  target_model: SUBJECT.checkpoint,
  benchmark_id: 'concurrency-sweep',
  hardware: { gpu: 'NVIDIA GB10' },
  params: { concurrencies: '1, 8', isls: '512', osl: '320', prompt_mode: 'natural' },
  serve_overrides: { kv_cache_dtype: 'fp8', max_batch_size: '128', max_model_len: '4096' },
  ...o,
  metrics: { [RUN_KEY.periodMs]: 250, ...(o.metrics ?? {}) }
});

/** A one-shot vLLM series on the SAME instrument as `rec()` above. */
const baselineSeries = (rungs, o = {}) => ({
  id: 'vllm-mtp',
  label: 'vLLM + MTP',
  role: 'baseline',
  instrument: {
    isl: 512,
    osl: 320,
    prompt_mode: 'natural',
    max_model_len: 4096,
    max_batch_size: 128,
    kv_cache_dtype: 'fp8'
  },
  rungs,
  ...o
});

/** One rung of a one-shot series, in the shape ladder-build.mjs#rungEnergy writes. */
const baseRung = (c, o = {}) => {
  const { watts = 40, tokS = 20, windowS = 100, samples = 400 } = o;
  return {
    c,
    tok_s: tokS,
    measured_utc: '2026-08-17T12:24:02Z',
    [KEY.energyJ]: watts * windowS,
    [KEY.tokens]: tokS * windowS,
    [KEY.windowS]: windowS,
    [KEY.samples]: samples,
    [RUN_KEY.periodMs]: 250
  };
};

const laddersWith = (series) => ({
  subjects: {
    [SUBJECT.id]: { workload: { checkpoint: SUBJECT.checkpoint }, box: { name: 'dgx2' }, series }
  }
});

// ---- the physics ------------------------------------------------------------

describe('the dollar arithmetic', () => {
  // 80 W at 25 tok/s is 3.2 J/token. A million tokens is 40,000 s (11.1 h) and
  // 3.2e6 J = 0.889 kWh, which at $0.15/kWh is $0.13.
  test('$ per 1M tokens matches the hand-worked fixture', () => {
    const jTok = jPerToken(8000, 2500);
    expect(jTok).toBe(3.2);
    const kwh = (1e6 * jTok) / 3.6e6;
    expect(kwh).toBeCloseTo(0.889, 3);
    expect(costPerMillion(jTok, DEFAULT_USD_PER_KWH)).toBeCloseTo(kwh * DEFAULT_USD_PER_KWH, 9);
    expect(costPerMillion(jTok, DEFAULT_USD_PER_KWH)).toBeCloseTo(0.1333, 4);
    expect(fmtUsd(costPerMillion(jTok, DEFAULT_USD_PER_KWH))).toBe('0.13');
  });

  test('the formula is J/token x $/kWh / 3.6, not some other constant', () => {
    // Independent derivation: joules -> kWh -> dollars, no shared constant.
    for (const [j, price] of [[3.2, 0.15], [0.13, 0.3], [1.9, 0.08]]) {
      expect(costPerMillion(j, price)).toBeCloseTo(((1e6 * j) / 3.6e6) * price, 12);
    }
  });

  test('tokens per Wh and watts are derived from the stored pair', () => {
    expect(tokPerWh(2500, 8000)).toBeCloseTo(1125, 6); // 2500 / 8000 * 3600
    expect(wattsOf(8000, 100)).toBe(80);
    // tok/Wh and $/1M are the same measurement seen twice: both are 1/(J/tok).
    expect(tokPerWh(2500, 8000)).toBeCloseTo(3600 / jPerToken(8000, 2500), 9);
  });

  test('prices are formatted to two significant figures below a dollar', () => {
    expect(fmtUsd(0.13333)).toBe('0.13');
    expect(fmtUsd(0.005417)).toBe('0.0054');
    expect(fmtUsd(12.4)).toBe('12.40');
  });
});

// ---- absent is never zero ---------------------------------------------------

describe('absent is not zero', () => {
  test('a record with no energy keys reports "not measured", with no number', () => {
    const e = energyOf(rec({ metrics: { c8_aggregate_tok_s: 120 } }), 8);
    expect(e.state).toBe('absent');
    expect(e.reason).toBe('not measured');
    expect(e.energyJ).toBeUndefined();
    expect(e.jPerToken).toBeUndefined();
  });

  test('a ZERO joule count is refused, never read as free', () => {
    const e = energyOf(rec({ metrics: { ...cell(8), [cellKey(8, KEY.energyJ)]: 0 } }), 8);
    expect(e.state).toBe('refused');
    expect(e.reason).toContain('a window cannot cost nothing');
  });

  test('joules with no token count are refused — a numerator with no denominator', () => {
    const m = cell(8);
    delete m[cellKey(8, KEY.tokens)];
    const e = energyOf(rec({ metrics: m }), 8);
    expect(e.state).toBe('refused');
    expect(e.reason).toContain('joules with no denominator');
  });

  test('an absent cell is skipped by the verdicts and by the trend, not counted as 0', () => {
    const r = rec({ metrics: { ...cell(8), c16_aggregate_tok_s: 200 } });
    const v = rungVerdicts([{ c: 16, energy: energyOf(r, 16) }], [
      { label: 'vLLM', points: [{ c: 16, energy: energyOf(rec({ metrics: cell(16) }), 16) }] }
    ]);
    expect(v.rungs).toEqual([]);
    expect(v.n).toBe(0);
    expect(costTrend(16, [r]).runs).toBe(0);
  });
});

// ---- the cross-check --------------------------------------------------------

describe('the joules and the tokens must be one window', () => {
  test('a token count that disagrees with the cell tok/s by more than the tolerance is refused by sha', () => {
    const m = cell(8);
    m[cellKey(8, KEY.tokens)] = 2500 * (1 + CROSS_CHECK_TOLERANCE * 2);
    const e = energyOf(rec({ git_sha: 'deadbeef12', metrics: m }), 8);
    expect(e.state).toBe('refused');
    expect(e.reason).toContain('deadbeef12');
    expect(e.reason).toContain('not from the same window');
  });

  test('a disagreement inside the tolerance is measured', () => {
    const m = cell(8);
    m[cellKey(8, KEY.tokens)] = 2500 * 1.02;
    expect(energyOf(rec({ metrics: m }), 8).state).toBe('measured');
  });
});

// ---- under-sampled ----------------------------------------------------------

describe('under-sampled windows are marked, never averaged in', () => {
  test('too few readings makes a cell untrusted and names the count', () => {
    const e = energyOf(rec({ metrics: cell(8, { samples: MIN_POWER_SAMPLES - 1 }) }), 8);
    expect(e.state).toBe('measured');
    expect(e.trusted).toBe(false);
    expect(e.concerns.join()).toContain(`under-sampled: ${MIN_POWER_SAMPLES - 1} readings`);
    // Still a real number — it is drawn, just never counted.
    expect(e.jPerToken).toBeCloseTo(3.2, 9);
  });

  test('enough readings but poor coverage of the window is untrusted', () => {
    // 100 s window, 250 ms cadence: 40 readings cover 10% of it.
    const e = energyOf(rec({ metrics: cell(8, { samples: 40 }) }), 8);
    expect(e.trusted).toBe(false);
    expect(e.concerns.join()).toContain('the sampler covered 10%');
  });

  test('no sample count at all is untrusted — a joule count with no audit', () => {
    const m = cell(8);
    delete m[cellKey(8, KEY.samples)];
    const e = energyOf(rec({ metrics: m }), 8);
    expect(e.trusted).toBe(false);
    expect(e.concerns.join()).toContain('unauditable');
  });

  test('an under-sampled rung is EXCLUDED from k-of-n but still returned for drawing', () => {
    const atlas = [{ c: 8, energy: energyOf(rec({ metrics: cell(8, { samples: 5 }) }), 8) }];
    const rival = [{ c: 8, energy: energyOf(rec({ metrics: cell(8, { watts: 200 }) }), 8) }];
    const v = rungVerdicts(atlas, [{ label: 'vLLM', points: rival }]);
    expect(v.rungs).toHaveLength(1); // drawn
    expect(v.rungs[0].counted).toBe(false); // not counted
    expect(v.n).toBe(0);
    expect(v.k).toBe(0);
  });

  test('an under-sampled record is excluded from the trend and NAMED', () => {
    const t = costTrend(8, [rec({ metrics: cell(8, { samples: 3 }) })]);
    expect(t.runs).toBe(0);
    expect(t.metrics).toEqual([]);
    expect(t.excluded[0].reason).toContain('under-sampled');
  });

  test('the SW power cap is reported but never disqualifying — it is this box normal state', () => {
    const e = energyOf(rec({ metrics: cell(8, { swCapFrac: 1 }) }), 8);
    expect(e.trusted).toBe(true);
    expect(e.throttled).toBe(false);
    expect(e.swCapFrac).toBe(1);
  });

  test('the HW power brake keeps the point but drops it from the verdicts', () => {
    const braked = energyOf(rec({ metrics: cell(8, { hwBrakeFrac: 0.4 }) }), 8);
    expect(braked.throttled).toBe(true);
    expect(braked.concerns.join()).toContain('HW power brake asserted on 40%');
    const v = rungVerdicts([{ c: 8, energy: braked }], [
      { label: 'vLLM', points: [{ c: 8, energy: energyOf(rec({ metrics: cell(8, { watts: 200 }) }), 8) }] }
    ]);
    expect(v.rungs).toHaveLength(1);
    expect(v.rungs[0].counted).toBe(false);
  });
});

// ---- losing rungs -----------------------------------------------------------

describe('a losing rung is returned like any other, and named', () => {
  const atlasAt = (c, watts) => ({ c, energy: energyOf(rec({ metrics: cell(c, { watts }) }), c) });
  const rivalAt = (c, watts, tokS) => ({
    c,
    energy: energyOf(rec({ metrics: cell(c, { watts, tokS }) }), c)
  });

  test('Atlas losing a rung carries the factor, in the same shape as a win', () => {
    // Atlas 80 W at 25 tok/s = 3.2 J/tok; vLLM 40 W at 20 tok/s = 2.0 J/tok.
    const v = rungVerdicts([atlasAt(8, 80)], [{ label: 'vLLM', points: [rivalAt(8, 40, 20)] }]);
    expect(v.rungs).toHaveLength(1);
    const r = v.rungs[0];
    expect(r.atlasCheaper).toBe(false);
    expect(r.ratio).toBeCloseTo(1.6, 6);
    expect(r.loseLabel).toBe('vLLM cheaper ×1.60');
    expect(r.counted).toBe(true); // a loss is counted, not filtered
  });

  test('k = 0 flips the headline tile to name vLLM', () => {
    const v = rungVerdicts([atlasAt(1, 80), atlasAt(8, 80)], [
      { label: 'vLLM', points: [rivalAt(1, 40, 20), rivalAt(8, 40, 20)] }
    ]);
    expect(v.k).toBe(0);
    expect(v.n).toBe(2);
    expect(verdictTile(v)).toBe('vLLM cheaper at 2 of 2 rungs');
  });

  test('a mixed result counts only the rungs Atlas actually wins', () => {
    const v = rungVerdicts([atlasAt(1, 80), atlasAt(8, 20)], [
      { label: 'vLLM', points: [rivalAt(1, 40, 20), rivalAt(8, 40, 20)] }
    ]);
    expect(verdictTile(v)).toBe('cheaper at 1 of 2 rungs');
    const { best, worst } = extremeRungs(v);
    expect(best.c).toBe(8);
    expect(worst.c).toBe(1);
  });

  test('with nothing measured on both sides the tile says so rather than claiming a win', () => {
    expect(verdictTile({ k: 0, n: 0 })).toBe('no rung measured on both engines');
  });

  test('the cheapest rival is the one compared against, not the first listed', () => {
    const v = rungVerdicts([atlasAt(8, 80)], [
      { label: 'vLLM slow', points: [rivalAt(8, 200, 20)] },
      { label: 'vLLM fast', points: [rivalAt(8, 40, 20)] }
    ]);
    expect(v.rungs[0].rivalLabel).toBe('vLLM fast');
    expect(v.rungs[0].atlasCheaper).toBe(false);
  });
});

// ---- instruments ------------------------------------------------------------

describe('two measurements on different instruments are never one series', () => {
  test('the cost instrument key carries the rail and the sampler cadence', () => {
    const a = rec({ metrics: cell(8) });
    const b = rec({ metrics: { ...cell(8), [RUN_KEY.periodMs]: 1000 } });
    expect(costInstrumentKey(a)).toContain('|gpu_rail|250');
    expect(costInstrumentKey(a)).not.toBe(costInstrumentKey(b));
  });

  test('an unrecorded cadence is its own regime, never a match with a recorded one', () => {
    const m = cell(8);
    const noPeriod = { ...rec({ metrics: m }), metrics: { ...m } };
    delete noPeriod.metrics[RUN_KEY.periodMs];
    expect(costInstrumentKey(noPeriod)).toContain('unrecorded-period');
    expect(costInstrumentKey(noPeriod)).not.toBe(costInstrumentKey(rec({ metrics: m })));
  });

  test('a cadence change splits the trend into two metric keys, so no line joins them', () => {
    const t = costTrend(8, [
      rec({ git_sha: 'aaaaaaaaaa', recorded_at: 1, metrics: cell(8) }),
      rec({ git_sha: 'bbbbbbbbbb', recorded_at: 2, metrics: { ...cell(8), [RUN_KEY.periodMs]: 1000 } })
    ]);
    expect(t.metrics).toHaveLength(2);
    expect(new Set(t.metrics.map((m) => m.key)).size).toBe(2);
    expect(t.metrics[1].key).toBe(trendMetricKey(8)); // newest keeps the plain key
    expect(t.metrics[0].dashed).toBe(true);
    // Each record carries only its own generation's key: GateChart splits on it.
    expect(t.records[0].metrics[t.metrics[0].key]).toBeCloseTo(1125, 6);
    expect(t.records[0].metrics[t.metrics[1].key]).toBeUndefined();
  });

  test('a workload change splits the trend too, and the label says what changed', () => {
    const t = costTrend(8, [
      rec({ recorded_at: 1, metrics: cell(8), params: { isls: '128', osl: '1024', prompt_mode: 'essay' } }),
      rec({ recorded_at: 2, metrics: cell(8) })
    ]);
    expect(t.metrics).toHaveLength(2);
    expect(t.metrics[0].label).toContain('isl 128 → 512');
  });

  test('runs on ONE instrument are one series', () => {
    const t = costTrend(8, [rec({ recorded_at: 1, metrics: cell(8) }), rec({ recorded_at: 2, metrics: cell(8) })]);
    expect(t.metrics).toHaveLength(1);
    expect(t.runs).toBe(2);
  });

  test('the derived key is added to a COPY — the stored record is untouched', () => {
    const r = rec({ metrics: cell(8) });
    const t = costTrend(8, [r]);
    expect(t.records[0]).not.toBe(r);
    expect(r.metrics[trendMetricKey(8)]).toBeUndefined();
    expect(t.records[0].metrics[trendMetricKey(8)]).toBeCloseTo(1125, 6);
  });
});

// ---- the whole chart --------------------------------------------------------

describe('costLadder', () => {
  test('with no energy anywhere it reports "none" and draws nothing', () => {
    const c = costLadder(SUBJECT, [rec({ metrics: { c8_aggregate_tok_s: 120 } })], laddersWith([]));
    expect(c.energyState).toBe('none');
    expect(c.verdicts.n).toBe(0);
  });

  test('Atlas energy with no comparable vLLM energy is "atlas-only", and the vLLM series is NAMED', () => {
    const ladders = laddersWith([baselineSeries([{ c: 8, tok_s: 20, measured_utc: '2026-08-17T00:00:00Z' }])]);
    const c = costLadder(SUBJECT, [rec({ metrics: cell(8) })], ladders);
    expect(c.energyState).toBe('atlas-only');
    expect(c.baselines).toEqual([]);
    expect(c.noEnergy[0].why).toContain('no energy-instrumented run');
  });

  test('a vLLM series on another instrument is refused and its differing axes are printed', () => {
    const other = baselineSeries([baseRung(8)], {
      instrument: { isl: 128, osl: 1024, prompt_mode: 'essay', max_model_len: 2048, max_batch_size: 128, kv_cache_dtype: 'bf16' }
    });
    const c = costLadder(SUBJECT, [rec({ metrics: cell(8) })], laddersWith([other]));
    expect(c.baselines).toEqual([]);
    expect(c.refused[0].why).toContain('isl 512 → 128');
  });

  test('a comparable, energy-carrying vLLM series pairs, and the losing rung survives to the chart', () => {
    const ladders = laddersWith([baselineSeries([baseRung(8)])]);
    const c = costLadder(SUBJECT, [rec({ metrics: cell(8) })], ladders);
    expect(c.energyState).toBe('paired');
    expect(c.verdicts.n).toBe(1);
    expect(verdictTile(c.verdicts)).toBe('vLLM cheaper at 1 of 1 rungs');
    expect(c.verdicts.rungs[0].loseLabel).toBe('vLLM cheaper ×1.60');
  });

  test('above idle is offered only when BOTH sides recorded an idle baseline', () => {
    const ladders = laddersWith([baselineSeries([baseRung(8)])]);
    const withIdle = rec({ metrics: { ...cell(8), [RUN_KEY.idleW]: 20 } });
    expect(costLadder(SUBJECT, [withIdle], ladders).idle.both).toBe(false);
    expect(costLadder(SUBJECT, [withIdle], ladders).idle.why).toContain('vLLM');

    const bothLadders = laddersWith([baselineSeries([{ ...baseRung(8), [RUN_KEY.idleW]: 15 }])]);
    expect(costLadder(SUBJECT, [withIdle], bothLadders).idle.both).toBe(true);
  });

  test('a refused cell of the DRAWN record is listed by sha, never coerced', () => {
    const broken = rec({ git_sha: 'facefeed11', metrics: { ...cell(8), [cellKey(8, KEY.energyJ)]: -1 } });
    const c = costLadder(SUBJECT, [rec({ metrics: cell(8) }), broken], laddersWith([]));
    expect(c.atlas.build).toBe('facefeed11'); // the newest passing run on main is what is drawn
    expect(c.refusals.join()).toContain('facefeed11');
    expect(c.refusals.join()).toContain('a window cannot cost nothing');
  });

  test('a broken record anywhere in the history is named by the trend, not dropped in silence', () => {
    const broken = rec({ git_sha: 'facefeed11', recorded_at: 1, metrics: { ...cell(8), [cellKey(8, KEY.energyJ)]: -1 } });
    const t = costTrend(8, [broken, rec({ recorded_at: 2, metrics: cell(8) })]);
    expect(t.runs).toBe(1);
    expect(t.excluded.map((e) => e.sha)).toEqual(['facefeed11']);
  });

  test('only a PASSing record on main is the live Atlas leg', () => {
    const onBranch = rec({ git_sha: 'branch1234', branch: 'pr-1', metrics: cell(8) });
    expect(costLadder(SUBJECT, [onBranch], laddersWith([])).atlas).toBeNull();
  });
});

describe('the one-shot snapshot', () => {
  test('a vLLM rung is a dated value, not a series', () => {
    const ladders = laddersWith([baselineSeries([baseRung(8)])]);
    const c = costLadder(SUBJECT, [rec({ metrics: cell(8) })], ladders);
    const [snap] = baselineSnapshots(8, c.baselines);
    expect(snap.date).toBe('2026-08-17');
    expect(snap.tokPerWh).toBeCloseTo(1800, 6); // 2000 tok / 4000 J * 3600
    expect(snap.label).toBe('vLLM + MTP');
  });
});

// ---- the empty state, which is what ships -----------------------------------

describe('the empty state', () => {
  test('a subject with no records at all says so, with a count and never a zero cost', () => {
    const e = emptyStateOf(SUBJECT, [], laddersWith([]));
    expect(e.atlas).toContain('0 records');
    expect(e.atlas).toContain(SUBJECT.checkpoint);
    expect(e.vllm).toContain('no vLLM series');
    expect(e.fills).toContain(SUBJECT.baselines_dir);
  });

  test('records without energy are counted and the missing KEY is named', () => {
    const e = emptyStateOf(SUBJECT, [rec({ metrics: { c8_aggregate_tok_s: 1 } }), rec({ metrics: {} })], laddersWith([]));
    expect(e.atlas).toContain('2 concurrency-sweep records');
    expect(e.atlas).toContain('c{C}_gpu_rail_energy_j');
  });

  test('a published vLLM series is named with its instrument and whether it carries joules', () => {
    const e = emptyStateOf(SUBJECT, [], laddersWith([baselineSeries([{ c: 8, tok_s: 20 }])]));
    expect(e.vllm).toContain('ISL 512 / OSL 320');
    expect(e.vllm).toContain('no energy recorded');
    expect(e.vllm).toContain('no Atlas run to fingerprint it against');
  });

  test('a vLLM series on ANOTHER instrument says so — it is not a comparison waiting for joules', () => {
    const other = baselineSeries([{ c: 8, tok_s: 20 }], {
      instrument: { isl: 128, osl: 1024, prompt_mode: 'essay', max_model_len: 2048, max_batch_size: 128, kv_cache_dtype: 'bf16' }
    });
    const e = emptyStateOf(SUBJECT, [rec({ metrics: { c8_aggregate_tok_s: 1 } })], laddersWith([other]));
    expect(e.vllm).toContain('another instrument');
    expect(e.vllm).toContain('isl 512 → 128');
    expect(e.vllm).toContain('more axes'); // the rest are counted, never dropped
  });

  test('a vLLM series ON the gate instrument says that instead', () => {
    const e = emptyStateOf(
      SUBJECT,
      [rec({ metrics: { c8_aggregate_tok_s: 1 } })],
      laddersWith([baselineSeries([{ c: 8, tok_s: 20 }])])
    );
    expect(e.vllm).toContain('the gate instrument');
    expect(e.vllm).not.toContain('another instrument');
  });
});

// ---- helpers ----------------------------------------------------------------

describe('reading the record', () => {
  test('rungsOf reads the rungs the record measured throughput at, ascending', () => {
    expect(rungsOf(rec({ metrics: { ...cell(128), ...cell(1), ...cell(16) } }))).toEqual([1, 16, 128]);
  });

  test('readEnergy works on a ladder rung with no prefix, as ladder-build writes it', () => {
    const e = readEnergy(baseRung(8), '', baseRung(8), 'vLLM C=8', 20);
    expect(e.state).toBe('measured');
    expect(e.jPerToken).toBeCloseTo(2, 9);
  });
});

describe('the checks can fail', () => {
  test('NEGATIVE CONTROL: a well-sampled cell IS trusted, so the trust assertions are not vacuous', () => {
    const e = energyOf(rec({ metrics: cell(8) }), 8);
    expect(e.trusted).toBe(true);
    expect(e.concerns).toEqual([]);
  });

  test('NEGATIVE CONTROL: Atlas winning a rung produces no lose label and k = n', () => {
    const atlas = { c: 8, energy: energyOf(rec({ metrics: cell(8, { watts: 20 }) }), 8) };
    const rival = { c: 8, energy: energyOf(rec({ metrics: cell(8, { watts: 40, tokS: 20 }) }), 8) };
    const v = rungVerdicts([atlas], [{ label: 'vLLM', points: [rival] }]);
    expect(v.rungs[0].loseLabel).toBeNull();
    expect(verdictTile(v)).toBe('cheaper at 1 of 1 rungs');
  });

  test('NEGATIVE CONTROL: a subject WITH energy is not reported as empty', () => {
    const e = emptyStateOf(SUBJECT, [rec({ metrics: cell(8) })], laddersWith([]));
    expect(e.atlas).toContain('1 of 1 records carry GPU-rail joules');
  });
});

// ---------------------------------------------------------------------------
// The measured run-to-run spread (issue #1214)
//
// The Cost tab's second dated point made its trend line slope for the first
// time, and the slope was smaller than the noise: J/token fell 5.6% at C=4
// between two anchors whose throughput draws (47.33 and 50.47 tok/s) sit inside
// a 6.1% run-to-run CoV over the 32 committed records that share their
// instrument. These tests pin the envelope that refuses such a claim, and --
// the load-bearing half -- pin that it can still ALLOW one.
// ---------------------------------------------------------------------------

/** A PASS run of `rec()`'s instrument carrying only a C=4 throughput. */
const tput = (c4, o = {}) => rec({ ...o, metrics: { c4_aggregate_tok_s: c4, ...(o.metrics ?? {}) } });

/** The same, on a DIFFERENT instrument (the old 1,4,8,16 ladder). */
const otherInstrument = (c4) =>
  rec({ params: { concurrencies: '1, 4, 8, 16', isls: '512', osl: '320', prompt_mode: 'natural' },
        metrics: { c4_aggregate_tok_s: c4 } });

const KEY_OF = (r) => instrumentKey(r);

describe('rungSpread', () => {
  test('a population under MIN_SPREAD_RUNS has no envelope, so nothing is claimed from it', () => {
    const pop = Array.from({ length: MIN_SPREAD_RUNS - 1 }, (_, i) => tput(45 + i));
    expect(rungSpread(4, KEY_OF(pop[0]), pop)).toBeNull();
  });

  test('at exactly MIN_SPREAD_RUNS it measures, and the envelope spans the observed range', () => {
    const vals = [44.66, 46, 47, 47.33, 48, 49, 50, 50.47, 52, 53.84];
    const pop = vals.map((v) => tput(v));
    const e = rungSpread(4, KEY_OF(pop[0]), pop);
    expect(e).not.toBeNull();
    expect(e.n).toBe(MIN_SPREAD_RUNS);
    expect(e.tMin).toBeCloseTo(44.66, 2);
    expect(e.tMax).toBeCloseTo(53.84, 2);
    expect(e.lo).toBeLessThan(1);
    expect(e.hi).toBeGreaterThan(1);
  });

  // ★ THE CONTROL FOR THE CONTAMINATION THAT PROMPTED THIS. Pooling the whole
  // .benchmarks history mixed two concurrency ladders and reported a 7.5% CoV
  // where the matched population gives 6.1%. A record on another instrument
  // must not move the envelope by so much as a digit.
  test('a run on a DIFFERENT instrument does not widen the envelope', () => {
    const pop = Array.from({ length: 12 }, (_, i) => tput(46 + i * 0.5));
    const before = rungSpread(4, KEY_OF(pop[0]), pop);
    const after = rungSpread(4, KEY_OF(pop[0]), [...pop, otherInstrument(9.9), otherInstrument(999)]);
    expect(after).toEqual(before);
  });

  // ★ THE CONTROL AGAINST A CIRCULAR BAND. costInstrumentKey appends the
  // sampler period, which almost no record carries; keying the population on it
  // would collapse n to the handful of energy-bearing points the envelope is
  // meant to judge.
  test('the population is not restricted to records carrying energy', () => {
    const pop = Array.from({ length: 12 }, (_, i) => tput(46 + i * 0.5));
    const e = rungSpread(4, KEY_OF(pop[0]), pop);
    expect(e.n).toBe(12);
    expect(pop.filter((r) => r.metrics[KEY.energyJ] !== undefined)).toHaveLength(0);
  });

  test('a non-PASS run is never averaged into the spread', () => {
    const pop = Array.from({ length: 12 }, (_, i) => tput(46 + i * 0.5));
    const withFail = [...pop, tput(9.9, { verdict: 'FAIL' }), tput(999, { verdict: 'FAIL' })];
    expect(rungSpread(4, KEY_OF(pop[0]), withFail)).toEqual(rungSpread(4, KEY_OF(pop[0]), pop));
  });
});

describe('spreadVerdict', () => {
  const envelope = { n: 32, lo: 0.94, hi: 1.06, tMin: 44.66, tMax: 53.84, tMedian: 48, powerMeasured: 2 };

  test("two points closer than the spread are NOT distinguishable — tonight's real case", () => {
    // 1279 vs 1277 tok/Wh: the step is 0.2%, the envelope is +/-6%.
    expect(spreadVerdict([1277, 1279], envelope).state).toBe('overlap');
  });

  test('the 5.6% C=4 step that started this is still overlap', () => {
    expect(spreadVerdict([1 / 1.3105, 1 / 1.2375], envelope).state).toBe('overlap');
  });

  // ★ WITHOUT THIS THE REFUSAL IS UNFALSIFIABLE DECORATION. A step that really
  // does clear the envelope must be reported as separated, and in the right
  // direction.
  test('a step larger than the spread IS distinguishable, and names its direction', () => {
    const v = spreadVerdict([1000, 1400], envelope);
    expect(v.state).toBe('separated');
    expect(v.direction).toBe('up');
    expect(spreadVerdict([1400, 1000], envelope).direction).toBe('down');
  });

  test('with no envelope, or fewer than two points, nothing is claimed either way', () => {
    expect(spreadVerdict([1000, 1400], null).state).toBe('unmeasured');
    expect(spreadVerdict([1000], envelope).state).toBe('unmeasured');
  });
});

// ---------------------------------------------------------------------------
// The sampler-coverage guard runs on every record, or says it could not (#1216)
//
// Coverage is samples x period / window. With no recorded cadence it cannot be
// computed, and the if/else-if chain used to fall off its end: ample samples
// and no period collected NO concern, so the record was trusted, drawn solid
// and counted everywhere, coverage unchecked. 2 of 72 committed records carry
// the key.
// ---------------------------------------------------------------------------
describe('readEnergy · coverage with no recorded cadence', () => {
  /** A well-formed window: 400 readings over 100 s, 80 W. */
  const window = (o = {}) => ({
    [KEY.energyJ]: 8000,
    [KEY.tokens]: 2000,
    [KEY.windowS]: 100,
    [KEY.samples]: 400,
    ...o
  });

  test('a record with ample samples and NO period is not silently trusted', () => {
    const e = readEnergy(window(), '', {}, 'r', null);
    expect(e.state).toBe('measured');
    expect(e.trusted).toBe(false);
    expect(e.concerns.join(' ')).toContain('cadence not recorded');
  });

  // ★ THE CONTROL. A record that DOES carry the cadence, and whose coverage is
  // fine, must be unaffected — otherwise the fix makes every good record hollow.
  test('a record that records its cadence, and is well covered, stays trusted', () => {
    const e = readEnergy(window(), '', { [RUN_KEY.periodMs]: 250 }, 'r', null);
    expect(e.trusted).toBe(true);
    expect(e.concerns).toEqual([]);
  });

  test('the existing under-coverage refusal still fires and still names the numbers', () => {
    // 100 readings at 250 ms = 25 s of a 100 s window = 25%.
    const e = readEnergy(window({ [KEY.samples]: 100 }), '', { [RUN_KEY.periodMs]: 250 }, 'r', null);
    expect(e.trusted).toBe(false);
    expect(e.concerns.join(' ')).toContain('25%');
  });

  test('too few samples is still reported as under-sampling, not as a missing cadence', () => {
    const e = readEnergy(window({ [KEY.samples]: MIN_POWER_SAMPLES - 1 }), '', {}, 'r', null);
    expect(e.concerns.join(' ')).toContain('under-sampled');
    expect(e.concerns.join(' ')).not.toContain('cadence not recorded');
  });

  test('no sample count at all is still reported as unauditable', () => {
    const e = readEnergy(window({ [KEY.samples]: undefined }), '', {}, 'r', null);
    expect(e.concerns.join(' ')).toContain('unauditable');
    expect(e.concerns.join(' ')).not.toContain('cadence not recorded');
  });
});
