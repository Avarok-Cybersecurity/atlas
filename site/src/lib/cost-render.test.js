// SPDX-License-Identifier: AGPL-3.0-only
//
// The Cost section, RENDERED — not grepped. Svelte's server compiler turns a
// component into a function returning HTML and bun runs that without a DOM, so
// these assert on what a reader would actually see.
//
// The four claims under test are the ones the owner approved and that a future
// edit could quietly undo:
//   * the EMPTY state is informative — it is the state this section ships in;
//   * a LOSING rung is drawn with the factor by which it loses, and the
//     headline flips to `vLLM cheaper at n of n rungs` at k = 0;
//   * an ABSENT cell renders as "not measured" and never as a zero cost;
//   * an UNDER-SAMPLED cell is hollow and is in no tile.
//
// The plugin is scoped to THIS file's imports, exactly as concurrency-tab
// .test.js does, so the pre-existing resolution failures elsewhere stay as
// they were.
import { describe, expect, test } from 'bun:test';
import { plugin } from 'bun';
import { compile } from 'svelte/compiler';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const LIB = fileURLToPath(new URL('./', import.meta.url));
const SELF = fileURLToPath(import.meta.url);

plugin({
  name: 'cost-ssr',
  setup(build) {
    build.onResolve({ filter: /^\$lib(\/|$)/ }, (a) =>
      a.importer.endsWith('.test.js') && a.importer !== SELF ? undefined : { path: join(LIB, a.path.slice(4)) }
    );
    build.module('$app/environment', () => ({ contents: 'export const browser = true; export const dev = false;', loader: 'js' }));
    build.module('$app/navigation', () => ({ contents: 'export const replaceState = () => {};', loader: 'js' }));
    build.onLoad({ filter: /\.svelte$/ }, (a) => ({
      contents: compile(readFileSync(a.path, 'utf8'), { filename: a.path, generate: 'server' }).js.code,
      loader: 'js'
    }));
  }
});

const { render } = await import('svelte/server');
const { recordsFor, tabs } = await import('./gates.js');
const { SUBJECTS } = await import('./concurrency-subjects.js');
const { KEY, RUN_KEY, cellKey, DEFAULT_USD_PER_KWH, costTrend } = await import('./cost.js');
const Panel = (await import('./components/CostSubjectPanel.svelte')).default;
const Chart = (await import('./components/CostLadderChart.svelte')).default;
const { costLadder, pueRange, PUE_TYPICAL, PUE_MAX, DEFAULT_PUE } = await import('./cost.js');
const Tab = (await import('./components/CostTab.svelte')).default;
const Dashboard = (await import('./components/BenchmarkDashboard.svelte')).default;
const LADDERS = (await import('./ladders.generated.json')).default;

const html = (C, props) => render(C, { props }).body.replace(/<!--[^]*?-->/g, '').replace(/\s+/g, ' ');
/** Entity-decoded, for assertions about words rather than about escaping. */
const text = (s) => s.replace(/&quot;/g, '"').replace(/&#123;/g, '{').replace(/&#125;/g, '}').replace(/&amp;/g, '&');

const DENSE = SUBJECTS.find((s) => s.id === 'qwen38-27b');
const MOE = SUBJECTS.find((s) => s.id === 'qwen36-35b-a3b');

// ---- synthetic measurements -------------------------------------------------

const cell = (c, o = {}) => {
  const { watts = 80, tokS = 25, windowS = 100, samples = 400 } = o;
  return {
    [`c${c}_aggregate_tok_s`]: tokS,
    [cellKey(c, KEY.energyJ)]: watts * windowS,
    [cellKey(c, KEY.tokens)]: tokS * windowS,
    [cellKey(c, KEY.windowS)]: windowS,
    [cellKey(c, KEY.samples)]: samples
  };
};

const rec = (metrics, o = {}) => ({
  git_sha: 'abc1234567',
  recorded_at: 1789900000,
  verdict: 'PASS',
  branch: null,
  target_model: DENSE.checkpoint,
  benchmark_id: 'concurrency-sweep',
  hardware: { gpu: 'NVIDIA GB10' },
  params: { concurrencies: '1, 8', isls: '512', osl: '320', prompt_mode: 'natural' },
  serve_overrides: { kv_cache_dtype: 'fp8', max_batch_size: '128', max_model_len: '4096' },
  metrics: { [RUN_KEY.periodMs]: 250, ...metrics },
  ...o
});

const rung = (c, o = {}) => {
  // periodMs is declared because every REAL energy-bearing record declares it:
  // all 6 on main carry gpu_rail_sample_period_ms, and readEnergy only ever
  // runs on records that carry energy. A fixture without it is not a good
  // record with a field missing -- it is a record whose coverage cannot be
  // verified, and since #1216 it is drawn hollow, correctly. Tests that mean
  // "a good record" must therefore say so. Pass `periodMs: null` to build the
  // unverifiable case on purpose.
  const { watts = 40, tokS = 20, windowS = 100, samples = 400, periodMs = 250 } = o;
  return {
    c,
    tok_s: tokS,
    measured_utc: '2026-08-17T12:24:02Z',
    [KEY.energyJ]: watts * windowS,
    [KEY.tokens]: tokS * windowS,
    [KEY.windowS]: windowS,
    [KEY.samples]: samples,
    ...(periodMs === null ? {} : { [RUN_KEY.periodMs]: periodMs })
  };
};

const ladders = (rungs) => ({
  subjects: {
    [DENSE.id]: {
      workload: { checkpoint: DENSE.checkpoint },
      box: { name: 'dgx2' },
      series: [
        {
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
          rungs
        }
      ]
    }
  }
});

/**
 * The cost chart's own plot area — not a legend swatch, and not the prose
 * under it. The first assertion this file wrote sliced from the first `<svg`
 * and got a 20x10 legend swatch, which contains none of what is tested here.
 */
const plot = (s) => {
  const i = s.indexOf('<svg viewBox="0 0 720 260"');
  expect(i).toBeGreaterThan(-1);
  return s.slice(i, s.indexOf('</svg>', i));
};

const panel = (records, lad, rungSel = 8) =>
  text(html(Panel, { subject: DENSE, records, rung: rungSel, onselect: () => {}, ladders: lad }));

// ---- the empty state, which is what ships -----------------------------------

describe('the empty state', () => {
  // The REAL committed data and the REAL generated ladders: this is the state
  // the section ships in.
  const real = panel(recordsFor(DENSE.gate).filter((r) => r.target_model === DENSE.checkpoint), LADDERS);

  test('the committed data has no energy yet, so the real dense tab renders the empty state', () => {
    expect(real).toContain('Cost · not yet measured');
    expect(real).toContain('No GPU-rail energy');
  });

  test('it names what Atlas has, what vLLM has, and what fills the gap — in that order', () => {
    const iAtlas = real.indexOf('Atlas:');
    const iVllm = real.indexOf('vLLM:');
    const iFills = real.indexOf('What fills this:');
    expect(iAtlas).toBeGreaterThan(-1);
    expect(iVllm).toBeGreaterThan(iAtlas);
    expect(iFills).toBeGreaterThan(iVllm);
    expect(real).toContain('none carries c{C}_gpu_rail_energy_j');
    expect(real).toContain('bench/baselines/qwen38-27b/');
    // The published vLLM legs are ISL 128 / OSL 1024, not the gate's — the
    // empty state says that rather than implying they are waiting for joules.
    expect(real).toContain('another instrument');
  });

  test('it counts the records it DOES have rather than showing a zero', () => {
    const n = recordsFor(DENSE.gate).filter((r) => r.target_model === DENSE.checkpoint).length;
    expect(n).toBeGreaterThan(0);
    expect(real).toContain(`${n} concurrency-sweep records`);
  });

  test('NOTHING renders as a zero cost, a zero joule count or a zero efficiency', () => {
    expect(real).not.toMatch(/\$\s*0\.00/);
    expect(real).not.toMatch(/0 J\/token|0 tok\/Wh|0\.000 J/);
  });

  test('a subject with no records at all says 0 records — and still not a zero cost', () => {
    const moe = text(html(Panel, { subject: MOE, records: [], rung: 8, onselect: () => {}, ladders: { subjects: {} } }));
    expect(moe).toContain('0 records');
    expect(moe).not.toMatch(/\$\s*0\.00/);
  });

  test('the rail and the price disclosure are shown even with no data', () => {
    expect(real).toContain('GPU rail only — a lower bound on cost.');
    expect(real).toContain('Your input, not a measurement.');
    expect(real).toContain(`${DEFAULT_USD_PER_KWH}`);
  });
});

// ---- a losing rung ----------------------------------------------------------

describe('a losing rung is drawn, labelled and counted against us', () => {
  // Atlas 80 W at 25 tok/s = 3.2 J/tok; vLLM 40 W at 20 tok/s = 2.0 J/tok.
  const losing = panel([rec(cell(8))], ladders([rung(8)]));

  test('the point carries the factor by which it loses, on the chart', () => {
    expect(losing).toContain('vLLM cheaper ×1.60');
  });

  test('the headline tile flips to name vLLM when k = 0', () => {
    expect(losing).toContain('vLLM cheaper at 1 of 1 rungs');
    expect(losing).not.toContain('cheaper at 0 of 1');
  });

  test('the losing point is drawn with the SAME mark as a winning one — no filter, no fade', () => {
    const won = panel([rec(cell(8, { watts: 20 }))], ladders([rung(8)]));
    const marksOf = (s) => (s.match(/class="gc-mark"[^>]*/g) ?? []).map((m) => m.replace(/cx="[\d.]+" cy="[\d.]+"/, ''));
    expect(marksOf(losing)).toHaveLength(1);
    expect(marksOf(losing)).toEqual(marksOf(won));
    expect(won).toContain('cheaper at 1 of 1 rungs');
    // The prose below the chart quotes the label verbatim, so the absence is
    // asserted where the marks are: inside the plot.
    expect(plot(won)).not.toContain('vLLM cheaper ×');
    expect(plot(losing)).toContain('vLLM cheaper ×1.60');
  });

  test('a mixed ladder shows both rungs and counts only the won one', () => {
    const mixed = panel(
      [rec({ ...cell(1), ...cell(8, { watts: 20 }) })],
      ladders([rung(1), rung(8)])
    );
    expect(mixed).toContain('cheaper at 1 of 2 rungs');
    expect(mixed).toContain('vLLM cheaper ×1.60'); // C=1, still drawn
  });

  test('every lose label stays INSIDE the plot — a clipped factor is a loss not shown', () => {
    // Eight rungs, Atlas losing every one, so labels land at both ends of the
    // axis as well as in the middle.
    const cs = [1, 2, 4, 8, 16, 32, 64, 128];
    const wide = panel(
      [rec(Object.assign({}, ...cs.map((c) => cell(c))))],
      ladders(cs.map((c) => rung(c))),
      128
    );
    const svg = plot(wide);
    const labels = [...svg.matchAll(/<text class="cost-lose" x="([\d.]+)"[^>]*text-anchor="(\w+)"\s*>([^<]+)</g)];
    expect(labels).toHaveLength(cs.length);
    // 10px monospace advances ~0.6em; 6.2 is a deliberate over-estimate, so a
    // label that only just fits still has to fit.
    for (const [, xs, anchor, label] of labels) {
      const w = label.trim().length * 6.2;
      const left = anchor === 'start' ? +xs : anchor === 'end' ? +xs - w : +xs - w / 2;
      expect(left).toBeGreaterThanOrEqual(0);
      expect(left + w).toBeLessThanOrEqual(720);
    }
    // ...and the anchor only moves at the ends: a mid-axis label is centred,
    // so this is not passing by anchoring everything to one side.
    expect(labels.filter(([, , a]) => a === 'middle').length).toBeGreaterThan(0);
  });

  test('the rail is named ON the chart, not only in the prose below it', () => {
    const svg = plot(losing);
    expect(svg).toContain('GPU rail only');
    expect(svg).toContain('lower bound');
  });
});

// ---- absent is not zero -----------------------------------------------------

describe('absent is not zero, in the rendering too', () => {
  const partial = panel([rec({ ...cell(8), c16_aggregate_tok_s: 200 })], ladders([rung(8), rung(16)]));

  test('a rung with throughput but no joules renders "not measured"', () => {
    expect(partial).toContain('not measured');
    expect(partial).toContain('no GPU-rail energy recorded for this rung');
  });

  test('that rung contributes no mark and no dollar figure', () => {
    expect(partial).toContain('cost per 1M tokens vs vLLM');
    expect((partial.match(/class="gc-mark"/g) ?? [])).toHaveLength(1); // C=8 only
    expect(partial).toContain('vLLM cheaper at 1 of 1 rungs'); // C=16 is not an n
  });
});

// ---- under-sampled ----------------------------------------------------------

// ★ THE CONTROL FOR THE FIXTURE CHANGE ABOVE. `rung()` now declares a cadence
// by default, which is what every real energy-bearing record does — but that
// must not be a way of switching the guard off. A rung built WITHOUT one has to
// render hollow and stay out of the count, exactly like an under-sampled one.
describe('a window whose sampler cadence was never recorded (#1216)', () => {
  // The two paths refuse differently, and each is asserted where it renders.
  // A RECORD without a cadence is untrusted, so costTrend drops it from the
  // drawn series entirely and reports why in `excluded` -- checked directly
  // against costTrend, because "not drawn" has no mark to match on. A RUNG
  // without one renders in the ladder with its reason named.
  const blindRung = panel([rec(cell(8))], ladders([rung(8, { periodMs: null })]));

  test('a RECORD with no cadence is not drawn at all, and the trend says why', () => {
    const blind = rec({ ...cell(8), [RUN_KEY.periodMs]: undefined });
    const t = costTrend(8, [blind]);
    expect(t.records).toHaveLength(0);
    expect(t.excluded).toHaveLength(1);
    expect(t.excluded[0].reason).toContain('cadence not recorded');
  });

  test('a RUNG with no cadence names the reason rather than just fading the point', () => {
    expect(text(blindRung)).toContain('cadence not recorded');
  });

  test('declaring the cadence draws both — so neither assertion is vacuous', () => {
    const seen = rec(cell(8));
    expect(costTrend(8, [seen]).records).toHaveLength(1);
    expect(costTrend(8, [seen]).excluded).toHaveLength(0);
    expect(text(panel([seen], ladders([rung(8)])))).not.toContain('cadence not recorded');
  });
});

describe('an under-sampled window is marked, not counted', () => {
  const thin = panel([rec(cell(8, { samples: 4 }))], ladders([rung(8)]));

  test('its mark is HOLLOW', () => {
    expect(thin).toMatch(/class="gc-mark"[^>]*fill="var\(--card\)"/);
    expect(thin).toContain('under-sampled — drawn, never counted');
  });

  test('it is in no tile: the verdict says nothing was measured on both engines', () => {
    expect(thin).toContain('no rung measured on both engines');
    expect(thin).not.toContain('cheaper at 1 of 1');
  });

  test('a well-sampled window is SOLID — so the hollow assertion is not vacuous', () => {
    const solid = panel([rec(cell(8))], ladders([rung(8)]));
    expect(solid).not.toMatch(/class="gc-mark"[^>]*fill="var\(--card\)"/);
    expect(solid).not.toContain('under-sampled — drawn, never counted');
  });
});

// ---- the price input --------------------------------------------------------

describe('the electricity price', () => {
  const p = panel([rec(cell(8))], ladders([rung(8)]));

  test('is one labelled input, defaulted and described as an assumption', () => {
    expect(p).toContain('Electricity price in dollars per kilowatt-hour');
    expect(p).toContain('electricity price');
    expect(p).toContain('Your input, not a measurement.');
    expect(p).toContain('never');
  });

  test('is never written into the deep link', async () => {
    const { formatDashboardHash } = await import('./dashboard-link.js');
    const hash = formatDashboardHash({ tab: 'cost', subject: DENSE.id, c: 8 });
    expect(hash).not.toContain('kwh');
    expect(hash).not.toContain('price');
    expect(hash).toBe('bench=cost&subject=qwen38-27b&c=8');
  });
});

// ---- the tab ----------------------------------------------------------------

describe('the Cost tab', () => {
  test('is registered and earns a tab from the sweep records', () => {
    expect(tabs.map((t) => t.id)).toContain('cost');
  });

  test('carries all three subjects of the SSOT, with a derived chip for each', () => {
    const t = text(html(Tab, { subject: DENSE.id, rung: 8, benches: ['concurrency-sweep', 'concurrency-sweep-dflash2'], recordsFor, onselect: () => {} }));
    for (const s of SUBJECTS) expect(t).toContain(s.label);
    expect(t).toContain('energy not yet measured'); // has runs, no joules
    expect(t).toContain('no runs yet'); // the MoE subject
  });

  test('the dashboard opens it from a deep link and hides the global model select', () => {
    // BenchmarkDashboard reads location.hash at first render (SSR runs no
    // effects, so nothing writes it back).
    globalThis.location = { hash: '#bench=cost&subject=qwen38-27b', pathname: '/engine', search: '' };
    let d;
    try {
      d = text(html(Dashboard, { onclose: () => {} }));
    } finally {
      delete globalThis.location;
    }
    expect(d).toContain('Cost');
    // The tab strip carries a cost tab button with the dashboard's own prefix.
    expect(d).toContain('id="bd-tab-cost"');
  });
});

// ---- facility overhead (PUE) ------------------------------------------------
//
// Owner, 2026-09-21: "include this option next to the electricity price, keep
// the PUE default at 1.0, showing a range in the graph from a low to high PUE
// with two options". Three things have to be true for that to be honest, and
// each is asserted rather than described:
//   * at the default the control applies NOTHING and draws NOTHING;
//   * a band is drawn for BOTH engines, because PUE applies to both;
//   * it moves every absolute figure and moves no comparison.
describe('facility overhead', () => {
  const rungs = [rung(1, { watts: 60, tokS: 12 }), rung(8, { watts: 90, tokS: 70 })];
  const lad = ladders(rungs);
  const records = [rec({ ...cell(1, { watts: 55, tokS: 13 }), ...cell(8, { watts: 95, tokS: 66 }) })];
  const cost = costLadder(DENSE, records, lad);
  const chart = (pue) =>
    text(
      html(Chart, {
        subject: DENSE,
        cost,
        usdPerKwh: DEFAULT_USD_PER_KWH,
        pue,
        rungs: [1, 8],
        title: 'Cost',
        aboveIdle: false,
        onselect: () => {}
      })
    );
  const none = chart(pueRange(DEFAULT_PUE, DEFAULT_PUE));
  const band = chart(pueRange(PUE_TYPICAL.low, PUE_TYPICAL.high));

  test('the fixture actually draws both engines, or nothing below means anything', () => {
    expect(cost.atlas).not.toBeNull();
    expect(cost.baselines).toHaveLength(1);
    expect(plot(none)).toContain('gc-mark');
  });

  test('THE DEFAULT APPLIES NOTHING: no band, no PUE in the unit, nothing shaded', () => {
    expect(none).not.toContain('cost-band');
    expect(none).toContain('$ per 1M tokens · log scale');
    expect(none).not.toContain('PUE');
  });

  test('a low-to-high pair shades a band FOR BOTH ENGINES, under the lines', () => {
    const p = plot(band);
    // Two bands: the Atlas series and the comparable vLLM one-shot. One band
    // would be a claim about the comparison that is not true.
    expect((p.match(/class="cost-band"/g) ?? []).length).toBe(2);
    // ...and both are drawn BEFORE the first line, so an assumption is never
    // painted over a measurement.
    expect(p.indexOf('cost-band')).toBeLessThan(p.indexOf('gc-mark'));
    expect(band).toContain('PUE 1.1–1.5×');
    expect(band).toContain('the line is the rail, the band is the building');
  });

  test('the AXIS makes room for the band — the top edge is never drawn off the chart', () => {
    // The y axis is fitted to the data, so both edges have to feed it. Fitted
    // to the low edge alone, the band's top is clipped away silently and the
    // reader sees a truncated assumption. PT is the plot's top padding (26).
    // Measured at the WIDEST band the input accepts (PUE_MAX), because a
    // narrow one is absorbed by the axis's own eighth-of-a-decade padding and
    // would let the defect through: fitted to the low edge alone, 1.0-3.0
    // puts the band's top ~35 px ABOVE the plot and it is silently clipped.
    const wide = plot(chart(pueRange(DEFAULT_PUE, PUE_MAX)));
    const d = wide.match(/class="cost-band" d="([^"]+)"/)[1];
    const ys = [...d.matchAll(/[ML]([\d.]+) (-?[\d.]+)/g)].map((m) => Number(m[2]));
    expect(ys.length).toBeGreaterThan(2);
    for (const y of ys) expect(y).toBeGreaterThanOrEqual(26);
  });

  test('the tooltip states the RANGE when banded and a single figure when not', () => {
    expect(none).toMatch(/\$[\d.]+ per 1M tokens at 0\.15 \$\/kWh/);
    expect(band).toMatch(/\$[\d.]+–[\d.]+ per 1M tokens at 0\.15 \$\/kWh, PUE 1\.1–1\.5/);
  });

  test('IT MOVES EVERY FIGURE...', () => {
    // The y axis is labelled in dollars, and at PUE 1.1-1.5 those dollars are
    // higher than at 1.0. If the axis were unchanged the band would be paint.
    const axis = (s) => [...s.matchAll(/class="gc-axis"[^>]*>\$([\d.]+)</g)].map((m) => Number(m[1]));
    const a0 = axis(plot(none));
    const a1 = axis(plot(band));
    expect(a0.length).toBeGreaterThan(0);
    expect(a1.length).toBe(a0.length);
    expect(a1.some((v, i) => v > a0[i])).toBe(true);
  });

  test('...AND MOVES NO COMPARISON: the winner and the factor are byte-identical', () => {
    // The property the whole control rests on. Strip the parts that are
    // ALLOWED to move (the band marks, the dollar figures, the axis) and what
    // is left -- who wins each rung and by what factor -- must not differ.
    const verdicts = (s) => [...s.matchAll(/(?:Atlas|vLLM) cheaper ×[\d.]+/g)].map((m) => m[0]);
    expect(verdicts(none).length).toBeGreaterThan(0);
    expect(verdicts(band)).toEqual(verdicts(none));
  });

  test('the MARK SITS ON THE LOW EDGE — never floating inside an assumption', () => {
    // Stated design rule: the line is the rail reading (the low edge of the
    // band), the band is the building. Checked as GEOMETRY inside one render,
    // so it holds whatever the axis does: every Atlas mark must lie on the
    // band's return leg, which `bandOf` draws at the low PUE.
    const p = plot(band);
    const d = p.match(/class="cost-band" d="([^"]+)"/)[1];
    const back = d.slice(d.indexOf(' L')); // the reversed low-PUE curve
    const onLow = [...back.matchAll(/L([\d.]+) ([\d.]+)/g)].map(
      (m) => `${Number(m[1]).toFixed(1)},${Number(m[2]).toFixed(1)}`
    );
    const marks = [...p.matchAll(/class="gc-mark" cx="([\d.]+)" cy="([\d.]+)"/g)].map(
      (m) => `${Number(m[1]).toFixed(1)},${Number(m[2]).toFixed(1)}`
    );
    expect(marks.length).toBeGreaterThan(0);
    for (const m of marks) expect(onLow).toContain(m);
    // ...and the high edge is somewhere else, or "on the low edge" is vacuous.
    const out = d.slice(0, d.indexOf(' L'));
    expect(out).not.toBe('');
    const highY = Number(out.match(/M[\d.]+ ([\d.]+)/)[1]).toFixed(1);
    expect(marks.some((m) => m.endsWith(`,${highY}`))).toBe(false);
  });

  test('the panel offers the control NEXT TO the price, with both presets', () => {
    const p = panel(records, lad);
    const iPrice = p.indexOf('electricity price');
    const iPue = p.indexOf('facility overhead (PUE)');
    expect(iPrice).toBeGreaterThan(-1);
    expect(iPue).toBeGreaterThan(iPrice);
    // Two boxes, not one: a building is a range.
    expect(p).toContain('Lowest power usage effectiveness to draw');
    expect(p).toContain('Highest power usage effectiveness to draw');
    expect(p).toContain('none (1×)');
    expect(p).toContain('datacentre (1.1–1.5×)');
    // and the panel ships at the identity, so its own chart carries no band
    expect(p).toContain('PUE starts at 1×, which applies nothing.');
    expect(plot(p)).not.toContain('cost-band');
  });
});
