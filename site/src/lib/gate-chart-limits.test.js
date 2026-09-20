// SPDX-License-Identifier: AGPL-3.0-only
//
// The gate charts, rendered — not grepped: Svelte's server compiler turns a
// component into a function that returns HTML, and these tests assert on what
// a reader would see. Two things the owner asked for, and each test traces
// to one:
//   - every metric with a declared floor or ceiling shows it as a labelled
//     rule, stepped where the bound changed, and a point past it is marked;
//   - a series is one line through its points in time order.
//
// Geometry is never recomputed here. A limit's row is proven by putting a
// point EXACTLY on the limit and asserting the rule and the mark share a y;
// a step is proven by the path's own `V`.
//
// The plugin is scoped to THIS file's imports, like concurrency-tab.test.js.
import { describe, expect, test } from 'bun:test';
import { plugin } from 'bun';
import { compile } from 'svelte/compiler';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const LIB = fileURLToPath(new URL('./', import.meta.url));
const SELF = fileURLToPath(import.meta.url);

plugin({
  name: 'gate-chart-limits-ssr',
  setup(build) {
    build.onResolve({ filter: /^\$lib(\/|$)/ }, (a) =>
      a.importer.endsWith('.test.js') && a.importer !== SELF ? undefined : { path: join(LIB, a.path.slice(4)) }
    );
    build.onLoad({ filter: /\.svelte$/ }, (a) => ({
      contents: compile(readFileSync(a.path, 'utf8'), { filename: a.path, generate: 'server' }).js.code,
      loader: 'js'
    }));
  }
});

const { render } = await import('svelte/server');
const { gateLimits, fmtDate } = await import('./gates.js');
// GateChart's own value formatting (fmtV): thousands grouped, else 2 decimals.
const fmtMs = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : String(+v.toFixed(2)));
const { SUBJECTS } = await import('./concurrency-subjects.js');
const GateChart = (await import('./components/GateChart.svelte')).default;
const GateLadderChart = (await import('./components/GateLadderChart.svelte')).default;
const ConcurrencyComparison = (await import('./components/ConcurrencyComparison.svelte')).default;

const html = (C, props) => render(C, { props }).body.replace(/<!--[^]*?-->/g, '').replace(/\s+/g, ' ');
// The plot alone: the legend's swatches reuse the plot's classes on purpose
// (a key that stops matching the chart is worse than none), so they must be
// kept out of counts of what the plot draws.
const plot = (page) => page.slice(page.indexOf('<svg viewBox'));

const MODEL = 'unsloth/Qwen3.8-27B-NVFP4';
const DAY = 86400;
// The declarations these fixtures lean on, read from the real table so the
// assertions move with BENCH.toml rather than pinning a stale number. Each is
// the newest entry of a dated series: {since, value}.
const DECLARED_FLOOR = gateLimits['decode-floor'][MODEL].server_decode_tok_s.min.at(-1);
// Fixture records are dated after every declaration in play unless a test
// says otherwise.
const T0 = DECLARED_FLOOR.since + DAY;
let seq = 0;
// Every params value is a STRING, as the harness records them.
const rec = (over = {}) => {
  seq += 1;
  return {
    benchmark_id: 'decode-floor',
    benchmark_name: 'decode floor',
    git_sha: `sha${seq}`,
    recorded_at: T0 + seq * DAY,
    target_model: MODEL,
    served_by: 'recipe',
    verdict: 'PASS',
    verdict_reason: '',
    branch: '',
    params: {},
    metrics: {},
    trend_predecessor: '',
    generated_ancestry: 'unknown',
    ...over
  };
};
const decode = (v, params = {}) => rec({ metrics: { server_decode_tok_s: v }, params });
const DECODE_PANEL = { title: 'decode floor', unit: 'tok/s', metrics: [{ key: 'server_decode_tok_s', label: 'tok/s' }] };

const limitPaths = (page) => [...plot(page).matchAll(/<path class="gc-limit" d="([^"]+)"/g)].map((m) => m[1]);
const limitLabels = (page) => [...plot(page).matchAll(/<text class="gc-limit-label"[^>]*>([^<]*)<\/text>/g)].map((m) => m[1]);
const markYs = (page) => [...plot(page).matchAll(/<circle class="gc-mark[^"]*" cx="[\d.]+" cy="([\d.]+)"/g)].map((m) => +m[1]);
const violations = (page) =>
  [...plot(page).matchAll(/<circle class="gc-viol" cx="([\d.]+)" cy="([\d.]+)" r="7\.5" data-limit="(\w+)"/g)];
const r1 = (n) => Math.round(n * 10) / 10;

describe('GateChart: the floor BENCH.toml declares, with no threshold in the record', () => {
  const declared = DECLARED_FLOOR.value;
  const page = html(GateChart, {
    panel: DECODE_PANEL,
    records: [decode(declared + 3), decode(declared), decode(declared + 1)],
    onselect: () => {}
  });

  test('one flat rule spans the plot field and is labelled with the bound and its value', () => {
    expect(declared).toBeGreaterThan(0);
    const paths = limitPaths(page);
    expect(paths).toHaveLength(1);
    expect(paths[0]).toMatch(/^M56\.0 [\d.]+ H704\.0$/); // PL to W-PR, no step
    expect(limitLabels(page)).toEqual([`floor ${declared} tok/s`]);
    expect(page).toContain('gate floor / ceiling');
  });

  test('the rule sits on the row of a point whose value equals the floor', () => {
    const ruleY = +/^M56\.0 ([\d.]+) H/.exec(limitPaths(page)[0])[1];
    const ys = markYs(page);
    expect(ys).toHaveLength(3);
    expect(r1(ys[1])).toBe(ruleY);
    expect(ys[0]).toBeLessThan(ruleY); // higher tok/s draws higher up
  });

  test('NEGATIVE CONTROL: a point on the line passes; nothing is marked', () => {
    expect(violations(page)).toHaveLength(0);
    expect(page).not.toContain('past its limit');
    expect(page).not.toContain('below floor');
  });
});

describe('GateChart: the floor each record was judged against, stepped where it was ratcheted', () => {
  const page = html(GateChart, {
    panel: DECODE_PANEL,
    records: [
      decode(26, { min_tok_s: '24.5' }),
      decode(24.7, { min_tok_s: '24.5' }),
      decode(26, { min_tok_s: '26' }), // ratchet: on the new line, passes
      decode(25.5, { min_tok_s: '26' }) // below the new floor, above the old one
    ],
    onselect: () => {}
  });

  test('the rule steps once, and both stretches are labelled', () => {
    const paths = limitPaths(page);
    expect(paths).toHaveLength(1);
    expect(paths[0]).toMatch(/^M56\.0 [\d.]+ H[\d.]+ V[\d.]+ H704\.0$/);
    expect(limitLabels(page)).toEqual(['floor 24.5 tok/s', 'floor 26 tok/s']);
  });

  test('the record-level floor wins over the declared 25: 24.7 under the old rule passes', () => {
    const ys = markYs(page);
    const [stepFrom, stepTo] = /^M56\.0 ([\d.]+) H[\d.]+ V([\d.]+) H/.exec(limitPaths(page)[0]).slice(1).map(Number);
    expect(r1(ys[2])).toBe(stepTo); // the point AT 26 sits on the second stretch
    expect(ys[1]).toBeLessThan(stepFrom); // 24.7 is above the 24.5 stretch
  });

  test('only the point below ITS OWN floor is ringed, and its description says by which bound', () => {
    const v = violations(page);
    expect(v).toHaveLength(1);
    expect(v[0][3]).toBe('floor');
    expect(+v[0][2]).toBe(markYs(page)[3]);
    expect(page).toContain('25.5 tok/s · below floor 26 ·');
    expect(page).toContain('past its limit');
  });
});

describe('GateChart: a ceiling', () => {
  const wall = (v, budget) =>
    rec({ benchmark_id: 'agentic-webserver', metrics: { sum_wall_s: v }, params: { wall_budget_s: budget } });
  const page = html(GateChart, {
    panel: { title: 'Σ wall time', unit: 's', metrics: [{ key: 'sum_wall_s', label: 'Σ wall (s)' }] },
    records: [wall(1500, '1800'), wall(1800, '1800'), wall(1950, '1800')],
    onselect: () => {}
  });

  test('is labelled as a ceiling with its value, and the point over it is ringed as such', () => {
    expect(limitLabels(page)).toEqual(['ceiling 1,800 s']);
    const v = violations(page);
    expect(v).toHaveLength(1);
    expect(v[0][3]).toBe('ceiling');
    expect(+v[0][2]).toBe(markYs(page)[2]);
    expect(page).toContain('1,950 s · over ceiling 1,800 ·');
  });

  test('the ceiling label rides above its rule; a floor label hangs below', () => {
    const ruleY = +/^M56\.0 ([\d.]+) H/.exec(limitPaths(page)[0])[1];
    const labelY = +/<text class="gc-limit-label" x="[\d.]+" y="([\d.]+)"/.exec(page)[1];
    expect(labelY).toBeLessThan(ruleY);
  });
});

describe('GateChart: a declared ceiling applies only from the date it took effect', () => {
  // The real warm-TTFT ceiling for the 35B FP8 checkpoint, and the day it
  // took effect. The records carry no absolute of their own, so this is the
  // case that once turned pre-ratchet history red.
  const CK = 'Qwen/Qwen3.6-35B-A3B-FP8';
  const { since, value: ceiling } = gateLimits['ttft-warm-gate'][CK].median_ms.max.at(-1);
  const ttft = (when, v) =>
    rec({ benchmark_id: 'ttft-warm-gate', target_model: CK, recorded_at: when, metrics: { median_ms: v } });
  const panel = { title: 'warm TTFT', unit: 'ms', metrics: [{ key: 'median_ms', label: 'median' }] };
  // Two points over today's ceiling: one recorded before it took effect, one after.
  const before = ttft(since - 30 * DAY, ceiling * 2);
  const after = ttft(since + 30 * DAY, ceiling * 2);
  const page = html(GateChart, { panel, records: [before, after, ttft(since + 31 * DAY, ceiling / 2)], onselect: () => {} });

  test('the point recorded BEFORE the ceiling took effect is not ringed; the one after it is', () => {
    expect(since).toBeGreaterThan(0);
    const v = violations(page);
    expect(v).toHaveLength(1);
    expect(v[0][3]).toBe('ceiling');
    expect(+v[0][2]).toBe(markYs(page)[1]); // the second point, not the first
    expect(page).not.toContain(`${fmtMs(ceiling * 2)} ms · over ceiling ${fmtMs(ceiling)} · ${fmtDate(before.recorded_at)}`);
    expect(page).toContain(`${fmtMs(ceiling * 2)} ms · over ceiling ${fmtMs(ceiling)} · ${fmtDate(after.recorded_at)}`);
  });

  test('the rule is drawn from the first point it governs, not from the left edge of history', () => {
    const paths = limitPaths(page);
    expect(paths).toHaveLength(1);
    const cxs = [...plot(page).matchAll(/<circle class="gc-mark" cx="([\d.]+)"/g)].map((m) => +m[1]);
    const x1 = cxs[1]; // the first point the ceiling governs is the second one drawn
    const start = +/^M([\d.]+) /.exec(paths[0])[1];
    expect(start).toBeCloseTo(x1, 0);
    expect(start).toBeGreaterThan(56); // PL: the left edge is history the rule never judged
  });

  test('NEGATIVE CONTROL: every point dated after the ceiling took effect is judged by it', () => {
    const all = html(GateChart, { panel, records: [ttft(since + DAY, ceiling * 2), ttft(since + 2 * DAY, ceiling * 2)], onselect: () => {} });
    expect(violations(all)).toHaveLength(2);
    expect(limitPaths(all)[0]).toMatch(/^M56\.0 /);
  });
});

describe('GateChart: a re-cut newer than every record is still drawn, on its own day', () => {
  // The real FP8 warm-TTFT ceiling has at least two entries; take the last
  // two and date every record between them, so the newest re-cut post-dates
  // the newest record — the state a chart is in right after a ratchet lands
  // and before the next gate run.
  const CK = 'Qwen/Qwen3.6-35B-A3B-FP8';
  const series = gateLimits['ttft-warm-gate'][CK].median_ms.max;
  const [old, recut] = series.slice(-2);
  const panel = { title: 'warm TTFT', unit: 'ms', metrics: [{ key: 'median_ms', label: 'median' }] };
  const ttft = (when, v) =>
    rec({ benchmark_id: 'ttft-warm-gate', target_model: CK, recorded_at: when, metrics: { median_ms: v } });
  // Both values sit between the two ceilings: over the new one, under the old.
  const v = (old.value + recut.value) / 2;
  const page = html(GateChart, { panel, records: [ttft(old.since + DAY, v), ttft(old.since + 5 * DAY, v)], onselect: () => {} });

  test('the axis reaches the re-cut, the rule steps down there, and both bounds are named', () => {
    expect(series.length).toBeGreaterThanOrEqual(2);
    expect(recut.since).toBeGreaterThan(old.since + 5 * DAY);
    const paths = limitPaths(page);
    expect(paths).toHaveLength(1);
    expect(paths[0]).toMatch(/^M56\.0 [\d.]+ H[\d.]+ V[\d.]+ H704\.0$/);
    const [yOld, yNew] = /^M56\.0 ([\d.]+) H[\d.]+ V([\d.]+) H/.exec(paths[0]).slice(1).map(Number);
    expect(yNew).toBeGreaterThan(yOld); // a lower ceiling draws lower
    expect(limitLabels(page)).toEqual([`ceiling ${fmtMs(old.value)} ms`, `ceiling ${fmtMs(recut.value)} ms`]);
    expect(plot(page)).toContain(`text-anchor="end">${fmtDate(recut.since)}</text>`);
  });

  test('NEGATIVE CONTROL: the records themselves are judged by the ceiling of their day, not the new one', () => {
    expect(violations(page)).toHaveLength(0);
  });
});

describe('GateChart: a metric with no limit anywhere', () => {
  test('draws no rule, no label and no legend key', () => {
    const page = html(GateChart, {
      panel: { title: 'warm TTFT', unit: 'ms', metrics: [{ key: 'median_ms', label: 'median' }] },
      records: [rec({ benchmark_id: 'ttft-warm-gate', metrics: { median_ms: 400 } }), rec({ benchmark_id: 'ttft-warm-gate', metrics: { median_ms: 410 } })],
      onselect: () => {}
    });
    expect(limitPaths(page)).toHaveLength(0);
    expect(limitLabels(page)).toHaveLength(0);
    expect(page).not.toContain('gate floor / ceiling');
  });
});

describe('GateChart: the series is one line through its points in time order', () => {
  test('two points already earn a line, and the line visits every point', () => {
    const records = [decode(30), decode(31), decode(29), decode(32)];
    const page = html(GateChart, { panel: DECODE_PANEL, records, onselect: () => {} });
    const lines = [...page.matchAll(/<path class="gc-line" d="([^"]+)"/g)].map((m) => m[1]);
    expect(lines).toHaveLength(1);
    const verts = [...lines[0].matchAll(/[ML]([\d.]+) ([\d.]+)/g)].map((m) => [+m[1], +m[2]]);
    expect(verts).toHaveLength(4);
    // The path is written to one decimal; the marks are not.
    const marks = [...page.matchAll(/<circle class="gc-mark" cx="([\d.]+)" cy="([\d.]+)"/g)].map((m) => [r1(+m[1]), r1(+m[2])]);
    expect(verts).toEqual(marks);
    for (let i = 1; i < verts.length; i += 1) expect(verts[i][0]).toBeGreaterThan(verts[i - 1][0]);

    const two = html(GateChart, { panel: DECODE_PANEL, records: [decode(30), decode(31)], onselect: () => {} });
    expect(two).toMatch(/<path class="gc-line" d="M[\d.]+ [\d.]+ L[\d.]+ [\d.]+"/);
  });

  test('NEGATIVE CONTROL: a lone point has nothing to join', () => {
    const one = html(GateChart, { panel: DECODE_PANEL, records: [decode(30)], onselect: () => {} });
    expect(one).not.toContain('class="gc-line"');
    expect(one).toContain('gc-lone');
  });
});

// ---- the concurrency ladder: a floor per rung ------------------------------
const sweep = (cells, floors, over = {}) =>
  rec({
    benchmark_id: 'concurrency-sweep',
    metrics: Object.fromEntries([
      ...Object.entries(cells).map(([c, v]) => [`c${c}_aggregate_tok_s`, v]),
      ['peak_aggregate_tok_s', Math.max(...Object.values(cells))]
    ]),
    params: {
      concurrencies: Object.keys(cells).join(', '),
      isls: '512',
      osl: '320',
      ...Object.fromEntries(Object.entries(floors).map(([c, v]) => [`min_c${c}`, String(v)]))
    },
    ...over
  });

describe('GateLadderChart: the newest run\'s per-rung floor', () => {
  // C=4 is above its own floor (37.5) but below C=8's (48): only a per-rung
  // rule leaves it unmarked. C=8 is below its own floor.
  const latest = sweep({ 1: 22, 2: 27, 4: 40, 8: 46 }, { 1: 20.4, 2: 25, 4: 37.5, 8: 48 });
  const page = html(GateLadderChart, {
    panel: { title: 'latest gate sweep', unit: 'tok/s' },
    records: [sweep({ 1: 21, 2: 26, 4: 39, 8: 50 }, { 1: 20, 2: 25, 4: 37.5, 8: 48 }), latest],
    onselect: () => {}
  });

  test('is one stepped rule with the bound named once and every rung\'s value on it', () => {
    const paths = limitPaths(page);
    expect(paths).toHaveLength(1);
    expect(paths[0].match(/V/g)).toHaveLength(3); // four distinct floors, three steps
    expect(limitLabels(page)).toEqual(['floor 20.4 tok/s', '25', '37.5', '48']);
    expect(page).toContain('gate floor per rung');
  });

  test('the rule at C=8 sits on the row of 48 tok/s, and only C=8 is ringed', () => {
    const v = violations(page);
    expect(v).toHaveLength(1);
    expect(page).toContain('C=8 · 46 tok/s · below floor 48 ·');
    expect(page).not.toContain('C=4 · 40 tok/s · below');
    expect(page).toContain('below its floor');
  });

  test('NEGATIVE CONTROL: a run with every rung clear of its floor marks nothing', () => {
    const clean = html(GateLadderChart, {
      panel: { title: 'latest gate sweep', unit: 'tok/s' },
      records: [sweep({ 1: 22, 2: 27, 4: 40, 8: 50 }, { 1: 20.4, 2: 25, 4: 37.5, 8: 48 })],
      onselect: () => {}
    });
    expect(violations(clean)).toHaveLength(0);
    expect(limitPaths(clean)).toHaveLength(1);
  });
});

describe('ConcurrencyComparison (live): the live record\'s per-rung floor', () => {
  const subject = SUBJECTS.find((s) => s.gate === 'concurrency-sweep' && s.checkpoint === MODEL);
  const live = sweep({ 1: 22, 2: 24, 4: 40 }, { 1: 20.4, 2: 25, 4: 37.5 });
  // An empty ladder set withholds the published pair so the live branch is drawn.
  const page = html(ConcurrencyComparison, {
    subject,
    records: [live],
    rungs: [1, 2, 4, 8],
    onselect: () => {},
    ladders: { subjects: {} }
  });

  test('draws the stepped floor under the Atlas curve and rings the rung below it', () => {
    expect(subject).toBeDefined();
    expect(limitPaths(page)).toHaveLength(1);
    expect(limitLabels(page)).toEqual(['floor 20.4 tok/s', '25', '37.5']);
    const v = violations(page);
    expect(v).toHaveLength(1);
    expect(page).toContain('C=2 · 24 tok/s · below floor 25 ·');
    expect(page).toContain('gate floor per rung');
  });

  test('the floor is held by the axis: the rule row is inside the plot field', () => {
    const y = +/^M[\d.-]+ ([\d.]+) H/.exec(limitPaths(page)[0])[1];
    expect(y).toBeGreaterThan(14);
    expect(y).toBeLessThan(232 - 30);
  });
});
