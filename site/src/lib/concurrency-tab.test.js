// SPDX-License-Identifier: AGPL-3.0-only
//
// The Concurrency tab, rendered — not grepped. Svelte's server compiler turns a
// component into a function that returns HTML, and bun can run that without a
// DOM, so these tests assert on what a reader would actually see: which tab is
// selected, whether a chart has marks, what a caption says and in what order.
//
// Two rules this tab exists to keep, and every test here traces to one:
//   - the published ladder (ISL 128 / OSL 1024, ~478 tok/s) and the live gate
//     (ISL 512 / OSL 320, ~116) are DIFFERENT INSTRUMENTS on one checkpoint;
//     each chart is labelled with its own and neither's number appears on the
//     other's axes;
//   - an empty subject keeps its chrome and says what is missing, why, and
//     what fills it — never a zero, never a mark.
//
// The plugin below is scoped to THIS file's imports so the two pre-existing
// `$lib/agent/*.svelte.js` resolution failures elsewhere stay exactly as they
// were: a harness that quietly fixed them here would change the baseline.
import { describe, expect, test } from 'bun:test';
import { plugin } from 'bun';
import { compile } from 'svelte/compiler';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const LIB = fileURLToPath(new URL('./', import.meta.url));
const SELF = fileURLToPath(import.meta.url);

plugin({
  name: 'concurrency-tab-ssr',
  setup(build) {
    build.onResolve({ filter: /^\$lib(\/|$)/ }, (a) =>
      a.importer.endsWith('.test.js') && a.importer !== SELF ? undefined : { path: join(LIB, a.path.slice(4)) }
    );
    // BenchmarkDashboard reads the hash through `browser` and writes it back
    // through replaceState; SSR runs neither effect, so the stubs only need to
    // exist. `browser: true` lets the initial parse read the fake location.
    build.module('$app/environment', () => ({ contents: 'export const browser = true; export const dev = false;', loader: 'js' }));
    build.module('$app/navigation', () => ({ contents: 'export const replaceState = () => {};', loader: 'js' }));
    build.onLoad({ filter: /\.svelte$/ }, (a) => ({
      contents: compile(readFileSync(a.path, 'utf8'), { filename: a.path, generate: 'server' }).js.code,
      loader: 'js'
    }));
  }
});

const { render } = await import('svelte/server');
const { recordsFor, tabs, ladderPoints, fmtDate, colorFor } = await import('./gates.js');
const { SUBJECTS, rungsDeclared } = await import('./concurrency-subjects.js');
const { liveRecordOf } = await import('./concurrency-comparison.js');
const publishedLadder = (await import('./ladder.generated.json')).default;
const ladders = (await import('./ladders.generated.json')).default;
const Tab = (await import('./components/ConcurrencyTab.svelte')).default;
const Ladder = (await import('./components/ConcurrencyLadder.svelte')).default;
const Comparison = await import('./components/ConcurrencyComparison.svelte');
const Dashboard = (await import('./components/BenchmarkDashboard.svelte')).default;

// Comments stripped and whitespace collapsed: the assertions are about what
// is said, not how the template was indented.
const html = (C, props) => render(C, { props }).body.replace(/<!--[^]*?-->/g, '').replace(/\s+/g, ' ');
const rungs = [...new Set(SUBJECTS.flatMap((s) => rungsDeclared(s, recordsFor)))].sort((a, b) => a - b);
const benches = tabs.find((t) => t.id === 'concurrency').benches;
const byId = (id) => SUBJECTS.find((s) => s.id === id);
/** Entity-decoded, for assertions about words rather than about escaping. */
const text = (x) => x.replace(/&quot;/g,'"').replace(/&amp;/g,'&').replace(/&#45;/g,'-').replace(/&minus;/g,'−');

const renderTab = (subject, rf = recordsFor) =>
  html(Tab, { subject, rungs, benches, recordsFor: rf, onselect: () => {} });

/** The comparison chart's own <svg> (the first one after its title). */
const comparisonSvg = (page) => {
  const from = page.indexOf('class="gate-panel-title">Atlas vs vLLM');
  const open = page.indexOf('<svg viewBox', from);
  return page.slice(open, page.indexOf('</svg>', open));
};
const gateSvg = (page) => {
  const from = page.indexOf('class="gate-panel-title">latest gate sweep');
  const open = page.indexOf('<svg viewBox', from);
  return page.slice(open, page.indexOf('</svg>', open));
};
const tile = (page, label) =>
  new RegExp(`<span class="gbs-tile-val[^"]*"[^>]*>([^<]*)</span>\\s*<span class="gbs-tile-label">${label}`).exec(page)?.[1];
/** Each phrase must appear, and appear AFTER the one before it. */
const inOrder = (page, ...needles) => {
  let at = 0;
  for (const n of needles) {
    const i = page.indexOf(n, at);
    expect(i, `"${n}" must appear after "${needles[needles.indexOf(n) - 1] ?? '(start)'}"`).not.toBe(-1);
    at = i + n.length;
  }
};
/** The published ladder's figure, and its plot svg (the legend swatches come first). */
const ladderFigure = (page) => {
  const open = page.indexOf('<figure class="cl-panel">');
  return page.slice(open, page.indexOf('</figure>', open));
};
const ladderSvg = (page) => {
  const fig = ladderFigure(page);
  const open = fig.indexOf('<svg viewBox');
  return fig.slice(open, fig.indexOf('</svg>', open));
};

const DENSE = byId('qwen38-27b');
const MOE = byId('qwen36-35b-a3b');
const DFLASH = byId('qwen38-27b-dflash');
const denseLatest = recordsFor('concurrency-sweep').at(-1);
// ★ THE PUBLISHED-PAIR STATE IS NOW A FALLBACK, NOT WHAT SHIPS. Since #1220 the
// gate measures the published instrument and its newest passing record on
// main PAIRS with the vLLM+MTP bar (the 'live' state below). The fallback is
// still reachable — every record on the retired 512/320 instrument refuses
// to pair — and `recordsFor` filtered to those records is how it is rendered
// here, so the fallback's wording stays under test without a second fixture.
const onRetiredInstrument = (r) => r.benchmark_id !== 'concurrency-sweep' || r.params?.prompt_mode !== 'essay';
const rfRetired = (bench) => recordsFor(bench).filter(onRetiredInstrument);
const retiredLatest = rfRetired('concurrency-sweep').at(-1);

// A passing main-branch record for any subject, shaped like the real ones.
const fakeRecord = (subject, cs, over = {}) => ({
  benchmark_id: subject.gate,
  target_model: subject.checkpoint,
  git_sha: 'feedfacedead'.slice(0, 10),
  recorded_at: 1_800_000_000,
  verdict: 'PASS',
  branch: '',
  params: { concurrencies: cs.join(', '), isls: '512', osl: '320', prompt_mode: 'natural' },
  serve_overrides: { max_batch_size: '128', kv_cache_dtype: 'fp8', max_model_len: '4096' },
  metrics: Object.fromEntries(cs.map((c) => [`c${c}_aggregate_tok_s`, 10 * c])),
  ...over
});
const withExtra = (extra) => (bench) => [...recordsFor(bench), ...extra.filter((r) => r.benchmark_id === bench)];

describe('the subject strip', () => {
  test('one tab per subject in SSOT order; the prop picks the selected one and its panel', () => {
    const page = renderTab('qwen36-35b-a3b');
    const ids = [...page.matchAll(/id="cs-tab-([^"]+)"/g)].map((m) => m[1]);
    expect(ids).toEqual(SUBJECTS.map((s) => s.id));
    for (const s of SUBJECTS) expect(page).toContain(`aria-controls="cs-panel-${s.id}"`);
    expect(page.match(/aria-selected="true"/g)).toHaveLength(1);
    expect(page).toMatch(/id="cs-tab-qwen36-35b-a3b"[^>]*aria-selected="true"/);
    expect(page).toContain('id="cs-panel-qwen36-35b-a3b" role="tabpanel" aria-labelledby="cs-tab-qwen36-35b-a3b"');
    expect(page.match(/role="tabpanel"/g)).toHaveLength(1);
  });

  test('an unknown subject is a wiring bug and throws, not a silent first tab', () => {
    expect(() => renderTab('nvidia-35b')).toThrow(/unknown subject "nvidia-35b"/);
  });

  test('"not yet measured" sits INSIDE the tab button of every subject with no records', () => {
    const page = renderTab('qwen38-27b');
    const chips = [...page.matchAll(/<button[^>]*id="cs-tab-([^"]+)"[^>]*>[^<]*<span class="cs-chip">not yet measured<\/span>/g)].map((m) => m[1]);
    expect(chips).toEqual(['qwen36-35b-a3b']);
  });

  test('NEGATIVE CONTROL: the chip follows the records, not the subject id', () => {
    const moeMeasured = withExtra([fakeRecord(MOE, [1, 4])]);
    expect(renderTab('qwen38-27b', moeMeasured)).not.toContain('not yet measured');
    const dflashGone = (bench) => (bench === DFLASH.gate ? [] : recordsFor(bench));
    const page = renderTab('qwen38-27b', dflashGone);
    expect(page).toMatch(/id="cs-tab-qwen38-27b-dflash"[^>]*>[^<]*<span class="cs-chip">not yet measured/);
  });

  test('records no subject claims are listed, never dropped', () => {
    expect(renderTab('qwen38-27b')).not.toContain('unassigned:');
    const stray = fakeRecord({ gate: 'concurrency-sweep', checkpoint: 'nvidia/Qwen3.6-35B-A3B-NVFP4' }, [1]);
    expect(renderTab('qwen38-27b', withExtra([stray]))).toContain(
      'unassigned: concurrency-sweep · nvidia/Qwen3.6-35B-A3B-NVFP4 (1)'
    );
  });
});

describe('the MoE tab today: a vLLM one-shot on the published instrument, no Atlas leg', () => {
  const page = renderTab('qwen36-35b-a3b');
  const moe = ladders.subjects['qwen36-35b-a3b'];
  const vllm = moe.series.find((s) => s.id === 'vllm-mtp');
  // The server renderer escapes `&` and `<` in dynamic text, not `>`; the
  // manifest's error string carries a `->`.
  const esc = (t) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;');

  test('header: checkpoint id verbatim, the one-shot dated from its rungs, and no verdict tile', () => {
    expect(page).toContain('>Qwen/Qwen3.6-35B-A3B-FP8</span>');
    expect(tile(page, 'records')).toBe('0');
    expect(tile(page, 'gate concurrency-sweep')).toBe('declared, unmeasured');
    expect(tile(page, 'vLLM baseline')).toBe('one-shot · 2026-09-19');
    expect(page).not.toContain('gbs-verdict');
  });

  test('the vLLM series is drawn as squares at its measured rungs only, every value from the ladder', () => {
    expect(page).toContain(
      'class="gate-panel-title">Atlas vs vLLM · published instrument · ISL 128 / OSL 1024 · vLLM one-shot, no Atlas run yet</span>'
    );
    const svg = comparisonSvg(page);
    expect(svg.match(/class="cmp-sq"/g)).toHaveLength(vllm.rungs.length);
    for (const r of vllm.rungs)
      expect(svg).toContain(`<title>vLLM + MTP · C=${r.c} · ${r.tok_s.toFixed(2)} tok/s · mean of ${r.reps} reps · spread ${r.spread_pct}% · ${r.source}</title>`);
    expect(svg).toContain(`>${vllm.engine} · 2026-09-19</text>`); // the dated stamp, inside the plot
    for (const c of rungs) expect(svg).toContain(`>C=${c}</text>`);
    // No Atlas marks, no Atlas line: the hollow key says why.
    expect(svg).not.toContain('gc-mark');
    expect(svg).not.toContain(`stroke="${colorFor(MOE.checkpoint)}"`);
    expect(page).toContain('Atlas · no run at this instrument yet');
    // Legend text, not a pill: nothing here is pressable, so nothing may look it.
    expect(page).toContain('</svg>vLLM + MTP · one-shot · measured 2026-09-19 · not re-run</span>');
    expect(page).not.toContain('cmp-chip');
  });

  test('C=32/64/128 are absent with the recorded reason where the point would be — no zero, no interpolation', () => {
    expect(vllm.rungs.map((r) => r.c)).toEqual([1, 2, 4, 8, 16]);
    expect(vllm.unmeasured.rungs).toEqual([32, 64, 128]);
    const svg = comparisonSvg(page);
    for (const c of vllm.unmeasured.rungs)
      expect(svg).toContain(`<title>C=${c} · not measured — ${vllm.unmeasured.reason}</title>`);
    expect(svg.match(/>not measured<\/text>/g)).toHaveLength(3);
    expect(svg).not.toContain('not in this manifest');
  });

  test('the copy: once, engine+build, the forced kernel and its crash, the absent rungs, then what fills it', () => {
    inOrder(
      page,
      'vLLM was measured <strong>once</strong>, on 2026-09-19',
      `<code>${vllm.engine} (${vllm.build})</code> on ${moe.box.name}`,
      'not re-measured when Atlas moves',
      "<strong>vLLM + MTP is not running vLLM's default kernel here.</strong>",
      'DeepGEMM',
      `<code>${vllm.kernel_override.env}</code>`,
      `<code>${esc(vllm.kernel_override.error)}</code>`,
      '<strong>C=32, 64, 128 not measured.</strong>',
      'physical powercycle',
      'No Atlas run at this instrument yet for <code>Qwen/Qwen3.6-35B-A3B-FP8</code>',
      'role <code>subject</code> in <code>bench/baselines/qwen36-35b-a3b/published.json</code>',
      'Exact configuration and provenance',
      `<td class="mono cl-src">${vllm.rungs[0].source}</td>`,
      `sha256 of the committed copy <code>${moe.harness_repo_sha256.slice(0, 10)}</code>`
    );
    expect(page).not.toContain('undefined');
    expect(page).not.toContain('No concurrency run on main yet');
    expect(page).not.toContain('latest gate sweep');
    expect(page).not.toContain('cc-bridge');
  });

  test('NEGATIVE CONTROL: a passing gate run on main is drawn ALONE — the one-shot is on another instrument and the caption names the axes', () => {
    const live = renderTab('qwen36-35b-a3b', withExtra([fakeRecord(MOE, [1, 2, 4])]));
    expect(live).not.toContain('not yet measured');
    const svg = comparisonSvg(live);
    expect(svg.match(/class="gc-mark"/g)).toHaveLength(3);
    expect(svg).not.toContain('cmp-sq');
    expect(live).toContain('vLLM + MTP · other instrument · not drawn');
    expect(tile(live, 'vLLM baseline')).toBe('other instrument');
    inOrder(
      live,
      '<strong>vLLM has not been run on this instrument</strong> (ISL 512 / OSL 320 · natural fixture · batch cap 128 · fp8 KV)',
      'The vLLM + MTP one-shot of 2026-09-19 is on another instrument and is not comparable: isl 512 → 128, osl 320 → 1024, prompt_mode natural → essay, max_model_len 4096 → 2048, kv_cache_dtype fp8 → bf16.',
      'filed under <code>bench/baselines/qwen36-35b-a3b/</code>, fills the comparison'
    );
    expect(live).toContain('latest gate sweep · ISL 512 / OSL 320 · natural fixture · batch cap 128 · fp8 KV');
  });

  test('POSITIVE CONTROL: a one-shot whose fingerprint equals the gate record IS drawn against it', () => {
    // The same ladder with its baseline re-fingerprinted on the gate's axes —
    // the shape a future bench/baselines/<subject>/ one-shot on the gate
    // instrument would generate to. Only `instrument` is read by the check.
    const onGate = { isl: 512, osl: 320, prompt_mode: 'natural', max_model_len: 4096, max_batch_size: 128, kv_cache_dtype: 'fp8' };
    const fake = { subjects: { 'qwen36-35b-a3b': { ...moe, series: [{ ...vllm, instrument: onGate }] } } };
    const props = { subject: MOE, records: [fakeRecord(MOE, [1, 2, 4])], rungs, onselect: () => {}, ladders: fake };
    const drawn = html(Comparison.default, props);
    const svg = comparisonSvg(drawn);
    expect(svg.match(/class="cmp-sq"/g)).toHaveLength(vllm.rungs.length);
    expect(svg.match(/class="gc-mark"/g)).toHaveLength(3);
    expect(drawn).toContain('vLLM + MTP · one-shot · measured 2026-09-19 · not re-run');
    inOrder(
      drawn,
      'vLLM was measured <strong>once</strong>, on 2026-09-19',
      'on this instrument (ISL 512 / OSL 320 · natural fixture · batch cap 128 · fp8 KV)',
      'Atlas is the newest passing run on main'
    );
    expect(drawn).not.toContain('not comparable');
    expect(drawn).not.toContain('not drawn');
    expect(Comparison.baselineTileOf(MOE, props.records, fake)).toBe('one-shot · 2026-09-19');
  });
});

describe('the DFlash tab today: Atlas only, absent rungs answered in place', () => {
  const page = renderTab('qwen38-27b-dflash');
  // Two records, two roles, as ConcurrencySubjectPanel keeps them: the header
  // tiles read the NEWEST record wherever it sits; the comparison chart and
  // the bridge draw the newest PASSING run ON MAIN (liveRecordOf). They were
  // one and the same until a DFlash record landed on an unmerged branch.
  const latest = recordsFor(DFLASH.gate).at(-1);
  const live = liveRecordOf(recordsFor(DFLASH.gate));
  const pts = ladderPoints(latest);
  const livePts = ladderPoints(live);

  test('header tiles come from the newest record', () => {
    const peak = pts.reduce((a, b) => (b.v > a.v ? b : a));
    expect(tile(page, `peak \\(C=${peak.c}\\) · ${fmtDate(latest.recorded_at)}`)).toBe(`${peak.v.toFixed(1)} tok/s`);
    expect(tile(page, 'rungs declared')).toBe(`${pts.length} of ${rungs.length}`);
    expect(tile(page, 'vLLM baseline')).toBe('none');
    expect(page).toContain('data-verdict="PASS"');
  });

  test('the Atlas curve is drawn dashed (same engine, another configuration) at its measured rungs only', () => {
    const svg = comparisonSvg(page);
    expect(svg).toMatch(/<path d="M[^"]+" fill="none" stroke="var\(--series-teal[^"]*" stroke-width="2" stroke-dasharray="5 4"/);
    expect(svg.match(/class="gc-mark"/g)).toHaveLength(livePts.length);
    for (const c of rungs) expect(svg).toContain(`>C=${c}</text>`);
  });

  test('each rung the gate does not run says so where the point would be, with the recorded reason', () => {
    const svg = comparisonSvg(page);
    const absent = rungs.filter((c) => !livePts.some((p) => p.c === c));
    expect(absent).toEqual([32, 64, 128]);
    for (const c of absent)
      expect(svg).toContain(
        `<title>C=${c} · not run at this rung — the concurrency-sweep-dflash2 gate stops at C=16 by design: max_batch_size = "16"</title>`
      );
    expect(svg.match(/>not run<\/text>/g)).toHaveLength(absent.length);
  });

  test('NEGATIVE CONTROL: without a recorded batch cap the reason is the declared rung list, never "undefined"', () => {
    const stripped = { ...live, serve_overrides: { kv_cache_dtype: 'fp8' } };
    const rf = (bench) => (bench === DFLASH.gate ? [stripped] : recordsFor(bench));
    const svg = comparisonSvg(renderTab('qwen38-27b-dflash', rf));
    expect(svg).toContain('the concurrency-sweep-dflash2 gate declares concurrencies = "1, 2, 4, 8, 16"');
    expect(svg).not.toContain('undefined');
  });

  test('the legend and caption say vLLM was not run on THIS instrument, and what would fill it', () => {
    expect(page).toContain(`Atlas · live · latest gate ${fmtDate(live.recorded_at)} · ${live.git_sha}`);
    expect(page).toMatch(/<rect [^>]*fill="none"[^>]*><\/rect>\s*<\/svg>vLLM · not measured on this instrument/);
    expect(page).toContain('class="gate-panel-title">Atlas vs vLLM · gate instrument · ISL 512 / OSL 200 · natural fixture · batch cap 16 · fp8 KV<');
    inOrder(
      page,
      '<strong>vLLM has not been run on this instrument</strong> (ISL 512 / OSL 200 · natural fixture · batch cap 16 · fp8 KV)',
      'ISL 128 / OSL 1024 at batch cap 128',
      'not comparable in either direction',
      'filed under <code>bench/baselines/qwen38-27b-dflash/</code>, fills the comparison'
    );
  });

  test('the gate sweep below carries the same instrument and the bridge names the record', () => {
    expect(page).toContain('class="gate-panel-title">latest gate sweep · ISL 512 / OSL 200 · natural fixture · batch cap 16 · fp8 KV<');
    expect(page).toContain(`newest passing run on main (${fmtDate(live.recorded_at)} · ${live.git_sha})`);
  });
});

describe('the dense tab today: the live gate record paired with the published vLLM bar', () => {
  // ★ WHAT SHIPS SINCE #1220. The newest passing concurrency-sweep record on
  // main is on the published instrument (ISL 128 / OSL 1024 / essay / ctx
  // 2048 / batch 128 / fp8 KV), so the top chart draws IT against the
  // vLLM+MTP one-shot instead of the frozen August Atlas series, and the two
  // charts are one instrument. Every expected string below is derived from
  // the record or the manifest; the only typed prose is the component's own.
  const page = renderTab('qwen38-27b');
  const live = liveRecordOf(recordsFor('concurrency-sweep'));
  const mtp = publishedLadder.series.find((s) => s.id === 'vllm-mtp');
  const nospec = publishedLadder.series.find((s) => s.id === 'vllm-nospec');
  const days = (s) => s.rungs.map((r) => r.measured_utc.slice(0, 10)).sort();
  const range = (s) => (days(s)[0] === days(s).at(-1) ? days(s)[0] : `${days(s)[0]} → ${days(s).at(-1)}`);
  const instrument = 'ISL 128 / OSL 1024 · essay fixture · batch cap 128 · fp8 KV';

  test('the live record is on the published instrument and the state is live', () => {
    expect(live).toBe(denseLatest);
    expect(live.params.prompt_mode).toBe('essay');
    expect(Comparison.comparisonStateOf(DENSE, recordsFor('concurrency-sweep'))).toBe('live');
  });

  test('both charts are titled with ONE instrument', () => {
    expect(page).toContain(`class="gate-panel-title">Atlas vs vLLM · gate instrument · ${instrument}</span>`);
    expect(page).toContain(`class="gate-panel-title">latest gate sweep · ${instrument}<`);
    expect(tile(page, 'vLLM baseline')).toBe(`one-shot · ${range(mtp)}`);
    expect(tile(page, 'rungs declared')).toBe(`${rungs.length} of ${rungs.length}`);
  });

  test('the bridge says the charts share an instrument and names the record — never the old 4x sentence', () => {
    expect(page).toContain(
      `Same instrument as the chart above, which draws the newest passing run on main (${fmtDate(live.recorded_at)} · ${live.git_sha}); this chart adds the run history around it.`
    );
    expect(page).not.toContain("Not the chart above's instrument");
    expect(page).not.toContain('never one against the other');
  });

  test('the gate record\'s numbers are drawn on the comparison chart; the frozen August Atlas series is not', () => {
    const svg = comparisonSvg(page);
    const gateC128 = (+live.metrics.c128_aggregate_tok_s).toFixed(1);
    expect(svg).toContain(`<title>Atlas · C=128 · ${gateC128} tok/s · gate record ${fmtDate(live.recorded_at)} · ${live.git_sha} · click for record</title>`);
    expect(svg).not.toContain('478.11');
    expect(svg.match(/class="gc-mark"/g)).toHaveLength(ladderPoints(live).length);
    expect(svg.match(/class="cmp-sq"/g)).toHaveLength(mtp.rungs.length);
    expect(gateSvg(page)).toContain(gateC128);
    expect(page).not.toContain('<figure class="cl-panel">');
  });

  test('legend chips: Atlas is the live record, vLLM+MTP a dated one-shot, the no-speculation leg refused by name', () => {
    expect(page).toContain(`Atlas · live · latest gate ${fmtDate(live.recorded_at)} · ${live.git_sha}`);
    expect(page).toContain(`${mtp.label} · one-shot · measured ${range(mtp)} · not re-run`);
    expect(page).toContain(`${nospec.label} · other instrument · not drawn`);
  });

  test('the caption says "once" in bold, names engine, build, box, the instrument and the record, and why nospec is not drawn', () => {
    inOrder(
      page,
      `vLLM was measured <strong>once</strong>, on ${range(mtp)}`,
      `<code>${mtp.engine} (${mtp.build})</code> on ${publishedLadder.box.name}`,
      `on this instrument (${instrument})`,
      'not re-measured when Atlas moves',
      `Atlas is the newest passing run on main (${fmtDate(live.recorded_at)} · <code>${live.git_sha}</code>)`,
      `The ${nospec.label} one-shot of ${range(nospec)} is on another instrument and is not drawn: max_model_len`
    );
  });
});

describe('the dense tab FALLBACK: the published pair when no live record pairs', () => {
  // Rendered on the retired-instrument records only (rfRetired): the state the
  // tab shipped in until #1220, and the state it returns to if the live series
  // ever stops pairing. Nothing here is typed from the live record.
  const page = renderTab('qwen38-27b', rfRetired);
  // ★ WHAT THE LADDER DRAWS, not every series in the manifest. A leg with
  // `scope: 'cost'` (vllm-mtp-energy: same engine and instrument as vllm-mtp,
  // re-measured with power sampling) is read by cost.js and deliberately NOT
  // drawn here — it would be a near-duplicate line that stops at C=16 on a
  // chart whose claim is about throughput. Filtering the same way the
  // component does keeps this test tracking the component rather than the
  // manifest's length.
  const series = publishedLadder.series.filter((s) => s.scope !== 'cost');
  const days = (s) => s.rungs.map((r) => r.measured_utc.slice(0, 10)).sort();
  const range = (s) => (days(s)[0] === days(s).at(-1) ? days(s)[0] : `${days(s)[0]} → ${days(s).at(-1)}`);

  test('each chart is titled with its own instrument', () => {
    expect(page).toContain(
      `class="gate-panel-title">Atlas vs vLLM · published campaign · ISL ${publishedLadder.workload.isl_tokens} / OSL ${publishedLadder.workload.osl_tokens}</span>`
    );
    expect(page).toContain('class="gate-panel-title">latest gate sweep · ISL 512 / OSL 320 · natural fixture · batch cap 128 · fp8 KV<');
    expect(tile(page, 'vLLM baseline')).toBe('published pair');
    expect(tile(page, 'rungs declared')).toBe(`${rungs.length} of ${rungs.length}`);
  });

  test('the ~4x gap is explained in words between the two charts', () => {
    inOrder(
      page,
      "Not the chart above's instrument: this is the gate's <code>ISL 512 / OSL 320",
      'ISL 128 / OSL 1024 with speculation pinned',
      'never one against the other'
    );
  });

  test('neither instrument\'s number is drawn on the other\'s axes', () => {
    const ladder = ladderFigure(page);
    const gateC128 = (+retiredLatest.metrics.c128_aggregate_tok_s).toFixed(1);
    expect(ladder).toContain('478.11');
    expect(ladder).not.toContain(gateC128);
    const gate = gateSvg(page);
    expect(gate).toContain(gateC128);
    expect(gate).not.toContain('478.11');
  });

  test('legend chips: Atlas is the campaign, every vLLM leg is a dated one-shot, dates from the series', () => {
    const atlas = series.find((s) => s.role === 'subject');
    expect(page).toContain(`>Atlas · published campaign · ${range(atlas)}</button>`);
    for (const b of series.filter((s) => s.role === 'baseline'))
      expect(page).toContain(
        `<button type="button" class="cmp-chip" aria-pressed="true">${b.label} · one-shot · measured ${range(b)} · not re-run</button>`
      );
    expect(range(series.find((s) => s.id === 'vllm-nospec'))).toBe('2026-08-16'); // one-day series prints one date
  });

  test('the caption says "once" in bold, names engine, build, box and the Atlas build', () => {
    const b = series.find((s) => s.id === 'vllm-mtp');
    inOrder(
      page,
      'vLLM was measured <strong>once</strong>, on 2026-08-16 → 2026-08-18',
      `<code>${b.engine} (${b.build})</code> on ${publishedLadder.box.name}`,
      'not re-measured when Atlas moves',
      `<code>${series.find((s) => s.role === 'subject').build}</code>`,
      '"Latest gate sweep" below'
    );
  });

  test('the dated stamp is INSIDE the ladder svg, at the last rung of each vLLM line, and never overprints', () => {
    const svg = ladderSvg(page);
    const stamp = (date) => +new RegExp(`<text class="gc-ref-label cl-stamp" x="740" y="([\\d.]+)" text-anchor="end">vLLM 0.27.1 · ${date}</text>`).exec(svg)?.[1];
    const point = (label) => +new RegExp(`<circle cx="740" cy="([\\d.]+)"[^>]*>\\s*<title>${label} · C=128`).exec(svg)?.[1];
    expect(stamp('2026-08-17')).toBeGreaterThan(point('vLLM \\+ MTP')); // lowest line: stamped below its point
    expect(stamp('2026-08-16')).toBeLessThan(point('vLLM, no speculation')); // the other: above
    expect(svg.match(/cl-stamp/g)).toHaveLength(2);
  });
});

describe('ConcurrencyLadder takes the ladder it draws', () => {
  test('defaults to the published JSON, and a different ladder changes the copy', () => {
    expect(html(Ladder, {})).toContain('Faster than vLLM at every concurrency, C=1 to 128');
    const lost = structuredClone(publishedLadder);
    lost.summary = { ...lost.summary, all_won: false, won: 7 };
    expect(html(Ladder, { ladder: lost })).toContain('Atlas vs vLLM, C=1 to 128 — 7 of 8 rungs');
  });
});

describe('the comparison state, decided once', () => {
  const { comparisonStateOf, publishedFor, baselineOnlyFor, liveRecordOf, instrumentLabel, measuredRange, batchCapOf } = Comparison;
  const strip = ({ generated_utc, ...rest }) => rest;

  test('published only when the generated ladder is for THIS checkpoint and has a subject series', () => {
    expect(strip(publishedFor(DENSE))).toEqual(strip(publishedLadder));
    expect(publishedFor({ ...DENSE, checkpoint: 'Qwen/Qwen3.6-35B-A3B-FP8' })).toBeNull();
    expect(publishedFor(MOE)).toBeNull();
    expect(baselineOnlyFor(MOE).manifest).toBe(MOE.published_manifest);
    expect(baselineOnlyFor(DENSE)).toBeNull();
    expect(comparisonStateOf(DENSE, [])).toBe('published');
    expect(comparisonStateOf(DFLASH, recordsFor(DFLASH.gate))).toBe('live');
    expect(comparisonStateOf(MOE, recordsFor(MOE.gate).filter((r) => r.target_model === MOE.checkpoint))).toBe('baseline');
    expect(comparisonStateOf(MOE, [fakeRecord(MOE, [1])])).toBe('live');
    expect(comparisonStateOf({ ...MOE, published_manifest: null }, [])).toBe('none');
  });

  test('a declared manifest without a generated ladder is said out loud', () => {
    const page = html(
      Comparison.default,
      { subject: { ...MOE, id: 'moe-x', published_manifest: 'bench/moe/published.json' }, records: [], rungs, onselect: () => {} }
    );
    expect(page).toContain('<code>bench/moe/published.json</code> is declared for this subject but the');
    expect(page).toContain('No concurrency run on main yet');
  });

  test('the live record is the newest PASS on main — branch runs and FAILs never qualify', () => {
    const old = fakeRecord(MOE, [1], { recorded_at: 1, git_sha: 'aaaaaaaaaa' });
    const branch = fakeRecord(MOE, [1], { recorded_at: 2, branch: 'pr/x', git_sha: 'bbbbbbbbbb' });
    const fail = fakeRecord(MOE, [1], { recorded_at: 3, verdict: 'FAIL', git_sha: 'cccccccccc' });
    expect(liveRecordOf([old, branch, fail])).toBe(old);
    expect(liveRecordOf([branch, fail])).toBeNull();
  });

  test('instrument, range and batch-cap readers print only what is recorded', () => {
    // The newest dense record is on the re-pointed (published) instrument; the
    // retired one is still in the record set and still labels itself honestly.
    expect(instrumentLabel(denseLatest)).toBe('ISL 128 / OSL 1024 · essay fixture · batch cap 128 · fp8 KV');
    expect(instrumentLabel(retiredLatest)).toBe('ISL 512 / OSL 320 · natural fixture · batch cap 128 · fp8 KV');
    expect(instrumentLabel({ params: { isls: '512', osl: '320' } })).toBe('ISL 512 / OSL 320');
    expect(instrumentLabel({})).toBe('');
    expect(measuredRange([{ measured_utc: '2026-08-18T15:20:55Z' }, { measured_utc: '2026-08-17T01:36:35Z' }])).toBe('2026-08-17 → 2026-08-18');
    expect(measuredRange([{ measured_utc: '2026-08-16T14:38:29Z' }])).toBe('2026-08-16');
    expect(batchCapOf('vllm serve --max-num-seqs 128 --x')).toBe('128');
    expect(batchCapOf('spark serve --max-batch-size 16')).toBe('16');
    expect(batchCapOf('spark serve')).toBeNull();
    expect(batchCapOf(undefined)).toBeNull();
  });
});

describe('the dashboard wires the hash to the subject', () => {
  const open = (hash) => {
    globalThis.location = { hash, pathname: '/engine', search: '' };
    try {
      return html(Dashboard, { onclose: () => {} });
    } finally {
      delete globalThis.location;
    }
  };

  test('a concurrency deep link selects the subject tab it names and hides the model select', () => {
    const page = open('#bench=concurrency&subject=qwen36-35b-a3b&c=64');
    expect(page).toMatch(/id="bd-tab-concurrency"[^>]*aria-selected="true"/);
    expect(page).toMatch(/id="cs-tab-qwen36-35b-a3b"[^>]*aria-selected="true"/);
    expect(page).toContain('id="cs-panel-qwen36-35b-a3b"');
    expect(page).not.toContain('aria-label="Filter by model"');
    // The subject tabs replace the per-bench section: one gate sweep, not two.
    expect(page.match(/latest gate sweep/g) ?? []).toHaveLength(0);
    expect(page).not.toContain('<article class="gbs"');
  });

  test('an unknown subject in the link lands on the first subject, never a blank panel', () => {
    const page = open('#bench=concurrency&subject=nvidia-35b');
    expect(page).toMatch(/id="cs-tab-qwen38-27b"[^>]*aria-selected="true"/);
    expect(page).toContain('Atlas vs vLLM · gate instrument · ISL 128 / OSL 1024');
  });

  test('other tabs keep the model select and the per-bench sections', () => {
    const page = open('#bench=ttft');
    expect(page).toContain('aria-label="Filter by model"');
    expect(page).toContain('<article class="gbs"');
    expect(page).not.toContain('role="tablist" aria-label="Concurrency subjects"');
  });
});

// The owner's screenshot: three legend pills that looked pressable and did
// nothing, a table cut off after C=2, and a top card narrower than the gate
// sweep under it. The pills are now buttons that drive the ladder through
// series-visibility.js; the two cards share .gate-panel. The clipping and the
// width were CSS (chat.css's `.cc`), which concurrency-css.test.js guards —
// SSR cannot see a stylesheet, so the table check here proves only that every
// rung is in the markup, never that it is on screen.
describe('the published pair: series pills and the ladder they drive', () => {
  // The published card is the FALLBACK state now (see rfRetired above); the
  // pills and the ladder they drive are unchanged and still ship in it.
  const page = renderTab('qwen38-27b', rfRetired);
  // ★ WHAT THE LADDER DRAWS, not every series in the manifest. A leg with
  // `scope: 'cost'` (vllm-mtp-energy: same engine and instrument as vllm-mtp,
  // re-measured with power sampling) is read by cost.js and deliberately NOT
  // drawn here — it would be a near-duplicate line that stops at C=16 on a
  // chart whose claim is about throughput. Filtering the same way the
  // component does keeps this test tracking the component rather than the
  // manifest's length.
  const series = publishedLadder.series.filter((s) => s.scope !== 'cost');
  const subject = series.find((s) => s.role === 'subject');
  const rows = (h) => [...h.matchAll(/<th scope="row" class="mono">(\d+)<\/th>/g)].map((m) => +m[1]);
  const cols = (h) => [...h.matchAll(/<th scope="col">([^<]*)<\/th>/g)].map((m) => m[1]);
  const topTick = (h) =>
    Math.max(...[...ladderSvg(h).matchAll(/<text class="gc-axis" x="54" y="[\d.]+" text-anchor="end">(\d+)<\/text>/g)].map((m) => +m[1]));
  const cells = (h, c) =>
    (new RegExp(`<th scope="row" class="mono">${c}</th>((?:\\s*<td[^>]*>[^<]*</td>)+)`).exec(h)?.[1].match(/<td/g) ?? []).length;
  const mainTable = (h) => h.slice(h.indexOf('<table class="cl-table">'), h.indexOf('</table>'));

  test('every rung of the ladder is a row of the table, C=1 to 128, in order', () => {
    expect(rows(mainTable(page))).toEqual(publishedLadder.concurrencies);
    expect(rows(mainTable(page))).toEqual([1, 2, 4, 8, 16, 32, 64, 128]);
  });

  test('the published card is the same chrome as the gate sweep: one figure.gate-panel each, head in a figcaption', () => {
    expect(page).toContain(
      '<figure class="gate-panel cmp"><figcaption class="gate-panel-head"><span class="gate-panel-title">Atlas vs vLLM · published campaign'
    );
    expect(page).toContain(
      '<figure class="gate-panel"><figcaption class="gate-panel-head"><span class="gate-panel-title">latest gate sweep'
    );
    // Nothing in the tab carries the chat modal's class family.
    expect(page).not.toMatch(/class="[^"]*\bcc(-[a-z]+)?\b/);
  });

  test('each series pill is a real button, pressed, one per series, and the group is named', () => {
    const at = page.indexOf('<div class="cmp-keys"');
    const keys = page.slice(at, page.indexOf('</div>', at));
    expect(keys).toContain('role="group" aria-label="Series drawn"');
    expect(keys.match(/<button type="button" class="cmp-chip"/g)).toHaveLength(series.length);
    expect(keys.match(/aria-pressed="true"/g)).toHaveLength(series.length);
    expect(keys).not.toContain('<span');
    expect(page).not.toContain('cmp-refused'); // nothing has been refused yet
  });

  // The pill's onclick is `toggleSeries` (series-visibility.test.js) fed back
  // into the ladder's `hidden` prop; the harness has no DOM to press it in, so
  // the prop is driven directly and the rendered result is asserted.
  test('hiding a vLLM series removes its line, its marks, its legend key and its table column; the rest stays', () => {
    const before = html(Ladder, { embedded: true });
    const after = html(Ladder, { embedded: true, hidden: ['vllm-mtp'] });
    expect(before.match(/<path /g)).toHaveLength(series.length);
    expect(after.match(/<path /g)).toHaveLength(series.length - 1);
    expect(before).toContain('<title>vLLM + MTP · C=128');
    expect(after).not.toContain('<title>vLLM + MTP · C=128');
    expect(after).toContain('<title>vLLM, no speculation · C=128');
    expect(after).toContain('<title>Atlas · C=128');
    expect(cols(mainTable(before))).toEqual(['C', 'Atlas', 'vLLM + MTP', 'vLLM, no speculation', 'Ratio']);
    expect(cols(mainTable(after))).toEqual(['C', 'Atlas', 'vLLM, no speculation', 'Ratio']);
    expect(cells(mainTable(before), 128)).toBe(4);
    expect(cells(mainTable(after), 128)).toBe(3);
    expect(rows(mainTable(after))).toEqual([1, 2, 4, 8, 16, 32, 64, 128]); // rows never go
    expect(after.match(/class="cl-key"/g)).toHaveLength(series.length - 1);
    expect(after).not.toContain('vLLM 0.27.1 · 2026-08-17</text>'); // its in-plot stamp goes with it
  });

  test('hiding the tall series rescales the y axis to what is still drawn', () => {
    const before = html(Ladder, { embedded: true });
    const after = html(Ladder, { embedded: true, hidden: [subject.id] });
    const max = (ids) =>
      Math.max(...series.filter((s) => ids.includes(s.id)).flatMap((s) => s.rungs.map((r) => r.tok_s)));
    expect(topTick(before)).toBe(Math.round(max(series.map((s) => s.id)) * 1.08));
    expect(topTick(after)).toBe(Math.round(max(['vllm-mtp', 'vllm-nospec']) * 1.08));
    expect(topTick(after)).toBeLessThan(topTick(before));
    // Atlas gone: its column and the ratio (Atlas over vLLM) go together.
    expect(cols(mainTable(after))).toEqual(['C', 'vLLM + MTP', 'vLLM, no speculation']);
    expect(mainTable(after)).not.toContain('cl-ratio');
  });

  test('every series hidden is a wiring bug: the ladder throws rather than drawing empty axes', () => {
    expect(() => html(Ladder, { embedded: true, hidden: series.map((s) => s.id) })).toThrow(/every series is hidden/);
  });

  test('NEGATIVE CONTROL: an id that hides nothing changes nothing', () => {
    expect(html(Ladder, { embedded: true, hidden: ['no-such-series'] })).toBe(html(Ladder, { embedded: true }));
  });
});

// ---------------------------------------------------------------------------
// The metric is called ITL on the page (#1218)
//
// "Call it ITL, and mention in the bottom disclaimer that it may be viewed as
// TPOT." The record keys deliberately stay `tpot_p50_ms` — renaming a recorded
// key would orphan every measurement already committed — so the rename lives in
// the VIEW only, and these pin both halves of that split.
// ---------------------------------------------------------------------------
describe('ITL naming', () => {
  const page = renderTab('qwen36-35b-a3b');

  test('the column is headed ITL, not TPOT', () => {
    expect(page).toContain('>ITL p50<');
    expect(page).not.toContain('>TPOT p50<');
  });

  test('the disclaimer names TPOT as the other name for it, and gives the definition', () => {
    const t = text(page);
    expect(t).toContain('ITL');
    expect(t).toContain('TPOT');
    expect(t).toContain('(request_latency − TTFT) / (OSL − 1)');
  });

  // ★ BOTH COMPONENTS, BECAUSE ONE TEST COVERED ONLY ONE OF THEM. `renderTab`
  // for a subject with no Atlas leg renders ConcurrencyBaseline; the ITL column
  // in ConcurrencyLadder went untested, and reverting ITL->TPOT there left the
  // suite green. Mutation-checked: reverting the header in EITHER component now
  // turns exactly one of these red.
  test('ConcurrencyLadder heads its column ITL too, and carries the disclaimer', () => {
    const page = html(Ladder, {});
    expect(page).toContain('>ITL p50<');
    expect(page).not.toContain('>TPOT p50<');
    const t = text(page);
    expect(t).toContain('TPOT');
    expect(t).toContain('(request_latency − TTFT) / (OSL − 1)');
  });

  // ★ THE CONTROL FOR THE SPLIT. The view renames; the data must not. If a
  // later tidy-up renames the record key too, every committed record is
  // orphaned — so assert the key is still what the records carry.
  test('the record key is still tpot_p50_ms — the rename is display-only', () => {
    const rung = publishedLadder?.subjects
      ? Object.values(publishedLadder.subjects).flatMap((s) =>
          Object.values(s.series ?? {}).flatMap((x) => x.rungs ?? [])
        )[0]
      : null;
    if (rung) expect(Object.keys(rung)).toContain('tpot_p50_ms');
    expect(page).not.toContain('itl_p50_ms');
  });
});
