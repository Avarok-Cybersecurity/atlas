// SPDX-License-Identifier: AGPL-3.0-only

// ladder-build.mjs — one published ladder from one manifest and its raw files.
//
// Pure: every raw file is read through `rawOf(file)`, and every violation is a
// thrown Error naming the file, the series and the rung. gen-ladder.mjs turns
// the throw into a build failure; the bun tests call this directly, against
// the committed manifests and against mutated copies, so each guard below is
// proven to bite rather than assumed to.
//
// The manifest names files, never numbers. Every figure the site shows —
// mean, median, spread, TTFT, TPOT, the measured date, the harness hash — is
// COMPUTED here from the raw harness output. A manifest that carries a
// `rungs`, `tok_s`, `rows` or `summary` key is refused outright: a typed
// number is how a chart quietly stops matching its receipts.
//
// Guards, each of which a test breaks on purpose:
//   * a raw file's header (model, isl, osl, reps, warmup, temperature, seed)
//     must equal the manifest's `workload` — the manifest describes the
//     instrument, the raw file proves it;
//   * the set of driver sha256 prefixes across the raw files must EQUAL the
//     keys of `harness_shas` — an unlisted revision is the silent
//     incomparability this file exists to prevent, and a listed one no file
//     carries is a phantom;
//   * every baseline declares an `instrument` (what ladder-baselines.js
//     fingerprints) that agrees with the workload and with its own command
//     line, and explains itself when the command spells a KV dtype
//     differently;
//   * a rung listed as `unmeasured` is absent everywhere: not in `sources`,
//     not in any raw file the series names (PCND — an absent rung has no
//     value, not a zero);
//   * a subject is optional (a baseline-only ladder is a state, not an error)
//     but when present every baseline must cover every subject rung.

export const KNOWN_ROLES = Object.freeze(['subject', 'baseline', 'variant']);

const TYPED_NUMBER_KEYS = Object.freeze(['rungs', 'tok_s', 'rows', 'summary']);

// Raw header field → manifest.workload field. The values are compared as
// JSON values, so `0.0` in the manifest equals `0.0` in the raw file.
const HEADER_AXES = Object.freeze([
  ['model', 'checkpoint'],
  ['isl', 'isl_tokens'],
  ['osl', 'osl_tokens'],
  ['reps', 'reps'],
  ['warmup', 'warmup'],
  ['temperature', 'temperature'],
  ['seed', 'seed']
]);

// Instrument field → manifest.workload field, for the axes both declare.
const INSTRUMENT_WORKLOAD = Object.freeze([
  ['isl', 'isl_tokens'],
  ['osl', 'osl_tokens'],
  ['reps', 'reps'],
  ['warmup', 'warmup'],
  ['temperature', 'temperature'],
  ['seed', 'seed']
]);

const REQUIRED_INSTRUMENT = Object.freeze(['max_model_len', 'max_batch_size', 'kv_cache_dtype']);

const fail = (msg) => {
  throw new Error(`gen-ladder: ${msg}`);
};

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
export const r2 = (v) => Math.round(v * 100) / 100;
// Ratios keep a third decimal: at C=2 the margin is 1.004x, and rounding that
// to 1.00x would present a real (if slim) win as a tie.
export const r3 = (v) => Math.round(v * 1000) / 1000;

const isPosInt = (v) => Number.isInteger(v) && v > 0;

/** The value a CLI flag was given, or null when the command does not carry it. */
export function cliFlag(cli, ...names) {
  for (const name of names) {
    const m = new RegExp(`(?:^|\\s)${name}\\s+(\\S+)`).exec(cli ?? '');
    if (m) return m[1];
  }
  return null;
}

function checkHeader(doc, file, workload) {
  for (const [raw, wl] of HEADER_AXES) {
    if (JSON.stringify(doc[raw]) !== JSON.stringify(workload[wl]))
      fail(`${file} header ${raw}=${JSON.stringify(doc[raw])} != workload.${wl}=${JSON.stringify(workload[wl])}`);
  }
  if (typeof doc.started_utc !== 'string' || doc.started_utc === '')
    fail(`${file} has no started_utc — a one-shot without a date cannot be stamped`);
  if (typeof doc.driver_sha256 !== 'string' || doc.driver_sha256.length < 10)
    fail(`${file} has no driver_sha256 — which harness produced it is unknown`);
}

/** One rung of one series: the reps for concurrency `c` inside `doc`. */
/**
 * The GPU-rail energy of one rung, when the harness recorded it.
 *
 * Joules and tokens ADD across reps (that is why the producer stores them and
 * not a ratio); the sample count takes the WORST rep, because the trust rule
 * the page applies must not be flattered by a well-sampled sibling. The key
 * names are the producer's own (`EnergyWindow::metrics`), carried unchanged so
 * a vLLM rung and an Atlas gate cell are read by one function on the page.
 *
 * Three refusals, none of which can be reached by an absence:
 *   * no rep carries energy -> the rung simply has none (PCND: absent is not
 *     zero, and an energyless rung draws no cost point);
 *   * SOME reps carry it -> refused, because a partial sum is not a
 *     measurement of the rung;
 *   * a rep's joules, tokens or window are not positive numbers -> refused by
 *     file, series and rung.
 */
function rungEnergy(seriesId, c, file, reps) {
  const where = `${file} rung C=${c} (series ${seriesId})`;
  const carried = reps.filter((r) => r.gpu_rail_energy_j !== undefined && r.gpu_rail_energy_j !== null);
  if (carried.length === 0) return {};
  if (carried.length !== reps.length)
    fail(`${where} recorded energy on ${carried.length} of ${reps.length} reps — a partial sum is not a measurement`);
  const positive = (r, k) => {
    const v = r[k];
    if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) fail(`${where} has ${k} = ${JSON.stringify(v)}`);
    return v;
  };
  const sum = (k) => reps.reduce((a, r) => a + positive(r, k), 0);
  // A joule count is only comparable to another taken at the same cadence, so
  // a rung whose reps disagree about it is refused rather than averaged.
  const periods = [...new Set(reps.map((r) => r.gpu_rail_sample_period_ms).filter((v) => v !== undefined && v !== null))];
  if (periods.length > 1) fail(`${where} mixes sampler cadences: ${periods.join(', ')} ms`);
  const samples = reps.map((r) => r.gpu_rail_power_samples).filter((v) => typeof v === 'number');
  // ★ THE WINDOW AND ITS DENOMINATOR HAVE TWO SPELLINGS, ONE PER PRODUCER.
  //
  //   RUST   `EnergySampler::metrics`      gpu_rail_energy_window_s
  //                                        gpu_rail_energy_window_tokens
  //          -> reaches the page through gen-gates.mjs, as a GATE RECORD
  //   PYTHON `bench/ladder38/power_window.py` + the ladder harness
  //                                        gpu_rail_window_s
  //                                        completion_tokens (on the rep)
  //          -> reaches the page through THIS function, as a BASELINE rung
  //
  // `rungEnergy` only ever sees Python-produced rungs, and until 2026-09-21 it
  // demanded the Rust spelling — keys nothing under bench/ has ever written.
  // `positive()` then failed the whole build on the first vLLM leg that
  // carried joules, which is why no vLLM baseline had ever had energy and the
  // Cost tab's "comparable but carries no joules" was structural, not a gap in
  // the data. Accept either spelling; a rung whose reps disagree about WHICH is
  // refused, because that is two measurements wearing one name.
  const pick = (ks) => {
    const found = [...new Set(reps.map((r) => ks.find((k) => typeof r[k] === 'number')))];
    if (found.length !== 1 || found[0] === undefined)
      fail(`${where} reps disagree about the energy-window key (looked for ${ks.join(' / ')})`);
    return found[0];
  };
  const winKey = pick(['gpu_rail_energy_window_s', 'gpu_rail_window_s']);
  const tokKey = pick(['gpu_rail_energy_window_tokens', 'completion_tokens']);
  return {
    gpu_rail_energy_j: r2(sum('gpu_rail_energy_j')),
    gpu_rail_energy_window_tokens: sum(tokKey),
    gpu_rail_energy_window_s: r2(sum(winKey)),
    ...(samples.length === reps.length ? { gpu_rail_power_samples: Math.min(...samples) } : {}),
    ...(periods.length === 1 ? { gpu_rail_sample_period_ms: periods[0] } : {})
  };
}

export function rungStats(seriesId, c, file, doc) {
  const rung = (doc.rungs ?? []).find((r) => r.concurrency === c);
  if (!rung) fail(`${file} has no rung for C=${c} (series ${seriesId})`);
  const reps = rung.reps ?? [];
  if (reps.length === 0) fail(`${file} rung C=${c} has no reps (series ${seriesId})`);

  const tok = reps.map((r) => r.tok_s);
  if (tok.some((v) => typeof v !== 'number' || !Number.isFinite(v)))
    fail(`${file} rung C=${c} has a non-numeric tok_s (series ${seriesId})`);
  const ttft = reps.map((r) => r.ttft_p50_ms).filter((v) => typeof v === 'number');
  const tpot = reps.map((r) => r.tpot_p50_ms).filter((v) => typeof v === 'number');
  const errs = reps.reduce((a, r) => a + (r.n_err ?? 0), 0);
  if (errs > 0) fail(`${file} rung C=${c} recorded ${errs} request errors — not publishable`);

  return {
    c,
    ...rungEnergy(seriesId, c, file, reps),
    tok_s: r2(mean(tok)),
    tok_s_median: r2(median(tok)),
    // Spread as a share of the mean: how much the rung moved run to run. A
    // published number whose reps disagree by 10% deserves to say so.
    spread_pct: r2(((Math.max(...tok) - Math.min(...tok)) / mean(tok)) * 100),
    reps: reps.length,
    ttft_p50_ms: ttft.length ? r2(median(ttft)) : null,
    tpot_p50_ms: tpot.length ? r2(median(tpot)) : null,
    source: file,
    measured_utc: doc.started_utc,
    harness_sha256: doc.driver_sha256.slice(0, 10)
  };
}

function checkInstrument(s, workload) {
  const inst = s.instrument;
  if (!inst || typeof inst !== 'object') fail(`baseline ${s.id} declares no instrument — nothing to fingerprint`);
  for (const [k, wl] of INSTRUMENT_WORKLOAD) {
    if (JSON.stringify(inst[k]) !== JSON.stringify(workload[wl]))
      fail(`baseline ${s.id} instrument.${k}=${JSON.stringify(inst[k])} != workload.${wl}=${JSON.stringify(workload[wl])}`);
  }
  for (const k of REQUIRED_INSTRUMENT) {
    if (inst[k] === undefined || inst[k] === null || inst[k] === '')
      fail(`baseline ${s.id} instrument.${k} is undeclared`);
  }
  const cap = cliFlag(s.cli, '--max-num-seqs', '--max-batch-size');
  if (cap !== String(inst.max_batch_size))
    fail(`baseline ${s.id} instrument.max_batch_size=${inst.max_batch_size} but its cli says ${cap ?? 'nothing'}`);
  const ctx = cliFlag(s.cli, '--max-model-len', '--max-seq-len');
  if (ctx !== String(inst.max_model_len))
    fail(`baseline ${s.id} instrument.max_model_len=${inst.max_model_len} but its cli says ${ctx ?? 'nothing'}`);
  const kv = cliFlag(s.cli, '--kv-cache-dtype');
  if (kv !== inst.kv_cache_dtype && !(typeof s.instrument_note === 'string' && s.instrument_note.trim()))
    fail(`baseline ${s.id} instrument.kv_cache_dtype=${inst.kv_cache_dtype} but its cli says ${kv ?? 'nothing'}; add instrument_note saying why`);
}

function checkUnmeasured(s, docs) {
  const u = s.unmeasured;
  if (u === undefined) return;
  if (!u || !Array.isArray(u.rungs) || u.rungs.length === 0 || !u.rungs.every(isPosInt))
    fail(`series ${s.id} unmeasured.rungs must be a non-empty list of rungs`);
  if (typeof u.reason !== 'string' || !u.reason.trim()) fail(`series ${s.id} unmeasured.reason is required`);
  for (const c of u.rungs) {
    if (s.sources[String(c)] !== undefined) fail(`series ${s.id} lists C=${c} as unmeasured but sources names a file for it`);
    for (const [file, doc] of docs) {
      if ((doc.rungs ?? []).some((r) => r.concurrency === c))
        fail(`series ${s.id} lists C=${c} as unmeasured but ${file} contains it`);
    }
  }
}

function buildSeries(s, manifest, rawOf, docs) {
  for (const k of TYPED_NUMBER_KEYS) {
    if (k in s) fail(`series ${s.id} carries a typed "${k}" — numbers come from raw files, never the manifest`);
  }
  if (!KNOWN_ROLES.includes(s.role)) fail(`series ${s.id} has unknown role ${JSON.stringify(s.role)}`);
  if (!s.sources || typeof s.sources !== 'object' || Object.keys(s.sources).length === 0)
    fail(`series ${s.id} names no sources`);
  const rungs = Object.entries(s.sources)
    .map(([c, file]) => {
      const cn = Number(c);
      if (!isPosInt(cn)) fail(`series ${s.id} has a non-rung source key ${JSON.stringify(c)}`);
      if (!docs.has(file)) {
        const doc = rawOf(file);
        checkHeader(doc, file, manifest.workload);
        docs.set(file, doc);
      }
      return rungStats(s.id, cn, file, docs.get(file));
    })
    .sort((a, b) => a.c - b.c);
  if (s.role === 'baseline') checkInstrument(s, manifest.workload);
  checkUnmeasured(s, [...docs].filter(([file]) => Object.values(s.sources).includes(file)));
  return { ...s, sources: undefined, rungs };
}

function checkHarnessShas(manifest, docs) {
  const listed = Object.keys(manifest.harness_shas ?? {}).filter((k) => k !== 'equivalence');
  const carried = [...new Set([...docs.values()].map((d) => d.driver_sha256.slice(0, 10)))];
  for (const sha of carried) {
    if (!listed.includes(sha)) {
      const files = [...docs].filter(([, d]) => d.driver_sha256.startsWith(sha)).map(([f]) => f);
      fail(`harness revision ${sha} produced ${files.join(', ')} but is not listed in harness_shas`);
    }
  }
  for (const sha of listed) {
    if (!carried.includes(sha)) fail(`harness_shas lists ${sha} but no raw file was produced by it`);
  }
}

/**
 * @param {object} manifest   a published.json document
 * @param {object} io
 * @param {{id:string, checkpoint:string, published_manifest:string}} io.subject
 * @param {(file:string) => object} io.rawOf  parsed raw harness JSON by file name
 * @param {string} io.harnessRepoSha256  hex sha256 of the harness file in the tree
 */
export function buildLadder(manifest, { subject, rawOf, harnessRepoSha256 }) {
  if (manifest.schema !== 1) fail(`${subject.published_manifest} has schema ${manifest.schema}, expected 1`);
  for (const k of TYPED_NUMBER_KEYS) {
    if (k in manifest) fail(`${subject.published_manifest} carries a typed "${k}" — numbers come from raw files`);
  }
  if (manifest.workload?.checkpoint !== subject.checkpoint)
    fail(`${subject.published_manifest} is for ${manifest.workload?.checkpoint}, subject ${subject.id} is ${subject.checkpoint}`);
  if (!Array.isArray(manifest.series) || manifest.series.length === 0)
    fail(`${subject.published_manifest} has no series`);

  const docs = new Map();
  const series = manifest.series.map((s) => buildSeries(s, manifest, rawOf, docs));
  checkHarnessShas(manifest, docs);

  const subjects = series.filter((s) => s.role === 'subject');
  if (subjects.length > 1) fail('manifest has more than one subject series');
  const subj = subjects[0] ?? null;
  const baselines = series.filter((s) => s.role === 'baseline');
  if (baselines.length === 0) fail('manifest has no baseline series');
  const at = (s, c) => s.rungs.find((r) => r.c === c);

  const out = {
    manifest: subject.published_manifest,
    title: manifest.title,
    subtitle: manifest.subtitle,
    aggregate: manifest.aggregate,
    results_doc: manifest.results_doc,
    results_doc_url: `https://github.com/Avarok-Cybersecurity/atlas/blob/main/${manifest.results_doc}`,
    workload: manifest.workload,
    box: manifest.box,
    harness_shas: manifest.harness_shas,
    harness_repo_sha256: harnessRepoSha256,
    concurrencies: [...new Set(series.flatMap((s) => s.rungs.map((r) => r.c)))].sort((a, b) => a - b),
    series
  };
  if (manifest.subject_note !== undefined) out.subject_note = manifest.subject_note;
  if (!subj) return out;

  // A pair: score the subject against the MATCHED-parity baseline. Ratios and
  // the win count are derived, never asserted, so a lost rung changes the copy
  // instead of leaving it stale.
  // ★ EXACTLY ONE, not "the first one found". `matched` names THE leg the
  // published ratio is computed against, and a `find` over two claimants picks
  // by array order -- so adding a second matched baseline would silently move
  // every ratio on the page. Caught on 2026-09-21 by the `no matched-parity
  // baseline` test, which stopped failing because a newly added energy leg had
  // quietly become the reference.
  // The reference is chosen below, from the FULL-LADDER baselines only.
  // `variant`: another configuration of the SUBJECT engine. Drawn, never
  // scored, and held to the subject's rung coverage for the same reason a
  // baseline is: a line that stops partway along a log2 axis reads as a
  // measurement, not as a gap.
  for (const v of series.filter((s) => s.role === 'variant')) {
    for (const row of subj.rungs) if (!at(v, row.c)) fail(`variant ${v.id} is missing rung C=${row.c}`);
  }
  out.concurrencies = subj.rungs.map((r) => r.c);
  // ★ THE PUBLISHED TABLE USES FULL-LADDER LEGS ONLY. `out.rows` is what
  // ConcurrencyLadder, Verified and marketing.js render -- the per-rung
  // "Atlas vs vLLM" claim -- so a leg that stops partway would either break
  // the row or, worse, silently enter `ratio_vs_fastest` at the rungs it does
  // have and change a published ratio at some rungs but not others.
  //
  // A PARTIAL leg is still a first-class series: it keeps its `rungs`, which
  // is what cost.js reads for the energy curves, and its `unmeasured.reason`,
  // which absentReasonOf prints. It simply does not vote in the throughput
  // table. The 2026-09-21 energy leg is the first of these -- driven only at
  // C=1..16 because vLLM+MTP has taken a GB10 down at the widest rungs.
  //
  // A rung missing WITHOUT being declared unmeasured is still a failure: that
  // is a gap nobody wrote down, which is the thing this file exists to refuse.
  const covers = (b) => subj.rungs.every((row) => at(b, row.c));
  // Name the RUNG, not just the fact. The old check said
  // `baseline X is missing rung C=64`, and a diagnostic that loses the number
  // is a worse diagnostic even when the refusal is the same -- so the message
  // keeps that shape and adds why it was not forgiven.
  for (const b of baselines)
    for (const row of subj.rungs)
      if (!at(b, row.c) && !(b.unmeasured?.rungs ?? []).includes(row.c))
        fail(`baseline ${b.id} is missing rung C=${row.c} and never declared it unmeasured`);
  const tableBaselines = baselines.filter(covers);
  // ★ THE REFERENCE COMES FROM THE FULL-LADDER LEGS, not from every baseline.
  // `parity: matched` describes the INSTRUMENT -- a leg can match on every
  // axis and still not be the leg the published ratio is computed against.
  // Selecting from `baselines` made a partial energy leg compete for that role
  // by array order; selecting from `tableBaselines` means a leg that cannot
  // fill the table cannot become its denominator either.
  const matchedAll = tableBaselines.filter((s) => s.parity === 'matched');
  if (matchedAll.length === 0) fail('no matched-parity baseline');
  if (matchedAll.length > 1)
    fail(`more than one full-ladder matched-parity baseline (${matchedAll.map((s) => s.id).join(', ')}) — exactly one leg is the published ratio's reference`);
  const matched = matchedAll[0];
  out.rows = subj.rungs.map((row) => {
    const perBaseline = tableBaselines.map((b) => {
      const r = at(b, row.c);
      if (!r) fail(`baseline ${b.id} is missing rung C=${row.c}`);
      return { id: b.id, label: b.label, parity: b.parity, tok_s: r.tok_s };
    });
    const fastest = perBaseline.reduce((a, b) => (b.tok_s > a.tok_s ? b : a));
    const m = perBaseline.find((b) => b.id === matched.id);
    return {
      c: row.c,
      atlas: row.tok_s,
      baselines: perBaseline,
      best_baseline_id: m.id,
      ratio_vs_best: r3(row.tok_s / m.tok_s),
      ratio_vs_matched: r3(row.tok_s / m.tok_s),
      ratio_vs_fastest: r3(row.tok_s / fastest.tok_s),
      wins: row.tok_s > m.tok_s
    };
  });
  out.summary = {
    rungs: out.rows.length,
    won: out.rows.filter((r) => r.wins).length,
    all_won: out.rows.every((r) => r.wins),
    min_ratio: r3(Math.min(...out.rows.map((r) => r.ratio_vs_matched))),
    max_ratio: r3(Math.max(...out.rows.map((r) => r.ratio_vs_matched)))
  };
  return out;
}
