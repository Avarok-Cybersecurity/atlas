// SPDX-License-Identifier: AGPL-3.0-only

// repro-steps.js — turn ONE gate record into the ordered pipeline that produced
// its number, so a third party can replay it and check every intermediate.
//
// Two rules govern everything here, and the tests hold both:
//
//   1. Every command shown is READ from the record. The run command is
//      `record.command`, written by `crates/avarok-plugin/src/gate/record.rs`
//      from the recorded inputs; a shard partition's commands are its MEMBER
//      records' own `command` fields. Nothing is reconstructed from `params`.
//      The two flags the record does not carry (`--checkpoint`, `--hardware`,
//      both gate-only options of `spark benchmark run`) are appended and
//      reported in `derived[]`, so a reader sees exactly which tokens the
//      record did not contain.
//   2. A field the record lacks is NAMED in `missing[]`, with what the reader
//      would have to supply, and the step that needed it goes without. A
//      plausible value in its place is the failure this module exists to
//      prevent: a diligence run that reproduces a command nobody ran.
//
// Pure and dependency-free of the DOM so `bun test` can measure it directly.

import { shardOf } from './bfcl-partition.js';

export const ATLAS_REPO = 'https://github.com/Avarok-Cybersecurity/atlas';
export const RECIPES_REPO = 'https://github.com/Avarok-Cybersecurity/atlas-recipes';

/** Tokens a POSIX shell reads back unchanged without quoting. */
const SAFE_TOKEN = /^[A-Za-z0-9_@%+=:,./-]+$/;

/**
 * Quote one argv element for a POSIX shell.
 *
 * The record stores multi-value params as ONE element containing spaces
 * (`concurrencies=1, 2, 4, 8`); pasted unquoted that is eight arguments and
 * `--param` refuses. Single quotes, with the only escape single quotes need.
 * @param {string} tok
 */
export function shellQuote(tok) {
  const s = String(tok);
  if (s === '') return "''";
  return SAFE_TOKEN.test(s) ? s : `'${s.replace(/'/g, `'\\''`)}'`;
}

/**
 * Split an argv into its positional head and `[flag, value|null]` pairs.
 *
 * A token after a `--flag` is that flag's value unless it is itself a flag —
 * which is how `--yes` (a bare boolean) and `--pull-request-gate` come out.
 * @param {string[]} argv
 * @returns {{head: string[], pairs: Array<[string, string|null]>}}
 */
export function splitFlags(argv) {
  const head = [];
  /** @type {Array<[string, string|null]>} */
  const pairs = [];
  let i = 0;
  while (i < argv.length && !argv[i].startsWith('--')) head.push(argv[i++]);
  while (i < argv.length) {
    const flag = argv[i++];
    const next = argv[i];
    if (next !== undefined && !next.startsWith('--')) {
      pairs.push([flag, next]);
      i += 1;
    } else {
      pairs.push([flag, null]);
    }
  }
  return { head, pairs };
}

/**
 * Render an argv as shell lines: the head, then one flag per line, every line
 * but the last ending in a `\` continuation. Every token is shell-quoted.
 * @param {string[]} head
 * @param {Array<[string, string|null]>} pairs
 * @returns {string[]}
 */
export function renderLines(head, pairs) {
  const lines = [head.map(shellQuote).join(' ')];
  for (const [flag, value] of pairs) {
    lines.push(`  ${shellQuote(flag)}${value === null ? '' : ' ' + shellQuote(value)}`);
  }
  return lines.map((l, k) => (k < lines.length - 1 ? `${l} \\` : l));
}

/** @param {string[]} argv */
export function renderArgv(argv) {
  const { head, pairs } = splitFlags(argv);
  return renderLines(head, pairs);
}

/**
 * The box class a gate record ran on, or null.
 *
 * Read from `hardware_state.perf_class` (`gb10@spark-28c2`), the one field
 * that carries the class as a token. `hardware.gpu` ("NVIDIA GB10") is a
 * marketing string that `hardware/ids.rs` maps to a class in Rust; mirroring
 * that table here would be a second copy of it, so a record without a
 * `perf_class` names the class as not recorded instead.
 * @param {{perf_class?: string}} record
 */
export function hardwareClass(record) {
  const pc = typeof record?.perf_class === 'string' ? record.perf_class : '';
  const cls = pc.split('@')[0];
  return cls === '' ? null : cls;
}

/**
 * Where the recorded command disagrees with the recorded inputs.
 *
 * `record.rs` assembles `command` from `params` and `serve_overrides` and
 * always ends it with `--pull-request-gate`. A record whose command does not
 * name every recorded input is not one that function wrote, and the reader
 * must be told before replaying it. Returns one sentence per discrepancy;
 * empty means the command covers the record.
 * @param {{command?: string[], params?: object, serve_overrides?: object}} record
 * @returns {string[]}
 */
export function commandCoversRecord(record) {
  const cmd = Array.isArray(record?.command) ? record.command : null;
  if (!cmd) return ['no command recorded'];
  const out = [];
  const has = (flag, value) => cmd.some((t, k) => t === flag && cmd[k + 1] === value);
  for (const [k, v] of Object.entries(record.params ?? {})) {
    if (!has('--param', `${k}=${v}`)) out.push(`params.${k}=${v} is not in the command`);
  }
  for (const [k, v] of Object.entries(record.serve_overrides ?? {})) {
    if (!has('--serve-override', `${k}=${v}`)) out.push(`serve_overrides.${k}=${v} is not in the command`);
  }
  if (!cmd.includes('--pull-request-gate')) out.push('the command lacks --pull-request-gate: not a gate-mode invocation');
  return out;
}

/** Keys of `params` that are thresholds the verdict was judged against. */
const FLOOR_KEY = /^min_|^max_|_limit_pct$|_budget_s$/;

const isoDate = (unix) => new Date(unix * 1000).toISOString().slice(0, 10);
const fmtNum = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : +(+v).toFixed(2));

/**
 * The run command as it will be shown: the record's argv, plus the gate-only
 * flags the record cannot carry, appended and reported. Both flags are real
 * options of `spark benchmark run` (`cli/bench_args.rs`); both are consulted
 * only under `--pull-request-gate`, so neither is added to an endpoint-form
 * invocation (no `served_by`, `--url/--model` in the argv).
 * @returns {{lines: string[], derivedLines: number[], derived: Array<{flag: string, from: string}>}}
 */
function runCommand(rec, argv, cls) {
  const { head, pairs } = splitFlags(argv);
  const derived = [];
  const gateMode = argv.includes('--pull-request-gate') && Boolean(rec.served_by);
  const present = (flag) => pairs.some(([f]) => f === flag);
  if (gateMode && rec.target_model && !present('--checkpoint')) {
    pairs.push(['--checkpoint', rec.target_model]);
    derived.push({ flag: '--checkpoint', from: 'target_model' });
  }
  if (gateMode && cls && !present('--hardware')) {
    pairs.push(['--hardware', cls]);
    derived.push({ flag: '--hardware', from: 'perf_class' });
  }
  const lines = renderLines(head, pairs);
  const derivedLines = derived.map((_, k) => lines.length - derived.length + k);
  return { lines, derivedLines, derived };
}

/** `--param shard=i/n` swapped for another index — the only edit made. */
function withShard(argv, j, n) {
  return argv.map((t, k) => (argv[k - 1] === '--param' && /^shard=\d+\/\d+$/.test(t) ? `shard=${j}/${n}` : t));
}

/**
 * Build the plan.
 *
 * @param {object} record a slim record from gates.generated.json, possibly a
 *   partition aggregate from `bfcl-partition.js` (`record.partition` set)
 * @param {{generated: {sha: string, date: string}, meta: {expected_secs: number, sensitivity: string}|null, serve_allowance_s: number|null}} ctx
 *   `generated` is the dashboard's own provenance and is REQUIRED — a plan
 *   that cannot say which build published the point is not a plan; `meta`
 *   and `serve_allowance_s` are null when the generator found no descriptor /
 *   limit, and the costs line then says so instead of quoting a number.
 */
export function reproSteps(record, ctx) {
  if (!record || typeof record !== 'object') throw new TypeError('reproSteps: a record is required');
  if (!ctx?.generated?.sha || !ctx.generated.date) {
    throw new TypeError('reproSteps: ctx.generated {sha, date} is required');
  }
  const rec = record;
  const missing = [];
  const caveats = [];
  const derived = [];
  const miss = (field, need) => missing.push({ field, need });

  const sha = typeof rec.git_sha === 'string' && rec.git_sha.trim() !== '' ? rec.git_sha : null;
  if (!sha) miss('git_sha', 'the commit the binary was built from; without it there is nothing to check out');
  const path = typeof rec.path === 'string' && rec.path !== '' ? rec.path : null;
  if (!path) miss('path', 'where the record file lives under .benchmarks/, so its .sig and signer can be found');
  const partition = rec.partition && Array.isArray(rec.partition.members) ? rec.partition : null;
  const cls = hardwareClass(rec);
  if (!cls) {
    miss(
      'hardware class',
      `hardware_state.perf_class is empty, so --hardware cannot be derived; hardware.gpu reads "${rec.hardware?.gpu ?? '?'}" and with a single registered class the gate infers it`
    );
  }
  const dirty = Array.isArray(rec.dirty_paths) ? rec.dirty_paths : [];
  if (dirty.length) {
    caveats.push(
      `This record was measured with uncommitted changes to ${dirty.join(', ')}; the commit alone does not reproduce the binary.`
    );
  }
  if (!rec.signer) caveats.push('This record has no .sig sidecar: nothing proves the file was not altered after it was written.');

  // 1. checkout ---------------------------------------------------------------
  const checkout = {
    id: 'checkout',
    title: 'Checkout',
    summary: `${sha ?? 'no sha'} · branch ${rec.branch || 'main'} · in dashboard history: ${rec.generated_ancestry ?? 'unknown'}`,
    facts: [
      ['commit', sha ? `${ATLAS_REPO}/commit/${sha}` : 'not recorded'],
      ['source branch', rec.branch || 'committed on main'],
      ['in dashboard history', rec.generated_ancestry ?? 'unknown']
    ],
    commands: [],
    notes: []
  };
  if (sha) {
    const lines = [`git clone ${ATLAS_REPO}.git && cd atlas`];
    if (rec.generated_ancestry === 'unknown') {
      checkout.notes.push(
        `Commit ${sha} was not found in the history the dashboard was generated from; it may sit on a squash-merged branch. Locate it before checking out.`
      );
    } else {
      lines.push(`git fetch origin ${sha} && git checkout ${sha}`);
    }
    checkout.commands.push({ label: 'clone and check out', lines, derivedLines: [] });
    if (dirty.length) checkout.notes.push(`Insufficient on its own: ${dirty.length} path(s) were dirty at measurement time (see caveats).`);
  }

  // 2. build ------------------------------------------------------------------
  // Same lines as the deck's Reproduce act (deck/acts/Reproduce.svelte), which
  // cannot import from here yet; keep them identical until it does.
  const build = {
    id: 'build',
    title: 'Build',
    summary: `cargo build --release -p spark-server --bin spark · atlas ${rec.atlas_version ?? 'version not recorded'}`,
    facts: [['atlas_version', rec.atlas_version ?? 'not recorded']],
    commands: [
      {
        label: 'binary',
        lines: [
          'export PATH=/usr/local/cuda/bin:$PATH',
          'cargo build --release -p spark-server --bin spark',
          `./target/release/spark benchmark list ${shellQuote(rec.benchmark_id)}`
        ],
        derivedLines: []
      }
    ],
    notes: ['CUDA 13.0 must be on PATH or the cudarc build script fails before anything informative.']
  };
  if (!rec.atlas_version) miss('atlas_version', 'the binary version string; compare `spark --version` after building');

  // 3. serve ------------------------------------------------------------------
  const overrides = Object.entries(rec.serve_overrides ?? {});
  const env = Object.entries(rec.perf_env ?? {});
  const serve = {
    id: 'serve',
    title: 'Serve',
    summary: `${rec.served_by ?? 'no recipe recorded'} · ${overrides.length} override(s) · ${env.length} env`,
    facts: [],
    commands: [],
    notes: []
  };
  if (rec.served_by) {
    const until = typeof rec.recorded_at === 'number' ? isoDate(rec.recorded_at) : '';
    serve.facts.push(
      ['recipe', `${RECIPES_REPO}/blob/main/recipes/${rec.served_by}.yaml`],
      ['recipe history before this run', `${RECIPES_REPO}/commits/main/recipes/${rec.served_by}.yaml?until=${until}`]
    );
    serve.notes.push(
      'The record names the recipe but does not pin its revision; use the newest commit of the recipe file dated before this run.'
    );
  } else {
    miss('served_by', 'which serve recipe started the server; an endpoint-form run (--url/--model) measures whatever was listening');
  }
  for (const [k, v] of overrides) serve.facts.push([`override ${k}`, String(v)]);
  if (env.length) {
    serve.commands.push({
      label: 'environment (resolved values, defaults substituted)',
      lines: env.map(([k, v]) => `export ${k}=${shellQuote(v)}`),
      derivedLines: []
    });
  } else {
    miss('perf_env', 'the environment that gated behaviour; a reproduction runs with the binary defaults, which may differ');
  }
  serve.notes.push(
    'Whether this run reused a warm server is not recorded (--serve-reuse is not echoed into the record); a cold reproduction may read slower first samples.'
  );

  // 4. warm-up ----------------------------------------------------------------
  const warmFacts = [];
  if (rec.params?.warmup !== undefined) warmFacts.push(['params.warmup', String(rec.params.warmup)]);
  for (const k of ['cache_uncontrolled_cells', 'min_cached_prompt_pct', 'min_cached_prompt_tokens']) {
    if (typeof rec.metrics?.[k] === 'number') warmFacts.push([`metrics.${k}`, String(fmtNum(rec.metrics[k]))]);
  }
  const warmup = {
    id: 'warmup',
    title: 'Warm-up & probe',
    summary: warmFacts.length ? warmFacts.map(([k, v]) => `${k.split('.').pop()} ${v}`).join(' · ') : 'this benchmark records no warm-up parameters',
    facts: warmFacts,
    commands: [],
    notes: ['Whether the coherence probe ran is not recorded (--skip-coherence-probe leaves no trace in the record).']
  };

  // 5. measure ----------------------------------------------------------------
  const measure = { id: 'measure', title: 'Measure', summary: '', facts: [], commands: [], notes: [] };
  const shard = shardOf(rec);
  const bench = shellQuote(rec.benchmark_id ?? '');
  const addRun = (label, argv, isDerived) => {
    const c = runCommand(rec, argv, cls);
    for (const d of c.derived) if (!derived.some((x) => x.flag === d.flag)) derived.push(d);
    measure.commands.push({
      label,
      lines: c.lines,
      derivedLines: isDerived ? c.lines.map((_, k) => k) : c.derivedLines
    });
  };
  if (partition) {
    const members = [...partition.members].sort((a, b) => (shardOf(a)?.index ?? 0) - (shardOf(b)?.index ?? 0));
    let shown = 0;
    for (const m of members) {
      const s = shardOf(m);
      const label = `shard ${s ? `${s.index + 1} of ${s.count}` : '?'}${m.path ? ` · ${m.path}` : ''}`;
      if (Array.isArray(m.command) && m.command.length) {
        addRun(label, m.command, false);
        shown += 1;
      } else {
        miss(`command (${label})`, 'this member record carries no command; the shard has to be re-measured to be reproducible');
      }
    }
    measure.commands.push({
      label: 'aggregate the partition',
      lines: [`spark benchmark aggregate ${bench}${sha ? ` --sha ${shellQuote(sha)}` : ''}`],
      derivedLines: []
    });
    measure.summary = `${shown} of ${partition.count} shard commands · ${fmtNum(rec.metrics?.samples ?? 0)} samples`;
    measure.notes.push(
      `This point is the aggregate of ${partition.count} shards, not one run: every shard runs at the same count, then the group is scored once over the union (this page does the same in bfcl-partition.js). The score is partition-dependent; run all ${partition.count} at ${partition.count}.`
    );
  } else if (Array.isArray(rec.command) && rec.command.length) {
    addRun('the recorded run', rec.command, false);
    measure.summary = renderArgv(rec.command)[0].replace(/ \\$/, '');
    if (shard) {
      for (let j = 0; j < shard.count; j += 1) {
        if (j !== shard.index) addRun(`sibling shard ${j + 1} of ${shard.count} (derived: shard=${j}/${shard.count} substituted)`, withShard(rec.command, j, shard.count), true);
      }
      measure.commands.push({
        label: 'aggregate the partition',
        lines: [`spark benchmark aggregate ${bench}${sha ? ` --sha ${shellQuote(sha)}` : ''}`],
        derivedLines: []
      });
      measure.notes.push(
        `This record is slice ${shard.index + 1} of ${shard.count}; its number is not the gate's. The sibling commands differ from the recorded one only in the shard parameter and are marked derived.`
      );
    }
    const gaps = commandCoversRecord(rec);
    if (gaps.length) caveats.push(`The recorded command does not cover the recorded inputs: ${gaps.join('; ')}.`);
  } else {
    miss(
      'command',
      'the argv that produced this number. Every gate-mode record carries one (gate/record.rs writes it); a record without it cannot be replayed from this page and has to be re-measured'
    );
    measure.summary = 'no command recorded';
  }
  if (typeof rec.metrics?.known_partition_sensitive === 'number') {
    measure.facts.push(['metrics.known_partition_sensitive', String(rec.metrics.known_partition_sensitive)]);
  }
  if (rec.params && ['non_live_pct', 'live_pct', 'hallucination_pct'].every((k) => k in rec.params)) {
    measure.facts.push(['dataset_fingerprint', rec.dataset_fingerprint ?? 'not recorded']);
    if (!rec.dataset_fingerprint) miss('dataset_fingerprint', 'the content hash of the provisioned BFCL draw; check your set reports the same sample count before comparing');
  }
  measure.notes.push(
    'The command is reconstructed by record.rs from the recorded inputs; flags that leave no trace in the record (--serve-reuse, --skip-coherence-probe, --hardware, --checkpoint) are absent whether or not they were passed.'
  );
  if (derived.length) {
    measure.notes.push(`Marked lines were added by this page and are not in the record: ${derived.map((d) => `${d.flag} from ${d.from}`).join(', ')}.`);
  }

  // 6. judge ------------------------------------------------------------------
  const floors = Object.entries(rec.params ?? {}).filter(([k]) => FLOOR_KEY.test(k));
  const judge = {
    id: 'judge',
    title: 'Judge',
    summary: `${rec.verdict ?? 'no verdict'} · ${rec.frame_status ?? 'frame status not recorded'}`,
    facts: [['verdict', rec.verdict ?? 'not recorded'], ...floors.map(([k, v]) => [`params.${k}`, String(v)])],
    commands: [],
    notes: rec.verdict_reason ? [rec.verdict_reason] : []
  };
  if (!rec.verdict) miss('verdict', 'the gate decision; a record without one was never judged');
  if (partition) judge.notes.push('Each shard record reads `info`; this verdict was re-derived by the dashboard from the aggregate against params.min_* exactly as the gate does.');

  // 7. record & sign ----------------------------------------------------------
  const recordStep = {
    id: 'record',
    title: 'Record & sign',
    summary: `${path ?? 'path not recorded'} · ${rec.signer ? `signed ${rec.signer}` : 'unsigned'}`,
    facts: [],
    commands: [],
    notes: ['A signature proves the file and its commit were not altered after signing; it does not prove the run happened.']
  };
  const files = partition ? partition.members : [rec];
  for (const f of files) {
    if (f.path) recordStep.facts.push(['record', `${ATLAS_REPO}/blob/main/${f.path}`]);
    if (f.signer) recordStep.facts.push(['signer', `${ATLAS_REPO}/blob/main/.github/record-signers/${f.signer}.pub`]);
  }
  recordStep.commands.push({
    label: 'what CI runs to verify every committed record',
    lines: ['cargo run --locked -p spark-server --bin spark -- benchmark --pull-request-gate-check'],
    derivedLines: []
  });
  if (rec.box_state) {
    const b = rec.box_state;
    recordStep.facts.push(['box state after the run', `${b.validity ?? '?'}${b.concerns?.length ? ` · ${b.concerns.join('; ')}` : ''}`]);
    if (typeof b.gpu_temp_delta_c === 'number') recordStep.facts.push(['gpu temp delta', `+${b.gpu_temp_delta_c} °C`]);
    if (typeof b.elapsed_s === 'number') recordStep.facts.push(['elapsed', `${b.elapsed_s} s`]);
    if (b.validity && b.validity !== 'valid' && b.validity !== 'not-applicable') caveats.push(`Box state ${b.validity}: SPEED numbers in this record are not comparable.`);
  } else {
    miss('box_state', 'the hardware pre/post-check; without it thermal or contention effects on this number are unknown');
  }

  // 8. publish ----------------------------------------------------------------
  const publish = {
    id: 'publish',
    title: 'Publish',
    summary: `site/scripts/gen-gates.mjs · site as of ${ctx.generated.date} (${ctx.generated.sha})`,
    facts: [
      ['generator', 'site/scripts/gen-gates.mjs'],
      ['generated from', `${ctx.generated.sha} · ${ctx.generated.date}`]
    ],
    commands: [{ label: 'regenerate the dashboard data', lines: ['node site/scripts/gen-gates.mjs'], derivedLines: [] }],
    notes: []
  };

  // preface -------------------------------------------------------------------
  const hw = rec.hardware;
  if (!hw?.gpu) miss('hardware', 'the GPU and driver the number was measured on');
  const required = `${hw?.gpu_count ?? 1}× ${hw?.gpu ?? 'GPU not recorded'}, driver ${hw?.driver ?? 'not recorded'}, an idle box: the gate refuses to self-start below 85 % free host memory and with more than one foreign GPU process. This record was produced by one process on one box.`;
  const elapsed = rec.box_state?.elapsed_s;
  const costs = ctx.meta
    ? `About ${ctx.meta.expected_secs} s of measurement plus up to ${ctx.serve_allowance_s ?? 'an unregistered number of'} s to load the checkpoint${typeof elapsed === 'number' ? `; this run's box-state window was ${elapsed} s` : ''}.`
    : `Expected duration is not registered for ${rec.benchmark_id} in the descriptor SSOT${typeof elapsed === 'number' ? `; this run's box-state window was ${elapsed} s` : ''}.`;
  const expect = [
    rec.verdict_reason ? `Measured: ${rec.verdict_reason}` : 'No verdict reason recorded.',
    'A value under the floor is a finding, not a mismatch to explain away — file an issue.'
  ];

  const steps = [checkout, build, serve, warmup, measure, judge, recordStep, publish];
  const headline = `${rec.verdict ?? '?'} · ${rec.benchmark_id ?? '?'} · ${sha ?? 'no sha'} · ${typeof rec.recorded_at === 'number' ? isoDate(rec.recorded_at) : 'no date'}`;
  return { headline, preface: { required, costs, expect }, steps, missing, caveats, derived, script: toScript(steps, headline, missing, caveats, derived) };
}

/** The copy-all form: prose as `#` comments, commands verbatim. */
function toScript(steps, headline, missing, caveats, derived) {
  const out = [`# ${headline}`];
  for (const m of missing) out.push(`# not recorded: ${m.field} — ${m.need}`);
  for (const c of caveats) out.push(`# caveat: ${c}`);
  steps.forEach((s, k) => {
    out.push('', `# ${k + 1}. ${s.title} — ${s.summary}`);
    for (const [f, v] of s.facts) out.push(`#   ${f}: ${v}`);
    for (const c of s.commands) {
      out.push(`# ${c.label}`);
      if (c.derivedLines.length && derived.length) {
        out.push(`# lines added by this page, not in the record: ${derived.map((d) => `${d.flag} (from ${d.from})`).join(', ')}`);
      }
      out.push(...c.lines);
    }
  });
  return out.join('\n') + '\n';
}
