import { describe, expect, it } from 'bun:test';
import { foldPartitions } from './bfcl-partition.js';
import {
  commandCoversRecord,
  hardwareClass,
  renderArgv,
  reproSteps,
  shellQuote,
  splitFlags
} from './repro-steps.js';

// Fixtures are trimmed copies of records on main; only the fields the module
// reads are kept, values verbatim.
const CTX = { generated: { sha: '4b18f7cec', date: '2026-09-16' }, meta: { expected_secs: 1560, sensitivity: 'Speed' }, serve_allowance_s: 600 };

const sweep = () => ({
  benchmark_id: 'concurrency-sweep',
  git_sha: 'a87b41905f',
  recorded_at: 1789803720,
  target_model: 'unsloth/Qwen3.8-27B-NVFP4',
  served_by: 'qwen3.8/qwen3.8-27b-nvfp4-unsloth',
  atlas_version: '1.0.0-beta-preview',
  hardware: { gpu: 'NVIDIA GB10', driver: '580.159.03', gpu_count: 1 },
  perf_class: 'gb10@spark-28c2',
  params: { concurrencies: '1, 2, 4, 8, 16, 32, 64, 128', isls: '512', min_c64: '107.6', warmup: '1' },
  command: [
    'spark', 'benchmark', 'run', 'concurrency-sweep',
    '--param', 'concurrencies=1, 2, 4, 8, 16, 32, 64, 128',
    '--param', 'isls=512', '--param', 'min_c64=107.6', '--param', 'warmup=1',
    '--serve-override', 'kv_cache_dtype=fp8', '--pull-request-gate'
  ],
  serve_overrides: { kv_cache_dtype: 'fp8' },
  perf_env: { AVAROK_PREFILL_CODISPATCH: '0' },
  metrics: { c64_aggregate_tok_s: 116.2, cache_uncontrolled_cells: 0 },
  frame_status: 'Completed',
  verdict: 'PASS',
  verdict_reason: 'every populated floor met (C64 116.2/107.6)',
  branch: '',
  generated_ancestry: 'yes',
  path: '.benchmarks/concurrency-sweep/2026-09-19-a87b41905f.json',
  signer: '02156264cbf75bd7',
  dirty_paths: [],
  dataset_fingerprint: null,
  box_state: { validity: 'valid', concerns: [], gpu_temp_delta_c: 16, elapsed_s: 1461 }
});

const shardRec = (index, count, extra = {}) => ({
  ...sweep(),
  benchmark_id: 'bfcl-subset',
  served_by: 'qwen3.8/qwen3.8-27b-nvfp4-unsloth-bfcl',
  recorded_at: 1789808000 + index,
  params: { non_live_pct: '62', live_pct: '10', hallucination_pct: '10', min_overall: '82.6', shard: `${index}/${count}` },
  command: [
    'spark', 'benchmark', 'run', 'bfcl-subset',
    '--param', 'non_live_pct=62', '--param', 'live_pct=10', '--param', 'hallucination_pct=10',
    '--param', 'min_overall=82.6', '--param', `shard=${index}/${count}`, '--pull-request-gate'
  ],
  serve_overrides: {},
  metrics: {
    'shard.index': index, 'shard.count': count, samples: 10, overall_accuracy: 90,
    'subset.simple_python.hits': 9, 'subset.simple_python.n': 10
  },
  verdict: 'info',
  verdict_reason: `shard ${index}/${count}`,
  path: `.benchmarks/bfcl-subset/2026-09-19-a87b41905f-s${index}of${count}.json`,
  ...extra
});

/** A POSIX-enough tokenizer: whitespace, single quotes, `\`-escapes, `\`+newline. */
function splitShell(text) {
  const src = text.replace(/\\\n/g, ' ');
  const out = [];
  let cur = '';
  let inTok = false;
  for (let i = 0; i < src.length; i += 1) {
    const c = src[i];
    if (c === '\\' && i + 1 < src.length) {
      inTok = true;
      cur += src[i + 1];
      i += 1;
    } else if (c === "'") {
      inTok = true;
      const end = src.indexOf("'", i + 1);
      if (end === -1) throw new Error('unterminated quote');
      cur += src.slice(i + 1, end);
      i = end;
    } else if (/\s/.test(c)) {
      if (inTok) out.push(cur);
      cur = '';
      inTok = false;
    } else {
      inTok = true;
      cur += c;
    }
  }
  if (inTok) out.push(cur);
  return out;
}

const measureOf = (plan) => plan.steps.find((s) => s.id === 'measure');
const linesOf = (plan) => plan.steps.flatMap((s) => s.commands.flatMap((c) => c.lines));

describe('shell safety', () => {
  it('round-trips the multi-value param through a POSIX tokenizer', () => {
    const argv = sweep().command;
    const rendered = renderArgv(argv).join('\n');
    expect(rendered).toContain("'concurrencies=1, 2, 4, 8, 16, 32, 64, 128'");
    expect(splitShell(rendered)).toEqual(argv);
    // Negative control: the quoting is load-bearing. A space join tokenizes
    // to a different argv (eight extra arguments), which `--param` refuses.
    expect(splitShell(argv.join(' ')).length).not.toBe(argv.length);
  });

  it('quotes exactly what needs it and escapes an embedded quote', () => {
    expect(shellQuote('shard=5/6')).toBe('shard=5/6');
    expect(shellQuote('')).toBe("''");
    expect(splitShell(shellQuote("it's"))).toEqual(["it's"]);
  });

  it('renders a bare boolean flag on its own line', () => {
    const argv = ['spark', 'benchmark', 'run', 'agentic-webserver', '--param', 'iterations=10', '--yes', '--pull-request-gate'];
    expect(splitFlags(argv).pairs).toEqual([['--param', 'iterations=10'], ['--yes', null], ['--pull-request-gate', null]]);
    const lines = renderArgv(argv);
    expect(lines).toContain('  --yes \\');
    expect(lines[lines.length - 1]).toBe('  --pull-request-gate');
    expect(splitShell(lines.join('\n'))).toEqual(argv);
  });
});

describe('a partition aggregate reproduces as N shard commands', () => {
  const aggregate = () => {
    const { records } = foldPartitions([shardRec(0, 2), shardRec(1, 2)]);
    expect(records).toHaveLength(1);
    expect(records[0].partition.count).toBe(2);
    return records[0];
  };

  it('lists every member command by shard and the real aggregate subcommand', () => {
    const plan = reproSteps(aggregate(), CTX);
    const cmds = measureOf(plan).commands;
    const runs = cmds.filter((c) => c.lines[0].startsWith('spark benchmark run'));
    expect(runs.map((c) => c.label)).toEqual([
      'shard 1 of 2 · .benchmarks/bfcl-subset/2026-09-19-a87b41905f-s0of2.json',
      'shard 2 of 2 · .benchmarks/bfcl-subset/2026-09-19-a87b41905f-s1of2.json'
    ]);
    expect(runs[0].lines).toContain('  --param shard=0/2 \\');
    expect(runs[1].lines).toContain('  --param shard=1/2 \\');
    // Member commands are the record's own, so none is marked derived except
    // the two flags every gate command gains.
    for (const r of runs) expect(r.derivedLines.length).toBe(2);
    expect(cmds[cmds.length - 1].lines).toEqual(['spark benchmark aggregate bfcl-subset --sha a87b41905f']);
    expect(measureOf(plan).summary).toBe('2 of 2 shard commands · 20 samples');
    // Every path is listed under record & sign, not just the inherited one.
    const rec = plan.steps.find((s) => s.id === 'record');
    expect(rec.facts.filter(([k]) => k === 'record')).toHaveLength(2);
    expect(plan.steps.find((s) => s.id === 'judge').notes.join(' ')).toContain('re-derived by the dashboard');
  });

  it('never presents the inherited single command as the run', () => {
    const agg = aggregate();
    const inherited = renderArgv(agg.command);
    const plan = reproSteps(agg, CTX);
    const runs = measureOf(plan).commands.filter((c) => c.lines[0].startsWith('spark benchmark run'));
    // The inherited argv IS member 0's argv: it appears once, labelled as
    // that shard, never as "the recorded run".
    const inheritedArgv = splitShell(inherited.join('\n'));
    expect(runs.filter((c) => splitShell(c.lines.join('\n')).slice(0, inheritedArgv.length).join('\0') === inheritedArgv.join('\0'))).toHaveLength(1);
    expect(runs.some((c) => c.label === 'the recorded run')).toBe(false);
    // Negative control: the same record without `partition` is one run and
    // no aggregate step — proving the partition branch is what produced N.
    const { partition, ...plain } = agg;
    const single = reproSteps({ ...plain, metrics: { ...plain.metrics, 'shard.index': undefined, 'shard.count': undefined } }, CTX);
    const singleRuns = measureOf(single).commands;
    expect(singleRuns.filter((c) => c.lines[0].startsWith('spark benchmark run'))).toHaveLength(1);
    expect(singleRuns[0].label).toBe('the recorded run');
    expect(singleRuns.some((c) => c.lines[0].startsWith('spark benchmark aggregate'))).toBe(false);
  });

  it('names a member that carries no command instead of filling it in', () => {
    const { records } = foldPartitions([shardRec(0, 2, { command: undefined }), shardRec(1, 2)]);
    const plan = reproSteps(records[0], CTX);
    expect(plan.missing.some((m) => m.field.startsWith('command (shard 1 of 2'))).toBe(true);
    expect(measureOf(plan).summary).toBe('1 of 2 shard commands · 20 samples');
    expect(plan.script).not.toContain('shard=0/2');
  });
});

describe('a lone shard record', () => {
  it('derives the sibling commands, marks them derived, and adds the aggregate', () => {
    const plan = reproSteps(shardRec(5, 6), CTX);
    const cmds = measureOf(plan).commands;
    const siblings = cmds.filter((c) => c.label.startsWith('sibling shard'));
    expect(siblings).toHaveLength(5);
    expect(siblings.map((c) => c.lines.find((l) => l.includes('shard=')))).toEqual(
      [0, 1, 2, 3, 4].map((j) => `  --param shard=${j}/6 \\`)
    );
    for (const s of siblings) expect(s.derivedLines).toEqual(s.lines.map((_, k) => k));
    expect(cmds[cmds.length - 1].lines[0]).toBe('spark benchmark aggregate bfcl-subset --sha a87b41905f');
    // Negative control: no shard → no siblings, no aggregate.
    const plain = reproSteps(sweep(), CTX);
    expect(measureOf(plain).commands.some((c) => c.label.startsWith('sibling') || c.lines[0].includes('aggregate'))).toBe(false);
  });
});

describe('a missing field is named, never filled in', () => {
  it('a record without a command shows no run command anywhere', () => {
    const { command, ...rec } = sweep();
    const plan = reproSteps(rec, CTX);
    expect(plan.missing.map((m) => m.field)).toContain('command');
    expect(measureOf(plan).commands).toHaveLength(0);
    // Only the command lines matter: prose may NAME the missing command.
    const cmdLines = plan.script.split('\n').filter((l) => l && !l.startsWith('#'));
    expect(cmdLines.some((l) => l.startsWith('spark benchmark run'))).toBe(false);
    expect(cmdLines.join('\n')).not.toMatch(/undefined|\bnull\b|--\S+\s+(--|$)/m);
    expect(plan.script).not.toMatch(/undefined|\bnull\b/);
    // Negative control: the complete record has an empty missing list and
    // does show the run.
    const full = reproSteps(sweep(), CTX);
    expect(full.missing).toEqual([]);
    expect(full.script).toMatch(/spark benchmark run concurrency-sweep/);
  });

  it('names served_by, perf_env, box_state, git_sha and hardware class when absent', () => {
    const rec = { ...sweep(), served_by: undefined, perf_env: {}, box_state: null, git_sha: '', perf_class: '' };
    const plan = reproSteps(rec, CTX);
    const fields = plan.missing.map((m) => m.field);
    for (const f of ['served_by', 'perf_env', 'box_state', 'git_sha', 'hardware class']) expect(fields).toContain(f);
    expect(plan.steps.find((s) => s.id === 'serve').commands).toEqual([]);
    expect(plan.script).not.toContain('git checkout');
    expect(plan.script).not.toMatch(/undefined|\bnull\b/);
    // No recipe → gate-only flags are not derived either.
    expect(plan.derived).toEqual([]);
    expect(linesOf(plan).some((l) => l.includes('--checkpoint') || l.includes('--hardware'))).toBe(false);
  });

  it('refuses to build without the dashboard provenance (PCND)', () => {
    expect(() => reproSteps(sweep(), { generated: { sha: '', date: '' }, meta: null, serve_allowance_s: null })).toThrow(TypeError);
    expect(() => reproSteps(sweep(), {})).toThrow(TypeError);
  });

  it('says when the expected duration is not registered instead of quoting one', () => {
    const plan = reproSteps(sweep(), { ...CTX, meta: null });
    expect(plan.preface.costs).toContain('not registered for concurrency-sweep');
    expect(plan.preface.costs).not.toMatch(/\d+ s of measurement/);
    expect(reproSteps(sweep(), CTX).preface.costs).toContain('About 1560 s of measurement plus up to 600 s');
  });
});

describe('derived flags are appended and reported', () => {
  const ttft = () => ({
    ...sweep(),
    benchmark_id: 'ttft-cold-gate',
    target_model: 'nvidia/Qwen3.6-35B-A3B-NVFP4',
    served_by: 'qwen3.6/qwen3.6-35b-a3b-nvfp4',
    params: { repeats: '12' },
    serve_overrides: {},
    command: ['spark', 'benchmark', 'run', 'ttft-cold-gate', '--param', 'repeats=12', '--pull-request-gate']
  });

  it('adds --checkpoint and --hardware as the last, marked lines', () => {
    const plan = reproSteps(ttft(), CTX);
    const run = measureOf(plan).commands[0];
    expect(run.lines.slice(-2)).toEqual(['  --checkpoint nvidia/Qwen3.6-35B-A3B-NVFP4 \\', '  --hardware gb10']);
    expect(run.derivedLines).toEqual([run.lines.length - 2, run.lines.length - 1]);
    expect(plan.derived).toEqual([{ flag: '--checkpoint', from: 'target_model' }, { flag: '--hardware', from: 'perf_class' }]);
    expect(splitShell(run.lines.join('\n'))).toEqual([...ttft().command, '--checkpoint', 'nvidia/Qwen3.6-35B-A3B-NVFP4', '--hardware', 'gb10']);
    expect(plan.script).toContain('# lines added by this page, not in the record: --checkpoint (from target_model), --hardware (from perf_class)');
  });

  it('does not add a flag the record already carries, or a class it cannot derive', () => {
    const withCkpt = { ...ttft(), command: [...ttft().command.slice(0, -1), '--checkpoint', 'X', '--pull-request-gate'] };
    const plan = reproSteps(withCkpt, CTX);
    const run = measureOf(plan).commands[0];
    expect(run.lines.filter((l) => l.includes('--checkpoint'))).toHaveLength(1);
    expect(plan.derived).toEqual([{ flag: '--hardware', from: 'perf_class' }]);
    const noClass = reproSteps({ ...ttft(), perf_class: '' }, CTX);
    expect(linesOf(noClass).some((l) => l.includes('--hardware'))).toBe(false);
    expect(noClass.missing.map((m) => m.field)).toContain('hardware class');
    expect(hardwareClass({ perf_class: 'gb10@spark-28c2' })).toBe('gb10');
    expect(hardwareClass({ perf_class: '' })).toBeNull();
    expect(hardwareClass({})).toBeNull();
  });
});

describe('caveats', () => {
  it('a dirty tree is a caveat and the checkout is marked insufficient', () => {
    const plan = reproSteps({ ...sweep(), dirty_paths: ['kernels/x.cu'] }, CTX);
    expect(plan.caveats.some((c) => c.includes('kernels/x.cu'))).toBe(true);
    expect(plan.steps.find((s) => s.id === 'checkout').notes.some((n) => n.startsWith('Insufficient'))).toBe(true);
    const clean = reproSteps(sweep(), CTX);
    expect(clean.caveats.some((c) => c.includes('uncommitted'))).toBe(false);
    expect(clean.steps.find((s) => s.id === 'checkout').notes).toEqual([]);
  });

  it('an unsigned record and an invalidated box state are said, a valid one is not', () => {
    const plan = reproSteps({ ...sweep(), signer: null, box_state: { validity: 'invalidated', concerns: ['throttled'], elapsed_s: 5 } }, CTX);
    expect(plan.caveats.some((c) => c.includes('.sig'))).toBe(true);
    expect(plan.caveats.some((c) => c.includes('not comparable'))).toBe(true);
    const ok = reproSteps(sweep(), CTX);
    expect(ok.caveats).toEqual([]);
    expect(ok.steps.find((s) => s.id === 'record').facts).toContainEqual(['signer', 'https://github.com/Avarok-Cybersecurity/atlas/blob/main/.github/record-signers/02156264cbf75bd7.pub']);
  });

  it('a command that does not cover the recorded inputs is reported', () => {
    expect(commandCoversRecord(sweep())).toEqual([]);
    const dropped = { ...sweep(), command: sweep().command.filter((t) => t !== 'isls=512' && t !== '--param' || true) };
    dropped.command = sweep().command.flatMap((t, k, a) => (t === 'isls=512' || (t === '--param' && a[k + 1] === 'isls=512') ? [] : [t]));
    expect(commandCoversRecord(dropped)).toEqual(['params.isls=512 is not in the command']);
    expect(reproSteps(dropped, CTX).caveats.some((c) => c.includes('params.isls=512'))).toBe(true);
    const noGate = { ...sweep(), command: sweep().command.filter((t) => t !== '--pull-request-gate') };
    expect(commandCoversRecord(noGate)).toEqual(['the command lacks --pull-request-gate: not a gate-mode invocation']);
  });
});

describe('the plan shape', () => {
  it('has the eight pipeline steps in execution order and real env exports', () => {
    const plan = reproSteps(sweep(), CTX);
    expect(plan.steps.map((s) => s.id)).toEqual(['checkout', 'build', 'serve', 'warmup', 'measure', 'judge', 'record', 'publish']);
    expect(plan.headline).toBe('PASS · concurrency-sweep · a87b41905f · 2026-09-19');
    expect(linesOf(plan)).toContain('export AVAROK_PREFILL_CODISPATCH=0');
    expect(linesOf(plan)).toContain('git fetch origin a87b41905f && git checkout a87b41905f');
    expect(plan.steps.find((s) => s.id === 'judge').facts).toContainEqual(['params.min_c64', '107.6']);
    expect(plan.steps.find((s) => s.id === 'publish').summary).toBe('site/scripts/gen-gates.mjs · site as of 2026-09-16 (4b18f7cec)');
    expect(plan.steps.find((s) => s.id === 'serve').facts[0][1]).toBe('https://github.com/Avarok-Cybersecurity/atlas-recipes/blob/main/recipes/qwen3.8/qwen3.8-27b-nvfp4-unsloth.yaml');
  });

  it('omits the checkout command when the commit is unknown to the dashboard history', () => {
    const plan = reproSteps({ ...sweep(), generated_ancestry: 'unknown' }, CTX);
    expect(linesOf(plan).some((l) => l.includes('git checkout'))).toBe(false);
    expect(plan.steps.find((s) => s.id === 'checkout').notes[0]).toContain('not found in the history');
  });
});
