# Agentic serving harnesses

Tools for measuring Atlas under *agentic* load — an LLM driving a coding agent
through multi-turn tool calls — as opposed to the throughput soaks in `../`.
Written while chasing KV-cache correctness and concurrency scaling on Laguna;
all are model-parameterised, so pointing them at another model is a flag.

Every script honours `ATLAS_URL` (default `http://localhost:8888/v1/chat/completions`).

| script | answers |
|---|---|
| `conc_harness.py` | how does the server hold up with C agents at once? |
| `coherence_check.py` | is output still correct after stress (cold **and** warm)? |
| `prefill_floor.py` | what is the fixed cost of one prefill dispatch? |
| `bench_decode.py` | single-stream decode tok/s |
| `two_wave.py` | reproduce classic swap-out (needs 2 waves) |
| `force_preempt.py` | force KV exhaustion during decode |

## conc_harness.py — concurrency

```bash
python3 conc_harness.py --levels 1,4,8 --include-csharp --model laguna-s-2.1
ATLAS_CONTAINER=laguna-xs python3 conc_harness.py --levels 4 --model laguna-xs-2.1
```

Reports pass rate, per-task median/p95 latency, batch **makespan**, aggregate
tok/s, and server-side KV counters (exhaustions, `dec_ref` bugs, unowned-evicts,
preempts) scraped from the container log for the batch's window.

Design notes that matter for interpreting results:

- **Each concurrent slot gets a DISTINCT task.** If all agents sent identical
  prompts the prefix cache would dedupe the batch and the run would measure
  cache hits instead of concurrent work.
- **The task pool is Python + C#, deliberately no Rust.** The Rust/axum task in
  `oc_harness.py` is unusable in parallel — cargo serialises on the shared
  target dir and one build costs 80-370s, so wall-clock measures cargo, not the
  server. C# gives compiled-language coverage without that: 8 parallel
  `dotnet new` + `dotnet run` complete in ~3s total.
- **`--max-repeat N` kills a task after N identical tool calls in a row.** A
  degenerating agent otherwise burns the entire wall budget looking "busy" and
  distorts every number in the run. Observed live: a model emitting
  `bash({"command":true})` — a boolean where a command string belongs — retried
  91 times in 3.5 minutes.
- **Per-task latency SHOULD rise with C.** The questions are whether aggregate
  throughput scales and whether pass rate and KV health hold.

## coherence_check.py — correctness after stress

```bash
COHERENCE_MODEL=laguna-xs-2.1 python3 coherence_check.py
```

Aliased KV blocks corrupt output rather than crashing, so a clean refcount log is
necessary but not sufficient. Asks questions with checkable answers twice — cold,
then warm through the prefix cache.

Two things it does deliberately:

- **Sends an explicit system prompt** rather than inheriting whatever the chat
  template bakes in, so the framing is identical across models and runs.
  Override with `COHERENCE_SYSTEM`.
- **Judges the STATED answer, not any substring.** An earlier version accepted a
  reply that opened with "271" for 17×23 and only reached 391 inside its
  working — a wrong headline scored as correct. It now checks the first or last
  non-empty line and reports a "WORKING-ONLY" case separately.

For KV correctness specifically, prefer a long-context needle test (a secret
phrase at the end of a multi-chunk prompt, retrieved cold and warm) — it
isolates the cache path from model capability, which a reasoning question does
not.

## prefill_floor.py — is slicing worth it?

```bash
python3 prefill_floor.py laguna-s-2.1 3
```

Fits `wall = floor + tokens/throughput` over prompt lengths that each fit in one
chunk, using a unique random prefix per request so the cache cannot skip the
work. Reports the floor, throughput, and **the floor's share of a full chunk** —
the number that decides whether chunk-slicing is affordable, since slicing into
N pieces pays the floor N times.

Measured on Laguna-S (GB10, NVFP4): **floor 203.7 ms, 3234 tok/s**, i.e. 24% of a
2048-token chunk and 39% of a 1024-token one. So slicing 8192 → 4×2048 costs
+22% prefill for a 3.3× shorter worst-case decode stall (viable), while a
256-token slice floor would be 72% overhead (not viable).

## Findings worth keeping

- **Decode dominates agentic wall time.** Over 64 real turns: TTFT median 483 ms
  vs decode median 6,945 ms — prefill is **3.2%** of generation wall. Agentic
  turns are cache-warm (e.g. 7984 of 8408 prompt tokens cached), so prefill
  optimisations have almost no headroom here. Optimise decode.
- **Thinking is expensive under agentic load.** Same model, same sampling, only
  thinking toggled: 2.4× slower makespan at C=8, aggregate tok/s down 24%, pass
  12/13 → 9/13, and repeat-loop kills 0/0/1 → 1/2/5 across C=1/4/8.
- **Sampling temperature is the lesser factor.** A checkpoint-default run
  (temp 1.0 + thinking) and a tuned-sampling thinking run degraded similarly,
  isolating thinking as the dominant cost.
- **Watch the loop detector count, not just pass rate.** Degeneration scales with
  concurrency and shows up as repeat-kills before it shows up as failures.

## Cross-model concurrency (2026-07-27, one GB10, thinking OFF on all three)

| model | agg tok/s C=1 / 4 / 8 | pass C=1 / 4 / 8 | loop-kills | makespan C=8 |
|---|---|---|---|---|
| Holo 3.1 35B-A3B-NVFP4 | 42.8 / 53.2 / **53.5** | 0/1, 3/4, **4/8** | 1 / 2 / 4 | 194 s |
| Laguna-XS-2.1 | 21.1 / 45.4 / 59.3 | 1/1, 4/4, 8/8 | 0 / 0 / 4 | 133 s |
| Laguna-S-2.1 | 11.0 / 25.1 / 46.0 | 1/1, 3/4, 8/8 | 0 / 0 / 1 | 210 s |

Holo is the fastest per stream (decode median 44.6 tok/s at C=1 vs 38.6 / 13.2)
and the worst at scaling: aggregate moves 53.2 → 53.5 from C=4 to C=8 (+0.6%)
while XS gains +31% and S +83%. It saturates at C≈4.

Two traps in reading that table:

- **Holo's lower C=8 makespan is an artifact.** Killed tasks exit early, so
  finishing 4/8 in 194 s is not better than finishing 8/8 in 210 s. Read pass
  rate before makespan.
- **`loop: () x6` — six identical tool calls with EMPTY arguments — now appears
  on all three models**, including Holo at C=1 with thinking off and tuned
  sampling. Three checkpoints, one signature, so suspect the Atlas tool-call
  path over a model quirk. Undiagnosed.

KV counters were clean (`decref=0 evict_unowned=0 exhaustions=0 preempts=0`) in
every run, on all three models.
