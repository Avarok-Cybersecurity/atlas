# Hopper A/B — Atlas vs vLLM on H100 / H200

Driver skeleton for the campaign in
the campaign issue (https://github.com/Avarok-Cybersecurity/atlas/issues/899).
Nothing here is new measurement machinery. The two components that produce the
numbers already exist and are reused verbatim:

- **[`bench/ladder38/harness_w55_conc_ladder.py`](../ladder38/harness_w55_conc_ladder.py)**
  — the engine-agnostic ladder client. ONE client drives both engines, which is
  the point: two harnesses measuring two engines is not an A/B. It owns the
  sampling parity pins (`presence_penalty`/`frequency_penalty` at 0,
  `chat_template_kwargs.enable_thinking` selected by `--enable-thinking`
  (default false), temp 0, seed 42), the
  per-request nonce that defeats prefix caching, and the output JSON schema
  everything downstream reads.
- **[`bench/phaseA_c_sweep.sh`](../phaseA_c_sweep.sh)** — the serve → health →
  bench → teardown orchestration, its skip-if-results-exist resumability, and
  its "Fairness notes" block. That block is the model for what a leg must write
  down about itself; copy its discipline, not its flags.

What is new here is the three things the ladder does not answer: how long each
engine took to become servable, whether what it served was coherent, and
whether the two result files are comparable at all.

## Files

| File | What it does |
|---|---|
| `workloads.json` | The frozen shapes. ISL/OSL, concurrencies, ladder rungs, sampling pins. The SSOT both legs read; a leg that does not match it is not in the campaign. |
| `time_to_ready.sh` | Measures boot: start → first `HTTP 200` on `/health` → first token. Requires health and a valid nonempty one-token completion within `--timeout-s` of launch (default 1800). |
| `coherency_gate.py` | Determinism, tool-call JSON and three known answers use `--think on\|off`; the separately labelled leakage probe requests off. Policies and HTTP exchanges are retained. Any failure is a non-zero exit. |
| `compare.py` | Two ladder JSONs in, the Pareto table out. Refuses to compare files whose workload axes differ. |
| `fixtures/` | Tiny hand-written ladder JSONs for `compare.py --selftest`, including the mismatched pair it must refuse, plus `degenerate_primes.txt` — a recorded GB10 failure the coherency gate must reject. |

Every script takes `--selftest`, which runs it against a local stub server (or
fixtures) and asserts the KNOWN answer. An instrument that has never been shown
to fail is not evidence; the selftests are where each one is shown to fail.

## The flow

Both legs run on the SAME node, sequentially, never concurrently — a second
engine resident on the GPU is a third variable.

```
0. prefetch weights          # once, outside both legs; a cold HF pull is not boot time
1. Atlas leg
   a. start `spark serve …`, note the epoch BEFORE the process starts
   b. time_to_ready.sh --engine atlas --start-epoch <that>   -> boot json
   c. coherency_gate.py --think <cell mode>                  -> gate json (hard stop on failure)
   d. harness_w55_conc_ladder.py --warmup 1 --reps 3         -> atlas ladder json
   e. tear the server down; confirm the GPUs are idle before continuing
2. vLLM leg — identical steps against the official recipe image
3. compare.py --atlas <a> --vllm <b>                         -> RESULTS.md rows + json
```

An on cell also passes `--enable-thinking` to the ladder. The campaign driver
does both and refuses modes excluded by `campaign_policy.json`; the assembler
checks gate and ladder policy evidence before certification. These probes judge
final answers and tool calls, not separately returned reasoning text. Model
templates that use another request-policy key need a verified adaptation before
their on/off labels can be trusted.

### Why boot time is measured, not estimated

The PRD makes a 30-minute boot cap a gate. `time_to_ready.sh` starts its clock
at an epoch the CALLER supplies — the moment before the serve process is
launched — because a script that starts its own clock measures its own startup
and silently forgives everything the launcher did first.

The two engines announce readiness differently and the script handles both:
Atlas answers `503 {"status":"loading"}` on `/health` while weights load and
then `200 {"status":"ready"}`; vLLM refuses the connection outright until its
server binds, then answers `200`. Connection-refused is therefore NOT an error
during the poll — it is vLLM's loading state — which is exactly the kind of
detail that turns into a wrong number when it is assumed instead of written
down.

Time-to-ready is not the whole story, so the script also issues a one-token
request afterwards and reports its latency separately. An engine that answers
`/health` before its graphs are captured is ready by the health check and not
by the clock.

### Why the fairness oracle lives in `compare.py`

The ladder JSON records `isl`, `osl`, `temperature`, `seed` and
`chat_template_kwargs` in its own header. `compare.py` refuses to emit a table
when those differ between the two files, because the failure it prevents is
silent: two legs run days apart with one flag changed produce a table that
looks exactly like a valid one. The refusal names the fields that differ.

`published.json` in `bench/ladder38/` is the precedent — its `harness_shas`
block exists because two legs really were run with two harness revisions, and
the equivalence had to be argued in prose afterwards. Refusing up front is
cheaper than arguing later.

## Running the selftests

```bash
bash time_to_ready.sh --selftest
python3 coherency_gate.py --selftest
python3 compare.py --selftest
```

They need `python3` and `curl`, no GPU and no network.

## What this skeleton does NOT do

- It does not start servers. The serve lines are the campaign's, and they
  belong in the campaign directory with the image digest that produced them —
  not hard-coded here where they would drift out of sight.
- It does not pick a campaign winner. `compare.py` labels each valid cell WIN/TIE/LOSS against the
  measured ratio; whether the campaign is won is a question about the whole
  table and the gates beside it.
- It has never been run against a Hopper box. Every number it can produce today
  came from a stub.

## Control validation and remaining limits

The [vLLM control report](https://github.com/Avarok-Cybersecurity/atlas/issues/899) (comment 3) records CPU red/green tests and Spark 2's read-only resource preflight. The requested Nano/image combination exceeds the task's 40 GB new-storage cap; no live engine run was performed.

Readiness now rejects HTTP errors, invalid/empty completion bodies, and boots whose first completion misses the process-start deadline. The first request pins the same sampling settings as the ladder. `first_token_s` is the latency of a one-token **non-streaming** completion including response framing, not the ladder's SSE TTFT; `total_s` includes health polling. `--model` is required to establish usability. `--out` write failures are nonzero. The readiness selftest also invokes `readiness_selftest.py` for HTTP boundary regressions.

The coherency selftest covers all four gates independently, malformed response envelopes and the declared tool schema. Every tool call must use the declared function name/type and required argument types. Deterministic text is not by itself proof of a correct answer.

### Why determinism grew two conditions, and where `known_answer_ok` came from

Measured on a GB10, the determinism check certified Nemotron 3 Nano FP8 (through the nvfp4 bundle) replying to "list exactly five prime numbers greater than one hundred" with `101, 103, 107, 107, 109, 109, 113, 109, 107, 109, …` — 256 tokens of decode loop, `finish_reason` `length`, and byte-identical across both greedy runs, because identical garbage is identical. That reply is now `fixtures/degenerate_primes.txt` and the selftest requires the gate to refuse it. Three things changed. Every reply the gate reads — determinism, think-leak and the new probes alike — goes through `scripts/test_coherence.py::_has_degeneration` as before **plus** a repetition-loop detector: over whitespace/comma-separated tokens, a reply of at least 40 tokens is a loop if one token is more than 50% of it or if fewer than 35% of its 3-grams are distinct, and any reply is a loop if one line or comma segment repeats 8 times in a row. On the fixture those read 0.73, 0.24 and 20, so all three fire with room to spare; running prose does not come close (English's most frequent word sits near 0.07 of a text), and the token-count floor keeps the two frequency signals off short answers like "Tokyo". Second, the determinism probe now fails on `finish_reason == "length"` and records `truncated_bounded_answer` — its prompt asks for five numbers, which cannot need 256 tokens, so hitting the cap means the model never stopped. Third, `known_answer_ok` runs the three probes from `bench/agentic/coherence_check.py` (17*23, the capital of Japan, "refrigerator" reversed), imported by path and judged the way that script judges them — the answer must appear in the first or last non-empty line, an answer reached only in the middle is reported as `WORKING-ONLY` rather than passed, and the printable-character ratio must stay above 0.98 — but at this gate's pinned sampling, not that script's. The other three checks all describe the SHAPE of a reply; a numerics bug that answers 271 for 17*23 produces a reproducible, well-formed, leak-free wrong answer and passes all of them.

What the thresholds miss: a loop whose period is longer than the reply (a 200-token answer that says the same thing twice in different words), a wrong answer to any question not in the three probes, and — deliberately — a short reply that repeats itself fewer than 8 times. Semantic repetition needs a model, not a counter. `known_answer_ok` is a fourth key in the gate JSON; the three original keys are unchanged.

The comparator requires matching valid `reps`, `warmup` and client `driver_sha256` as well as the original parity fields. Equal throughput is TIE; missing rungs appear as NO-PAIR on either side. Request errors, missing or short per-request usage, incomplete reps, invalid metrics and more than 10% rate spread remain visible as INVALID with reasons and no ratio. INVALID/NO-PAIR reports exit 0 because report generation succeeded; callers must inspect verdicts. Header/schema mismatches exit 2. Old tiny fixtures without request usage are no longer evidence of valid rungs.

Bare ladder JSON cannot establish model revision, hardware, server speculation, cache state or prompt-mode parity. Its latency columns are means of per-rep percentiles, not pooled request percentiles. Its first saved rep follows the discarded warmup. Its nominal ISL is word-based, and its nonce restarts across separate invocations. See the schema-gaps section of https://github.com/Avarok-Cybersecurity/atlas/issues/899 (comment 3) before producing a campaign receipt. The ladder measurement code was not changed.
