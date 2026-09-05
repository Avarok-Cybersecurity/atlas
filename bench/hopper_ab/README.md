# Hopper A/B — Atlas vs vLLM on H100 / H200

Driver skeleton for the campaign in
the campaign issue (https://github.com/Avarok-Cybersecurity/atlas/issues/899).
Nothing here is new measurement machinery. The two components that produce the
numbers already exist and are reused verbatim:

- **[`bench/ladder38/harness_w55_conc_ladder.py`](../ladder38/harness_w55_conc_ladder.py)**
  — the engine-agnostic ladder client. ONE client drives both engines, which is
  the point: two harnesses measuring two engines is not an A/B. It owns the
  sampling parity pins (`presence_penalty`/`frequency_penalty` at 0,
  `chat_template_kwargs.enable_thinking=false`, temp 0, seed 42), the
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
| `coherency_gate.py` | Determinism, tool-call JSON, and `<think>` containment against a live endpoint. Any failure is a non-zero exit. |
| `compare.py` | Two ladder JSONs in, the Pareto table out. Refuses to compare files whose workload axes differ. |
| `fixtures/` | Tiny hand-written ladder JSONs for `compare.py --selftest`, including the mismatched pair it must refuse. |

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
   c. coherency_gate.py                                      -> gate json (hard stop on failure)
   d. harness_w55_conc_ladder.py --warmup 1 --reps 3         -> atlas ladder json
   e. tear the server down; confirm the GPUs are idle before continuing
2. vLLM leg — identical steps against the official recipe image
3. compare.py --atlas <a> --vllm <b>                         -> RESULTS.md rows + json
```

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

The coherency selftest covers all three gates independently, malformed response envelopes and the declared tool schema. Every tool call must use the declared function name/type and required argument types. Deterministic text is not by itself proof of a correct answer.

The comparator requires matching valid `reps`, `warmup` and client `driver_sha256` as well as the original parity fields. Equal throughput is TIE; missing rungs appear as NO-PAIR on either side. Request errors, missing or short per-request usage, incomplete reps, invalid metrics and more than 10% rate spread remain visible as INVALID with reasons and no ratio. INVALID/NO-PAIR reports exit 0 because report generation succeeded; callers must inspect verdicts. Header/schema mismatches exit 2. Old tiny fixtures without request usage are no longer evidence of valid rungs.

Bare ladder JSON cannot establish model revision, hardware, server speculation, cache state or prompt-mode parity. Its latency columns are means of per-rep percentiles, not pooled request percentiles. Its first saved rep follows the discarded warmup. Its nominal ISL is word-based, and its nonce restarts across separate invocations. See the schema-gaps section of https://github.com/Avarok-Cybersecurity/atlas/issues/899 (comment 3) before producing a campaign receipt. The ladder measurement code was not changed.
