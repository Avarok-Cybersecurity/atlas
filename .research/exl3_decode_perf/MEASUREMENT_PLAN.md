# EXL3 on GB10 — GPU measurement plan (prefill profile + concurrency sweep)

Date 2026-09-05. Branch `research/exl3-gpu-measurement-plan` on top of `wip/exl3-research`
(`ffdffa467`, the named preset `spark serve qwen3.8-flash-next-exl3` at 4 sequences x 128K).
Checkpoint `/tank/exl3-ckpt/qwen38-flash-next-4.05bpw` (K=6 dense, K=4 experts, 512 experts
top-10, hidden 2560, expert inter 640, 48 layers). Box dgx-00, one GB10 (48 SMs, 231 GB/s
measured streaming read).

This document is written BEFORE the GPU runs. Every number below that is not marked "measured"
is a hypothesis from `EXL3_DECODE_PERF.md`, `vllm_exl3_prefill_review.md` and arithmetic; the
scripts exist so that the next engineering decision is made from a trace and a table, not from
these guesses. Read with `/measurement-discipline`: fingerprint (Rule 1), one variable per
comparison (Rule 2), n>=3 for a headline (Rule 4), roofline arithmetic is a hypothesis (Rule 7).

## The two open questions

1. **Prefill is ~390-407 tok/s cold at 8K-11K** (measured 2026-09-05, `measure_prefill.py`,
   MTP profile and the preset; `prefill_baseline_8k_11k.txt`, job-dir `prefill8k_nsys.txt`).
   Reference points the user supplied: other engines ~1.1K tok/s at 8K on this model, NVFP4 on
   Atlas ~2.6K. The routed-weight bandwidth bound alone allows ~15.6K tok/s and the tensor-core
   bound ~5.9K (DERIVED in the vllm review) — so a >2.5x gap to the nearest reference is
   structural, and there is NO EXL3 prefill kernel breakdown yet. Three candidate levers (R2
   raise the fused MoE tier's 128-row cap, R3 dense reconstruct+BF16 GEMM tier, R4 fat-tile
   grouped expert GEMM) target DIFFERENT kernels; without the split their order is a guess.
2. **The preset serves 4 sequences x 128K with 2 MTP drafts and prefix caching**, sized by the
   self-relative KV budget at boot, and the recipe commit says plainly that this envelope is
   "not separately validated": the gates ran at 32K / one sequence. The preset's own comment
   says "the mHC MTP verify runs per sequence (serialized) above one". An earlier C=4 attempt at
   util 0.85 swapped the box. What C buys — or costs — is unmeasured, and so is whether 4 x 30K
   of context fits without host swap.

## Script 1 — `prefill_profile.sh` (answers question 1)

What it does: refuses to start beside another `spark serve` / nsys session / low MemAvailable;
boots this branch's release binary through the preset under `nsys launch --trace=cuda,nvtx`
(TMPDIR under the run dir); warms with one 512-token cold request; measures two UNCAPTURED ~8K
cold prefills (the reference wall under injection); then captures exactly ONE ~8K cold prefill
(fresh salt, `max_tokens=1`, i.e. one 8192-token prefill chunk plus one decode step) between
`nsys start`/`nsys stop`; stops the server; runs `nsys stats --report
cuda_gpu_kern_sum,cuda_api_sum` and `nsys_kern_table.py` (per-kernel and per-FAMILY % of captured
GPU time, host API sync/launch counts, GPU busy fraction against the captured wall). It also
widens `RUST_LOG` for `forward_prefill_exl3` to `trace` so the `overflow_experts` count per
4096-token MoE batch lands in `serve.log` (`overflow_stats.txt`).

Sanity checks before reading the table:
- `prefill_captured.txt` wall within ~10% of `prefill_reference.txt` — otherwise capture
  perturbed the run and the family split is suspect (the decode capture cost ~3%).
- `prompt_tokens` ~ 7,900-8,100 so the prompt is ONE prefill chunk (PLE cap 9216; a two-chunk
  prompt double-counts per-chunk fixed costs).
- GPU busy fraction: if < 70%, the first lever is host-side, not a kernel — see D1.

### Decision table (family shares are % of captured GPU kernel time)

| ID | Observation | Decision |
|---|---|---|
| D1 | GPU busy fraction < 70% of the captured wall, and/or `cuda_api_sum` shows > 200 `cuMemcpyDtoH*`/`cuStreamSynchronize` calls in the request | Host-bound. Work the syncs first: the one D2H of `expert_offsets` per MoE batch (`moe_prefill.rs`, when S>128), the PLE host hash/gather per chunk, the QSA host top-k. No kernel work until busy > 85%. |
| D2 | EXL3 MoE overflow tier (`exl3_gemm_k4_*` + gather/store glue) >= 15% | **R2** — raise `EXL3_MOE_MAX_TOKENS_PER_EXPERT` 128 -> 512 (then 1024), slabs in `ptr_table_build.rs`; A/B on `measure_prefill.py` 8K/11K with the constant the only variable. `overflow_stats.txt` must show `overflow_experts` falling to ~0 or the arm is inert (Rule 4). Expected (HYP): removes >= 5 serialized launches per 1024 overflow rows per expert; effect size unknown. |
| D3 | EXL3 MoE fused tier (`exl3_moe_k4_n128_cb2`) >= 40% AND achieved bandwidth per launch (2.46 MB x active experts / avg kernel time) < 40% of 231 GB/s | ALU-bound trellis re-decode (16-row M slices re-decode each B fragment ~5x per 4K chunk). **R4** — a grouped 128-row-M-tile expert GEMM, deterministic slot-store epilogue; 1-2 weeks. Only after D2 is exhausted. |
| D4 | Fused tier >= 40% but achieved bandwidth >= 60% of peak | Bandwidth-bound already; R4 buys little. Lever is bytes: fewer expert re-reads per chunk (larger `ATLAS_EXL3_MOE_PREFILL_BATCH_TOKENS`, bounded by slab memory) — cheap A/B via the env knob first. |
| D5 | EXL3 dense K=6 GEMM (`exl3_gemm_k6_*`) >= 25% | **R3** — reconstruct-to-BF16 + fixed-config BF16 GEMM tier above a row threshold in `exl3_dense.rs`, transient inside the util pledge; not bit-identical, needs the greedy-sample and agentic gates. Upstream's threshold is 144 rows. |
| D6 | QSA (`qsa_*`) >= 30% | Not an EXL3 lever. The 2026-08-27 NVFP4 profile had QSA at 34%; this is the shared sparse-attention path (`qsa_score_rows`, `qsa_topk_rows`, `qsa_prefill_attn`) and its fix is model-wide. Park the EXL3 tiers; open a QSA ticket with this table attached. |
| D7 | GDN (`gated_delta_rule_*`, `causal_conv1d_*`, gated norm) >= 20% | GDN chunk kernels; the FlashInfer GDN path exists but fails open silently — verify `ATLAS_GDN_LIB` engaged in `serve.log` before concluding anything (memory: `holo-gdn-fails-open`). |
| D8 | cuBLASLt / `hc_*` (mHC collapse) >= 15% | Shared with the NVFP4 path; the 08-27 lever ladder already halved it. Lower priority than any EXL3-specific family above 15%. |
| D9 | `w4a16_*` (NVFP4 shared expert) >= 10% at prefill | The shared expert still runs its NVFP4 materialised twin in prefill; the native K=4 shared-expert slot (decode ranking item 1) would also remove this. |

Ranking rule: take the highest-share family whose decision is EXL3-specific (D2-D5, D9) unless
D1 fires. Two families within 5 points of each other: do the cheaper one first (D2 is a
constant + slab sizes; D5 is a new tier; D3 is a new kernel).

Thresholds are deliberately coarse. A family at 12% is not a lever worth a week; a family at
40% is a lever even if the kernel is "at peak" (then the lever is bytes, D4).

### What would make the prefill number itself suspect

- `prefill_captured.txt` shows `prompt_tokens` far from 8K: the tokenizer ratio drifted; fix the
  word count, not the conclusion.
- The preset's prefix cache served part of the prompt (a warm restore instead of a prefill):
  impossible by construction (salt leads the prompt) — but check `serve.log` for a prefix hit
  on the captured request anyway.
- The reference and captured walls disagree by > 10%: report the family split as "under
  capture, perturbed" and rerun with `--trace=cuda` only.

## Script 2 — `concurrency_sweep.sh` (answers question 2)

What it does: same refusals; boots the preset (util 0.72 — NOT raised); warms; then per arm runs
`measure_concurrency.py`, which launches C streams at once with distinct salted prompts
(natural/code answer class: a ~2K or ~30K salted log followed by the LRU-cache Rust task,
`reasoning_effort:low`, `max_tokens=300`, temp 0) at C = 1, 2, 4 and records per stream TTFT,
decode wall, server-attested `completion_tokens`, per-stream decode tok/s
`(completion_tokens-1)/decode wall` (the MTP-safe definition — gap medians are meaningless under
drafting), aggregate decode tok/s (sum tokens over the union of decode windows) and aggregate
incl. TTFT; samples MemAvailable, swap, nvidia-smi util/power every 5 s (nvidia-smi memory is
`[N/A]` on GB10's unified memory — `/proc/meminfo` is the truth); and aborts — TERMing the
server — if MemAvailable < 8 GB or swap grows > 4 GB. The shell has a second watchdog covering
boot. `ATLAS_MTP_ACCEPT_DEBUG=1` is set so accepted-per-step lines land in `serve.log`.

Arms: `short` (~2K prompt, REPEATS_SHORT=3 — the headline cells) and `long` (~30K prompt,
REPEATS_LONG=1 — preliminary, single-shot; 4 x 30K ~ 120K of the 128K x 4 envelope plus GDN/PLE
state). Expected wall (HYP at ~400 tok/s prefill, serialized): the long arm's C=4 cell spends
~5 min in prefill before the last stream's first token; the whole long arm ~10 min.

Baselines to compare against (measured 2026-09-05, one sequence, 32K, `measure_decode.py`
300-token code prompt): MTP 2 drafts + prefix cache **30.3-30.7 tok/s** (`ab_abf16_mtp.txt`),
serial no-spec **23.5-23.9 tok/s**. The sweep's C=1 short cell is the same prompt class at a
~2K prefix instead of ~60 tokens, under the 4-seq/128K preset instead of 1-seq/32K — a related
observation, not the same fingerprint.

### Decision table

| ID | Observation | Decision |
|---|---|---|
| S1 | Short C=1 median per-stream < 27 tok/s (>10% below the 30.3 one-sequence number) | The 4-sequence / 128K preset itself costs serial decode (KV-pool geometry, `ATLAS_MTP_MAX_SEQS=4`, or the per-sequence verify path). One-variable A/B: `EXTRA_SERVE_ARGS="--max-num-seqs 1 --max-batch-size 1"` vs preset, same script. Fix this before reading any C>1 cell. |
| S2 | Aggregate decode tok/s at C=4 >= 2.5x C=1 and per-stream at C=4 >= 60% of C=1 | Batching works under MTP. No concurrency lever; spend the effort on prefill (Script 1). |
| S3 | Aggregate at C=2 <= 1.3x C=1 (flat), per-stream roughly halves | The serialized per-sequence mHC verify is the ceiling (each sequence's 3-row step runs alone, ~86 ms; two sequences = two steps). Levers, in order: (a) A/B the draft width at C>=2 — boot with `EXTRA_SERVE_ARGS="--num-drafts 1"` (an explicit flag beats the preset's 2; the verify narrows from 3 rows to 2) and compare aggregate tok/s WITH acceptance; a fully spec-OFF arm cannot be expressed through the preset (`--speculative` is a presence flag with no negation and there is no global MTP kill switch), so it must be a preset-free serve with every other preset flag/env spelled out, labelled as such; if the narrower or spec-off arm's aggregate at C=4 beats MTP's, gate MTP on batch size in the scheduler (`mtp_gate`) — a bounded change; (b) cross-sequence batched verify (the DFlash C=2 +70% precedent, `dflash-batched-verify`) — weeks, and the drafter's acceptance is sensitive to any numerics change on this model. |
| S4 | Aggregate grows C=1 -> C=2 but flattens C=2 -> C=4 | Same as S3 at the wider point; also read the MTP accept lines: if accepted/step falls with C (drafts starved by scheduling), the lever is scheduling, not kernels. |
| S5 | Long-arm TTFT of stream k ~ k x (30K / ~400 tok/s) — the last of 4 waits ~5 min | Prefill is not interleaved with other sequences' decode (or is, and is simply slow). Either way the lever is prefill throughput — Script 1's outcome — not the scheduler; do not chase chunked-prefill scheduling until prefill is >1K tok/s. |
| S6 | Any long-arm stream errors or finishes with `length` far below `max_tokens`, or `serve.log` shows KV/GDN-slot exhaustion or preemption at C=4 x 30K | The 128K x 4 envelope does not fit at util 0.72 as sized. Reduce `--max-seq-len` (fp8 KV is NOT available: QSA requires bf16 KV here) or `--max-num-seqs` in the preset; record the pool sizes from the alloc ledger in `serve.log`. |
| S7 | `swap_delta_gb` > 2 or `min_avail_gb` < 10 in any cell, or the watchdog aborts | Host memory pressure inside the pledge: the earlier 0.85 swap event reproduces at 0.72. Stop; attribute with the >= 32 MB allocation trace (`gx10-memory-growth-attribution`) before any further C=4 run. Do not raise util. |
| S8 | Per-stream tok/s at C=1 short vs the 30.3 baseline differ by > 15% in EITHER direction with S1 excluded | Prompt-length effect on the verify (2K vs 60-token prefix) or the preset's `preserve_thinking`; report as an observation and add a 60-token-prompt cell (`SHORT_TOKENS=64`) before quoting either number. |

Headline rule: only the short arm (n=3) may be quoted as a rate; the long arm is "preliminary,
single-shot" until rerun with `REPEATS_LONG=3`. Per-stream tok/s is quoted with its
accepted-per-step; a cell whose acceptance differs from C=1's by > 0.1 is not a kernel result.

## A/B discipline for both scripts

The preset is the only configuration. An arm is `ARM=<name> <KILL_SWITCH>=1 ./script.sh`: the
preset sets its env defaults only when the variable is unset and logs the deviation, so the
exported switch is the single variable and `fingerprint.txt` records it, the preset lines the
server actually applied, the binary sha256, the git head and `free -g`. Fresh server per arm,
control arm rerun on the same day (run-to-run spread on this box was <= 0.05 ms on the serial
gap, but the clippy-compile contamination on 09-05 shows the box must be otherwise idle:
check `pgrep -af cargo` and `nvidia-smi` before each arm).

No GPU program was started to write this plan; nothing here is measured.

## Files

- `prefill_profile.sh`, `nsys_kern_table.py` — Script 1 and its summariser (the summariser
  reproduces the decode baseline's table from `nsys_baseline_kern_sum.csv`: `w4a16_gemm` 50.7%).
- `concurrency_sweep.sh`, `measure_concurrency.py` — Script 2 and its driver.
- `measure_common.sh` — shared: refusals, preset boot (plain or under nsys), readiness wait,
  memory watchdog, fingerprint, cleanup on exit.
- `runs/` (gitignored) — where both scripts write; copy the tables you cite into the write-up.
- Binary: `$REPO/target/release/spark` by default (`SPARK_BIN=` to point at another build).
