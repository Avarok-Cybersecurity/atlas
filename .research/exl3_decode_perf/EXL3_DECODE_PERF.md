# EXL3 native decode on GB10 — where the token goes, and how to get to the llama.cpp band

Date 2026-09-05. Branch `wip/exl3-research` (PR #834) at `2274d01d7` plus the fix in this
directory's commit. Box dgx-00 (one GB10, 48 SMs, 24 MB L2, SM clock pinned ~2.3 GHz).
Checkpoint `turboderp/Qwen3.8-Flash-Next-exl3` 4.05bpw (`/tank/exl3-ckpt/qwen38-flash-next-4.05bpw`).
Measurement discipline: every number below carries its fingerprint; isolated-kernel numbers are
hypotheses, the e2e A/B decides.

## The question

EXL3 native serial decode was ~12.6-13 tok/s. The NVFP4 checkpoint on the same engine decodes at
17.9-20.5 tok/s and the llama.cpp band for this model on GB10 is 19-21 tok/s ("the llama speeds").
Weight bytes per token are about the same for both quantizations, so the ~25 ms/token gap had to
be structural, not bandwidth.

## Roofline (arithmetic, Rule 7)

Per decode token at 4.05bpw (K=6 dense/lm_head, K=4 experts), m=1:

| block | weights read | at 231 GB/s (measured streaming peak) |
|---|---:|---:|
| 36 GDN layers: in_proj_qkv + in_proj_z + out_proj (K=6) | 1.56 GB | 6.7 ms |
| 12 attention layers: q(12288)/k/v(512)/o (K=6) | 0.45 GB | 1.9 ms |
| 48 MoE layers: top-10 of 512 experts, gate/up/down 2560x640 (K=4) | 1.18 GB | 5.1 ms |
| 48 shared experts 2560x640x3 (NVFP4 today) | ~0.12 GB | 0.5 ms |
| lm_head 248320x2560 (K=6) | 0.48 GB | 2.1 ms |
| router 512x2560 bf16 x48, mHC low-rank, norms | ~0.13 GB | 0.6 ms |
| **total** | **~3.9 GB** | **~17 ms → ~59 tok/s ceiling** |

`exl3_decode_bench.cu` measured the device's streaming read at 231 GB/s (LPDDR5X 8533 MHz,
256-bit; 273 GB/s nominal).

## Microbench: the EXL3 kernels themselves are NOT the problem

`exl3_decode_bench.cu` compiles Atlas's exact `exl3_matmul.cu` wrappers and times each decode
shape at m=1 with 24-48 distinct weight copies cycled per launch (working set >> L2). Atlas's own
shape/grid selection (ported from `exl3_matmul.rs` / `mgemm_grid.rs`) versus the best of a sweep:

| projection (m=1) | Atlas config | Atlas | best found | gain |
|---|---|---:|---|---:|
| gdn.in_proj_qkv 2560→10240 K6 | gemm sh3 grid 48 | 93.9 us, 210 GB/s | sh4 grid 40: 89.5 us | 1.05x |
| gdn.in_proj_z 2560→6144 K6 | sh3 grid 48 | 57.7 us, 205 GB/s | sh2 grid 48 | 1.01x |
| gdn.out_proj 6144→2560 K6 | sh2 grid 48 | 59.5 us, 198 GB/s | — | 1.01x |
| attn.q_proj 2560→12288 K6 | sh3 grid 48 | 115.6 us, 204 GB/s | sh4 grid 48: 105 us | 1.10x |
| attn.k/v_proj 2560→512 K6 | sh2 grid 48 | 22.8 us, 43 GB/s | sh2 grid 20: 16 us | 1.4x (tiny abs.) |
| lm_head 2560→248320 K6 | sh4 grid 48 | 2039 us, 234 GB/s | — | 1.00x |
| routed experts T=1, S=10, K4 (gate+up+silu+down) | sh2 (8,6) / sh4 (8,6) | 171 us/layer, 143 GB/s | gate (4,10) 47.6 vs 57.1 us; down sh3 (4,10) 49.9 vs 58.2 us | ~1.2x |
| routed experts T=3, S=30 (MTP verify width) | (2,24) | 518 us/layer | (3,16): 121 vs 160 us per proj | ~1.3x |

Dense K=6 GEMMs at m=1 already run at 85-100% of the measured peak. Summed with Atlas's configs
the EXL3 trellis kernels cost **~21 ms/token** (dense 12.3 + routed 8.2 + converters ~0.5), which
is within 25% of the 17 ms roofline. Launch floor on this box: plain launch 2.0 us, cooperative
launch 4.1 us back-to-back (`cuLaunchCooperativeKernel` is a plain driver call here, no host sync).

Upstream's Blackwell notes (issue #242, `exl3_gemv_int8.cu`) say the fp16 kernel is per-SM
INT-throughput bound at ~65-78% of DRAM peak on a 5090 — GB10 has 3x less bandwidth per SM
(48 SMs / 231 GB/s vs 170 SMs / 1.8 TB/s), so the same kernel is DRAM-bound here. An int8-GEMV
port therefore cannot buy much on GB10 for dense; it is not a lever on this part.

## In-situ: the stage profiler and nsys

Baseline fingerprint (pass A/B): binary `/home/ms/atlas/target/release/spark` built from
`2274d01d7`, `serve_exl3.sh` (native MoE+dense+lm_head, no `--speculative`, C=1, util 0.72,
bf16 KV, 32K ctx, `reasoning_effort:low`), `measure_decode.py` (code prompt, 300 tokens, temp 0,
streaming, gaps after the first 5), port 8890, dgx-00, 2026-09-05.

- Pass A (`ATLAS_QWEN4EXP_DECODE_PROF=1`): **12.21 tok/s, 81.3 ms median gap** (n=3).
  Stage means per layer over 150 probes: **moe 1074 us**, ssm_forward 371, hc_post+hc_pre_ffn 158,
  hc_pre_attn 121, ple 18 → ~83 ms/token, of which the MoE stage is ~52 ms.
- Pass B (profiler off, under `nsys launch`, one 200-token run): 11.8 tok/s, 84.6 ms. GPU kernel
  time 15.6 s over 200 tokens = **78 ms/token busy** (GPU-bound; only 1.3 D2H syncs/token,
  ~2170 launches/token: 1567 `cuLaunchKernel` + 300 cooperative + 203 runtime + 97 cuBLASLt).

nsys kernel table (`nsys_baseline_kern_sum.csv`), per token:

| kernel | % GPU | ms/token | what |
|---|---:|---:|---|
| `w4a16_gemm` (M_TILE 64 prefill GEMM) x144 @ 274 us | **50.7** | **39.6** | **the NVFP4 shared expert at m=1: 3 launches x 48 layers** |
| `exl3_gemm_k6_cb2_sh3_f32` x84 | 8.9 | 6.9 | GDN in_proj qkv/z + attn q |
| `exl3_mgemm_k4_cb2_sh2_f16` x96 | 7.4 | 5.8 | routed gate/up |
| cutlass wmma bf16 (cuBLASLt) x97 | 9.6 | 7.5 | mHC collapse GEMMs |
| `exl3_gemm_k6_cb2_sh2_f32` x72 | 4.6 | 3.6 | GDN out_proj, attn k/v/o |
| `exl3_mgemm_k4_cb2_sh4_f32` x48 | 3.7 | 2.9 | routed down |
| `exl3_gemm_k6_cb2_sh4_f32` x1 | 2.6 | 2.1 | lm_head |
| `gated_delta_rule_decode_f32` x36 | 1.6 | 1.3 | GDN recurrence |
| hc_pre_mix / hc_pre_stage / hc_post / hc_silu x~97 each | 3.9 | 3.0 | mHC glue |
| everything else | ~7 | ~5 | router gemv, top-k, blend, converters, attention, conv, norms |

**Root cause.** `forward_exl3_after_routing` evaluated the shared expert through
`run_shared_expert_prefill`, which for the NVFP4-materialized shared weights is the prefill-tiled
`w4a16_gemm` (`M_TILE = 64`) — at m=1 that is 274 us per 0.8 MB projection (~3 GB/s). The NVFP4
decode path never pays this: it fuses the shared expert into the routed gate-up / silu-down kernels
as an extra slot (`forward.rs:517/599`). The EXL3 arm reused a prefill routine for a decode step.
That single mis-dispatch is ~40 of the ~81 ms token — the whole EXL3-vs-NVFP4 gap and then some.

## The fix (this commit)

`layers/moe/forward_exl3_shared.rs`: `run_shared_expert_exl3_decode` — for 1..=8 rows and an
NVFP4 shared expert, per-row `w4a16_decode_gemv` (the router's single-warp `w4a16_gemv_sw`,
~9 us at these shapes) for gate/up, the same `silu_mul`, per-row GEMV for down. Same scratch
buffers and output as the old arm, so `moe_batched_blend` is untouched. BF16 shared experts keep
`run_bf16_shared_expert` (already GEMV at one row); FP8 twins and >8 rows fall back to the prefill
arm. Kill switch `ATLAS_EXL3_SHARED_PREFILL_GEMM=1` restores the old dispatch for A/B.

Numerics: GEMV and tiled GEMM compute the same fp32 dot with the FP8 group scale factored out;
reduction order differs, so outputs are not bit-identical to the old arm (same contract as every
gemm-vs-gemv decode dispatch in the crate). Greedy output equality is checked in the A/B below.

### A/B — same binary, same flags, back-to-back, kill switch the only variable

Fingerprint: binary `spark-sharedgemv` built from this worktree (sha256
`4b896a5b6a2b9a8e…a68847`, single kernel target `qwen3.8-flash-next`, `decode_arm_build.sh` env),
`run_arm_serve_fix.sh` (identical flags to the baseline passes: native MoE+dense+lm_head, no
`--speculative`, C=1, util 0.72, bf16 KV, 32K ctx, `reasoning_effort:low`), `measure_decode.py`
code prompt / 300 tokens / temp 0 / streaming, port 8890, dgx-00, 2026-09-05 19:49-20:00, box
otherwise idle (the first fix-on run overlapped a clippy kernel compile; it is kept but the clean
repeat is the headline). Fresh server per arm.

| arm | boot | median gap (ms) | decode tok/s (server-attested tokens / wall) | per-run tok/s |
|---|---|---:|---:|---|
| control: `ATLAS_EXL3_SHARED_PREFILL_GEMM=1` (old arm, same binary) | fresh | **81.39** | **12.25** | 11.83 / 12.26 / 12.25 |
| fix on, run 1 (clippy compile in background) | fresh | 43.43 | 22.96 | 21.3 / 22.9 / 22.9 |
| fix on, run 2 (clean) | fresh | **43.29** | **23.03** | 21.5 / 23.05 / 23.03 |
| reference: baseline binary `2274d01d7`, pass A (profiler on) | fresh | 81.26 | 12.21 | 11.77 / 12.29 / 12.21 |

**1.88x decode (12.25 → 23.03 tok/s) with the kill switch the only variable.** The control arm
reproduces the baseline binary's number to 0.3%, so the old dispatch is the whole gap. 23 tok/s
serial is above the NVFP4 path's 17.9-20.5 band and at the top of the llama.cpp 19-21 band; the
remaining distance to the ~59 tok/s roofline is itemised below. TTFT is unchanged (~390 ms warm),
as expected — prefill never took this arm.

Greedy 200-token sample (`ab_greedy_sample_*.txt`): the fix-on arm is deterministic across the two
boots (identical bytes); it diverges from the control arm after ~40 tokens, both coherent and
on-task. That is the gemv-vs-tiled-gemm reduction-order difference, not a defect — but it means
this is not a bit-exact change, and the agentic/quality gates (`agentic-webserver` under
`ATLAS_AGENTIC_SAMPLING=model-card`) have NOT been rerun on this arm yet. Speculative decode was
off in every arm; the MTP verify width (3 rows) takes the same per-row GEMV path (rows ≤ 8) and
is unmeasured here.

### Agentic check on the fixed binary (one iteration)

User-directed profile (2026-09-05 20:13 local): fixed binary `spark-sharedgemv`, **prefix caching
ON** (`--enable-prefix-caching --ssm-cache-slots 64`, the user's standing rule — realistic agentic
configs depend on it and warm-restore bugs must be caught, not dodged), `--speculative
--num-drafts 2`, one sequence, 32K ctx, util 0.72, `reasoning_effort:low`; benchmark client
`agentic-webserver` with **greedy** sampling (no `ATLAS_AGENTIC_SAMPLING`) and
`ATLAS_AGENTIC_PRESERVE_THINKING=1`, `iterations=1`, `wall_budget_s=1000`, port 8888.

| result | value |
|---|---|
| verdict | **Pass** — `webserver_ok` 1/1, `followed_directions` 1/1, all six steps |
| turns / tool calls | 7 / 7 |
| Σ wall | 112 s (15.8 s/turn) |
| completion tokens | 1944 |
| harness decode_tps (incl. TTFT) | 17.5 |
| mangling markers in trajectory | 0 (greedy sampling is known to hide this defect, so this is not warm-restore evidence) |
| run record | `agentic/run-1788653752697008545.json`, trajectory beside it |

Reference from the PR's own agentic passes (old binary, model-card sampling, no prefix cache):
1 draft 11 turns / 390 s, 2 drafts 16 turns / 672 s. Different sampling and cache config, so an
observation, not an A/B — but the fixed arm completes the task in a fraction of the wall.

An earlier iteration in this session (fix on, model-card sampling, NO prefix cache) was cancelled
by the user at turn 8 while the agent's own `cargo test` was running; through those 8 turns the
server logged 20.5-25.0 tok/s per completion with two drafts.

## Routed-expert mgemm grid (second lever, this commit)

`mgemm_grid.rs` gains a `onewave` policy (default; `ATLAS_EXL3_MGEMM_GRID=legacy` restores the
ported heuristic): `per_slot = clamp(sms / top_k, 1, tiles)`, concurrency `min(sms / per_slot,
slots)`. On GB10 that is 4 blocks x 10 slots for one token instead of the legacy 8 x 6 (two waves,
the second 2/3 empty), and 4 x 12 in waves for the 30-slot MTP verify batch instead of 8 x 6 in
five waves. Per-slot split-K is identical between serial and verify under both policies (the
stable-grid contract; asserted by the tests for every replay width).

Clean microbench on the idle GPU (`microbench_moe_grid_plans.txt`), routed gate+up+down chain per
layer: S=10 legacy 163 us → onewave 144 us (1.13x); S=30 legacy-stable 416 us → 405 us (1.03x);
(3,16) at S=30 is 374 us but would change the serial split. Expected e2e: ~0.9 ms of a ~43 ms
token (~2%), i.e. at the edge of run-to-run spread. The first `grid` microbench pass ran while the
agentic server was decoding and read 268 us for the legacy chain — contaminated, discarded.

A/B (binary `spark-grid` = shared-expert fix + grid policy, built from this worktree; same
`measure_decode.py` fingerprint as above, fresh server per arm, port 8888, dgx-00 2026-09-05
20:20-20:33 local, box idle; `ATLAS_EXL3_MGEMM_GRID=legacy` the only variable within each profile):

| profile | legacy plan | one-wave plan | delta |
|---|---:|---:|---:|
| serial, prefix cache off (`serve_exl3.sh`) | 43.17 ms · 23.10 tok/s | **42.00 ms · 23.78 tok/s** | -1.17 ms (+2.9%) |
| MTP 2 drafts, prefix cache on (`serve_exl3_fix_agentic.sh`), server-attested tokens / wall | 25.95 tok/s | **26.58 tok/s** | +2.4% |

Per-run spread inside each arm was ≤0.05 ms on the serial gap, so the 1.2 ms delta is
resolvable and matches the microbench's 0.9 ms prediction. Greedy samples: the serial one-wave arm
differs from the legacy arm (the 4-block split changes the fp32 reduction order in every routed
projection — expected, same class as the shared-expert change); the two MTP arms produced
byte-identical 200-token samples. Under MTP the streaming gap median is meaningless (drafted tokens
arrive in bursts), hence the tokens/wall column.

Cumulative under the serial fingerprint: 12.25 → 23.78 tok/s (1.94x). Under the user's operating
profile (2 drafts + prefix cache): 26.6 tok/s on a 300-token greedy code prompt, versus 12.9-13.7
tok/s logged by the pre-fix TUI server earlier the same day at a 17-20K context (different
context length — an observation, not an A/B).

## The MTP step, profiled (where the next 2x is)

nsys on the user's operating profile (`spark-grid`, 2 drafts, prefix cache on, 300-token greedy
code prompt after one warm-up request; `nsys_mtp2_fixed_*.csv`): 25.8 tok/s at `tok_step 2.49`
→ ~120 steps → **~97 ms per 3-row verify step** (83 ms GPU busy, ~4,800 launches and 9 D2H syncs
per step), versus 42 ms per serial token. Speculation buys 2.49 tokens for 2.3x the cost, which is
why 2 drafts (26.6) barely beats serial (23.8). Per step:

| kernel | ms/step | inst/step | what it says |
|---|---:|---:|---|
| `exl3_mgemm` sh2 + sh4 (routed gate/up/down) | 19.7 | 144 + 72 | S=30 chain, 2.3x the serial 8.7 ms — bytes: 30 expert slots vs 10 |
| `exl3_gemm_k6` sh3 + sh2 (GDN/attn dense) | 15.7 | 108 + 144 | ~1.3x serial: the m=3 GEMM costs the same as m=1, but extra launches appear (draft module rows) |
| cuBLASLt cutlass wmma x2 (mHC collapse) | 14.1 | 295 | **3x serial** — the verify body runs the hyper-connection GEMMs once per row |
| `exl3_gemm_k6_cb2_sh4` (lm_head) | 6.1 | 3.0 | **three lm_head passes per step** (verify + two drafts), 2 ms each |
| `gated_delta_rule_decode_f32` (GDN recurrence) | 5.4 | 107 | one launch per row per layer — inherent to a recurrent verify |
| `w4a16_gemv_sw` (router + shared expert GEMV) | 3.8 | 577 | 48 x 3 rows router + 3 x 48 x 3 rows shared — the fix IS on this path |
| hc glue (`hc_pre_mix`, `hc_pre_stage`, `hc_post`, silu) | 4.9 | ~610 | per row |
| `exl3_moe_k4_n128_cb2` (prefill tier) | 1.8 | 0.8 | prefill of the prompt, amortised |
| everything else (attention, conv, blend, top-k, converters, argmax) | ~5 | | |

Levers this exposes, in order of size: (a) run the mHC collapse and glue once at M=3 instead of
per row (~9 ms/step, cuBLASLt at M=3 costs what M=1 does); (b) one lm_head pass per step or a
narrower draft head (~4 ms/step); (c) the routed mgemm at S=30 is 1.03x from its one-wave best
on this kernel — the remaining gain there is bytes, not scheduling; (d) launch count: ~4,800
launches/step at ~12 us host each is ~57 ms of host time hiding under 83 ms of GPU time — a graph
of the routed block would also take the 9 syncs/step off the critical path.

## MTP step: one step's anatomy and the row-exact chain (third lever, this commit)

One step from the trace (`nsys_mtp2_fixed_*`, `spark-grid`, 2 drafts, prefix cache on):
verify lm_head at t=0 → 92.5 ms 3-row verify pass over 48 layers → draft 1 (lm_head at 92.6) →
draft 2 (lm_head at 97.2) → next step at ~100 ms. 5,085 launches, 83.2 ms GPU busy, 17.0 ms idle:

| idle component | ms/step | mechanism |
|---|---:|---|
| ~2.6 us gaps between 5,085 consecutive launches | 13.4 | launch count, not host syncs |
| PLE at layer 2: 2 x (D2H carry snapshot → host hash+gather → H2D) + the first row's host gather | 2.3 | per-row `forward_row` + `push_verify_row` (`copy_d2h_on_stream`) |
| end of step: argmax D2H → host accept decision → next step | 1.3 | scheduler |

Inside the pass, per GDN layer: hc_pre attn ~0.19 ms (row 0's three GEMMs 35+12+12 us, rows 1-2
~29 us each — L2 hits), qkv/z 101+59 us (m=3 costs m=1), GDN recurrence 3 x ~55 us + conv/norm/
memcpy per row, out_proj 67 us, hc_pre ffn ~0.2 ms, router 3 rows 37 us, routed mgemm 285+143 us
(S=30, the one-wave plan), egress. ≈1.35 ms x 36 GDN layers + 12 attention layers ≈ 90 ms.

### The row-exact chain A/B (env only, `spark-grid`, MTP profile, fresh server per arm)

The mHC verify sets `gdn_exact_replay`, which by default ran the `hc_pre` and conv+GDN legs once
per row so the K-row verify reproduces K serial decode rows bit-for-bit (`gdn_flags.rs`
`RowExactLeg`). The crate's own tests pin "exact verify is opt-in" for every other body because of
its 22-36% cost; this was the one body running it by default.

| arm | tok/s | accepted/step | 200-token greedy sample |
|---|---:|---:|---|
| default (chain armed) | 26.56 | 1.49 | `c5d02cf5…` |
| `ATLAS_NO_VERIFY_ROW_HC=1` (hc leg off) | 26.19 | 1.32 | `5aee81be…` (differs) |
| `…_HC=1 …_GDN=1` (both off) | 29.84 | 1.47 | `c5d02cf5…` (identical to default) |
| `ATLAS_NO_VERIFY_ROW_EXACT=1` (all legs off) | **29.92** | 1.47 | `c5d02cf5…` (identical to default) |

Disarming only the hc leg buys nothing (its extra rows were already L2-served) and costs
acceptance; the GDN leg is the time. All legs off: **+12.6%** with the greedy sample unchanged.
Quality gate on the disarmed arm: `agentic-webserver` **Pass** — webserver_ok 1/1,
followed_directions 1/1, 8 turns / 118 s, harness decode_tps 19.2 (vs 17.5 with the chain armed),
0 mangling markers (`agentic/run-1788655475375096580.json`).

Change: `row_exact_lever` polarity flipped — the chain is opt-in via `ATLAS_VERIFY_ROW_EXACT`;
`ATLAS_NO_VERIFY_ROW_EXACT` still disarms and wins. Consequence, stated plainly: spec-on output is
no longer guaranteed bit-equal to spec-off on the mHC MTP path (the #435 divergence the rest of the
engine already ships by default); it was byte-equal on the probes above.

Confirmation on the rebuilt binary (`spark-noexact`, no env overrides, same MTP profile):
**29.89 tok/s**, accepted/step 1.47, greedy sample `c5d02cf5…` — identical to the env arm.
Cumulative on the operating profile (2 drafts, prefix cache on): 25.95 → 29.89 tok/s across the
grid and row-exact changes; on top of the shared-expert fix, the pre-fix TUI server logged
12.9-13.7 tok/s under this profile earlier the same day (longer context, observation not A/B).

### Model-card agentic gate on the final binary

The two passes above used greedy sampling, which hides the warm-restore string-mangling. Rerun on
`spark-plesnap` (all four changes) with `ATLAS_AGENTIC_SAMPLING=model-card`,
`ATLAS_AGENTIC_PRESERVE_THINKING=1`, `reasoning_effort:low`, prefix cache on, 2 drafts, one
iteration: **Pass** — webserver_ok 1/1, followed_directions 1/1, 6 turns / 97 s, 1834 completion
tokens, harness decode_tps 19.1, **0 mangling markers** in the 8.6 KB trajectory
(`agentic/run-1788657364366230696.json`). One sampled trajectory, not a rate.

Same again with thinking OFF (`--default-chat-template-kwargs '{"enable_thinking":false}'`, the
card's non-thinking preset applies): **Pass** — webserver_ok 1/1, followed_directions 1/1,
11 turns / 126 s, 0 `[reasoning]` blocks in the trajectory (thinking confirmed off), 0 mangling
markers (`agentic/run-1788657924008552574.json`).

## Launch count (fifth lever): batched shared-expert GEMV; the router must stay per-row

Fresh trace on the row-exact-off binary (`nsys_mtp2b_*`, `spark-plesnap`, 2 drafts, prefix cache
on): 86.0 ms/step wall, 72.5 ms busy, 13.5 ms idle = 10.8 ms of ~2.6 us gaps over 4,147
launches + 2.7 ms in four stalls. Launches per step, top of the list: `w4a16_gemv_sw` 580
(router 3 rows x 48 + shared expert 3 projections x 3 rows x 48), cuBLASLt cutlass 378 + splitK
reduce 225 (hc collapse at M=3, 11.2 ms busy ≈ 48% of DRAM peak), EXL3 converters 510
(`f32_to_bf16` 252, `bf16_to_f16` 183, `_2d` 75), D2D copies 197 (GDN conv-state shifts and hc
staging, 1-3 us each), hc glue ~610, per-row router+top-k 290.

Step tail: verify lm_head → three separate argmax+D2H pairs → **one 1.55 ms host gap** → draft 1
(MTP layer ~0.8 ms + lm_head 2 ms + argmax/D2H) → draft 2 with no gap → next step. The host gap
is `verify_mtp_wide::finish`: it copies all K verify logits rows to the host (3 x 248320 x bf16)
and runs `process_seq_logits` per row on the CPU — bf16→f32 dequant of 248K entries, penalties,
top-k/top-p/min-p — rows sequentially dependent. Removing it needs a GPU sampler (the GPU argmax
already runs and is ignored on this path); ~1.8%/step, ranked below.

`nvfp4_rows_proj` (forward_exl3_shared.rs): rows 1 → single-warp GEMV, 2/3 → the existing
`w4a16_gemv_batch2/3` (weights read once for all rows), else per-row loop; never the prefill-tiled
`w4a16_gemm`. Used by the shared expert (3 launches per layer per step instead of 9) and by the
verify router's non-row path, which had been falling into `w4a16_gemm` at M=3 — the reason the
profile had to arm `ATLAS_VERIFY_EXL3_ROW_ROUTER=1`.

A/B (`spark-batchgemv`, MTP profile, fresh server per arm, greedy 200-token sample vs the
`c5d02cf5…` reference):

| arm | tok/s | accepted/step | sample |
|---|---:|---:|---|
| previous binary (`spark-plesnap`) | 29.89 | 1.47 | identical |
| batched shared GEMV, per-row router (flag on) | **30.45** | 1.47 | identical |
| batched shared GEMV, batched router (batch3 GEMV + batched top-k) | 28.25 | **1.27** | differs |

Batching the router loses: verify routing that does not reproduce the serial rows' numerics
lowers draft agreement, the same effect as the hc leg earlier — routing and hc feed the drafter's
conditioning. So the per-row router and the stable one-token grid become the DEFAULTS
(`ATLAS_NO_VERIFY_EXL3_ROW_ROUTER` / `ATLAS_NO_VERIFY_EXL3_STABLE_GRID` disarm; the old `=1` arms
stay accepted as no-ops), and a bare serve gets the fast configuration.

Confirmation (`spark-defaults`, MTP profile with NO `ATLAS_VERIFY_EXL3_*` env at all): **30.34
tok/s**, accepted/step 1.47, sample identical. Cumulative on the operating profile today:
12.9-13.7 (pre-fix TUI log) → 25.95 → 26.58 → 29.89 → 30.34 tok/s.

## Hyper-connection collapse through the batched dense GEMV (sixth lever)

The mHC `hc_pre` low-rank collapse is three projections per site (down `[320 x 10240]`, up
`[10240 x 320]`, inject `[4 x 10240]`, all BF16), two sites per layer, 96 sites per token. They
went through cuBLASLt (`bf16_gemm_act_weight_t`): a heuristic per call, a split-K kernel + a
reduce launch per projection, ~48% of DRAM peak on the two 6.5 MB matrices — 7.5 ms per serial
token and 11.2 ms per 3-row step (378 cutlass + 225 splitK launches), and under the row-exact
contract one call per row because cuBLASLt picks a different kernel per M.

Tried: route `m <= 8` to `dense_gemv_bf16_batchm`, Atlas's own batched BF16 GEMV (one pass over
the weight for all rows, bit-identical per row to its M=1 form by construction: fixed K order,
`--fmad=false`), so serial, verify and the MTP draft module reduce each row identically with one
launch and no reduce kernel.

**Negative result** (`spark-hcgemv`, fresh server per arm, `ATLAS_NO_HC_DENSE_GEMV` the only
variable per profile):

| profile | cuBLASLt (default) | dense GEMV | accepted/step |
|---|---:|---:|---|
| MTP 2 drafts, prefix on | **30.36** | 29.05 | 1.47 → **1.37** |
| serial, prefix off | 23.48 | 23.70 | — (noise) |

The kernels were faster, but draft acceptance fell 7%, which outweighs it. All three consumers
took the same row-invariant kernel, so this is not a serial/verify mismatch — the dense GEMV's
numerics themselves cost draft agreement (the model's next-token distribution got noisier).
Shipped as OPT-IN (`ATLAS_HC_DENSE_GEMV=1`), cuBLASLt stays the default. Lesson, third time
today: on this model the drafter's agreement is sensitive to any numerics change in the hc / router
path; every such change must be judged on tokens/s WITH acceptance, not on kernel time. Records:
`ab_hc_*`.

Prefill note: the vllm-exl3 review (`vllm_exl3_prefill_review.md`) ranks the EXL3 prefill
levers: nsys an 8K EXL3 prefill first; raise the fused MoE tier's 128-row-per-expert cap (theirs
is 2048; ours overflows at every 4K chunk); a dense reconstruct-to-BF16 GEMM tier above a row
threshold using the byte-identical `exl3_reconstruct.cu` Atlas already has.

### Prefill lever 3, measured: dense reconstruct-to-BF16 tier (branch `perf/exl3-dense-reconstruct-prefill-tier`, DEFAULT 512 rows since the merged gate)

`ops/exl3_dense/reconstruct.rs`: for `m >= ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS` (presence arms; unset =
off, the default; `ATLAS_NO_EXL3_DENSE_RECONSTRUCT` kills) each dense weight is reconstructed once per
call — `exl3_reconstruct_had_k{K}_cb{cb}` (f16 `[in, out]`) + `exl3_f16_to_bf16_t` (bf16 `[out, in]`)
into two stage-owned slabs of `max_in x max_out x 2 B` (302 MB at 6144 x 12288, allocated at load
inside the util pledge only when armed) — and multiplied with the fixed-config tensor-core
`dense_gemm_bf16_pipelined` (128x128 tile, no split-K, no cuBLASLt heuristics; run-to-run
deterministic, m-independent per element). Contiguous destinations take the GEMM's BF16 C directly
(zero egress launches); the GDN `[Q|K|V|Z]` arena stages through `c_f16` per row batch + one
`cudaMemcpy2DAsync`. The m <= 8 decode arm is untouched (asserted by the mock plan tests).

Not bit-identical to the trellis tier: the reconstructed weight is rounded once to BF16 (2^-9
relative, a rounding the trellis path never takes), the activation is read unrounded (the trellis
path rounds it to f16 and rotates in f16), and the reduction order differs. HYPOTHESIS: swamped by
the 4-bit trellis noise, judged by the greedy samples and the model-card agentic gate. HYPOTHESIS on
speed: one trellis decode per weight per call instead of `rows/16` (512x at an 8192-row chunk) and a
~50 TFLOP/s GEMM in place of a kernel upstream measured at 0.59 TFLOP/s at m=128; nothing is a number
until `ab_dense_reconstruct.sh` runs (arms: off / 512 / 1024 on `measure_prefill.py` 8000/11000,
decode sanity, greedy samples recorded — expected to differ from the off arm).

**Measured 2026-09-06 00:32-00:41** (`ab_dense_reconstruct/`, binary `0ca060220467bfea` at `5e8b5cb46`
— the tier on the 128-row MoE cap base, so its "off" arm is also a clean 128-cap prefill baseline;
named preset verbatim, fresh server per arm, `measure_prefill.py` 8000/11000 x2, `measure_decode.py` x3):

| arm | 8K prefill tok/s | 11K prefill tok/s | decode tok/s | long greedy sample | short sample |
|---|---:|---:|---:|---|---|
| off (default) | 411 | 406 | 30.61 | `aef9543d…` 830 B | `c5d02cf5…` |
| `ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=512` | **439 (+6.8%)** | **433 (+6.7%)** | 30.59 | `69e8a436…` 834 B (differs, coherent) | identical |
| `ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=1024` | 435 | 427 | 30.47 | `69e8a436…` = 512 arm | identical |

Everything the module predicted held: decode (m ≤ 8) untouched, short prompt untouched (m < threshold),
the two armed arms byte-identical to each other, the long sample a one-time numerics change. The gain
is +7%, smaller than the "15% of GPU time" share suggested because the BF16 GEMM at 4096 rows is not
free either. On the merged binary (row cap 1024 + egress; "Merged-binary gate" below) the arm at 512 rows gives
491 / 485 tok/s against 454 / 453 (+8%) and PASSES the model-card agentic gate, so the DEFAULT is now
512 rows (`EXL3_DENSE_RECONSTRUCT_DEFAULT_ROWS`; `ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=<n>` moves it,
`ATLAS_NO_EXL3_DENSE_RECONSTRUCT` restores the trellis-only prefill).

## BF16 ingress fused into the dense GEMM prologue (seventh lever, bit-exact)

The dense decode arm ran `exl3_bf16_to_f16` before every projection group (183 launches per
MTP step, ~61 per serial token), then the cooperative GEMM, then the f32→bf16 egress. The GEMM's
input-Hadamard prologue already loads every activation element once; the `_abf16` kernel twins
(`exl3_gemm_k{K}_cb{cb}_sh{S}_f32_abf16`, all K/cb/shapes) load BF16 there and convert with the
converter's exact `__float2half_rn(__bfloat162float(x))` before the identical rotate. So the
fused path is bit-identical to convert-then-GEMM by construction — no acceptance risk, unlike the
numerics-changing levers above. `exl3_dense_linear_shared_a` skips the ingress launch whenever
the group has no GEMV-tier weight (K ∉ 2..=4), i.e. every dense projection on the 4.05bpw
checkpoint; the GEMV tier keeps the f16 ingress. Egress (`f32_to_bf16`, 252 launches per step)
is the remaining half and needs an epilogue store variant inside the inner kernel (not done).

A/B (`spark-abf16` vs the previous binary's control arms, fresh server per arm):

| profile | before | fused ingress | sample |
|---|---:|---:|---|
| MTP 2 drafts, prefix on | 30.36 tok/s (1.47 acc/step) | **30.71** (1.47) | byte-identical |
| serial, prefix off | 23.48 | **23.89** | byte-identical |

+1.2% / +1.7%, exactly the launch-gap arithmetic (≈61 launches x ~2.6 us gaps + ~1.3 us kernels
per token), with both greedy samples unchanged — the first lever today with zero numerics
exposure. Records: `ab_abf16_*`.

## BF16 egress fused into the dense GEMM epilogue (eighth lever, bit-exact, measured)

Branch `perf/exl3-gemm-bf16-egress`. The other half of the seventh lever: after the `_abf16`
GEMM the dense decode arm still launched `exl3_f32_to_bf16` (contiguous destination) or
`exl3_f32_to_bf16_2d` (a column block of the GDN `[Q|K|V|Z]` arena row) per projection — 252
launches per 2-draft MTP step in the `nsys_mtp2b_*` trace, ~61 per serial token. The inner
kernel's output-Hadamard epilogue (`output_had_sh_gl` → `had_ff_r_128_inner<false, true>`) is
where the final f32 C values exist in registers: only the LAST split-K block of a column runs
it, after the rotate and the `svh` post-scale, so the values it stores are exactly the ones the
converter would read back. The new `out_bf16` template arm (`had_ff_r_128_inner_obf16`) keeps
the f32 store and ALSO stores `__float2bfloat16_rn(v)` of the same four registers into a pitched
BF16 destination (`C_bf16 + row * ld_bf16 + col`), which is the converter's rounding on the
identical value — byte-identical to convert-after by construction, and the `_2d` pitch becomes a
kernel argument. Kernel twins `exl3_gemm_k{K}_cb{cb}_sh{S}_f32_abf16_obf16` (two extra trailing
args; `EXL3_GEMM_ARGS` and every existing instance untouched), host `exl3_gemm_abf16_obf16`
(`exl3_matmul/gemm.rs`, the GEMM launchers split out of `exl3_matmul.rs` for the LoC cap),
probed at load by `exl3_dense_kernel_names`, dispatched by `exl3_dense_linear_shared_a` on the
m ≤ 8, K ∉ 2..=4 path — one launch per dense projection instead of two (three before the
ingress lever). The GEMV tier (K in 2..=4) and the m > 8 tier are untouched.

Kill switch `ATLAS_EXL3_NO_FUSED_EGRESS` (presence) restores the `_abf16` + converter form;
the arm logs `EXL3 dense decode egress: …` once so an A/B record can prove which arm was live.
Launch-plan tests pin both plans (`exl3_dense/tests.rs`), the name list pins the probe.

**Hypothesis, not a measurement:** the same launch-gap arithmetic as the ingress lever
(~61 launches x ~2.6 us + ~1-2 us of converter kernel per serial token ≈ 0.2-0.3 ms of a ~42 ms
token; ~252 per MTP step) puts the expected gain at +0.5-1.5% in each profile, i.e. at the edge
of run-to-run spread, and the greedy samples must be byte-identical in both profiles (a
differing sample is a bug). Nothing here is measured until `ab_fused_egress.sh` has run on the
GPU: `.research/exl3_decode_perf/ab_fused_egress.sh` boots the branch's release binary with the
named preset (MTP profile) and the `serve_exl3.sh` flag set (serial profile — the preset cannot
drop `--speculative`), kill switch off then on per profile, `measure_decode.py` x3 + the
200-token greedy sample, asserts byte-equality and that the logged arm matches, writes
`ab_fused_egress_<stamp>/SUMMARY.txt`.

**Measured 2026-09-06 00:16-00:23** (`ab_fused_egress/ab_fused_egress_20260906_001602/`, binary
`d086972fa45bf1cd` at `6a54a005f`, dgx-00, port 8899, 4.05bpw checkpoint, kill switch the only
variable, fresh server per arm, `measure_decode.py` x3, code prompt, 300 tokens, temp 0, effort low):

| profile | arm | decode tok/s (tokens/wall) | gap median ms | greedy sample |
|---|---|---:|---:|---|
| MTP-2 preset | off (`ATLAS_EXL3_NO_FUSED_EGRESS=1`) | 30.05 | — | `c5d02cf5…` 775 B |
| MTP-2 preset | on (default) | **30.66 (+2.0%)** | — | `c5d02cf5…` byte-identical |
| serial (`serve_exl3.sh`) | off | 22.63 | 43.38 | `36dc8fc6…` 814 B |
| serial (`serve_exl3.sh`) | on (default) | **23.51** | **42.16 (+2.9%)** | `36dc8fc6…` byte-identical |

Same order as the ingress lever and bit-exact in both profiles, as predicted. One caveat on the
record: the script's arm-liveness grep for the "on" arm is vacuous (review finding), so the arm is
proven live by the timing difference plus the kill-switch default, not by the log line. Ships as the
default.

## GPU-greedy pick for the wide-verify tail (tenth lever, NEGATIVE, opt-in scaffold not merged)

Branch `perf/mtp-gpu-sampler-scaffold` (`scheduler/verify_mtp_wide/gpu_greedy.rs`, design in
`GPU_SAMPLER_DESIGN.md`): `verify_mtp_wide::finish` copies the `[K, vocab]` BF16 verify logits to the
host and runs `process_seq_logits` per row (the 1.55 ms end-of-step gap in the launch-count trace).
The arm (`ATLAS_MTP_GPU_GREEDY=1`) runs a device argmax for greedy-eligible rows and only copies the
picks; `ATLAS_MTP_GPU_GREEDY_CHECK=1` runs both pickers, emits the host pick, and logs every
disagreement as either an exact bf16 tie or a defect.

**Measured 2026-09-06 00:23-00:32** (`ab_mtp_gpu_greedy/`, binary `66fe19e15e8f8dfa` at `fdacc652c`,
named preset verbatim, arms off/on/off/on, `measure_decode.py` x3 each, code prompt, 300 tokens,
temp 0, effort low):

| arm | decode tok/s (tokens/wall) | greedy sample |
|---|---:|---|
| off (control) | 30.49, 30.51 | `c5d02cf5…` 775 B |
| on (`ATLAS_MTP_GPU_GREEDY=1`) | **29.36, 29.37 (−3.7%)** | `5c619b45…` 812 B — DIFFERS |
| check (both pickers, host pick emitted) | 29.05 | `c5d02cf5…` (= control); **4 exact bf16 ties, 0 defects** |

Two findings. (1) The device argmax over 248K x 3 BF16 logits plus its own sync costs MORE than the
host gap it removes — the hypothesis (+1.8%) had the sign wrong; the step tail was never the host
sampler alone. (2) The sample divergence is NOT a defect: `check_ties.txt` shows the disagreements
are exact bf16 ties at the maximum (e.g. `host=2577 device=586 logit=22.25`, twice at the same
position across requests), where the host picker keeps the LAST index and the device kernel the
FIRST. bf16 logits tie at the top a few times per 300 tokens on this model, so any GPU picker that
ships must reproduce the host tie-break or greedy output changes. The branch stays unmerged (kept
locally as the design record); the residual lever here is a GPU sampler that also does penalties /
top-k / top-p (the design doc), and it only pays if the whole finish path moves, not the argmax.

## PLE verify snapshots on device (fourth lever, this commit)

`push_verify_row` snapshotted the PLE carry to a host blob with `copy_d2h_on_stream` after
each verify row a partial accept can land on — a stream sync per row. In the step timeline each
of the two per-step snapshots drained the queue and then idled the GPU 0.7-0.8 ms while the host
hashed and gathered the next row's n-gram embeddings. The conv state (~160 KB FP32) now copies
device-to-device into a per-row slot (`PleSeqState::verify_conv`, allocated on first use, freed
with `conv`), and the token history — host state anyway — is cloned. `rewind_verify_row` copies
back from the slot. Numerics unchanged by construction; partial accepts exercise the rewind every
few steps (accepted/step ≈1.47 of 2), so a wrong rewind would show up as a diverging sample.

Measured (`spark-plesnap`, MTP profile, same fingerprint as the row-exact confirmation, fresh
server): **29.86 tok/s** vs 29.89 on the previous binary — neutral within run-to-run spread;
accepted/step 1.47 unchanged; greedy sample `c5d02cf5…` byte-identical (the rewind path is
exercised and correct). The two syncs per step were cheaper than the 0.7-0.8 ms stalls around
them suggested: most of that idle was the host hash + gather of the next row, which this change
does not touch. Kept because it removes two host syncs per step from the critical path at no
cost and closes the host-blob asymmetry with the SSM intermediates; the host-side prestage of
all K rows' gathers at pass start is the follow-up that would recover the remaining ~2 ms.
Record: `ab_plesnap_mtp.txt`.

## What is left after the fix, ranked (hypotheses from the trace, not yet measured e2e)

Post-fix serial token budget is roughly: EXL3 trellis kernels ~21 ms + cuBLASLt mHC ~7.5 ms +
mHC glue ~3 ms + GDN/attention/router/blend/converters ~5 ms + launch gaps ~7 ms. (The routed
grid item below has since been implemented and measured: +2.9% serial / +2.4% MTP, see above.)

1. **Native EXL3 shared expert as an 11th mgemm slot** (~0.5 ms/token, plus fidelity/memory).
   The checkpoint ships the shared expert packed at K=4 with exactly the routed-expert geometry
   (2560→640→2560). Add it to the per-projection pointer tables as index `num_local`, stage one
   extra slot per token with `b_weights = sigmoid(input @ w_sg)` (one tiny kernel), and drop the
   NVFP4 shared pass + `moe_batched_blend`. Removes the last EXL3→BF16→NVFP4 double quantization on
   the MoE and 5 launches/layer. Touches `exl3_materialize_moe.rs` (keep predicate),
   `ptr_table_build.rs`, `exl3_moe_stage_ingress`.
2. **MoE mgemm grid** (~1.5 ms/token at T=1, ~4 ms at the T=3 verify width). `mgemm_grid` puts
   8 blocks per slot x 6 slots and walks 10 slots in two waves; the sweep prefers 4 x 10 (one wave)
   at T=1 and 3 x 16 at T=3, 1.2-1.3x per kernel. Upstream fixes this with a cooperative
   autotuner (`coop_autotune.cu`: sweeps `num_sms` per shape under L2 thrash, disk-cached). A
   frozen per-(K, k, n, S) table is enough here — but keep serial and verify on the same plan
   (the exact-mismatch that `stable_token_grid` repaired).
3. **mHC collapse cuBLASLt GEMMs** (7.5 ms/token, 97 calls): two cutlass wmma kernels at ~39 us
   each per call; the 2026-08-27 lever ladder already moved these from 254 to 122 us/layer. Not
   EXL3-specific; shared with the NVFP4 path.
4. **Launch gaps (~7 ms/token, ~2170 launches).** CUDA graphs are vetoed here (QSA host top-k,
   PLE host hash, cooperative launches). Upstream captures cooperative mgemm launches in graphs
   (`blocksparse_mlp.cpp run_bszN_gr`), and the vendor review's smoke test proved cooperative
   stream capture works on this CUDA 13.0 / GB10 host — a routed-expert-only graph per layer is the
   bounded next step, after the shared-expert slot lands (so the graph covers the whole FFN).
5. **Dense K=6 shape picks** (~0.3 ms/token): sh4 over sh3 for n ≥ 10240, grid 20 for the 512-wide
   k/v. Real but small; fold into the same frozen plan table as item 2.

Not levers on GB10: int8-activation GEMV (dense already at DRAM peak here), GEMV tier for K=6
(upstream has none; dense GEMM is at peak anyway), the two plain ingress kernels (already fused).

## Prefill baseline (measured 2026-09-05, not yet investigated)

`measure_prefill.py`: unique salted prompt per request (no prefix-cache hit possible),
`max_tokens=1`, prefill tok/s = server `prompt_tokens` / wall. Binary `spark-plesnap` (all four
changes above), MTP profile (2 drafts, prefix cache on, `--ssm-cache-slots 64`, util 0.72, 32K
ctx, default `--max-prefill-tokens` 8192 so the 11K prompt is two chunks), port 8888, dgx-00, idle.

| prompt tokens | wall | cold prefill tok/s (2 repeats) |
|---:|---:|---:|
| 8006 / 7977 | 20.9 / 20.4 s | 383 / 391 → **387** |
| 10986 / 10978 | 28.3 / 27.9 s | 389 / 393 → **391** |
| (6379 / 8777, first sizing pass) | 17.2 / 22.8 s | 377 / 389 |

Flat at ~390 tok/s from 6K to 11K. User-supplied reference points: other engines ~1.1K tok/s at
8K on this model, NVFP4 ~2.6K; Atlas historically ≤~500.

### Prefill profile (nsys, one cold 7,993-token prefill, preset binary at `ffdffa467`)

`nsys_prefill8k_kern_sum.csv`; 407 tok/s under the profiler, 18.7 s of GPU kernel time in a 19.6 s wall.

| kernel | % | ms | instances | what |
|---|---:|---:|---:|---|
| `exl3_gemm_k4_cb2_sh2_f16` | 34.0 | 6364 | 11748 | **MoE OVERFLOW tier**: chunked per-expert GEMM for experts above the 128-row cap (gate/up) |
| `qsa_prefill_attn` | 13.4 | 2509 | 48 | QSA selected attention |
| `exl3_gemm_k4_cb2_sh1_f32` | 10.5 | 1959 | 5874 | overflow tier, down projection |
| `exl3_gemm_k6_cb2_sh3_f16` | 9.1 | 1705 | 252 | dense GDN/attn trellis GEMM at 4096 rows (16-row slabs) |
| `gated_delta_rule_prefill_regresident` | 4.8 | 894 | 72 | GDN chunked recurrence |
| `exl3_gemm_k6_cb2_sh2_f32` | 4.5 | 840 | 144 | dense trellis GEMM |
| `w4a16_gemm` | 4.1 | 770 | 384 | shared expert (NVFP4) at prefill M |
| `dense_gemm_bf16_pipelined` | 3.3 | 611 | 1164 | mHC collapse + PLE projections |
| `exl3_moe_k4_n128_cb2` | 3.1 | 585 | 144 | the FUSED MoE tier itself |
| overflow gather / store / reduce (`exl3_moe_gather_rows_h16`, `_store_slots_f32`, `_reduce_slots_f32`) | 3.0 | 567 | 11892 | overflow tier plumbing |
| `inferspark_prefill_64` | 1.6 | 301 | 12 | full attention |

**48% of prefill GPU time is the overflow tier** (8.9 s of 18.7 s), while the fused kernel that the
overflow spills from is 3%. At the default 4096-token chunk the mean is 80 rows per expert with a heavy
tail, so most experts exceed the 128-row cap and take ~245 serialized cooperative GEMM launches per
layer. Raising the cap (upstream derives it from the temp slab; vllm-exl3 runs 2048) is therefore the
first prefill lever, worth up to ~1.9x if the fused kernel absorbs those rows at its current
efficiency — hypothesis until the A/B. Next: QSA attention (13%), then the dense K=6 trellis GEMMs
(15% combined) which a reconstruct-to-BF16 GEMM tier would replace.

## MoE prefill per-expert row cap 128 -> 1024 (ninth lever, prefill, measured)

Branch `perf/exl3-moe-prefill-row-cap` (`moe_prefill/cap.rs`): the fused `exl3_moe` prefill kernel's
`max_tokens_per_expert` was only the temp-slab height — the kernel's GEMM loops walk any row count in
16-row tiles — so the 128-row constant that sent 48% of prefill GPU time through the serialized
overflow tier was a slab-sizing choice, not a kernel limit. The cap is now 1024 rows per expert by
default (`ATLAS_EXL3_MOE_ROWS_PER_EXPERT=<n>` to override, `ATLAS_NO_EXL3_MOE_WIDE_ROWS=1` restores
128); the boot line `EXL3 native MoE state allocated: … fused C=6 x 1024 rows/expert [Default] =
78.6 MB temp slabs` proves the arm.

**Measured 2026-09-06 00:06-00:13** (`ab_moe_row_cap/20260906T000603/`, binary `c1001dfb90b9d0dd` at
`b72022fe5`, the named preset verbatim: 128K x 4, util 0.72, 2 MTP drafts, prefix cache ON, prefill
chunk 8192 so the 11K prompt is two chunks; MoE prefill batch 4096; salted-unique cold prompts, x2 per
size; fresh server per arm, kill switch the only variable):

| arm | 8K prefill tok/s (2 reps → median) | 11K prefill tok/s | decode tok/s (MTP) |
|---|---:|---:|---:|
| legacy 128 (`ATLAS_NO_EXL3_MOE_WIDE_ROWS=1`) | 379, 304 → 341 | 312, 386 → 349 | 27.88 |
| **default 1024** | **454, 456 → 455** | **447, 448 → 448** | 30.18 |

The legacy arm's reps are 20% apart because workflow agents were compiling on the host during that arm
(the same perturbation seen on 2026-09-05); the clean baseline for this preset measured the day before
is 387 / 391 (section "Prefill baseline"). Read the gain as **+15-17% against the clean baseline and up
to +30% against the in-run control**, not 1.9x: the overflow tier is gone from the common case but the
fused kernel now spends longer per layer (it was 3% of GPU time at 128 rows). The decode column is a
sanity check only (the cap has no decode path); the MTP profile's 27.9 vs 30.2 is the same host-load
effect. Outputs: the 200-token greedy sample and a 6K cold prompt are text-identical across arms
(the harness's byte-compare flagged the usage JSON line, whose only differences are
`time_to_first_token_ms` and `response_token/s`; the compare must strip timing fields — review
finding, fixed on the branch). The 6K cold-vs-warm-restore difference is the known prefix-cache
restore class ([[qwen4exp-phantom-corruption]]), present in both arms.

Remaining prefill headroom after this: QSA prefill attention (13% of the old profile, now a larger
share), the dense K=6 trellis GEMMs at 4096 rows (15%; the reconstruct-to-BF16 tier below), the shared
expert through `w4a16_gemm` (4%), and a fresh nsys to re-rank.

### Mechanism, numerics, and the parity gate (from the implementing commits)

Branch `perf/exl3-moe-prefill-row-cap`. Mechanism, from `vllm_exl3_prefill_review.md` R2: the
fused `exl3_moe` kernel's `max_tokens_per_expert` is only the temp-slab height (a runtime
argument; upstream derives it from `temp_state_g.shape[1]`, vllm-exl3 runs 2048). Atlas passed
the constant 128, so at the 4096-token prefill batch (40,960 slots over 512 experts, mean 80
rows/expert, heavy tail) most hot experts fell to the overflow tier — per expert, per 1024-row
chunk, five host-issued launches, three of them cooperative `exl3_gemm` grids of ≤48 blocks that
serialize behind each other. The cap is now resolved at model build
(`ops/exl3_matmul/moe_prefill_cap.rs`): default **1024**, `ATLAS_EXL3_MOE_ROWS_PER_EXPERT=<n>`
overrides, kill switch `ATLAS_NO_EXL3_MOE_WIDE_ROWS` (presence) pins the legacy 128 and wins over
the numeric knob. Clamped loudly to the token-batch cap (no expert can hold more rows than tokens)
and to the kernel's only bound, its 32-bit slab-index arithmetic (`C·rows·max(H,I) < 2^31`; at
C=6, H=2560 that is ~139K rows). The boot log prints the resolved cap, its source and the temp-slab
bytes: `fused C=6 x 1024 rows/expert [Default] = 78.6 MB temp slabs`.

Slab arithmetic (C=6, H=2560, I=640, four f16 slabs): 128 → 9.8 MB, 512 → 39.3 MB, **1024 →
78.6 MB**, 2048 → 157 MB, 4096 → 315 MB — versus the 419 MB deterministic slot slab already paid.
1024 = 12.8× the mean and equals the overflow chunk; beyond it a single expert holding a quarter of
the chunk is routing pathology where an 8-SM group walking 64 sixteen-row passes is the
load-balance risk (HYP) and the overflow tier's 48-block GEMMs fit better.

Numerics: same kernel, same grid policy, same deterministic per-slot epilogue for every expert the
fused tier serves — the cap only changes WHICH experts it serves. Experts with 128 < rows ≤ 1024
move from the overflow tier's cooperative `exl3_gemm` onto the fused kernel's 16×32×128 MoE tile:
same trellis decode and f16 activation precision, different fp32 accumulation order — a one-time
bit change, not run-to-run. That was a code-reading argument (the kernel consumes
`max_tokens_per_expert` only as the per-group slab stride and the `token_count > cap` skip; the
tile walk and the `output_slots` epilogue depend on `token_count` alone); the review of the first
commit correctly noted that no gate exercised the fused kernel above 128 rows. **Fix round
(2026-09-06): the parity harness now does** — `legs_moe_prefill_wide.rs` re-runs sub-leg 3's T=192
skewed batch (expert 0 at ~460 rows, overflow at cap 128) on slabs sized at cap 512 (host-sync, every
expert fused, asserted `overflow_experts == 0` and `num_active` == the host count of non-empty
experts) and at cap 768 = S (the no-sync shortcut, asserted `num_active == -1`), both gated vs the
f64 reference at the same rel 8e-3 / z 8e-2, and diffs each run's bf16 bits against the cap-128 run
of the SAME inputs per token class: tokens with no expert-0 slot (every expert they touch fused in
both runs) are ASSERTED bit-identical; the expert-0 tokens' mismatch fraction, max |Δ| and max
bf16-ulp are printed — the "one-time fp32-order change" becomes a measured number, not a claim. The
512 and 768 runs are asserted bit-identical to each other where their grids coincide (GB10: C=6,
48 SMs → 8 SMs/group either way). `EXL3_MOE_PREFILL_ONLY=1` runs just legs G+G2, and
`ab_moe_row_cap.sh` runs that as step 0 before any server boots — a parity failure aborts the A/B.
**RUN 2026-09-06 00:47 on the merged binary** (`parity_moe_prefill_wide/parity_moe_prefill_20260906.txt`,
`EXL3_MOE_PREFILL_ONLY=1`, exit 0): every assertion held. At cap 512 (host-sync) the fused tier served
all 16 non-empty experts including expert 0 at 487 rows (`num_active=16, overflow_experts=0`); at cap
768 the no-sync shortcut (`num_active=-1, overflow 0`); both match the f64 truth at rel_rms 1.860e-3 /
max_z 2.515e-2 — the SAME max_z as the cap-128 overflow run, i.e. the worst token is unchanged and the
moved experts sit inside the existing gate. The 5,120 bf16 outputs of tokens that never touched expert
0 are bit-identical to the cap-128 run; the expert-0 tokens differ in 26% of their bf16 bits (max |Δ|
128 on outputs whose scale makes that z < 0.025; the 33,164-ulp figure is sign flips of near-zero
values, not a numerics defect). Cap 512 and cap 768 are bit-identical to each other (same 6x8-SM grid). A batch with no expert above 128 rows (any prompt ≤ 128 tokens) must
be BYTE-IDENTICAL across arms; the A/B asserts it on the 200-token greedy LRU-cache sample. Side
effect, also upstream's semantics: the S ≤ cap no-sync shortcut now covers S ≤ 1024 slots
(≤ 102 tokens at top-10), removing the one host D2H for short batches (HYP: small TTFT win at
MTP-verify/short-prompt shapes).

The A/B script (its results are in the table above):
`.research/exl3_decode_perf/ab_moe_row_cap.sh` — preset
`spark serve qwen3.8-flash-next-exl3 --model-from-path /tank/exl3-ckpt/qwen38-flash-next-4.05bpw
--bind 127.0.0.1 --port 8899`, fresh server per arm, kill switch the only variable, boot-log
cap-line gate against inert arms, `measure_prefill.py --tokens 8000 11000 --repeats 2`,
`measure_decode.py --repeats 2 --max-tokens 300`, short-sample byte-equality assertion, long-prompt
cold/warm and cross-arm samples (reported; the numerics gate is step 0's parity run, whose
`moe-prefill WIDE` lines are copied into the verdicts). Baseline: ~390 tok/s flat 6K–11K (`prefill_baseline_8k_11k.txt`); measured 455 / 448 above.

## Merged-binary gate (2026-09-06 00:47-00:56, all round-2 levers on `wip/exl3-research`)

`merged_gate.sh` → `merged_gate_20260906T004709/`: the named preset verbatim on the merged binary
(fused egress default, MoE row cap 1024 default, reconstruct tier still opt-in at this commit), one boot
per arm, then prefill 8K/11K x2 (cold, salted), decode x3, a 200-token greedy sample, and ONE
agentic-webserver iteration with model-card sampling and preserved thinking, prefix caching ON.

| arm | 8K prefill tok/s | 11K | decode tok/s (MTP-2) | greedy sample | agentic (model-card) |
|---|---:|---:|---:|---|---|
| default | 454 | 453 | 30.82 | `b6e9fa3c…` 779 B | **PASS** 1/1 webserver_ok, 1/1 followed, 127 s, 11 MTP accept lines |
| `ATLAS_EXL3_DENSE_RECONSTRUCT_ROWS=512` | **491** | **485** | 30.55 | `b6e9fa3c…` identical (92-token prompt is below the threshold) | **PASS** 1/1, 1/1, 88 s |

Against the session's opening state (prefill ~390 flat, decode 12.25 serial / ~13 MTP) the merged
binary reads 491 / 485 prefill and 30.8 MTP-2 decode. Both arms produce identical greedy text on the
short prompt and coherent webserver trajectories (`agentic_*_trajectories/`). The reconstruct arm's
agentic pass is what flips its default to 512 rows in the following commit; the parity gate for the
row cap (`parity_moe_prefill_wide/`) ran on this same binary just before the gate.

**Final gate on the shipped defaults** (`merged_gate_20260906T010442/`, binary rebuilt with
`EXL3_DENSE_RECONSTRUCT_DEFAULT_ROWS = 512`, no env at all): boot log shows `1024 rows/expert
[Default]` and `reconstruct tier ARMED: m >= 512`; prefill **478 / 486** tok/s (8K / 11K), decode
**30.64** tok/s (MTP-2), greedy sample `b6e9fa3c…` identical to both arms above, agentic model-card
**PASS** 1/1 webserver_ok, 1/1 followed, 92 s, 9 MTP accept lines. Session totals: prefill ~390 → ~480
tok/s (+23%), decode 12.25 serial / ~13 MTP → 23.9 serial / 30.7 MTP-2 (2.5x / 2.4x).

## Concurrency (measured 2026-09-05 late, preset `qwen3.8-flash-next-exl3`: 4 slots, 128K)

`measure_concurrency.py`: C simultaneous streaming requests with distinct salted ~2K prompts,
300 tokens, temp 0; per-stream decode tok/s from server-attested completion_tokens / decode wall;
host memory sampled every 5 s. Records `concurrency_*.txt`.

| arm | C=1 | C=2 | C=4 |
|---|---:|---:|---:|
| MTP 2 drafts (preset) — per-stream | 27.6 | 13.2 / 10.8 | 5.2–6.3 |
| MTP 2 drafts — aggregate | 19.5* | 18.6 | 18.3 |
| speculation OFF (same flags otherwise) — per-stream | 12.7 | 11.6 / 14.4 | 5.7–7.9 |
| speculation OFF — aggregate | 10.1* | 19.3 | 20.9 |

\*aggregate at C=1 includes the ~5 s TTFT in its wall. **With MTP on, aggregate is flat at 18-19
tok/s from C=1 to C=4**: the mHC K-row verify runs per sequence (serialized), so drafts buy nothing
once a second sequence is active. Without MTP the batched highway decode scales (10 → 19 → 21) but
from a lower base. The C=1 speculation-off figure of 12.7 is under investigation (an nsys of the same
path at C=1 with a short prompt decoded 19.1 tok/s with a healthy kernel mix — `nsys_batched_c1_*`);
see the follow-up measurement below.

At 30K-token prompts: C=2 completed with 141–157 s TTFT (two ~30K prefills queue at ~400 tok/s);
at C=4 three of four streams ended `finish_reason=timeout` at 250–275 s TTFT — the 300 s default
request timeout, not KV (the pool holds 183K tokens at util 0.72 with 70.1 GB pre-KV, so 4 x 128K
is NOT resident either: the 128K preset bounds one sequence, four of them queue/preempt). Host
available memory fell to 7 GB during that arm. The preset now sets `request_timeout = 1800` (the
TUI launcher's value).

Implications, ranked: (1) prefill throughput is the concurrency wall at long prompts (the overflow
tier above); (2) an MTP policy that disables speculation once ≥2 sequences are active would recover
the batched path's scaling (hypothesis: ~21 aggregate at C=4 vs 18 today) — the `mtp_gate` already
arbitrates by measured verify cost per sequence, not by concurrency; (3) KV at util 0.72 is 4.2 GB
on this checkpoint — 4 x 128K needs either fp8 KV (refused: QSA needs bf16) or a higher util with the
host-memory watch the 30K arm just tripped.

Follow-up on the 12.7 (fresh speculation-off server, same preset): short prompt 300 tokens x2 →
18.0 tok/s (gap median 45.2 ms but gap MEAN 56.1 ms — ~25% of decode time in periodic stalls, 5
`Saved SSM snapshot` decode-time checkpoints logged); 2K prompt C=1 three times → 19.5, 20.6, 21.8
tok/s (the third a prefix-cache hit, TTFT 0.2 s). So 12.7 was a cold first-request outlier (host
swap use peaked at 5 GB during it). Steady state on the 4-slot preset without speculation is 18-22
tok/s at C=1 versus 23.9 on the single-slot serial profile with prefix caching off, whose gap mean
equalled its median. Two levers fall out: the decode-time snapshot stalls (~20% at C=1 on this path;
prefix caching stays on by rule, so the fix is making the snapshot D2H asynchronous or less frequent,
not turning it off), and host memory headroom on the 128K x 4 preset (available fell to 7 GB and swap
to 5 GB in the 30K arm; the box has swapped under similar pressure before).

## Warm-restore TTFT probe (2026-09-06 02:21, explains the 128-token anomaly)

`probe_warm_restore_ttft.py` + `warm_restore_probe_20260906T022156/`: four requests per prompt length
(cold, the SAME prompt again, a tail-modified variant, a second cold), `max_tokens=1`, temp 0, server
TTFT and `cached_tokens` from the usage block. Arms: the named preset (prefix caching ON) and the same
flags without `--enable-prefix-caching`. Median TTFT in ms, two repeats:

| prompt tokens | arm | cold | exact repeat | tail-modified (~95% cached) | second cold |
|---:|---|---:|---:|---:|---:|
| ~195 | prefix ON | 663 | **640** | 650 | 662 |
| ~625 | prefix ON | 1183 | **197** | 757 (one rep 334, one 1180) | 1169 |
| ~2340 | prefix ON | 4250 | **220** | **4215** | 4157 |
| ~195 | prefix OFF | 528 | 506 | 509 | 518 |
| ~625 | prefix OFF | 1023 | 979 | 992 | 1024 |
| ~2340 | prefix OFF | 4047 | 3964 | 4034 | 4004 |

Two defects, both visible in the `cached_tokens` column of the raw log:

1. **A short exact hit is reported but not honoured.** At ~195 tokens the repeat reports
   `cached=192/193` and still costs 640 ms — the same as its own cold run, and MORE than the
   prefix-OFF cold (506 ms). At ~2340 tokens the identical situation costs 220 ms. So below some
   length the restore path recomputes the prompt while still counting it as cached. That is exactly
   the operator's TUI row: a 128-token prompt at 97% cache with the WORST TTFT at every concurrency
   (729 / 1515 / 2194 ms) while a 512-token prompt at 100% cache read 209 / 404 / 651 ms.
2. **A partial hit is charged the full cold price.** The tail variant at ~2340 tokens reports
   `cached=2272` (94%) and costs 4215 ms, indistinguishable from cold; at ~625 tokens one repeat took
   the fast path (334 ms) and the other did not (1180 ms) on the same shape. Reuse is therefore
   all-or-nothing *and* nondeterministic near the boundary — a partial prefix does not save prefill,
   it only saves the lookup.

Prefix caching also costs ~20-30% on genuinely cold prompts (663 vs 528, 1183 vs 1023, 4250 vs 4047):
the snapshot/insert work on the ingest path. That is the price of the hit when the hit works, and it
is why defect 1 is worse than "no cache": short prompts pay the insert and never collect.

Suspects, in the order a fix should test them (UNVERIFIED): the block granularity of the radix match
(the log shows `cached` moving in 16/32-token steps, so a short prompt may match blocks the restore
then declines to use); the Marconi SSM snapshot policy declining to snapshot/restore below a token
threshold, forcing a full recompute of the GDN state while the KV blocks are reported as hits; and the
PLE row-cache re-stage, which is per-request host work the restore path cannot skip. Records:
`warm_restore_probe_20260906T022156/` (fingerprint, per-request rows, both arms).

## Cross-sequence batched mHC verify — NEGATIVE (2026-09-06, opt-in, not shipped)

The concurrency wall the operator's TUI sweep showed (aggregate flat ~30 tok/s from C=1 to C=4 while
TPOT grows with C) was attributed to the mHC K-row verify running once per sequence. Branch
`perf/batched-hc-verify` implements the obvious fix: `decode_verify_hc_multi` verifies n sequences x
k_i rows in ONE R=Σk_i-row forward, sharing only the row-invariant sweeps (GDN in/out projections in
≤8-row chunks, BA gates, gated RMS norm, hc_expand/hc_post, embeds, final norm, lm_head in ≤8-row
chunks, one argmax + one D2H) and keeping every acceptance-sensitive op per row (hc_pre at M=k_i, the
per-row MoE FFN, per-row PLE, R one-row attention decode bodies, per-sequence conv+GDN).

**Two defects had to be fixed before the arm could even be measured** — both of the class "a lever that
never engages reads as no gain":

1. `verify_hidden_stash` is allocated in `TransformerModel::new` only when the generic `proposer`
   field is filled. qwen4_exp installs its MTP head post-construction
   (`set_qwen4_exp_mtp_head(install_as_proposer=true)`), so the buffer stayed NULL and the batched path
   declined every tick with `reason=no MTP proposer (verify_hidden_stash null)` while its boot line
   said ARMED. First run (`ab_batched-hc-verify_20260906T042014`): `live=INERT, multi_lines=0`. Fixed
   by allocating the stash where the head is installed (the batched step's Phase 2 genuinely needs it:
   every accepted row's hidden must be stashed before any sequence's propose overwrites the shared
   buffer).
2. The first-tick gate line printed `arm=KILLED(ATLAS_NO_MTP_HC_BATCH_VERIFY)` whenever the path was
   off, including when nothing set that variable. It now distinguishes OFF(default) from KILLED.

**Measured** (`ab_batched-hc-verify_20260906T044450/`, binary `f938cdf84151fb9b74c4` at
`perf/batched-hc-verify`, counterbalanced serial / hcbatch / hcbatch / serial, fresh server per arm,
named preset, 2K salted prompts, 300 tokens, temp 0, effort low, C=1/2/4 x3 reps, kill switch the only
variable; every hcbatch boot proven live with `multi_lines>=1, declined=0, verify_errors=0`):

| aggregate decode tok/s | C=1 | C=2 | C=4 |
|---|---:|---:|---:|
| serial (per-sequence verify, today's default) | 26.7-27.6 | **23.7-24.3** | **21.8-22.4** |
| hcbatch (cross-sequence batched verify) | 26.7-27.5 | 16.3-18.9 (−25%) | 14.6-15.1 (−33%) |

C=1 is identical to three decimal places across all four boots (a single sequence never batches), so
the entire loss is inside the multi-sequence pass. **The hypothesis is refuted for this model**: the
flat C=1→C=4 aggregate is NOT the verify's weight sweeps. Sharing them cannot pay for what the mHC
path must still do per row — R one-row attention decode bodies, per-row router and PLE, per-sequence
conv+GDN and per-sequence metadata staging with a stream sync each — and batching adds staging on top.
A 2-stream probe reads 33 tok/s aggregate with the arm off vs 28 with it on, the same direction.

Shipped as OPT-IN (`ATLAS_MTP_HC_BATCH_VERIFY` presence arms it; `ATLAS_NO_MTP_HC_BATCH_VERIFY` still
wins) so the arm survives as a measurement tool. Where the concurrency wall actually lives is now an
open question: the per-row work inside the verify, not the sweeps. The next probe should price the R
one-row attention bodies and the per-sequence GDN/metadata staging directly (nsys of one C=4 step),
before anyone writes another batching lever.

## Marconi restore floor — the 128-token TTFT anomaly, FIXED for this preset (2026-09-06)

The warm-restore probe above showed a short exact prefix hit being reported as cached and then
recomputed. The cause is a deliberate floor: `mtp_carry::marconi_min_tokens()` declines a Marconi SSM
snapshot restore below **256 tokens**, set on 2026-07-28 from a DECODE measurement (restoring a
~99-token prefix cost -9.7% tok/s; the crossover sat between 99 and 219 matched tokens). Anchors below
the floor are declined by name in `prefix_lookup.rs`, so the KV blocks count as cached while the pass
is recomputed — which is what the operator's TUI row and my probe both saw.

**Measured** (`ab_marconi_floor/`, branch `perf/concurrent-prefill-ttft` binary `b4ea6fb127f07f48`,
named preset, one boot per arm, `ATLAS_MARCONI_MIN_TOKENS` the only variable, primed (cached) prompts,
128 output tokens, 3 reps per cell, arm proven live from the boot line `Marconi min-tokens floor: 64
tokens [env]`):

| primed prompt | floor | TTFT C=1 | TTFT C=4 (per stream) | aggregate incl. TTFT, C=4 |
|---|---:|---:|---|---:|
| ~168 tokens | 256 (compiled default) | 0.65-0.66 s | 0.65 / 1.35 / 2.01 / 2.65 | 25.1-26.0 |
| ~168 tokens | **64** | **0.21-0.22 s** | **0.22 / 0.49 / 0.70 / 0.91** | **28.0-28.2** |
| ~550 tokens | 256 | 0.21-0.22 s | 0.21 / 0.46 / 0.66 / 0.87 | 27.6-28.6 |
| ~550 tokens | 64 | 0.21-0.23 s | 0.21 / 0.46 / 0.66 / 0.88 | 27.2-28.6 |

Three things are true at once. The short-prompt TTFT improves **3x** at C=1 and **2.9x** on the last
stream at C=4. The 512-token cell is untouched, which is the control that proves the lever is scoped to
prompts below the floor. And the 2026-07-28 decode finding REPRODUCES: pure decode at C=1 on the short
prompt falls from 29.6 to ~27.5 tok/s (-7%) because the restore is now taken. So this is a TRADEOFF,
not a free win — it pays whenever a request's output is short relative to its TTFT, which is exactly
agentic and concurrent traffic, and the aggregate INCLUDING TTFT wins at every concurrency here.
C=1 greedy samples are byte-identical across arms.

Shipped as a PRESET pin (`ATLAS_MARCONI_MIN_TOKENS = "64"` in the `qwen3.8-flash-next-exl3` preset),
NOT as a compiled-default change: the 256 default is shared with models whose traffic was measured
differently. Set the variable to 256 to restore it. Floor 0 (always restore) is UNMEASURED — the E1
arm `B0` exists for it.

## The partial-hit reuse cliff, MEASURED (2026-09-06 19:33) — it is the chunk boundary

`probe_partial_hit_cliff.py`, records in `cliff_20260906T193328/`: prime a prompt, then send variants
whose tail is regenerated from a given kept-prefix fraction, `max_tokens=1`, temp 0, server-attested
TTFT and `cached_tokens`, 2 repeats. Named preset on the shipped binary (Marconi floor 64, MoE row cap
1024, reconstruct tier 512). Median TTFT in ms:

| kept prefix | ~2350-token prompt | ~8960-token prompt |
|---|---:|---:|
| cold (0%) | 4249 | 18333 |
| 25% | 4220 | 18342 |
| 50% | 4131 | 18243 |
| 75% | 4152 | 18352 |
| 90% | 4147 | 18123 |
| **95%** | 4136 | **1802** |
| **99%** | 4146 | **1793** |
| exact (100%) | **219** | **242** |

**The cliff is the prefill CHUNK boundary, exactly as `prefill_b/save_checkpoint.rs` says it must be.**
At `--max-prefill-tokens 8192` the ~8960-token prompt is two chunks, so snapshots exist at 8192 and in
the last blocks near 8960. Keeping 95% leaves the divergence above 8192, the 8192 anchor is usable, and
the cost falls to a ~770-token replay: 18.3 s → 1.8 s, a **10x**. Keeping 90% puts the divergence at
~8064, BELOW the only chunk anchor, and the price is a full cold prefill even though the server reports
8048 of 8960 tokens cached. The ~2350-token prompt is a SINGLE chunk, so it has no interior anchor at
all: every fraction from 25% to 99% pays full cold price, and only a byte-exact repeat is fast.

So on this engine prefix reuse is not proportional to how much of the prompt matched — it is
**all-or-nothing at chunk granularity**. An agentic turn that appends to a conversation lands in the
final chunk and reuses; edit a tool result, a system prompt, or anything mid-context and the entire
prompt is recomputed. At the measured ~480 tok/s a 30K context costs ~62 s of that.

The fix is the one the source file already names as untested: make the checkpoint interval GENERATE
boundaries instead of only filtering chunk ends (`save_checkpoint.rs`: "Making the interval a real
generator (splitting chunks at interval boundaries) is a behaviour change with a prefill-throughput
cost and is deliberately NOT made here; it needs its own measured A/B"). The cost is more snapshots
competing for the `--ssm-cache-slots` pool, which is the pressure the operator has separately observed
dropping checkpoints; the benefit, from the 8960-token row, is turning an 18.3 s recompute into a
replay bounded by the interval. UNMEASURED until that A/B runs.

## PLE n-gram fault-in: the worker cap was 4x below the drive's knee (measured 2026-09-06)

A live prefill logged `PLE gather: 127488 ids, 95743 hits / 31745 misses, resolve 288342us`. The
misses are O_DIRECT 4 KiB positional reads of the n-gram table, already fanned out to scoped threads
by `spark-storage/src/ngram_cache_fault.rs` — capped at `MAX_WORKERS = 16` with the comment "NVMe queue
depth benefits flatten out well below this". That comment was an assumption, and it is wrong on this
drive.

`ple_fault_bench.c` replicates `run_one` exactly (O_DIRECT, 4096-byte blocks, random offsets, one
thread per worker, its own aligned bounce buffer) against the real 36.4 GiB pinned n-gram file,
32768 reads per sweep — the miss count of a real 8K prefill. Three repeats, wall ms:

| workers | 16 | 24 | 32 | 48 | **64** | 96 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| run 1 | 332 | 268 | 220 | 171 | **146** | 139 | 138 |
| run 2 | 334 | 262 | 222 | 170 | **147** | 138 | 139 |
| run 3 | 335 | 269 | 224 | 172 | **148** | 137 | 137 |
| IOPS | 98K | 122K | 148K | 191K | **223K** | 237K | 237K |

The knee is **64** (2.3x over the shipped 16) and the curve is flat past 96 — the drive saturates
around 237K IOPS. Per-read latency rises from 77 µs at one worker to 286 µs at 64, which is the
queue filling, not the drive degrading.

Change: `MAX_WORKERS` 16 → 64, plus a `JOBS_PER_WORKER = 8` floor so a small gather does not spawn a
thread per read (31745 misses → 64 workers, 100 → 13, 32 → 4; thread spawn is ~20 µs, comparable to a
read). `ATLAS_PLE_FAULT_WORKERS=<n>` overrides the cap for in-situ re-measurement, and
`ATLAS_PLE_SERIAL_FAULT=1` still restores the pre-parallel arm.

**Expected but UNMEASURED end to end**: the 288 ms resolve above should fall to ~130 ms, i.e. ~160 ms
off a cold prefill's TTFT. That is a bench-derived hypothesis — the in-engine A/B (same prompt, the
env knob the only variable) has not been run. Note also that `resolve` holds the PLE table mutex, so
this time is serialized across concurrent streams and the saving compounds at concurrency.

## Concurrent prefill: the gate opens, the forward is still per-stream (2026-09-06 late)

Chasing "concurrent prefills serialize" produced three wrong suspects before the right one. Recorded
so nobody re-walks it.

**Not the admission window.** One already exists: `mod_helpers.rs::drain_pending_requests` waits up to
`ATLAS_PREFILL_CODISPATCH_WINDOW_MS` (default 100) on the idle condvar, growth-aware — it keeps
collecting while the queue grows and stops after one quiet `settle`, precisely so a C=4 burst is not
split. **Not admission accounting** either: `seq_commitment_blocks` reserves `prompt + min(max_tokens,
watermark)`, so four 8K prompts with 300-token budgets reserve ~2K blocks of an ~11K-block pool.

**Not the co-dispatch gate.** A one-shot attribution line was added to `phase_start_prefills.rs`
(same pattern as the DFlash batched-verify gate), and on a fresh idle server with four concurrent
~2150-token prompts it reads:

```
prefill co-dispatch gate (first tick with requests) new_reqs=4 active=0 prefilling=0
    chunked=true is_ep=false vision=false want_codispatch=true
```

followed by `Batched prefill[0/4] … [3/4]`. All four streams then return TTFT within 17 ms of each
other — synchronized, no staircase. So co-dispatch works.

**It is the forward.** `phase_continue_prefills.rs` says it outright: *"Both call the default trait
impl today (per-stream loops); Q12 Phase 2/3 replace with kernel-level batched dispatch."* The four
streams share a scheduler step, then are forwarded one at a time.

Measured, fresh idle server, 4 x ~2150-token prompts, `ATLAS_PREFILL_CODISPATCH=1`:

| binary | per-stream TTFT | vs one stream alone (4249 ms, cliff probe) |
|---|---:|---:|
| `wip/exl3-research` (d09ab8477) | 14364 / 14369 / 14358 / 14375 ms | 3.4x |
| `perf/qsa-concurrent-prefill` (503dad960) | 14304 / 14310 / 14315 / 14320 ms | 3.4x |

**The QSA batching branch measures as no change** (0.4%, inside noise). It is necessary but not
sufficient: it removes `ensure!(batched_meta.is_none())` so QSA stops refusing the batched path, but
the path it is admitted to still loops per stream. Batching four streams costs 14.3 s against 17 s of
serial prefill — a 1.18x from sharing scheduler overhead, not from sharing weight sweeps.

### Kernel-level fused prefill EXISTS, engages, and measures 0.4% — prefill is saturated

The obvious next step was "concatenate the streams' rows into ONE forward so the MoE and dense GEMMs
see 4x the rows", on the theory that this engine is weakest at small M. **It is already built**
(`prefill_b/batch_kernel.rs`, `prefill_batch_chunk_kernel_batched`, the varlen wave planner), and two
preconditions had merely never been met at once. `check_kernel_batched_eligible` refuses when chunk
lengths differ across streams unless `ATLAS_PREFILL_VARLEN=1`, and when Σ effective tokens exceeds the
token arena — a 4 x ~2150-token burst is 8600 > 8192 and unequal, so it tripped both.

With `ATLAS_PREFILL_CODISPATCH=1 ATLAS_PREFILL_VARLEN=1` and 4 x ~1700-token prompts (Σ 6825 < 8192),
on `perf/qsa-concurrent-prefill`, the whole chain engages:

```
Varlen prefill waves: 4 streams -> 1 wave(s), M per wave [6825] (cap 8192)
QSA batched prefill select: ARM=batched streams=4 selective=0 inert_bound=2051
Q12 kernel-batched prefill dispatched (fused large-M) n=4 total_tokens=6825
```

One-variable A/B, `ATLAS_Q12_BATCHED` the only difference (prefix caching OFF in BOTH arms, because
the fused path cannot be reached with it on — see below):

| arm | tokens | worst TTFT | prefill tok/s |
|---|---:|---:|---:|
| fused large-M | 6825 | 10647 ms | **641** |
| per-stream (`ATLAS_Q12_BATCHED=0`) | 6820 | 10691 ms | **638** |
| solo reference, 1 stream | 1714 | 2731 ms | 628 |

**0.4%.** And the solo rate is the same 628 tok/s, so four streams fused into a 6825-row forward run at
the same rate as one 1714-row forward. Prefill is throughput-SATURATED at ~630 tok/s: the TTFT
staircase under concurrency is queueing at a fixed service rate, not waste that batching can recover.
The small-M argument does not apply — it is about DECODE at 1 row and MoE experts receiving few rows;
at 1700 rows the dense GEMMs, the 1024-row MoE tier and the 512-row reconstruct tier are all already
in their efficient regime, so there was nothing left to amortize.

Two constraints on the claim. The prompts sit below the 2051-token QSA inert bound (`selective=0`), so
the fused dispatch was exercised but QSA's selective path was not. And prefix caching had to be OFF in
both arms: `batched_reserve_hybrid_ssm_ok` admits a hybrid-SSM batch only when EVERY reservation is
cold (`matched_tokens == 0`), and the shared chat-template preamble matches ~32 tokens, so under the
house rule (prefix caching always on) the fused path is unreachable in production. Making it
warm-match-capable is therefore work that would unlock the 0.4% above.

**Consequence:** `perf/qsa-concurrent-prefill` is correct and now proven to engage (`ARM=batched
streams=4`), but it is not a win — it removes a refusal in front of a path that does not pay. Left
unmerged. The remaining prefill lever is making each token cheaper (the kernel mix: MoE tiers, QSA
attention, dense trellis), not grouping more tokens per forward.

## Cross-stream serialization levers: checkpoint cadence REFUTED, PLE lock UNPROVEN (2026-09-06)

`ab_concurrency_serialization.sh` + `probe_conc_serialization.py`, records in
`ab_conc_serial_20260906T220433/`. Named preset, prefix caching ON in every arm (house rule), one
variable per arm, 2 repeats.

| arm | decode agg C=1 | decode agg C=4 | scale | `decode-ckpt SAVE` lines | warm4 TTFT |
|---|---:|---:|---:|---:|---:|
| base (64 fault workers, ckpt every 4 blocks) | 31.5 | 28.5 | 0.90x | 13 | 387 ms |
| `ATLAS_PLE_FAULT_WORKERS=16` | 31.9 | 32.6 | 1.02x | 10 | 326 ms |
| `ATLAS_DECODE_CKPT_BLOCKS=16` | 30.3 | 29.8 | 0.98x | 3 | 406 ms |

**Checkpoint cadence: REFUTED.** Raising the interval did exactly what it claims — decode checkpoint
saves fell 13 → 3 — and bought nothing in decode aggregate, while warm TTFT drifted slightly worse.
The decode-time snapshot stalls measured earlier (~20-25% of decode at C=1 on the 4-slot preset) are
evidently not dominated by save FREQUENCY. Do not ship a cadence change on this evidence.

**PLE fault workers: UNPROVEN, and this workload cannot decide it.** The gathers here missed 8-192
rows and resolved in 1-4 ms; the operator's production line was `127488 ids, 95743 hits / 31745
misses, resolve 288342us`. Synthetic random-word prompts reuse the same n-gram rows, so the cache runs
at ~98% hit rate and the worker count is irrelevant (the 16-worker arm even reads nominally *better*,
which is noise at this scale). The bench-measured 2.3x (`ple_fault_bench.c`) therefore stands
un-contradicted but also un-confirmed in-engine.

**Harness defect, recorded so the number is not reused:** `cold_ratio` came out exactly 1.00 in all
three arms because the server's `time_to_first_token_ms` is measured from the start of a request's
PROCESSING, not from its arrival. With co-dispatch off, four queued streams each report their own
~6.6 s prefill and the queue wait lands in wall time instead. The cold cells therefore measure
per-request service time, not end-to-end concurrency; a concurrency metric must use wall-clock from
submission (the probe records `wall_s` but did not report it per cell).

**Consequence for the PLE-lock refactor** (`resolve` holds the table mutex across both its phases —
bookkeeping, then `fault_all`): phase 1 already pins every slot it assigns, so the I/O provably does
not need the lock, and the refactor is contained. But at 1-4 ms of held lock it would be optimizing
noise. The prerequisite is a workload that reproduces the miss-heavy gather (~127K ids implies ~30K
tokens of genuinely diverse text, not synthetic filler).

## Queueing: SLAI, the admission window, and a cold-start artifact I nearly shipped (2026-09-06)

The hypothesis was that concurrent queueing needs the SLAI policy plus a ~100 ms admission delay.
Measured with `ab_queueing_policy.sh` + `probe_queueing.py` on a deliberately heterogeneous burst —
one ~8000-token prompt submitted FIRST, then three ~600s, all timings CLIENT-SIDE from submission
(the server's `time_to_first_token_ms` starts at processing and hides queue wait entirely).

**First pass, 2 repeats per arm, arms in order fifo → slai → slai_cd → slai_win → fifo_win:**

| arm | short med TTFT | long TTFT | burst wall |
|---|---:|---:|---:|
| fifo | 12102 | 20508 | 32.35 s |
| slai | 12172 | 24066 | 32.81 s |
| slai + co-dispatch | 19488 | 29049 | 31.75 s |
| fifo + 100 ms window | 2670 | 22321 | 31.92 s |
| slai + 100 ms window | 2901 | 30791 | 33.69 s |

That reads as a 4.5x for the admission window. **It is an artifact.** Re-measured properly — same
binary, `ATLAS_PREFILL_ADMISSION_WINDOW_MS` the only variable, FIVE repeats per arm
(`window_reps_20260906T225903/`, `window_reps_20260906T230244/`):

| arm | short med TTFT | per-rep | long | burst wall |
|---|---:|---|---:|---:|
| window 100 ms | **2607** | 2679/2594/2613/2579/2607 | 22145 | 31.77 s |
| window off (`=0`) | **2670** | 2835/2631/2678/2664/2670 | 22097 | 31.39 s |

Identical. The 12102 came from the FIRST burst against a freshly booted server (cold PLE row cache,
cold page cache) dragging a 2-repeat median — the no-window arm's own rep1 was already 2675 ms, and
the window arms simply ran later when the box was warm. Two repeats plus arm-ordering, exactly what
the n>=3 and counterbalancing rules exist to prevent. **No preset change was justified and none was
made.**

What DOES survive, both from warm-vs-warm arms:

* **Co-dispatch batching is worse for mixed traffic.** `slai_cd` made every request uniform at
  19.4-19.6 s instead of 1.3 / 2.7 / 4.0 / 20.5 — nobody gets a token until the whole cohort's prefill
  completes. Fairness, not latency.
* **SLAI does not help and hurts the tail.** Warm arms: `fifo_win` short 2670 / long 22321 vs
  `slai_win` short 2901 / long 30791. Shortest-pending-first has nothing to gain when prefill is
  saturated, and it defers the long request substantially.
* **Burst wall is invariant across every arm** (31.4-33.7 s). Scheduling redistributes; it cannot
  create throughput. Consistent with the fused-prefill result above.

The one shipped change is a code decoupling, not a default:
`ATLAS_PREFILL_ADMISSION_WINDOW_MS=<ms>` now opens the admission window WITHOUT arming the batched
co-dispatch. Previously both lived behind `ATLAS_PREFILL_CODISPATCH`, so a scheduling policy could
never see an ordered queue — either no window (nothing pending to choose between) or a window whose
cohort was immediately fused (ordering moot). The knob now exists; on this workload it measures flat.

## Attacking the reuse cliff: interior SSM anchors — BLOCKED on two independent gates (2026-09-07)

The plan was to give a chunk interior anchors so a divergence anywhere can re-anchor near it, using the
existing `midchunk_capture.rs` mechanism (capture GDN recurrent + conv state mid-chunk by splitting the
two cheap per-token kernels, no extra pass) generalized from its two tail-hugging points to N.
Branch `perf/ssm-interior-anchors` (`16dd1b6ae`), knob `ATLAS_SSM_PREFILL_ANCHOR_TOKENS=<n>`, default
OFF. Records: `ab_ssm_interior_anchors/20260907T001432/`.

**Discovery that reframes the cliff: the mid-chunk mechanism was DEAD on GB10.**
`prepare_midchunk_capture` opened with `if !cfg!(atlas_scale) { return None; }` — the h_state capture
existed only in the gfx1151 split4 arm, while the NVIDIA FLA/WY ladder allocated destination pointers
and never wrote them. So the tail and sibling anchors this file previously described as "default-on"
have never existed on this hardware, and chunk-end checkpoints were genuinely the ONLY prefill anchors.
That is why the cliff is exactly at the chunk boundary.

**Arms were live.** `off` verified inert (no ACTIVE line, 0 anchors). `anchor1024` registered 44
interior anchors, `anchor512` registered 55, both logging `ssm interior anchors ACTIVE`.

**Gate 1 — correctness: FAILED, blocking.** The A/B's P4 asserts a cold pass is unchanged by the
capture splits (the segmentation is supposed to be byte-identical — same algebra as the >4096
sub-chunk loop):

```
anchor1024 COLD != control COLD  <- the SPLIT changed a cold pass. BLOCK.
anchor512  COLD != control COLD  <- BLOCK.
```

Splitting the recurrence at a capture point regroups the FLA `CHUNK = 64` ladder and drifts the bf16
W/U/uc/S_c intermediates, so the NVIDIA arm's capture is NOT free of numerics. (`cold vs
warm-restore: DIFFER` appears in every arm INCLUDING the control — that is the pre-existing
warm-restore class, not this change.)

**Gate 2 — the anchors are unusable on this model anyway.** TTFT did not move at any divergence
fraction: the 25-90% cells stayed at ~18.2 s (L=8000) and ~4.15 s (L=2048), identical to the control,
despite 44-55 anchors being registered per arm. Cause: interior anchors carry no aux blob
(`collect_aux_states` reads live state at the PASS END), and `requires_aux_state()` is
`layers.iter().any(|l| l.has_aux_state())` where the attention layer returns `self.qsa.is_some()` —
TRUE on qwen3.8-flash-next. So the restore path declines every interior anchor. They were written and
never read. `anchor512` also regressed the EXACT-repeat cell badly (L=8000 exact 1018 ms vs the
control's 218; L=2048 exact 2114 vs 232), i.e. the extra slots evict the anchors that do work.

**Verdict: not shippable, left unmerged, default OFF so nothing changed.** To actually close the cliff
on THIS model both must be solved: (1) a capture that is byte-exact on the FLA ladder — most likely
capture only on the `CHUNK = 64` grid AND prove it, or capture from a boundary the ladder already
lands on rather than forcing one; and (2) snapshotting the QSA/PLE aux state at an interior boundary,
not just the SSM state, since this model refuses aux-less anchors at restore. Item 2 is the larger
piece and it is a prerequisite — without it, item 1 buys nothing here.

## Rollback: rewind QSA/PLE aux state with the SSM state, or decline (2026-09-07)

A latent 500 on this model class, found by comparison with a downstream fork. The content-loop
watchdog's rollback (`scheduler/rollback.rs`) restored the SSM recurrent state from the decode ring
and lowered `seq_len`, but nothing rewound the auxiliary per-sequence state that hybrid models also
carry — the QSA indexer cursor and the PLE n-gram history. The very next decode then failed

```
QSA: decode at pos 2864 but 2932 tokens ingested — the indexer cache lost sync
```

and the request died with a 500. The only aux rewind in the tree was the speculative-verify commit
hook; the watchdog path never touched aux state at all. Confirmed present here before the fix:
`RollbackFallback` had `NoSsmSnapshot` but no aux variant, and neither `rollback.rs` nor
`ssm_decode_ring.rs` mentioned aux — while `requires_aux_state()` is
`layers.iter().any(|l| l.has_aux_state())` and the attention layer returns `self.qsa.is_some()`,
which is TRUE on qwen3.8-flash-next.

**Fix — mirror the SSM treatment with the AUDITED SNAPSHOT PATH, not a new arithmetic rewind.**
`model/decode_aux_ring.rs` holds host-side blobs from the same `collect_aux_states` /
`apply_aux_states` (`snapshot_aux` / `restore_aux`) hooks Marconi prefix caching already uses, so a
restore is byte-exact by construction and cannot get cursor arithmetic subtly wrong. Entries are keyed
by the SAME `(seq.slot_idx, ring_slot)` pair as the SSM ring, so the halves cannot drift:

* `snapshot_boundary_if_ssm` saves BOTH or drops the ring entry — a boundary with only its SSM half
  would let the rollback select a slot it must then decline on, turning a recoverable loop into a hard
  stop.
* `rollback_to_boundary` restores the aux companion right after the SSM state and BEFORE any buffer
  truncation (same reason: a failure leaves the sequence untouched for the caller's hard stop), and
  declines with the new `RollbackFallback::NoAuxSnapshot` when it is missing or fails.

Declining is the honest outcome, not a regression: truncation is stream-safe and is what the
pre-existing `NoSsmSnapshot` arm already does.

Inert by construction for models with no aux-carrying layer (`requires_aux_state()` false → every aux
path returns `Ok(())` and no ring entry is made), so pure-attention and plain-SSM models are byte-
identical to before.

Gates: `cargo test -p spark-model --lib` 800 passed, `-p spark-server --bin spark rollback` 21 passed,
new `decode_aux_ring` unit test (keying, precise forget, non-consuming get so a declined rollback may
retry), clippy `-D warnings` clean on both crates, rustfmt clean, release build. **UNMEASURED on the
GPU:** the fix's own trigger needs the watchdog to fire on a degenerate generation, which this session
did not reproduce — what is verified is that the code compiles, is inert for non-aux models, and that
the decline path is taken when the companion is absent.

## EP=2 on two GB10s: works, and MTP works EXCEPT across a warm restore (2026-09-07)

Two-node EP=2 (gx10-9959 head :8890 + dgx-00 worker, RDMA over enp1s0f1np1), EXL3 4.05bpw, preset-parity
env, prefix caching ON, `--ep-size 2`. Binary sha256 identical on both nodes (a mismatch here desyncs
ranks in ways that read as a model bug). Scripts: `~/run_exl3_ep2_mtp.sh` on both hosts.

**Three rank-asymmetries had to be fixed to get MTP to load at all**, each hidden behind the last:

1. The draft MoE was sharded by `is_local_expert` while the upload never sharded `mtp.*` → replicate
   with `force_all_experts` (43f82882d).
2. The native-EXL3 expert loader shards independently via `local_expert_range()` — same flag needed
   there, which the error message alone would not have revealed.
3. `factory/build.rs` gated the WHOLE MTP module load on `config.ep_rank == 0`, citing the
   precondition (1) had just removed. Left up, rank 1 warned "no MTP weights were loaded", set
   `has_mtp=false`, allocated `num_intermediates=0`, and died three layers away in the first mHC
   verify with "SSM MTP intermediate buffers not allocated (h_state_intermediates.len()=0)"
   (b0b918064).

**Single-turn generation: WORKS, at single-node parity.** 200-token generation, correct Python output:

| | EP=2 (2 nodes) | single node |
|---|---:|---:|
| decode | **30.3 tok/s** | 30.6-30.8 |
| acceptance `mean_na` | **1.653** | 1.47-1.49 |
| tok/step | 2.653 | ~2.5 |
| boot | ~62 s | ~50 s |
| experts/GPU | **256** | 512 |

`mtp=1.00 serial=0.00 p1=0.893` — every step took the speculative path. Acceptance at or ABOVE the
single-rank baseline is the evidence that drafts are not diverging across ranks: divergence collapses
acceptance, it does not raise it.

**Agentic, one iteration, reasoning_effort low + preserve_thinking — BISECTED:**

| arm | result |
|---|---|
| EP=2, MTP **off** | **PASS** — 1/1 webserver_ok, 1/1 followed_directions, 137 s wall, 15.1 s/turn |
| EP=2, MTP **on** | worker dies on the first WARM turn |

```
EP worker error: QSA: decode at pos 3739 but 3741 tokens ingested
  — the indexer cache lost sync (prefix-cache skip or a rewound sequence)
```

The gap is exactly 2 tokens = the draft width. It is NOT a rank asymmetry in the drafter: head and
worker agree exactly on `MTP drafter coverage` at every turn (25/25, 3512/3512, and `captured=0` on
the third — the first turn served from the prefix cache). So EP=2 itself is sound for agentic work,
and the defect is specifically **EP + speculation + a prefix-cache warm restore**, where the worker's
QSA ingest count runs ahead of the committed decode position by the draft width.

This is the same defect CLASS as the rollback fix in `fix(rollback): rewind QSA/PLE aux state with the
SSM state` (dba869c04) — QSA aux state not rewound to the committed position — but on the EP verify
path rather than the watchdog path, so that fix does not cover it.

**FIXED (2026-09-07, same session).** The worker's verify arms in `impl_a2.rs`
(`0xFFFFFFF2/3/4`) received `num_accepted`, trimmed the proposer state, rewound `seq_len`/`tokens` and
rolled back the SSM — but never landed the aux carries. The head does that inside
`commit_accepted_prefix`, whose own comment names the class: *"the OTHER TWO per-row carries — PLE's
rolling conv/history window and QSA's ingested/pooled marks — are still `k - num_accepted` rows
ahead"*. So `commit_verify_aux_rows` (visibility widened to `pub(in crate::model)`) is now called from
each worker arm.

ORDERING MATTERS and cost one run: `commit_verify_aux_rows` asserts
`seq_len == base + num_accepted`, so it must be called AFTER the arm rewinds `seq_len`, not before.
Calling it first fails with `batched mHC commit: 1/3 rows from 3513, but seq_len=3516`.

**Agentic under EP=2 with MTP: PASS.** 1/1 webserver_ok, 1/1 followed_directions, 15.9 s/turn, 112 s
wall, reasoning_effort low + preserve_thinking. Per-turn MTP telemetry stayed healthy across the run:

```
127 tok  24.7 tok/s  mtp=1.00  p1=0.638  mean_na=1.172  tok_step=2.172
161 tok  30.6 tok/s  mtp=1.00  p1=0.917  mean_na=1.667  tok_step=2.667
147 tok  26.2 tok/s  mtp=0.78  p1=0.768  mean_na=1.321  tok_step=2.321
475 tok  25.2 tok/s  mtp=1.00  p1=0.742  mean_na=1.225  tok_step=2.225
```

**PREFILL AND CONCURRENCY under EP=2** (2026-09-07, 4 sequence slots, 32K ctx, MTP on, prefix caching
on, `EP v2 active: honoring max_batch_size=4`, 2 reps per cell):

| metric | EP=2 (2 GB10s) | single node | delta |
|---|---:|---:|---:|
| prefill 8K | **542 tok/s** | 478 | **+13%** |
| prefill 11K | **561 tok/s** | 486 | **+15%** |
| decode aggregate C=1 | 26.4-27.2 | 26.7-27.6 | parity |
| decode aggregate C=2 | 22.2-23.7 | 23.7-24.3 | parity |
| decode aggregate C=4 | 22.3-22.8 | 21.8-22.4 | parity |

**Prefill is the win, and the mechanism is the obvious one:** prefill on this model is dominated by
MoE weight traffic, and EP=2 halves the experts each rank must read (256 instead of 512). Decode is
unchanged because it is bound by the per-sequence mHC verify, not by weight bandwidth — the same
reason batching the verify across sequences was a NEGATIVE result above.

Speculation stays healthy under concurrency: at C=4 every stream logged `mtp=1.00` with `mean_na`
0.88-1.51 (acceptance falls at width, as single-node does).

**Status:** `ATLAS_EP_MTP` remains the opt-in — one agentic iteration and 2-rep sweeps are a pass, not
a certification — but EP=2 with MTP and prefix caching now completes a multi-turn agentic run AND is
faster at prefill than a single node. Set `ATLAS_EP_MTP=1` to use it. EP=2 without MTP also passes
(137 s wall, 15.1 s/turn) and needs no flag.

Harnesses: `measure_prefill.py` / `measure_concurrency.py` gained `--host` so they can target the EP
head from another node; `~/run_exl3_ep2_mtp.sh <rank>` on both hosts carries the fabric + preset-parity
env.

## Host D2H drains are ~free on GB10: the batched aux collect measures +1.2% (2026-09-07)

A four-dimension read-only review (locks, host-sync, serialization, memory) converged on one
mechanism: `copy_d2h` and `copy_d2h_on_stream` both `cuStreamSynchronize` INSIDE the call
(`cuda_backend/gpu_copy.rs`), and **no D2H destination anywhere in the engine is pinned** — the
tree's page-locked memory (`PinnedMetaStaging`, `alloc_host_pinned`) is H2D-only. The largest
instance was the Marconi aux collect (`trait_impl/mod.rs` `collect_aux_states`), which ran one
draining, fresh-`Vec` D2H per aux-carrying layer, with the QSA half O(context)
(`ingested * head_dim * 2`), on every chunk-boundary snapshot save and every
`ATLAS_DECODE_CKPT_BLOCKS` (4) blocks = 64 decode tokens.

**Measured: +1.1–1.2%, which is 10x smaller than the arithmetic predicted.**

Counterbalanced B A A B, one binary, one variable (`ATLAS_AUX_COLLECT_BATCHED`), 3 reps/arm,
pinned harness salt so both arms see byte-identical prompts. Arm liveness proved per arm from the
server's own log line (`13 batched layer(s), 0 legacy, 24944844 B` vs `0 batched, 48 legacy`).

| metric | batched (b1, b2) | legacy (a1, a2) | delta |
|---|---:|---:|---:|
| prefill 8K tok/s | 494, 496 | 492, 487 | +1.2% |
| prefill 11K tok/s | 480, 488 | 479, 478 | +1.1% |
| decode rep0 tok/s | 25.41, 26.11 | 25.50, 25.30 | +1.4% |
| decode rep1 tok/s | 26.35, 26.58 | 26.18, 26.00 | +1.5% |
| decode rep2 tok/s | 28.03, 28.40 | 28.15, 27.72 | +1.0% |

Same sign in all five paired comparisons, so the direction is probably real, but the size is small
against the within-arm rep spread (25.3 -> 28.4 tok/s) and this is a single fingerprint. Claim
language: *observed +1.2% under this fingerprint*, not a validated headline.

### Why the projection was wrong — GB10 is unified memory

The projection (~15%) came from the `GpuBackend::copy_d2h_async` doc's measured shape: the SSM
spill moved 66.8 MB as 60 blocking `copy_d2h` in ~400 ms (~165 MB/s) versus ~28 ms for the async
form plus one `synchronize`. **That 165 MB/s is not transfer bandwidth — it is mostly the host
waiting for already-queued GPU work, attributed to the copy call.** On GB10 host and device are the
same LPDDR5X, so a D2H is not a bus transfer at all, and "pageable vs pinned" does not mean what it
means on a discrete GPU. When the queue is shallow a drain costs almost nothing.

`ssm_spill_staging.rs` (now `pinned_host_staging.rs`) already said as much and was not taken
seriously enough: *"whether pinning helps at all on a UMA box (GB10: host and device are the same
LPDDR5X) is exactly the question a GPU run has to answer, not one to assume."* It has now been
answered: **~1% per 13 eliminated drains plus 25 MB of eliminated pageable allocation churn.**

### What this rules out, and what it does not

Ruled out as levers on this box, by direct implication (each is FEWER drains than the 13 measured
here, on the same hardware):

- the per-draft argmax D2H in the qwen4_exp draft chain (`qwen4_exp_mtp.rs:476`) — one drain per
  step at `--num-drafts 2`, on the speculative draft path;
- the MoE prefill `expert_offsets` drain (`exl3_matmul/moe_prefill.rs:416`) — prefill measured flat
  across both arms here.

NOT ruled out, because the defect is allocation and residency rather than drain count:

- KV spill save/restore (`sequence/state_io.rs:40`) collects the ENTIRE KV image into host `Vec`s
  before writing a byte — ~3.1 GB at 32K, larger than the whole 3 GB `--swap-space-gb` budget — via
  ~196,608 fresh pageable allocations at 32K. A bounded pinned window fixes the residency; the drain
  count is now known to be the lesser half.
- The Marconi aux blobs are host RAM the preflight does not budget (`preflight.rs:187` reserves only
  the device half): 3072 B per token of context per snapshot slot, ~25.8 GB at 128K x 64 slots.

Records: `aux_ab/` (per-arm serve logs, prefill and decode logs, liveness lines) from
`ab_aux_collect.sh` + `run_exl3_single.sh` (the shipped preset spelled out so one variable can move).

## Agentic gate at 6c4285e1a: five arms, all PASS (2026-09-07)

Post-change validation of the four commits in this session (batched aux collect,
batched verify-tail argmax, EXL3 smem-memo teardown, KV-spill windowing) on the
shipped preset, 32K x 4 seqs, prefix caching on, preserve-thinking on both sides
(`ATLAS_AGENTIC_PRESERVE_THINKING=1` client, `preserve_thinking:true` +
`reasoning_effort:low` in the serve default chat kwargs).

| arm | sampling | loop guards | verdict | per-iter wall s | turns | Swall | s/turn | decode tok/s |
|---|---|---|---|---|---|---|---|---|
| A | greedy | on | Pass 3/3 | 112/82/195 | 7/7/13 | 389 s | 14.03 | 21.9 |
| B | greedy | on | Pass 3/3 | 117/168/456 | 9/12/18 | 740 s | 18.91 | 16.3 |
| C | model-card | on | Pass 3/3 | 124/118/146 | 9/9/11 | 387 s | 13.26 | 21.9 |
| D | model-card | PARTIAL off | Pass 3/3 | 189/265/96 | 14/18/8 | 550 s | 13.67 | 21.5 |
| E | model-card | ALL off | Pass 3/3 | 113/84/98 | 10/6/7 | 294 s | 12.68 | 21.8 |

Every arm: 3/3 `webserver_ok`, 3/3 `followed_directions`, 6/6 steps, zero
desyncs / CUDA errors / panics / `NoAuxSnapshot` declines, all requests finishing
`(stop)`.

### Greedy is the DEFAULT, and it reproduces — the harness does not

`agent.rs:87` pins `TEMPERATURE = 0.0` and `SEED = 0` into every request unless
`ATLAS_AGENTIC_SAMPLING=model-card` removes them. Arms A and B are therefore the
same greedy configuration run twice, and they differ 1.9x in wall (389 s vs
740 s). That is NOT engine nondeterminism:

- `run-00` is byte-identical through turn 3 — same reasoning, same tool calls —
  and the first difference is INSIDE a tool result: cargo printed
  `Adding matchit v0.8.4 (available: v0.8.6)` in B and not in A.
- `run-01`/`run-02` differ only in the tool-call ID counter (`call_...0007` vs
  `call_...0009`), which is monotonic across iterations, so B's longer first
  iteration shifts it.

**Consequence: agentic wall-clock is not a usable perf metric.** One line of
cargo output moved it 1.9x on identical config. Turn count is the dominant term
and it is trajectory noise. Use `measure_prefill.py` / `measure_concurrency.py`
for throughput; use this gate for correctness. Note decode tok/s is FLAT at
21.5-21.9 across C/D/E — the axis that would move if any of this were expensive.

### Arm liveness, proved from the server not the env

- greedy: `temp=Some(0.0)` on all 34 requests; model-card: `temp=None` on 29/32
  (the 3 pinned ones are the harness's own grading calls).
- E: boot WARN `ALL auto-watchdogs DISABLED via ATLAS_DISABLE_WATCHDOGS=1
  (content-loop, inter-tool prose, F2 confidence early-stop, mid-word </think>
  defer, thinking-loop)`.

### Loop-guard knobs are THREE independent mechanisms, and one flag is a trap

- `ATLAS_SIMHASH_LOOP=0` — F4 SimHash semantic-loop guard (`handle_token.rs`).
  Fired once in C (`paraphrased sentence repeat`, request completed normally).
- `ATLAS_LOOP_NO_SUPPRESS=1` — soft `<tool_call>` bias decay (`loop_detect.rs`).
  One HINT in C.
- `ATLAS_DISABLE_WATCHDOGS=1` — ALL auto-watchdogs at once. **This is the knob to
  use**; it does NOT cover the two above, so a complete arm sets all three.
- **`--content-loop-watchdog false` is INERT on this model**: `[behavior]` never
  sets `enable_loop_watchdog`, so it is already off and the flag changes nothing.
  Arm D was built on that flag and is therefore NOT a middle setting — it differs
  from C only in SimHash + suppression, and left inter-tool prose, F2 confidence,
  mid-word `</think>` and thinking-loop armed. E is the only true guards-off arm.
- `watchdog rollback+re-steer ENABLED` prints in every arm: it is the RESPONSE
  policy for a firing, a separate `[behavior]` key, not a detector.

### Speculation runs INSIDE `<think>` on this preset

`MODEL.toml:362` sets `ATLAS_DFLASH_SPEC_THINK = "1"`. The gate it lifts
(`mtp_gate.rs:147`) carries an explicit warning: batch-K verify is not
byte-lossless at T=0 and the 2026-08-16 bisect measured **deterministic 8-9/10
agentic trajectory failures** with it on. That bisect was the DFlash raw-argmax
lane; this is the qwen4_exp MTP lane with the row-exact router chain
(`VERIFY_EXL3_ROW_ROUTER` + `STABLE_GRID` + `NO_VERIFY_ROW_FFN`). 15 iterations
across five arms passed with it on, and the greedy determinism check above
localised all trajectory divergence to tool output rather than flipped tokens —
so it is not biting here. Unmeasured: the thinking-phase throughput it buys.

Records: `agentic_arms/` (per-arm result JSON + iteration tables). Harnesses:
`run_exl3_single.sh` (now with an `EXTRA_ARGS` pass-through and a boot line that
echoes the guard env, so an arm proves itself).

## MoE row cap: +29% prefill, and the cap was sized for HALF the real chunk (2026-09-07)

The overflow tier was assumed near-dead at the shipped cap of 1024
(`moe_prefill_cap.rs`: "1024 is 12.8x the 80-row mean — beyond it the tail is
routing pathology"). Telemetry says otherwise, and the cause is a units error in
that rationale.

### Measurement 1 — the tier fires on EVERY prefill MoE call

`ATLAS_EXL3_MOE_TIER_STATS=1` (counter-only, default-off) on the host side of the
tier select, which already reads `expert_offsets`, so it observes the real
routing distribution with no added device work:

```
cap=1024  sync_calls=832  nosync_calls=528
calls_with_overflow=832 (100.00%)  overflow_experts_total=8868
max_rows_any_expert=4096
```

100% of sync-path calls, ~10.6 overflow experts each. Every one of those costs
~6 host-issued launches (three cooperative grids that serialize) behind the
fused launch.

### Why: the rationale was computed at a 4096-token chunk; the server uses 8192

`--max-prefill-tokens` defaults to **8192** (`serve_args.rs:694`) and neither the
preset nor the serve script overrides it. The doc's mean is 80 rows/expert,
which is 4096 x top_k 10 / 512 experts. At the real 8192 chunk the mean is
**160**, so cap 1024 is **6.4x** the mean, not 12.8x — small enough that overflow
is structural rather than a tail event. Nothing in the code couples the cap to
`max_prefill_tokens`, so the cap silently became half-sized when the chunk
default moved.

### Measurement 2 — counterbalanced cap sweep

1024 -> 2048 -> 4096 -> 4096 -> 2048 -> 1024, one variable
(`ATLAS_EXL3_MOE_ROWS_PER_EXPERT` via the harness's new `MOE_ROWS`), 3 reps per
length, tier telemetry on in every arm so each arm reports its own cap and
overflow rate.

| cap | prefill 4K | 8K | 11K | overflow experts | calls w/ overflow |
|---|---:|---:|---:|---:|---:|
| 1024 (shipped) | 522, 523 | 502, 504 | 493, 493 | 8868, 8867 | 100% |
| 2048 | 566, 567 | 543, 546 | 524, 534 | 4075, 4082 | 98% |
| 4096 | 680, 677 | 652, 649 | 627, 626 | 0, 0 | 0% |

**+29% at 8K (503 -> 650), +30% at 4K, +27% at 11K.** The baseline returned to
523/504/493 against its opening 522/502/493, so there is no drift. Each arm
reproduced twice across independent boots within ~2%. The dose-response is
monotonic and the telemetry supplies the mechanism independently of the
throughput number: overflow experts 8868 -> 4075 -> 0 tracks prefill
503 -> 545 -> 650.

### Validation at cap 4096

- **128K x 4 envelope**: boots OK at util 0.72 (the 315 MB of temp slabs fit).
- **Agentic gate x3** (greedy, guards on): PASS 3/3 webserver_ok + followed
  directions, 6/6 steps, 12.73 s/turn, zero errors / desyncs / CUDA faults.
- **Decode**: C=1 27.4/28.2, C=2 23.9/23.7 aggregate — parity with the recorded
  26.7-27.6 and 23.7-24.3. Prefill-tier change, decode untouched, as expected.

### A second benefit: short batches stop syncing at all

The agentic leg reported `nosync_calls=1728` vs `sync_calls=576` — three quarters
of MoE calls take the `s <= cap` shortcut and never do the host readback. Raising
the cap widens that shortcut, which is the TTFT hypothesis
`exl3_moe_needs_host_sync`'s doc flagged as unmeasured. It also largely
neutralises the "remove the expert_offsets host drain" item: the sync it targets
stops firing for short batches on its own.

### Not yet established

- **Byte-parity.** Three agentic iterations passing is not parity. Experts move
  from the chunked overflow path into the fused kernel — a different reduction —
  and the 128 -> 1024 change carried an `exl3_native_parity` leg. Run echoprobe
  at cap 1024 vs 4096 before treating outputs as identical.
- **Skew headroom.** Zero overflow at 4096 is boundary-dependent:
  `max_rows_any_expert` was exactly 4096 against an 8192 chunk, and 1988 on the
  agentic leg. A more concentrated routing distribution would overflow again, so
  record the result as the PAIR (chunk 8192, cap 4096) — quoting a cap without
  its chunk is how the original rationale drifted.

Records: `cap_sweep/` (per-arm prefill logs + tier-stats lines), `cap_validate/`
(128K boot, agentic JSON, decode). Harness: `cap_sweep.sh`, `tier_probe.sh`,
`run_exl3_single.sh` gained `MOE_ROWS`.

## Files

- `exl3_decode_bench.cu` — standalone microbench (nvcc `-arch=sm_121a -O3 -std=c++17
  --expt-relaxed-constexpr -I kernels/gb10/common`); `sweep` / `moe` modes.
- `microbench_dense_sweep.txt`, `microbench_moe_sweep.txt` — raw microbench output.
- `measure_decode.py` — streaming gap measurement with fingerprint line.
- `serve_exl3.sh`, `serve_nvfp4.sh` — the serve profiles used (`SPARK_BIN` selects the binary).
- `baseline_*` — pass A/B measurements and the raw stage-profile probes.
- `nsys_baseline_kern_sum.csv`, `nsys_baseline_api_sum.csv` — the trace summaries.
- `ab_fused_egress/`, `ab_moe_row_cap/`, `ab_mtp_gpu_greedy/`, `ab_dense_reconstruct/` — round-2 lever
  A/B records (2026-09-06; fingerprint, per-arm measurements, samples; server logs dropped) from
  `ab_fused_egress.sh`, `ab_moe_row_cap.sh`, `ab_mtp_gpu_greedy.sh`, `ab_dense_reconstruct.sh`.
- `parity_moe_prefill_wide/` — the `exl3_native_parity` legs G+G2 run with the wide-cap sub-leg.
- `merged_gate.sh`, `merged_gate_<stamp>/` — the merged-binary gate (prefill, decode, sample, model-card
  agentic run per arm) on the named preset.
- `GPU_SAMPLER_DESIGN.md` — design record for the (negative) GPU-greedy scaffold and a full GPU sampler.
- `MEASUREMENT_PLAN.md`, `prefill_profile.sh`, `concurrency_sweep.sh`, `measure_common.sh`,
  `nsys_kern_table.py` — the operator measurement plan (nsys prefill split, C=1/2/4 sweep with memory watchdogs).
