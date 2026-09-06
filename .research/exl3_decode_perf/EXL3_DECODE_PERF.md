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
8K on this model, NVFP4 ~2.6K; Atlas historically ≤~500. The EXL3 prefill tier
(`exl3_moe_k4_n128_cb2`, 3.2 ms per launch in the baseline trace) and the 2026-08-27 prefill
profile (QSA 34%, grouped MoE 31%) are the starting points for that separate investigation.

## Files

- `exl3_decode_bench.cu` — standalone microbench (nvcc `-arch=sm_121a -O3 -std=c++17
  --expt-relaxed-constexpr -I kernels/gb10/common`); `sweep` / `moe` modes.
- `microbench_dense_sweep.txt`, `microbench_moe_sweep.txt` — raw microbench output.
- `measure_decode.py` — streaming gap measurement with fingerprint line.
- `serve_exl3.sh`, `serve_nvfp4.sh` — the serve profiles used (`SPARK_BIN` selects the binary).
- `baseline_*` — pass A/B measurements and the raw stage-profile probes.
- `nsys_baseline_kern_sum.csv`, `nsys_baseline_api_sum.csv` — the trace summaries.
