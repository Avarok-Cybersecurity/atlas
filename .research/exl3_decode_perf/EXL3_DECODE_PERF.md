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

## What is left after the fix, ranked (hypotheses from the trace, not yet measured e2e)

Post-fix token budget is roughly: EXL3 trellis kernels ~21 ms + cuBLASLt mHC ~7.5 ms + mHC glue
~3 ms + GDN/attention/router/blend/converters ~5 ms + launch gaps ~7 ms.

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

## Prefill (context, not investigated here)

User-supplied reference points: other engines ~1.1K tok/s prefill at 8K on this model, NVFP4 up to
2.6K tok/s; Atlas has not exceeded ~500 tok/s. The EXL3 prefill tier (`exl3_moe_k4_n128_cb2`,
3.2 ms per launch in the baseline trace) and the 2026-08-27 prefill profile (QSA 34%, grouped MoE
31%) are the starting points for that separate investigation.

## Files

- `exl3_decode_bench.cu` — standalone microbench (nvcc `-arch=sm_121a -O3 -std=c++17
  --expt-relaxed-constexpr -I kernels/gb10/common`); `sweep` / `moe` modes.
- `microbench_dense_sweep.txt`, `microbench_moe_sweep.txt` — raw microbench output.
- `measure_decode.py` — streaming gap measurement with fingerprint line.
- `serve_exl3.sh`, `serve_nvfp4.sh` — the serve profiles used (`SPARK_BIN` selects the binary).
- `baseline_*` — pass A/B measurements and the raw stage-profile probes.
- `nsys_baseline_kern_sum.csv`, `nsys_baseline_api_sum.csv` — the trace summaries.
