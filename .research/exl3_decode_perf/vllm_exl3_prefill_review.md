# vllm-exl3 vs Atlas EXL3 prefill — read-only review (2026-09-05)

Sources: `https://github.com/vcruz305/vllm-exl3` cloned to
`/home/ms/.claude/jobs/5a7bd33d/tmp/vllm-exl3` (HEAD `de65e22`, 2026-09-05; 41 commits, all
2026-08-31..09-05); upstream ExLlamaV3 v1.4.6 at `/tmp/atlas-exl3-upstream-review`; Atlas worktree
`/home/ms/atlas/.claude/worktrees/exl3-research` (read only). No GPU process was started; nothing was
written into the Atlas repo. Every number below is either quoted from their repo (marked THEIRS),
derived arithmetically from published shapes (DERIVED), or a hypothesis (HYP).

## 0. One-paragraph verdict

vllm-exl3 is a thin vLLM plugin around ExLlamaV3's kernels for GLM-5.3-Flash / DeepSeek-V4 on DGX
Spark. Its prefill story has exactly two mechanisms Atlas lacks: (1) it runs upstream's fused
`exl3_moe` kernel with a **2048-row per-expert cap** instead of Atlas's 128 (so no overflow tier at
all until >2048 rows/expert), and (2) a copied third-party **128x128-tile "fat" expert GEMM** that
decodes each trellis B-fragment once per 128 rows instead of once per 16 rows, with the output
Hadamard, routing weight and token scatter fused into the tile. For dense (non-expert) linears it
simply inherits upstream's policy of **reconstruct-to-fp16 + cuBLAS above 144 rows**, which Atlas's
dense tier does not do. Their own "prefill GEMM" (`exl3_gemm.cu`) is a chunked <=8-row GEMV loop
and should not be imported. Nothing on split-K, multi-stream overlap, or prefill CUDA graphs exists
in the repo. Their fat-GEMM path was unreachable in the v0.3.1 release (fixed 09-04 in `0f644f7`) and
the post-fix commits state Spark execution was unavailable, so treat the fat-path numbers as
microbenchmarks of the kernel, not of the serving path.

## 1. What vllm-exl3 does for PREFILL (files + commits)

**Routed experts — `src/vllm_exl3/exl3.py::apply_exl3_fused_moe` (lines 1095-1250).**
- Sorts by expert (argsort of local ids), bincounts on device, and calls upstream
  `exllamav3_ext.exl3_moe` ONCE per layer with temp slabs of `TEMP_ROWS_FUSED = 2048` rows
  (`exl3.py:99`, `build_exl3_fused_state` 755-758). Upstream derives `max_tokens_per_expert` from the
  temp-slab shape at runtime (`exl3_moe.cu:168`) — it is a host sizing choice, not a kernel constant.
  CHANGELOG 0.1.1 (THEIRS): raising 128 -> 2048 "fix[ed] the >163k-token prefill stall where fat
  experts fell back to a slow per-expert reconstruction path"; AGENTS.md: "do not lower it".
- If any expert exceeds 2048 rows the token batch is re-sliced into 2048-token slices (1155-1170).
- Experts with `count > FAT_EXPERT_THRESHOLD` (env `VLLM_EXL3_FAT_THRESHOLD`, default 256, `exl3.py:101`)
  are masked out of the fused launch (sentinel id, zero weight) and routed to
  `apply_exl3_batched_fat` (959-1093) — reachable only since `0f644f7` (2026-09-04, "Eliminate early
  return bug in apply_exl3_fused_moe to make fat GEMM dispatch reachable").
- `apply_exl3_batched_fat`, per fat expert (Python loop, one launch set per expert): index_select the
  rows -> `had_r_128(h, suh)` -> concat gate+up trellis into a `packed13` scratch (a device copy of
  both weights per call) -> `exl3_fat_gemm` (fp32 out, svh+Hadamard fused) -> SiLU*up in torch ->
  `had_r_128(act, down.suh)` -> `exl3_fat_gemm_scatter` (down proj, routing weight multiply and
  `out[token] +=` fused). Fallback when not K4/MCG: `ext.reconstruct` + `ext.hgemm` + `had_r_128`.

**The fat kernel — `csrc/exl3_fat_gemm.cu` (305 lines), copied verbatim from Mia's AI Lab
`GLM-5.3-Flash-EXL3-2x-DGX-Sparks` commit `4b8d3c7` ("perf(exl3): accelerate fat-expert prefill").**
- 256 threads = 8 warps; tile M=128, N=128, K=16; grid `(N/128, ceil(M/128))`; each warp owns one
  16-col N block and calls `dq_dispatch<4,1>` (K=4, MCG only — `check_common` rejects everything
  else) once per K-step, then reuses `frag_b0/b1` across 8 m16 blocks (`#pragma unroll mb`). A tile
  loaded to smem with an XOR swizzle; B tile (8 x 64 words) loaded by 64 threads; **no cp.async, no
  double buffering, two `__syncthreads` per 16-wide K step.** fp32 accumulators; epilogue writes the
  128-col row to smem, applies `fat_had_ff_128` (H128 * 0.0884 * svh) per row, then plain store
  (`scatter=false`) or `out[token_idx[row]*N + col] += value * route_weight[row]` (`scatter=true`,
  **non-atomic**, correctness argued from single-stream ordering + one route per token per expert).
- THEIRS (README v0.3.1 table, microbench, "cosine similarity 1.000000" as the only parity metric):
  down-proj fat-scatter vs "Stock Reconstruct + GEMM" at GLM shapes: M=256 144.3->86.9 us (1.66x),
  M=512 222.8->106.4 (2.09x), M=1024 400.4->195.3 (2.05x), M=2048 733.1->508.7 (1.44x). Note the
  baseline is reconstruct+cuBLAS, NOT the 16-row `exl3_moe` kernel, so the gain relative to Atlas's
  current fused tier is unquantified anywhere in their repo.

**Dense linears — `Exl3LinearMethod.apply` (`exl3.py:2251-2320`)** just calls upstream
`LinearEXL3.forward` per shard, i.e. upstream v1.4.6 policy (`modules/quant/exl3.py:10,135,184`):
`rows <= AUTO_RECONSTRUCT_THRESHOLD (144)` -> cooperative `exl3_gemm`; otherwise
`reconstruct` (fp16 weight) + `hgemm` + `had_r_128`; at `rows >= 1024` the fused
`reconstruct_had_slice` folds both Hadamards and sign vectors into the weight so the GEMM runs on
raw x (upstream comment: the standalone had launches were "~14% of long-chunk prefill GPU time";
"fused kernel costs ~4x plain reconstruct ... breakeven rows ~400-900").

**Their "Power-of-Two Chunked Prefill GEMM" (`csrc/exl3_gemm.cu`, 31 lines, commit `3fbb56e`)** is
a host loop calling the cooperative GEMV on <=8-row slices. THEIRS: "7.85 TFLOPS (13.0x faster than
legacy prefill)", test at m=128, K=2, 4096x4096 (`tests/test_native_gemm.py:17,58`), which also
asserts upstream `exl3_gemm` runs 0.59 TFLOPS there and cuBLAS fp16 27.6 TFLOPS. DERIVED: 16
launches x 4 MB packed = 64 MB streamed per 128 rows, so at 231 GB/s the ceiling is ~15 TFLOPS and
it degrades linearly with M (weights re-read every 8 rows). Not a prefill GEMM; do not import.

**Absent from the repo:** split-K, large-M tiling beyond the fat kernel, any multi-stream overlap,
autotuning (GEMV `cfg` is hard-coded 0, `exl3_gemv.cu:38`), prefill CUDA graphs, reconstruct caching
(reconstruct is redone per call), expert-level batching of the fat kernel (per-expert Python loop).

## 2. What vllm-exl3 does for DECODE (brief)

- `csrc/p2b_moe.cu` (commit `c403165`): single cooperative launch, 4 phases with `grid.sync()`
  (input Hadamard -> batched gate/up GEMV over all active experts -> SwiGLU + down-Hadamard -> down
  GEMV -> atomicAdd into one fp32 row). m=1 and hidden=4096 hard-coded; batch >1 is a Python loop of
  one launch per row (`_apply_native_fused_moe`, `exl3.py:865`). THEIRS: 497 -> 287.8 us/layer on
  GLM-5.3-Flash K2, +45.6% avg tok/s vs the exllamav3 baseline (README). Atlas's batched mgemm over
  S = T x top_k slots is structurally ahead of this; nothing to take.
- `csrc/p2b_batched.cu` `launch_batched_fast_2` (commit `0ae7e63`, 09-05): replaces the cooperative
  worklist with THREE plain launches (input had / per-(expert,col-group) GEMV / output had) for
  K2-MCG m=1 — the reason is not stated, but plain launches are trivially graph-capturable and their
  contract test does eager-vs-`torch.cuda.graph` replay (`tests/test_native_moe_contract.py:137`).
  Relevant to Atlas's decode graph veto (see `bench/qwen4_exp/EXL3_VENDOR_REVIEW.md` follow-up 1).
- Per-kernel occupancy query (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) to size cooperative
  grids (`exl3_gemv.cu:47`, `p2b_moe.cu`) — same as vendor-review follow-up 3.
- No int8-activation GEMV, no autotuner, no MTP/spec kernels (their "speculative scheduler" is a pure
  Python K-by-batch-size table).

## 3. Atlas's EXL3 prefill path, as read

- MoE: `forward_prefill_exl3.rs` -> per 4096-token batch (`pf_t_cap`, env
  `ATLAS_EXL3_MOE_PREFILL_BATCH_TOKENS`) -> `moe_sort_by_expert` -> `exl3_moe_prefill_routed`
  (`moe_prefill.rs:339`): stage, bf16->f16, ONE D2H of expert_offsets when S>128, fused
  `exl3_moe_k4_n128_cb2` over experts with `0 < count <= 128` (`EXL3_MOE_MAX_TOKENS_PER_EXPERT`,
  `moe_prefill.rs:56`; slabs `pf_concurrency=6 x 128 x {H,I}` in `ptr_table_build.rs:302-305`),
  then the overflow tier (`moe_prefill_overflow.rs`): per expert with count>128, per 1024-row chunk
  (`EXL3_MOE_OVERFLOW_CHUNK_ROWS`, `tables.rs:174`), 5 launches: gather, gate `exl3_gemm`, up
  `exl3_gemm`, silu_mul, down `exl3_gemm`, store-to-slots. Deterministic epilogue: every expert
  plain-stores its weighted row to its own sorted slot (`output_slots`, 31st kernel arg) and
  `exl3_moe_reduce_slots_f32` sums each token's top_k slots in fixed order (`moe_prefill_det.rs`).
- Both the fused kernel and `exl3_gemm` decode B once per **16-row** M slice
  (`exl3_vendor/exl3_gemm_kernel.cuh:61-76`, `exl3_moe_kernel.cuh` `while (size_m > 0) ... -= 16`,
  `MOE_TILESIZE_M 16`). DERIVED for qwen4_exp (512 experts, top-10, 4096-token chunk): 40,960 slot
  rows -> 80 rows/expert average -> each expert's gate/up/down B is re-decoded ~5x per chunk; the
  module doc itself says overflow (>128) is "ROUTINE at serving shapes".
- Dense (`exl3_dense.rs`): m<=8 GEMV (K in 2..4) else cooperative `exl3_gemm` row-batched at
  `EXL3_DENSE_STAGE_ROWS_DEFAULT = 4096` (`exl3_dense/stage.rs:20`), grid = min(tiles, 48). A
  4096-row chunk = 256 sixteen-row passes over the full weight. There is NO reconstruct tier at any
  m, although `kernels/gb10/common/exl3_reconstruct.cu` (fused reconstruct_had, 24 instances,
  GPU-vs-CPU byte-identical per `.research/EXL3_DECODE_FINDINGS.md`) already exists for the
  materialise path.
- Contracts to preserve: deterministic per-slot epilogue; stable grid (exl3_gemm grid is
  m-independent); one in-flight launch per locks buffer; no graph capture on these arms.
- Measured baseline (`.research/exl3_decode_perf/EXL3_DECODE_PERF.md` "Prefill baseline"): ~390 tok/s
  flat 6K-11K. No EXL3 prefill kernel breakdown exists yet; the decode trace shows
  `exl3_moe_k4_n128_cb2` at 3.19 ms x 48 for one (unknown-length) prompt prefill.
- DERIVED bounds per 4096-token chunk, qwen4_exp K=4 experts (2560x640x3 x 0.5 B = 2.46 MB/expert):
  weights streamed once = 1.26 GB/layer -> 5.4 ms at 231 GB/s -> ~15.6K tok/s ceiling over 48 layers
  from routed-weight bandwidth alone; MMA work 2 x 40960 x 3 x 1.64M = 0.40 TFLOP/layer -> at the
  27.6 TFLOPS cuBLAS reference THEY quote, ~0.7 s/chunk -> ~5.9K tok/s. Observed 390 tok/s (10.5 s
  per 4096 tokens) is far below both, so (HYP) the cost is trellis-decode ALU x5 re-decode, the
  serialized per-expert overflow launches at <=48 blocks each, and the non-MoE layers (the 08-27
  NVFP4 profile had QSA at 34%).

## 4. Transferable ideas, ranked by payoff / effort

**R1. Profile first (effort: hours; risk: none).** nsys one 8K cold prefill on the EXL3 binary and
read `exl3_moe_*`, `exl3_gemm_k6_*` (dense K=6), `exl3_gemm_k4_*` (overflow), gather/store, QSA, GDN
shares; also log `Exl3MoePrefillStats.overflow_experts` (already traced at `trace!` level,
`forward_prefill_exl3.rs:230`) per batch. R2-R4 below target different kernels; without this split
their ordering is a guess.

**R2. Raise `EXL3_MOE_MAX_TOKENS_PER_EXPERT` from 128 toward vllm-exl3's 2048 (effort: ~1 day;
numerics risk: low).**
- Mechanism: the fused kernel already handles any count via its 16-row loop; the cap is only the
  temp-slab height (runtime arg `max_tokens_per_expert`). Every expert moved off the overflow tier
  saves >=5 host-issued launches per 1024 rows, three of them cooperative `exl3_gemm` grids of <=48
  blocks that serialize behind each other, plus the D2H-dependent host loop; the fused kernel's
  ticket scheduler instead keeps ~6 expert groups (48/8 SMs) busy.
- Atlas touch: `moe_prefill.rs:56` const; the four `128` literals in `ptr_table_build.rs:302-305`
  and the `total` formula at :317; header comments in `kernels/gb10/common/exl3_moe.cu`. The det
  slot slab is per sorted slot and unaffected. DERIVED slab cost at C=6, H=2560, I=640: 128 rows
  9.8 MB; 512 rows 39 MB; 2048 rows 157 MB.
- Risk: numerics stay deterministic (same plain-store epilogue), but experts that used to take the
  overflow `exl3_gemm` shapes now take the MoE tile shape -> different fp32 accumulation order for
  those experts (a one-time bit change, not run-to-run). Load-balance tail (HYP): one 2048-row expert
  costs 16x a 128-row one on an 8-SM group, and `group_size` only widens when `num_active` is small
  (`moe_prefill.rs:292-296`); start at 512-1024 and keep the overflow tier for extremes.
- Evidence: THEIRS is only the 0.1.1 CHANGELOG stall fix and the fact that their 1,875 tok/s cold
  65K-context GLM prefill (README, v0.3.0, K2, BF16 dense layers, fat path unreachable at the time)
  ran entirely on this configuration. No isolated speedup number exists; effect on GB10 is HYP.

**R3. Dense prefill: reconstruct + BF16 GEMM above a row threshold (effort: 2-4 days; risk: medium).**
- Mechanism: upstream's own policy (threshold 144; fused reconstruct_had at >=1024 rows) — one pass
  of trellis decode per chunk instead of rows/16 passes, then a tensor-core GEMM. Atlas has the
  reconstruct kernel (byte-identical parity) and BF16 GEMMs (`dense_gemm_bf16_pipelined`, cuBLASLt);
  what is missing is the tier in `exl3_dense.rs::exl3_dense_linear_shared_a` GEMM branch plus a
  weight transient in `exl3_dense/stage.rs` (fp16/bf16 `in_dim x out_dim`, e.g. 13 MB for 2560^2,
  84 MB for the 2560x16384 GDN `[Q|K|V|Z]` arena row — or N-slice like upstream
  `MAX_RECONSTRUCT_SLICE_N`). The 4.05bpw attention/GDN projections are K=6, the most ALU-heavy dq
  path, so (HYP) this is where the 16-row re-decode hurts most.
- Risk: not bit-identical to the current tier (Hadamard folded into fp16 weights vs fp16 activation
  rotation; different accumulation). Run-to-run determinism requires a GEMM whose config does not
  depend on M — prefer Atlas's own fixed-config BF16 GEMM over cuBLASLt heuristics (memory: cuBLASLt
  picks per M). Honor `Exl3DenseOut::fp32` for residual-bound projections (fp32 accumulate, bf16 out).
  Reconstruct transient must sit inside the util pledge (allocate at stage construction, not per call).
- Evidence: THEIRS (test docstring) upstream `exl3_gemm` 0.59 TFLOPS vs cuBLAS 27.6 TFLOPS at m=128 on
  GB10; upstream's own "~14% of long-chunk prefill" comment for the had launches. Both unverified here.

**R4. A 128-row-M-tile grouped expert GEMM for hot experts (effort: 1-2 weeks; risk: medium).**
- Mechanism (the one genuinely new kernel idea in their tree): decode each B fragment once per 128
  rows (8 m16 blocks share it) -> 8x less trellis-decode ALU per MMA than `exl3_moe`/`exl3_gemm`;
  fuse svh+Hadamard and the routing weight into the fp32 epilogue. Do NOT copy their kernel as is:
  it is K4/MCG-only (qwen4_exp is MUL1, cb=2 -> template on `cb`), unpipelined (no cp.async, two
  barriers per 16-K step; K-tile 16 is too shallow for 48 SMs, HYP), and launched per expert with grid
  `(N/128, ceil(M/128))` — for qwen4_exp gate/up N=640 that is 5 column blocks per row block, i.e. a
  per-expert launch leaves ~43 of 48 SMs idle. Atlas's version must be grouped: `blockIdx.z` or a
  work list over experts via the existing pointer tables (`Exl3MoeProj`), replacing the chunked
  `exl3_gemm` calls in `moe_prefill_overflow.rs::run_overflow_expert` (gate/up as one launch with two
  B pointers, down as a second), or becoming the fused kernel's inner for `count > 128`.
- Determinism: the `scatter=false` form (plain store of the weighted row to the sorted slot) is
  exactly Atlas's `output_slots` contract; never adopt the non-atomic `+=` scatter. Fixed tile/split
  -> stable grid. Parity: extend the MoE parity example against the fused tier at count>128.
- Evidence: THEIRS 1.44-2.09x vs reconstruct+cuBLAS at M=256-2048 (microbench, GLM shapes, cosine
  parity only, path unreachable in the shipped 0.3.1, post-fix commits not GPU-validated per
  `4f605b1` message). Against Atlas's 16-row tier: no number anywhere; HYP: bounded by the 8x ALU
  reduction, realized only if R1 shows the MoE tier is decode-ALU-bound rather than launch-bound.

**R5. (Decode, already on the vendor-review list) plain-launch restructuring for graph capture**
(`0ae7e63`) and per-kernel occupancy caching (`exl3_gemv.cu:47`). Nothing else decode-relevant.

## 5. Do NOT import / caveats on their claims

- `exl3_gemm.cu` chunked-GEMV "GEMM" (section 1) — O(M) weight re-reads.
- `packed13` concat: copies both trellis weights to scratch on every fat call (`exl3.py:1000-1002`).
- Fat scatter's non-atomic `+=` relies on stream serialization of per-expert launches.
- Non-`.item()`-free host path: `bool(fat.any().item())` and `counts.tolist()` syncs per layer
  (`exl3.py:1175,1236,1243`) — Atlas already does one D2H per batch.
- README e2e numbers (+45.6% decode, 1,875 tok/s prefill at 65K) are vs the exllamav3 plugin
  baseline on GLM-5.3-Flash K2 with BF16 dense layers, not comparable to qwen4_exp on Atlas.
- The actual kernel author is Mia's AI Lab (`4b8d3c7`); their repo was not reviewed here and may
  have moved past the copied snapshot — worth a direct look before building R4.

## 6. Suggested order

R1 (profile) -> R2 (cap 128->512/1024, one-line + slabs, A/B on the 8K/11K `measure_prefill.py`
harness with the same binary and flags) -> R3 if K=6 dense GEMMs are a top-3 kernel -> R4 only if
the fused MoE tier remains the top kernel after R2 and the trace shows it ALU-bound.
