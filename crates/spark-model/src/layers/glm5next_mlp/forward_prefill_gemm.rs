// SPDX-License-Identifier: AGPL-3.0-only

//! GLM routed-MoE **prefill** through the shared tensor-core grouped W4A16 GEMM.
//!
//! # What this replaces, and why it is not a new kernel
//!
//! `forward_moe`'s routed arm runs `w4a16_gemv_sw_moe_batchm_mR` — a software-dequant
//! GEMV with **no `mma.sync`** — and `MOE_ROW_BATCH_MAX_ROWS` caps it at 8 rows per
//! launch *regardless of `AVAROK_GLM_PREFILL_ROWS`*. A 256-row prefill sub-chunk is
//! therefore 32 separate 8-row sweeps, and each sweep re-reads the weights of every
//! expert its 8 rows selected. Measured share of prefill: **36.9 % (2026-09-02) to
//! 49.3 % (2026-09-06) of the GPU-busy window** — the single largest bucket in both
//! profiles, and at 94-100 % of its own DRAM roofline, so the win cannot come from
//! more FLOP/s. It has to come from reading each expert's weights FEWER TIMES.
//!
//! `kernels/gb10/common/moe_w4a16_grouped_gemm.cu::moe_w4a16_grouped_gemm_ptrtable`
//! does exactly that: **one launch for all experts**, `mma.sync.aligned.m16n8k16`,
//! `expert_offsets` prefix sum, in-kernel A-gather through `sorted_token_ids`. Thirteen
//! other MoE models already run their prefill on it through
//! `layers::moe::forward_prefill_routed`. GLM never did, because `glm5next_mlp` is a
//! separate bespoke implementation.
//!
//! # 🔴 Layout compatibility — VERIFIED, not assumed
//!
//! The two kernels read the SAME bytes with the same convention. Checked term by term
//! against `w4a16_gemv.cu::w4a16_gemv_partial_rows` (the GEMV this replaces) and
//! `moe_w4a16_grouped_gemm.cu::moe_w4a16_grouped_gemm_ptrtable`:
//!
//! | term | GEMV (`w4a16_gemv.cu`) | grouped GEMM | same? |
//! |---|---|---|---|
//! | packed byte for logical `k` | `B_packed[n * (K/2) + k/2]` (`kk*8 + b`, `k = kk*16 + 2b(+1)`) | `B_expert[gn * (K/2) + gk/2]` | ✅ N-major `[N, K/2]` |
//! | nibble | even `k` → `byte & 0xF`, odd → `byte >> 4` | `(gk & 1) ? (byte >> 4) : (byte & 0xF)` | ✅ |
//! | codebook | `E2M1_LUT` | `E2M1_LUT_MOE` (same 16 values) | ✅ |
//! | block scale | `B_scale[n * (K/16) + k/16]`, E4M3 | `S_expert[gn * (K/GROUP_SIZE) + k_base/GROUP_SIZE]`, E4M3, `GROUP_SIZE 16` | ✅ |
//! | per-tensor scale | `scale2_vals[eid]`, multiplied into the block scale | `scale2_vals[expert_id]`, multiplied into the dequant | ✅ |
//! | remote expert (EP) | `packed_ptrs[eid] == 0` → return, caller's zeros stand | `B_expert == 0` → return, caller's zeros stand | ✅ |
//! | table index | GLOBAL expert id, `[num_experts]` | GLOBAL expert id, `[num_experts]` | ✅ |
//!
//! `weight_loader/glm5_next_load/nvfp4_dequant.rs` states the same three conventions
//! (ModelOpt direct-multiplier order, `GROUP_SIZE = 16` not DeepSeek-V4's 32, even flat
//! index = LOW nibble). **No permutation, no repack, no extra bytes at load** — the
//! already-uploaded `Glm5NextMoePtrTables` are passed straight through.
//!
//! # 🔴 What is NOT bit-identical — and it is NOT only re-association
//!
//! Two things move, and the second one is the bigger one:
//!
//! 1. **Association.** The GEMM accumulates a `k`-major `mma.sync` FP32 chain over
//!    `K_STEP = 16` tiles; the GEMV accumulates two interleaved FP32 `fmaf` chains per
//!    orig-lane and combines them through a warp shuffle tree.
//! 2. **Operand precision.** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` takes
//!    BF16 operands, so the grouped kernel stages its dequantised weight tile as
//!    `__float2bfloat16(E2M1_LUT[nibble] * e4m3 * scale2)` in shared memory. The GEMV keeps
//!    that product in **FP32** all the way into its `fmaf`. The grouped path therefore
//!    carries an extra ~2^-9 relative rounding PER WEIGHT ELEMENT that the GEMV does not.
//!
//! MEASURED 2026-09-22, `examples/glm5next_moe_grouped_prefill_microtest.rs` on n1 (GB10),
//! M=64, top_k=8, 16 experts, N=256, K=512, random NVFP4, against an FP32 host reference:
//!
//! | arm | max_abs | max_abs / ‖ref‖∞ | max_rel |
//! |---|---:|---:|---:|
//! | grouped GEMM vs exact FP32 ref | 1.462067 | 0.003306 | 0.128986 |
//! | GEMV (production) vs the same ref | 0.999542 | 0.002260 | 0.003889 |
//! | **grouped GEMM vs BF16-WEIGHT ref** | **0.998566** | **0.002258** | **0.003888** |
//! | grouped GEMM vs GEMV | 2.000000 | 0.004522 | 0.130725 |
//!
//! The third row is the finding: against a reference that rounds the dequantised weight to
//! BF16 first, the grouped GEMM scores `0.003888` — the GEMV's own `0.003889` to five
//! figures. **All of the residual gap is operand precision; none of it is layout.** That is
//! what makes the layout claim above evidence rather than assertion.
//!
//! It is still a numerics change, so it is gated OFF for decode and for the speculative
//! verify (`rows <= MOE_ROW_BATCH_MAX_ROWS`), which must stay bit-identical, and ON only
//! for prefill — which already left the bit-identical tier when the dense projections moved
//! to cuBLASLt above the same width.
//!
//! # Buffer discipline
//!
//! 🪤 **A59**: every buffer this path needs is allocated in `Glm5NextMlpWorkspace::new`,
//! at load, sized from `max_rows`. Nothing here allocates. The only per-call host traffic
//! is one `[num_experts + 1]` i32 read of `expert_offsets` (see `max_m_tiles_from_offsets`
//! in `tile.rs`).
//!
//! Split into `tile.rs` (tile geometry + env-lever resolution) and `dispatch.rs` (the
//! grouped-GEMM launch + the prefill driver) to keep each file under the 500-LoC cap; both
//! are re-exported here so callers keep using `forward_prefill_gemm::<name>` unchanged.

mod dispatch;
mod tile;

pub(crate) use tile::*;

pub(super) use dispatch::forward_moe_grouped_prefill;

#[cfg(test)]
mod tests;
