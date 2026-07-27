// SPDX-License-Identifier: AGPL-3.0-only

//! MoE (Mixture of Experts) FFN component.
//!
//! Batched expert dispatch: top-K experts run in 2 fused kernel launches
//! (gate+up, silu+down) instead of 10 × 5 individual launches. Expert indices
//! and weights stay on device — zero D2H synchronization.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8ExpertWeight, MoeWeights, QuantizedWeight};

/// Device-side pointer table for one projection across all experts.
///
/// Enables GPU-side expert dispatch: the batched GEMV kernel reads
/// expert_id from device memory, then indexes these tables to find
/// the correct weight pointers — no CPU involvement needed.
pub(crate) struct ExpertPtrTable {
    /// `[num_experts]` u64 device pointers to each expert's B_packed.
    pub(crate) packed_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's B_scale.
    pub(crate) scale_ptrs: DevicePtr,
    /// `[num_experts]` f32 per-expert scale2 values.
    pub(crate) scale2_vals: DevicePtr,
}

/// Device-side pointer table for FP8 expert dispatch (one projection).
///
/// FP8 experts use 2 pointer arrays (weight + block_scale) instead of
/// NVFP4's 3 (packed + scale + scale2). The fused FP8 MoE kernel indexes
/// these tables by expert_id to load the correct FP8 weight matrix.
pub(crate) struct Fp8ExpertPtrTable {
    /// `[num_experts]` u64 device pointers to each expert's FP8 weight.
    pub(crate) weight_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's block scales.
    pub(crate) scale_ptrs: DevicePtr,
}

/// Unified expert pointer table for any quantization format.
///
/// Replaces the separate `ExpertPtrTable` (NVFP4) and `Fp8ExpertPtrTable` (FP8)
/// with a single enum. The MoE forward path matches on this to select the
/// correct fused kernel (moe_shared_expert_fused vs moe_shared_expert_fused_fp8).
#[allow(dead_code)]
pub(crate) enum ExpertPtrSet {
    /// NVFP4: 3 pointer arrays (packed_ptrs, scale_ptrs, per-expert scale2 f32).
    Nvfp4 {
        packed_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
        scale2_vals: DevicePtr,
    },
    /// FP8: 2 pointer arrays (weight_ptrs, block_scale_ptrs).
    Fp8 {
        weight_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
    },
}

/// MoE feed-forward network component.
///
/// Not a `TransformerLayer` — used as a component inside layers
/// for the FFN/MoE block after post-attention norm.
#[allow(dead_code)]
pub struct MoeLayer {
    pub weights: MoeWeights,
    /// Quant format of the ROUTED experts as landed in GPU memory. `Nvfp4`
    /// (default) = packed E2M1 + FP8-E4M3 per-16 block scales + f32 per-tensor
    /// global. Set to `Mxfp4E8m0` by the DeepSeek-V4 native-MXFP4 loader
    /// (transcode-free: E8M0 per-32 scales, no global) so the Phase-K E8M0
    /// GEMM variants dispatch on it instead of the NVFP4 kernels. Consumed at
    /// the grouped/decode GEMM call sites (assert via `WeightQuantFormat::expect`).
    // Written by the loader (Phase L); READ at the GEMM dispatch sites in Phase K.
    // Until Phase K wires the read, `deny(warnings)` would flag it never-read.
    #[allow(dead_code)]
    pub(crate) experts_scale_kind: crate::weight_map::WeightQuantFormat,
    /// Quant format of the SHARED expert (ARM-2 Phase-K RIDER A1). The native
    /// V4 ckpt is heterogeneous: routed experts `Mxfp4E8m0` but the shared
    /// expert is FP8→`Nvfp4`. Keyed off the weight tag (not `is_shared`
    /// positionality) so the dual-format decode kernel's `expect` net fires if
    /// a future ckpt ships a different shared format. Default `Nvfp4`.
    #[allow(dead_code)]
    pub(crate) shared_experts_scale_kind: crate::weight_map::WeightQuantFormat,
    // NVFP4-quantized gate weight (quarters bandwidth for routing)
    gate_nvfp4: Option<QuantizedWeight>,
    /// Pre-expert norm: applied to input AFTER routing but BEFORE expert dispatch.
    /// Gemma-4 26B: router sees raw residual, experts see pre_feedforward_layernorm_2(residual).
    pub pre_expert_norm: Option<crate::weight_map::DenseWeight>,
    pre_expert_norm_k: spark_runtime::gpu::KernelHandle,
    dense_gemv: KernelHandle,
    w4a16_gemv: KernelHandle,
    w4a16_gemm: KernelHandle,
    dense_gemm: KernelHandle,
    /// FP32-output router GEMM + FP32-input top-K for the ATLAS_FP32_GATE path.
    /// Zero (unresolved) when the kernels are absent; dispatch falls back to BF16.
    dense_gemm_f32out: KernelHandle,
    /// FP32-in/FP32-out router GEMM for ATLAS_FP32_ROUTING (reads the FP32
    /// router_in from residual_add_rms_norm_gatef32). Zero if absent.
    dense_gemm_f32in: KernelHandle,
    moe_topk_f32: KernelHandle,
    moe_expert_gate_up_shared: KernelHandle,
    moe_expert_silu_down_shared: KernelHandle,
    moe_topk: KernelHandle,
    moe_weighted_sum_blend: KernelHandle,
    residual_add: KernelHandle,
    moe_topk_batched: KernelHandle,
    // K=2 fused MoE kernel handles
    moe_expert_gate_up_shared_batch2: KernelHandle,
    moe_expert_silu_down_shared_batch2: KernelHandle,
    moe_weighted_sum_blend_batch2: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    // K=3 fused MoE kernel handles
    moe_expert_gate_up_shared_batch3: KernelHandle,
    moe_expert_silu_down_shared_batch3: KernelHandle,
    moe_weighted_sum_blend_batch3: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    // Generic token-major NVFP4 MoE kernels. Used as an opt-in decode
    // concurrency experiment for N>=4 without grouped-GEMM sorting.
    moe_expert_gate_up_shared_token_major: KernelHandle,
    moe_expert_silu_down_shared_token_major: KernelHandle,
    moe_weighted_sum_blend_token_major: KernelHandle,
    moe_decode_atomic_c4_silu_down_accum_k: KernelHandle,
    moe_decode_atomic_c4_finalize_k: KernelHandle,
    // Sorted/grouped prefill path
    moe_sort_by_expert: KernelHandle,
    moe_sorted_gate_up: KernelHandle,
    moe_sorted_silu_down: KernelHandle,
    moe_grouped_gemm: KernelHandle,
    moe_silu_mul: KernelHandle,
    /// Activation kernel for sorted/unfused path. SiLU by default, GeGLU for Gemma-4.
    moe_act_mul: KernelHandle,
    /// When true, decode uses the sorted prefill path (avoids fused SiLU kernels).
    gelu_activation: bool,
    moe_unpermute_reduce: KernelHandle,
    moe_batched_blend: KernelHandle,
    /// Pointer tables for batched expert dispatch.
    gate_ptrs: ExpertPtrTable,
    up_ptrs: ExpertPtrTable,
    down_ptrs: ExpertPtrTable,
    /// Transposed pointer tables for coalesced prefill GEMM.
    gate_ptrs_t: Option<ExpertPtrTable>,
    up_ptrs_t: Option<ExpertPtrTable>,
    down_ptrs_t: Option<ExpertPtrTable>,
    /// CUTLASS grouped-NVFP4 swizzled SFB weight-scale tables
    /// (`ATLAS_HOLO_MOE_GROUPED_CUTLASS`). Device `[num_experts]` u64 arrays of
    /// per-expert SFB pointers, built at load by `build_cutlass_grouped_sfb` from
    /// the `gate_ptrs_t`/`up_ptrs_t` `[K/16,N]` scales (`pack_weight_sfb` swizzle).
    /// The grouped kernel reads `gate_ptrs.packed` (`[N,K/2]`) + these SFB + the
    /// real per-expert `scale2`. `None` => the CUTLASS grouped path is unavailable.
    gate_sfb_cutlass: Option<DevicePtr>,
    up_sfb_cutlass: Option<DevicePtr>,
    down_sfb_cutlass: Option<DevicePtr>,
    /// Keeps the per-expert SFB buffers + the two pointer arrays alive.
    _cutlass_sfb_owned: Vec<DevicePtr>,
    /// Lazy down_proj transpose scratch — populated at the start of each
    /// prefill call when the persistent transpose pass couldn't fit
    /// down_proj. Decode keeps using `down_ptrs` (untransposed); prefill
    /// uses `down_ptrs_t` pointing into this scratch. Shared across all
    /// MoE layers (the same scratch is overwritten layer-by-layer during
    /// the sequential forward).
    ///
    /// `down_t_scratch_packed`: contiguous `[num_experts × N × K/2]` bytes.
    /// `down_t_scratch_scale`:  contiguous `[num_experts × N × K/16]` bytes.
    /// Both `None` when the persistent transpose pass already covered
    /// down (full-fits path) or when the layer doesn't need scratch
    /// transpose (FP8 experts, etc.).
    down_t_scratch_packed: Option<DevicePtr>,
    down_t_scratch_scale: Option<DevicePtr>,
    /// Kernel handle for the batched per-expert uint8 transpose.
    moe_transpose_u8_batched_k: KernelHandle,
    // ── Phase 8a transposed-layout decode kernels (unified-layout MoE).
    // Loaded eagerly at construction. Currently NOT wired into the
    // dispatch — Phase 8a part 3/3 will route decode through these once
    // the weight loader produces transposed-only pointer tables.
    moe_expert_gate_up_shared_t_k: KernelHandle,
    moe_expert_silu_down_shared_t_k: KernelHandle,
    // ARM-2 Phase-K: native-MXFP4 (E8M0 routed / NVFP4 shared) dual-format
    // decode variants. KernelHandle(0) on models that don't ship them.
    moe_expert_gate_up_shared_t_e8m0_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_k: KernelHandle,
    // ── sqrtsoftplus routing (DeepSeek-V4) ──
    moe_topk_sqrtsoftplus_k: KernelHandle,
    moe_topk_sqrtsoftplus_batched_k: KernelHandle,
    // ── hash routing (DeepSeek-V4 first `num_hash_layers` MoE layers) ──
    moe_hash_route_k: KernelHandle,
    moe_hash_route_batched_k: KernelHandle,
    /// Static `tid2eid` table [vocab_size, top_k] i64 — present ONLY for the
    /// hash-routed layers (the loader supplies it only for those). `Some`
    /// here is the SSOT that this layer routes via the static hash table
    /// instead of the learned gate's top-K.
    tid2eid_dev: Option<DevicePtr>,
    moe_expert_gate_up_shared_batch2_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch2_t_k: KernelHandle,
    moe_expert_gate_up_shared_batch3_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch3_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch2_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch2_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch3_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch3_t_k: KernelHandle,
    /// `ATLAS_UNIFIED_MOE_LAYOUT=1` opts in to the unified-layout decode
    /// path: gate/up/down all use transposed `[K/2, N]` layout, decode
    /// dispatches to `moe_expert_*_shared_t` kernels. Default off — the
    /// dispatch falls through to the original `[N, K/2]` kernels.
    /// Resolved once at construction.
    unified_layout: bool,
    /// `ATLAS_NVFP4_GATE_UP_M128=1` opts in to the M=128 fused gate+up
    /// kernel (Block D #3, Avarok tile-shape rewrite). Halves block count
    /// at large prefill — better SM amortization on GB10's 25-SM budget.
    /// Currently only minimax-m2-229b ships the kernel; other models keep
    /// `moe_fused_gate_up_t_k64_m128 == KernelHandle(0)` and dispatch
    /// falls through to the M=64 path even when the env var is set.
    nvfp4_gate_up_m128: bool,
    /// `ATLAS_HOLO_MOE_GATEUP_FP4=1` opts the prefill fused gate_up onto the
    /// block-scaled FP4 kernel. Reads the SHARED FAST_MOE=full `gate_ptrs_t`/
    /// `up_ptrs_t` `[K/2,N]` tables (no extra MoE memory); dispatch also requires
    /// those tables present + the FP4 kernel handle != 0.
    gateup_fp4: bool,
    /// `ATLAS_HOLO_MOE_DOWN_FP4=1` — same, for the prefill down projection over
    /// the shared `down_ptrs_t` table.
    down_fp4: bool,
    /// `ATLAS_HYBRID_MOE_LAYOUT=1` opts in to the hybrid-layout path:
    /// keep BOTH original `[N, K/2]` weights (for decode + MTP verify) AND
    /// transposed `[K/2, N]` weights (for prefill). Doubles MoE-weight
    /// memory but recovers the ~15 % decode regression that pure unified
    /// layout suffers from. Resolved once at construction; mutually
    /// exclusive with `unified_layout` at the dispatch level (hybrid wins
    /// on decode paths since it preserves untransposed warp-reduction
    /// parallelism).
    hybrid_layout: bool,
    /// Transposed shared expert weights for prefill.
    shared_gate_t: Option<QuantizedWeight>,
    shared_up_t: Option<QuantizedWeight>,
    shared_down_t: Option<QuantizedWeight>,
    moe_grouped_gemm_t: KernelHandle,
    moe_grouped_gemm_t_k64: KernelHandle,
    moe_fused_gate_up_t: KernelHandle,
    moe_fused_gate_up_t_k64: KernelHandle,
    // ARM-2 Phase-K: native-MXFP4 (E8M0 per-32) prefill variants of the W4A16
    // routed-expert GEMMs. KernelHandle(0) on models that don't ship them
    // (only the deepseek-v4-flash target compiles the `_e8m0` entries).
    moe_grouped_gemm_e8m0: KernelHandle,
    moe_grouped_gemm_t_e8m0: KernelHandle,
    moe_grouped_gemm_t_k64_e8m0: KernelHandle,
    moe_fused_gate_up_t_e8m0: KernelHandle,
    moe_fused_gate_up_t_k64_e8m0: KernelHandle,
    /// M=128 variant of the K64 fused gate+up kernel (Block D #3, Avarok
    /// tile-shape rewrite). Loaded with `try_kernel` — falls back to
    /// `KernelHandle(0)` on models that don't ship the kernel; dispatch
    /// gates on `nvfp4_gate_up_m128` AND handle non-zero.
    moe_fused_gate_up_t_k64_m128: KernelHandle,
    /// FUSED FP4 (block-scaled e2m1) variant of the K64 fused gate+up kernel
    /// (`ATLAS_HOLO_MOE_GATEUP_FP4`). Same signature as `moe_fused_gate_up_t_k64`
    /// but runs one `mma.sync.kind::mxf4nvf4.scale_vec::4X.m16n8k64` per k64
    /// tile (vs 2× m16n8k32 e4m3). `try_kernel` — `KernelHandle(0)` on images
    /// lacking it; the dispatch in `forward_prefill_routed` only fires when this
    /// handle != 0, `gateup_fp4` is set, and the shared `gate_ptrs_t`/`up_ptrs_t`
    /// tables are present (FAST_MOE=full).
    moe_fused_gate_up_t_k64_fp4: KernelHandle,
    moe_fp8_grouped_gemm_t: KernelHandle,
    w4a16_gemm_t: KernelHandle,
    bf16_to_fp8_k: KernelHandle,
    /// Pre-dequanted FP8 weights for zero-overhead prefill GEMMs.
    gate_fp8: Option<DevicePtr>,
    shared_gate_fp8: Option<DevicePtr>,
    shared_up_fp8: Option<DevicePtr>,
    shared_down_fp8: Option<DevicePtr>,
    fp8_gemm_k: KernelHandle,
    /// Secondary CUDA stream for overlapping shared expert with routed experts.
    prefill_stream: u64,
    /// Event pair for stream synchronization (input_ready, shared_done).
    event_a: u64,
    event_b: u64,
    // ── Sigmoid + correction-bias routing (DeepSeek-V3 / MiniMax-M2 style) ──
    /// Device pointer to `[num_experts]` correction bias. Populated from
    /// `MoeWeights.correction_bias` in `new()` when the loader sets it.
    /// `None` = Atlas's default softmax path. When `Some`, every top-k
    /// dispatch site branches to `moe_topk_sigmoid` with this bias arg.
    correction_bias_dev: Option<DevicePtr>,
    /// Handle to `moe_topk_sigmoid` kernel. Lazy-loaded in `new()` even
    /// when bias is `None` (harmless if kernel isn't used).
    moe_topk_sigmoid_k: KernelHandle,
    /// Batched variant for prefill / MTP-verify (one block per token).
    /// Loaded via `try_kernel` — returns KernelHandle(0) on models whose
    /// KERNEL.toml doesn't register the sigmoid kernels (e.g. Mistral).
    /// Never dispatched on those paths because `correction_bias_dev` is
    /// `None` there.
    moe_topk_sigmoid_batched_k: KernelHandle,
    // FP8 fused MoE kernels (used when experts are FP8)
    moe_expert_gate_up_shared_fp8: KernelHandle,
    moe_expert_silu_down_shared_fp8: KernelHandle,
    // FP8 batch2/3 fused MoE kernels (for MTP K=2/K=3 verify)
    moe_expert_gate_up_shared_fp8_batch2: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch2: KernelHandle,
    moe_weighted_sum_blend_fp8_batch2: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch3: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch3: KernelHandle,
    moe_weighted_sum_blend_fp8_batch3: KernelHandle,
    // THE routed-expert FP8 grouped GEMM for sorted MoE prefill: grid-compaction
    // (persistent 96-CTA grid over a COMPACTED (expert, m_tile, n_tile) work-list
    // built by `moe_build_tile_worklist`). Handle may be 0 on images that don't
    // ship the kernel.
    moe_fp8_grouped_gemm_k: KernelHandle,
    // Builds the grouped-GEMM work-list (moe_build_tile_worklist, module "moe").
    // Launched on the SAME stream as the grouped GEMM (read-after-write of
    // total_tiles). Handle may be 0 on older images.
    moe_build_tile_worklist_k: KernelHandle,
    // W8A8 + FP32 epilogue MoE GEMM (vLLM-equivalent). Opt-in via
    // ATLAS_FP8_W8A8=1. Requires per-token-quanted A_fp8 + a_scale.
    moe_w8a8_grouped_gemm_k: KernelHandle,
    per_token_group_quant_fp8_k: KernelHandle,
    // Dense W8A8 (same kernel used by attention QKV/O proj) for shared-expert path.
    fp8_gemm_t_blockscaled_k: KernelHandle,
    // BF16 grouped GEMM — for FP8-source models dequanted to BF16 at load.
    // Activates the high-precision MoE path that closes the per-layer
    // 0.989 FP8 cosine ceiling. Handle may be 0 on images that don't ship
    // the kernel; dispatch site is gated on Some(bf16_*_weight_ptrs).
    moe_bf16_grouped_gemm_k: KernelHandle,
    // Fused BF16 decode kernels (mirror moe_expert_*_shared_fp8 layout).
    moe_expert_gate_up_shared_bf16_k: KernelHandle,
    moe_expert_silu_down_shared_bf16_k: KernelHandle,
    // Fused BF16 K=2 batch kernels for MTP verify (mirror the FP8 batch2 layout).
    // Handle may be 0 on images that don't ship the kernel; the K=2 BF16
    // dispatch site is gated on this being non-null and falls back to the
    // per-token batched path otherwise.
    moe_expert_gate_up_shared_bf16_batch2_k: KernelHandle,
    moe_expert_silu_down_shared_bf16_batch2_k: KernelHandle,
    w8a16_gemm_k: KernelHandle,           // for shared expert FP8 prefill
    w8a16_gemm_pipelined_k: KernelHandle, // ATLAS_W8A16_PIPELINED shared-expert variant
    // Fused gate GEMV + topK softmax (saves 1 kernel launch per layer)
    moe_gate_topk_fused_k: KernelHandle,
    // FP8 expert pointer tables (None when experts are NVFP4)
    fp8_gate_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_up_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_down_weight_ptrs: Option<Fp8ExpertPtrTable>,
    // BF16 expert pointer tables — populated by the FP8-dequant-on-load
    // path. When Some, the routed-expert dispatch in `forward_prefill_fp8`
    // routes through `moe_bf16_grouped_gemm` instead of the FP8 grouped
    // GEMM, eliminating the per-layer FP8 quantization ceiling.
    bf16_gate_weight_ptrs: Option<DevicePtr>,
    bf16_up_weight_ptrs: Option<DevicePtr>,
    bf16_down_weight_ptrs: Option<DevicePtr>,
    // BF16 shared expert weights — direct device pointers for the
    // fused-decode dispatch. Mirrors `fp8_shared_expert` but with raw
    // BF16 pointers (no scale).
    bf16_shared_gate: Option<DevicePtr>,
    bf16_shared_up: Option<DevicePtr>,
    bf16_shared_down: Option<DevicePtr>,
    // FP8 shared expert weights (None when shared expert is NVFP4)
    fp8_shared_expert: Option<Fp8ExpertWeight>,
    /// FP4 down kernel handle (`moe_w4a16_down_t_k64_fp4`). `try_kernel` =>
    /// `KernelHandle(0)` on images lacking it; the FP4-down dispatch checks this
    /// handle != 0, `down_fp4` is set, and the shared `down_ptrs_t` table is present.
    pub(crate) moe_down_t_k64_fp4: KernelHandle,
    /// `moe_permute_tokens` gather kernel — only needed by the FP4 escape-hatch
    /// (which consumes expert-sorted contiguous rows, unlike the FP8 fused
    /// kernel that gathers via `sorted_token_ids` internally). `try_kernel`
    /// (handle may be 0 on images lacking it). Now unused — the CUTLASS grouped
    /// path fuses the gather into its A-pack — kept for potential reuse.
    #[allow(dead_code)]
    pub(crate) moe_permute_tokens_k: KernelHandle,
    // Phase 2.7 Tier C — Frankenstein dispatch flag.
    // True when this layer's index is in `config.dflash_capture_layers`.
    // When the env var `ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL=1` is set,
    // `forward()` (single-token decode) will route through `forward_prefill`
    // (tensor-core grouped GEMM kernel) on this layer only, so the captured
    // hidden states use a different numerical recipe than the scalar GEMV
    // path. Used to test whether the kernel choice is the dominant cause
    // of low DFlash drafter acceptance on FP4/FP8 targets.
    pub is_dflash_capture_layer: bool,
}

impl MoeLayer {
    /// ARM-2 Phase-K routed-expert kernel-handle select. Returns the E8M0
    /// variant when the routed experts are native MXFP4 (`Mxfp4E8m0`), else the
    /// NVFP4 handle. Panics if E8M0 is selected but the `_e8m0` kernel is
    /// absent from this target (`try_kernel` gave 0) — that means a native
    /// checkpoint reached a build that never compiled the variant, which must
    /// be loud, not silent NVFP4-on-E8M0 garbage (the straggler net).
    #[inline]
    fn e8m0_or(
        &self,
        nvfp4: spark_runtime::gpu::KernelHandle,
        e8m0: spark_runtime::gpu::KernelHandle,
        site: &str,
    ) -> spark_runtime::gpu::KernelHandle {
        if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
            assert!(
                e8m0.0 != 0,
                "ARM-2 Phase-K: routed experts tagged Mxfp4E8m0 at {site}, but the \
                 _e8m0 kernel handle is unresolved (not compiled into this target)."
            );
            e8m0
        } else {
            nvfp4
        }
    }
}

// ── Sub-files (split for ≤500 LoC) ────────────────────────────────────────
mod dump;
mod forward;
mod forward_atomic_c4;
mod forward_batched;
mod forward_ep;
mod forward_k2;
mod forward_k3;
mod forward_phase;
mod forward_prefill;
mod forward_prefill_bf16;
mod forward_prefill_fp8;
mod forward_prefill_phase;
mod forward_prefill_routed;
mod forward_token_major;
mod helpers_a;
mod helpers_b;
mod helpers_c;
mod init;
#[cfg(test)]
mod mod_tests;
mod ptr_table_build;
mod union_stats;
pub(crate) use ptr_table_build::*;
