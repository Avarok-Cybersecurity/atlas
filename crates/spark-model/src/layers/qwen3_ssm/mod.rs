// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3-Next SSM (Gated Delta Net) layer implementing TransformerLayer.
//!
//! Corrected pipeline matching the HuggingFace reference implementation:
//!   1. QKVZ projection (interleaved output)
//!   2. Deinterleave QKVZ → sequential [Q | K | V | Z]
//!   3. BA projection (interleaved output)
//!   4. Compute GDN gates: gate = exp(-A * softplus(alpha + dt_bias)), beta = sigmoid(b)
//!   5. Conv1d update on [Q | K | V] concatenated (d_inner=8192)
//!   6. Split conv output → Q', K', V'
//!   7. GDN decode (Q', K', V', gate, beta) — kernel handles GQA internally
//!   8. Gated RMS norm (GDN output, Z gate)
//!   9. Output projection [value_dim → hidden_size]
//!  10. MoE FFN

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState};
use crate::layers::FfnComponent;
use crate::layers::ops;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::{DenseWeight, Fp8Weight, QuantizedWeight, SsmWeights};

/// Qwen3-Next SSM/GDN layer (36 of 48 layers).
///
/// Supports two QKVZ projection modes:
/// - **Interleaved** (80B): `w4a16_gemv_qkvz` or GEMV + `deinterleave_qkvz`
/// - **Sequential** (3.5-35B): plain GEMV → `[Q|K|V|Z]` already in order
#[allow(dead_code)]
pub struct Qwen3SsmLayer {
    /// mHC weights when the model carries a `hc_mult`-wide highway; see `hc`.
    pub(crate) hc: Option<crate::layers::qwen3_attention::HcWeights>,
    /// PLE n-gram injection. `Some` on exactly ONE model layer (layer 1 on
    /// this checkpoint); it runs at the TOP of the mHC forward, before this
    /// layer's own hyper-connection, matching the reference's
    /// `hidden_states = hidden_states + self.ple(...)`.
    pub(crate) ple: Option<crate::layers::ple::PleLayer>,
    /// mHC kernel handles. Resolved only when `config.hc_mult > 0`, so a
    /// plain GDN model issues no lookup and leaves no row in the startup
    /// audit. See `qwen3_attention::init_arch_gates`.
    pub(super) hc_pre_k: KernelHandle,
    pub(super) hc_post_k: KernelHandle,
    /// Seeds the highway on MODEL layer 0 — which on a 3:1 GDN:attention
    /// interleave is a GDN layer, so this side owns the expand that the
    /// attention side used to do.
    pub(super) hc_expand_k: KernelHandle,
    input_norm: DenseWeight,
    ssm: SsmWeights,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    /// GDN `out_proj` LoRA delta for this layer, with the kernels to apply it.
    /// `None` on every base serve, which keeps the base path byte-identical.
    lora_out_proj: Option<(
        crate::layers::ops::lora_delta::LoraPair,
        crate::layers::ops::lora_delta::LoraKernels,
    )>,
    // NVFP4-quantized QKVZ weight (quarters bandwidth vs BF16)
    qkvz_nvfp4: Option<QuantizedWeight>,
    // Transposed [K/2, N] copy for coalesced w4a16_gemm reads (prefill)
    qkvz_nvfp4_t: Option<QuantizedWeight>,
    // Transposed out_proj for prefill GEMM
    out_proj_nvfp4_t: Option<QuantizedWeight>,
    // BF16 out_proj for models where SSM weights are not pre-quantized
    pub out_proj_dense: Option<DenseWeight>,
    /// Natively-served EXL3 GDN family (`ATLAS_EXL3_NATIVE_DENSE=1`): the
    /// packed `in_proj_qkv` / `in_proj_z` / `out_proj` trellis weights.
    /// When `Some`, `ssm.in_proj_qkvz`, `ssm.out_proj` and `out_proj_dense`
    /// are NULL/None (the FP8-native precedent) and the projections dispatch
    /// through the cooperative `exl3_gemv`/`exl3_gemm` arms — which also
    /// vetoes decode-graph capture (`exl3_graph_veto`). Installed by the
    /// loader via [`Self::set_exl3_gdn_weights`]; `None` on every other
    /// serve keeps the existing arms byte-identical.
    exl3_gdn: Option<crate::layers::exl3_dense::Exl3GdnWeights>,
    // FP8 E4M3 checkpoint weights for native FP8 serving (w8a16_gemv LUT kernel)
    qkvz_fp8w: Option<Fp8Weight>,
    out_proj_fp8w: Option<Fp8Weight>,
    /// PER-ROW FP8 (`Fp8PerRow`) for PREFILL ONLY, from mixed-precision
    /// compressed-tensors checkpoints (`ATLAS_FP8_ROWWISE=1`).
    ///
    /// Separate fields rather than reusing `qkvz_fp8w`/`out_proj_fp8w`, and
    /// that separation is the safety property: those two are read by
    /// `w8a16_gemv` in `ssm_forward.rs` and `trait_decode_batched.rs`, which
    /// index the scale as a `[N/128, K/128]` block grid. A per-row buffer is
    /// SMALLER than that index space, so it would not fault — it would return
    /// plausible garbage. Only the row-wise cuBLASLt prefill arm reads these;
    /// decode keeps the NVFP4 copy.
    qkvz_fp8w_rowwise: Option<Fp8Weight>,
    out_proj_fp8w_rowwise: Option<Fp8Weight>,
    /// Tier-1c keep-packed ternary Q2_0 fused in_proj_qkvz (`ATLAS_GGUF_NATIVE_Q2`).
    /// [Q|K|V|Z] rows byte-concatenated from packed `in_proj_qkv` (V-region
    /// row-permuted) + `in_proj_z` (row-permuted) at load, so the 2-bit weight is
    /// HF-correct. `out_proj` stays NVFP4 (column reorder not packed-permutable here).
    qkvz_q2: Option<crate::weight_map::PackedQ2Weight>,
    /// Q2_0 kernels for the packed qkvz: `gemv` = `q2_0_gemv_vec` decode; `dequant`
    /// = load-time packed→BF16 for the transient-dequant prefill fallback;
    /// `mmq_{nc,wc}` = Tier-2 keep-packed tensor-core MMQ prefill (`KernelHandle(0)`
    /// → fallback); `q4k_quant_act` = shared q8_1 activation quantizer.
    q2_0_gemv_k: KernelHandle,
    dequant_q2_0_gn_k: KernelHandle,
    q2_0_mmq_nc_k: KernelHandle,
    q2_0_mmq_wc_k: KernelHandle,
    q4k_quant_act_k: KernelHandle,
    /// When true, QKVZ projection output is already sequential [Q|K|V|Z].
    /// Skips the deinterleave kernel (used by Qwen3.5 where QKV+Z are
    /// concatenated at load time rather than interleaved per-group).
    sequential_qkvz: bool,
    /// Streaming multiprocessor count, read from the driver ONCE at
    /// construction (`GpuBackend::sm_count`). `ms_proj_gemm` needs to know how
    /// wide the machine is to decide whether halving the CTA rows is a saving
    /// or an under-fill; a compiled-in constant would be wrong on every part
    /// that is not the one it was tuned on.
    sm_count: u32,
    // Kernels — decode path (single-token GEMV)
    rms_norm_residual_k: KernelHandle,
    gated_rms_norm_k: KernelHandle,
    gated_rms_norm_f32_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    /// K=2 verify: batched (M=2) BF16 GDN in_proj_qkvz — one weight pass for
    /// both verify tokens instead of two M=1 `dense_gemv` reads.
    dense_gemv_batch2_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    /// Single-warp `w4a16_gemv_sw`. `KernelHandle(0)` on miss → base GEMV.
    w4a16_gemv_sw_k: KernelHandle,
    w8a16_gemv_k: KernelHandle,
    w4a16_gemv_qkvz_k: KernelHandle,
    deinterleave_k: KernelHandle,
    conv1d_k: KernelHandle,
    conv1d_l2norm_k: KernelHandle,
    conv1d_l2norm_f32_k: KernelHandle,
    /// `conv1d_l2norm_f32_k` with explicit input/output row strides, letting
    /// the concurrent-decode path batch all N sequences into one launch.
    /// `KernelHandle(0)` on kernel sets that predate it — the multi-seq path
    /// then falls back to the per-sequence conv loop.
    conv1d_l2norm_f32_strided_k: KernelHandle,
    gdn_k: KernelHandle,
    gdn_f32_k: KernelHandle,
    gdn_f32_norm_k: KernelHandle,
    gdn_f32_conv_norm_k: KernelHandle,
    gdn_f32_strided_k: KernelHandle,
    gdn_f32_strided_norm_k: KernelHandle,
    /// Half-width register retention (k_dim==v_dim==128): retains the first 64 H
    /// columns so the update re-reads only the rest (2R+1W -> 1.5R+1W).
    gdn_f32_strided_norm_half_k: KernelHandle,
    /// SRAM-staged full retention (k_dim==v_dim==128): the columns the register
    /// file cannot hold are staged in shared memory on the first pass instead of
    /// being re-read from H (1.5R+1W -> 1.0R+1W). Bit-identical to
    /// `gdn_f32_strided_norm_half_k` but measured throughput-NEUTRAL, so it is
    /// OPT-IN via `gdn_smem_stage_enabled()` (`ATLAS_GDN_SMEM_STAGE`).
    gdn_f32_strided_norm_smem_k: KernelHandle,
    /// FP16 h-state twin of `gdn_f32_strided_norm_half_k` (`ATLAS_SSM_H_FP16`).
    /// Additive: it never replaces the FP32 kernel, it is selected instead of
    /// it when the sequence's `SsmLayerState::h_is_f16` is set.
    gdn_f16_strided_norm_half_k: KernelHandle,
    /// FP16 h-state twin of `gdn_f32_norm_k` — the per-sequence arm the batched
    /// dispatch falls back to at n == 1 and whenever pool slots fragment out of
    /// slice order. Without it the FP16 pool would be read as FP32 on exactly
    /// those steps.
    gdn_f16_norm_k: KernelHandle,
    ba_gates_k: KernelHandle,
    residual_add_k: KernelHandle,
    l2_norm_k: KernelHandle,
    residual_add_rms_norm_k: KernelHandle,
    /// Dual-output (bf16 + f32) MoE-input norm for ATLAS_FP32_ROUTING. Zero if absent.
    residual_add_rms_norm_gatef32_k: KernelHandle,
    gated_rms_norm_prefill_k: KernelHandle,
    // Kernels — batched verification path (multi-token GEMM)
    w4a16_gemm_k: KernelHandle,
    w4a16_gemm_t_k: KernelHandle, // Transposed B layout [K/2, N] — K_STEP_T=32
    w4a16_gemm_t_k64_k: KernelHandle, // K64 variant: K_STEP_T=64, halves outer loop
    /// K64 with a 64-wide N tile: same math, 2x the CTAs. `KernelHandle(0)`
    /// when absent or killed by `ATLAS_NO_K64_N64`.
    w4a16_gemm_t_k64_n64_k: KernelHandle,
    w4a16_gemm_t_m128_k: KernelHandle, // M128 variant: 2 M-chunks per CTA, halves B re-reads
    w4a16_gemm_t_m128_v2_k: KernelHandle, // M128 8-warp pipelined (fast at small M; the FFN's kernel)
    w4a16_gemv_batch2_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    dense_gemm_pipelined_k: KernelHandle,
    gdn_prefill_k: KernelHandle,
    gdn_prefill_split_k: KernelHandle,
    gdn_prefill_split4_k: KernelHandle,
    gdn_prefill_persistent_k: KernelHandle,
    gdn_prefill_persistent_wy4_k: KernelHandle,
    /// Register-resident token-sequential warm-replay recurrence (H in regs, >=2
    /// CTA/SM, no barriers). Token-equal to WY4 (cosine 1.0), ~2.9x faster.
    /// DEFAULT-ON since 2026-07-25 (serve-validated: full MLPerf-edge e2e, wall
    /// −7.25%, BFCL identical); kill switch `ATLAS_NO_GDN_REGRESIDENT=1`.
    gdn_prefill_regresident_k: KernelHandle,
    /// FLA multi-kernel chunked prefill (baked default for 128-dim GDN): recompute_wu →
    /// chunk_delta_h_ksplit (k-split occupancy) → chunk_fwd_o. 1.75x vs wy4 @16k,
    /// token-equal (cos=1.0 vs scalar). Three handles; all must be non-null.
    gdn_prefill_fla_recompute_wu_k: KernelHandle,
    gdn_prefill_fla_chunk_delta_h_k: KernelHandle,
    /// Tensor-core / DV-block-split variant of the FLA chunk_delta_h spine
    /// (`gated_delta_rule_chunk_delta_h_tc_vblock`). Loaded by default but not
    /// yet wired into the prefill dispatch — the cos-gate validates it in
    /// isolation first. `allow(dead_code)` until the launch site reads it.
    #[allow(dead_code)]
    gdn_prefill_fla_chunk_delta_h_tc_vblock_k: KernelHandle,
    /// Warp-dense fused GDN state spine (`gated_delta_rule_chunk_delta_h_vtile`).
    /// 512 threads = 16 warps/CTA against ksplit's 8, with the SAME grid (one CTA
    /// per head) so `W`/`K` global loads are not duplicated — an ncu profile put
    /// ksplit at L2 60.6% / L1 56.2%, i.e. memory-pipeline bound, which is why the
    /// DV-split variants (which duplicate those loads) lost. Fusing the two
    /// per-chunk passes deletes ksplit's `duc[CHUNK]` register array, and that is
    /// what pays for the extra warps: 118 registers, no spills. Measured 2.15-2.18x
    /// vs ksplit at 2048/8192/16384 with cos=1.0000 (`gdn_chunk_shapetest`).
    gdn_prefill_fla_chunk_delta_h_fused_k: KernelHandle,
    /// TMA (`cp.async.bulk.tensor`) build of the state spine, behind
    /// `ATLAS_GDN_TMA=1`. `try_kernel` => 0 on images that lack it, and the
    /// launcher additionally refuses varlen and any head narrower than the
    /// compile-time tile — the descriptors encode that tile, and a mismatched
    /// shape loads the wrong columns without erroring.
    gdn_prefill_fla_chunk_delta_h_tma_k: KernelHandle,
    gdn_prefill_fla_chunk_fwd_o_k: KernelHandle,
    /// WY32 chunked prefill: processes 32 tokens per WY iteration with H in
    /// shared memory. ~30x faster than per-token for 14k+ sequences.
    gdn_prefill_wy32_k: KernelHandle,
    // ── Q12 Phase 2b: same-chunk-len batched GDN prefill kernels ──
    // Each takes `float* const* h_state_ptrs` plus stacked QKV/gate/beta/output.
    // Used by `Qwen3SsmLayer::prefill_batched` when N≥2 streams have matching
    // chunk_len. Null on targets that don't carry the corresponding kernel.
    gdn_prefill_wy32_batched_k: KernelHandle,
    gdn_prefill_persistent_batched_k: KernelHandle,
    gdn_prefill_persistent_wy4_batched_k: KernelHandle,
    gdn_prefill_split4_batched_k: KernelHandle,
    compute_gdn_gates_k: KernelHandle,
    ba_gates_prefill_k: KernelHandle,
    // Kernels — prefill (multi-token sequential)
    conv1d_prefill_k: KernelHandle,
    /// Token-parallel prefill conv1d (`causal_conv1d_update_prefill_tp`).
    conv1d_prefill_tp_k: KernelHandle,
    // Kernels — fused chunk2 path (2-token verification)
    gdn_chunk2_k: KernelHandle,
    conv1d_chunk2_k: KernelHandle,
    // Kernels — fused chunk3 path (3-token verification)
    gdn_chunk3_k: KernelHandle,
    w4a16_gemv_batch3_k: KernelHandle,
    // NVFP4 batched decode GEMV (multi-seq concurrency + chain verify):
    // the narrow batch{4,5,6,7,8} family plus batch16 (M<=16) — siblings of
    // w8a16_gemv_batch4/16 for the FP4 QKVZ + out_proj, so FP4 decode
    // amortizes the weight read at C=4..16 like FP8.
    w4a16_batchm: W4a16BatchmTiers,
    w4a16_gemv_batch16_k: KernelHandle,
    // Kernels — WY-chunkwise path (2-pass verification)
    gdn_wy2_k: KernelHandle,
    /// Register-resident wy2 twin (K=2 verify, the C=32 hot shape): Pass 2
    /// is served from the Pass 1 H read retained in registers
    /// (`__launch_bounds__(128,1)`, 128 floats/thread — the regresident
    /// prefill pattern), cutting the kernel's HBM state traffic from 2R+2W
    /// to 1R+2W. Byte-identical accumulation order to `gdn_wy2_k`
    /// (bitwise-asserted by gdn_wy_verify_microtest's parity leg).
    /// KernelHandle(0) when not linked (e.g. strix module sets). Selection +
    /// kd/vd==128 guard + width gate (n >= wy_resident_min_width(); the
    /// 1-block/SM kernel loses at narrow launches) live in `wy2_kernel`
    /// (trait_decode_batched_conv_gdn);
    /// kill switch ATLAS_NO_GDN_WY2_RESIDENT (PRESENCE — `=0` is NOT off).
    gdn_wy2_resident_k: KernelHandle,
    gdn_wy3_k: KernelHandle,
    /// Register-resident wy3 twin (K=3 verify — the 16:2 ladder rung's 3
    /// rows/seq shape, plus the 24:2/32:2 rungs of the 96-row envelope):
    /// Pass 2 served from the Pass 1 H read retained in registers, cutting
    /// HBM state traffic from 2R+3W to 1R+3W. Byte-identical accumulation
    /// order to `gdn_wy3_k` (bitwise-asserted by gdn_wy_verify_microtest's
    /// wy3 parity leg). KernelHandle(0) when not linked. Selection +
    /// kd/vd==128 guard + width gate (n >= wy_resident_min_width()) live in
    /// `wy3_kernel` (trait_decode_batched_conv_gdn);
    /// kill switch ATLAS_NO_GDN_WY3_RESIDENT (PRESENCE — `=0` is NOT off).
    gdn_wy3_resident_k: KernelHandle,
    gdn_wy4_k: KernelHandle,
    /// FP16 h-state twins of the five WY verify kernels above
    /// (`ATLAS_SSM_H_FP16` stage 2). Same launch contracts, same float
    /// expressions and accumulation orders as their FP32 parents — the h-state
    /// and its rollback intermediates are simply `__half` in memory, with the
    /// state rounded once per token boundary so a rollback checkpoint holds
    /// exactly the bits the forward chain carried.
    ///
    /// Stage 1 narrowed only the NON-speculative decode scan, so `--speculative`
    /// and the flag were mutually exclusive (preflight refused). These close
    /// that: with speculation on, the WY kernels are the only GDN h-state
    /// readers/writers in the step, so the rungs whose best config is spec-ON
    /// could not use FP16 at all.
    ///
    /// KernelHandle(0) when not linked. The selectors (`wy2_kernel`,
    /// `wy3_kernel`, and the K=4 sites) gate on `.0 != 0` and fall back to the
    /// FP32 parent — which is why preflight must independently refuse the flag
    /// when a reachable K has no twin, since that fallback would read an FP16
    /// pool through an FP32 kernel and produce fluent garbage.
    gdn_wy2_f16_k: KernelHandle,
    gdn_wy2_resident_f16_k: KernelHandle,
    gdn_wy3_f16_k: KernelHandle,
    gdn_wy3_resident_f16_k: KernelHandle,
    gdn_wy4_f16_k: KernelHandle,
    /// Stage-3 f16-SIZED pool (`--ssm-h-dtype f16-pool`): the two h-state
    /// width converters (`ssm_h_dtype.cu`). PREFILL uses them as a matched
    /// pair around its FP32 kernels — widen the narrow slot into the
    /// sequence's FP32 staging blob, run, narrow back — so unlike the
    /// decode-side one-shot conversion these launch once per SSM layer per
    /// prefill pass and are self-cancelling. A 0 handle is a hard error at
    /// the first prefill, never an FP32 fallback: an FP32 kernel writing a
    /// 2-byte-sized slot is an OOB write into the neighbouring slot.
    ssm_h_f16_to_f32_k: KernelHandle,
    ssm_h_f32_to_f16_k: KernelHandle,
    /// STAGE 1 fused K=2 MTP-verify epilogue: conv1d+L2norm ×2 and
    /// gated-RMS-norm ×2 each folded into a single launch. Dispatched only
    /// when the `ATLAS_GDN_FUSED_VERIFY` env flag is set (default OFF); the
    /// per-token path runs unchanged otherwise. Bit-identical (cos == 1.0).
    gdn_verify_fused_conv_k2_k: KernelHandle,
    gdn_verify_fused_norm_k2_k: KernelHandle,
    /// Fused generic-K verify conv1d+L2norm (one launch for all K positions,
    /// rollback snapshots written inline). Used by the K=17 DFlash verify arm;
    /// default ON when present, kill-switch `ATLAS_GDN_FUSED_CONV17=0`.
    /// NULL handle on targets lacking the .cu → per-token loop unchanged.
    gdn_verify_fused_conv_kn_k: KernelHandle,
    /// Batched twin (gridDim.y = n_seq) — batched spec decode. 0 when absent.
    gdn_verify_fused_conv_kn_batched_k: KernelHandle,
    /// Exact-verify `_snap` twins (issue #435 route (a)): the fused-norm
    /// decode kernels with an inline per-token h-state rollback snapshot, and
    /// the FP32-output fused verify conv. All OPTIONAL (model-shadow staged,
    /// currently qwen3.6-27b/nvfp4 only): a 0 handle makes the exact arm fall
    /// back to the parent kernel + `copy_d2d_async` snapshots — the same
    /// bits, more launches.
    gdn_f32_norm_snap_k: KernelHandle,
    gdn_f32_strided_norm_snap_k: KernelHandle,
    gdn_verify_fused_conv_kn_f32_k: KernelHandle,
    /// WY-Chunkwise K=17 GDN verify (DFlash γ+1). Only present in
    /// qwen3.6-35b-a3b's PTX module set; NULL handle for other targets,
    /// in which case decode_batched(K=17) falls through to the sequential
    /// per-token path.
    gdn_wy17_k: KernelHandle,
    /// WY-Chunkwise K∈{5..16} GDN verify (every chain-verify width between
    /// the dedicated wy4 and the DFlash wy17; K=9..16 added 2026-08-29 for
    /// the γ>8 window class, which previously fell to the sequential
    /// per-token loop — the measured γ10 tax). One K-templated source
    /// (`gated_delta_rule_wyn.cu`, gb10 common) instantiates wy5..wy16 with
    /// the same pool-layout intermediates contract as wy17. Index = K-5;
    /// NULL handles on targets lacking the module → sequential fallback.
    /// Kill-switch: `ATLAS_GDN_WYN=0` (default ON).
    gdn_wyn_k: [KernelHandle; 12],
    /// FP16 h-state twins of the wyN family (K=5..16), stage 2 of
    /// `ATLAS_SSM_H_FP16` — added 2026-08-29 (#812: the FP16 pool is the
    /// lever that lets MTP serve wide batch; DFlash was refused it for
    /// want of these twins). Same index contract (K-5); zero handles on
    /// targets lacking the module. Under the f16 pool a missing twin is a
    /// HARD ERROR at dispatch (never a silent FP32 fallback over FP16
    /// state). provenance-id: 526f6e616c6420522e205374657369616b
    gdn_wyn_f16_k: [KernelHandle; 12],
    // State allocation sizes (pre-computed from config)
    h_state_bytes: usize,
    conv_state_bytes: usize,
    // Pre-dequanted FP8 weights for zero-overhead prefill GEMMs
    qkvz_fp8: Option<DevicePtr>,
    out_proj_fp8: Option<DevicePtr>,
    fp8_gemm_k: KernelHandle,
    fp8_gemm_t_m128_k: KernelHandle, // M128: halves B re-reads for out_proj at ISL > 128
    // Block-scaled W8A16 prefill kernels (preferred over single-scale
    // fp8_gemm_n128 when block-scaled FP8 weights are available — matches
    // vLLM's per-128-block scale precision instead of single-scale).
    w8a16_gemm_k: KernelHandle,
    // Pipelined (cp.async) rewrite of w8a16_gemm: bit-identical, ~4.6× faster.
    // KernelHandle(0) when not linked into the image. Gated ON only when
    // ATLAS_W8A16_PIPELINED=1 (default OFF — production dispatch unchanged).
    w8a16_gemm_pipelined_k: KernelHandle,
    // M<=4 weight-streaming block-scaled FP8 GEMV. Replaces the M-padded
    // w8a16_gemm_pipelined for n<=4 batched decode (qkvz + out_proj): pipelined
    // pads M=4 to a 128-row MMA tile (32× compute over-provision, issue-bound);
    // this streams the weight once with 4 FP32 accumulators. Bit-identical per
    // row to w8a16_gemv. KernelHandle(0) when not linked.
    w8a16_gemv_batch4_k: KernelHandle,
    // M<=16 sibling of batch4 for high-concurrency decode (n=5..16): same
    // weight-streaming GEMV, avoids the M-padded MMA at C=8/16.
    w8a16_gemv_batch16_k: KernelHandle,
    w8a16_gemm_t_k: KernelHandle,
    // W8A8 + FP32 epilogue (vLLM-equivalent) prefill kernels.
    // `per_token_group_quant_fp8` produces FP8 activations + per-token-per-128
    // FP32 scale; `fp8_gemm_t_blockscaled` consumes both with FP8 MMA and
    // applies a_scale × b_scale in the FP32 epilogue. Gated behind
    // `ATLAS_FP8_W8A8=1` for staged rollout.
    per_token_group_quant_fp8_k: KernelHandle,
    fp8_gemm_t_blockscaled_k: KernelHandle,
}

// Kernel-selection helpers moved to `kernel_select.rs` (≤500 LoC split).

// ── Sub-files (split for ≤500 LoC) ────────────────────────────────────────
mod debug;
pub mod gdn_flags;
mod hc_ffn_dispatch;
mod init;
mod init_exl3;
mod init_fp8;
mod init_q2;
mod kernel_select;
mod lora;
mod ssm_forward;
pub(crate) mod ssm_h_fp16;
mod trait_decode;
mod trait_decode_batched;
pub(crate) mod trait_decode_batched_hc;
mod trait_decode_batched_conv_gdn;
mod trait_decode_batched_conv_gdn_exact;
mod trait_decode_batched_conv_gdn_multi;
mod trait_decode_batched_conv_gdn_multi_exact;
mod trait_decode_batched_conv_gdn_wyn;
mod trait_decode_hc;
mod trait_decode_multi_seq;
mod trait_layer;
mod trait_prefill;
mod trait_prefill_block;
mod trait_prefill_gdn;
mod trait_prefill_hc;
mod trait_prefill_helper;
mod trait_prefill_phase1;
mod trait_prefill_phase3;
mod trait_prefill_proj;
mod trait_prefill_recur;

pub use gdn_flags::{
    GdnFlags, MAX_F16_TWIN_DFLASH_GAMMA, MAX_F16_TWIN_K, default_dflash_gamma,
    gdn_fused_norm_enabled, ssm_batched_recurrent_enabled, ssm_h_dtype_bits,
    ssm_h_f16_pool_enabled, ssm_h_fp16_enabled, verify_exact_enabled,
    RowExactLeg, verify_row_exact_for_pass, verify_row_exact_leg, verify_row_exact_required,
};

// ── TransformerLayer impl (delegates to per-file inherent _inner methods) ──

#[cfg(test)]
mod tests;

#[path = "hc.rs"]
mod hc;
