// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3 attention struct definitions: `MlaWeights` (latent attention
//! 2-step decode) and `Qwen3AttentionLayer` (full attention layer).

use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

use crate::layers::FfnComponent;
use crate::layers::fp8_calibration::Fp8KvCalibration;
use crate::weight_map::{AttentionWeights, DenseWeight, QuantWeight, QuantizedWeight};

/// MLA (Multi-head Latent Attention) weight components for 2-step decode.
///
/// Instead of a single Q GEMV: `input × Q_expanded → Q[n_heads*hd]`,
/// MLA does: `input × wq_a → latent[q_lora]` → `norm` → `latent × wq_b → Q`.
/// This preserves the latent normalization that's critical for output quality.
pub struct MlaWeights {
    pub wq_a: DenseWeight, // [q_lora, h] — Q down-projection (BF16)
    pub wq_a_nvfp4: Option<QuantizedWeight>, // NVFP4 for fast decode
    pub wq_b: DenseWeight, // [n_heads*hd, q_lora] — Q up-projection (BF16)
    pub wq_b_nvfp4: Option<QuantizedWeight>, // NVFP4 for fast decode
    pub q_a_norm: DenseWeight, // [q_lora] — RMS norm weight
    pub wkv_a: DenseWeight, // [kv_lora, h] — KV down-projection (BF16)
    pub wkv_a_nvfp4: Option<QuantizedWeight>, // NVFP4 for fast decode
    pub wkv_b: DenseWeight, // [n_kv*(nope+v), kv_lora] — KV up-projection (BF16)
    pub kv_a_norm: DenseWeight, // [kv_lora] — RMS norm weight
    pub wkv_a_rope: DenseWeight, // [rope, h] — K RoPE projection (BF16)
    /// Merged wkv_a + wkv_a_rope for prefill: [kv_lora+rope, h] — single GEMM replaces 2
    pub wkv_a_merged: DenseWeight,
    pub wo: DenseWeight, // [h, n_heads*v_dim] — O projection BF16 (for prefill accuracy)
    pub wo_nvfp4: Option<QuantizedWeight>, // O projection NVFP4 (for fast decode GEMV)
    /// Absorbed MLA weights for decode (avoid full K/V expansion, preserve precision).
    /// W_UK_T: [n_heads, nope, kv_lora] — Q_nope absorption: Q_absorbed = Q_nope @ W_UK_T
    pub w_uk_t: DenseWeight,
    /// W_UV: [n_heads, kv_lora, v_dim] — V extraction: v_out = attn_latent @ W_UV
    pub w_uv: DenseWeight,
    /// Q rope projection: wq_b_rope[nq*rope, q_lora] — Q_rope = wq_b_rope @ Q_latent
    /// Extracted from wq_b rows [n*hd+nope .. n*hd+nope+rope] for each head.
    pub wq_b_rope: DenseWeight,
    /// Fused Q absorption: `W_QK_absorbed[nq*kv_lora, q_lora]` — Q_absorbed = W_QK @ Q_latent
    /// Precomputed as: `W_QK[n, lkv, l] = sum_p wq_b_nope[n, p, l] * W_UK[n, p, lkv]`
    /// Enables single GEMV: `Q_absorbed[nq*kv_lora] = W_QK[nq*kv_lora, q_lora] @ Q_latent[q_lora]`
    pub w_qk_absorbed: DenseWeight,
    /// Block-diagonal W_UK for prefill batched GEMM: [nq*kv_lora, nq*nope]
    /// Single GEMM replaces 32*N per-head GEMV calls for Q absorption in prefill.
    pub w_uk_block_diag: DenseWeight,
    /// Block-diagonal W_UV for prefill batched GEMM: [nq*v_dim, nq*kv_lora]
    /// Single GEMM replaces 32*N per-head GEMV calls for V extraction in prefill.
    pub w_uv_block_diag: DenseWeight,
    /// Precomputed YaRN inv_freq table [rotary_dim/2] FP32 on GPU.
    /// NULL = use standard theta computation in the RoPE kernel.
    pub yarn_inv_freq: spark_runtime::gpu::DevicePtr,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_dim: usize,
}

/// Qwen3-Next full attention layer (12 of 48 layers).
#[allow(dead_code)]
pub struct Qwen3AttentionLayer {
    pub(super) input_norm: DenseWeight,
    pub(crate) attn: AttentionWeights,
    pub(super) post_attn_norm: DenseWeight,
    pub(super) ffn: FfnComponent,
    pub(super) attn_layer_idx: usize,
    /// Whether Q projection includes an output gate (Q+Gate interleaved).
    /// When true, q_proj output is 2× q_dim; attn output is gated by sigmoid.
    /// When false (e.g. Qwen3-VL), q_proj output is q_dim; no gating applied.
    pub(super) gated: bool,
    /// Whether this layer should apply MRoPE-interleaved instead of scalar
    /// RoPE. Set when `config.mrope_interleaved = true` (Qwen3.6).
    pub(crate) mrope_interleaved: bool,
    /// Per-layer dimension overrides for heterogeneous models (Gemma-4).
    pub(crate) head_dim_override: Option<usize>,
    pub(crate) num_q_heads_override: Option<usize>,
    pub(crate) num_kv_heads_override: Option<usize>,
    /// Per-layer sliding-window size for Gemma-4 hybrid attention.
    pub(crate) sliding_window: Option<u32>,
    /// Per-layer RoPE overrides for heterogeneous models (Gemma-4).
    pub(crate) rope_theta_override: Option<f32>,
    pub(crate) rotary_dim_override: Option<u32>,
    /// Proportional RoPE (Gemma-4 full-attention).
    pub(crate) rope_proportional: bool,
    /// Per-layer attention scale override (Gemma-4: 1.0 because QK-norm
    /// handles scaling). When None, uses the standard 1/sqrt(head_dim).
    pub(crate) attn_scale_override: Option<f32>,
    /// K=V mode: V comes from raw K projection output (no separate v_proj).
    pub(crate) k_eq_v: bool,
    /// Ones-filled BF16 weight buffer for the pure-RMSNorm v_norm path.
    pub(crate) v_norm_weight: Option<DenseWeight>,
    /// Post-attention output norm (Gemma-4).
    pub(crate) post_attn_out_norm: Option<DenseWeight>,
    /// Post-FFN output norm (Gemma-4).
    pub(crate) post_ffn_out_norm: Option<DenseWeight>,
    /// Per-layer scalar (Gemma-4): hidden_states *= layer_scalar at end of forward.
    pub(crate) layer_scalar: Option<f32>,
    /// Secondary FFN (Gemma-4 26B MoE): runs in parallel with primary FFN (dense).
    pub(crate) moe_ffn: Option<FfnComponent>,
    /// Pre-norm for MoE input (pre_feedforward_layernorm_2).
    pub(crate) pre_moe_norm: Option<DenseWeight>,
    /// Post-norm for MoE output (post_feedforward_layernorm_2).
    pub(crate) post_moe_out_norm: Option<DenseWeight>,
    /// Post-norm for dense FFN output only (post_feedforward_layernorm_1).
    pub(crate) post_dense_ffn_norm: Option<DenseWeight>,
    pub(super) kv_dtype: KvCacheDtype,
    /// Turbo4 sparse-V pruning threshold (0.0 = disabled).
    pub(super) sparse_v_threshold: f32,
    // ── Decode weights (QuantWeight enum: Nvfp4 | Fp8 | Dense) ──
    pub(super) q_weight: Option<QuantWeight>,
    pub(super) k_weight: Option<QuantWeight>,
    pub(super) v_weight: Option<QuantWeight>,
    pub(super) o_weight: Option<QuantWeight>,
    /// BF16 dense fallback for the output projection. When `Some`, the
    /// decode/prefill o_proj GEMV uses this BF16 weight instead of the
    /// NVFP4 path (`attn.o_proj`). Used by Gemma-4 dense which honors
    /// Nvidia ModelOpt's official ignore list.
    pub(super) o_dense_bf16: Option<DenseWeight>,
    // ── MLA (Multi-head Latent Attention) — 2-step decode ──
    pub(crate) mla: Option<MlaWeights>,
    // ── Transposed weights for prefill GEMM ──
    pub(super) q_nvfp4_t: Option<QuantizedWeight>,
    pub(super) k_nvfp4_t: Option<QuantizedWeight>,
    pub(super) v_nvfp4_t: Option<QuantizedWeight>,
    pub(super) o_nvfp4_t: Option<QuantizedWeight>,
    pub(super) q_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) k_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) v_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) o_fp8w_t: Option<crate::weight_map::Fp8WeightTransposed>,
    pub(super) w8a16_gemm_t_k: KernelHandle,
    pub(super) w8a16_gemm_t_pipelined_k: KernelHandle,
    // W8A8 + FP32 epilogue (vLLM-equivalent) — gated by ATLAS_FP8_W8A8=1.
    pub(super) per_token_group_quant_fp8_k: KernelHandle,
    pub(super) fp8_gemm_t_blockscaled_k: KernelHandle,
    // Kernels — decode (GEMV M=1)
    pub(super) rms_norm_k: KernelHandle,
    pub(super) rms_norm_residual_k: KernelHandle,
    /// Gemma-4 FP32-input rms_norm (absolute formula).
    pub(super) rms_norm_f32_in_k: KernelHandle,
    pub(super) dense_gemv_k: KernelHandle,
    pub(super) w4a16_gemv_k: KernelHandle,
    pub(super) w8a16_gemv_k: KernelHandle,
    pub(super) w8a16_gemm_k: KernelHandle,
    pub(super) w8a16_gemm_pipelined_k: KernelHandle,
    pub(super) w4a16_gemv_dual_k: KernelHandle,
    pub(super) rope_k: KernelHandle,
    /// MRoPE-interleaved kernel.
    pub(super) rope_mrope_interleaved_k: KernelHandle,
    /// YaRN RoPE kernel using pre-computed inv_freq table (Mistral, etc.)
    pub(super) rope_yarn_k: KernelHandle,
    /// Proportional RoPE kernel (Gemma-4 full-attention layers).
    pub(super) rope_proportional_k: KernelHandle,
    pub(super) reshape_cache_k: KernelHandle,
    /// Fused k_norm + RoPE + paged BF16 cache write — eliminates two
    /// intermediate BF16 rounding steps that cause the documented L35-L39
    /// cliff in chunked-prefill BF16 KV mode (memory:
    /// `project_qwen36_phase2b_softmax_expf.md`).
    pub(super) fused_k_norm_rope_cache_write_bf16_k: KernelHandle,
    /// MRoPE-interleaved variant of the above. Same precision regime.
    /// Dispatched when `mrope_interleaved` is true.
    pub(super) fused_k_norm_rope_mrope_cache_write_bf16_k: KernelHandle,
    /// V-only paged cache write. Used alongside the fused K-path so the
    /// K side of the cache stays single-rounded.
    pub(super) reshape_and_cache_flash_v_only_k: KernelHandle,
    /// WHT kernel for turbo KV cache.
    pub(super) wht_bf16_k: KernelHandle,
    /// Inverse WHT. With TQ_PLUS_SIGNS off this aliases the forward kernel
    /// (plain WHT is self-inverse); with TQ+ signs the inverse reverses the
    /// signs1/signs2 order, which is required because (S2·H·S1)·(S2·H·S1) ≠ I.
    pub(super) wht_bf16_k_inv: KernelHandle,
    /// InnerQ application kernels (Q pre-WHT scale_inv, K post-WHT scale).
    /// Returns 0 handle when InnerQ kernel module isn't loaded — caller should
    /// guard launches with `.0 != 0`.
    pub(super) innerq_apply_q_k: KernelHandle,
    pub(super) innerq_apply_k_k: KernelHandle,
    pub(super) paged_decode_k: KernelHandle,
    /// HDIM=512 paged decode kernel for Gemma-4 full-attention layers
    pub(super) paged_decode_512_k: KernelHandle,
    /// MLA absorbed paged decode kernel (HDIM=320).
    pub(super) paged_decode_mla_k: KernelHandle,
    /// MLA batched GEMV for Q absorption and V extraction.
    pub(super) mla_batched_gemv_k: KernelHandle,
    /// MLA fused kernels — decode.
    pub(super) mla_q_rope_scatter_k: KernelHandle,
    pub(super) mla_q_rope_writeback_k: KernelHandle,
    pub(super) mla_cache_assemble_k: KernelHandle,
    /// MLA fused kernels — prefill.
    pub(super) mla_q_rope_extract_batched_k: KernelHandle,
    pub(super) mla_q_rope_writeback_batched_k: KernelHandle,
    pub(super) mla_kv_assemble_batched_k: KernelHandle,
    pub(super) mla_cache_assemble_batched_k: KernelHandle,
    /// MLA absorbed prefill flash attention (HDIM=320, GQA 32:1)
    pub(super) prefill_attn_mla320_k: KernelHandle,
    /// Grouped GEMM for MLA Q absorption + V extraction.
    pub(super) grouped_gemm_mla_k: KernelHandle,
    /// Q_final assembly: [absorbed|rope] per head.
    pub(super) mla_q_final_assemble_k: KernelHandle,
    /// Fused MLA prefill: Q_absorb + attention + V_extract in one kernel.
    pub(super) mla_fused_prefill_k: KernelHandle,
    /// Split-K GEMM for skinny prefill matrices (M < 64).
    pub(super) gemm_splitk_partial_k: KernelHandle,
    pub(super) gemm_splitk_reduce_k: KernelHandle,
    /// Tensor-core BF16 GEMM (m16n8k16 MMA).
    pub(super) dense_gemm_tc_k: KernelHandle,
    pub(super) paged_decode_splitk_k: Option<KernelHandle>,
    pub(super) paged_decode_reduce_k: Option<KernelHandle>,
    pub(super) residual_add_k: KernelHandle,
    pub(super) sigmoid_gate_mul_k: KernelHandle,
    pub(super) deinterleave_qg_k: KernelHandle,
    pub(super) w4a16_gemv_qg_k: KernelHandle,
    pub(super) residual_add_rms_norm_k: KernelHandle,
    // Kernels — batch2 (K=2 verify)
    pub(super) w4a16_gemv_qg_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_dual_batch2_k: KernelHandle,
    pub(super) w4a16_gemv_batch2_k: KernelHandle,
    // Kernels — batch3 (K=3 verify)
    pub(super) w4a16_gemv_qg_batch3_k: KernelHandle,
    pub(super) w4a16_gemv_dual_batch3_k: KernelHandle,
    pub(super) w4a16_gemv_batch3_k: KernelHandle,
    // Kernels — prefill (GEMM M=N + Flash Attention)
    pub(super) w4a16_gemm_k: KernelHandle,
    pub(super) w4a16_gemm_t_k: KernelHandle,
    pub(super) w4a16_gemm_t_k64_k: KernelHandle,
    pub(super) w4a16_gemm_t_m128_k: KernelHandle,
    /// MiniMax-only shadow kernel.
    pub(super) w4a16_gemm_t_m128_v2_k: KernelHandle,
    /// v3 variant: K_STEP=64.
    pub(super) w4a16_gemm_t_m128_v3_k: KernelHandle,
    pub(super) dense_gemm_k: KernelHandle,
    pub(super) prefill_attn_k: KernelHandle,
    /// HDIM=512 contiguous prefill for Gemma-4 full-attention layers
    pub(super) prefill_attn_512_k: KernelHandle,
    /// HDIM=512 paged prefill (BF16 KV) for Gemma-4 chunked long-context prefill
    pub(super) prefill_attn_paged_512_k: KernelHandle,
    pub(super) prefill_attn_64_k: KernelHandle,
    pub(super) prefill_attn_paged_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4_k: KernelHandle,
    // BR=64 variants for long-context prefill (q_len >= 256)
    pub(super) prefill_attn_paged_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_64_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo2_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo3_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo8_64_k: KernelHandle,
    // ── TurboQuant+ asymmetric BR=64 prefill kernels ──
    // Combined-dtype kernels that read K and V with different on-disk layouts.
    // Currently: Bf16K + Turbo3V (safer-asym variant — K kept at bf16 precision,
    // V aggressively compressed to 3-bit Lloyd-Max + FP8 group scale).
    pub(super) prefill_attn_paged_bf16k_turbo3v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_bf16k_turbo4v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_bf16k_turbo2v_64_k: KernelHandle,
    // Fp8K + TurboNV variants — same shape as bf16k_turbo*v_64 but threads
    // the FP8 K-side per-tensor `k_scale` through to the dequant in
    // LOAD_K_TILE. Targets FP8-attention models (Qwen3.6-35B-FP8 etc.).
    pub(super) prefill_attn_paged_fp8k_turbo3v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8k_turbo4v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8k_turbo2v_64_k: KernelHandle,
    // Both-sides-quantized TurboQuant+ asym (K and V both turbo, separate
    // pool strides). K-side WHT bookend + Q WHT both fire because K is turbo.
    pub(super) prefill_attn_paged_turbo4k_turbo3v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo4k_turbo8v_64_k: KernelHandle,
    pub(super) prefill_attn_paged_turbo3k_turbo8v_64_k: KernelHandle,
    // ── Q12 Phase 3: same-chunk-len batched paged-prefill kernels ──
    // Each takes `const int* const* block_table_ptrs` + per-batch Q/O
    // offsets. Used by `Qwen3AttentionLayer::prefill_batched` when N≥2
    // streams share the same chunk_len. Null on targets that don't
    // carry the corresponding kernel (e.g. CPU backend).
    pub(super) prefill_attn_paged_batched_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_batched_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_batched_k: KernelHandle,
    pub(super) prefill_attn_paged_batched_64_k: KernelHandle,
    pub(super) prefill_attn_paged_fp8_batched_64_k: KernelHandle,
    pub(super) prefill_attn_paged_nvfp4_batched_64_k: KernelHandle,
    // Batched prefill kernels
    pub(super) deinterleave_qg_split_k: KernelHandle,
    pub(super) deinterleave_qg_split_qnorm_k: KernelHandle,
    pub(super) sigmoid_gate_mul_batched_k: KernelHandle,
    // Pre-dequanted FP8 weights for zero-overhead prefill GEMMs
    pub(super) q_fp8: Option<DevicePtr>,
    pub(super) k_fp8: Option<DevicePtr>,
    pub(super) v_fp8: Option<DevicePtr>,
    pub(super) o_fp8: Option<DevicePtr>,
    pub(super) fp8_gemm_k: KernelHandle,
    // FP8×FP8 GEMM
    pub(super) bf16_to_fp8_k: KernelHandle,
    pub(super) fp8_fp8_gemm_k: KernelHandle,
    // M128 variants
    pub(super) fp8_gemm_t_m128_k: KernelHandle,
    pub(super) fp8_fp8_gemm_t_m128_k: KernelHandle,
    /// Online FP8 KV scale calibration.
    pub(super) fp8_calibration: Option<Fp8KvCalibration>,
}
