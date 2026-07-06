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
    /// Native block-scaled FP8 weight (the checkpoint ships these projections as
    /// FP8-E4M3 + 128×128 block scales). Used by the decode GEMV (w8a16_gemv) so
    /// the hot path reads 1 byte/elem instead of the BF16-dequant's 2 — lossless
    /// (the in-kernel dequant keeps F32 precision before the BF16 activation MAC).
    pub wq_a_fp8: Option<crate::weight_map::Fp8Weight>,
    pub wq_b: DenseWeight, // [n_heads*hd, q_lora] — Q up-projection (BF16)
    pub wq_b_nvfp4: Option<QuantizedWeight>, // NVFP4 for fast decode
    pub wq_b_fp8: Option<crate::weight_map::Fp8Weight>,
    pub q_a_norm: DenseWeight,                // [q_lora] — RMS norm weight
    pub wkv_a: DenseWeight,                   // [kv_lora, h] — KV down-projection (BF16)
    pub wkv_a_nvfp4: Option<QuantizedWeight>, // NVFP4 for fast decode
    pub wkv_a_fp8: Option<crate::weight_map::Fp8Weight>,
    pub wkv_b: DenseWeight, // [n_kv*(nope+v), kv_lora] — KV up-projection (BF16)
    pub kv_a_norm: DenseWeight, // [kv_lora] — RMS norm weight
    pub wkv_a_rope: DenseWeight, // [rope, h] — K RoPE projection (BF16)
    /// Merged wkv_a + wkv_a_rope for prefill: [kv_lora+rope, h] — single GEMM replaces 2
    pub wkv_a_merged: DenseWeight,
    pub wo: DenseWeight, // [h, n_heads*v_dim] — O projection BF16 (for prefill accuracy)
    pub wo_nvfp4: Option<QuantizedWeight>, // O projection NVFP4 (for fast decode GEMV)
    /// Grouped low-rank O down-projection (wo_a → wo_b) for DeepSeek-V4-Flash.
    /// When `o_lora_rank > 0`, the decode/prefill paths use wo_a→wo_b instead of `wo`.
    pub wo_a: DenseWeight, // [o_lora_rank, n_heads*v_dim]
    pub wo_a_nvfp4: Option<QuantizedWeight>,
    /// Native block-scaled FP8 wo_a for the grouped decode O-projection. Sliced
    /// per o_group (block-diagonal) into w8a16_gemv calls.
    pub wo_a_fp8: Option<crate::weight_map::Fp8Weight>,
    pub wo_b: DenseWeight, // [h, o_lora_rank]
    pub wo_b_nvfp4: Option<QuantizedWeight>,
    pub wo_b_fp8: Option<crate::weight_map::Fp8Weight>,
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
    pub o_lora_rank: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_dim: usize,
    /// DeepSeek Sparse Attention compressor (CSA ratio-4 / HCA ratio-128).
    /// `None` for full-attention layers (`compress_ratios[L]` == 0).
    pub compressor: Option<CompressorWeights>,
    /// Per-head attention sink logit `[num_q_heads]` BF16 (DeepSeek-V4 s_aux).
    /// NULL if the checkpoint has no attn_sink for this layer.
    pub attn_sink: spark_runtime::gpu::DevicePtr,
}

/// DeepSeek-V4 compressed-attention compressor weights (one per compressed layer).
/// Produces `n_win = usable/ratio` compressed KV entries that are concatenated to
/// the raw sliding-window KV before core attention. CSA (ratio 4) uses a 2×ratio
/// overlap window (Ca/Cb); HCA (ratio 128) uses a single non-overlapping window.
#[derive(Debug, Clone, Copy)]
pub struct CompressorWeights {
    /// kv_proj: [proj_dim, hidden]. proj_dim = 2*head_dim (CSA) or head_dim (HCA).
    pub wkv: DenseWeight,
    /// gate_proj: same shape as wkv.
    pub wgate: DenseWeight,
    /// kv_norm weight `[head_dim]` — STANDARD RMSNorm (loaded via dense_minus_one).
    pub norm: DenseWeight,
    /// position_bias / ape: [ratio, proj_dim] BF16, added to the gate before softmax.
    pub ape: spark_runtime::gpu::DevicePtr,
    /// compress_rate for this layer (4 = CSA, 128 = HCA).
    pub ratio: usize,
    /// proj_dim of wkv/wgate output (2*head_dim for CSA, head_dim for HCA).
    pub proj_dim: usize,
    /// true = CSA (2×ratio overlap window); false = HCA (single window).
    pub is_csa: bool,
    /// 4b: persistent flat compressed-KV pool (decode reads it; inc-3 appends).
    /// Layout `[pool_blocks × hd_mla]` FP8-E4M3, each block = one rope'd `comp_k`
    /// entry quantized at the raw KV arm's scale (k_scale=1.0 for V4) so decode
    /// reads raw+compressed at one dtype/scale (single online softmax). Flat
    /// per-seq (V4 serves max_batch=1), NOT paged — mirrors the reference
    /// `Compressor.kv_cache` contiguous buffer so `block_idx = pos/ratio` matches
    /// prefill's index set exactly (no ring, no block-table remap).
    /// Prefill fills blocks `[0, n_win)`; decode appends after.
    pub pool: spark_runtime::gpu::DevicePtr,
    /// Capacity in compressed blocks = `max_position_embeddings.div_ceil(ratio)`.
    pub pool_blocks: usize,
    /// 4b inc-3: persistent decode-time normed-x ring `[ratio × hidden]` BF16.
    /// Each decode token's compressor input (`normed`, the layer-input RMSNorm
    /// output — the SAME tensor prefill's `cache_skip_v4` feeds `wkv`/`wgate`) is
    /// written to slot `pos % ratio`. At a window boundary the ring holds the
    /// `ratio` tokens of the just-completed window in order, and decode reruns the
    /// prefill compress pipeline over it to append one pool block. BF16 (not FP8):
    /// quantize only at the pool write, so decode's compressor input matches
    /// prefill's bit-for-bit (fp8-ing the input would add a stage prefill never
    /// sees and make the golden-vector gate uninterpretable).
    pub ring: spark_runtime::gpu::DevicePtr,
    /// 4b inc-3 (CSA only): previous completed window's normed-x `[ratio × hidden]`
    /// BF16. CSA reads a 2×ratio overlap (prev window's Ca + current window's Cb);
    /// after each append the ring is copied here to feed the next window's Ca.
    /// `DevicePtr::NULL` for HCA (no overlap). The first decode window has no valid
    /// prev (it would be a prefill window absent from the decode ring) → Ca masked.
    pub prev_win: spark_runtime::gpu::DevicePtr,
    /// 4b inc-3 (CSA only): concat staging `[2×ratio × hidden]` BF16 = prev_win ‖
    /// ring, the 2×ratio-token input the CSA compress kernel indexes for one
    /// overlapped window. `DevicePtr::NULL` for HCA.
    pub stage: spark_runtime::gpu::DevicePtr,
}

/// Per-block Manifold-Constrained Hyper-Connection (mHC) parameters for one
/// site (attention or FFN). All buffers are float32 device pointers, matching
/// the checkpoint dtype. See `ops::hc_pre` / `ops::hc_post`.
pub struct HcSiteWeights {
    /// Mix projection `fn`: `[mix_hc, hc_mult*hidden]` f32, where
    /// `mix_hc = (2 + hc_mult) * hc_mult`.
    pub hc_fn: DevicePtr,
    /// Mix bias `base`: `[mix_hc]` f32.
    pub hc_base: DevicePtr,
    /// Mix scale: `[3]` f32 (pre / post / comb scalars).
    pub hc_scale: DevicePtr,
}

/// Both HC sites for a DeepSeek-V4 block: the attention site runs before/after
/// attention, the FFN site before/after the MoE FFN.
/// Model-level HC head parameters (final collapse before LM head).
/// Loaded once, attached to every layer, but only used by the last layer.
#[derive(Clone)]
pub struct HcHeadWeights {
    /// Mix projection `head_fn`: `[hc_mult, hc_mult*hidden]` f32.
    pub hc_fn: DevicePtr,
    /// Mix bias `head_base`: `[hc_mult]` f32.
    pub hc_base: DevicePtr,
    /// Mix scale: `[1]` f32.
    pub hc_scale: DevicePtr,
}

pub struct HcWeights {
    pub attn: HcSiteWeights,
    pub ffn: HcSiteWeights,
    /// Model-level head weights. `Some` on all layers (replicated pointer),
    /// consumed only by the last layer's `hc_head` call.
    pub head: Option<HcHeadWeights>,
    pub hc_mult: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
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
    /// Per-head attention gate weight (Step 3.7 g_proj).
    /// Shape: [num_q_heads, hidden_size] BF16. Applied as:
    /// attn_out = attn_out * sigmoid(g_proj @ hidden_states)
    /// with broadcast over head_dim.
    pub(crate) head_gate_weight: Option<DenseWeight>,
    /// Kernel handle for per-head sigmoid gate broadcast multiply.
    pub(super) sigmoid_gate_head_broadcast_k: KernelHandle,
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
    // ── Manifold-Constrained Hyper-Connections (mHC) — DeepSeek-V4 ──
    /// Per-block HC parameters. `Some` only for DeepSeek-V4 (`hc_mult > 0`),
    /// in which case the attn/ffn residual sites use `hc_pre`/`hc_post`
    /// against the `hc_streams` buffer instead of the standard residual add.
    pub(crate) hc: Option<HcWeights>,
    /// HC `hc_pre` kernel handle (NULL when HC disabled).
    pub(super) hc_pre_k: KernelHandle,
    /// HC `hc_post` kernel handle (NULL when HC disabled).
    pub(super) hc_post_k: KernelHandle,
    /// HC `hc_expand` kernel handle (NULL when HC disabled).
    pub(super) hc_expand_k: KernelHandle,
    /// HC `hc_head` kernel handle (NULL when HC disabled).
    pub(super) hc_head_k: KernelHandle,
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
    // Fast transposed FP8 prefill GEMM (128x128 / 8-warp / two-level FP32 fold).
    // Consumes the SAME B_t[K,N] + block_scale_t[K/128,N/128] that
    // transpose_fp8 / transpose_block_scale already produce. KernelHandle(0) on
    // miss → fall back to w8a16_gemm_t.
    pub(super) w8a16_gemm_t_m128_k: KernelHandle,
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
    /// K-only MRoPE kernel used when Q RoPE is fused into Q deinterleave/norm.
    pub(super) rope_mrope_interleaved_k_only_k: KernelHandle,
    /// YaRN RoPE kernel using pre-computed inv_freq table (Mistral, etc.)
    pub(super) rope_yarn_k: KernelHandle,
    /// Interleaved (GPT-J / is_neox_style=False) YaRN RoPE kernel — DeepSeek MLA.
    pub(super) rope_yarn_interleaved_k: KernelHandle,
    /// Conjugate (negated-sin) interleaved YaRN RoPE — DeepSeek-V4 attention
    /// output de-rotation (eq.26).
    pub(super) rope_yarn_interleaved_inv_k: KernelHandle,
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
    /// MLA paged decode kernel for DeepSeek-V4-Flash (compressed KV cache: 576 dims)
    pub(super) mla_paged_decode_k: KernelHandle,
    /// MLA paged decode kernel for DeepSeek-V4-Flash with FP8 KV cache
    pub(super) mla_paged_decode_fp8_k: KernelHandle,
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
    /// Dual-output (bf16 + f32) MoE-input norm for ATLAS_FP32_ROUTING. Zero if absent.
    pub(super) residual_add_rms_norm_gatef32_k: KernelHandle,
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
    /// LOSSLESS BF16-TC variant of t_m128 for QKV/o projection prefill (FP4→BF16
    /// dequant + BF16 MMA, no FP8 activation crush). Opt-in via ATLAS_BF16_TC_PROJ
    /// (default off → t_m128 path unchanged). KernelHandle(0) on miss.
    pub(super) w4a16_gemm_t_m128_bf16_k: KernelHandle,
    /// MiniMax-only shadow kernel.
    pub(super) w4a16_gemm_t_m128_v2_k: KernelHandle,
    /// v3 variant: K_STEP=64.
    pub(super) w4a16_gemm_t_m128_v3_k: KernelHandle,
    pub(super) dense_gemm_k: KernelHandle,
    /// Tensor-core pipelined BF16 GEMM (mma.sync + cp.async, 128×128 tile) —
    /// ~40× the scalar `dense_gemm_k` on large-M prefill projections, same math
    /// (cosine 1.0). Used for the BF16-fallback Q/K/V/O projections (Holo's
    /// native-FP8-dequant-to-BF16 attention path).
    pub(super) dense_gemm_pipelined_k: KernelHandle,
    pub(super) prefill_attn_k: KernelHandle,
    /// HDIM=512 contiguous prefill for Gemma-4 full-attention layers
    pub(super) prefill_attn_512_k: KernelHandle,
    /// DeepSeek-V4 CSA compressor: window softmax-gated KV compression.
    pub(super) csa_compress_k: KernelHandle,
    /// DeepSeek-V4 CSA prefill attention over [raw | compressed] KV + sink.
    pub(super) prefill_attn_compressed_k: KernelHandle,
    /// 4b: # compressed blocks prefill wrote to `mla.compressor.pool` for the
    /// active sequence (= prefill_len / ratio). Decode's compressed arm attends
    /// blocks `[0, this)`. AtomicU32 for interior mutability under prefill's
    /// `&self`; V4 serves max_batch=1 so one counter suffices (inc-3: per-seq
    /// tracking + decode-time append will grow this each boundary crossing).
    pub(super) v4_comp_pool_filled: std::sync::atomic::AtomicU32,
    /// 4b inc-3 decode-append state (V4 serves max_batch=1 → scalar per layer).
    /// `prev_valid`: the CSA `prev_win` ring holds a real previous decode window
    /// (false until the first decode append, and reset each prefill) — when false
    /// the CSA append masks Ca (window-0 semantics). `decode_started`/`first_pos`:
    /// the absolute position of the first decode token this sequence, used to skip
    /// any prefill/decode straddle window whose ring slots aren't all decode-written
    /// (that one block is left as prefill/zero — a documented seam, not corruption).
    pub(super) v4_comp_prev_valid: std::sync::atomic::AtomicBool,
    pub(super) v4_decode_started: std::sync::atomic::AtomicBool,
    pub(super) v4_decode_first_pos: std::sync::atomic::AtomicU32,
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
    pub(super) deinterleave_qg_split_qnorm_mrope_k: KernelHandle,
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
